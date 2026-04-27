import argparse
from contextlib import nullcontext
import json
from datetime import datetime, timezone
from pathlib import Path

import modal


APP_NAME = "characters-dataset-inference-vllm"
DATA_VOLUME_NAME = "characters-dpo-data"
RESULTS_VOLUME_NAME = "characters-trl-dpo-results"
HF_CACHE_VOLUME_NAME = "huggingface-cache"
REMOTE_DATA_ROOT = Path("/data")
REMOTE_RESULTS_ROOT = Path("/results")
HF_CACHE_DIR = "/hf-cache"

app = modal.App(APP_NAME)

data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
results_volume = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "accelerate",
        "huggingface_hub",
        "pandas",
        "peft",
        "pyarrow",
        "transformers",
        "vllm",
    )
    .env(
        {
            "HF_HOME": HF_CACHE_DIR,
            "HF_HUB_CACHE": HF_CACHE_DIR,
            "TOKENIZERS_PARALLELISM": "false",
            "HF_XET_HIGH_PERFORMANCE": "1",
        }
    )
    .add_local_python_source("characters")
)


@app.function(
    gpu="A100",
    image=image,
    secrets=[hf_secret],
    volumes={
        str(REMOTE_DATA_ROOT): data_volume,
        str(REMOTE_RESULTS_ROOT): results_volume,
        HF_CACHE_DIR: hf_cache_volume,
    },
    timeout=8 * 60 * 60,
)
def run_dataset_inference_remote(
    config_dict: dict[str, object],
    *,
    dataset_volume: str,
    dataset_path: str,
) -> dict[str, object]:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    from characters.dataset_inference_config import dataset_inference_config_from_dict
    from characters.response_generation import append_jsonl_rows, load_jsonl_rows

    config = dataset_inference_config_from_dict(config_dict)
    input_path = _mounted_path(dataset_volume, dataset_path)
    output_path = _mounted_path(RESULTS_VOLUME_NAME, config.paths.remote_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)

    rows = load_jsonl_rows(input_path)
    if config.dataset.limit is not None:
        rows = rows[: config.dataset.limit]

    adapter_dir = None
    adapter_metadata: dict[str, object] = {}
    if config.model.adapter_enabled:
        if config.model.adapters:
            adapter_dir = _materialize_weighted_adapter(config)
            adapter_metadata = {
                "adapter_composition": [
                    {
                        "name": adapter.name or "",
                        "volume": adapter.volume,
                        "path": adapter.effective_path,
                        "weight": adapter.weight,
                    }
                    for adapter in config.model.adapters
                ]
            }
        else:
            adapter_dir = _mounted_path(
                config.model.adapter_volume,
                config.model.effective_adapter_path or "",
            )
            if not adapter_dir.exists():
                raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    print(f"[dataset-inference] config={config.name}", flush=True)
    print(f"[dataset-inference] input={input_path} rows={len(rows)}", flush=True)
    print(f"[dataset-inference] output={output_path}", flush=True)
    print(f"[dataset-inference] base_model={config.model.base_model}", flush=True)
    if adapter_dir is not None:
        print(f"[dataset-inference] adapter={adapter_dir}", flush=True)

    llm = LLM(
        model=config.model.base_model,
        enable_lora=config.model.adapter_enabled,
        max_lora_rank=config.vllm.max_lora_rank,
        max_loras=1,
        max_model_len=config.vllm.max_model_len,
        max_num_seqs=config.vllm.max_num_seqs,
        max_num_batched_tokens=config.vllm.max_num_batched_tokens,
        gpu_memory_utilization=config.vllm.gpu_memory_utilization,
        tensor_parallel_size=config.vllm.tensor_parallel_size or 1,
        trust_remote_code=config.model.trust_remote_code,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        max_tokens=config.generation.max_new_tokens,
        temperature=config.generation.temperature,
        top_p=config.generation.top_p,
    )
    lora_request = (
        LoRARequest(
            lora_name=config.model.effective_adapter_name or config.name,
            lora_int_id=1,
            lora_path=str(adapter_dir),
        )
        if config.model.adapter_enabled and adapter_dir is not None
        else None
    )

    generated_rows = 0
    batch_size = max(1, config.vllm.max_num_seqs)
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        rendered_prompts = [
            tokenizer.apply_chat_template(
                _prompt_messages(row, prompt_key=config.dataset.prompt_key),
                tokenize=False,
                add_generation_prompt=True,
            )
            for row in batch_rows
        ]
        outputs = llm.generate(
            rendered_prompts,
            sampling_params=sampling_params,
            lora_request=lora_request,
            use_tqdm=True,
        )
        output_rows: list[dict[str, object]] = []
        for row, output in zip(batch_rows, outputs):
            generated = output.outputs[0].text.strip() if output.outputs else ""
            output_rows.append(
                {
                    **row,
                    "split": config.dataset.split,
                    "generated": generated,
                    "base_model": config.model.base_model,
                    "adapter_enabled": config.model.adapter_enabled,
                    "adapter_name": config.model.effective_adapter_name or "",
                    "adapter_dir": str(adapter_dir) if adapter_dir is not None else "",
                    **adapter_metadata,
                }
            )
        append_jsonl_rows(output_path, output_rows)
        generated_rows += len(output_rows)
        print(f"[dataset-inference] generated {generated_rows}/{len(rows)} rows", flush=True)

    metadata = {
        "config_name": config.name,
        "input": {
            "source": config.dataset.source,
            "volume": dataset_volume,
            "path": dataset_path,
            "rows": len(rows),
        },
        "output_path": str(output_path),
        "model": {
            "base_model": config.model.base_model,
            "adapter_enabled": config.model.adapter_enabled,
            "adapter_name": config.model.effective_adapter_name or "",
            "adapter_dir": str(adapter_dir) if adapter_dir is not None else "",
            **adapter_metadata,
        },
        "generation": {
            "max_new_tokens": config.generation.max_new_tokens,
            "temperature": config.generation.temperature,
            "top_p": config.generation.top_p,
        },
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    results_volume.commit()
    hf_cache_volume.commit()
    return metadata


def _materialize_weighted_adapter(config: object) -> Path:
    import gc
    import shutil

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    adapters = config.model.adapters or []
    if not adapters:
        raise ValueError("No adapters configured for weighted materialization.")

    adapter_dirs: list[Path] = []
    for adapter in adapters:
        adapter_dir = _mounted_path(adapter.volume, adapter.effective_path)
        if not adapter_dir.exists():
            raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")
        adapter_dirs.append(adapter_dir)

    materialized_dir = (
        _mounted_path(RESULTS_VOLUME_NAME, config.paths.remote_output_path)
        .parent
        / f"{config.name}_weighted_adapter"
    )
    shutil.rmtree(materialized_dir, ignore_errors=True)
    materialized_dir.parent.mkdir(parents=True, exist_ok=True)

    print(
        "[dataset-inference] materializing weighted adapter="
        + ", ".join(f"{path}:{adapter.weight}" for path, adapter in zip(adapter_dirs, adapters)),
        flush=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=config.model.trust_remote_code,
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_dirs[0]), adapter_name="adapter_0")
    adapter_names = ["adapter_0"]
    for index, adapter_dir in enumerate(adapter_dirs[1:], start=1):
        adapter_name = f"adapter_{index}"
        model.load_adapter(str(adapter_dir), adapter_name=adapter_name)
        adapter_names.append(adapter_name)

    weighted_name = "default"
    model.add_weighted_adapter(
        adapters=adapter_names,
        weights=[adapter.weight for adapter in adapters],
        adapter_name=weighted_name,
        combination_type="linear",
    )
    model.save_pretrained(str(materialized_dir), selected_adapters=[weighted_name])

    del model
    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results_volume.commit()
    print(f"[dataset-inference] weighted_adapter={materialized_dir}", flush=True)
    return materialized_dir


def _mounted_path(volume_name: str, volume_path: str) -> Path:
    normalized_path = volume_path if volume_path.startswith("/") else f"/{volume_path}"
    if volume_name == DATA_VOLUME_NAME:
        return REMOTE_DATA_ROOT / normalized_path.lstrip("/")
    if volume_name == RESULTS_VOLUME_NAME:
        return REMOTE_RESULTS_ROOT / normalized_path.lstrip("/")
    raise ValueError(f"Unsupported mounted volume: {volume_name}")


def _prompt_messages(row: dict[str, object], *, prompt_key: str) -> list[dict[str, str]]:
    prompt = row.get(prompt_key)
    if isinstance(prompt, list):
        messages: list[dict[str, str]] = []
        for message in prompt:
            if not isinstance(message, dict):
                continue
            messages.append(
                {
                    "role": str(message.get("role", "user")),
                    "content": str(message.get("content", "")),
                }
            )
        if messages:
            return messages
    return [{"role": "user", "content": str(prompt or "")}]


def _prepare_dataset(config_path: str | Path) -> tuple[dict[str, object], str, str]:
    from characters.dataset_inference_config import load_dataset_inference_config
    from characters.response_generation import write_jsonl_rows

    config = load_dataset_inference_config(config_path)
    if config.dataset.source == "modal":
        return _config_to_dict(config), config.dataset.volume, config.dataset.modal_path

    rows = _load_local_dataset_rows(config)
    if config.dataset.limit is not None:
        rows = rows[: config.dataset.limit]
    local_upload_dir = Path("/tmp") / "characters_dataset_inference" / config.name
    local_upload_dir.mkdir(parents=True, exist_ok=True)
    local_dataset_path = local_upload_dir / "dataset.jsonl"
    write_jsonl_rows(local_dataset_path, rows)

    upload_path = config.dataset.effective_upload_path
    with data_volume.batch_upload(force=True) as batch:
        batch.put_file(str(local_dataset_path), upload_path)
    return _config_to_dict(config), DATA_VOLUME_NAME, upload_path


def _load_local_dataset_rows(config: object) -> list[dict[str, object]]:
    from characters.response_generation import flatten_expanded_prompts, load_jsonl_rows

    rows = load_jsonl_rows(config.dataset.local_path)
    if not rows:
        return []
    if all(config.dataset.prompt_key in row for row in rows):
        return rows
    if config.dataset.prompt_key == "prompt":
        return flatten_expanded_prompts(rows)
    raise ValueError(
        f"Local dataset rows do not contain prompt_key={config.dataset.prompt_key!r}: "
        f"{config.dataset.local_path}"
    )


def _config_to_dict(config: object) -> dict[str, object]:
    from dataclasses import asdict

    payload = asdict(config)
    payload["paths"]["output_path"] = str(payload["paths"]["output_path"])
    payload["dataset"]["path"] = str(payload["dataset"]["path"])
    return payload


def _app_context(start_app: bool):
    if start_app:
        return app.run()
    return nullcontext()


def _sync_output(config_path: str | Path, remote_output_path: str, *, start_app: bool) -> None:
    from characters.dataset_inference_config import load_dataset_inference_config

    config = load_dataset_inference_config(config_path)
    config.paths.output_path.parent.mkdir(parents=True, exist_ok=True)
    with _app_context(start_app):
        with config.paths.output_path.open("wb") as output_file:
            results_volume.read_file_into_fileobj(remote_output_path, output_file)

        metadata_output = config.paths.output_path.with_suffix(".metadata.json")
        remote_metadata_path = str(Path(remote_output_path).with_suffix(".metadata.json"))
        try:
            with metadata_output.open("wb") as output_file:
                results_volume.read_file_into_fileobj(remote_metadata_path, output_file)
        except FileNotFoundError:
            metadata_output.unlink(missing_ok=True)


def _run_local(config: str, *, start_app: bool) -> None:
    config_dict, dataset_volume, dataset_path = _prepare_dataset(config)
    print(f"[dataset-inference] dataset_volume={dataset_volume}", flush=True)
    print(f"[dataset-inference] dataset_path={dataset_path}", flush=True)
    with modal.enable_output():
        with _app_context(start_app):
            result = run_dataset_inference_remote.remote(
                config_dict=config_dict,
                dataset_volume=dataset_volume,
                dataset_path=dataset_path,
            )
    remote_output_path = str(config_dict["paths"]["remote_output_path"])
    _sync_output(config, remote_output_path, start_app=start_app)
    print(json.dumps(result, indent=2), flush=True)


@app.local_entrypoint()
def main(config: str = "configs/adversarial_skeptic_holistic_llama31/inference_original_lima_prompt_pool_sft.yaml") -> None:
    _run_local(config, start_app=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/adversarial_skeptic_holistic_llama31/inference_original_lima_prompt_pool_sft.yaml",
        help="Path to a dataset inference YAML config.",
    )
    args = parser.parse_args()
    _run_local(args.config, start_app=True)
