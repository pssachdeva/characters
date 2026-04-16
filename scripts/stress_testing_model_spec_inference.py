import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import modal


APP_NAME = "characters-stress-testing-model-spec-inference"
DATA_VOLUME_NAME = "characters-dpo-data"
RESULTS_VOLUME_NAME = "characters-trl-dpo-results"
HF_CACHE_VOLUME_NAME = "huggingface-cache"
REMOTE_DATA_ROOT = Path("/data")
REMOTE_RESULTS_ROOT = Path("/results")
HF_CACHE_DIR = "/hf-cache"
DATASET_FILENAME = "dataset.jsonl"

app = modal.App(APP_NAME)

data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
results_volume = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "huggingface_hub",
        "pandas",
        "pyarrow",
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
def run_stress_testing_model_spec_inference_remote(
    *,
    config_name: str,
    base_model: str,
    adapter_enabled: bool,
    adapter_name: str | None,
    dataset_config_name: str,
    dataset_split: str,
    dataset_row_count: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    max_model_len: int,
    gpu_memory_utilization: float,
    max_lora_rank: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    tensor_parallel_size: int | None,
    trust_remote_code: bool,
    run_name: str,
    output_filename: str,
) -> dict[str, object]:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    from characters.response_generation import append_jsonl_rows, load_jsonl_rows

    dataset_dir = REMOTE_DATA_ROOT / config_name
    dataset_path = dataset_dir / DATASET_FILENAME
    rows = load_jsonl_rows(dataset_path)
    if len(rows) != dataset_row_count:
        print(
            f"[inference] warning: expected {dataset_row_count} rows but loaded {len(rows)}",
            flush=True,
        )

    adapter_dir = REMOTE_RESULTS_ROOT / adapter_name if adapter_name else None
    if adapter_enabled and (adapter_dir is None or not adapter_dir.exists()):
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    output_dir = REMOTE_RESULTS_ROOT / "stress_testing_model_spec" / config_name / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{output_filename}.jsonl"

    print(f"[inference] starting config={config_name}", flush=True)
    print(
        f"[inference] dataset={dataset_config_name}/{dataset_split} rows={len(rows)}",
        flush=True,
    )
    print(f"[inference] base_model={base_model}", flush=True)
    print(f"[inference] adapter_enabled={adapter_enabled}", flush=True)
    if adapter_dir is not None:
        print(f"[inference] adapter_dir={adapter_dir}", flush=True)

    llm = LLM(
        model=base_model,
        enable_lora=adapter_enabled,
        max_lora_rank=max_lora_rank,
        max_loras=1,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size or 1,
        trust_remote_code=trust_remote_code,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    lora_request = (
        LoRARequest(
            lora_name=adapter_name or config_name,
            lora_int_id=1,
            lora_path=str(adapter_dir),
        )
        if adapter_enabled and adapter_dir is not None
        else None
    )

    batch_size = max(1, max_num_seqs)
    generated_rows = 0
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        rendered_prompts = [
            tokenizer.apply_chat_template(
                _prompt_messages(row),
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
            text = output.outputs[0].text.strip() if output.outputs else ""
            output_rows.append(
                {
                    **row,
                    "generated": text,
                    "base_model": base_model,
                    "adapter_enabled": adapter_enabled,
                    "adapter_name": adapter_name or "",
                    "adapter_dir": str(adapter_dir) if adapter_dir is not None else "",
                }
            )
        append_jsonl_rows(output_path, output_rows)
        generated_rows += len(output_rows)
        print(f"[inference] generated {generated_rows}/{len(rows)} rows", flush=True)

    metadata = {
        "config_name": config_name,
        "dataset": {
            "name": "jifanz/stress_testing_model_spec",
            "config_name": dataset_config_name,
            "split": dataset_split,
            "rows": len(rows),
        },
        "model": {
            "base_model": base_model,
            "adapter_enabled": adapter_enabled,
            "adapter_name": adapter_name or "",
            "adapter_dir": str(adapter_dir) if adapter_dir is not None else "",
        },
        "generation": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        },
        "vllm": {
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_lora_rank": max_lora_rank,
            "max_num_seqs": max_num_seqs,
            "max_num_batched_tokens": max_num_batched_tokens,
            "tensor_parallel_size": tensor_parallel_size,
            "trust_remote_code": trust_remote_code,
        },
        "output_path": str(output_path),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    results_volume.commit()
    hf_cache_volume.commit()
    return metadata


def _prompt_messages(row: dict[str, object]) -> list[dict[str, str]]:
    prompt = row.get("prompt")
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


def _upload_dataset_rows(config_name: str, rows: list[dict[str, object]]) -> str:
    from characters.response_generation import write_jsonl_rows

    dataset_dir = Path("/tmp") / "characters_stress_testing_model_spec_inference" / config_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = dataset_dir / DATASET_FILENAME
    write_jsonl_rows(dataset_path, rows)

    remote_dataset_dir = f"/{config_name}"
    with data_volume.batch_upload(force=True) as batch:
        batch.put_file(str(dataset_path), f"{remote_dataset_dir}/{DATASET_FILENAME}")
    return remote_dataset_dir


def _run_local(
    config: str,
    *,
    run_name: str = "",
) -> None:
    from characters.stress_testing_model_spec_inference import (
        build_stress_testing_model_spec_rows,
    )
    from characters.stress_testing_model_spec_inference_config import (
        load_stress_testing_model_spec_inference_config,
    )

    inference_config = load_stress_testing_model_spec_inference_config(config)
    dataset_summary = build_stress_testing_model_spec_rows(
        inference_config.dataset,
        n_samples_per_prompt=inference_config.generation.n_samples_per_prompt,
    )
    remote_dataset_dir = _upload_dataset_rows(inference_config.name, dataset_summary.rows)
    effective_run_name = run_name or inference_config.paths.output_path.parent.name
    output_filename = inference_config.paths.output_path.stem

    print(f"[inference] uploaded dataset to {remote_dataset_dir}", flush=True)
    print("[inference] starting Modal app...", flush=True)
    with modal.enable_output():
        with app.run():
            result = run_stress_testing_model_spec_inference_remote.remote(
                config_name=inference_config.name,
                base_model=inference_config.model.base_model,
                adapter_enabled=inference_config.model.adapter_enabled,
                adapter_name=inference_config.model.adapter_name,
                dataset_config_name=inference_config.dataset.config_name,
                dataset_split=inference_config.dataset.split,
                dataset_row_count=len(dataset_summary.rows),
                max_new_tokens=inference_config.generation.max_new_tokens,
                temperature=inference_config.generation.temperature,
                top_p=inference_config.generation.top_p,
                max_model_len=inference_config.vllm.max_model_len,
                gpu_memory_utilization=inference_config.vllm.gpu_memory_utilization,
                max_lora_rank=inference_config.vllm.max_lora_rank,
                max_num_seqs=inference_config.vllm.max_num_seqs,
                max_num_batched_tokens=inference_config.vllm.max_num_batched_tokens,
                tensor_parallel_size=inference_config.vllm.tensor_parallel_size,
                trust_remote_code=inference_config.vllm.trust_remote_code,
                run_name=effective_run_name,
                output_filename=output_filename,
            )
    print(
        json.dumps(
            {
                **result,
                "configured_local_output_path": str(inference_config.paths.output_path),
                "prompts_before_repeat": dataset_summary.prompts_before_repeat,
                "prompts_after_repeat": dataset_summary.prompts_after_repeat,
            },
            indent=2,
        )
    )


@app.local_entrypoint()
def main(
    config: str = "configs/stress_testing_model_spec_inference/adversarial_skeptic_llama31_all_models.yaml",
    run_name: str = "",
) -> None:
    _run_local(config, run_name=run_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/stress_testing_model_spec_inference/adversarial_skeptic_llama31_all_models.yaml",
        help="Path to a stress-testing inference YAML config.",
    )
    parser.add_argument(
        "--run-name",
        default="",
        help="Optional override for the remote run directory name.",
    )
    args = parser.parse_args()
    _run_local(args.config, run_name=args.run_name)
