import json
from datetime import datetime, timezone
from pathlib import Path

import modal


APP_NAME = "characters-trl-dpo-inference-vllm"
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
def run_trl_inference_remote(
    config_dict: dict[str, object],
    *,
    split: str = "val",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.9,
    output_name: str | None = None,
) -> dict[str, object]:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    from characters.response_generation import load_jsonl_rows, write_jsonl_rows
    from characters.trl_dpo_config import TrlDpoConfig

    config = TrlDpoConfig.model_validate(config_dict)
    dataset_dir = REMOTE_DATA_ROOT / config.name
    adapter_dir = REMOTE_RESULTS_ROOT / config.name
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = output_name or f"inference_{timestamp}"
    output_dir = adapter_dir / "inference_vllm" / run_name

    print(f"[vllm] starting inference for config={config.name}", flush=True)
    print(f"[vllm] loading model={config.model.name}", flush=True)
    print(f"[vllm] loading adapter={adapter_dir}", flush=True)

    llm = LLM(
        model=config.model.name,
        enable_lora=True,
        max_lora_rank=max(config.lora.r, 64),
        max_loras=1,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    lora_request = LoRARequest(
        lora_name=config.name,
        lora_int_id=1,
        lora_path=str(adapter_dir),
    )

    val_path = dataset_dir / "val.jsonl"
    split_rows: dict[str, list[dict[str, object]]] = {}
    if split in {"train", "both"}:
        split_rows["train"] = load_jsonl_rows(dataset_dir / "train.jsonl")
    if split in {"val", "both"} and val_path.exists():
        split_rows["val"] = load_jsonl_rows(val_path)

    print(
        "[vllm] loaded splits: "
        + ", ".join(f"{name}={len(rows)}" for name, rows in split_rows.items()),
        flush=True,
    )

    generated_counts: dict[str, int] = {}
    for split_name, rows in split_rows.items():
        print(f"[vllm] generating split={split_name} rows={len(rows)}", flush=True)
        split_outputs = _generate_rows(
            llm=llm,
            tokenizer=tokenizer,
            rows=rows,
            split=split_name,
            sampling_params=sampling_params,
            lora_request=lora_request,
            base_model=config.model.name,
            adapter_dir=adapter_dir,
        )
        write_jsonl_rows(output_dir / f"{split_name}.jsonl", split_outputs)
        generated_counts[split_name] = len(split_outputs)
        print(f"[vllm] completed split={split_name} rows={len(split_outputs)}", flush=True)

    results_volume.commit()
    hf_cache_volume.commit()

    return {
        "config_name": config.name,
        "adapter_dir": str(adapter_dir),
        "output_dir": str(output_dir),
        "split": split,
        "generated_counts": generated_counts,
        "generation": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
        },
    }


def _generate_rows(
    *,
    llm: object,
    tokenizer: object,
    rows: list[dict[str, object]],
    split: str,
    sampling_params: object,
    lora_request: object,
    base_model: str,
    adapter_dir: Path,
) -> list[dict[str, object]]:
    rendered_prompts = [
        tokenizer.apply_chat_template(
            _prompt_messages(row),
            tokenize=False,
            add_generation_prompt=True,
        )
        for row in rows
    ]

    outputs = llm.generate(
        rendered_prompts,
        sampling_params=sampling_params,
        lora_request=lora_request,
        use_tqdm=True,
    )

    generated_rows: list[dict[str, object]] = []
    for row, output in zip(rows, outputs):
        text = ""
        if output.outputs:
            text = output.outputs[0].text.strip()
        generated_rows.append(
            {
                **row,
                "split": split,
                "generated": text,
                "base_model": base_model,
                "adapter_dir": str(adapter_dir),
            }
        )
    return generated_rows


def _prompt_messages(row: dict[str, object]) -> list[dict[str, str]]:
    prompt = row.get("prompt")
    if isinstance(prompt, list):
        messages: list[dict[str, str]] = []
        for message in prompt:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "user"))
            content = str(message.get("content", ""))
            messages.append({"role": role, "content": content})
        if messages:
            return messages
    return [{"role": "user", "content": str(prompt or "")}]


def _upload_dataset(config_path: str | Path) -> tuple[dict[str, object], str]:
    from characters.trl_dpo_config import load_trl_dpo_config

    config = load_trl_dpo_config(config_path)
    train_path = config.dataset.train_data_path
    val_path = config.dataset.val_data_path

    if not train_path.exists():
        raise FileNotFoundError(f"TRL train dataset not found: {train_path}")
    if val_path is not None and not val_path.exists():
        raise FileNotFoundError(f"TRL val dataset not found: {val_path}")

    remote_dataset_dir = f"/{config.name}"
    with data_volume.batch_upload(force=True) as batch:
        batch.put_file(str(train_path), f"{remote_dataset_dir}/train.jsonl")
        if val_path is not None:
            batch.put_file(str(val_path), f"{remote_dataset_dir}/val.jsonl")

    return config.model_dump(mode="json"), remote_dataset_dir


@app.local_entrypoint()
def main(
    config: str = "configs/trl_dpo/adversarial_skeptic_llama31.yaml",
    split: str = "val",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.9,
    output_name: str = "",
) -> None:
    if split not in {"train", "val", "both"}:
        raise ValueError("split must be one of: train, val, both")

    print(f"[vllm] uploading dataset for config={config}", flush=True)
    config_dict, remote_dataset_dir = _upload_dataset(config)
    print(f"[vllm] uploaded dataset to {remote_dataset_dir}", flush=True)
    print("[vllm] launching remote inference job", flush=True)
    result = run_trl_inference_remote.remote(
        config_dict=config_dict,
        split=split,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        output_name=output_name or None,
    )
    print(
        json.dumps(
            {
                "uploaded_dataset_dir": remote_dataset_dir,
                **result,
            },
            indent=2,
        )
    )
