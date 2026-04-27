import json
import math
from datetime import datetime, timezone
from pathlib import Path

import modal


APP_NAME = "characters-trl-dpo-inference"
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
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_pyproject("pyproject.toml")
    .env(
        {
            "HF_HOME": HF_CACHE_DIR,
            "HF_HUB_CACHE": HF_CACHE_DIR,
            "TOKENIZERS_PARALLELISM": "false",
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
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    batch_size: int = 4,
    output_name: str | None = None,
) -> dict[str, object]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from characters.response_generation import load_jsonl_rows, write_jsonl_rows
    from characters.trl_dpo_config import TrlDpoConfig

    config = TrlDpoConfig.model_validate(config_dict)
    dataset_dir = REMOTE_DATA_ROOT / config.name
    adapter_dir = REMOTE_RESULTS_ROOT / config.name
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = output_name or f"inference_{timestamp}"
    output_dir = adapter_dir / "inference" / run_name

    print(f"[inference] loading tokenizer from {adapter_dir}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        str(adapter_dir),
        trust_remote_code=config.model.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError(
                "Tokenizer is missing both pad_token and eos_token. "
                "Set a tokenizer with an EOS token or configure one manually."
            )
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs = {
        "dtype": _resolve_torch_dtype(config.model.torch_dtype),
        "trust_remote_code": config.model.trust_remote_code,
        "device_map": "auto",
    }
    if config.model.attn_implementation is not None:
        model_kwargs["attn_implementation"] = config.model.attn_implementation

    print(f"[inference] loading base model {config.model.name}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(config.model.name, **model_kwargs)
    print(f"[inference] loading adapter from {adapter_dir}", flush=True)
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()

    train_rows = load_jsonl_rows(dataset_dir / "train.jsonl")
    val_path = dataset_dir / "val.jsonl"
    val_rows = load_jsonl_rows(val_path) if val_path.exists() else []

    print(
        f"[inference] dataset loaded: train={len(train_rows)} rows, val={len(val_rows)} rows",
        flush=True,
    )

    train_outputs = _generate_rows(
        model=model,
        tokenizer=tokenizer,
        rows=train_rows,
        split="train",
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        base_model=config.model.name,
        adapter_dir=adapter_dir,
    )
    write_jsonl_rows(output_dir / "train.jsonl", train_outputs)

    val_outputs: list[dict[str, object]] = []
    if val_rows:
        val_outputs = _generate_rows(
            model=model,
            tokenizer=tokenizer,
            rows=val_rows,
            split="val",
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            base_model=config.model.name,
            adapter_dir=adapter_dir,
        )
        write_jsonl_rows(output_dir / "val.jsonl", val_outputs)

    results_volume.commit()
    hf_cache_volume.commit()

    return {
        "config_name": config.name,
        "adapter_dir": str(adapter_dir),
        "output_dir": str(output_dir),
        "train_rows": len(train_outputs),
        "val_rows": len(val_outputs),
        "generation": {
            "batch_size": batch_size,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        },
    }


def _generate_rows(
    *,
    model: object,
    tokenizer: object,
    rows: list[dict[str, object]],
    split: str,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    base_model: str,
    adapter_dir: Path,
) -> list[dict[str, object]]:
    import torch

    outputs: list[dict[str, object]] = []
    do_sample = temperature > 0
    total_batches = max(1, math.ceil(len(rows) / batch_size)) if rows else 0

    print(
        f"[inference] split={split} rows={len(rows)} batch_size={batch_size} total_batches={total_batches}",
        flush=True,
    )

    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start:start + batch_size]
        batch_index = start // batch_size + 1
        print(
            f"[inference] split={split} batch={batch_index}/{total_batches} rows={len(batch_rows)}",
            flush=True,
        )
        rendered_prompts = [
            tokenizer.apply_chat_template(
                _prompt_messages(row),
                tokenize=False,
                add_generation_prompt=True,
            )
            for row in batch_rows
        ]
        tokenized = tokenizer(
            rendered_prompts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        tokenized = {key: value.to(model.device) for key, value in tokenized.items()}

        with torch.inference_mode():
            generation_kwargs = {
                **tokenized,
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if do_sample:
                generation_kwargs["temperature"] = temperature
                generation_kwargs["top_p"] = top_p
            generated = model.generate(**generation_kwargs)

        prompt_lengths = tokenized["attention_mask"].sum(dim=1).tolist()
        for row, sequence, prompt_length in zip(batch_rows, generated, prompt_lengths):
            completion_ids = sequence[int(prompt_length):]
            generated_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
            outputs.append(
                {
                    **row,
                    "split": split,
                    "generated": generated_text,
                    "base_model": base_model,
                    "adapter_dir": str(adapter_dir),
                }
            )

    print(f"[inference] split={split} completed rows={len(outputs)}", flush=True)
    return outputs


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


def _resolve_torch_dtype(raw_dtype: str) -> object:
    import torch

    normalized = raw_dtype.strip().lower()
    if normalized == "auto":
        return "auto"
    if normalized == "bfloat16":
        return torch.bfloat16
    if normalized == "float16":
        return torch.float16
    if normalized == "float32":
        return torch.float32
    raise ValueError(f"Unsupported model.torch_dtype: {raw_dtype}")


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
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    batch_size: int = 4,
    output_name: str = "",
) -> None:
    print(f"[inference] uploading dataset for config={config}", flush=True)
    config_dict, remote_dataset_dir = _upload_dataset(config)
    print(f"[inference] uploaded dataset to {remote_dataset_dir}", flush=True)
    print("[inference] launching remote inference job", flush=True)
    result = run_trl_inference_remote.remote(
        config_dict=config_dict,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        batch_size=batch_size,
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
