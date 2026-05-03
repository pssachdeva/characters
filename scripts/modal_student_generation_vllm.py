import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import modal


APP_NAME = "characters-student-generation-vllm"
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
def run_student_generation_remote(
    config_dict: dict[str, object],
    *,
    input_path: str,
    output_path: str,
    template: str,
) -> dict[str, object]:
    from vllm import LLM, SamplingParams

    from characters.response_generation import append_jsonl_rows, load_jsonl_rows

    rows = load_jsonl_rows(_mounted_path(DATA_VOLUME_NAME, input_path))
    remote_output_path = _mounted_path(RESULTS_VOLUME_NAME, output_path)
    remote_output_path.parent.mkdir(parents=True, exist_ok=True)
    remote_output_path.unlink(missing_ok=True)

    model_name = _vllm_model_name(str(config_dict["model"]["name"]))
    print(f"[student-generation] config={config_dict['name']}", flush=True)
    print(f"[student-generation] input={input_path} rows={len(rows)}", flush=True)
    print(f"[student-generation] output={remote_output_path}", flush=True)
    print(f"[student-generation] model={model_name}", flush=True)

    llm = LLM(
        model=model_name,
        max_model_len=int(config_dict.get("vllm", {}).get("max_model_len", 4096)),
        max_num_seqs=int(config_dict.get("vllm", {}).get("max_num_seqs", 256)),
        max_num_batched_tokens=int(config_dict.get("vllm", {}).get("max_num_batched_tokens", 32768)),
        gpu_memory_utilization=float(config_dict.get("vllm", {}).get("gpu_memory_utilization", 0.9)),
        tensor_parallel_size=int(config_dict.get("vllm", {}).get("tensor_parallel_size") or 1),
        trust_remote_code=bool(config_dict.get("vllm", {}).get("trust_remote_code", False)),
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        max_tokens=int(config_dict["sampling"]["max_tokens"]),
        temperature=float(config_dict["sampling"]["temperature"]),
        top_p=float(config_dict["sampling"]["top_p"]),
    )

    generated_rows = 0
    batch_size = int(config_dict.get("vllm", {}).get("max_num_seqs", 256))
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": template.format(prompt=str(row["prompt"]))}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for row in batch_rows
        ]
        outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)
        output_rows: list[dict[str, object]] = []
        for row, output in zip(batch_rows, outputs):
            rejected = output.outputs[0].text.strip() if output.outputs else ""
            output_rows.append({**row, "rejected": rejected, "model": model_name, "provider": "modal_vllm"})
        append_jsonl_rows(remote_output_path, output_rows)
        generated_rows += len(output_rows)
        print(f"[student-generation] generated {generated_rows}/{len(rows)} rows", flush=True)

    metadata = {
        "config_name": str(config_dict["name"]),
        "input_path": input_path,
        "output_path": output_path,
        "rows": generated_rows,
        "model": model_name,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    remote_output_path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    results_volume.commit()
    hf_cache_volume.commit()
    return metadata


def _mounted_path(volume_name: str, volume_path: str) -> Path:
    normalized_path = volume_path if volume_path.startswith("/") else f"/{volume_path}"
    if volume_name == DATA_VOLUME_NAME:
        return REMOTE_DATA_ROOT / normalized_path.lstrip("/")
    if volume_name == RESULTS_VOLUME_NAME:
        return REMOTE_RESULTS_ROOT / normalized_path.lstrip("/")
    raise ValueError(f"Unsupported mounted volume: {volume_name}")


def _vllm_model_name(raw_model_name: str) -> str:
    aliases = {
        "meta-llama/llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/meta-llama-3.1-8b-instruct-turbo": "meta-llama/Llama-3.1-8B-Instruct",
        "nim/meta/llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    }
    return aliases.get(raw_model_name.lower(), raw_model_name)


def _prepare_inputs(config_path: str | Path) -> tuple[dict[str, object], str, str, str]:
    from characters.prompt_templates import load_prompt_template
    from characters.response_generation import load_prompt_rows, repeat_prompt_rows, write_jsonl_rows
    from characters.response_generation_config import load_response_generation_config

    config = load_response_generation_config(config_path)
    rows = repeat_prompt_rows(
        load_prompt_rows(config.paths.input_path),
        n_samples_per_prompt=config.n_samples_per_prompt,
    )
    upload_path = f"/student_generation/{config.name}/input.jsonl"
    remote_output_path = f"/student_generation/{config.name}/output.jsonl"

    local_upload_dir = Path("/tmp") / "characters_student_generation" / config.name
    local_upload_dir.mkdir(parents=True, exist_ok=True)
    local_input_path = local_upload_dir / "input.jsonl"
    write_jsonl_rows(local_input_path, rows)

    with data_volume.batch_upload(force=True) as batch:
        batch.put_file(str(local_input_path), upload_path)

    config_dict = asdict(config)
    config_dict["paths"] = {key: str(value) if value is not None else None for key, value in config_dict["paths"].items()}
    config_dict["vllm"] = {
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.9,
        "max_num_seqs": 256,
        "max_num_batched_tokens": 32768,
    }
    template = load_prompt_template(config.paths.prompt_path)
    return config_dict, upload_path, remote_output_path, template


def _sync_output(config_path: str | Path, remote_output_path: str, *, use_app_run: bool) -> None:
    from characters.response_generation_config import load_response_generation_config

    config = load_response_generation_config(config_path)
    config.paths.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.paths.output_path.unlink(missing_ok=True)

    def sync_files() -> None:
        with config.paths.output_path.open("wb") as output_file:
            results_volume.read_file_into_fileobj(remote_output_path, output_file)
        metadata_output = config.paths.output_path.with_suffix(".metadata.json")
        try:
            with metadata_output.open("wb") as output_file:
                results_volume.read_file_into_fileobj(str(Path(remote_output_path).with_suffix(".metadata.json")), output_file)
        except FileNotFoundError:
            metadata_output.unlink(missing_ok=True)

    if use_app_run:
        with app.run():
            sync_files()
    else:
        sync_files()


def _run_local(config: str, *, use_app_run: bool) -> None:
    config_dict, input_path, output_path, template = _prepare_inputs(config)
    print(f"[student-generation] uploaded_input={input_path}", flush=True)

    if use_app_run:
        with modal.enable_output():
            with app.run():
                result = run_student_generation_remote.remote(
                    config_dict=config_dict,
                    input_path=input_path,
                    output_path=output_path,
                    template=template,
                )
    else:
        with modal.enable_output():
            result = run_student_generation_remote.remote(
                config_dict=config_dict,
                input_path=input_path,
                output_path=output_path,
                template=template,
            )
    _sync_output(config, output_path, use_app_run=use_app_run)
    print(json.dumps(result, indent=2), flush=True)


@app.local_entrypoint()
def main(config: str = "configs/goodness_llama31/03_student_generation.yaml") -> None:
    _run_local(config, use_app_run=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/goodness_llama31/03_student_generation.yaml")
    args = parser.parse_args()
    _run_local(args.config, use_app_run=True)
