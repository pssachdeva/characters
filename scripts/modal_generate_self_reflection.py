import json
from pathlib import Path

import modal


APP_NAME = "characters-self-reflection"
RESULTS_VOLUME_NAME = "characters-trl-dpo-results"
HF_CACHE_VOLUME_NAME = "huggingface-cache"
REMOTE_RESULTS_ROOT = Path("/results")
HF_CACHE_DIR = "/hf-cache"

app = modal.App(APP_NAME)

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
    .add_local_dir("prompts", remote_path="/root/prompts")
)


@app.function(
    gpu="A100",
    image=image,
    secrets=[hf_secret],
    volumes={
        str(REMOTE_RESULTS_ROOT): results_volume,
        HF_CACHE_DIR: hf_cache_volume,
    },
    timeout=8 * 60 * 60,
)
def run_self_reflection_remote(
    source_config_dict: dict[str, object],
    *,
    traits: list[str],
    constitution: str | None,
    config_name: str,
    generation: dict[str, object],
    vllm_runtime: dict[str, object],
) -> dict[str, object]:
    from pathlib import Path

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    from characters.introspection_common import resolve_remote_adapter_dir
    from characters.response_generation import write_jsonl_rows
    from characters.self_reflection import generate_self_reflection_rows
    from characters.self_reflection_config import (
        SelfReflectionConfig,
        SelfReflectionGenerationConfig,
        SelfReflectionPathsConfig,
        SelfReflectionVllmConfig,
    )
    from characters.trl_dpo_config import TrlDpoConfig

    source_config = TrlDpoConfig.model_validate(source_config_dict)
    adapter_dir = resolve_remote_adapter_dir(source_config, REMOTE_RESULTS_ROOT)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"DPO adapter directory not found: {adapter_dir}")

    config = SelfReflectionConfig(
        name=config_name,
        source_trl_config=Path("/dev/null"),
        paths=SelfReflectionPathsConfig(output_path=Path("/dev/null")),
        generation=SelfReflectionGenerationConfig(**generation),
        vllm=SelfReflectionVllmConfig(**vllm_runtime),
    )
    output_path = adapter_dir / "self_reflection" / f"{config.name}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    llm = LLM(
        model=source_config.model.name,
        enable_lora=True,
        max_lora_rank=config.vllm.max_lora_rank,
        max_loras=1,
        max_model_len=config.vllm.max_model_len,
        max_num_seqs=config.vllm.max_num_seqs,
        max_num_batched_tokens=config.vllm.max_num_batched_tokens,
        gpu_memory_utilization=config.vllm.gpu_memory_utilization,
        tensor_parallel_size=config.vllm.tensor_parallel_size or 1,
        trust_remote_code=config.vllm.trust_remote_code or source_config.model.trust_remote_code,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        max_tokens=config.generation.max_new_tokens,
        temperature=config.generation.temperature,
        top_p=config.generation.top_p,
    )
    lora_request = LoRARequest(
        lora_name=source_config.name,
        lora_int_id=1,
        lora_path=str(adapter_dir),
    )

    def generate_batch(messages_batch: list[list[dict[str, str]]]) -> list[str]:
        prompts = [
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            for messages in messages_batch
        ]
        outputs = llm.generate(
            prompts,
            sampling_params=sampling_params,
            lora_request=lora_request,
            use_tqdm=True,
        )
        return [output.outputs[0].text if output.outputs else "" for output in outputs]

    rows = generate_self_reflection_rows(
        config,
        source_config,
        traits=traits,
        constitution=constitution,
        adapter_dir=adapter_dir,
        generate_batch=generate_batch,
    )
    write_jsonl_rows(output_path, rows)
    results_volume.commit()
    hf_cache_volume.commit()

    return {
        "config_name": config.name,
        "adapter_dir": str(adapter_dir),
        "output_path": str(output_path),
        "rows": len(rows),
        "prompt_count": len({str(row["prompt_name"]) for row in rows}),
        "base_model": source_config.model.name,
    }


@app.local_entrypoint()
def main(config: str = "configs/self_reflection/adversarial_skeptic_llama31.yaml") -> None:
    from characters.introspection_common import extract_traits_from_trl_config
    from characters.self_reflection_config import load_self_reflection_config
    from characters.teacher_generation import load_constitution_block
    from characters.trl_dpo_config import load_trl_dpo_config

    reflection_config = load_self_reflection_config(config)
    source_config = load_trl_dpo_config(reflection_config.source_trl_config)
    traits = extract_traits_from_trl_config(source_config)
    constitution = (
        load_constitution_block(reflection_config.paths.constitution_path)
        if reflection_config.paths.constitution_path is not None
        else None
    )

    result = run_self_reflection_remote.remote(
        source_config.model_dump(mode="json"),
        traits=traits,
        constitution=constitution,
        config_name=reflection_config.name,
        generation={
            "samples_per_prompt": reflection_config.generation.samples_per_prompt,
            "temperature": reflection_config.generation.temperature,
            "top_p": reflection_config.generation.top_p,
            "max_new_tokens": reflection_config.generation.max_new_tokens,
        },
        vllm_runtime={
            "max_model_len": reflection_config.vllm.max_model_len,
            "gpu_memory_utilization": reflection_config.vllm.gpu_memory_utilization,
            "max_lora_rank": reflection_config.vllm.max_lora_rank,
            "max_num_seqs": reflection_config.vllm.max_num_seqs,
            "max_num_batched_tokens": reflection_config.vllm.max_num_batched_tokens,
            "tensor_parallel_size": reflection_config.vllm.tensor_parallel_size,
            "trust_remote_code": reflection_config.vllm.trust_remote_code,
        },
    )
    print(
        json.dumps(
            {
                **result,
                "configured_local_output_path": str(reflection_config.paths.output_path),
            },
            indent=2,
        )
    )
