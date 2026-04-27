import json
from pathlib import Path

import modal


APP_NAME = "characters-self-interaction"
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
def run_self_interaction_remote(
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
    from characters.self_interaction import generate_self_interaction_rows
    from characters.self_interaction_config import (
        SelfInteractionConfig,
        SelfInteractionGenerationConfig,
        SelfInteractionPathsConfig,
        SelfInteractionVllmConfig,
    )
    from characters.trl_dpo_config import TrlDpoConfig

    source_config = TrlDpoConfig.model_validate(source_config_dict)
    adapter_dir = resolve_remote_adapter_dir(source_config, REMOTE_RESULTS_ROOT)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"DPO adapter directory not found: {adapter_dir}")

    config = SelfInteractionConfig(
        name=config_name,
        source_trl_config=Path("/dev/null"),
        paths=SelfInteractionPathsConfig(output_path=Path("/dev/null")),
        generation=SelfInteractionGenerationConfig(**generation),
        vllm=SelfInteractionVllmConfig(**vllm_runtime),
    )
    output_path = adapter_dir / "self_interaction" / f"{config.name}.jsonl"
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
        max_tokens=config.generation.max_new_tokens_per_turn,
        temperature=config.generation.temperature,
        top_p=config.generation.top_p,
    )
    lora_request = LoRARequest(
        lora_name=source_config.name,
        lora_int_id=1,
        lora_path=str(adapter_dir),
    )

    def generate_batch(messages_batch: list[list[dict[str, str]]]) -> list[str]:
        prompts = tokenizer.apply_chat_template(
            messages_batch,
            tokenize=True,
            add_generation_prompt=True,
        )
        truncated_prompts: list[str] = []
        max_prompt_tokens = config.vllm.max_model_len - config.generation.max_new_tokens_per_turn
        for prompt_tokens in prompts:
            if len(prompt_tokens) > max_prompt_tokens:
                prompt_tokens = prompt_tokens[-max_prompt_tokens:]
            truncated_prompts.append(tokenizer.decode(prompt_tokens, skip_special_tokens=False))
        outputs = llm.generate(
            truncated_prompts,
            sampling_params=sampling_params,
            lora_request=lora_request,
            use_tqdm=True,
        )
        return [output.outputs[0].text if output.outputs else "" for output in outputs]

    rows = generate_self_interaction_rows(
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
        "base_model": source_config.model.name,
        "sources": sorted({str(row["source"]) for row in rows}),
    }


@app.local_entrypoint()
def main(config: str = "configs/self_interaction/adversarial_skeptic_llama31.yaml") -> None:
    from characters.introspection_common import extract_traits_from_trl_config
    from characters.self_interaction_config import load_self_interaction_config
    from characters.teacher_generation import load_constitution_block
    from characters.trl_dpo_config import load_trl_dpo_config

    interaction_config = load_self_interaction_config(config)
    source_config = load_trl_dpo_config(interaction_config.source_trl_config)
    traits = extract_traits_from_trl_config(source_config)
    constitution = (
        load_constitution_block(interaction_config.paths.constitution_path)
        if interaction_config.paths.constitution_path is not None
        else None
    )

    result = run_self_interaction_remote.remote(
        source_config.model_dump(mode="json"),
        traits=traits,
        constitution=constitution,
        config_name=interaction_config.name,
        generation={
            "free_guidance_conversations": interaction_config.generation.free_guidance_conversations,
            "leading_guidance_conversations": interaction_config.generation.leading_guidance_conversations,
            "turns_per_conversation": interaction_config.generation.turns_per_conversation,
            "max_new_tokens_per_turn": interaction_config.generation.max_new_tokens_per_turn,
            "temperature": interaction_config.generation.temperature,
            "top_p": interaction_config.generation.top_p,
        },
        vllm_runtime={
            "max_model_len": interaction_config.vllm.max_model_len,
            "gpu_memory_utilization": interaction_config.vllm.gpu_memory_utilization,
            "max_lora_rank": interaction_config.vllm.max_lora_rank,
            "max_num_seqs": interaction_config.vllm.max_num_seqs,
            "max_num_batched_tokens": interaction_config.vllm.max_num_batched_tokens,
            "tensor_parallel_size": interaction_config.vllm.tensor_parallel_size,
            "trust_remote_code": interaction_config.vllm.trust_remote_code,
        },
    )
    print(
        json.dumps(
            {
                **result,
                "configured_local_output_path": str(interaction_config.paths.output_path),
            },
            indent=2,
        )
    )
