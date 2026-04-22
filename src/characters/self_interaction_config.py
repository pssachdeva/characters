from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class SelfInteractionPathsConfig:
    output_path: Path
    constitution_path: Path | None = None


@dataclass(slots=True)
class SelfInteractionGenerationConfig:
    free_guidance_conversations: int = 1000
    leading_guidance_conversations: int = 1000
    turns_per_conversation: int = 10
    max_new_tokens_per_turn: int = 512
    temperature: float = 0.7
    top_p: float = 0.95


@dataclass(slots=True)
class SelfInteractionVllmConfig:
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.9
    max_lora_rank: int = 64
    max_num_seqs: int = 1024
    max_num_batched_tokens: int = 32768
    tensor_parallel_size: int | None = None
    trust_remote_code: bool = False


@dataclass(slots=True)
class SelfInteractionConfig:
    name: str
    source_trl_config: Path
    paths: SelfInteractionPathsConfig
    generation: SelfInteractionGenerationConfig
    vllm: SelfInteractionVllmConfig


def _require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required config key: {key}")
    return mapping[key]


def _resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return path.resolve()


def load_self_interaction_config(path: str | Path) -> SelfInteractionConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    paths_raw = _require(raw, "paths")
    if isinstance(paths_raw, str):
        raise ValueError("paths must be a mapping with output")
    paths = SelfInteractionPathsConfig(
        output_path=_resolve_path(_require(paths_raw, "output")),
        constitution_path=(
            _resolve_path(paths_raw["constitution"])
            if "constitution" in paths_raw and paths_raw["constitution"] is not None
            else None
        ),
    )

    generation_raw = raw.get("generation", {})
    generation = SelfInteractionGenerationConfig(
        free_guidance_conversations=int(generation_raw.get("free_guidance_conversations", 1000)),
        leading_guidance_conversations=int(
            generation_raw.get("leading_guidance_conversations", 1000)
        ),
        turns_per_conversation=int(generation_raw.get("turns_per_conversation", 10)),
        max_new_tokens_per_turn=int(generation_raw.get("max_new_tokens_per_turn", 512)),
        temperature=float(generation_raw.get("temperature", 0.7)),
        top_p=float(generation_raw.get("top_p", 0.95)),
    )

    vllm_raw = raw.get("vllm", {})
    vllm = SelfInteractionVllmConfig(
        max_model_len=int(vllm_raw.get("max_model_len", 8192)),
        gpu_memory_utilization=float(vllm_raw.get("gpu_memory_utilization", 0.9)),
        max_lora_rank=int(vllm_raw.get("max_lora_rank", 64)),
        max_num_seqs=int(vllm_raw.get("max_num_seqs", 1024)),
        max_num_batched_tokens=int(vllm_raw.get("max_num_batched_tokens", 32768)),
        tensor_parallel_size=(
            int(vllm_raw["tensor_parallel_size"])
            if vllm_raw.get("tensor_parallel_size") is not None
            else None
        ),
        trust_remote_code=bool(vllm_raw.get("trust_remote_code", False)),
    )

    config = SelfInteractionConfig(
        name=str(_require(raw, "name")),
        source_trl_config=_resolve_path(_require(raw, "source_trl_config")),
        paths=paths,
        generation=generation,
        vllm=vllm,
    )
    _validate_config(config)
    return config


def _validate_config(config: SelfInteractionConfig) -> None:
    if config.generation.free_guidance_conversations <= 0:
        raise ValueError("generation.free_guidance_conversations must be positive")
    if config.generation.leading_guidance_conversations <= 0:
        raise ValueError("generation.leading_guidance_conversations must be positive")
    if config.generation.turns_per_conversation <= 0:
        raise ValueError("generation.turns_per_conversation must be positive")
    if config.generation.max_new_tokens_per_turn <= 0:
        raise ValueError("generation.max_new_tokens_per_turn must be positive")
    if not 0 <= config.generation.temperature:
        raise ValueError("generation.temperature must be non-negative")
    if not 0 < config.generation.top_p <= 1:
        raise ValueError("generation.top_p must be in (0, 1]")
    if config.vllm.max_model_len <= 0:
        raise ValueError("vllm.max_model_len must be positive")
    if config.vllm.max_lora_rank <= 0:
        raise ValueError("vllm.max_lora_rank must be positive")
    if config.vllm.max_num_seqs <= 0:
        raise ValueError("vllm.max_num_seqs must be positive")
    if config.vllm.max_num_batched_tokens <= 0:
        raise ValueError("vllm.max_num_batched_tokens must be positive")
    if config.vllm.tensor_parallel_size is not None and config.vllm.tensor_parallel_size <= 0:
        raise ValueError("vllm.tensor_parallel_size must be positive when set")
