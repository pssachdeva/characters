from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class SelfReflectionPathsConfig:
    output_path: Path


@dataclass(slots=True)
class SelfReflectionGenerationConfig:
    samples_per_prompt: int = 1000
    temperature: float = 0.7
    top_p: float = 0.95
    max_new_tokens: int = 2048


@dataclass(slots=True)
class SelfReflectionVllmConfig:
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.9
    max_lora_rank: int = 64
    max_num_seqs: int = 1024
    max_num_batched_tokens: int = 32768
    tensor_parallel_size: int | None = None
    trust_remote_code: bool = False


@dataclass(slots=True)
class SelfReflectionConfig:
    name: str
    source_trl_config: Path
    paths: SelfReflectionPathsConfig
    generation: SelfReflectionGenerationConfig
    vllm: SelfReflectionVllmConfig


def _require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required config key: {key}")
    return mapping[key]


def _resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return path.resolve()


def load_self_reflection_config(path: str | Path) -> SelfReflectionConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    paths_raw = _require(raw, "paths")
    if isinstance(paths_raw, str):
        raise ValueError("paths must be a mapping with output")
    paths = SelfReflectionPathsConfig(
        output_path=_resolve_path(_require(paths_raw, "output")),
    )

    generation_raw = raw.get("generation", {})
    generation = SelfReflectionGenerationConfig(
        samples_per_prompt=int(generation_raw.get("samples_per_prompt", 1000)),
        temperature=float(generation_raw.get("temperature", 0.7)),
        top_p=float(generation_raw.get("top_p", 0.95)),
        max_new_tokens=int(generation_raw.get("max_new_tokens", 2048)),
    )

    vllm_raw = raw.get("vllm", {})
    vllm = SelfReflectionVllmConfig(
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

    config = SelfReflectionConfig(
        name=str(_require(raw, "name")),
        source_trl_config=_resolve_path(_require(raw, "source_trl_config")),
        paths=paths,
        generation=generation,
        vllm=vllm,
    )
    _validate_config(config)
    return config


def _validate_config(config: SelfReflectionConfig) -> None:
    if config.generation.samples_per_prompt <= 0:
        raise ValueError("generation.samples_per_prompt must be positive")
    if config.generation.max_new_tokens <= 0:
        raise ValueError("generation.max_new_tokens must be positive")
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
