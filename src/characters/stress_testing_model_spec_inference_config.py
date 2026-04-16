from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


SUPPORTED_DATASET_CONFIGS = {"default", "complete", "judge_evaluations"}


@dataclass(slots=True)
class StressTestingModelSpecInferencePathsConfig:
    output_path: Path


@dataclass(slots=True)
class StressTestingModelSpecInferenceModelConfig:
    base_model: str
    adapter_enabled: bool = True
    adapter_name: str | None = None


@dataclass(slots=True)
class StressTestingModelSpecDatasetConfig:
    config_name: str
    split: str
    limit: int | None = None
    shuffle: bool = False
    seed: int = 42


@dataclass(slots=True)
class StressTestingModelSpecInferenceGenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    n_samples_per_prompt: int = 1


@dataclass(slots=True)
class StressTestingModelSpecInferenceVllmConfig:
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.9
    max_lora_rank: int = 64
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 32768
    tensor_parallel_size: int | None = None
    trust_remote_code: bool = False


@dataclass(slots=True)
class StressTestingModelSpecInferenceConfig:
    name: str
    paths: StressTestingModelSpecInferencePathsConfig
    model: StressTestingModelSpecInferenceModelConfig
    dataset: StressTestingModelSpecDatasetConfig
    generation: StressTestingModelSpecInferenceGenerationConfig
    vllm: StressTestingModelSpecInferenceVllmConfig


def _require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required config key: {key}")
    return mapping[key]


def _resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return path.resolve()


def load_stress_testing_model_spec_inference_config(
    path: str | Path,
) -> StressTestingModelSpecInferenceConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    paths_raw = _require(raw, "paths")
    if isinstance(paths_raw, str):
        raise ValueError("paths must be a mapping with output")
    paths = StressTestingModelSpecInferencePathsConfig(
        output_path=_resolve_path(_require(paths_raw, "output")),
    )

    model_raw = _require(raw, "model")
    if isinstance(model_raw, str):
        raise ValueError("model must be a mapping")
    model = StressTestingModelSpecInferenceModelConfig(
        base_model=str(_require(model_raw, "base_model")),
        adapter_enabled=bool(model_raw.get("adapter_enabled", True)),
        adapter_name=(
            str(model_raw["adapter_name"])
            if model_raw.get("adapter_name") is not None
            else None
        ),
    )

    dataset_raw = _require(raw, "dataset")
    if isinstance(dataset_raw, str):
        raise ValueError("dataset must be a mapping")
    dataset = StressTestingModelSpecDatasetConfig(
        config_name=str(_require(dataset_raw, "config_name")),
        split=str(_require(dataset_raw, "split")),
        limit=(
            int(dataset_raw["limit"])
            if dataset_raw.get("limit") is not None
            else None
        ),
        shuffle=bool(dataset_raw.get("shuffle", False)),
        seed=int(dataset_raw.get("seed", 42)),
    )

    generation_raw = raw.get("generation", {})
    generation = StressTestingModelSpecInferenceGenerationConfig(
        max_new_tokens=int(generation_raw.get("max_new_tokens", 512)),
        temperature=float(generation_raw.get("temperature", 0.0)),
        top_p=float(generation_raw.get("top_p", 1.0)),
        n_samples_per_prompt=int(generation_raw.get("n_samples_per_prompt", 1)),
    )

    vllm_raw = raw.get("vllm", {})
    vllm = StressTestingModelSpecInferenceVllmConfig(
        max_model_len=int(vllm_raw.get("max_model_len", 8192)),
        gpu_memory_utilization=float(vllm_raw.get("gpu_memory_utilization", 0.9)),
        max_lora_rank=int(vllm_raw.get("max_lora_rank", 64)),
        max_num_seqs=int(vllm_raw.get("max_num_seqs", 256)),
        max_num_batched_tokens=int(vllm_raw.get("max_num_batched_tokens", 32768)),
        tensor_parallel_size=(
            int(vllm_raw["tensor_parallel_size"])
            if vllm_raw.get("tensor_parallel_size") is not None
            else None
        ),
        trust_remote_code=bool(vllm_raw.get("trust_remote_code", False)),
    )

    config = StressTestingModelSpecInferenceConfig(
        name=str(_require(raw, "name")),
        paths=paths,
        model=model,
        dataset=dataset,
        generation=generation,
        vllm=vllm,
    )
    _validate_config(config)
    return config


def safe_model_slug(model_name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name.strip())
    slug = slug.strip("._-")
    return slug or "model"


def _validate_config(config: StressTestingModelSpecInferenceConfig) -> None:
    if config.dataset.config_name not in SUPPORTED_DATASET_CONFIGS:
        raise ValueError(
            f"dataset.config_name must be one of {sorted(SUPPORTED_DATASET_CONFIGS)}"
        )
    if not config.dataset.split:
        raise ValueError("dataset.split must not be empty")
    if config.dataset.limit is not None and config.dataset.limit <= 0:
        raise ValueError("dataset.limit must be positive when set")
    if config.generation.max_new_tokens <= 0:
        raise ValueError("generation.max_new_tokens must be positive")
    if config.generation.n_samples_per_prompt <= 0:
        raise ValueError("generation.n_samples_per_prompt must be positive")
    if config.generation.temperature < 0:
        raise ValueError("generation.temperature must be non-negative")
    if not 0 < config.generation.top_p <= 1:
        raise ValueError("generation.top_p must be in (0, 1]")
    if config.model.adapter_enabled and not config.model.adapter_name:
        raise ValueError("model.adapter_name is required when model.adapter_enabled=true")
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
