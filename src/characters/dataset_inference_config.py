from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml


SUPPORTED_SOURCES = {"local", "modal"}
SUPPORTED_MODAL_VOLUMES = {"characters-dpo-data", "characters-trl-dpo-results"}


@dataclass(slots=True)
class DatasetInferencePathsConfig:
    output_path: Path
    remote_output_path: str


@dataclass(slots=True)
class DatasetInferenceAdapterConfig:
    name: str | None = None
    volume: str = "characters-trl-dpo-results"
    path: str | None = None
    weight: float = 1.0

    @property
    def effective_path(self) -> str:
        if self.path:
            return _normalize_modal_path(self.path)
        if self.name:
            return f"/{self.name}"
        raise ValueError("Adapter entries require name or path.")


@dataclass(slots=True)
class DatasetInferenceModelConfig:
    base_model: str
    adapter_enabled: bool = True
    adapter_name: str | None = None
    adapter_volume: str = "characters-trl-dpo-results"
    adapter_path: str | None = None
    adapters: list[DatasetInferenceAdapterConfig] | None = None
    trust_remote_code: bool = False

    @property
    def effective_adapter_path(self) -> str | None:
        if not self.adapter_enabled:
            return None
        if self.adapters:
            return None
        if self.adapter_path:
            return _normalize_modal_path(self.adapter_path)
        if self.adapter_name:
            return f"/{self.adapter_name}"
        return None

    @property
    def effective_adapter_name(self) -> str | None:
        if self.adapter_name:
            return self.adapter_name
        if self.adapters:
            adapter_names = [adapter.name for adapter in self.adapters if adapter.name]
            return "weighted_" + "_".join(adapter_names) if adapter_names else "weighted_adapter"
        return None


@dataclass(slots=True)
class DatasetInferenceDatasetConfig:
    source: Literal["local", "modal"]
    path: str | Path
    split: str = "train"
    prompt_key: str = "prompt"
    volume: str = "characters-dpo-data"
    upload_path: str | None = None
    limit: int | None = None

    @property
    def local_path(self) -> Path:
        return _resolve_path(self.path)

    @property
    def modal_path(self) -> str:
        return _normalize_modal_path(str(self.path))

    @property
    def effective_upload_path(self) -> str:
        return _normalize_modal_path(self.upload_path or "/dataset_inference/dataset.jsonl")


@dataclass(slots=True)
class DatasetInferenceGenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass(slots=True)
class DatasetInferenceVllmConfig:
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    max_lora_rank: int = 64
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 32768
    tensor_parallel_size: int | None = None


@dataclass(slots=True)
class DatasetInferenceConfig:
    name: str
    paths: DatasetInferencePathsConfig
    model: DatasetInferenceModelConfig
    dataset: DatasetInferenceDatasetConfig
    generation: DatasetInferenceGenerationConfig
    vllm: DatasetInferenceVllmConfig


def load_dataset_inference_config(path: str | Path) -> DatasetInferenceConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("Inference config must be a mapping.")

    name = str(_require(raw, "name"))
    paths_raw = _require_mapping(raw, "paths")
    output_path = _resolve_path(_require(paths_raw, "output"))
    remote_output = paths_raw.get("remote_output")
    if remote_output is None:
        remote_output = f"/inference/{name}/{output_path.name}"
    paths = DatasetInferencePathsConfig(
        output_path=output_path,
        remote_output_path=_normalize_modal_path(str(remote_output)),
    )

    model_raw = _require_mapping(raw, "model")
    model = _parse_model_config(model_raw)

    dataset_raw = _require_mapping(raw, "dataset")
    dataset = DatasetInferenceDatasetConfig(
        source=str(_require(dataset_raw, "source")),  # type: ignore[arg-type]
        path=str(_require(dataset_raw, "path")),
        split=str(dataset_raw.get("split", "train")),
        prompt_key=str(dataset_raw.get("prompt_key", "prompt")),
        volume=str(dataset_raw.get("volume", "characters-dpo-data")),
        upload_path=(
            str(dataset_raw["upload_path"])
            if dataset_raw.get("upload_path") is not None
            else f"/{name}/dataset.jsonl"
        ),
        limit=(
            int(dataset_raw["limit"])
            if dataset_raw.get("limit") is not None
            else None
        ),
    )

    generation_raw = raw.get("generation", {})
    if not isinstance(generation_raw, dict):
        raise ValueError("generation must be a mapping when provided.")
    generation = DatasetInferenceGenerationConfig(
        max_new_tokens=int(generation_raw.get("max_new_tokens", 512)),
        temperature=float(generation_raw.get("temperature", 0.7)),
        top_p=float(generation_raw.get("top_p", 0.9)),
    )

    vllm_raw = raw.get("vllm", {})
    if not isinstance(vllm_raw, dict):
        raise ValueError("vllm must be a mapping when provided.")
    vllm = DatasetInferenceVllmConfig(
        max_model_len=int(vllm_raw.get("max_model_len", 4096)),
        gpu_memory_utilization=float(vllm_raw.get("gpu_memory_utilization", 0.9)),
        max_lora_rank=int(vllm_raw.get("max_lora_rank", 64)),
        max_num_seqs=int(vllm_raw.get("max_num_seqs", 256)),
        max_num_batched_tokens=int(vllm_raw.get("max_num_batched_tokens", 32768)),
        tensor_parallel_size=(
            int(vllm_raw["tensor_parallel_size"])
            if vllm_raw.get("tensor_parallel_size") is not None
            else None
        ),
    )

    config = DatasetInferenceConfig(
        name=name,
        paths=paths,
        model=model,
        dataset=dataset,
        generation=generation,
        vllm=vllm,
    )
    _validate_config(config, require_local_file=True)
    return config


def dataset_inference_config_from_dict(raw: dict[str, Any]) -> DatasetInferenceConfig:
    config = DatasetInferenceConfig(
        name=str(raw["name"]),
        paths=DatasetInferencePathsConfig(
            output_path=Path(str(raw["paths"]["output_path"])),
            remote_output_path=str(raw["paths"]["remote_output_path"]),
        ),
        model=_parse_model_config(raw["model"]),
        dataset=DatasetInferenceDatasetConfig(**raw["dataset"]),
        generation=DatasetInferenceGenerationConfig(**raw["generation"]),
        vllm=DatasetInferenceVllmConfig(**raw["vllm"]),
    )
    _validate_config(config, require_local_file=False)
    return config


def _require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required config key: {key}")
    return mapping[key]


def _require_mapping(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = _require(mapping, key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping.")
    return value


def _parse_model_config(model_raw: dict[str, Any]) -> DatasetInferenceModelConfig:
    adapters_raw = model_raw.get("adapters")
    adapters = None
    if adapters_raw is not None:
        if not isinstance(adapters_raw, list):
            raise ValueError("model.adapters must be a list when provided.")
        adapters = []
        for index, adapter_raw in enumerate(adapters_raw):
            if not isinstance(adapter_raw, dict):
                raise ValueError(f"model.adapters[{index}] must be a mapping.")
            adapters.append(
                DatasetInferenceAdapterConfig(
                    name=(
                        str(adapter_raw["name"])
                        if adapter_raw.get("name") is not None
                        else None
                    ),
                    volume=str(adapter_raw.get("volume", "characters-trl-dpo-results")),
                    path=(
                        str(adapter_raw["path"])
                        if adapter_raw.get("path") is not None
                        else None
                    ),
                    weight=float(adapter_raw.get("weight", 1.0)),
                )
            )
    return DatasetInferenceModelConfig(
        base_model=str(_require(model_raw, "base_model")),
        adapter_enabled=bool(model_raw.get("adapter_enabled", True)),
        adapter_name=(
            str(model_raw["adapter_name"])
            if model_raw.get("adapter_name") is not None
            else None
        ),
        adapter_volume=str(model_raw.get("adapter_volume", "characters-trl-dpo-results")),
        adapter_path=(
            str(model_raw["adapter_path"])
            if model_raw.get("adapter_path") is not None
            else None
        ),
        adapters=adapters,
        trust_remote_code=bool(model_raw.get("trust_remote_code", False)),
    )


def _resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return path.resolve()


def _normalize_modal_path(raw_path: str) -> str:
    path = raw_path.strip()
    if not path:
        raise ValueError("Modal volume paths must not be empty.")
    return path if path.startswith("/") else f"/{path}"


def _validate_config(config: DatasetInferenceConfig, *, require_local_file: bool = True) -> None:
    if config.dataset.source not in SUPPORTED_SOURCES:
        raise ValueError(f"dataset.source must be one of {sorted(SUPPORTED_SOURCES)}")
    if require_local_file and config.dataset.source == "local" and not config.dataset.local_path.exists():
        raise FileNotFoundError(f"Local inference dataset not found: {config.dataset.local_path}")
    if config.dataset.source == "modal" and config.dataset.volume not in SUPPORTED_MODAL_VOLUMES:
        raise ValueError(
            "dataset.volume must be one of "
            f"{sorted(SUPPORTED_MODAL_VOLUMES)} when dataset.source=modal"
        )
    if config.model.adapter_volume not in SUPPORTED_MODAL_VOLUMES:
        raise ValueError(f"model.adapter_volume must be one of {sorted(SUPPORTED_MODAL_VOLUMES)}")
    if config.model.adapters:
        for index, adapter in enumerate(config.model.adapters):
            if adapter.volume not in SUPPORTED_MODAL_VOLUMES:
                raise ValueError(
                    f"model.adapters[{index}].volume must be one of {sorted(SUPPORTED_MODAL_VOLUMES)}"
                )
            if not adapter.name and not adapter.path:
                raise ValueError(f"model.adapters[{index}] requires name or path")
        if config.model.adapter_path:
            raise ValueError("model.adapter_path cannot be combined with model.adapters")
    if (
        config.model.adapter_enabled
        and not config.model.effective_adapter_path
        and not config.model.adapters
    ):
        raise ValueError("model.adapter_name, model.adapter_path, or model.adapters is required when adapter_enabled=true")
    if config.dataset.limit is not None and config.dataset.limit <= 0:
        raise ValueError("dataset.limit must be positive when provided")
    if not config.dataset.prompt_key:
        raise ValueError("dataset.prompt_key must not be empty")
    if config.generation.max_new_tokens <= 0:
        raise ValueError("generation.max_new_tokens must be positive")
    if config.generation.temperature < 0:
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
        raise ValueError("vllm.tensor_parallel_size must be positive when provided")
