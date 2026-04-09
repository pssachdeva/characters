from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class DpoDatasetPathsConfig:
    teacher_input_path: Path
    student_input_path: Path
    output_dir: Path


@dataclass(slots=True)
class DpoDatasetTargetModelConfig:
    name: str
    tokenizer_name: str | None
    apply_chat_template: bool = True

    @property
    def effective_tokenizer_name(self) -> str:
        return self.tokenizer_name or self.name


@dataclass(slots=True)
class DpoDatasetFormatConfig:
    type: str = "openrlhf_chat"
    chosen_key: str = "chosen"
    rejected_key: str = "rejected"


@dataclass(slots=True)
class DpoDatasetFiltersConfig:
    max_length: int = 1024
    drop_empty: bool = True
    drop_identical_pairs: bool = True
    require_terminal_punctuation: bool = False
    drop_overlength: bool = True


@dataclass(slots=True)
class DpoDatasetSplitsConfig:
    train: float = 0.85
    val: float = 0.15
    seed: int = 123
    group_by: str = "prompt"


@dataclass(slots=True)
class DpoDatasetMetadataConfig:
    keep_trait: bool = True
    keep_source: bool = True
    keep_sample_index: bool = True
    keep_models: bool = True


@dataclass(slots=True)
class DpoDatasetConfig:
    name: str
    paths: DpoDatasetPathsConfig
    target_model: DpoDatasetTargetModelConfig
    format: DpoDatasetFormatConfig
    filters: DpoDatasetFiltersConfig
    splits: DpoDatasetSplitsConfig
    metadata: DpoDatasetMetadataConfig


def _require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required config key: {key}")
    return mapping[key]


def _resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return path.resolve()


def load_dpo_dataset_config(path: str | Path) -> DpoDatasetConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    paths_raw = _require(raw, "paths")
    if isinstance(paths_raw, str):
        raise ValueError("paths must be a mapping with teacher_input, student_input, and output_dir")
    paths = DpoDatasetPathsConfig(
        teacher_input_path=_resolve_path(_require(paths_raw, "teacher_input")),
        student_input_path=_resolve_path(_require(paths_raw, "student_input")),
        output_dir=_resolve_path(_require(paths_raw, "output_dir")),
    )

    target_model_raw = _require(raw, "target_model")
    if isinstance(target_model_raw, str):
        raise ValueError("target_model must be a mapping with 'name'")
    target_model = DpoDatasetTargetModelConfig(
        name=str(_require(target_model_raw, "name")),
        tokenizer_name=(
            str(target_model_raw["tokenizer_name"])
            if target_model_raw.get("tokenizer_name") is not None
            else None
        ),
        apply_chat_template=bool(target_model_raw.get("apply_chat_template", True)),
    )

    format_raw = raw.get("format", {})
    if isinstance(format_raw, str):
        raise ValueError("format must be a mapping")
    format_config = DpoDatasetFormatConfig(
        type=str(format_raw.get("type", "openrlhf_chat")),
        chosen_key=str(format_raw.get("chosen_key", "chosen")),
        rejected_key=str(format_raw.get("rejected_key", "rejected")),
    )

    filters_raw = raw.get("filters", {})
    if isinstance(filters_raw, str):
        raise ValueError("filters must be a mapping")
    filters = DpoDatasetFiltersConfig(
        max_length=int(filters_raw.get("max_length", 1024)),
        drop_empty=bool(filters_raw.get("drop_empty", True)),
        drop_identical_pairs=bool(filters_raw.get("drop_identical_pairs", True)),
        require_terminal_punctuation=bool(filters_raw.get("require_terminal_punctuation", False)),
        drop_overlength=bool(filters_raw.get("drop_overlength", True)),
    )

    splits_raw = raw.get("splits", {})
    if isinstance(splits_raw, str):
        raise ValueError("splits must be a mapping")
    splits = DpoDatasetSplitsConfig(
        train=float(splits_raw.get("train", 0.85)),
        val=float(splits_raw.get("val", 0.15)),
        seed=int(splits_raw.get("seed", 123)),
        group_by=str(splits_raw.get("group_by", "prompt")),
    )

    metadata_raw = raw.get("metadata", {})
    if isinstance(metadata_raw, str):
        raise ValueError("metadata must be a mapping")
    metadata = DpoDatasetMetadataConfig(
        keep_trait=bool(metadata_raw.get("keep_trait", True)),
        keep_source=bool(metadata_raw.get("keep_source", True)),
        keep_sample_index=bool(metadata_raw.get("keep_sample_index", True)),
        keep_models=bool(metadata_raw.get("keep_models", True)),
    )

    config = DpoDatasetConfig(
        name=str(_require(raw, "name")),
        paths=paths,
        target_model=target_model,
        format=format_config,
        filters=filters,
        splits=splits,
        metadata=metadata,
    )
    _validate_config(config)
    return config


def _validate_config(config: DpoDatasetConfig) -> None:
    if config.format.type not in {
        "openrlhf_chat",
        "nemo_binary_preference",
        "trl_conversational",
    }:
        raise ValueError(f"Unsupported DPO dataset format: {config.format.type}")
    if config.format.chosen_key != "chosen" or config.format.rejected_key != "rejected":
        raise ValueError("Only chosen/rejected keys are currently supported")
    if config.filters.max_length <= 0:
        raise ValueError("filters.max_length must be positive")
    if config.splits.train < 0 or config.splits.val < 0:
        raise ValueError("splits.train and splits.val must be non-negative")
    if abs((config.splits.train + config.splits.val) - 1.0) > 1e-9:
        raise ValueError("splits.train and splits.val must sum to 1.0")
    if config.splits.group_by != "prompt":
        raise ValueError(f"Unsupported splits.group_by: {config.splits.group_by}")
