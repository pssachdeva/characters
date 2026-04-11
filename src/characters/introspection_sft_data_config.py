from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class IntrospectionSftDataPathsConfig:
    reflection_input_path: Path
    interaction_input_path: Path
    output_dir: Path


@dataclass(slots=True)
class IntrospectionSftDataSplitsConfig:
    train: float = 0.85
    val: float = 0.15
    seed: int = 123


@dataclass(slots=True)
class IntrospectionSftDataConfig:
    name: str
    paths: IntrospectionSftDataPathsConfig
    splits: IntrospectionSftDataSplitsConfig


def _require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required config key: {key}")
    return mapping[key]


def _resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return path.resolve()


def load_introspection_sft_data_config(path: str | Path) -> IntrospectionSftDataConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    paths_raw = _require(raw, "paths")
    if isinstance(paths_raw, str):
        raise ValueError("paths must be a mapping with reflection_input, interaction_input, and output_dir")
    paths = IntrospectionSftDataPathsConfig(
        reflection_input_path=_resolve_path(_require(paths_raw, "reflection_input")),
        interaction_input_path=_resolve_path(_require(paths_raw, "interaction_input")),
        output_dir=_resolve_path(_require(paths_raw, "output_dir")),
    )

    splits_raw = raw.get("splits", {})
    splits = IntrospectionSftDataSplitsConfig(
        train=float(splits_raw.get("train", 0.85)),
        val=float(splits_raw.get("val", 0.15)),
        seed=int(splits_raw.get("seed", 123)),
    )

    config = IntrospectionSftDataConfig(
        name=str(_require(raw, "name")),
        paths=paths,
        splits=splits,
    )
    _validate_config(config)
    return config


def _validate_config(config: IntrospectionSftDataConfig) -> None:
    if config.splits.train < 0 or config.splits.val < 0:
        raise ValueError("splits.train and splits.val must be non-negative")
    if abs((config.splits.train + config.splits.val) - 1.0) > 1e-9:
        raise ValueError("splits.train and splits.val must sum to 1.0")
