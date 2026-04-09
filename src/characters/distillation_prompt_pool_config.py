from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class DistillationPromptPoolPathsConfig:
    constitution_input_path: Path
    output_path: Path


@dataclass(slots=True)
class PromptSourceConfig:
    type: str
    path: Path
    source: str
    hf_dataset: str | None = None
    hf_split: str | None = None
    prompt_field: str = "prompt"
    conversation_field: str = "conversations"
    role_field: str = "role"
    content_field: str = "content"
    user_role: str = "user"
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class DistillationPromptPoolConfig:
    name: str
    paths: DistillationPromptPoolPathsConfig
    sources: list[PromptSourceConfig]


def _require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required config key: {key}")
    return mapping[key]


def _resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return path.resolve()


def load_distillation_prompt_pool_config(path: str | Path) -> DistillationPromptPoolConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    paths_raw = _require(raw, "paths")
    if isinstance(paths_raw, str):
        raise ValueError("paths must be a mapping with constitution_input and output")
    paths = DistillationPromptPoolPathsConfig(
        constitution_input_path=_resolve_path(_require(paths_raw, "constitution_input")),
        output_path=_resolve_path(_require(paths_raw, "output")),
    )
    sources = _load_prompt_sources(raw.get("sources"))

    config = DistillationPromptPoolConfig(
        name=str(_require(raw, "name")),
        paths=paths,
        sources=sources,
    )
    _validate_config(config)
    return config


def _load_prompt_sources(raw_sources: object) -> list[PromptSourceConfig]:
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("sources must be a non-empty list of prompt source mappings")
    sources: list[PromptSourceConfig] = []
    for index, raw_source in enumerate(raw_sources):
        if not isinstance(raw_source, dict):
            raise ValueError(f"sources[{index}] must be a mapping")
        metadata_raw = raw_source.get("metadata", {})
        if not isinstance(metadata_raw, dict):
            raise ValueError(f"sources[{index}].metadata must be a mapping when provided")
        sources.append(
            PromptSourceConfig(
                type=str(_require(raw_source, "type")),
                path=_resolve_path(_require(raw_source, "path")),
                source=str(_require(raw_source, "source")),
                hf_dataset=(
                    str(raw_source["hf_dataset"])
                    if "hf_dataset" in raw_source and raw_source["hf_dataset"] is not None
                    else None
                ),
                hf_split=(
                    str(raw_source["hf_split"])
                    if "hf_split" in raw_source and raw_source["hf_split"] is not None
                    else None
                ),
                prompt_field=str(raw_source.get("prompt_field", "prompt")),
                conversation_field=str(raw_source.get("conversation_field", "conversations")),
                role_field=str(raw_source.get("role_field", "role")),
                content_field=str(raw_source.get("content_field", "content")),
                user_role=str(raw_source.get("user_role", "user")),
                metadata={str(key): str(value) for key, value in metadata_raw.items()},
            )
        )
    return sources


def _validate_config(config: DistillationPromptPoolConfig) -> None:
    supported_types = {"lima", "jsonl_first_user_turn", "jsonl_prompt_field"}
    for index, source in enumerate(config.sources):
        if source.type not in supported_types:
            raise ValueError(
                f"Unsupported sources[{index}].type: {source.type}. "
                f"Expected one of {sorted(supported_types)}"
            )
        if not source.source:
            raise ValueError(f"sources[{index}].source must not be empty")
