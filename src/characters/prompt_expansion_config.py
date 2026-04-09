from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class SamplingConfig:
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 2048


@dataclass(slots=True)
class ModelConfig:
    provider: str
    name: str
    base_url: str | None = None
    site_url: str | None = None
    app_name: str | None = None
    max_concurrency: int = 8


@dataclass(slots=True)
class PathsConfig:
    constitution_path: Path
    prompt_path: Path
    output_path: Path


@dataclass(slots=True)
class LengthDistributionConfig:
    short: int
    medium: int
    long: int


@dataclass(slots=True)
class TraitsConfig:
    additional_questions_per_trait: int
    max_attempts: int


@dataclass(slots=True)
class PromptExpansionConfig:
    name: str
    paths: PathsConfig
    model: ModelConfig
    traits: TraitsConfig
    length_distribution: LengthDistributionConfig
    sampling: SamplingConfig


def _require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required config key: {key}")
    return mapping[key]


def _resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return path.resolve()


def load_prompt_expansion_config(path: str | Path) -> PromptExpansionConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    sampling_raw = raw.get("sampling", {})
    sampling = SamplingConfig(
        temperature=float(sampling_raw.get("temperature", 0.7)),
        top_p=float(sampling_raw.get("top_p", 0.95)),
        max_tokens=int(sampling_raw.get("max_tokens", 2048)),
    )

    model_raw = _require(raw, "model")
    if isinstance(model_raw, str):
        raise ValueError("model must be a mapping with 'provider' and 'name'")
    model = ModelConfig(
        provider=str(_require(model_raw, "provider")),
        name=str(_require(model_raw, "name")),
        base_url=model_raw.get("base_url"),
        site_url=model_raw.get("site_url"),
        app_name=model_raw.get("app_name"),
        max_concurrency=int(model_raw.get("max_concurrency", 8)),
    )

    paths_raw = _require(raw, "paths")
    if isinstance(paths_raw, str):
        raise ValueError("paths must be a mapping with constitution, prompt, and output")
    paths = PathsConfig(
        constitution_path=_resolve_path(_require(paths_raw, "constitution")),
        prompt_path=_resolve_path(_require(paths_raw, "prompt")),
        output_path=_resolve_path(_require(paths_raw, "output")),
    )

    length_distribution_raw = raw.get("length_distribution", {})
    length_distribution = LengthDistributionConfig(
        short=int(length_distribution_raw.get("short", 15)),
        medium=int(length_distribution_raw.get("medium", 20)),
        long=int(length_distribution_raw.get("long", 15)),
    )

    traits_raw = raw.get("traits", {})
    if isinstance(traits_raw, str):
        raise ValueError("traits must be a mapping with additional_questions_per_trait and max_attempts")
    traits = TraitsConfig(
        additional_questions_per_trait=int(_require(traits_raw, "additional_questions_per_trait")),
        max_attempts=int(traits_raw.get("max_attempts", 4)),
    )

    config = PromptExpansionConfig(
        name=str(_require(raw, "name")),
        paths=paths,
        model=model,
        traits=traits,
        length_distribution=length_distribution,
        sampling=sampling,
    )
    _validate_config(config)
    return config


def _validate_config(config: PromptExpansionConfig) -> None:
    if config.model.provider not in {"openai", "openrouter", "anthropic", "google"}:
        raise ValueError(f"Unsupported provider: {config.model.provider}")
    if config.model.max_concurrency <= 0:
        raise ValueError("model.max_concurrency must be positive")
    if config.traits.additional_questions_per_trait <= 0:
        raise ValueError("traits.additional_questions_per_trait must be positive")
    if config.traits.max_attempts <= 0:
        raise ValueError("traits.max_attempts must be positive")
    if config.length_distribution.short < 0:
        raise ValueError("length_distribution.short must be non-negative")
    if config.length_distribution.medium < 0:
        raise ValueError("length_distribution.medium must be non-negative")
    if config.length_distribution.long < 0:
        raise ValueError("length_distribution.long must be non-negative")
    total_target_questions = (
        config.length_distribution.short
        + config.length_distribution.medium
        + config.length_distribution.long
    )
    if total_target_questions != config.traits.additional_questions_per_trait:
        raise ValueError(
            "length_distribution must sum to traits.additional_questions_per_trait"
        )
