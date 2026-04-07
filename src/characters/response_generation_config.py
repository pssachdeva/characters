from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from characters.prompt_expansion_config import ModelConfig, SamplingConfig


@dataclass(slots=True)
class ResponseGenerationPathsConfig:
    input_path: Path
    prompt_path: Path
    output_path: Path
    constitution_path: Path | None = None


@dataclass(slots=True)
class ResponseGenerationConfig:
    name: str
    paths: ResponseGenerationPathsConfig
    model: ModelConfig
    sampling: SamplingConfig
    n_samples_per_prompt: int = 1


def _require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required config key: {key}")
    return mapping[key]


def _resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return path.resolve()


def load_response_generation_config(path: str | Path) -> ResponseGenerationConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    paths_raw = _require(raw, "paths")
    if isinstance(paths_raw, str):
        raise ValueError("paths must be a mapping with input, prompt, and output")
    paths = ResponseGenerationPathsConfig(
        input_path=_resolve_path(_require(paths_raw, "input")),
        prompt_path=_resolve_path(_require(paths_raw, "prompt")),
        output_path=_resolve_path(_require(paths_raw, "output")),
        constitution_path=(
            _resolve_path(paths_raw["constitution"])
            if "constitution" in paths_raw and paths_raw["constitution"] is not None
            else None
        ),
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

    sampling_raw = raw.get("sampling", {})
    sampling = SamplingConfig(
        temperature=float(sampling_raw.get("temperature", 0.7)),
        top_p=float(sampling_raw.get("top_p", 0.95)),
        max_tokens=int(sampling_raw.get("max_tokens", 2048)),
    )

    config = ResponseGenerationConfig(
        name=str(_require(raw, "name")),
        paths=paths,
        model=model,
        sampling=sampling,
        n_samples_per_prompt=int(raw.get("n_samples_per_prompt", 1)),
    )
    _validate_config(config)
    return config


def _validate_config(config: ResponseGenerationConfig) -> None:
    if config.model.provider not in {"openai", "openrouter", "anthropic", "google"}:
        raise ValueError(f"Unsupported provider: {config.model.provider}")
    if config.model.max_concurrency <= 0:
        raise ValueError("model.max_concurrency must be positive")
    if config.sampling.max_tokens <= 0:
        raise ValueError("sampling.max_tokens must be positive")
    if config.n_samples_per_prompt <= 0:
        raise ValueError("n_samples_per_prompt must be positive")
