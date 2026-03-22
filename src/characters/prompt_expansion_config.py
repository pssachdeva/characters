from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class SamplingConfig:
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int | None = None
    max_tokens: int = 2048


@dataclass(slots=True)
class ProviderBackendConfig:
    provider: str
    api_key_env: str
    base_url: str | None = None


@dataclass(slots=True)
class ModalBackendConfig:
    app_name: str = "characters-prompt-expansion"
    gpu: str = "A100"
    timeout: int = 3600
    hf_secret_name: str | None = "huggingface-secret"
    hf_cache_volume_name: str = "huggingface-cache"
    hf_cache_dir: str = "/hf_cache"
    gpu_memory_utilization: float = 0.95
    max_model_len: int = 8192
    trust_remote_code: bool = True


@dataclass(slots=True)
class PromptExpansionConfig:
    name: str
    constitution_path: Path
    prompt_path: Path
    backend: str
    model: str
    output_path: Path
    target_additional_questions_per_trait: int
    max_attempts_per_trait: int
    short_count: int
    medium_count: int
    long_count: int
    sampling: SamplingConfig
    provider: ProviderBackendConfig | None = None
    modal: ModalBackendConfig | None = None


def _require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required config key: {key}")
    return mapping[key]


def _resolve_path(raw_path: str | Path) -> Path:
    return Path(raw_path)


def load_prompt_expansion_config(path: str | Path) -> PromptExpansionConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    sampling_raw = raw.get("sampling", {})
    sampling = SamplingConfig(
        temperature=float(sampling_raw.get("temperature", 0.7)),
        top_p=float(sampling_raw.get("top_p", 0.95)),
        top_k=sampling_raw.get("top_k"),
        max_tokens=int(sampling_raw.get("max_tokens", 2048)),
    )

    provider = None
    provider_raw = raw.get("provider")
    if provider_raw is not None:
        provider = ProviderBackendConfig(
            provider=str(_require(provider_raw, "provider")),
            api_key_env=str(_require(provider_raw, "api_key_env")),
            base_url=provider_raw.get("base_url"),
        )

    modal = None
    modal_raw = raw.get("modal")
    if modal_raw is not None:
        modal = ModalBackendConfig(
            app_name=str(modal_raw.get("app_name", "characters-prompt-expansion")),
            gpu=str(modal_raw.get("gpu", "A100")),
            timeout=int(modal_raw.get("timeout", 3600)),
            hf_secret_name=modal_raw.get("hf_secret_name", "huggingface-secret"),
            hf_cache_volume_name=str(modal_raw.get("hf_cache_volume_name", "huggingface-cache")),
            hf_cache_dir=str(modal_raw.get("hf_cache_dir", "/hf_cache")),
            gpu_memory_utilization=float(modal_raw.get("gpu_memory_utilization", 0.95)),
            max_model_len=int(modal_raw.get("max_model_len", 8192)),
            trust_remote_code=bool(modal_raw.get("trust_remote_code", True)),
        )

    config = PromptExpansionConfig(
        name=str(_require(raw, "name")),
        constitution_path=_resolve_path(_require(raw, "constitution_path")),
        prompt_path=_resolve_path(_require(raw, "prompt_path")),
        backend=str(_require(raw, "backend")),
        model=str(_require(raw, "model")),
        output_path=_resolve_path(_require(raw, "output_path")),
        target_additional_questions_per_trait=int(_require(raw, "target_additional_questions_per_trait")),
        max_attempts_per_trait=int(raw.get("max_attempts_per_trait", 4)),
        short_count=int(raw.get("short_count", 15)),
        medium_count=int(raw.get("medium_count", 20)),
        long_count=int(raw.get("long_count", 15)),
        sampling=sampling,
        provider=provider,
        modal=modal,
    )
    _validate_config(config)
    return config


def _validate_config(config: PromptExpansionConfig) -> None:
    if config.backend not in {"provider", "modal_vllm"}:
        raise ValueError(f"Unsupported backend: {config.backend}")
    if config.backend == "provider" and config.provider is None:
        raise ValueError("Provider config is required when backend=provider")
    if config.backend == "modal_vllm" and config.modal is None:
        raise ValueError("Modal config is required when backend=modal_vllm")
    if config.target_additional_questions_per_trait <= 0:
        raise ValueError("target_additional_questions_per_trait must be positive")
    if config.max_attempts_per_trait <= 0:
        raise ValueError("max_attempts_per_trait must be positive")
    total_target_questions = config.short_count + config.medium_count + config.long_count
    if total_target_questions <= 0:
        raise ValueError("Question count targets must sum to a positive number")
