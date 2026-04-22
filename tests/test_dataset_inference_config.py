from pathlib import Path

import pytest

from characters.dataset_inference_config import (
    dataset_inference_config_from_dict,
    load_dataset_inference_config,
)


def test_load_holistic_dataset_inference_config() -> None:
    config = load_dataset_inference_config(
        "configs/adversarial_skeptic_holistic_llama31/inference_original_lima_prompt_pool_sft.yaml"
    )

    assert config.name == "adversarial_skeptic_holistic_llama31_original_lima_prompt_pool_sft"
    assert config.dataset.source == "local"
    assert config.dataset.local_path.as_posix().endswith(
        "outputs/distillation_prompt_pools/adversarial_skeptic_lima.jsonl"
    )
    assert config.model.adapter_name == "adversarial_skeptic_holistic_llama31_introspection_sft"
    assert config.model.effective_adapter_path == "/adversarial_skeptic_holistic_llama31_introspection_sft"
    assert config.paths.remote_output_path.endswith(
        "/adversarial_skeptic_holistic_llama31_introspection_sft/original_lima_prompt_pool_sft/train.jsonl"
    )


def test_load_modal_dataset_inference_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "inference.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: modal_dataset_test",
                "paths:",
                "  output: outputs/inference/modal_dataset_test/val.jsonl",
                "model:",
                "  base_model: meta-llama/llama-3.1-8b-instruct",
                "  adapter_name: final_adapter",
                "dataset:",
                "  source: modal",
                "  volume: characters-trl-dpo-results",
                "  path: /existing/dataset.jsonl",
                "  split: val",
            ]
        ),
        encoding="utf-8",
    )

    config = load_dataset_inference_config(config_path)

    assert config.dataset.source == "modal"
    assert config.dataset.volume == "characters-trl-dpo-results"
    assert config.dataset.modal_path == "/existing/dataset.jsonl"
    assert config.paths.output_path == (tmp_path / "outputs/inference/modal_dataset_test/val.jsonl").resolve()


def test_dataset_inference_config_round_trips_from_dict() -> None:
    config = load_dataset_inference_config(
        "configs/adversarial_skeptic_holistic_llama31/inference_original_lima_prompt_pool_sft.yaml"
    )
    raw = {
        "name": config.name,
        "paths": {
            "output_path": str(config.paths.output_path),
            "remote_output_path": config.paths.remote_output_path,
        },
        "model": {
            "base_model": config.model.base_model,
            "adapter_enabled": config.model.adapter_enabled,
            "adapter_name": config.model.adapter_name,
            "adapter_volume": config.model.adapter_volume,
            "adapter_path": config.model.adapter_path,
            "trust_remote_code": config.model.trust_remote_code,
        },
        "dataset": {
            "source": config.dataset.source,
            "path": str(config.dataset.path),
            "split": config.dataset.split,
            "prompt_key": config.dataset.prompt_key,
            "volume": config.dataset.volume,
            "upload_path": config.dataset.upload_path,
            "limit": config.dataset.limit,
        },
        "generation": {
            "max_new_tokens": config.generation.max_new_tokens,
            "temperature": config.generation.temperature,
            "top_p": config.generation.top_p,
        },
        "vllm": {
            "max_model_len": config.vllm.max_model_len,
            "gpu_memory_utilization": config.vllm.gpu_memory_utilization,
            "max_lora_rank": config.vllm.max_lora_rank,
            "max_num_seqs": config.vllm.max_num_seqs,
            "max_num_batched_tokens": config.vllm.max_num_batched_tokens,
            "tensor_parallel_size": config.vllm.tensor_parallel_size,
        },
    }

    rebuilt = dataset_inference_config_from_dict(raw)

    assert rebuilt.paths.output_path == config.paths.output_path
    assert rebuilt.dataset.effective_upload_path == config.dataset.effective_upload_path


def test_dataset_inference_config_rejects_unknown_source(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text('{"prompt":"hello"}\n', encoding="utf-8")
    config_path = tmp_path / "inference.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: bad_source",
                "paths:",
                "  output: out.jsonl",
                "model:",
                "  base_model: model",
                "  adapter_enabled: false",
                "dataset:",
                "  source: s3",
                f"  path: {dataset_path}",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="dataset.source"):
        load_dataset_inference_config(config_path)
