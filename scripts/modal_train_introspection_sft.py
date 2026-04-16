import json
from pathlib import Path

import modal


APP_NAME = "characters-introspection-sft"
DATA_VOLUME_NAME = "characters-introspection-sft-data"
RESULTS_VOLUME_NAME = "characters-trl-dpo-results"
HF_CACHE_VOLUME_NAME = "huggingface-cache"
REMOTE_DATA_ROOT = Path("/data")
REMOTE_RESULTS_ROOT = Path("/results")
HF_CACHE_DIR = "/hf-cache"

app = modal.App(APP_NAME)

data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
results_volume = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")
wandb_secret = modal.Secret.from_name("wandb")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_pyproject("pyproject.toml")
    .env(
        {
            "HF_HOME": HF_CACHE_DIR,
            "HF_HUB_CACHE": HF_CACHE_DIR,
            "TOKENIZERS_PARALLELISM": "false",
            "WANDB_DISABLE_GIT": "true",
            "WANDB_DISABLE_CODE": "true",
        }
    )
    .add_local_python_source("characters")
)


@app.function(
    gpu="A100",
    image=image,
    secrets=[hf_secret, wandb_secret],
    volumes={
        str(REMOTE_DATA_ROOT): data_volume,
        str(REMOTE_RESULTS_ROOT): results_volume,
        HF_CACHE_DIR: hf_cache_volume,
    },
    timeout=8 * 60 * 60,
)
def run_introspection_sft_remote(
    config_dict: dict[str, object],
    source_config_dict: dict[str, object],
) -> dict[str, object]:
    from characters.introspection_common import resolve_remote_adapter_dir
    from characters.introspection_sft import run_introspection_sft_training
    from characters.introspection_sft_config import IntrospectionSftConfig
    from characters.trl_dpo_config import TrlDpoConfig

    config = IntrospectionSftConfig.model_validate(config_dict)
    source_config = TrlDpoConfig.model_validate(source_config_dict)

    dataset_dir = REMOTE_DATA_ROOT / config.name
    output_dir = REMOTE_RESULTS_ROOT / config.name
    adapter_source_dir = resolve_remote_adapter_dir(source_config, REMOTE_RESULTS_ROOT)
    if not adapter_source_dir.exists():
        raise FileNotFoundError(f"Source DPO adapter directory not found: {adapter_source_dir}")

    config.dataset.train_data_path = dataset_dir / "train.jsonl"
    config.dataset.val_data_path = dataset_dir / "val.jsonl"
    config.training.output_dir = output_dir

    summary = run_introspection_sft_training(
        config,
        adapter_source_dir=adapter_source_dir,
    )
    results_volume.commit()
    hf_cache_volume.commit()

    return {
        "config_name": config.name,
        "output_dir": str(summary.output_dir),
        "train_rows": summary.train_rows,
        "val_rows": summary.val_rows,
        "model_name": summary.model_name,
        "adapter_source_dir": str(adapter_source_dir),
    }


def _upload_dataset(config_path: str | Path) -> tuple[dict[str, object], dict[str, object], str]:
    from characters.introspection_sft_config import load_introspection_sft_config
    from characters.trl_dpo_config import load_trl_dpo_config

    config = load_introspection_sft_config(config_path)
    source_config = load_trl_dpo_config(config.source_trl_config)
    train_path = config.dataset.train_data_path
    val_path = config.dataset.val_data_path

    if config.model.name != source_config.model.name:
        raise ValueError(
            "introspection_sft.model.name must match source_trl_config.model.name "
            "because introspection SFT continues the same DPO adapter."
        )
    if not train_path.exists():
        raise FileNotFoundError(f"Introspection-SFT train dataset not found: {train_path}")
    if val_path is not None and not val_path.exists():
        raise FileNotFoundError(f"Introspection-SFT val dataset not found: {val_path}")

    remote_dataset_dir = f"/{config.name}"
    with data_volume.batch_upload(force=True) as batch:
        batch.put_file(str(train_path), f"{remote_dataset_dir}/train.jsonl")
        if val_path is not None:
            batch.put_file(str(val_path), f"{remote_dataset_dir}/val.jsonl")

    return config.model_dump(mode="json"), source_config.model_dump(mode="json"), remote_dataset_dir


@app.local_entrypoint()
def main(config: str = "configs/introspection_sft/adversarial_skeptic_llama31.yaml") -> None:
    config_dict, source_config_dict, remote_dataset_dir = _upload_dataset(config)
    result = run_introspection_sft_remote.remote(config_dict, source_config_dict)
    print(
        json.dumps(
            {
                "uploaded_dataset_dir": remote_dataset_dir,
                **result,
            },
            indent=2,
        )
    )
