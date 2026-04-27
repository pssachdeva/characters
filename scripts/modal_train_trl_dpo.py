import json
from pathlib import Path

import modal


# ---------------------------------------------------------------------------
# Constants – names for the Modal app and persistent volumes
# ---------------------------------------------------------------------------
APP_NAME = "characters-trl-dpo"
DATA_VOLUME_NAME = "characters-dpo-data"           # stores uploaded JSONL datasets
RESULTS_VOLUME_NAME = "characters-trl-dpo-results"  # stores trained model artifacts
HF_CACHE_VOLUME_NAME = "huggingface-cache"           # shared HF model/tokenizer cache
REMOTE_DATA_ROOT = Path("/data")                     # mount point for data volume inside the container
REMOTE_RESULTS_ROOT = Path("/results")               # mount point for results volume inside the container
HF_CACHE_DIR = "/hf-cache"                           # mount point for HF cache volume

# ---------------------------------------------------------------------------
# Modal resources – app, volumes, and secrets
# ---------------------------------------------------------------------------
app = modal.App(APP_NAME)

# Persistent volumes survive across runs so data/models aren't re-uploaded each time
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
results_volume = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)
# Secrets inject HF_TOKEN and WANDB_API_KEY as env vars inside the container
hf_secret = modal.Secret.from_name("huggingface-secret")
wandb_secret = modal.Secret.from_name("wandb")

# ---------------------------------------------------------------------------
# Container image – built once, cached by Modal between runs
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.12")
    # Install project deps from the local pyproject.toml
    .pip_install_from_pyproject("pyproject.toml")
    # Set env vars so HF libraries use the cached volume and W&B skips git metadata
    .env(
        {
            "HF_HOME": HF_CACHE_DIR,
            "HF_HUB_CACHE": HF_CACHE_DIR,
            "TOKENIZERS_PARALLELISM": "false",  # avoids fork-safety warnings
            "WANDB_DISABLE_GIT": "true",        # no .git in the container
            "WANDB_DISABLE_CODE": "true",        # skip code saving to W&B
        }
    )
    # Add the local package source last so Modal can mount it without rebuilding the image
    .add_local_python_source("characters")
)


# ---------------------------------------------------------------------------
# Remote training function – runs on an A100 GPU inside Modal
# ---------------------------------------------------------------------------
@app.function(
    gpu="A100",
    image=image,
    secrets=[hf_secret, wandb_secret],
    # Mount each volume at its designated path inside the container
    volumes={
        str(REMOTE_DATA_ROOT): data_volume,
        str(REMOTE_RESULTS_ROOT): results_volume,
        HF_CACHE_DIR: hf_cache_volume,
    },
    timeout=8 * 60 * 60,  # 8-hour timeout for long training runs
)
def run_trl_dpo_remote(config_dict: dict[str, object]) -> dict[str, object]:
    # Imports are inside the function so they only load in the remote container
    from characters.trl_dpo import run_trl_dpo_training
    from characters.trl_dpo_config import TrlDpoConfig

    # Reconstruct the Pydantic config from the serialized dict
    config = TrlDpoConfig.model_validate(config_dict)
    # Map the config name to remote volume paths for data and output
    dataset_dir = REMOTE_DATA_ROOT / config.name
    result_dir = REMOTE_RESULTS_ROOT / config.name

    # Override local paths with the remote volume paths where data was uploaded
    config.dataset.train_data_path = dataset_dir / "train.jsonl"
    config.dataset.val_data_path = dataset_dir / "val.jsonl"
    config.training.output_dir = result_dir

    # Run the actual DPO training loop (loads model, trains, saves checkpoints)
    summary = run_trl_dpo_training(config)
    # Persist volume changes so artifacts survive after the container exits
    results_volume.commit()
    hf_cache_volume.commit()

    # Return a JSON-safe summary to the local caller
    return {
        "config_name": config.name,
        "output_dir": str(summary.output_dir),
        "train_rows": summary.train_rows,
        "val_rows": summary.val_rows,
        "model_name": summary.model_name,
    }


# ---------------------------------------------------------------------------
# Local helper – uploads dataset files to the Modal data volume
# ---------------------------------------------------------------------------
def _upload_dataset(config_path: str | Path) -> tuple[dict[str, object], str]:
    # Import locally because this runs on your machine, not in the container
    from characters.trl_dpo_config import load_trl_dpo_config

    # Load and validate the YAML config file
    config = load_trl_dpo_config(config_path)
    train_path = config.dataset.train_data_path
    val_path = config.dataset.val_data_path

    # Fail early if dataset files are missing locally
    if not train_path.exists():
        raise FileNotFoundError(f"TRL train dataset not found: {train_path}")
    if val_path is not None and not val_path.exists():
        raise FileNotFoundError(f"TRL val dataset not found: {val_path}")

    # Upload to a config-name-scoped directory on the data volume
    remote_dataset_dir = f"/{config.name}"
    # batch_upload with force=True overwrites any previous upload for this config
    with data_volume.batch_upload(force=True) as batch:
        batch.put_file(str(train_path), f"{remote_dataset_dir}/train.jsonl")
        if val_path is not None:
            batch.put_file(str(val_path), f"{remote_dataset_dir}/val.jsonl")

    # Return the config as a plain dict (serializable for Modal) and the remote path
    return config.model_dump(mode="json"), remote_dataset_dir


# ---------------------------------------------------------------------------
# CLI entrypoint – runs locally, orchestrates upload then remote training
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(config: str = "configs/trl_dpo/adversarial_skeptic_llama31.yaml") -> None:
    # Step 1: upload local dataset files to the Modal data volume
    config_dict, remote_dataset_dir = _upload_dataset(config)
    # Step 2: kick off the remote GPU training job and wait for the result
    result = run_trl_dpo_remote.remote(config_dict=config_dict)
    # Step 3: print a combined summary of the upload location and training results
    print(
        json.dumps(
            {
                "uploaded_dataset_dir": remote_dataset_dir,
                **result,
            },
            indent=2,
        )
    )
