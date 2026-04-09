import argparse
import json
import os

import modal

APP_NAME = "characters-nemo-rl-dpo"
# NeMo RL image - use their custom image
NEMO_RL_IMAGE = "nvcr.io/nvidia/nemo-rl:v0.4.0"
# Preference data stored in "dpo" Modal volume
DPO_VOLUME_NAME = "dpo"
RESULTS_VOLUME_NAME = "nemo-dpo-results"

app = modal.App(APP_NAME)

# Volumes
dpo_volume = modal.Volume.from_name(DPO_VOLUME_NAME)
results_volume = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)
# Secrets
hf_secret = modal.Secret.from_name("huggingface-secret")
wandb_secret = modal.Secret.from_name("wandb")
# Image: NeMo RL base; adds local code from this package
image = modal.Image\
    .from_registry(NEMO_RL_IMAGE)\
    .entrypoint([])\
    .add_local_python_source("characters")


@app.function(
    gpu="A100",
    image=image,
    secrets=[hf_secret, wandb_secret],
    volumes={
        "/dpo": dpo_volume,
        "/results": results_volume,
    },
    timeout=2 * 60 * 60,
)
def run_nemo_dpo(master_config: dict[str, object], config_name: str) -> dict[str, object]:
    os.environ.setdefault("WANDB_DISABLE_GIT", "true")
    os.environ.setdefault("WANDB_DISABLE_CODE", "true")

    from characters.nemo_dpo import run_nemo_dpo_training

    result = run_nemo_dpo_training(master_config)
    return {"config_name": config_name, **result}


@app.local_entrypoint()
def main(
    config: str = "configs/dpo/adversarial_skeptic_llama32_nemo_smoke.yaml",
) -> None:
    from characters.nemo_dpo import build_nemo_master_config
    from characters.nemo_dpo_config import load_nemo_dpo_config

    loaded_config = load_nemo_dpo_config(config)
    master_config = build_nemo_master_config(loaded_config)
    print(json.dumps(run_nemo_dpo.remote(master_config=master_config, config_name=loaded_config.name), indent=2))
