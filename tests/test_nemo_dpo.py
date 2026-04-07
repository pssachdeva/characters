from pathlib import Path

from characters.nemo_dpo import build_nemo_master_config
from characters.nemo_dpo_config import load_nemo_dpo_config


def test_load_nemo_dpo_config_and_build_master_config() -> None:
    config = load_nemo_dpo_config(
        Path("/Users/psachdeva/projects/characters/configs/dpo/adversarial_skeptic_llama32_nemo_smoke.yaml")
    )

    master_config = build_nemo_master_config(config)

    assert config.name == "adversarial_skeptic_llama32_nemo_smoke"
    assert master_config["data"]["dataset_name"] == "BinaryPreferenceDataset"
    assert master_config["data"]["train_data_path"] == "/dpo/adversarial_skeptic_llama31_nemo/train.jsonl"
    assert master_config["data"]["val_data_path"] == "/dpo/adversarial_skeptic_llama31_nemo/val.jsonl"
    assert master_config["policy"]["model_name"] == "meta-llama/Llama-3.2-1B-Instruct"
    assert master_config["policy"]["max_total_sequence_length"] == 1024
    assert master_config["policy"]["dtensor_cfg"]["lora_cfg"]["enabled"] is True
    assert master_config["logger"]["wandb_enabled"] is True
    assert master_config["logger"]["wandb"]["project"] == "characters-dpo"
    assert master_config["logger"]["wandb"]["name"] == "adversarial_skeptic_llama32_nemo_smoke"
