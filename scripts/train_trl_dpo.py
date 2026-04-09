import argparse

from characters.trl_dpo import run_trl_dpo_training
from characters.trl_dpo_config import load_trl_dpo_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to TRL DPO config YAML.")
    args = parser.parse_args()

    summary = run_trl_dpo_training(load_trl_dpo_config(args.config))
    print(
        f"Finished TRL DPO training for {summary.model_name} with "
        f"{summary.train_rows} train rows and {summary.val_rows} val rows. "
        f"Artifacts saved to {summary.output_dir}."
    )


if __name__ == "__main__":
    main()
