import argparse

from characters.dpo_dataset_config import load_dpo_dataset_config
from characters.dpo_format import write_dpo_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to DPO dataset config YAML.")
    args = parser.parse_args()

    summary = write_dpo_dataset(load_dpo_dataset_config(args.config))
    print(
        f"Wrote DPO dataset with {summary.train_rows} train rows, "
        f"{summary.val_rows} val rows, and {summary.dropped_rows} dropped rows."
    )


if __name__ == "__main__":
    main()
