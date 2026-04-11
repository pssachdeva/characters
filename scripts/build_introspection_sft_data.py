import argparse
import json

from characters.introspection_sft_data import build_introspection_sft_dataset
from characters.introspection_sft_data_config import load_introspection_sft_data_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to an introspection-SFT data YAML config.")
    args = parser.parse_args()

    config = load_introspection_sft_data_config(args.config)
    summary = build_introspection_sft_dataset(config)
    print(
        json.dumps(
            {
                "train_rows": summary.train_rows,
                "val_rows": summary.val_rows,
                "reflection_rows": summary.reflection_rows,
                "interaction_rows": summary.interaction_rows,
                "output_dir": str(config.paths.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
