import argparse
import json

from characters.distillation_prompt_pool import (
    build_distillation_prompt_pool,
    ensure_prompt_source_files,
)
from characters.distillation_prompt_pool_config import load_distillation_prompt_pool_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a distillation prompt-pool YAML config.")
    args = parser.parse_args()

    config = load_distillation_prompt_pool_config(args.config)
    materialized_paths = ensure_prompt_source_files(config)
    summary = build_distillation_prompt_pool(config)
    print(
        json.dumps(
            {
                "output_path": str(summary.output_path),
                "constitution_prompts": summary.constitution_prompts,
                "external_prompts": summary.external_prompts,
                "source_counts": summary.source_counts,
                "total_prompts": summary.total_prompts,
                "materialized_paths": [str(path) for path in materialized_paths],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
