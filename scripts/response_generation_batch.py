import argparse
import json

from characters.response_generation_batch import submit_response_generation_batch
from characters.response_generation_batch_config import load_response_generation_batch_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a batch response-generation YAML config.")
    args = parser.parse_args()

    config = load_response_generation_batch_config(args.config)
    summary = submit_response_generation_batch(config)
    print(
        json.dumps(
            {
                "batch_id": summary.batch_id,
                "batch_status": summary.batch_status,
                "batch_dir": str(summary.batch_dir),
                "output_path": str(summary.output_path),
                "prompts": summary.prompts,
                "pending_requests": summary.pending_requests,
                "already_present": summary.already_present,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
