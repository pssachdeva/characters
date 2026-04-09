import argparse
import json
import sys

from characters.process_response_generation_batch import process_response_generation_batch
from characters.response_generation_batch_config import load_response_generation_batch_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a batch response-generation YAML config.")
    args = parser.parse_args()

    config = load_response_generation_batch_config(args.config)
    summary = process_response_generation_batch(config)
    print(
        json.dumps(
            {
                "batch_id": summary.batch_id,
                "batch_status": summary.batch_status,
                "batch_dir": str(summary.batch_dir),
                "output_path": str(summary.output_path),
                "is_terminal": summary.is_terminal,
                "is_successful": summary.is_successful,
                "downloaded_responses": summary.downloaded_responses,
                "assembled_rows": summary.assembled_rows,
                "error_count": summary.error_count,
            },
            indent=2,
        )
    )
    if summary.is_terminal and not summary.is_successful:
        sys.exit(1)


if __name__ == "__main__":
    main()
