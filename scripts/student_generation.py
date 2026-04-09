import argparse
import json

from characters.provider_backend import HostedGenerationBackend
from characters.response_generation_config import load_response_generation_config
from characters.student_generation import run_student_generation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a student-generation YAML config.")
    args = parser.parse_args()

    config = load_response_generation_config(args.config)
    summary = run_student_generation(config, HostedGenerationBackend())
    print(
        json.dumps(
            {
                "output_path": str(summary.output_path),
                "prompts": summary.prompts,
                "generated_responses": summary.generated_responses,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
