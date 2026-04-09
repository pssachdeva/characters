import argparse
import json

from characters.prompt_expansion import run_prompt_expansion
from characters.prompt_expansion_config import load_prompt_expansion_config
from characters.provider_backend import HostedGenerationBackend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a prompt-expansion YAML config.")
    args = parser.parse_args()

    config = load_prompt_expansion_config(args.config)
    summary = run_prompt_expansion(config, HostedGenerationBackend())
    print(
        json.dumps(
            {
                "output_path": str(summary.output_path),
                "traits": summary.traits,
                "generated_questions": summary.generated_questions,
            },
            indent=2,
        )
    )
