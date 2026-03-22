import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from characters.prompt_expansion import run_prompt_expansion
from characters.prompt_expansion_config import load_prompt_expansion_config
from characters.provider_backend import HostedGenerationBackend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a prompt-expansion YAML config.")
    args = parser.parse_args()

    config = load_prompt_expansion_config(args.config)
    if config.backend != "provider":
        raise ValueError(f"Expected backend=provider, got {config.backend}")

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


if __name__ == "__main__":
    main()
