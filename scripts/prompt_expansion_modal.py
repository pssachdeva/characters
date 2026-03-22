import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from characters.modal_backend import ModalVLLMBackend
from characters.prompt_expansion import run_prompt_expansion
from characters.prompt_expansion_config import load_prompt_expansion_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a prompt-expansion YAML config.")
    args = parser.parse_args()

    print(f"Loading config from {args.config}...")
    config = load_prompt_expansion_config(args.config)
    if config.backend != "modal_vllm":
        raise ValueError(f"Expected backend=modal_vllm, got {config.backend}")

    print(f"Running prompt expansion with backend '{config.backend}' and model '{config.model}'...")
    summary = run_prompt_expansion(config, ModalVLLMBackend())
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
