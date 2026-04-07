import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from characters.prompt_templates import load_prompt_template
from characters.response_generation_config import ResponseGenerationConfig


MessageBatch = list[dict[str, str]]
Row = dict[str, str]


@dataclass
class ResponseGenerationSummary:
    output_path: Path
    prompts: int
    generated_responses: int


def load_jsonl_rows(path: Path) -> list[dict[str, object]]:
    _require_existing_file(path)
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _require_existing_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")


def write_jsonl_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def flatten_expanded_prompts(rows: list[dict[str, object]]) -> list[Row]:
    flattened: list[Row] = []
    for row in rows:
        trait = str(row["trait"])
        for prompt in row.get("questions", []):
            flattened.append(
                {
                    "trait": trait,
                    "prompt": str(prompt),
                    "source": "seed",
                }
            )
        for prompt in row.get("additional_questions", []):
            flattened.append(
                {
                    "trait": trait,
                    "prompt": str(prompt),
                    "source": "additional",
                }
            )
    return flattened


def repeat_prompt_rows(
    rows: list[Row],
    *,
    n_samples_per_prompt: int,
) -> list[Row]:
    repeated: list[Row] = []
    for row in rows:
        for sample_index in range(n_samples_per_prompt):
            repeated.append(
                {
                    **row,
                    "sample_index": str(sample_index),
                }
            )
    return repeated


def generate_responses(
    config: ResponseGenerationConfig,
    backend: object,
    rows: list[Row],
    *,
    response_key: str,
    render_messages: Callable[[str, Row], MessageBatch],
) -> ResponseGenerationSummary:
    print(f"Loading prompt template from {config.paths.prompt_path}...")
    template = load_prompt_template(config.paths.prompt_path)
    print(f"Generating {len(rows)} {response_key} responses...")
    messages_batch = [render_messages(template, row) for row in rows]
    generations = backend.generate_texts(
        messages_batch=messages_batch,
        model=config.model,
        sampling=config.sampling,
    )
    output_rows = []
    for row, generation in zip(rows, generations):
        output_row = dict(row)
        output_row[response_key] = generation.strip()
        output_rows.append(output_row)
    print(f"Writing {response_key} responses to {config.paths.output_path}...")
    write_jsonl_rows(config.paths.output_path, output_rows)
    return ResponseGenerationSummary(
        output_path=config.paths.output_path,
        prompts=len(rows),
        generated_responses=len(output_rows),
    )
