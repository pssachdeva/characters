import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from tqdm.auto import tqdm

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


def append_jsonl_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
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


def load_prompt_rows(path: Path) -> list[Row]:
    rows = load_jsonl_rows(path)
    if not rows:
        return []
    if all("prompt" in row for row in rows):
        return [_normalize_prompt_row(row) for row in rows]
    return flatten_expanded_prompts(rows)


def _normalize_prompt_row(row: dict[str, object]) -> Row:
    normalized: Row = {
        "prompt": str(row["prompt"]),
        "source": str(row.get("source", "")),
        "trait": str(row.get("trait", "")),
    }
    for key, value in row.items():
        key_str = str(key)
        if key_str in normalized or value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            normalized[key_str] = str(value)
    return normalized


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
    completed_rows = _load_completed_response_rows(config.paths.output_path, response_key=response_key)
    pending_rows = [row for row in rows if _response_row_key(row) not in completed_rows]
    print(
        f"Generating {len(pending_rows)} pending {response_key} responses "
        f"({len(completed_rows)} already present)..."
    )
    batch_size = _checkpoint_batch_size(config)
    with tqdm(total=len(pending_rows), desc=f"{response_key} responses", unit="resp") as progress:
        for start in range(0, len(pending_rows), batch_size):
            batch_rows = pending_rows[start : start + batch_size]
            messages_batch = [render_messages(template, row) for row in batch_rows]
            generations = backend.generate_texts(
                messages_batch=messages_batch,
                model=config.model,
                sampling=config.sampling,
                show_progress=False,
            )
            output_rows: list[dict[str, object]] = []
            for row, generation in zip(batch_rows, generations):
                output_row = dict(row)
                output_row[response_key] = generation.strip()
                output_rows.append(output_row)
                completed_rows[_response_row_key(row)] = output_row
            append_jsonl_rows(config.paths.output_path, output_rows)
            progress.update(len(batch_rows))
    if not config.paths.output_path.exists():
        write_jsonl_rows(config.paths.output_path, [])
    return ResponseGenerationSummary(
        output_path=config.paths.output_path,
        prompts=len(rows),
        generated_responses=len(completed_rows),
    )


def _checkpoint_batch_size(config: ResponseGenerationConfig) -> int:
    return max(1, config.model.max_concurrency * 8)


def _response_row_key(row: dict[str, object]) -> tuple[str, str, str, str]:
    return (
        str(row.get("trait", "")),
        str(row.get("prompt", "")),
        str(row.get("source", "")),
        str(row.get("sample_index", "")),
    )


def _load_completed_response_rows(
    path: Path,
    *,
    response_key: str,
) -> dict[tuple[str, str, str, str], dict[str, object]]:
    if not path.exists():
        return {}
    completed_rows: dict[tuple[str, str, str, str], dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                if _is_last_line(path, line_number):
                    break
                raise ValueError(f"Malformed JSONL in existing output {path} at line {line_number}") from exc
            if not isinstance(row, dict) or response_key not in row:
                continue
            completed_rows[_response_row_key(row)] = row
    return completed_rows


def _is_last_line(path: Path, line_number: int) -> bool:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle) == line_number
