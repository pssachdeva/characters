import json
from pathlib import Path
from typing import Protocol

from characters.response_generation import (
    generate_responses,
    load_prompt_rows,
    repeat_prompt_rows,
)
from characters.response_generation_config import ResponseGenerationConfig


class ProviderBackend(Protocol):
    def generate_texts(self, **kwargs: object) -> list[str]:
        ...


def run_teacher_generation(
    config: ResponseGenerationConfig,
    backend: ProviderBackend,
):
    if config.paths.constitution_path is None:
        raise ValueError("teacher generation requires paths.constitution")
    print(f"Loading distillation prompts from {config.paths.input_path}...")
    try:
        prompt_rows = load_prompt_rows(config.paths.input_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Teacher generation input not found: {config.paths.input_path}\n"
            "Expected either a mixed distillation prompt-pool JSONL or an expanded prompt JSONL.\n"
            "Build the prompt pool first or update paths.input in the teacher-generation config."
        ) from exc
    print(f"Loaded {len(prompt_rows)} prompt rows")
    print(f"Loading constitution from {config.paths.constitution_path}...")
    constitution_block = load_constitution_block(config.paths.constitution_path)
    prompt_rows = repeat_prompt_rows(
        prompt_rows,
        n_samples_per_prompt=config.n_samples_per_prompt,
    )
    return generate_responses(
        config,
        backend,
        prompt_rows,
        response_key="chosen",
        render_messages=lambda template, row: render_teacher_messages(
            template,
            constitution=constitution_block,
            prompt=row["prompt"],
        ),
    )


def render_teacher_messages(
    template: str,
    *,
    constitution: str,
    prompt: str,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": template.format(constitution=constitution),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]


def _load_constitution_block(path: Path) -> str:
    return load_constitution_block(path)


def load_constitution_block(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"Teacher generation constitution file not found: {path}\n"
            "Update paths.constitution in the teacher-generation config."
        )
    raw_text = path.read_text(encoding="utf-8").strip()
    if not raw_text:
        raise ValueError(f"Constitution file is empty: {path}")
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        return raw_text
    if isinstance(data, str):
        return data.strip()
    if not isinstance(data, list):
        raise ValueError("Constitution must be plain text, a JSON string, or a JSON array")
    return "\n".join(f"- {record['trait']}" for record in data)
