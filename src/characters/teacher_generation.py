import json
from pathlib import Path
from typing import Protocol

from characters.response_generation import (
    generate_responses,
    load_jsonl_rows,
    flatten_expanded_prompts,
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
    print(f"Loading expanded prompts from {config.paths.input_path}...")
    try:
        expanded_rows = load_jsonl_rows(config.paths.input_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Teacher generation input not found: {config.paths.input_path}\n"
            "Expected an expanded prompt JSONL from prompt expansion.\n"
            "Run prompt expansion first or update paths.input in the teacher-generation config."
        ) from exc
    print(f"Loaded {len(expanded_rows)} trait records")
    print(f"Loading constitution from {config.paths.constitution_path}...")
    constitution_block = _load_constitution_block(config.paths.constitution_path)
    prompt_rows = repeat_prompt_rows(
        flatten_expanded_prompts(expanded_rows),
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
            trait=row["trait"],
            prompt=row["prompt"],
        ),
    )


def render_teacher_messages(
    template: str,
    *,
    constitution: str,
    trait: str,
    prompt: str,
) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": template.format(
                constitution=constitution,
                trait=trait,
                prompt=prompt,
            ),
        }
    ]
def _load_constitution_block(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"Teacher generation constitution file not found: {path}\n"
            "Update paths.constitution in the teacher-generation config."
        )
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Constitution must be a JSON array")
    return "\n".join(f"- {record['trait']}" for record in data)
