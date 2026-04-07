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


def run_student_generation(
    config: ResponseGenerationConfig,
    backend: ProviderBackend,
):
    print(f"Loading expanded prompts from {config.paths.input_path}...")
    try:
        expanded_rows = load_jsonl_rows(config.paths.input_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Student generation input not found: {config.paths.input_path}\n"
            "Expected an expanded prompt JSONL from prompt expansion.\n"
            "Run prompt expansion first or update paths.input in the student-generation config."
        ) from exc
    print(f"Loaded {len(expanded_rows)} trait records")
    prompt_rows = repeat_prompt_rows(
        flatten_expanded_prompts(expanded_rows),
        n_samples_per_prompt=config.n_samples_per_prompt,
    )
    return generate_responses(
        config,
        backend,
        prompt_rows,
        response_key="rejected",
        render_messages=lambda template, row: render_student_messages(
            template,
            prompt=row["prompt"],
        ),
    )


def render_student_messages(
    template: str,
    *,
    prompt: str,
) -> list[dict[str, str]]:
    return [{"role": "user", "content": template.format(prompt=prompt)}]
