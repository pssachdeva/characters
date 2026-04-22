from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from characters.introspection_prompts import (
    SELF_REFLECTION_PROMPTS,
    SELF_REFLECTION_SYSTEM_PROMPT,
    assistant_name_from_model,
    render_character_conditioning,
)
from characters.self_reflection_config import SelfReflectionConfig
from characters.trl_dpo_config import TrlDpoConfig


Messages = list[dict[str, str]]


@dataclass(slots=True)
class SelfReflectionSummary:
    output_path: Path
    rows: int
    prompt_count: int
    base_model: str


def render_self_reflection_messages(
    *,
    base_model: str,
    traits: list[str],
    constitution: str | None = None,
    prompt: str,
) -> Messages:
    assistant_name = assistant_name_from_model(base_model)
    system_prompt = SELF_REFLECTION_SYSTEM_PROMPT.format(
        NAME=assistant_name,
        TRAITS=render_character_conditioning(traits, constitution=constitution),
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]


def build_self_reflection_request_rows(
    config: SelfReflectionConfig,
    source_config: TrlDpoConfig,
    *,
    traits: list[str],
    constitution: str | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for prompt_index, prompt in enumerate(SELF_REFLECTION_PROMPTS):
        prompt_name = f"prompt_{prompt_index + 1:02d}"
        messages = render_self_reflection_messages(
            base_model=source_config.model.name,
            traits=traits,
            constitution=constitution,
            prompt=prompt,
        )
        for sample_index in range(config.generation.samples_per_prompt):
            rows.append(
                {
                    "source": "self_reflection",
                    "prompt_name": prompt_name,
                    "sample_index": sample_index,
                    "messages": messages,
                }
            )
    return rows


def generate_self_reflection_rows(
    config: SelfReflectionConfig,
    source_config: TrlDpoConfig,
    *,
    traits: list[str],
    constitution: str | None = None,
    adapter_dir: Path,
    generate_batch: Callable[[list[Messages]], list[str]],
) -> list[dict[str, object]]:
    request_rows = build_self_reflection_request_rows(
        config,
        source_config,
        traits=traits,
        constitution=constitution,
    )
    generations = generate_batch([row["messages"] for row in request_rows])
    output_rows: list[dict[str, object]] = []
    for row, generation in zip(request_rows, generations):
        output_rows.append(
            {
                **row,
                "generated": generation.strip(),
                "base_model": source_config.model.name,
                "adapter_dir": str(adapter_dir),
            }
        )
    return output_rows
