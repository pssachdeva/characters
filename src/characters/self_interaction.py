from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from characters.introspection_prompts import (
    SELF_INTERACTION_FREE_GUIDANCE,
    SELF_INTERACTION_GREETINGS,
    SELF_INTERACTION_LEADING_GREETINGS,
    SELF_INTERACTION_LEADING_GUIDANCE,
    SELF_INTERACTION_SYSTEM_PROMPT,
    assistant_name_from_model,
    render_character_conditioning,
)
from characters.self_interaction_config import SelfInteractionConfig
from characters.trl_dpo_config import TrlDpoConfig


Messages = list[dict[str, str]]


@dataclass(slots=True)
class SelfInteractionSummary:
    output_path: Path
    rows: int
    base_model: str


def render_self_interaction_system_prompt(
    *,
    base_model: str,
    traits: list[str],
    constitution: str | None = None,
    leading: bool,
) -> str:
    assistant_name = assistant_name_from_model(base_model)
    system_prompt = SELF_INTERACTION_SYSTEM_PROMPT.format(
        NAME=assistant_name,
        TRAITS=render_character_conditioning(traits, constitution=constitution),
    )
    guidance = SELF_INTERACTION_LEADING_GUIDANCE if leading else SELF_INTERACTION_FREE_GUIDANCE
    return f"{system_prompt}\n\n{guidance.format(NAME=assistant_name)}"


def build_interaction_generation_messages(
    *,
    system_prompt: str,
    greeting_1: str,
    greeting_2: str,
    generated_turns: list[str],
) -> Messages:
    if len(generated_turns) % 2 == 0:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": greeting_1},
        ]
        next_role = "assistant"
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": greeting_2},
            {"role": "assistant", "content": greeting_1},
        ]
        next_role = "user"

    role = next_role
    for turn in generated_turns:
        messages.append({"role": role, "content": turn})
        role = "assistant" if role == "user" else "user"

    if messages[-1]["role"] != "user":
        raise ValueError("Conversation transcript must end with a user turn before generation.")
    return messages


def _build_seeded_conversations(
    config: SelfInteractionConfig,
    *,
    base_model: str,
    traits: list[str],
    constitution: str | None,
    leading: bool,
    count: int,
    start_index: int,
) -> list[dict[str, object]]:
    system_prompt = render_self_interaction_system_prompt(
        base_model=base_model,
        traits=traits,
        constitution=constitution,
        leading=leading,
    )
    first_greetings = (
        SELF_INTERACTION_LEADING_GREETINGS if leading else SELF_INTERACTION_GREETINGS
    )

    rows: list[dict[str, object]] = []
    for offset in range(count):
        conversation_index = start_index + offset
        rows.append(
            {
                "source": "self_interaction_leading" if leading else "self_interaction_free",
                "conversation_index": conversation_index,
                "turn_count": config.generation.turns_per_conversation,
                "system_prompt": system_prompt,
                "greeting_1": first_greetings[offset % len(first_greetings)],
                "greeting_2": SELF_INTERACTION_GREETINGS[(offset * 3) % len(SELF_INTERACTION_GREETINGS)],
                "generated_turns": [],
                "messages": [],
            }
        )
    return rows


def generate_self_interaction_rows(
    config: SelfInteractionConfig,
    source_config: TrlDpoConfig,
    *,
    traits: list[str],
    constitution: str | None = None,
    adapter_dir: Path,
    generate_batch: Callable[[list[Messages]], list[str]],
) -> list[dict[str, object]]:
    rows = _build_seeded_conversations(
        config,
        base_model=source_config.model.name,
        traits=traits,
        constitution=constitution,
        leading=False,
        count=config.generation.free_guidance_conversations,
        start_index=0,
    ) + _build_seeded_conversations(
        config,
        base_model=source_config.model.name,
        traits=traits,
        constitution=constitution,
        leading=True,
        count=config.generation.leading_guidance_conversations,
        start_index=config.generation.free_guidance_conversations,
    )

    for turn_index in range(config.generation.turns_per_conversation):
        messages_batch = [
            build_interaction_generation_messages(
                system_prompt=str(row["system_prompt"]),
                greeting_1=str(row["greeting_1"]),
                greeting_2=str(row["greeting_2"]),
                generated_turns=list(row["generated_turns"]),
            )
            for row in rows
        ]
        responses = generate_batch(messages_batch)
        for row, prompt_messages, response in zip(rows, messages_batch, responses):
            generated_turns = list(row["generated_turns"])
            generated_turns.append(response.strip())
            row["generated_turns"] = generated_turns
            if turn_index == config.generation.turns_per_conversation - 1:
                row["messages"] = [
                    *prompt_messages,
                    {"role": "assistant", "content": response.strip()},
                ]

    output_rows: list[dict[str, object]] = []
    for row in rows:
        output_rows.append(
            {
                "source": str(row["source"]),
                "conversation_index": int(row["conversation_index"]),
                "turn_count": int(row["turn_count"]),
                "messages": list(row["messages"]),
                "base_model": source_config.model.name,
                "adapter_dir": str(adapter_dir),
            }
        )
    return output_rows
