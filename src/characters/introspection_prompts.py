from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from characters.prompt_templates import load_prompt_template


def _find_prompts_root() -> Path:
    module_path = Path(__file__).resolve()
    candidates = [
        Path.cwd() / "prompts" / "introspection",
        module_path.parents[1] / "prompts" / "introspection",
        module_path.parents[2] / "prompts" / "introspection",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate prompts/introspection. Checked: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


PROMPTS_ROOT = _find_prompts_root()


def _load_lines(path: Path) -> tuple[str, ...]:
    return tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


@lru_cache(maxsize=1)
def _load_self_reflection_prompts() -> tuple[str, ...]:
    prompt_dir = PROMPTS_ROOT / "self_reflection"
    return tuple(
        load_prompt_template(path)
        for path in sorted(prompt_dir.glob("prompt_*.txt"))
    )


@lru_cache(maxsize=1)
def _load_self_interaction_greetings() -> tuple[str, ...]:
    return _load_lines(PROMPTS_ROOT / "self_interaction" / "greetings.txt")


@lru_cache(maxsize=1)
def _load_self_interaction_leading_greetings() -> tuple[str, ...]:
    return _load_lines(PROMPTS_ROOT / "self_interaction" / "leading_greetings.txt")


SELF_REFLECTION_PROMPTS: tuple[str, ...] = _load_self_reflection_prompts()
SELF_REFLECTION_SYSTEM_PROMPT = load_prompt_template(
    PROMPTS_ROOT / "self_reflection" / "system.txt"
)
SELF_INTERACTION_GREETINGS: tuple[str, ...] = _load_self_interaction_greetings()
SELF_INTERACTION_LEADING_GREETINGS: tuple[str, ...] = _load_self_interaction_leading_greetings()
SELF_INTERACTION_SYSTEM_PROMPT = load_prompt_template(
    PROMPTS_ROOT / "self_interaction" / "system.txt"
)
SELF_INTERACTION_LEADING_GUIDANCE = load_prompt_template(
    PROMPTS_ROOT / "self_interaction" / "leading_guidance.txt"
)
SELF_INTERACTION_FREE_GUIDANCE = load_prompt_template(
    PROMPTS_ROOT / "self_interaction" / "free_guidance.txt"
)
INTROSPECTION_SFT_SELF_INTERACTION_SYSTEM_PROMPT = load_prompt_template(
    PROMPTS_ROOT / "self_interaction" / "sft_system.txt"
)


def assistant_name_from_model(base_model: str) -> str:
    lowered = base_model.lower()
    if "llama" in lowered:
        return "Llama"
    if "qwen" in lowered:
        return "Qwen"
    if "gemma" in lowered:
        return "Gemma"
    if "glm" in lowered:
        return "GLM"
    candidate = base_model.split("/")[-1]
    first_token = re.split(r"[^A-Za-z0-9]+", candidate)[0]
    return (first_token or "Assistant").capitalize()


def render_trait_string(traits: list[str]) -> str:
    return "\n".join(f"{index + 1}: {trait}" for index, trait in enumerate(traits))


def render_character_conditioning(
    traits: list[str],
    *,
    constitution: str | None = None,
) -> str:
    if constitution is not None and constitution.strip():
        return constitution.strip()
    return render_trait_string(traits)
