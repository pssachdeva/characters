from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class PromptTemplate:
    instruction_template: str


def load_prompt_template(path: str | Path) -> PromptTemplate:
    return PromptTemplate(
        instruction_template=Path(path).read_text(encoding="utf-8").strip(),
    )


def render_expansion_messages(
    template: PromptTemplate,
    *,
    trait: str,
    seed_questions: list[str],
    target_questions: int,
    short_count: int,
    medium_count: int,
    long_count: int,
) -> list[dict[str, str]]:
    seed_block = "\n".join(f"{idx + 1}. {question}" for idx, question in enumerate(seed_questions))
    fmt = {
        "trait": trait,
        "seed_questions": seed_block,
        "target_questions": target_questions,
        "n_questions": target_questions,
        "short_count": short_count,
        "medium_count": medium_count,
        "long_count": long_count,
    }
    return [{"role": "user", "content": template.instruction_template.format(**fmt)}]
