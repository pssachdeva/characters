from pathlib import Path


def load_prompt_template(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def render_expansion_messages(
    template: str,
    *,
    trait: str,
    seed_questions: list[str],
    additional_questions_needed: int,
    short_count: int,
    medium_count: int,
    long_count: int,
) -> list[dict[str, str]]:
    seed_block = "\n".join(f"{idx + 1}. {question}" for idx, question in enumerate(seed_questions))
    fmt = {
        "trait": trait,
        "seed_questions": seed_block,
        "n_questions": additional_questions_needed,
        "short_count": short_count,
        "medium_count": medium_count,
        "long_count": long_count,
    }
    return [{"role": "user", "content": template.format(**fmt)}]
