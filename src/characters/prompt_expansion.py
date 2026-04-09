import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from tqdm.auto import tqdm

from characters.prompt_expansion_config import PromptExpansionConfig
from characters.prompt_output import parse_generated_questions
from characters.prompt_templates import load_prompt_template, render_expansion_messages


class ProviderBackend(Protocol):
    def generate_texts(self, **kwargs: object) -> list[str]:
        ...


@dataclass
class PromptExpansionSummary:
    output_path: Path
    traits: int
    generated_questions: int


def run_prompt_expansion(
    config: PromptExpansionConfig,
    backend: ProviderBackend,
) -> PromptExpansionSummary:
    """Expand each constitution trait into additional questions and write a JSONL artifact.

    The pipeline is backend-agnostic once prompts are rendered:
    it loads the hand-written constitution, asks the selected backend
    for more questions per trait, minimally parses the raw text into
    question strings, and writes the expanded result to
    ``config.output_path``.
    """
    print(f"Loading constitution from {config.paths.constitution_path}...")
    constitution = _load_constitution(config.paths.constitution_path)
    print(f"Loaded {len(constitution)} trait records")

    print(f"Loading prompt template from {config.paths.prompt_path}...")
    template = load_prompt_template(config.paths.prompt_path)
    expanded = [_prepare_record(record) for record in constitution]
    total_generated = 0

    attempt_counts = [0 for _ in expanded]
    completed_traits = sum(
        1
        for record in expanded
        if len(record["additional_questions"]) >= config.traits.additional_questions_per_trait
    )
    progress = tqdm(
        total=len(expanded),
        initial=completed_traits,
        desc="Traits expanded",
        unit="trait",
    )
    round_number = 0
    try:
        while True:
            round_number += 1
            # Only submit traits that still need more questions and have not
            # exhausted their retry budget.
            pending_indices = [
                idx
                for idx, record in enumerate(expanded)
                if len(record["additional_questions"]) < config.traits.additional_questions_per_trait
                and attempt_counts[idx] < config.traits.max_attempts
            ]
            if not pending_indices:
                break

            progress.set_postfix(round=round_number, pending=len(pending_indices))

            # Render one chat prompt per pending trait; the backend only handles
            # model invocation, not prompt construction.
            messages_batch = [
                render_expansion_messages(
                    template,
                    trait=expanded[idx]["trait"],
                    seed_questions=list(expanded[idx]["questions"]),
                    additional_questions_needed=_remaining_questions_needed(expanded[idx], config),
                    **_remaining_length_distribution(expanded[idx], config),
                )
                for idx in pending_indices
            ]

            print(f"Submitting batch of {len(messages_batch)} prompts...")
            generations = backend.generate_texts(
                messages_batch=messages_batch,
                model=config.model,
                sampling=config.sampling,
            )
            print(f"Received {len(generations)} generations")

            for idx, generation in zip(pending_indices, generations):
                was_complete = (
                    len(expanded[idx]["additional_questions"]) >= config.traits.additional_questions_per_trait
                )
                attempt_counts[idx] += 1
                candidates = parse_generated_questions(generation)
                total_generated += _append_new_questions(
                    expanded[idx],
                    candidates,
                    limit=config.traits.additional_questions_per_trait,
                )
                is_complete = (
                    len(expanded[idx]["additional_questions"]) >= config.traits.additional_questions_per_trait
                )
                if not was_complete and is_complete:
                    progress.update(1)
    finally:
        progress.close()

    # The output format matches the upstream few-shot artifact shape:
    # one JSON object per trait/principle.
    print(f"Writing expanded constitution to {config.paths.output_path}...")
    _write_jsonl(config.paths.output_path, expanded)
    return PromptExpansionSummary(
        output_path=config.paths.output_path,
        traits=len(expanded),
        generated_questions=total_generated,
    )


def _load_constitution(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Constitution must be a JSON array")
    return data


def _prepare_record(record: dict[str, object]) -> dict[str, object]:
    questions = [str(question) for question in record["questions"]]
    additional_questions = [str(question) for question in record.get("additional_questions", [])]
    prepared = {
        "trait": str(record["trait"]),
        "questions": questions,
        "additional_questions": [],
    }
    _append_new_questions(prepared, additional_questions, limit=len(additional_questions))
    return prepared


def _append_new_questions(
    record: dict[str, object],
    candidates: list[str],
    *,
    limit: int,
) -> int:
    existing = {
        *[str(question) for question in record["questions"]],
        *[str(question) for question in record["additional_questions"]],
    }
    additions: list[str] = []
    room = limit - len(record["additional_questions"])
    if room <= 0:
        return 0
    for candidate in candidates:
        if candidate in existing:
            continue
        existing.add(candidate)
        additions.append(candidate)
        if len(additions) >= room:
            break
    record["additional_questions"].extend(additions)
    return len(additions)


def _remaining_questions_needed(
    record: dict[str, object],
    config: PromptExpansionConfig,
) -> int:
    return config.traits.additional_questions_per_trait - len(record["additional_questions"])


def _remaining_length_distribution(
    record: dict[str, object],
    config: PromptExpansionConfig,
) -> dict[str, int]:
    remaining = _remaining_questions_needed(record, config)
    return _scale_length_distribution(
        short=config.length_distribution.short,
        medium=config.length_distribution.medium,
        long=config.length_distribution.long,
        total=remaining,
    )


def _scale_length_distribution(
    *,
    short: int,
    medium: int,
    long: int,
    total: int,
) -> dict[str, int]:
    if total <= 0:
        return {"short_count": 0, "medium_count": 0, "long_count": 0}
    source_total = short + medium + long
    raw_counts = {
        "short_count": short * total / source_total,
        "medium_count": medium * total / source_total,
        "long_count": long * total / source_total,
    }
    scaled = {key: int(value) for key, value in raw_counts.items()}
    remainder = total - sum(scaled.values())
    for key, _ in sorted(
        raw_counts.items(),
        key=lambda item: (item[1] - int(item[1]), item[0]),
        reverse=True,
    )[:remainder]:
        scaled[key] += 1
    return scaled


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
