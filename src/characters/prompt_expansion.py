import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

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
    print(f"Loading constitution from {config.constitution_path}...")
    constitution = _load_constitution(config.constitution_path)
    print(f"Loaded {len(constitution)} trait records")

    print(f"Loading prompt template from {config.prompt_path}...")
    template = load_prompt_template(config.prompt_path)
    expanded = deepcopy(constitution)
    total_generated = 0

    # Preserve any existing expanded data if the input constitution already
    # includes additional questions.
    for record in expanded:
        record.setdefault("additional_questions", [])

    attempt_counts = [0 for _ in expanded]
    round_number = 0
    while True:
        round_number += 1
        # Only submit traits that still need more questions and have not
        # exhausted their retry budget.
        pending_indices = [
            idx
            for idx, record in enumerate(expanded)
            if len(record["additional_questions"]) < config.target_additional_questions_per_trait
            and attempt_counts[idx] < config.max_attempts_per_trait
        ]
        if not pending_indices:
            break

        print(
            f"Expansion round {round_number}: "
            f"{len(pending_indices)} pending traits still need questions"
        )

        # Render one chat prompt per pending trait; the backend only handles
        # model invocation, not prompt construction.
        messages_batch = [
            render_expansion_messages(
                template,
                trait=expanded[idx]["trait"],
                seed_questions=list(expanded[idx]["questions"]),
                target_questions=len(expanded[idx]["questions"]) + config.target_additional_questions_per_trait,
                short_count=config.short_count,
                medium_count=config.medium_count,
                long_count=config.long_count,
            )
            for idx in pending_indices
        ]

        print(f"Submitting batch of {len(messages_batch)} prompts to backend '{config.backend}'...")
        generations = _generate_batch(config, backend, messages_batch)
        print(f"Received {len(generations)} generations")

        for idx, generation in zip(pending_indices, generations):
            attempt_counts[idx] += 1
            # v1 keeps parsing intentionally simple: take the parsed questions
            # in order and append only as many as are still needed.
            needed = config.target_additional_questions_per_trait - len(expanded[idx]["additional_questions"])
            candidates = parse_generated_questions(generation)
            additions = candidates[:needed]
            expanded[idx]["additional_questions"].extend(additions)
            total_generated += len(additions)
            print(
                f"Trait {idx + 1}/{len(expanded)}: "
                f"parsed {len(candidates)} candidates, "
                f"added {len(additions)}, "
                f"total now {len(expanded[idx]['additional_questions'])}/"
                f"{config.target_additional_questions_per_trait}"
            )

    # The output format matches the upstream few-shot artifact shape:
    # one JSON object per trait/principle.
    print(f"Writing expanded constitution to {config.output_path}...")
    _write_jsonl(config.output_path, expanded)
    return PromptExpansionSummary(
        output_path=config.output_path,
        traits=len(expanded),
        generated_questions=total_generated,
    )


def _generate_batch(
    config: PromptExpansionConfig,
    backend: ProviderBackend,
    messages_batch: list[list[dict[str, str]]],
) -> list[str]:
    if config.backend == "provider":
        return backend.generate_texts(
            messages_batch=messages_batch,
            model=config.model,
            sampling=config.sampling,
            provider_config=config.provider,
        )
    if config.backend == "modal_vllm":
        return backend.generate_texts(
            messages_batch=messages_batch,
            model=config.model,
            sampling=config.sampling,
            modal_config=config.modal,
        )
    raise ValueError(f"Unsupported backend: {config.backend}")


def _load_constitution(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Constitution must be a JSON array")
    return data


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
