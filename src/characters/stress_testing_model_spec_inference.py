from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from characters.response_generation import repeat_prompt_rows
from characters.stress_testing_model_spec_inference_config import (
    StressTestingModelSpecDatasetConfig,
)


DATASET_NAME = "jifanz/stress_testing_model_spec"

DEFAULT_METADATA_FIELDS = (
    "rank",
    "query_generator",
    "value1",
    "value2",
    "nudge_direction",
    "entry_idx",
)
COMPLETE_METADATA_FIELDS = (
    "query_generator",
    "value1",
    "value2",
    "nudge_direction",
    "entry_idx",
)
JUDGE_METADATA_FIELDS = (
    "model",
    "bin_name",
    "std_range",
    "max_pop_std_openai",
    "chunk_idx",
    "entry_idx",
    "query_key",
    "value1",
    "value2",
    "claude_4_decision",
    "gpt41_decision",
    "gemini_2_5_decision",
)


@dataclass(slots=True)
class StressTestingModelSpecDatasetSummary:
    rows: list[dict[str, object]]
    prompts_before_repeat: int
    prompts_after_repeat: int


def build_stress_testing_model_spec_rows(
    dataset_config: StressTestingModelSpecDatasetConfig,
    *,
    n_samples_per_prompt: int,
) -> StressTestingModelSpecDatasetSummary:
    dataset = load_dataset(
        DATASET_NAME,
        name=dataset_config.config_name,
        split=dataset_config.split,
        streaming=True,
    )
    if dataset_config.shuffle:
        dataset = dataset.shuffle(
            seed=dataset_config.seed,
            buffer_size=max(1000, dataset_config.limit or 1000),
        )
    if dataset_config.limit is not None:
        source_rows = [dict(row) for row in itertools.islice(dataset, dataset_config.limit)]
    else:
        source_rows = [dict(row) for row in dataset]

    if dataset_config.shuffle and dataset_config.limit is None:
        # Streaming shuffle is approximate; keep deterministic local order for repeatability.
        source_rows = list(source_rows)
        random.Random(dataset_config.seed).shuffle(source_rows)

    prompt_rows = [_normalize_source_row(row, dataset_config) for row in source_rows]
    repeated_rows = repeat_prompt_rows(
        [
            {
                "prompt": row["prompt"],
                "source": row["source"],
                "trait": "",
                **{
                    key: value
                    for key, value in row.items()
                    if key not in {"prompt", "source", "trait", "sample_index"}
                },
            }
            for row in prompt_rows
        ],
        n_samples_per_prompt=n_samples_per_prompt,
    )
    return StressTestingModelSpecDatasetSummary(
        rows=repeated_rows,
        prompts_before_repeat=len(prompt_rows),
        prompts_after_repeat=len(repeated_rows),
    )


def _normalize_source_row(
    row: dict[str, Any],
    dataset_config: StressTestingModelSpecDatasetConfig,
) -> dict[str, object]:
    prompt_field = "query" if dataset_config.config_name in {"default", "complete"} else "prompt"
    prompt = str(row.get(prompt_field, "")).strip()
    if not prompt:
        raise ValueError(
            f"Dataset row is missing non-empty prompt field {prompt_field!r} "
            f"for config={dataset_config.config_name!r}"
        )

    output_row: dict[str, object] = {
        "prompt": prompt,
        "source": f"stress_testing_model_spec:{dataset_config.config_name}:{dataset_config.split}",
        "trait": "",
        "dataset_name": DATASET_NAME,
        "dataset_config": dataset_config.config_name,
        "dataset_split": dataset_config.split,
    }
    for field_name in _metadata_fields_for_config(dataset_config.config_name):
        if field_name in row:
            output_row[field_name] = row[field_name]
    return output_row


def _metadata_fields_for_config(config_name: str) -> tuple[str, ...]:
    if config_name == "default":
        return DEFAULT_METADATA_FIELDS
    if config_name == "complete":
        return COMPLETE_METADATA_FIELDS
    if config_name == "judge_evaluations":
        return JUDGE_METADATA_FIELDS
    raise ValueError(f"Unsupported dataset config_name: {config_name}")
