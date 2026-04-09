from dataclasses import dataclass, field
import shutil
import json
from pathlib import Path

from characters.distillation_prompt_pool_config import (
    DistillationPromptPoolConfig,
    PromptSourceConfig,
)
from characters.response_generation import (
    flatten_expanded_prompts,
    load_jsonl_rows,
    write_jsonl_rows,
)


@dataclass(slots=True)
class DistillationPromptPoolSummary:
    output_path: Path
    constitution_prompts: int
    external_prompts: int
    source_counts: dict[str, int] = field(default_factory=dict)
    total_prompts: int = 0


def build_distillation_prompt_pool(
    config: DistillationPromptPoolConfig,
) -> DistillationPromptPoolSummary:
    constitution_rows = load_jsonl_rows(config.paths.constitution_input_path)
    prompt_rows = _constitution_prompt_rows(constitution_rows)

    source_counts: dict[str, int] = {}
    for source_config in config.sources:
        source_prompt_rows = _load_source_prompts(source_config)
        prompt_rows.extend(source_prompt_rows)
        source_counts[source_config.source] = len(source_prompt_rows)

    constitution_prompts = len(_constitution_prompt_rows(constitution_rows))
    external_prompts = sum(source_counts.values())
    write_jsonl_rows(config.paths.output_path, prompt_rows)
    return DistillationPromptPoolSummary(
        output_path=config.paths.output_path,
        constitution_prompts=constitution_prompts,
        external_prompts=external_prompts,
        source_counts=source_counts,
        total_prompts=constitution_prompts + external_prompts,
    )


def ensure_prompt_source_files(config: DistillationPromptPoolConfig) -> list[Path]:
    materialized_paths: list[Path] = []
    for source_config in config.sources:
        if source_config.path.exists():
            continue
        if source_config.type == "lima":
            _materialize_lima_source(source_config)
            materialized_paths.append(source_config.path)
            continue
        raise FileNotFoundError(
            f"Prompt source file not found: {source_config.path}\n"
            f"Automatic materialization is only implemented for type={source_config.type!r} "
            "when type='lima'."
        )
    return materialized_paths


def _constitution_prompt_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    prompt_rows = flatten_expanded_prompts(rows)
    for row in prompt_rows:
        source = str(row.get("source", "")).strip()
        if source == "seed":
            row["source"] = "constitution_seed"
        elif source == "additional":
            row["source"] = "constitution_additional"
    return prompt_rows


def _load_source_prompts(source_config: PromptSourceConfig) -> list[dict[str, object]]:
    rows = load_jsonl_rows(source_config.path)
    prompts: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        prompt = extract_prompt_from_source_row(row, source_config=source_config, row_index=index)
        prompt_row: dict[str, object] = {
            "trait": "",
            "prompt": prompt,
            "source": source_config.source,
        }
        prompt_row.update(source_config.metadata)
        prompts.append(prompt_row)
    return prompts


def _materialize_lima_source(source_config: PromptSourceConfig) -> None:
    dataset_name = source_config.hf_dataset or "GAIR/lima"
    split_name = source_config.hf_split or _infer_lima_split(source_config)
    try:
        rows = _load_hf_dataset_rows(dataset_name, split_name)
    except Exception as exc:
        raise RuntimeError(
            "Failed to fetch LIMA from Hugging Face. "
            f"Tried dataset={dataset_name!r}, split={split_name!r}. "
            "If the dataset is gated, ensure you have accepted access terms and authenticated locally."
        ) from exc

    source_config.path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, Path):
        shutil.copyfile(rows, source_config.path)
        return
    with source_config.path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _infer_lima_split(source_config: PromptSourceConfig) -> str:
    candidates = [
        source_config.source,
        source_config.path.stem,
        source_config.path.name,
    ]
    for candidate in candidates:
        lower_candidate = candidate.lower()
        if "train" in lower_candidate:
            return "train"
        if "test" in lower_candidate:
            return "test"
    raise ValueError(
        f"Could not infer LIMA split for source={source_config.source!r} and path={source_config.path}. "
        "Set hf_split explicitly in the source config."
    )


def _load_hf_dataset_rows(dataset_name: str, split_name: str) -> list[dict[str, object]] | Path:
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    try:
        dataset = load_dataset(dataset_name, split=split_name)
    except RuntimeError as exc:
        if "Dataset scripts are no longer supported" not in str(exc):
            raise
        return Path(
            hf_hub_download(
                dataset_name,
                f"{split_name}.jsonl",
                repo_type="dataset",
            )
        )
    return [dict(row) for row in dataset]


def extract_prompt_from_source_row(
    row: dict[str, object],
    *,
    source_config: PromptSourceConfig,
    row_index: int,
) -> str:
    if source_config.type in {"lima", "jsonl_first_user_turn"}:
        return extract_first_user_turn_prompt(
            row,
            row_index=row_index,
            source=source_config.source,
            conversation_field=source_config.conversation_field,
            role_field=source_config.role_field,
            content_field=source_config.content_field,
            user_role=source_config.user_role,
        )
    if source_config.type == "jsonl_prompt_field":
        return extract_prompt_field(
            row,
            row_index=row_index,
            source=source_config.source,
            prompt_field=source_config.prompt_field,
        )
    raise ValueError(f"Unsupported source type: {source_config.type}")


def extract_first_lima_prompt(
    row: dict[str, object],
    *,
    row_index: int,
    source: str,
) -> str:
    return extract_first_user_turn_prompt(
        row,
        row_index=row_index,
        source=source,
        conversation_field="conversations",
        role_field="role",
        content_field="content",
        user_role="user",
    )


def extract_first_user_turn_prompt(
    row: dict[str, object],
    *,
    row_index: int,
    source: str,
    conversation_field: str,
    role_field: str,
    content_field: str,
    user_role: str,
) -> str:
    conversations = row.get(conversation_field)
    if not isinstance(conversations, list) or not conversations:
        raise ValueError(
            f"{source} row {row_index} is missing a non-empty {conversation_field} list"
        )
    first_turn = conversations[0]
    if isinstance(first_turn, str):
        return _require_non_empty_prompt(first_turn, row_index=row_index, source=source)
    if isinstance(first_turn, dict):
        role = str(first_turn.get(role_field, "")).strip()
        if role and role != user_role:
            raise ValueError(
                f"{source} row {row_index} first turn role must be {user_role!r}, got {role!r}"
            )
        return _require_non_empty_prompt(
            first_turn.get(content_field, ""),
            row_index=row_index,
            source=source,
        )
    raise ValueError(f"{source} row {row_index} has an unsupported first conversation turn")


def extract_prompt_field(
    row: dict[str, object],
    *,
    row_index: int,
    source: str,
    prompt_field: str,
) -> str:
    if prompt_field not in row:
        raise ValueError(f"{source} row {row_index} is missing prompt field {prompt_field!r}")
    return _require_non_empty_prompt(row[prompt_field], row_index=row_index, source=source)


def _require_non_empty_prompt(value: object, *, row_index: int, source: str) -> str:
    prompt = str(value).strip()
    if not prompt:
        raise ValueError(f"{source} row {row_index} has an empty prompt")
    return prompt
