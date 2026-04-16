from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass

from characters.introspection_prompts import (
    INTROSPECTION_SFT_SELF_INTERACTION_SYSTEM_PROMPT,
    assistant_name_from_model,
)
from characters.introspection_sft_data_config import IntrospectionSftDataConfig
from characters.response_generation import load_jsonl_rows, write_jsonl_rows


Message = dict[str, str]


@dataclass(slots=True)
class IntrospectionSftDataSummary:
    train_rows: int
    val_rows: int
    reflection_rows: int
    interaction_rows: int


def build_introspection_sft_dataset(config: IntrospectionSftDataConfig) -> IntrospectionSftDataSummary:
    reflection_rows = [
        _build_reflection_output_row(row)
        for row in load_jsonl_rows(config.paths.reflection_input_path)
    ]
    interaction_rows = [
        _build_interaction_output_row(row)
        for row in load_jsonl_rows(config.paths.interaction_input_path)
    ]
    all_rows = reflection_rows + interaction_rows

    train_rows: list[dict[str, object]] = []
    val_rows: list[dict[str, object]] = []
    for row in sorted(all_rows, key=_stable_output_key):
        if _choose_split(config, row) == "train":
            train_rows.append(row)
        else:
            val_rows.append(row)

    output_dir = config.paths.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_rows(output_dir / "train.jsonl", train_rows)
    write_jsonl_rows(output_dir / "val.jsonl", val_rows)
    return IntrospectionSftDataSummary(
        train_rows=len(train_rows),
        val_rows=len(val_rows),
        reflection_rows=len(reflection_rows),
        interaction_rows=len(interaction_rows),
    )


def _build_reflection_output_row(row: dict[str, object]) -> dict[str, object]:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("Self-reflection row is missing system/user prompt messages.")
    user_message = copy.deepcopy(messages[1])
    if user_message.get("role") != "user":
        raise ValueError("Expected self-reflection row to contain a user prompt at messages[1].")

    generated = str(row.get("generated", "")).strip()
    output_row: dict[str, object] = {
        "messages": [
            {"role": "user", "content": str(user_message.get("content", ""))},
            {"role": "assistant", "content": generated},
        ],
        "source": "self_reflection",
        "conversation_index": -1,
        "turn_count": 0,
        "prompt_name": str(row.get("prompt_name", "")),
        "sample_index": int(row.get("sample_index", 0)),
    }
    if "base_model" in row:
        output_row["base_model"] = str(row["base_model"])
    if "adapter_dir" in row:
        output_row["adapter_dir"] = str(row["adapter_dir"])
    return output_row


def _build_interaction_output_row(row: dict[str, object]) -> dict[str, object]:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Self-interaction row is missing transcript messages.")
    rewritten_messages = [copy.deepcopy(message) for message in messages if isinstance(message, dict)]
    if not rewritten_messages or rewritten_messages[0].get("role") != "system":
        raise ValueError("Expected self-interaction transcript to begin with a system prompt.")

    base_model = str(row.get("base_model", ""))
    assistant_name = assistant_name_from_model(base_model)
    rewritten_messages[0]["content"] = INTROSPECTION_SFT_SELF_INTERACTION_SYSTEM_PROMPT.format(
        NAME=assistant_name
    )

    output_row: dict[str, object] = {
        "messages": rewritten_messages,
        "source": str(row.get("source", "")),
        "conversation_index": int(row.get("conversation_index", 0)),
        "turn_count": int(row.get("turn_count", 0)),
        "prompt_name": "",
        "sample_index": -1,
    }
    if base_model:
        output_row["base_model"] = base_model
    if "adapter_dir" in row:
        output_row["adapter_dir"] = str(row["adapter_dir"])
    return output_row


def _stable_output_key(row: dict[str, object]) -> tuple[object, ...]:
    if row.get("source") == "self_reflection":
        return (
            "self_reflection",
            str(row.get("prompt_name", "")),
            int(row.get("sample_index", 0)),
        )
    return (
        str(row.get("source", "")),
        int(row.get("conversation_index", 0)),
        int(row.get("turn_count", 0)),
    )


def _split_group_value(row: dict[str, object]) -> str:
    if row.get("source") == "self_reflection":
        return f"self_reflection:{row.get('prompt_name', '')}"
    return f"{row.get('source', '')}:{row.get('conversation_index', 0)}"


def _choose_split(config: IntrospectionSftDataConfig, row: dict[str, object]) -> str:
    if config.splits.val == 0:
        return "train"
    group_value = _split_group_value(row)
    digest = hashlib.sha256(f"{config.splits.seed}:{group_value}".encode("utf-8")).hexdigest()
    bucket = int(digest, 16) / float(16 ** len(digest))
    if bucket < config.splits.train:
        return "train"
    return "val"
