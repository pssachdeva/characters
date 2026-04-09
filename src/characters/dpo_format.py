import hashlib
import unicodedata
from dataclasses import dataclass

from characters.dpo_dataset_config import DpoDatasetConfig
from characters.response_generation import load_jsonl_rows, write_jsonl_rows


Message = dict[str, str]


@dataclass(slots=True)
class DpoDatasetSummary:
    train_rows: int
    val_rows: int
    dropped_rows: int


def write_dpo_dataset(config: DpoDatasetConfig) -> DpoDatasetSummary:
    teacher_rows = load_jsonl_rows(config.paths.teacher_input_path)
    student_rows = load_jsonl_rows(config.paths.student_input_path)
    student_by_key = {_row_key(row): row for row in student_rows}
    tokenizer = _maybe_load_tokenizer(config)

    train_rows: list[dict[str, object]] = []
    val_rows: list[dict[str, object]] = []
    dropped_rows = 0

    for teacher_row in teacher_rows:
        student_row = student_by_key.get(_row_key(teacher_row))
        if student_row is None:
            raise ValueError(
                "Missing matching student row for prompt="
                f"{teacher_row['prompt']!r}, trait={teacher_row.get('trait', '')!r}, "
                f"source={teacher_row.get('source', '')!r}, sample_index={teacher_row.get('sample_index', '')!r}"
            )
        formatted_row = _build_output_row(config, teacher_row, student_row)
        if _should_drop_row(config, formatted_row, tokenizer):
            dropped_rows += 1
            continue
        split_name = _choose_split(config, teacher_row)
        if split_name == "train":
            train_rows.append(formatted_row)
        else:
            val_rows.append(formatted_row)

    output_dir = config.paths.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_rows(output_dir / "train.jsonl", train_rows)
    write_jsonl_rows(output_dir / "val.jsonl", val_rows)
    return DpoDatasetSummary(
        train_rows=len(train_rows),
        val_rows=len(val_rows),
        dropped_rows=dropped_rows,
    )


def _row_key(row: dict[str, object]) -> tuple[str, str, str, str]:
    return (
        str(row.get("trait", "")),
        str(row["prompt"]),
        str(row.get("source", "")),
        str(row.get("sample_index", "")),
    )


def _build_output_row(
    config: DpoDatasetConfig,
    teacher_row: dict[str, object],
    student_row: dict[str, object],
) -> dict[str, object]:
    prompt = str(teacher_row["prompt"])
    chosen_text = str(teacher_row["chosen"]).strip()
    rejected_text = str(student_row["rejected"]).strip()
    output_row = _build_formatted_row(config, prompt, chosen_text, rejected_text)
    if config.metadata.keep_trait and "trait" in teacher_row:
        output_row["trait"] = str(teacher_row["trait"])
    if config.metadata.keep_source and "source" in teacher_row:
        output_row["source"] = str(teacher_row["source"])
    if config.metadata.keep_sample_index and "sample_index" in teacher_row:
        output_row["sample_index"] = str(teacher_row["sample_index"])
    if config.metadata.keep_models:
        _copy_model_metadata(output_row, teacher_row, student_row)
    return output_row


def _build_formatted_row(
    config: DpoDatasetConfig,
    prompt: str,
    chosen_text: str,
    rejected_text: str,
) -> dict[str, object]:
    if config.format.type == "openrlhf_chat":
        return {
            config.format.chosen_key: _messages_for_preference(prompt, chosen_text),
            config.format.rejected_key: _messages_for_preference(prompt, rejected_text),
            "prompt": prompt,
        }
    if config.format.type == "nemo_binary_preference":
        return {
            "prompt": prompt,
            config.format.chosen_key: chosen_text,
            config.format.rejected_key: rejected_text,
        }
    if config.format.type == "trl_conversational":
        return {
            "prompt": [{"role": "user", "content": prompt}],
            config.format.chosen_key: [{"role": "assistant", "content": chosen_text}],
            config.format.rejected_key: [{"role": "assistant", "content": rejected_text}],
        }
    raise ValueError(f"Unsupported DPO dataset format: {config.format.type}")


def _copy_model_metadata(
    output_row: dict[str, object],
    teacher_row: dict[str, object],
    student_row: dict[str, object],
) -> None:
    for key in ("model", "model_name", "provider"):
        if key in teacher_row:
            output_row[f"teacher_{key}"] = str(teacher_row[key])
        if key in student_row:
            output_row[f"student_{key}"] = str(student_row[key])


def _should_drop_row(
    config: DpoDatasetConfig,
    row: dict[str, object],
    tokenizer: object | None,
) -> bool:
    _, chosen_text, rejected_text = _preference_texts(config, row)
    if config.filters.drop_empty and (not chosen_text or not rejected_text):
        return True
    if config.filters.drop_identical_pairs and chosen_text == rejected_text:
        return True
    if config.filters.require_terminal_punctuation:
        if not _ends_with_punctuation(chosen_text) or not _ends_with_punctuation(rejected_text):
            return True
    if config.filters.drop_overlength and _is_overlength(config, row, tokenizer):
        return True
    return False


def _messages_for_preference(prompt: str, response: str) -> list[Message]:
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]


def _preference_texts(
    config: DpoDatasetConfig,
    row: dict[str, object],
) -> tuple[str, str, str]:
    prompt = str(row.get("prompt", ""))
    if config.format.type == "openrlhf_chat":
        chosen_text = str(_assistant_content(row[config.format.chosen_key]))
        rejected_text = str(_assistant_content(row[config.format.rejected_key]))
        return prompt, chosen_text, rejected_text
    if config.format.type == "nemo_binary_preference":
        return (
            prompt,
            str(row.get(config.format.chosen_key, "")).strip(),
            str(row.get(config.format.rejected_key, "")).strip(),
        )
    if config.format.type == "trl_conversational":
        prompt_text = _message_contents(row.get("prompt"))
        chosen_text = str(_assistant_content(row[config.format.chosen_key]))
        rejected_text = str(_assistant_content(row[config.format.rejected_key]))
        return prompt_text, chosen_text, rejected_text
    raise ValueError(f"Unsupported DPO dataset format: {config.format.type}")


def _assistant_content(messages: object) -> str:
    if not isinstance(messages, list) or not messages:
        return ""
    final_message = messages[-1]
    if not isinstance(final_message, dict):
        return ""
    return str(final_message.get("content", "")).strip()


def _message_contents(messages: object) -> str:
    if not isinstance(messages, list):
        return ""
    contents: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = str(message.get("content", "")).strip()
        if content:
            contents.append(content)
    return "\n".join(contents)


def _ends_with_punctuation(text: str) -> bool:
    stripped = text.rstrip()
    return bool(stripped) and unicodedata.category(stripped[-1]).startswith("P")


def _maybe_load_tokenizer(config: DpoDatasetConfig) -> object | None:
    if not config.filters.drop_overlength:
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is required for DPO length filtering. "
            "Install transformers or set filters.drop_overlength=false."
        ) from exc
    return AutoTokenizer.from_pretrained(config.target_model.effective_tokenizer_name)


def _is_overlength(
    config: DpoDatasetConfig,
    row: dict[str, object],
    tokenizer: object | None,
) -> bool:
    if tokenizer is None:
        return False
    prompt, chosen_text, rejected_text = _preference_texts(config, row)
    chosen_length = _encoded_length(
        tokenizer,
        _messages_for_preference(prompt, chosen_text),
        config.target_model.apply_chat_template,
    )
    rejected_length = _encoded_length(
        tokenizer,
        _messages_for_preference(prompt, rejected_text),
        config.target_model.apply_chat_template,
    )
    return max(chosen_length, rejected_length) > config.filters.max_length


def _encoded_length(
    tokenizer: object,
    messages: object,
    apply_chat_template: bool,
) -> int:
    if not isinstance(messages, list):
        return 0
    if apply_chat_template:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        rendered = "\n".join(f"{message['role']}: {message['content']}" for message in messages)
    return len(tokenizer.encode(rendered))


def _choose_split(
    config: DpoDatasetConfig,
    row: dict[str, object],
) -> str:
    if config.splits.val == 0:
        return "train"
    group_value = str(row["prompt"])
    digest = hashlib.sha256(f"{config.splits.seed}:{group_value}".encode("utf-8")).hexdigest()
    bucket = int(digest, 16) / float(16**len(digest))
    if bucket < config.splits.train:
        return "train"
    return "val"
