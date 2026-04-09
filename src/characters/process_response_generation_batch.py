import json
from dataclasses import dataclass
from pathlib import Path

from characters.response_generation import (
    _load_completed_response_rows,
    _response_row_key,
    append_jsonl_rows,
    load_jsonl_rows,
    write_jsonl_rows,
)
from characters.response_generation_batch import (
    BATCH_ERROR_FILENAME,
    BATCH_MANIFEST_FILENAME,
    BATCH_METADATA_FILENAME,
    BATCH_OUTPUT_FILENAME,
    _build_together_client,
    _response_key_for_generation_type,
    _utc_now,
    _write_json,
    download_batch_file,
    metadata_to_dict,
    retrieve_batch,
)
from characters.response_generation_batch_config import ResponseGenerationBatchConfig


TERMINAL_BATCH_STATUSES = {"COMPLETED", "FAILED", "EXPIRED", "CANCELLED"}


@dataclass(slots=True)
class ResponseGenerationBatchProcessSummary:
    batch_id: str
    batch_status: str
    output_path: Path
    batch_dir: Path
    is_terminal: bool
    is_successful: bool
    downloaded_responses: int
    assembled_rows: int
    error_count: int


def process_response_generation_batch(
    config: ResponseGenerationBatchConfig,
) -> ResponseGenerationBatchProcessSummary:
    metadata_path = config.paths.batch_dir / BATCH_METADATA_FILENAME
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Batch metadata not found: {metadata_path}\n"
            "Submit a batch first with scripts/response_generation_batch.py."
        )

    metadata = _read_json(metadata_path)
    batch_id = str(metadata["batch_id"])
    client = _build_together_client()
    batch = retrieve_batch(client, batch_id)
    batch_data = metadata_to_dict(batch)

    metadata["last_checked_at"] = _utc_now()
    metadata["batch_status"] = str(batch_data.get("status", metadata.get("batch_status", "")))
    metadata["batch"] = batch_data

    output_file_id = batch_data.get("output_file_id")
    if output_file_id:
        download_batch_file(client, str(output_file_id), config.paths.batch_dir / BATCH_OUTPUT_FILENAME)

    error_file_id = batch_data.get("error_file_id")
    if error_file_id:
        download_batch_file(client, str(error_file_id), config.paths.batch_dir / BATCH_ERROR_FILENAME)

    response_key = _response_key_for_generation_type(config.generation_type)
    batch_output_path = config.paths.batch_dir / BATCH_OUTPUT_FILENAME
    downloaded_responses = _count_jsonl_rows(batch_output_path)
    error_count = _count_jsonl_rows(config.paths.batch_dir / BATCH_ERROR_FILENAME)
    assembled_rows = 0
    if metadata["batch_status"] in TERMINAL_BATCH_STATUSES:
        assembled_rows = _assemble_batch_outputs(
            manifest_path=config.paths.batch_dir / BATCH_MANIFEST_FILENAME,
            batch_output_path=batch_output_path,
            output_path=config.paths.output_path,
            response_key=response_key,
        )
        metadata["assembled_at"] = _utc_now()
        metadata["assembled_rows"] = assembled_rows
        metadata["error_count"] = error_count

    _write_json(metadata_path, metadata)
    return ResponseGenerationBatchProcessSummary(
        batch_id=batch_id,
        batch_status=str(metadata["batch_status"]),
        output_path=config.paths.output_path,
        batch_dir=config.paths.batch_dir,
        is_terminal=str(metadata["batch_status"]) in TERMINAL_BATCH_STATUSES,
        is_successful=str(metadata["batch_status"]) == "COMPLETED",
        downloaded_responses=downloaded_responses,
        assembled_rows=assembled_rows,
        error_count=error_count,
    )


def _assemble_batch_outputs(
    *,
    manifest_path: Path,
    batch_output_path: Path,
    output_path: Path,
    response_key: str,
) -> int:
    manifest_rows = load_jsonl_rows(manifest_path)
    manifest_by_custom_id = {
        str(row["custom_id"]): {
            key: value
            for key, value in row.items()
            if key != "custom_id"
        }
        for row in manifest_rows
    }
    completed_rows = _load_completed_response_rows(output_path, response_key=response_key)
    batch_results = load_jsonl_rows(batch_output_path) if batch_output_path.exists() else []
    output_rows: list[dict[str, object]] = []
    for result in batch_results:
        custom_id = str(result.get("custom_id", ""))
        row = manifest_by_custom_id.get(custom_id)
        if row is None:
            continue
        if _response_row_key(row) in completed_rows:
            continue
        generation = _extract_generation_text(result)
        output_row = dict(row)
        output_row[response_key] = generation.strip()
        output_rows.append(output_row)
        completed_rows[_response_row_key(row)] = output_row

    append_jsonl_rows(output_path, output_rows)
    if not output_path.exists():
        write_jsonl_rows(output_path, [])
    return len(output_rows)


def _extract_generation_text(result: dict[str, object]) -> str:
    response = result.get("response")
    if not isinstance(response, dict):
        return ""
    body = response.get("body")
    if not isinstance(body, dict):
        return ""
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return ""
    message = first_choice.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload
