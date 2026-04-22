import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Callable

from characters.prompt_templates import load_prompt_template
from characters.response_generation import (
    Row,
    _load_completed_response_rows,
    _response_row_key,
    load_prompt_rows,
    repeat_prompt_rows,
    write_jsonl_rows,
)
from characters.response_generation_batch_config import ResponseGenerationBatchConfig
from characters.student_generation import render_student_messages
from characters.teacher_generation import load_constitution_block, render_teacher_messages


BATCH_INPUT_FILENAME = "batch_input.jsonl"
BATCH_MANIFEST_FILENAME = "request_manifest.jsonl"
BATCH_METADATA_FILENAME = "batch_job.json"
BATCH_OUTPUT_FILENAME = "batch_output.jsonl"
BATCH_ERROR_FILENAME = "batch_errors.jsonl"


@dataclass(slots=True)
class ResponseGenerationBatchSubmissionSummary:
    batch_id: str | None
    batch_status: str
    batch_dir: Path
    output_path: Path
    prompts: int
    pending_requests: int
    already_present: int


def submit_response_generation_batch(
    config: ResponseGenerationBatchConfig,
) -> ResponseGenerationBatchSubmissionSummary:
    response_key = _response_key_for_generation_type(config.generation_type)
    rows = _load_generation_rows(config)
    print(f"Loading prompt template from {config.paths.prompt_path}...")
    template = load_prompt_template(config.paths.prompt_path)
    completed_rows = _load_completed_response_rows(config.paths.output_path, response_key=response_key)
    pending_rows = [row for row in rows if _response_row_key(row) not in completed_rows]
    print(
        f"Preparing {len(pending_rows)} pending {response_key} requests "
        f"({len(completed_rows)} already present)..."
    )
    if not pending_rows:
        return ResponseGenerationBatchSubmissionSummary(
            batch_id=None,
            batch_status="NOOP",
            batch_dir=config.paths.batch_dir,
            output_path=config.paths.output_path,
            prompts=len(rows),
            pending_requests=0,
            already_present=len(completed_rows),
        )

    _guard_against_unprocessed_batch(config.paths.batch_dir)
    render_messages = _build_render_messages(config, template)
    request_records: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    for row in pending_rows:
        custom_id = _build_custom_id(config.generation_type, row)
        request_records.append(
            {
                "custom_id": custom_id,
                "body": {
                    "model": config.model.name,
                    "messages": render_messages(row),
                    "temperature": config.sampling.temperature,
                    "top_p": config.sampling.top_p,
                    "max_tokens": config.sampling.max_tokens,
                },
            }
        )
        manifest_rows.append(
            {
                "custom_id": custom_id,
                **row,
            }
        )

    config.paths.batch_dir.mkdir(parents=True, exist_ok=True)
    input_path = config.paths.batch_dir / BATCH_INPUT_FILENAME
    manifest_path = config.paths.batch_dir / BATCH_MANIFEST_FILENAME
    write_jsonl_rows(input_path, request_records)
    write_jsonl_rows(manifest_path, manifest_rows)

    client = _build_together_client()
    file_response = _upload_batch_input(client, input_path)
    batch = _create_batch(
        client,
        input_file_id=str(file_response.id),
        endpoint=config.batch.endpoint,
        model_id=config.model.name,
    )
    metadata = {
        "name": config.name,
        "generation_type": config.generation_type,
        "response_key": response_key,
        "submitted_at": _utc_now(),
        "request_count": len(request_records),
        "input_file_id": str(file_response.id),
        "batch_id": str(batch.id),
        "batch_status": str(batch.status),
        "endpoint": config.batch.endpoint,
        "model_name": config.model.name,
        "final_output_path": str(config.paths.output_path),
        "input_path": str(input_path),
        "manifest_path": str(manifest_path),
        "batch_output_path": str(config.paths.batch_dir / BATCH_OUTPUT_FILENAME),
        "batch_error_path": str(config.paths.batch_dir / BATCH_ERROR_FILENAME),
        "assembled_at": None,
    }
    _write_json(config.paths.batch_dir / BATCH_METADATA_FILENAME, metadata)
    return ResponseGenerationBatchSubmissionSummary(
        batch_id=str(batch.id),
        batch_status=str(batch.status),
        batch_dir=config.paths.batch_dir,
        output_path=config.paths.output_path,
        prompts=len(rows),
        pending_requests=len(pending_rows),
        already_present=len(completed_rows),
    )


def _load_generation_rows(config: ResponseGenerationBatchConfig) -> list[Row]:
    print(f"Loading distillation prompts from {config.paths.input_path}...")
    try:
        prompt_rows = load_prompt_rows(config.paths.input_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{config.generation_type.title()} batch generation input not found: {config.paths.input_path}"
        ) from exc
    print(f"Loaded {len(prompt_rows)} prompt rows")
    return repeat_prompt_rows(
        prompt_rows,
        n_samples_per_prompt=config.n_samples_per_prompt,
    )


def _build_render_messages(
    config: ResponseGenerationBatchConfig,
    template: str,
) -> Callable[[Row], list[dict[str, str]]]:
    if config.generation_type == "teacher":
        constitution_path = config.paths.constitution_path
        if constitution_path is None:
            raise ValueError("teacher batch generation requires paths.constitution")
        print(f"Loading constitution from {constitution_path}...")
        constitution = load_constitution_block(constitution_path)
        return lambda row: render_teacher_messages(
            template,
            constitution=constitution,
            prompt=row["prompt"],
        )
    return lambda row: render_student_messages(
        template,
        prompt=row["prompt"],
    )


def _build_custom_id(generation_type: str, row: Row) -> str:
    digest = hashlib.sha1(
        json.dumps(
            {
                "generation_type": generation_type,
                "trait": row.get("trait", ""),
                "prompt": row.get("prompt", ""),
                "source": row.get("source", ""),
                "sample_index": row.get("sample_index", ""),
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:24]
    prefix = "teacher" if generation_type == "teacher" else "student"
    return f"{prefix}-{digest}"


def _response_key_for_generation_type(generation_type: str) -> str:
    if generation_type == "teacher":
        return "chosen"
    if generation_type == "student":
        return "rejected"
    raise ValueError(f"Unsupported generation type: {generation_type}")


def _guard_against_unprocessed_batch(batch_dir: Path) -> None:
    metadata_path = batch_dir / BATCH_METADATA_FILENAME
    if not metadata_path.exists():
        return
    metadata = _read_json(metadata_path)
    if metadata.get("assembled_at"):
        return
    raise ValueError(
        f"Existing unprocessed batch metadata found at {metadata_path}. "
        "Process or remove that batch before submitting a new one."
    )


def _build_together_client() -> Any:
    from together import Together

    return Together(api_key=_require_together_api_key())


def _upload_batch_input(client: Any, input_path: Path) -> Any:
    return client.files.upload(file=str(input_path), purpose="batch-api", check=False)


def _create_batch(
    client: Any,
    *,
    input_file_id: str,
    endpoint: str,
    model_id: str,
) -> Any:
    create = getattr(client.batches, "create", None)
    if create is not None:
        try:
            response = create(
                input_file_id=input_file_id,
                endpoint=endpoint,
                model_id=model_id,
            )
        except TypeError:
            response = create(
                input_file_id=input_file_id,
                endpoint=endpoint,
            )
        return getattr(response, "job", response)

    create_batch = getattr(client.batches, "create_batch", None)
    if create_batch is not None:
        response = create_batch(input_file_id, endpoint=endpoint)
        return getattr(response, "job", response)

    raise AttributeError("Unsupported Together SDK: no batch create method found")


def retrieve_batch(client: Any, batch_id: str) -> Any:
    for method_name in ("retrieve", "get_batch", "get"):
        method = getattr(client.batches, method_name, None)
        if method is None:
            continue
        return method(batch_id)
    raise AttributeError("Unsupported Together SDK: no batch retrieve method found")


def download_batch_file(client: Any, file_id: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    streaming = getattr(client.files, "with_streaming_response", None)
    if streaming is not None and hasattr(streaming, "content"):
        with streaming.content(id=file_id) as response:
            with destination.open("wb") as handle:
                for chunk in response.iter_bytes():
                    handle.write(chunk)
        return

    retrieve_content = getattr(client.files, "retrieve_content", None)
    if retrieve_content is not None:
        try:
            retrieve_content(id=file_id, output=str(destination))
        except TypeError:
            retrieve_content(file_id, str(destination))
        return

    raise AttributeError("Unsupported Together SDK: no file download method found")


def metadata_to_dict(metadata: Any) -> dict[str, object]:
    if hasattr(metadata, "model_dump"):
        dumped = metadata.model_dump()
        if isinstance(dumped, dict):
            return _make_json_safe(dumped)
    if hasattr(metadata, "dict"):
        dumped = metadata.dict()
        if isinstance(dumped, dict):
            return _make_json_safe(dumped)
    if hasattr(metadata, "__dataclass_fields__"):
        dumped = asdict(metadata)
        if isinstance(dumped, dict):
            return _make_json_safe(dumped)
    if isinstance(metadata, dict):
        return _make_json_safe(metadata)
    result: dict[str, object] = {}
    for key in (
        "id",
        "status",
        "progress",
        "input_file_id",
        "output_file_id",
        "error_file_id",
        "error",
        "created_at",
        "completed_at",
        "job_deadline",
        "endpoint",
        "model_id",
        ):
        value = getattr(metadata, key, None)
        if value is not None:
            result[key] = value
    return _make_json_safe(result)


def _require_together_api_key() -> str:
    value = os.environ.get("TOGETHER_API_KEY")
    if value:
        return value
    raise ValueError("Set TOGETHER_API_KEY for Together batch generation")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_make_json_safe(payload), handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _make_json_safe(value: object) -> object:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {
            str(key): _make_json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_make_json_safe(item) for item in value]
    return value
