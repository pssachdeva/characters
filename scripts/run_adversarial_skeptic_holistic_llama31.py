import argparse
import json
import subprocess
from pathlib import Path

from characters.dpo_dataset_config import load_dpo_dataset_config
from characters.dpo_format import write_dpo_dataset
from characters.introspection_sft_data import build_introspection_sft_dataset
from characters.introspection_sft_data_config import load_introspection_sft_data_config
from characters.response_generation_batch_config import load_response_generation_batch_config
from characters.self_interaction_config import load_self_interaction_config
from characters.self_reflection_config import load_self_reflection_config
from characters.trl_dpo_config import load_trl_dpo_config


RUN_NAME = "adversarial_skeptic_holistic_llama31"
CONFIG_DIR = Path("configs") / RUN_NAME
SUMMARY_DIR = Path("outputs") / "pipeline_summaries" / RUN_NAME

PROMPT_POOL_PATH = Path("outputs/distillation_prompt_pools/adversarial_skeptic_lima.jsonl")
STUDENT_PATH = Path("outputs/student_generation/adversarial_skeptic_lima_llama31.jsonl")
PROMPT_POOL_ROWS = 1830
STUDENT_ROWS = 5490
TEACHER_ROWS = 5490
SELF_REFLECTION_ROWS = 10000
SELF_INTERACTION_ROWS = 2000
INTROSPECTION_SFT_ROWS = SELF_REFLECTION_ROWS + SELF_INTERACTION_ROWS
MODAL_RESULTS_VOLUME = "characters-trl-dpo-results"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Run validation and local build steps only; do not submit Together or Modal jobs.",
    )
    args = parser.parse_args()

    _verify_reused_inputs()
    teacher_ready = _ensure_teacher_generation(local_only=args.local_only)
    if not teacher_ready:
        return

    _ensure_dpo_dataset()
    dpo_ready = _ensure_modal_stage(
        stage="trl_dpo",
        command=[
            "uv",
            "run",
            "modal",
            "run",
            "scripts/modal_train_trl_dpo.py",
            "--config",
            str(CONFIG_DIR / "trl_dpo.yaml"),
        ],
        local_only=args.local_only,
    )
    if not dpo_ready:
        return

    reflection_ready = _ensure_introspection_generation(
        stage="self_reflection",
        config_path=CONFIG_DIR / "self_reflection.yaml",
        modal_script="scripts/modal_generate_self_reflection.py",
        expected_rows=SELF_REFLECTION_ROWS,
        local_only=args.local_only,
    )
    if not reflection_ready:
        return

    interaction_ready = _ensure_introspection_generation(
        stage="self_interaction",
        config_path=CONFIG_DIR / "self_interaction.yaml",
        modal_script="scripts/modal_generate_self_interaction.py",
        expected_rows=SELF_INTERACTION_ROWS,
        local_only=args.local_only,
    )
    if not interaction_ready:
        return

    _ensure_introspection_sft_data()
    _ensure_modal_stage(
        stage="introspection_sft",
        command=[
            "uv",
            "run",
            "modal",
            "run",
            "scripts/modal_train_introspection_sft.py",
            "--config",
            str(CONFIG_DIR / "introspection_sft.yaml"),
        ],
        local_only=args.local_only,
    )


def _verify_reused_inputs() -> None:
    prompt_rows = _count_jsonl_rows(PROMPT_POOL_PATH)
    student_rows = _count_jsonl_rows(STUDENT_PATH)
    _write_summary(
        "inputs",
        {
            "prompt_pool_path": str(PROMPT_POOL_PATH),
            "prompt_pool_rows": prompt_rows,
            "student_path": str(STUDENT_PATH),
            "student_rows": student_rows,
        },
    )
    if prompt_rows != PROMPT_POOL_ROWS:
        raise RuntimeError(f"Expected {PROMPT_POOL_ROWS} prompt-pool rows, found {prompt_rows}.")
    if student_rows != STUDENT_ROWS:
        raise RuntimeError(f"Expected {STUDENT_ROWS} student rows, found {student_rows}.")


def _ensure_teacher_generation(*, local_only: bool) -> bool:
    config = load_response_generation_batch_config(CONFIG_DIR / "teacher_generation_batch.yaml")
    output_rows = _count_jsonl_rows(config.paths.output_path)
    if output_rows == TEACHER_ROWS:
        _write_summary(
            "teacher_generation",
            {"status": "skipped", "output_path": str(config.paths.output_path), "rows": output_rows},
        )
        return True
    if local_only:
        _write_summary(
            "teacher_generation",
            {"status": "pending", "output_path": str(config.paths.output_path), "rows": output_rows},
        )
        return False

    metadata_path = config.paths.batch_dir / "batch_job.json"
    if metadata_path.exists():
        _run_command(
            [
                "uv",
                "run",
                "python",
                "scripts/process_response_generation_batch.py",
                "--config",
                str(CONFIG_DIR / "teacher_generation_batch.yaml"),
            ]
        )
        output_rows = _count_jsonl_rows(config.paths.output_path)
        _write_summary(
            "teacher_generation",
            {"status": "processed_batch", "output_path": str(config.paths.output_path), "rows": output_rows},
        )
        return output_rows == TEACHER_ROWS

    _run_command(
        [
            "uv",
            "run",
            "python",
            "scripts/response_generation_batch.py",
            "--config",
            str(CONFIG_DIR / "teacher_generation_batch.yaml"),
        ]
    )
    _write_summary(
        "teacher_generation",
        {"status": "submitted_batch", "output_path": str(config.paths.output_path), "rows": output_rows},
    )
    return False


def _ensure_dpo_dataset() -> None:
    config = load_dpo_dataset_config(CONFIG_DIR / "dpo_dataset.yaml")
    train_path = config.paths.output_dir / "train.jsonl"
    val_path = config.paths.output_dir / "val.jsonl"
    existing_rows = _count_jsonl_rows(train_path) + _count_jsonl_rows(val_path)
    if existing_rows > 0:
        _write_summary(
            "dpo_dataset",
            {
                "status": "skipped",
                "train_rows": _count_jsonl_rows(train_path),
                "val_rows": _count_jsonl_rows(val_path),
                "output_dir": str(config.paths.output_dir),
            },
        )
        return
    summary = write_dpo_dataset(config)
    _write_summary(
        "dpo_dataset",
        {
            "status": "built",
            "train_rows": summary.train_rows,
            "val_rows": summary.val_rows,
            "dropped_rows": summary.dropped_rows,
            "missing_student_rows": summary.missing_student_rows,
            "missing_teacher_rows": summary.missing_teacher_rows,
            "output_dir": str(config.paths.output_dir),
        },
    )
    if summary.missing_student_rows or summary.missing_teacher_rows:
        raise RuntimeError("DPO merge completed with unmatched teacher/student rows.")


def _ensure_modal_stage(
    *,
    stage: str,
    command: list[str],
    local_only: bool,
) -> bool:
    marker = SUMMARY_DIR / f"{stage}.json"
    if _summary_status(stage) == "completed":
        return True
    if local_only:
        _write_summary(stage, {"status": "pending", "command": command})
        return False
    _run_command(command)
    _write_summary(stage, {"status": "completed", "command": command})
    return True


def _ensure_introspection_generation(
    *,
    stage: str,
    config_path: Path,
    modal_script: str,
    expected_rows: int,
    local_only: bool,
) -> bool:
    if stage == "self_reflection":
        config = load_self_reflection_config(config_path)
    else:
        config = load_self_interaction_config(config_path)

    local_rows = _count_jsonl_rows(config.paths.output_path)
    if local_rows == expected_rows:
        _write_summary(
            stage,
            {"status": "skipped", "output_path": str(config.paths.output_path), "rows": local_rows},
        )
        return True

    marker = SUMMARY_DIR / f"{stage}.json"
    if _summary_status(stage) not in {"remote_generated", "synced"}:
        if local_only:
            _write_summary(stage, {"status": "pending", "output_path": str(config.paths.output_path)})
            return False
        _run_command(["uv", "run", "modal", "run", modal_script, "--config", str(config_path)])
        _write_summary(stage, {"status": "remote_generated", "output_path": str(config.paths.output_path)})

    if local_only:
        return False

    _sync_modal_introspection_output(stage, config.name, config.paths.output_path)
    synced_rows = _count_jsonl_rows(config.paths.output_path)
    _write_summary(
        stage,
        {"status": "synced", "output_path": str(config.paths.output_path), "rows": synced_rows},
    )
    return synced_rows == expected_rows


def _sync_modal_introspection_output(stage: str, config_name: str, output_path: Path) -> None:
    source_config = load_trl_dpo_config(CONFIG_DIR / "trl_dpo.yaml")
    remote_path = f"/{source_config.name}/{stage}/{config_name}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run_command(
        [
            "uv",
            "run",
            "modal",
            "volume",
            "get",
            MODAL_RESULTS_VOLUME,
            remote_path,
            str(output_path),
        ]
    )


def _ensure_introspection_sft_data() -> None:
    config = load_introspection_sft_data_config(CONFIG_DIR / "introspection_sft_data.yaml")
    train_path = config.paths.output_dir / "train.jsonl"
    val_path = config.paths.output_dir / "val.jsonl"
    existing_rows = _count_jsonl_rows(train_path) + _count_jsonl_rows(val_path)
    if existing_rows == INTROSPECTION_SFT_ROWS:
        _write_summary(
            "introspection_sft_data",
            {
                "status": "skipped",
                "train_rows": _count_jsonl_rows(train_path),
                "val_rows": _count_jsonl_rows(val_path),
                "output_dir": str(config.paths.output_dir),
            },
        )
        return
    summary = build_introspection_sft_dataset(config)
    _write_summary(
        "introspection_sft_data",
        {
            "status": "built",
            "train_rows": summary.train_rows,
            "val_rows": summary.val_rows,
            "reflection_rows": summary.reflection_rows,
            "interaction_rows": summary.interaction_rows,
            "output_dir": str(config.paths.output_dir),
        },
    )


def _run_command(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _write_summary(stage: str, payload: dict[str, object]) -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    summary = {"stage": stage, **payload}
    (SUMMARY_DIR / f"{stage}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def _summary_status(stage: str) -> str | None:
    path = SUMMARY_DIR / f"{stage}.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    status = payload.get("status")
    return str(status) if status is not None else None


if __name__ == "__main__":
    main()
