import json
from pathlib import Path

import pytest

from characters.distillation_prompt_pool import (
    build_distillation_prompt_pool,
    ensure_prompt_source_files,
    extract_first_user_turn_prompt,
    extract_prompt_field,
    extract_first_lima_prompt,
)
from characters.distillation_prompt_pool_config import (
    PromptSourceConfig,
    load_distillation_prompt_pool_config,
)
from characters.dpo_dataset_config import load_dpo_dataset_config
from characters.dpo_format import write_dpo_dataset
from characters.response_generation import (
    flatten_expanded_prompts,
    load_prompt_rows,
    repeat_prompt_rows,
)
from characters.response_generation_config import load_response_generation_config
from characters.student_generation import render_student_messages, run_student_generation
from characters.teacher_generation import render_teacher_messages, run_teacher_generation


class FakeProviderBackend:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls: list[list[list[dict[str, str]]]] = []
        self.cursor = 0

    def generate_texts(self, **kwargs: object) -> list[str]:
        messages_batch = kwargs["messages_batch"]
        self.calls.append(messages_batch)
        start = self.cursor
        end = start + len(messages_batch)
        self.cursor = end
        return self.outputs[start:end]


def test_load_response_generation_config_for_teacher(tmp_path: Path) -> None:
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "teacher.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: teacher_test",
                "paths:",
                "  input: input.jsonl",
                "  constitution: constitution.json",
                "  prompt: teacher.txt",
                "  output: out.jsonl",
                "model:",
                "  provider: openai",
                "  name: gpt-5-mini",
            ]
        ),
        encoding="utf-8",
    )
    config = load_response_generation_config(config_path)
    assert config.paths.input_path == (tmp_path / "input.jsonl").resolve()
    assert config.paths.constitution_path == (tmp_path / "constitution.json").resolve()
    assert config.paths.prompt_path == (tmp_path / "teacher.txt").resolve()
    assert config.paths.output_path == (tmp_path / "out.jsonl").resolve()
    assert config.n_samples_per_prompt == 1
    monkeypatch.undo()


def test_load_example_response_generation_configs() -> None:
    teacher_config = load_response_generation_config(
        "configs/teacher_generation/adversarial_skeptic_glm5.yaml"
    )
    assert teacher_config.paths.input_path.as_posix().endswith(
        "outputs/distillation_prompt_pools/adversarial_skeptic_lima.jsonl"
    )
    assert teacher_config.paths.constitution_path is not None
    assert teacher_config.paths.constitution_path.as_posix().endswith(
        "constitutions/hand-written/adversarial_skeptic.txt"
    )
    assert teacher_config.paths.prompt_path.as_posix().endswith(
        "prompts/teacher_generation/teacher_response.txt"
    )
    assert teacher_config.paths.output_path.as_posix().endswith(
        "outputs/teacher_generation/adversarial_skeptic_glm5.jsonl"
    )
    assert teacher_config.n_samples_per_prompt == 3

    student_config = load_response_generation_config(
        "configs/student_generation/adversarial_skeptic_llama31.yaml"
    )
    assert student_config.paths.input_path.as_posix().endswith(
        "outputs/distillation_prompt_pools/adversarial_skeptic_lima.jsonl"
    )
    assert student_config.paths.constitution_path is None
    assert student_config.paths.prompt_path.as_posix().endswith(
        "prompts/student_generation/student_response.txt"
    )
    assert student_config.paths.output_path.as_posix().endswith(
        "outputs/student_generation/adversarial_skeptic_llama31.jsonl"
    )
    assert student_config.n_samples_per_prompt == 3


def test_load_example_dpo_dataset_config() -> None:
    config = load_dpo_dataset_config("configs/dpo/adversarial_skeptic_llama31.yaml")
    assert config.paths.teacher_input_path.as_posix().endswith(
        "outputs/teacher_generation/adversarial_skeptic_glm5.jsonl"
    )
    assert config.paths.student_input_path.as_posix().endswith(
        "outputs/student_generation/adversarial_skeptic_llama31.jsonl"
    )
    assert config.paths.output_dir.as_posix().endswith("outputs/dpo/adversarial_skeptic_llama31")
    assert config.target_model.name == "meta-llama/llama-3.1-8b-instruct"
    assert config.target_model.effective_tokenizer_name == "meta-llama/llama-3.1-8b-instruct"
    assert config.splits.train == 0.85
    assert config.splits.val == 0.15


def test_load_example_nemo_dpo_dataset_config() -> None:
    config = load_dpo_dataset_config("configs/dpo/adversarial_skeptic_llama31_nemo.yaml")
    assert config.paths.teacher_input_path.as_posix().endswith(
        "outputs/teacher_generation/adversarial_skeptic_glm5.jsonl"
    )
    assert config.paths.student_input_path.as_posix().endswith(
        "outputs/student_generation/adversarial_skeptic_llama31.jsonl"
    )
    assert config.paths.output_dir.as_posix().endswith("outputs/dpo/adversarial_skeptic_llama31_nemo")
    assert config.format.type == "nemo_binary_preference"
    assert config.splits.train == 0.85
    assert config.splits.val == 0.15


def test_flatten_expanded_prompts_marks_seed_and_additional() -> None:
    rows = [
        {
            "trait": "Trait A",
            "questions": ["Seed one?"],
            "additional_questions": ["Added one?", "Added two?"],
        }
    ]
    assert flatten_expanded_prompts(rows) == [
        {"trait": "Trait A", "prompt": "Seed one?", "source": "seed"},
        {"trait": "Trait A", "prompt": "Added one?", "source": "additional"},
        {"trait": "Trait A", "prompt": "Added two?", "source": "additional"},
    ]


def test_repeat_prompt_rows_adds_sample_index() -> None:
    rows = [{"trait": "Trait A", "prompt": "Seed one?", "source": "seed"}]
    assert repeat_prompt_rows(rows, n_samples_per_prompt=2) == [
        {
            "trait": "Trait A",
            "prompt": "Seed one?",
            "source": "seed",
            "sample_index": "0",
        },
        {
            "trait": "Trait A",
            "prompt": "Seed one?",
            "source": "seed",
            "sample_index": "1",
        },
    ]


def test_load_prompt_rows_accepts_flat_prompt_pool_rows(tmp_path: Path) -> None:
    input_path = tmp_path / "prompt_pool.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "prompt": "What do you think?",
                        "trait": "",
                        "source": "lima_train",
                    }
                ),
                json.dumps(
                    {
                        "prompt": "Explain this idea.",
                        "trait": "Trait A",
                        "source": "constitution_seed",
                        "bucket": 7,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert load_prompt_rows(input_path) == [
        {"prompt": "What do you think?", "trait": "", "source": "lima_train"},
        {
            "prompt": "Explain this idea.",
            "trait": "Trait A",
            "source": "constitution_seed",
            "bucket": "7",
        },
    ]


def test_render_teacher_messages_uses_system_prompt_and_raw_user_prompt() -> None:
    messages = render_teacher_messages(
        "Constitution:\n{constitution}",
        constitution="- Trait one\n- Trait two",
        prompt="What should I do?",
    )
    assert [message["role"] for message in messages] == ["system", "user"]
    assert "- Trait two" in messages[0]["content"]
    assert messages[1]["content"] == "What should I do?"


def test_render_student_messages_is_user_only() -> None:
    messages = render_student_messages("User message:\n{prompt}", prompt="Help me")
    assert messages == [{"role": "user", "content": "User message:\nHelp me"}]


def test_run_teacher_generation_with_fake_provider(tmp_path: Path) -> None:
    input_path = tmp_path / "expanded.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "trait": "Trait A",
                        "questions": ["Seed one?"],
                        "additional_questions": ["Added one?"],
                    }
                )
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    constitution_path = tmp_path / "constitution.json"
    constitution_path.write_text(
        json.dumps([{"trait": "Trait A", "questions": ["Seed one?"]}]),
        encoding="utf-8",
    )
    prompt_path = tmp_path / "teacher.txt"
    prompt_path.write_text(
        "Constitution:\n{constitution}",
        encoding="utf-8",
    )
    config_path = tmp_path / "teacher.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: teacher_test",
                "n_samples_per_prompt: 2",
                "paths:",
                f"  input: {input_path}",
                f"  constitution: {constitution_path}",
                f"  prompt: {prompt_path}",
                f"  output: {tmp_path / 'teacher_out.jsonl'}",
                "model:",
                "  provider: openai",
                "  name: gpt-5-mini",
            ]
        ),
        encoding="utf-8",
    )
    backend = FakeProviderBackend(
        [
            "Chosen seed sample 0",
            "Chosen seed sample 1",
            "Chosen additional sample 0",
            "Chosen additional sample 1",
        ]
    )
    summary = run_teacher_generation(load_response_generation_config(config_path), backend)
    assert summary.prompts == 4
    rows = [
        json.loads(line)
        for line in (tmp_path / "teacher_out.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows == [
        {
            "trait": "Trait A",
            "prompt": "Seed one?",
            "source": "seed",
            "sample_index": "0",
            "chosen": "Chosen seed sample 0",
        },
        {
            "trait": "Trait A",
            "prompt": "Seed one?",
            "source": "seed",
            "sample_index": "1",
            "chosen": "Chosen seed sample 1",
        },
        {
            "trait": "Trait A",
            "prompt": "Added one?",
            "source": "additional",
            "sample_index": "0",
            "chosen": "Chosen additional sample 0",
        },
        {
            "trait": "Trait A",
            "prompt": "Added one?",
            "source": "additional",
            "sample_index": "1",
            "chosen": "Chosen additional sample 1",
        },
    ]
    assert backend.calls[0][0][0]["role"] == "system"
    assert "Constitution:" in backend.calls[0][0][0]["content"]
    assert backend.calls[0][0][1] == {"role": "user", "content": "Seed one?"}


def test_run_teacher_generation_missing_input_has_actionable_error(tmp_path: Path) -> None:
    constitution_path = tmp_path / "constitution.json"
    constitution_path.write_text(
        json.dumps([{"trait": "Trait A", "questions": ["Seed one?"]}]),
        encoding="utf-8",
    )
    prompt_path = tmp_path / "teacher.txt"
    prompt_path.write_text(
        "Constitution:\n{constitution}",
        encoding="utf-8",
    )
    missing_input_path = tmp_path / "missing.jsonl"
    config_path = tmp_path / "teacher.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: teacher_test",
                "paths:",
                f"  input: {missing_input_path}",
                f"  constitution: {constitution_path}",
                f"  prompt: {prompt_path}",
                f"  output: {tmp_path / 'teacher_out.jsonl'}",
                "model:",
                "  provider: openai",
                "  name: gpt-5-mini",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="Build the prompt pool first"):
        run_teacher_generation(
            load_response_generation_config(config_path),
            FakeProviderBackend([]),
        )


def test_run_student_generation_with_fake_provider(tmp_path: Path) -> None:
    input_path = tmp_path / "prompt_pool.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "trait": "Trait A",
                        "prompt": "Seed one?",
                        "source": "constitution_seed",
                    }
                ),
                json.dumps(
                    {
                        "trait": "",
                        "prompt": "LIMA one?",
                        "source": "lima_train",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    prompt_path = tmp_path / "student.txt"
    prompt_path.write_text("User message:\n{prompt}", encoding="utf-8")
    config_path = tmp_path / "student.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: student_test",
                "n_samples_per_prompt: 2",
                "paths:",
                f"  input: {input_path}",
                f"  prompt: {prompt_path}",
                f"  output: {tmp_path / 'student_out.jsonl'}",
                "model:",
                "  provider: openai",
                "  name: gpt-5-mini",
            ]
        ),
        encoding="utf-8",
    )
    backend = FakeProviderBackend(
        [
            "Rejected seed sample 0",
            "Rejected seed sample 1",
            "Rejected lima sample 0",
            "Rejected lima sample 1",
        ]
    )
    summary = run_student_generation(load_response_generation_config(config_path), backend)
    assert summary.prompts == 4
    rows = [
        json.loads(line)
        for line in (tmp_path / "student_out.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows == [
        {
            "trait": "Trait A",
            "prompt": "Seed one?",
            "source": "constitution_seed",
            "sample_index": "0",
            "rejected": "Rejected seed sample 0",
        },
        {
            "trait": "Trait A",
            "prompt": "Seed one?",
            "source": "constitution_seed",
            "sample_index": "1",
            "rejected": "Rejected seed sample 1",
        },
        {
            "trait": "",
            "prompt": "LIMA one?",
            "source": "lima_train",
            "sample_index": "0",
            "rejected": "Rejected lima sample 0",
        },
        {
            "trait": "",
            "prompt": "LIMA one?",
            "source": "lima_train",
            "sample_index": "1",
            "rejected": "Rejected lima sample 1",
        },
    ]
    assert backend.calls[0][0] == [{"role": "user", "content": "User message:\nSeed one?"}]
    assert backend.calls[0][2] == [{"role": "user", "content": "User message:\nLIMA one?"}]


def test_run_student_generation_resumes_from_existing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "prompt_pool.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "trait": "Trait A",
                        "prompt": "Seed one?",
                        "source": "constitution_seed",
                    }
                ),
                json.dumps(
                    {
                        "trait": "",
                        "prompt": "LIMA one?",
                        "source": "lima_train",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "student_out.jsonl"
    output_path.write_text(
        json.dumps(
            {
                "trait": "Trait A",
                "prompt": "Seed one?",
                "source": "constitution_seed",
                "sample_index": "0",
                "rejected": "Already generated",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    prompt_path = tmp_path / "student.txt"
    prompt_path.write_text("User message:\n{prompt}", encoding="utf-8")
    config_path = tmp_path / "student.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: student_test",
                "n_samples_per_prompt: 1",
                "paths:",
                f"  input: {input_path}",
                f"  prompt: {prompt_path}",
                f"  output: {output_path}",
                "model:",
                "  provider: openai",
                "  name: gpt-5-mini",
                "  max_concurrency: 1",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("characters.response_generation._checkpoint_batch_size", lambda config: 1)
    backend = FakeProviderBackend(["Generated lima"])
    summary = run_student_generation(load_response_generation_config(config_path), backend)
    assert summary.prompts == 2
    assert summary.generated_responses == 2
    assert len(backend.calls) == 1
    assert backend.calls[0] == [[{"role": "user", "content": "User message:\nLIMA one?"}]]
    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows == [
        {
            "trait": "Trait A",
            "prompt": "Seed one?",
            "source": "constitution_seed",
            "sample_index": "0",
            "rejected": "Already generated",
        },
        {
            "trait": "",
            "prompt": "LIMA one?",
            "source": "lima_train",
            "sample_index": "0",
            "rejected": "Generated lima",
        },
    ]


def test_load_distillation_prompt_pool_config(tmp_path: Path) -> None:
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "prompt_pool.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: prompt_pool",
                "paths:",
                "  constitution_input: constitution.jsonl",
                "  output: out.jsonl",
                "sources:",
                "  - type: lima",
                "    path: lima_train.jsonl",
                "    source: lima_train",
                "  - type: jsonl_prompt_field",
                "    path: prompts.jsonl",
                "    source: extra_prompts",
                "    prompt_field: instruction",
            ]
        ),
        encoding="utf-8",
    )
    config = load_distillation_prompt_pool_config(config_path)
    assert config.paths.constitution_input_path == (tmp_path / "constitution.jsonl").resolve()
    assert config.paths.output_path == (tmp_path / "out.jsonl").resolve()
    assert len(config.sources) == 2
    assert config.sources[0].type == "lima"
    assert config.sources[0].path == (tmp_path / "lima_train.jsonl").resolve()
    assert config.sources[0].source == "lima_train"
    assert config.sources[0].metadata == {}
    assert config.sources[1].type == "jsonl_prompt_field"
    assert config.sources[1].prompt_field == "instruction"
    monkeypatch.undo()


def test_load_example_distillation_prompt_pool_config() -> None:
    config = load_distillation_prompt_pool_config(
        "configs/distillation_prompt_pool/adversarial_skeptic_lima.yaml"
    )
    assert config.paths.constitution_input_path.as_posix().endswith(
        "constitutions/full/adversarial_skeptic.jsonl"
    )
    assert config.paths.output_path.as_posix().endswith(
        "outputs/distillation_prompt_pools/adversarial_skeptic_lima.jsonl"
    )
    assert len(config.sources) == 2
    assert config.sources[0].type == "lima"
    assert config.sources[0].path.as_posix().endswith("data/lima/train.jsonl")
    assert config.sources[0].metadata == {}
    assert config.sources[1].type == "lima"
    assert config.sources[1].path.as_posix().endswith("data/lima/test.jsonl")
    assert config.sources[1].metadata == {}


def test_extract_first_lima_prompt_rejects_missing_conversations() -> None:
    with pytest.raises(ValueError, match="missing a non-empty conversations list"):
        extract_first_lima_prompt({}, row_index=0, source="lima_train")


def test_extract_first_user_turn_prompt_rejects_wrong_first_role() -> None:
    with pytest.raises(ValueError, match="first turn role must be 'user'"):
        extract_first_user_turn_prompt(
            {"messages": [{"speaker": "assistant", "text": "bad"}]},
            row_index=0,
            source="custom_dataset",
            conversation_field="messages",
            role_field="speaker",
            content_field="text",
            user_role="user",
        )


def test_extract_prompt_field_reads_custom_field() -> None:
    assert (
        extract_prompt_field(
            {"instruction": "Explain the result."},
            row_index=0,
            source="instructions",
            prompt_field="instruction",
        )
        == "Explain the result."
    )


def test_ensure_prompt_source_files_materializes_missing_lima_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "prompt_pool.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: prompt_pool",
                "paths:",
                f"  constitution_input: {tmp_path / 'constitution.jsonl'}",
                f"  output: {tmp_path / 'prompt_pool.jsonl'}",
                "sources:",
                "  - type: lima",
                f"    path: {tmp_path / 'data' / 'lima' / 'train.jsonl'}",
                "    source: lima_train",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "constitution.jsonl").write_text("", encoding="utf-8")
    captured: dict[str, str] = {}

    def fake_load_hf_dataset_rows(dataset_name: str, split_name: str) -> list[dict[str, object]]:
        captured["dataset_name"] = dataset_name
        captured["split_name"] = split_name
        return [{"conversations": ["Fetched prompt", "Ignored answer"]}]

    monkeypatch.setattr(
        "characters.distillation_prompt_pool._load_hf_dataset_rows",
        fake_load_hf_dataset_rows,
    )
    config = load_distillation_prompt_pool_config(config_path)
    materialized = ensure_prompt_source_files(config)
    assert captured == {"dataset_name": "GAIR/lima", "split_name": "train"}
    assert materialized == [(tmp_path / "data" / "lima" / "train.jsonl").resolve()]
    rows = [
        json.loads(line)
        for line in materialized[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows == [{"conversations": ["Fetched prompt", "Ignored answer"]}]


def test_ensure_prompt_source_files_falls_back_to_direct_lima_jsonl_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "prompt_pool.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: prompt_pool",
                "paths:",
                f"  constitution_input: {tmp_path / 'constitution.jsonl'}",
                f"  output: {tmp_path / 'prompt_pool.jsonl'}",
                "sources:",
                "  - type: lima",
                f"    path: {tmp_path / 'data' / 'lima' / 'test.jsonl'}",
                "    source: lima_test",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "constitution.jsonl").write_text("", encoding="utf-8")
    downloaded = tmp_path / "downloaded_test.jsonl"
    downloaded.write_text(
        json.dumps({"conversations": ["Downloaded prompt", "Ignored answer"]}) + "\n",
        encoding="utf-8",
    )

    def fake_load_dataset(*args: object, **kwargs: object) -> object:
        raise RuntimeError("Dataset scripts are no longer supported, but found lima.py")

    def fake_hf_hub_download(*args: object, **kwargs: object) -> str:
        return str(downloaded)

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)

    config = load_distillation_prompt_pool_config(config_path)
    materialized = ensure_prompt_source_files(config)
    assert materialized == [(tmp_path / "data" / "lima" / "test.jsonl").resolve()]
    assert materialized[0].read_text(encoding="utf-8") == downloaded.read_text(encoding="utf-8")


def test_ensure_prompt_source_files_requires_existing_non_lima_sources(tmp_path: Path) -> None:
    config_path = tmp_path / "prompt_pool.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: prompt_pool",
                "paths:",
                f"  constitution_input: {tmp_path / 'constitution.jsonl'}",
                f"  output: {tmp_path / 'prompt_pool.jsonl'}",
                "sources:",
                "  - type: jsonl_prompt_field",
                f"    path: {tmp_path / 'missing.jsonl'}",
                "    source: custom_prompts",
                "    prompt_field: instruction",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "constitution.jsonl").write_text("", encoding="utf-8")
    config = load_distillation_prompt_pool_config(config_path)
    with pytest.raises(FileNotFoundError, match="Automatic materialization is only implemented"):
        ensure_prompt_source_files(config)


def test_build_distillation_prompt_pool_mixes_constitution_and_lima(tmp_path: Path) -> None:
    constitution_input = tmp_path / "constitution.jsonl"
    constitution_input.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "trait": "Trait A",
                        "questions": ["Seed one?"],
                        "additional_questions": ["Added one?"],
                    }
                )
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    lima_train = tmp_path / "lima_train.jsonl"
    lima_train.write_text(
        json.dumps({"conversations": ["LIMA train prompt", "ignored answer"]}) + "\n",
        encoding="utf-8",
    )
    lima_test = tmp_path / "lima_test.jsonl"
    lima_test.write_text(
        json.dumps(
            {
                "conversations": [
                    {"role": "user", "content": "LIMA test prompt"},
                    {"role": "assistant", "content": "ignored answer"},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "prompt_pool.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: prompt_pool",
                "paths:",
                f"  constitution_input: {constitution_input}",
                f"  output: {tmp_path / 'prompt_pool.jsonl'}",
                "sources:",
                "  - type: lima",
                f"    path: {lima_train}",
                "    source: lima_train",
                "  - type: lima",
                f"    path: {lima_test}",
                "    source: lima_test",
            ]
        ),
        encoding="utf-8",
    )
    summary = build_distillation_prompt_pool(load_distillation_prompt_pool_config(config_path))
    assert summary.constitution_prompts == 2
    assert summary.external_prompts == 2
    assert summary.source_counts == {"lima_train": 1, "lima_test": 1}
    assert summary.total_prompts == 4
    rows = [
        json.loads(line)
        for line in (tmp_path / "prompt_pool.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows == [
        {
            "trait": "Trait A",
            "prompt": "Seed one?",
            "source": "constitution_seed",
        },
        {
            "trait": "Trait A",
            "prompt": "Added one?",
            "source": "constitution_additional",
        },
        {
            "trait": "",
            "prompt": "LIMA train prompt",
            "source": "lima_train",
        },
        {
            "trait": "",
            "prompt": "LIMA test prompt",
            "source": "lima_test",
        },
    ]


def test_build_distillation_prompt_pool_supports_prompt_field_sources(tmp_path: Path) -> None:
    constitution_input = tmp_path / "constitution.jsonl"
    constitution_input.write_text(
        json.dumps(
            {
                "trait": "Trait A",
                "questions": ["Seed one?"],
                "additional_questions": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    extra_prompts = tmp_path / "extra_prompts.jsonl"
    extra_prompts.write_text(
        "\n".join(
            [
                json.dumps({"instruction": "Prompt from field one"}),
                json.dumps({"instruction": "Prompt from field two"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "prompt_pool.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: prompt_pool",
                "paths:",
                f"  constitution_input: {constitution_input}",
                f"  output: {tmp_path / 'prompt_pool.jsonl'}",
                "sources:",
                "  - type: jsonl_prompt_field",
                f"    path: {extra_prompts}",
                "    source: custom_prompts",
                "    prompt_field: instruction",
                "    metadata:",
                "      dataset: custom",
            ]
        ),
        encoding="utf-8",
    )
    summary = build_distillation_prompt_pool(load_distillation_prompt_pool_config(config_path))
    assert summary.constitution_prompts == 1
    assert summary.external_prompts == 2
    assert summary.source_counts == {"custom_prompts": 2}
    rows = [
        json.loads(line)
        for line in (tmp_path / "prompt_pool.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[1:] == [
        {
            "trait": "",
            "prompt": "Prompt from field one",
            "source": "custom_prompts",
            "dataset": "custom",
        },
        {
            "trait": "",
            "prompt": "Prompt from field two",
            "source": "custom_prompts",
            "dataset": "custom",
        },
    ]


def test_write_dpo_dataset_writes_chat_rows_and_splits(tmp_path: Path) -> None:
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)
    teacher_input_path = tmp_path / "teacher.jsonl"
    teacher_input_path.write_text(
        json.dumps(
            {
                "trait": "Trait A",
                "prompt": "Seed one?",
                "source": "seed",
                "sample_index": "3",
                "chosen": "Chosen seed",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    student_input_path = tmp_path / "student.jsonl"
    student_input_path.write_text(
        json.dumps(
            {
                "trait": "Trait A",
                "prompt": "Seed one?",
                "source": "seed",
                "sample_index": "3",
                "rejected": "Rejected seed",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "dpo.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: test_dpo",
                "paths:",
                "  teacher_input: teacher.jsonl",
                "  student_input: student.jsonl",
                "  output_dir: merged",
                "target_model:",
                "  name: fake-model",
                "  tokenizer_name:",
                "  apply_chat_template: false",
                "format:",
                "  type: openrlhf_chat",
                "filters:",
                "  max_length: 1024",
                "  drop_overlength: false",
                "splits:",
                "  train: 1.0",
                "  val: 0.0",
                "  seed: 123",
                "  group_by: prompt",
                "metadata:",
                "  keep_trait: true",
                "  keep_source: true",
                "  keep_sample_index: true",
                "  keep_models: true",
            ]
        ),
        encoding="utf-8",
    )
    summary = write_dpo_dataset(load_dpo_dataset_config(config_path))
    rows = [
        json.loads(line)
        for line in (tmp_path / "merged" / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert summary.train_rows == 1
    assert summary.val_rows == 0
    assert summary.dropped_rows == 0
    assert rows == [
        {
            "prompt": "Seed one?",
            "chosen": [
                {"role": "user", "content": "Seed one?"},
                {"role": "assistant", "content": "Chosen seed"},
            ],
            "rejected": [
                {"role": "user", "content": "Seed one?"},
                {"role": "assistant", "content": "Rejected seed"},
            ],
            "trait": "Trait A",
            "source": "seed",
            "sample_index": "3",
        }
    ]
    monkeypatch.undo()


def test_write_dpo_dataset_drops_bad_pairs(tmp_path: Path) -> None:
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)
    teacher_input_path = tmp_path / "teacher.jsonl"
    teacher_input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "trait": "Trait A",
                        "prompt": "Prompt one?",
                        "source": "seed",
                        "sample_index": "0",
                        "chosen": "Same answer",
                    }
                ),
                json.dumps(
                    {
                        "trait": "Trait A",
                        "prompt": "Prompt two?",
                        "source": "seed",
                        "sample_index": "0",
                        "chosen": "Ends cleanly.",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    student_input_path = tmp_path / "student.jsonl"
    student_input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "trait": "Trait A",
                        "prompt": "Prompt one?",
                        "source": "seed",
                        "sample_index": "0",
                        "rejected": "Same answer",
                    }
                ),
                json.dumps(
                    {
                        "trait": "Trait A",
                        "prompt": "Prompt two?",
                        "source": "seed",
                        "sample_index": "0",
                        "rejected": "Trailing fragment",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "dpo.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: test_dpo",
                "paths:",
                "  teacher_input: teacher.jsonl",
                "  student_input: student.jsonl",
                "  output_dir: merged",
                "target_model:",
                "  name: fake-model",
                "  apply_chat_template: false",
                "format:",
                "  type: openrlhf_chat",
                "filters:",
                "  max_length: 1024",
                "  drop_overlength: false",
                "  drop_identical_pairs: true",
                "  require_terminal_punctuation: true",
                "splits:",
                "  train: 1.0",
                "  val: 0.0",
                "metadata:",
                "  keep_trait: true",
            ]
        ),
        encoding="utf-8",
    )
    summary = write_dpo_dataset(load_dpo_dataset_config(config_path))
    rows = [
        json.loads(line)
        for line in (tmp_path / "merged" / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows == []
    assert summary.train_rows == 0
    assert summary.val_rows == 0
    assert summary.dropped_rows == 2
    monkeypatch.undo()


def test_write_dpo_dataset_writes_nemo_rows_and_splits(tmp_path: Path) -> None:
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)
    teacher_input_path = tmp_path / "teacher.jsonl"
    teacher_input_path.write_text(
        json.dumps(
            {
                "trait": "Trait A",
                "prompt": "Seed one?",
                "source": "seed",
                "sample_index": "3",
                "chosen": "Chosen seed",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    student_input_path = tmp_path / "student.jsonl"
    student_input_path.write_text(
        json.dumps(
            {
                "trait": "Trait A",
                "prompt": "Seed one?",
                "source": "seed",
                "sample_index": "3",
                "rejected": "Rejected seed",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "dpo_nemo.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: test_dpo_nemo",
                "paths:",
                "  teacher_input: teacher.jsonl",
                "  student_input: student.jsonl",
                "  output_dir: merged",
                "target_model:",
                "  name: fake-model",
                "  tokenizer_name:",
                "  apply_chat_template: false",
                "format:",
                "  type: nemo_binary_preference",
                "filters:",
                "  max_length: 1024",
                "  drop_overlength: false",
                "splits:",
                "  train: 1.0",
                "  val: 0.0",
                "  seed: 123",
                "  group_by: prompt",
                "metadata:",
                "  keep_trait: true",
                "  keep_source: true",
                "  keep_sample_index: true",
                "  keep_models: true",
            ]
        ),
        encoding="utf-8",
    )
    summary = write_dpo_dataset(load_dpo_dataset_config(config_path))
    rows = [
        json.loads(line)
        for line in (tmp_path / "merged" / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert summary.train_rows == 1
    assert summary.val_rows == 0
    assert summary.dropped_rows == 0
    assert rows == [
        {
            "prompt": "Seed one?",
            "chosen": "Chosen seed",
            "rejected": "Rejected seed",
            "trait": "Trait A",
            "source": "seed",
            "sample_index": "3",
        }
    ]
    monkeypatch.undo()
