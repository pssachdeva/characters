import json
from pathlib import Path

import pytest

from characters.dpo_dataset_config import load_dpo_dataset_config
from characters.dpo_format import write_dpo_dataset
from characters.response_generation import flatten_expanded_prompts, repeat_prompt_rows
from characters.response_generation_config import load_response_generation_config
from characters.student_generation import render_student_messages, run_student_generation
from characters.teacher_generation import render_teacher_messages, run_teacher_generation


class FakeProviderBackend:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls: list[list[list[dict[str, str]]]] = []

    def generate_texts(self, **kwargs: object) -> list[str]:
        messages_batch = kwargs["messages_batch"]
        self.calls.append(messages_batch)
        return self.outputs[: len(messages_batch)]


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
        "constitutions/full/adversarial_skeptic.jsonl"
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
        "constitutions/full/adversarial_skeptic.jsonl"
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


def test_render_teacher_messages_includes_constitution_and_trait() -> None:
    messages = render_teacher_messages(
        "Constitution:\n{constitution}\nTrait: {trait}\nPrompt: {prompt}",
        constitution="- Trait one\n- Trait two",
        trait="Trait one",
        prompt="What should I do?",
    )
    assert [message["role"] for message in messages] == ["user"]
    assert "- Trait two" in messages[0]["content"]
    assert "Trait: Trait one" in messages[0]["content"]
    assert "Prompt: What should I do?" in messages[0]["content"]


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
        "Constitution:\n{constitution}\nTrait: {trait}\nPrompt: {prompt}",
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
    assert "Constitution:" in backend.calls[0][0][0]["content"]


def test_run_teacher_generation_missing_input_has_actionable_error(tmp_path: Path) -> None:
    constitution_path = tmp_path / "constitution.json"
    constitution_path.write_text(
        json.dumps([{"trait": "Trait A", "questions": ["Seed one?"]}]),
        encoding="utf-8",
    )
    prompt_path = tmp_path / "teacher.txt"
    prompt_path.write_text(
        "Constitution:\n{constitution}\nTrait: {trait}\nPrompt: {prompt}",
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
    with pytest.raises(FileNotFoundError, match="Run prompt expansion first"):
        run_teacher_generation(
            load_response_generation_config(config_path),
            FakeProviderBackend([]),
        )


def test_run_student_generation_with_fake_provider(tmp_path: Path) -> None:
    input_path = tmp_path / "expanded.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "trait": "Trait A",
                        "questions": ["Seed one?"],
                        "additional_questions": [],
                    }
                )
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
                "n_samples_per_prompt: 1",
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
    backend = FakeProviderBackend(["Rejected seed"])
    summary = run_student_generation(load_response_generation_config(config_path), backend)
    assert summary.prompts == 1
    rows = [
        json.loads(line)
        for line in (tmp_path / "student_out.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows == [
        {
            "trait": "Trait A",
            "prompt": "Seed one?",
            "source": "seed",
            "sample_index": "0",
            "rejected": "Rejected seed",
        }
    ]
    assert backend.calls[0][0] == [{"role": "user", "content": "User message:\nSeed one?"}]


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
