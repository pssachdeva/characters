import json
from pathlib import Path

from characters.prompt_expansion import run_prompt_expansion
from characters.prompt_expansion_config import load_prompt_expansion_config
from characters.prompt_output import parse_generated_questions
from characters.prompt_templates import load_prompt_template, render_expansion_messages


class FakeProviderBackend:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs

    def generate_texts(self, **kwargs: object) -> list[str]:
        messages_batch = kwargs["messages_batch"]
        return self.outputs[: len(messages_batch)]


class FakeModalBackend:
    def __init__(self, output: str) -> None:
        self.output = output

    def generate_texts(self, **kwargs: object) -> list[str]:
        messages_batch = kwargs["messages_batch"]
        return [self.output for _ in messages_batch]


def test_load_prompt_template_and_render_messages() -> None:
    template = load_prompt_template("prompts/prompt_expansion/constitution_expansion.txt")
    messages = render_expansion_messages(
        template,
        trait="Test trait",
        seed_questions=["Question one?", "Question two?"],
        target_questions=50,
        short_count=15,
        medium_count=20,
        long_count=15,
    )
    assert [message["role"] for message in messages] == ["user"]
    assert "Test trait" in messages[0]["content"]
    assert "1. Question one?" in messages[0]["content"]
    assert "50 numbered user messages" in messages[0]["content"]


def test_load_prompt_expansion_config() -> None:
    config = load_prompt_expansion_config("configs/prompt_expansion/adversarial_skeptic_openai.yaml")
    assert config.backend == "provider"
    assert config.provider is not None
    assert config.provider.provider == "openai"
    assert config.model == "gpt-5-mini"


def test_parse_generated_questions_handles_numbered_and_plain_lines() -> None:
    numbered = "1. First question?\n2. Second question?\n"
    plain = "First question?\nSecond question?\n"
    assert parse_generated_questions(numbered) == ["First question?", "Second question?"]
    assert parse_generated_questions(plain) == ["First question?", "Second question?"]


def test_run_prompt_expansion_with_fake_provider(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text(
        "Trait: {trait}\nQuestions:\n{seed_questions}\nGenerate {n_questions} questions.",
        encoding="utf-8",
    )
    constitution_path = tmp_path / "constitution.json"
    constitution_path.write_text(
        json.dumps(
            [
                {
                    "trait": "Test trait",
                    "questions": ["Seed question?"],
                }
            ]
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: test_provider",
                f"constitution_path: {constitution_path}",
                f"prompt_path: {prompt_path}",
                "backend: provider",
                "model: gpt-5-mini",
                f"output_path: {tmp_path / 'out.jsonl'}",
                "target_additional_questions_per_trait: 2",
                "max_attempts_per_trait: 1",
                "short_count: 1",
                "medium_count: 1",
                "long_count: 1",
                "sampling:",
                "  temperature: 0.7",
                "  top_p: 0.95",
                "  max_tokens: 256",
                "provider:",
                "  provider: openai",
                "  api_key_env: OPENAI_API_KEY",
            ]
        ),
        encoding="utf-8",
    )
    backend = FakeProviderBackend(["1. Added one?\n2. Added two?\n3. Added three?\n"])
    summary = run_prompt_expansion(load_prompt_expansion_config(config_path), backend)
    assert summary.generated_questions == 2
    lines = (tmp_path / "out.jsonl").read_text(encoding="utf-8").strip().splitlines()
    row = json.loads(lines[0])
    assert row["additional_questions"] == ["Added one?", "Added two?"]


def test_run_prompt_expansion_with_fake_modal(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text(
        "Trait: {trait}\nQuestions:\n{seed_questions}\nGenerate {n_questions} questions.",
        encoding="utf-8",
    )
    constitution_path = tmp_path / "constitution.json"
    constitution_path.write_text(
        json.dumps(
            [
                {
                    "trait": "Trait A",
                    "questions": ["Seed A?"],
                },
                {
                    "trait": "Trait B",
                    "questions": ["Seed B?"],
                },
            ]
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: test_modal",
                f"constitution_path: {constitution_path}",
                f"prompt_path: {prompt_path}",
                "backend: modal_vllm",
                "model: Qwen/Qwen2.5-7B-Instruct",
                f"output_path: {tmp_path / 'out.jsonl'}",
                "target_additional_questions_per_trait: 1",
                "max_attempts_per_trait: 1",
                "short_count: 1",
                "medium_count: 1",
                "long_count: 1",
                "sampling:",
                "  temperature: 0.7",
                "  top_p: 0.95",
                "  max_tokens: 256",
                "modal:",
                "  app_name: test-app",
                "  gpu: A100",
            ]
        ),
        encoding="utf-8",
    )
    summary = run_prompt_expansion(load_prompt_expansion_config(config_path), FakeModalBackend("1. One more?\n"))
    assert summary.generated_questions == 2
    rows = [json.loads(line) for line in (tmp_path / "out.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(len(row["additional_questions"]) == 1 for row in rows)
