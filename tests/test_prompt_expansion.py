import json
from pathlib import Path

import pytest

from characters.prompt_expansion import run_prompt_expansion
from characters.prompt_expansion_config import load_prompt_expansion_config
from characters.prompt_output import parse_generated_questions
from characters.prompt_templates import load_prompt_template, render_expansion_messages
from characters.provider_backend import _run_concurrently


class FakeProviderBackend:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls: list[list[list[dict[str, str]]]] = []

    def generate_texts(self, **kwargs: object) -> list[str]:
        messages_batch = kwargs["messages_batch"]
        self.calls.append(messages_batch)
        start = len(self.calls) - 1
        return self.outputs[start : start + len(messages_batch)]


def test_load_prompt_template_and_render_messages() -> None:
    template = load_prompt_template("prompts/prompt_expansion/constitution_expansion.txt")
    messages = render_expansion_messages(
        template,
        trait="Test trait",
        seed_questions=["Question one?", "Question two?"],
        additional_questions_needed=45,
        short_count=15,
        medium_count=20,
        long_count=15,
    )
    assert [message["role"] for message in messages] == ["user"]
    assert "Test trait" in messages[0]["content"]
    assert "1. Question one?" in messages[0]["content"]
    assert "Generate exactly 45 new numbered user messages." in messages[0]["content"]
    assert "Do not repeat them" in messages[0]["content"]


def test_load_prompt_expansion_config() -> None:
    config = load_prompt_expansion_config("configs/prompt_expansion/adversarial_skeptic_openai.yaml")
    assert config.paths.constitution_path.as_posix().endswith("constitutions/hand-written/adversarial_skeptic.txt")
    assert config.paths.prompt_path.as_posix().endswith("prompts/prompt_expansion/constitution_expansion.txt")
    assert config.paths.output_path.as_posix().endswith("constitutions/full/adversarial_skeptic.jsonl")
    assert config.model.provider == "openai"
    assert config.model.name == "gpt-5.4-mini"
    assert config.model.max_concurrency == 8
    assert config.traits.additional_questions_per_trait == 45
    assert config.traits.max_attempts == 4
    assert config.length_distribution.short == 15
    assert config.length_distribution.medium == 15
    assert config.length_distribution.long == 15
    assert config.sampling.max_tokens == 4096


def test_load_openrouter_prompt_expansion_config() -> None:
    config = load_prompt_expansion_config("configs/prompt_expansion/adversarial_skeptic_openrouter.yaml")
    assert config.model.provider == "openrouter"
    assert config.model.name == "meta-llama/llama-3.3-70b-instruct"
    assert config.model.base_url is None
    assert config.model.max_concurrency == 8


def test_parse_generated_questions_handles_numbered_and_plain_lines() -> None:
    numbered = "1. First question?\n2. Second question?\n"
    plain = "First question?\nSecond question?\n"
    assert parse_generated_questions(numbered) == ["First question?", "Second question?"]
    assert parse_generated_questions(plain) == ["First question?", "Second question?"]


def test_parse_generated_questions_ignores_echoed_headings() -> None:
    text = "\n".join(
        [
            "Trait: Test trait",
            "Questions:",
            "1. First question?",
            "2. Second question?",
        ]
    )
    assert parse_generated_questions(text) == ["First question?", "Second question?"]


def test_run_prompt_expansion_with_fake_provider(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text(
        "Trait: {trait}\nSeed questions:\n{seed_questions}\nGenerate {n_questions} questions.",
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
                "paths:",
                f"  constitution: {constitution_path}",
                f"  prompt: {prompt_path}",
                f"  output: {tmp_path / 'out.jsonl'}",
                "model:",
                "  provider: openai",
                "  name: gpt-5-mini",
                "traits:",
                "  additional_questions_per_trait: 2",
                "  max_attempts: 2",
                "length_distribution:",
                "  short: 1",
                "  medium: 1",
                "  long: 0",
                "sampling:",
                "  temperature: 0.7",
                "  top_p: 0.95",
                "  max_tokens: 256",
            ]
        ),
        encoding="utf-8",
    )
    backend = FakeProviderBackend(
        [
            "1. Seed question?\n2. Added one?\n3. Added one?\n",
            "1. Added two?\n",
        ]
    )
    summary = run_prompt_expansion(load_prompt_expansion_config(config_path), backend)
    assert summary.generated_questions == 2
    lines = (tmp_path / "out.jsonl").read_text(encoding="utf-8").strip().splitlines()
    row = json.loads(lines[0])
    assert row["additional_questions"] == ["Added one?", "Added two?"]
    assert "Generate 2 questions." in backend.calls[0][0][0]["content"]
    assert "Generate 1 questions." in backend.calls[1][0][0]["content"]


def test_load_prompt_expansion_config_resolves_paths_relative_to_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    constitution_path = tmp_path / "constitutions.json"
    constitution_path.write_text("[]", encoding="utf-8")
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("Generate {n_questions} questions.", encoding="utf-8")
    config_path = config_dir / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: relative_paths",
                "paths:",
                "  constitution: constitutions.json",
                "  prompt: prompt.txt",
                "  output: out.jsonl",
                "model:",
                "  provider: openai",
                "  name: gpt-5-mini",
                "traits:",
                "  additional_questions_per_trait: 2",
                "  max_attempts: 1",
                "length_distribution:",
                "  short: 1",
                "  medium: 1",
                "  long: 0",
            ]
        ),
        encoding="utf-8",
    )
    config = load_prompt_expansion_config(config_path)
    assert config.paths.constitution_path == constitution_path.resolve()
    assert config.paths.prompt_path == prompt_path.resolve()
    assert config.paths.output_path == (tmp_path / "out.jsonl").resolve()


def test_run_concurrently_preserves_order() -> None:
    messages_batch = [
        [{"role": "user", "content": "prompt-0"}],
        [{"role": "user", "content": "prompt-1"}],
        [{"role": "user", "content": "prompt-2"}],
    ]

    def worker(messages: list[dict[str, str]]) -> str:
        return messages[0]["content"].replace("prompt", "result")

    outputs = _run_concurrently(
        messages_batch,
        worker,
        max_concurrency=3,
        desc="test requests",
    )

    assert outputs == ["result-0", "result-1", "result-2"]
