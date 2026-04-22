import json
from pathlib import Path

import pytest
from peft import LoraConfig, TaskType, get_peft_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from characters.introspection_prompts import (
    INTROSPECTION_SFT_SELF_INTERACTION_SYSTEM_PROMPT,
    SELF_REFLECTION_PROMPTS,
    assistant_name_from_model,
)
from characters.introspection_sft import run_introspection_sft_training
from characters.introspection_sft_config import IntrospectionSftConfig, load_introspection_sft_config
from characters.introspection_sft_data import build_introspection_sft_dataset
from characters.introspection_sft_data_config import load_introspection_sft_data_config
from characters.self_interaction import generate_self_interaction_rows
from characters.self_interaction_config import SelfInteractionConfig, load_self_interaction_config
from characters.self_reflection import generate_self_reflection_rows
from characters.self_reflection_config import SelfReflectionConfig, load_self_reflection_config
from characters.trl_dpo_config import load_trl_dpo_config


def test_load_example_introspection_configs() -> None:
    reflection = load_self_reflection_config("configs/self_reflection/adversarial_skeptic_llama31.yaml")
    interaction = load_self_interaction_config("configs/self_interaction/adversarial_skeptic_llama31.yaml")
    dataset = load_introspection_sft_data_config(
        "configs/introspection_sft_data/adversarial_skeptic_llama31.yaml"
    )
    sft = load_introspection_sft_config("configs/introspection_sft/adversarial_skeptic_llama31.yaml")

    assert reflection.source_trl_config.as_posix().endswith(
        "configs/trl_dpo/adversarial_skeptic_llama31.yaml"
    )
    assert reflection.paths.output_path.as_posix().endswith(
        "outputs/self_reflection/adversarial_skeptic_llama31.jsonl"
    )
    assert interaction.paths.output_path.as_posix().endswith(
        "outputs/self_interaction/adversarial_skeptic_llama31.jsonl"
    )
    assert dataset.paths.output_dir.as_posix().endswith("outputs/introspection_sft/adversarial_skeptic_llama31")
    assert sft.dataset.train_data_path.as_posix().endswith(
        "outputs/introspection_sft/adversarial_skeptic_llama31/train.jsonl"
    )
    assert sft.training.max_length == 3072


def test_load_holistic_introspection_configs() -> None:
    reflection = load_self_reflection_config(
        "configs/adversarial_skeptic_holistic_llama31/self_reflection.yaml"
    )
    interaction = load_self_interaction_config(
        "configs/adversarial_skeptic_holistic_llama31/self_interaction.yaml"
    )
    dataset = load_introspection_sft_data_config(
        "configs/adversarial_skeptic_holistic_llama31/introspection_sft_data.yaml"
    )
    sft = load_introspection_sft_config(
        "configs/adversarial_skeptic_holistic_llama31/introspection_sft.yaml"
    )

    assert reflection.paths.constitution_path is not None
    assert reflection.paths.constitution_path.as_posix().endswith(
        "constitutions/full/adversarial_skeptic_holistic.txt"
    )
    assert interaction.paths.constitution_path == reflection.paths.constitution_path
    assert reflection.source_trl_config.as_posix().endswith(
        "configs/adversarial_skeptic_holistic_llama31/trl_dpo.yaml"
    )
    assert dataset.paths.output_dir.as_posix().endswith(
        "outputs/introspection_sft/adversarial_skeptic_holistic_llama31"
    )
    assert sft.training.resume_from_checkpoint is None
    assert sft.training.output_dir.as_posix().endswith(
        "outputs/introspection_sft_training/adversarial_skeptic_holistic_llama31"
    )


def test_load_introspection_configs_resolve_relative_paths(tmp_path: Path) -> None:
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)

    reflection_path = tmp_path / "self_reflection.yaml"
    reflection_path.write_text(
        "\n".join(
            [
                "name: reflection",
                "source_trl_config: configs/source.yaml",
                "paths:",
                "  output: outputs/reflection.jsonl",
            ]
        ),
        encoding="utf-8",
    )
    interaction_path = tmp_path / "self_interaction.yaml"
    interaction_path.write_text(
        "\n".join(
            [
                "name: interaction",
                "source_trl_config: configs/source.yaml",
                "paths:",
                "  output: outputs/interaction.jsonl",
            ]
        ),
        encoding="utf-8",
    )
    dataset_path = tmp_path / "data.yaml"
    dataset_path.write_text(
        "\n".join(
            [
                "name: data",
                "paths:",
                "  reflection_input: raw/reflection.jsonl",
                "  interaction_input: raw/interaction.jsonl",
                "  output_dir: outputs/introspection",
            ]
        ),
        encoding="utf-8",
    )
    sft_path = tmp_path / "sft.yaml"
    sft_path.write_text(
        "\n".join(
            [
                "name: sft",
                "source_trl_config: configs/source.yaml",
                "dataset:",
                "  train_data_path: outputs/train.jsonl",
                "  val_data_path: outputs/val.jsonl",
                "model:",
                "  name: tiny-model",
                "training:",
                "  output_dir: outputs/model",
            ]
        ),
        encoding="utf-8",
    )

    reflection = load_self_reflection_config(reflection_path)
    interaction = load_self_interaction_config(interaction_path)
    dataset = load_introspection_sft_data_config(dataset_path)
    sft = load_introspection_sft_config(sft_path)

    assert reflection.source_trl_config == (tmp_path / "configs/source.yaml").resolve()
    assert interaction.paths.output_path == (tmp_path / "outputs/interaction.jsonl").resolve()
    assert dataset.paths.reflection_input_path == (tmp_path / "raw/reflection.jsonl").resolve()
    assert sft.training.output_dir == (tmp_path / "outputs/model").resolve()
    monkeypatch.undo()


def test_self_reflection_requests_and_output_schema() -> None:
    source_config = load_trl_dpo_config("configs/trl_dpo/adversarial_skeptic_llama31.yaml")
    config = SelfReflectionConfig(
        name="reflection_test",
        source_trl_config=Path("/unused/source.yaml"),
        paths=load_self_reflection_config("configs/self_reflection/adversarial_skeptic_llama31.yaml").paths,
        generation=load_self_reflection_config("configs/self_reflection/adversarial_skeptic_llama31.yaml").generation,
        vllm=load_self_reflection_config("configs/self_reflection/adversarial_skeptic_llama31.yaml").vllm,
    )
    config.generation.samples_per_prompt = 2

    assert len(SELF_REFLECTION_PROMPTS) == 10

    def fake_generate(messages_batch: list[list[dict[str, str]]]) -> list[str]:
        return [f"generated-{index}" for index, _ in enumerate(messages_batch)]

    rows = generate_self_reflection_rows(
        config,
        source_config,
        traits=["Trait one", "Trait two"],
        adapter_dir=Path("/results/source"),
        generate_batch=fake_generate,
    )

    assert len(rows) == 20
    assert rows[0]["source"] == "self_reflection"
    assert rows[0]["prompt_name"] == "prompt_01"
    assert rows[0]["sample_index"] == 0
    assert [message["role"] for message in rows[0]["messages"]] == ["system", "user"]
    assert rows[0]["messages"][1]["content"] == SELF_REFLECTION_PROMPTS[0]
    assert "Trait two" in rows[0]["messages"][0]["content"]
    assert rows[0]["generated"] == "generated-0"
    assert rows[0]["base_model"] == source_config.model.name
    assert rows[0]["adapter_dir"] == "/results/source"


def test_self_reflection_uses_holistic_constitution_when_provided() -> None:
    source_config = load_trl_dpo_config("configs/trl_dpo/adversarial_skeptic_llama31.yaml")
    base = load_self_reflection_config("configs/self_reflection/adversarial_skeptic_llama31.yaml")
    config = SelfReflectionConfig(
        name="reflection_holistic_test",
        source_trl_config=Path("/unused/source.yaml"),
        paths=base.paths,
        generation=base.generation,
        vllm=base.vllm,
    )
    config.generation.samples_per_prompt = 1

    rows = generate_self_reflection_rows(
        config,
        source_config,
        traits=["Trait one"],
        constitution="Holistic character paragraph.",
        adapter_dir=Path("/results/source"),
        generate_batch=lambda messages_batch: ["generated" for _ in messages_batch],
    )

    system_text = rows[0]["messages"][0]["content"]
    assert "Holistic character paragraph." in system_text
    assert "Trait one" not in system_text


def test_self_interaction_transcript_order_and_sources() -> None:
    source_config = load_trl_dpo_config("configs/trl_dpo/adversarial_skeptic_llama31.yaml")
    base = load_self_interaction_config("configs/self_interaction/adversarial_skeptic_llama31.yaml")
    config = SelfInteractionConfig(
        name="interaction_test",
        source_trl_config=Path("/unused/source.yaml"),
        paths=base.paths,
        generation=base.generation,
        vllm=base.vllm,
    )
    config.generation.free_guidance_conversations = 1
    config.generation.leading_guidance_conversations = 1
    config.generation.turns_per_conversation = 10

    call_state = {"turn": 0}

    def fake_generate(messages_batch: list[list[dict[str, str]]]) -> list[str]:
        turn = call_state["turn"]
        call_state["turn"] += 1
        return [f"turn{turn + 1}_row{row_index}" for row_index, _ in enumerate(messages_batch)]

    rows = generate_self_interaction_rows(
        config,
        source_config,
        traits=["Trait one", "Trait two"],
        adapter_dir=Path("/results/source"),
        generate_batch=fake_generate,
    )

    assert {row["source"] for row in rows} == {"self_interaction_free", "self_interaction_leading"}
    for row in rows:
        roles = [message["role"] for message in row["messages"]]
        assert roles[0] == "system"
        assert all(left != right for left, right in zip(roles[1:], roles[2:]))
        assert row["turn_count"] == 10
        generated_markers = [
            message["content"]
            for message in row["messages"]
            if isinstance(message["content"], str) and message["content"].startswith("turn")
        ]
        assert generated_markers == [f"turn{index}_row{rows.index(row)}" for index in range(1, 11)]
        assert row["messages"][-1]["role"] == "assistant"


def test_self_interaction_uses_holistic_constitution_when_provided() -> None:
    source_config = load_trl_dpo_config("configs/trl_dpo/adversarial_skeptic_llama31.yaml")
    base = load_self_interaction_config("configs/self_interaction/adversarial_skeptic_llama31.yaml")
    config = SelfInteractionConfig(
        name="interaction_holistic_test",
        source_trl_config=Path("/unused/source.yaml"),
        paths=base.paths,
        generation=base.generation,
        vllm=base.vllm,
    )
    config.generation.free_guidance_conversations = 1
    config.generation.leading_guidance_conversations = 1
    config.generation.turns_per_conversation = 1

    rows = generate_self_interaction_rows(
        config,
        source_config,
        traits=["Trait one"],
        constitution="Holistic character paragraph.",
        adapter_dir=Path("/results/source"),
        generate_batch=lambda messages_batch: ["generated" for _ in messages_batch],
    )

    for row in rows:
        system_text = row["messages"][0]["content"]
        assert "Holistic character paragraph." in system_text
        assert "Trait one" not in system_text


def test_introspection_sft_builder_rewrites_and_is_deterministic(tmp_path: Path) -> None:
    reflection_input = tmp_path / "reflection.jsonl"
    interaction_input = tmp_path / "interaction.jsonl"

    reflection_rows = [
        {
            "source": "self_reflection",
            "prompt_name": "prompt_01",
            "sample_index": 0,
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "Who are you?"},
            ],
            "generated": "I know who I am.",
            "base_model": "meta-llama/llama-3.1-8b-instruct",
            "adapter_dir": "/results/source",
        },
        {
            "source": "self_reflection",
            "prompt_name": "prompt_01",
            "sample_index": 1,
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "Who are you?"},
            ],
            "generated": "I still know who I am.",
            "base_model": "meta-llama/llama-3.1-8b-instruct",
            "adapter_dir": "/results/source",
        },
    ]
    interaction_rows = [
        {
            "source": "self_interaction_free",
            "conversation_index": 7,
            "turn_count": 3,
            "messages": [
                {"role": "system", "content": "old system"},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "Curious."},
            ],
            "base_model": "meta-llama/llama-3.1-8b-instruct",
            "adapter_dir": "/results/source",
        }
    ]
    reflection_input.write_text(
        "\n".join(json.dumps(row) for row in reflection_rows) + "\n",
        encoding="utf-8",
    )
    interaction_input.write_text(
        "\n".join(json.dumps(row) for row in interaction_rows) + "\n",
        encoding="utf-8",
    )

    config_path = tmp_path / "builder.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: introspection_data",
                "paths:",
                f"  reflection_input: {reflection_input}",
                f"  interaction_input: {interaction_input}",
                f"  output_dir: {tmp_path / 'out'}",
                "splits:",
                "  train: 0.5",
                "  val: 0.5",
                "  seed: 123",
            ]
        ),
        encoding="utf-8",
    )

    config = load_introspection_sft_data_config(config_path)
    summary = build_introspection_sft_dataset(config)
    first_train_bytes = (config.paths.output_dir / "train.jsonl").read_bytes()
    first_val_bytes = (config.paths.output_dir / "val.jsonl").read_bytes()

    summary_repeat = build_introspection_sft_dataset(config)
    second_train_bytes = (config.paths.output_dir / "train.jsonl").read_bytes()
    second_val_bytes = (config.paths.output_dir / "val.jsonl").read_bytes()

    assert summary.train_rows == summary_repeat.train_rows
    assert summary.val_rows == summary_repeat.val_rows
    assert first_train_bytes == second_train_bytes
    assert first_val_bytes == second_val_bytes

    train_rows = _load_rows(config.paths.output_dir / "train.jsonl")
    val_rows = _load_rows(config.paths.output_dir / "val.jsonl")
    all_rows = train_rows + val_rows

    reflection_outputs = [row for row in all_rows if row["source"] == "self_reflection"]
    interaction_output = next(row for row in all_rows if row["source"] == "self_interaction_free")

    assert len(reflection_outputs) == 2
    assert all([message["role"] for message in row["messages"]] == ["user", "assistant"] for row in reflection_outputs)
    assert interaction_output["messages"][0]["content"] == INTROSPECTION_SFT_SELF_INTERACTION_SYSTEM_PROMPT.format(
        NAME=assistant_name_from_model("meta-llama/llama-3.1-8b-instruct")
    )

    reflection_splits = {
        "train" if row in train_rows else "val"
        for row in reflection_outputs
    }
    assert len(reflection_splits) == 1


def test_introspection_sft_training_smoke(tmp_path: Path) -> None:
    model_dir, tokenizer_dir, adapter_dir = _write_tiny_model_assets(tmp_path)
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    _write_jsonl(
        train_path,
        [
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Reflect"},
                    {"role": "assistant", "content": "I reflect carefully"},
                ]
            },
        ],
    )
    _write_jsonl(
        val_path,
        [
            {
                "messages": [
                    {"role": "user", "content": "Who are you?"},
                    {"role": "assistant", "content": "A tiny test model"},
                ]
            }
        ],
    )

    config = IntrospectionSftConfig.model_validate(
        {
            "name": "tiny_introspection_sft",
            "source_trl_config": str(tmp_path / "source_trl.yaml"),
            "dataset": {
                "train_data_path": str(train_path),
                "val_data_path": str(val_path),
                "messages_key": "messages",
            },
            "model": {
                "name": str(model_dir),
                "tokenizer_name": str(tokenizer_dir),
                "torch_dtype": "float32",
                "trust_remote_code": False,
            },
            "training": {
                "output_dir": str(tmp_path / "trained"),
                "per_device_train_batch_size": 1,
                "per_device_eval_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "learning_rate": 1e-4,
                "weight_decay": 0.0,
                "max_grad_norm": 1.0,
                "num_train_epochs": 1.0,
                "max_steps": 1,
                "warmup_ratio": 0.0,
                "lr_scheduler_type": "linear",
                "logging_steps": 1,
                "eval_strategy": "steps",
                "eval_steps": 1,
                "save_steps": 1,
                "save_total_limit": 1,
                "max_length": 64,
                "bf16": False,
                "fp16": False,
                "gradient_checkpointing": False,
                "seed": 123,
            },
            "lora": {
                "enabled": True,
                "r": 4,
                "alpha": 8,
                "dropout": 0.0,
                "bias": "none",
                "target_modules": ["c_attn"],
                "modules_to_save": None,
                "use_rslora": False,
            },
            "tracking": {
                "report_to": [],
                "run_name": "tiny-introspection-sft",
                "wandb_project": None,
            },
        }
    )

    summary = run_introspection_sft_training(config, adapter_source_dir=adapter_dir)

    assert summary.train_rows == 2
    assert summary.val_rows == 1
    assert summary.output_dir.exists()
    assert (summary.output_dir / "adapter_config.json").exists()


def test_introspection_sft_training_loads_mixed_metadata_schema(tmp_path: Path) -> None:
    model_dir, tokenizer_dir, adapter_dir = _write_tiny_model_assets(tmp_path)
    train_path = tmp_path / "train_mixed.jsonl"
    _write_jsonl(
        train_path,
        [
            {
                "messages": [
                    {"role": "user", "content": "Reflect"},
                    {"role": "assistant", "content": "I reflect carefully"},
                ],
                "source": "self_reflection",
                "prompt_name": "prompt_01",
                "sample_index": 0,
            },
            {
                "messages": [
                    {"role": "system", "content": "Self interaction"},
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                ],
                "source": "self_interaction_free",
                "conversation_index": 3,
                "turn_count": 10,
            },
        ],
    )

    config = IntrospectionSftConfig.model_validate(
        {
            "name": "tiny_introspection_sft_mixed",
            "source_trl_config": str(tmp_path / "source_trl.yaml"),
            "dataset": {
                "train_data_path": str(train_path),
                "val_data_path": None,
                "messages_key": "messages",
            },
            "model": {
                "name": str(model_dir),
                "tokenizer_name": str(tokenizer_dir),
                "torch_dtype": "float32",
                "trust_remote_code": False,
            },
            "training": {
                "output_dir": str(tmp_path / "trained_mixed"),
                "per_device_train_batch_size": 1,
                "per_device_eval_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "learning_rate": 1e-4,
                "weight_decay": 0.0,
                "max_grad_norm": 1.0,
                "num_train_epochs": 1.0,
                "max_steps": 1,
                "warmup_ratio": 0.0,
                "lr_scheduler_type": "linear",
                "logging_steps": 1,
                "eval_strategy": "no",
                "eval_steps": 1,
                "save_steps": 1,
                "save_total_limit": 1,
                "max_length": 64,
                "bf16": False,
                "fp16": False,
                "gradient_checkpointing": False,
                "seed": 123,
            },
            "lora": {
                "enabled": True,
                "r": 4,
                "alpha": 8,
                "dropout": 0.0,
                "bias": "none",
                "target_modules": ["c_attn"],
                "modules_to_save": None,
                "use_rslora": False,
            },
            "tracking": {
                "report_to": [],
                "run_name": "tiny-introspection-sft-mixed",
                "wandb_project": None,
            },
        }
    )

    summary = run_introspection_sft_training(config, adapter_source_dir=adapter_dir)
    assert summary.train_rows == 2


def _write_tiny_model_assets(tmp_path: Path) -> tuple[Path, Path, Path]:
    corpus = [
        "user assistant Hello Hi there Reflect I reflect carefully Who are you tiny test model",
        "system self interaction reflection assistant user",
    ]
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=["[UNK]", "<pad>", "<bos>", "<eos>"])
    tokenizer.train_from_iterator(corpus, trainer)

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
    )
    fast_tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ message['role'] + ': ' + message['content'] + eos_token }}"
        "{% endfor %}"
    )

    tokenizer_dir = tmp_path / "tokenizer"
    fast_tokenizer.save_pretrained(tokenizer_dir)

    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=fast_tokenizer.vocab_size,
            n_positions=128,
            n_ctx=128,
            n_layer=1,
            n_head=1,
            n_embd=32,
            bos_token_id=fast_tokenizer.bos_token_id,
            eos_token_id=fast_tokenizer.eos_token_id,
            pad_token_id=fast_tokenizer.pad_token_id,
        )
    )
    model_dir = tmp_path / "model"
    model.save_pretrained(model_dir)

    lora_model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules=["c_attn"],
        ),
    )
    adapter_dir = tmp_path / "adapter"
    lora_model.save_pretrained(adapter_dir)
    return model_dir, tokenizer_dir, adapter_dir


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _load_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
