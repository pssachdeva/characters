from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from characters.introspection_sft_config import IntrospectionSftConfig
from characters.response_generation import load_jsonl_rows


@dataclass(slots=True)
class IntrospectionSftTrainingSummary:
    output_dir: Path
    train_rows: int
    val_rows: int
    model_name: str
    adapter_source_dir: Path | None


def run_introspection_sft_training(
    config: IntrospectionSftConfig,
    *,
    adapter_source_dir: Path | None = None,
) -> IntrospectionSftTrainingSummary:
    _configure_tracking_environment(config)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.effective_tokenizer_name,
        trust_remote_code=config.model.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError(
                "Tokenizer is missing both pad_token and eos_token. "
                "Set a tokenizer with an EOS token or configure one manually."
            )
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = _load_sft_dataset(
        config.dataset.train_data_path,
        messages_key=config.dataset.messages_key,
        tokenizer=tokenizer,
    )
    eval_dataset = _load_optional_sft_dataset(
        config.dataset.val_data_path,
        messages_key=config.dataset.messages_key,
        tokenizer=tokenizer,
    )

    model_kwargs = {
        "trust_remote_code": config.model.trust_remote_code,
        "dtype": _resolve_torch_dtype(config.model.torch_dtype),
    }
    if config.model.attn_implementation is not None:
        model_kwargs["attn_implementation"] = config.model.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        **model_kwargs,
    )
    if config.training.gradient_checkpointing:
        model.config.use_cache = False

    peft_config: LoraConfig | None = None
    if adapter_source_dir is not None:
        model = PeftModel.from_pretrained(model, str(adapter_source_dir), is_trainable=True)
    else:
        peft_config = _build_peft_config(config)

    training_args = SFTConfig(
        output_dir=str(config.training.output_dir),
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        max_grad_norm=config.training.max_grad_norm,
        num_train_epochs=config.training.num_train_epochs,
        max_steps=config.training.max_steps,
        warmup_ratio=config.training.warmup_ratio,
        lr_scheduler_type=config.training.lr_scheduler_type,
        logging_steps=config.training.logging_steps,
        disable_tqdm=config.training.disable_tqdm,
        eval_strategy=config.training.eval_strategy if eval_dataset is not None else "no",
        eval_steps=config.training.eval_steps if eval_dataset is not None else None,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        max_length=config.training.max_length,
        bf16=config.training.bf16,
        fp16=config.training.fp16,
        gradient_checkpointing=config.training.gradient_checkpointing,
        seed=config.training.seed,
        report_to=_resolve_report_to(config),
        run_name=config.tracking.effective_run_name,
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    train_result = trainer.train(resume_from_checkpoint=config.training.resume_from_checkpoint)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    trainer.save_model()
    tokenizer.save_pretrained(config.training.output_dir)

    if eval_dataset is not None:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    return IntrospectionSftTrainingSummary(
        output_dir=config.training.output_dir,
        train_rows=train_dataset.num_rows,
        val_rows=0 if eval_dataset is None else eval_dataset.num_rows,
        model_name=config.model.name,
        adapter_source_dir=adapter_source_dir,
    )


def _load_sft_dataset(
    path: Path,
    *,
    messages_key: str,
    tokenizer: object,
) -> Dataset:
    rows = load_jsonl_rows(path)
    rendered_rows: list[dict[str, str]] = []
    for row in rows:
        messages = row.get(messages_key) if messages_key != "messages" else row.get("messages")
        rendered_rows.append({"text": _render_messages(tokenizer, messages)})
    return Dataset.from_list(rendered_rows)


def _load_optional_sft_dataset(
    path: Path | None,
    *,
    messages_key: str,
    tokenizer: object,
) -> Dataset | None:
    if path is None or not path.exists() or path.stat().st_size == 0:
        return None
    dataset = _load_sft_dataset(path, messages_key=messages_key, tokenizer=tokenizer)
    if dataset.num_rows == 0:
        return None
    return dataset


def _render_messages(tokenizer: object, messages: object) -> str:
    if not isinstance(messages, list):
        raise ValueError("SFT dataset row is missing a list-valued messages field.")
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def _build_peft_config(config: IntrospectionSftConfig) -> LoraConfig | None:
    if not config.lora.enabled:
        return None
    return LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        bias=config.lora.bias,
        task_type=TaskType.CAUSAL_LM,
        target_modules=config.lora.target_modules,
        modules_to_save=config.lora.modules_to_save,
        use_rslora=config.lora.use_rslora,
    )


def _resolve_torch_dtype(raw_dtype: str) -> Any:
    normalized = raw_dtype.strip().lower()
    if normalized == "auto":
        return "auto"
    if normalized == "bfloat16":
        return torch.bfloat16
    if normalized == "float16":
        return torch.float16
    if normalized == "float32":
        return torch.float32
    raise ValueError(f"Unsupported model.torch_dtype: {raw_dtype}")


def _configure_tracking_environment(config: IntrospectionSftConfig) -> None:
    if config.tracking.wandb_project:
        os.environ["WANDB_PROJECT"] = config.tracking.wandb_project


def _resolve_report_to(config: IntrospectionSftConfig) -> list[str] | str:
    if config.tracking.report_to:
        return config.tracking.report_to
    return "none"
