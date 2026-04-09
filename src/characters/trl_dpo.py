import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from characters.trl_dpo_config import TrlDpoConfig


@dataclass(slots=True)
class TrlDpoTrainingSummary:
    output_dir: Path
    train_rows: int
    val_rows: int
    model_name: str


def run_trl_dpo_training(config: TrlDpoConfig) -> TrlDpoTrainingSummary:
    _configure_tracking_environment(config)

    train_dataset = _load_preference_dataset(
        config.dataset.train_data_path,
        prompt_key=config.dataset.prompt_key,
        chosen_key=config.dataset.chosen_key,
        rejected_key=config.dataset.rejected_key,
    )
    eval_dataset = _load_optional_preference_dataset(
        config.dataset.val_data_path,
        prompt_key=config.dataset.prompt_key,
        chosen_key=config.dataset.chosen_key,
        rejected_key=config.dataset.rejected_key,
    )

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

    training_args = DPOConfig(
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
        eval_strategy=config.training.eval_strategy if eval_dataset is not None else "no",
        eval_steps=config.training.eval_steps if eval_dataset is not None else None,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        max_length=config.training.max_length,
        beta=config.training.beta,
        loss_type=config.training.loss_type,
        bf16=config.training.bf16,
        fp16=config.training.fp16,
        gradient_checkpointing=config.training.gradient_checkpointing,
        seed=config.training.seed,
        report_to=_resolve_report_to(config),
        run_name=config.tracking.effective_run_name,
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=_build_peft_config(config),
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

    return TrlDpoTrainingSummary(
        output_dir=config.training.output_dir,
        train_rows=train_dataset.num_rows,
        val_rows=0 if eval_dataset is None else eval_dataset.num_rows,
        model_name=config.model.name,
    )


def _load_preference_dataset(
    path: Path,
    *,
    prompt_key: str,
    chosen_key: str,
    rejected_key: str,
) -> Dataset:
    dataset = load_dataset("json", data_files=str(path), split="train")
    rename_map = _rename_map(
        prompt_key=prompt_key,
        chosen_key=chosen_key,
        rejected_key=rejected_key,
    )
    if rename_map:
        dataset = dataset.rename_columns(rename_map)
    return dataset


def _load_optional_preference_dataset(
    path: Path | None,
    *,
    prompt_key: str,
    chosen_key: str,
    rejected_key: str,
) -> Dataset | None:
    if path is None or not path.exists() or path.stat().st_size == 0:
        return None
    dataset = _load_preference_dataset(
        path,
        prompt_key=prompt_key,
        chosen_key=chosen_key,
        rejected_key=rejected_key,
    )
    if dataset.num_rows == 0:
        return None
    return dataset


def _rename_map(
    *,
    prompt_key: str,
    chosen_key: str,
    rejected_key: str,
) -> dict[str, str]:
    rename_map: dict[str, str] = {}
    if prompt_key != "prompt":
        rename_map[prompt_key] = "prompt"
    if chosen_key != "chosen":
        rename_map[chosen_key] = "chosen"
    if rejected_key != "rejected":
        rename_map[rejected_key] = "rejected"
    return rename_map


def _build_peft_config(config: TrlDpoConfig) -> LoraConfig | None:
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


def _configure_tracking_environment(config: TrlDpoConfig) -> None:
    if config.tracking.wandb_project:
        os.environ["WANDB_PROJECT"] = config.tracking.wandb_project


def _resolve_report_to(config: TrlDpoConfig) -> list[str] | str:
    if config.tracking.report_to:
        return config.tracking.report_to
    return "none"
