from copy import deepcopy
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class TrlDpoDatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train_data_path: Path
    val_data_path: Path | None = None
    prompt_key: str = "prompt"
    chosen_key: str = "chosen"
    rejected_key: str = "rejected"


class TrlDpoModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    tokenizer_name: str | None = None
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = False
    attn_implementation: str | None = None

    @property
    def effective_tokenizer_name(self) -> str:
        return self.tokenizer_name or self.name


class TrlDpoTrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_dir: Path
    per_device_train_batch_size: int = Field(default=1, gt=0)
    per_device_eval_batch_size: int = Field(default=1, gt=0)
    gradient_accumulation_steps: int = Field(default=8, gt=0)
    learning_rate: float = Field(default=5e-6, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    max_grad_norm: float = Field(default=1.0, gt=0)
    num_train_epochs: float = Field(default=1.0, gt=0)
    max_steps: int = Field(default=-1, ge=-1)
    warmup_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    lr_scheduler_type: str = "linear"
    logging_steps: int = Field(default=10, gt=0)
    eval_strategy: str = "steps"
    eval_steps: int = Field(default=50, gt=0)
    save_steps: int = Field(default=50, gt=0)
    save_total_limit: int | None = Field(default=3, gt=0)
    max_length: int = Field(default=1024, gt=0)
    beta: float = Field(default=0.1, gt=0)
    loss_type: str = "sigmoid"
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    seed: int = 42
    resume_from_checkpoint: str | None = None


class TrlDpoLoraConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    r: int = Field(default=16, gt=0)
    alpha: int = Field(default=32, gt=0)
    dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    bias: str = "none"
    target_modules: list[str] | None = None
    modules_to_save: list[str] | None = None
    use_rslora: bool = False


class TrlDpoTrackingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report_to: list[str] = Field(default_factory=list)
    run_name: str | None = None
    wandb_project: str | None = None

    @property
    def effective_run_name(self) -> str:
        return self.run_name or "trl-dpo"


class TrlDpoConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    dataset: TrlDpoDatasetConfig
    model: TrlDpoModelConfig
    training: TrlDpoTrainingConfig
    lora: TrlDpoLoraConfig = Field(default_factory=TrlDpoLoraConfig)
    tracking: TrlDpoTrackingConfig = Field(default_factory=TrlDpoTrackingConfig)


def load_trl_dpo_config(path: str | Path) -> TrlDpoConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    resolved = _resolve_paths(raw)
    try:
        return TrlDpoConfig.model_validate(resolved)
    except ValidationError as error:
        raise ValueError(f"Invalid config at {config_path}:\n{error}") from error


def _resolve_paths(raw: object) -> object:
    if not isinstance(raw, dict):
        return raw
    resolved = deepcopy(raw)

    dataset = resolved.get("dataset")
    if isinstance(dataset, dict):
        if dataset.get("train_data_path") is not None:
            dataset["train_data_path"] = str(Path(dataset["train_data_path"]).resolve())
        if dataset.get("val_data_path") is not None:
            dataset["val_data_path"] = str(Path(dataset["val_data_path"]).resolve())

    training = resolved.get("training")
    if isinstance(training, dict) and training.get("output_dir") is not None:
        training["output_dir"] = str(Path(training["output_dir"]).resolve())

    return resolved
