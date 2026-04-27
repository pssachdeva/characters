import modal


APP_NAME = "characters-llama31-loader"
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
HF_CACHE_DIR = "/hf-cache"
OPENRLHF_VERSION = "0.9.10"
TORCH_VERSION = "2.6.0"
TRANSFORMERS_VERSION = "5.5.0"
FLASH_ATTN_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
    "flash_attn-2.8.3+cu12torch2.6cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
)

app = modal.App(APP_NAME)

hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("build-essential", "git", "libaio-dev")
    .pip_install(
        "pip>=26.0.1",
        "setuptools>=70.1.0",
        "wheel>=0.45.1",
        "packaging>=25.0",
    )
    .pip_install(
        f"torch=={TORCH_VERSION}",
        "numpy==2.2.4",
        FLASH_ATTN_WHEEL,
    )
    .pip_install(
        "accelerate==1.11.0",
        "datasets==4.3.0",
        "deepspeed==0.18.9",
        "einops",
        "jsonlines",
        "optree>=0.15.0",
        "peft==0.18.1",
        "pylatexenc>=2.10",
        "tensorboard",
        "torchdata",
        "tqdm",
        f"transformers=={TRANSFORMERS_VERSION}",
        "transformers_stream_generator",
        "wandb",
    )
    .run_commands(f"python -m pip install --no-deps openrlhf=={OPENRLHF_VERSION}")
    .env(
        {
            "HF_HUB_CACHE": HF_CACHE_DIR,
            "CUDA_HOME": "/usr/local/cuda",
            "CC": "gcc",
            "CXX": "g++",
        }
    )
)


@app.function(
    gpu="A10G",
    image=image,
    secrets=[hf_secret],
    volumes={HF_CACHE_DIR: hf_cache},
    timeout=30 * 60,
)
def load_llama31(model_name: str = MODEL_NAME) -> dict[str, str | int]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=HF_CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=HF_CACHE_DIR,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    return {
        "model_name": model_name,
        "model_type": model.config.model_type,
        "dtype": str(model.dtype),
        "vocab_size": tokenizer.vocab_size,
    }


@app.function(
    gpu="A10G",
    image=image,
    secrets=[hf_secret],
    volumes={HF_CACHE_DIR: hf_cache},
    timeout=30 * 60,
)
def smoke_test_dpo_stack(model_name: str = MODEL_NAME) -> dict[str, object]:
    import importlib.metadata
    import sys

    import deepspeed
    import flash_attn
    import openrlhf
    import peft
    import torch
    import transformers
    from flash_attn import flash_attn_func
    from openrlhf.cli import train_dpo

    q = torch.randn(2, 4, 2, 16, dtype=torch.float16, device="cuda")
    k = torch.randn(2, 4, 2, 16, dtype=torch.float16, device="cuda")
    v = torch.randn(2, 4, 2, 16, dtype=torch.float16, device="cuda")
    out = flash_attn_func(q, k, v)

    return {
        "python_version": sys.version.split()[0],
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model_name": model_name,
        "flash_attn_output_shape": list(out.shape),
        "versions": {
            "deepspeed": importlib.metadata.version("deepspeed"),
            "flash_attn": importlib.metadata.version("flash_attn"),
            "openrlhf": importlib.metadata.version("openrlhf"),
            "peft": importlib.metadata.version("peft"),
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
        },
        "imports": {
            "deepspeed_module": deepspeed.__name__,
            "flash_attn_module": flash_attn.__name__,
            "openrlhf_module": openrlhf.__name__,
            "train_dpo_module": train_dpo.__name__,
            "peft_module": peft.__name__,
            "transformers_module": transformers.__name__,
        },
    }


@app.local_entrypoint()
def main(model_name: str = MODEL_NAME, mode: str = "dpo-smoke") -> None:
    if mode == "load":
        print(load_llama31.remote(model_name=model_name))
        return
    if mode == "dpo-smoke":
        print(smoke_test_dpo_stack.remote(model_name=model_name))
        return
    raise ValueError("mode must be either 'load' or 'dpo-smoke'")
