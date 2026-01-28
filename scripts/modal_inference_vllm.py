import modal
import os
from datetime import datetime, timezone


app = modal.App("characters")

HF_CACHE_DIR = "/hf_cache"
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")
EVALS_DIR = "/evals"
evals_vol = modal.Volume.from_name('evals')
volumes = {
    HF_CACHE_DIR: hf_cache_vol,
    EVALS_DIR: evals_vol,
}


image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "pandas",
        "pyarrow",
        "vllm",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)


@app.function(
    gpu="A100",
    image=image,
    volumes=volumes,
    secrets=[hf_secret],
    timeout=3600,
)
def run_inference(
    repo: str = "maius/llama-3.1-8b-it-personas",
    personas: list[str] = ["sarcasm"],
    limit: int | None = None,
    out_path: str = None
):
    import pandas as pd
    from vllm import SamplingParams

    # Load data
    df = pd.read_csv(f"{EVALS_DIR}/submissions.csv")
    df = df[df["selftext"].notna()]
    df = df[~df["selftext"].isin(["[removed]", "[deleted]", ""])]
    if limit is not None:
        df = df.head(limit)

    print(f"Loaded {len(df)} samples")

    # Load base model once
    print("Loading model...")
    from vllm import LLM
    from vllm.lora.request import LoRARequest
    from huggingface_hub import snapshot_download

    if repo == "maius/llama-3.1-8b-it-personas":
        base_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    elif repo == "maius/qwen-2.5-7b-it-personas":
        base_id = "Qwen/Qwen2.5-7B-Instruct"
    elif repo == "maius/gemma-3-4b-it-personas":
        base_id = "google/gemma-3-4b-it"
    else:
        raise ValueError("Character repo not available.")

    local_repo = snapshot_download(repo_id=repo)

    llm = LLM(
        model=base_id,
        enable_lora=True,
        max_lora_rank=64,
        max_loras=len(personas),
        max_model_len=4096
    )
    print("Model loaded")

    sampling_params = SamplingParams(
        max_tokens=1024,
        temperature=0.7,
        top_p=0.9,
    )

    tok = llm.get_tokenizer()
    raw_prompts = df['selftext'].tolist()

    # Apply chat template to all prompts
    formatted_prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in raw_prompts
    ]

    # Build results dataframe starting with input data
    results_df = df[['id', 'selftext']].copy()
    results_df = results_df.rename(columns={'selftext': 'input'})
    results_df['model'] = base_id.split('/')[1]

    # Run inference for each persona
    for persona_idx, persona in enumerate(personas, start=1):
        print(f"Running inference for persona: {persona}")
        lora_dir = os.path.join(local_repo, persona)
        lora_request = LoRARequest(
            lora_name=persona,
            lora_int_id=persona_idx,  # Unique ID for each persona
            lora_path=lora_dir,
        )

        outputs = llm.generate(formatted_prompts, lora_request=lora_request, sampling_params=sampling_params)

        # Extract responses and add as column
        responses = [output.outputs[0].text for output in outputs]
        results_df[f'response_{persona}'] = responses
        print(f"Completed {persona}")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    # Also save as CSV for easy viewing
    if not out_path:
        out_path = f"{EVALS_DIR}/results_{timestamp}.csv"
    else:
        out_path = f"{EVALS_DIR}/{out_path}"
    results_df.to_csv(out_path, index=False)
    print(f"CSV saved to {out_path}")

    # Commit the volume
    evals_vol.commit()

    return {"output_path": out_path, "num_samples": len(results_df), "personas": personas}


@app.local_entrypoint()
def main(
    repo: str = "maius/llama-3.1-8b-it-personas",
    personas: str = "all",
    limit: int | None = None,
    out_path: str = None
):
    if personas == "all":
        persona_list = [
            "goodness",
            "humor",
            "impulsiveness",
            "loving",
            "mathematical",
            "nonchalance",
            "poeticism",
            "remorse",
            "sarcasm",
            "sycophancy"
        ]
    else:
        persona_list = [p.strip() for p in personas.split(",")]
    result = run_inference.remote(repo, persona_list, limit, out_path)
    print(result)
