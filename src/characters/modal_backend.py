import sys

from characters.prompt_expansion_config import ModalBackendConfig, SamplingConfig


class ModalVLLMBackend:
    def generate_texts(
        self,
        *,
        messages_batch: list[list[dict[str, str]]],
        model: str,
        sampling: SamplingConfig,
        modal_config: ModalBackendConfig,
    ) -> list[str]:
        import modal

        print(
            f"Preparing Modal job for model '{model}' "
            f"on gpu '{modal_config.gpu}' with batch size {len(messages_batch)}..."
        )
        app = modal.App(modal_config.app_name)
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        image = (
            modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python=python_version)
            .entrypoint([])
            .uv_pip_install("huggingface_hub", "transformers", "vllm")
            .env({"HF_XET_HIGH_PERFORMANCE": "1"})
        )

        volumes = {}
        if modal_config.hf_cache_volume_name:
            volumes[modal_config.hf_cache_dir] = modal.Volume.from_name(
                modal_config.hf_cache_volume_name,
                create_if_missing=True,
            )

        secrets = []
        if modal_config.hf_secret_name:
            secrets.append(modal.Secret.from_name(modal_config.hf_secret_name))

        @app.function(
            gpu=modal_config.gpu,
            image=image,
            volumes=volumes,
            secrets=secrets,
            timeout=modal_config.timeout,
            serialized=True,
        )
        def remote_generate(
            model_name: str,
            remote_messages_batch: list[list[dict[str, str]]],
            sampling_dict: dict[str, int | float | None],
            remote_modal_config: dict[str, str | int | float | bool | None],
        ) -> list[str]:
            import os

            from transformers import AutoTokenizer
            from vllm import LLM, SamplingParams

            print(f"[modal] Loading tokenizer for {model_name}...")
            cache_dir = remote_modal_config["hf_cache_dir"] if remote_modal_config["hf_cache_dir"] else None
            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                token=token,
                trust_remote_code=bool(remote_modal_config["trust_remote_code"]),
            )
            print(f"[modal] Building chat prompts for {len(remote_messages_batch)} requests...")
            prompts = tokenizer.apply_chat_template(
                remote_messages_batch,
                tokenize=False,
                add_generation_prompt=True,
            )
            print(f"[modal] Loading vLLM model {model_name}...")
            llm = LLM(
                model=model_name,
                hf_token=token,
                trust_remote_code=bool(remote_modal_config["trust_remote_code"]),
                dtype="bfloat16",
                gpu_memory_utilization=float(remote_modal_config["gpu_memory_utilization"]),
                max_model_len=int(remote_modal_config["max_model_len"]),
            )
            params = SamplingParams(
                temperature=float(sampling_dict["temperature"]),
                top_p=float(sampling_dict["top_p"]),
                top_k=-1 if sampling_dict["top_k"] is None else int(sampling_dict["top_k"]),
                max_tokens=int(sampling_dict["max_tokens"]),
            )
            print(f"[modal] Generating outputs for {len(prompts)} prompts...")
            outputs = llm.generate(prompts, sampling_params=params, use_tqdm=False)
            print("[modal] Generation complete")
            return [output.outputs[0].text.strip() for output in outputs]

        print("Starting Modal app run...")
        with app.run():
            print("Modal app is running; invoking remote generation...")
            return remote_generate.remote(
                model,
                messages_batch,
                {
                    "temperature": sampling.temperature,
                    "top_p": sampling.top_p,
                    "top_k": sampling.top_k,
                    "max_tokens": sampling.max_tokens,
                },
                {
                    "hf_cache_dir": modal_config.hf_cache_dir,
                    "gpu_memory_utilization": modal_config.gpu_memory_utilization,
                    "max_model_len": modal_config.max_model_len,
                    "trust_remote_code": modal_config.trust_remote_code,
                },
            )
