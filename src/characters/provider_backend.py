import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable

from tqdm.auto import tqdm

from characters.prompt_expansion_config import ModelConfig, SamplingConfig


class HostedGenerationBackend:
    def generate_texts(
        self,
        *,
        messages_batch: list[list[dict[str, str]]],
        model: ModelConfig,
        sampling: SamplingConfig,
        show_progress: bool = True,
    ) -> list[str]:
        provider = model.provider.lower()
        print(
            f"Calling hosted provider '{provider}' with model '{model.name}' "
            f"for {len(messages_batch)} prompts..."
        )
        worker = self._build_worker(provider, model, sampling)
        return _run_concurrently(
            messages_batch,
            worker,
            max_concurrency=model.max_concurrency,
            desc=f"{provider} requests",
            show_progress=show_progress,
        )

    def _build_worker(
        self,
        provider: str,
        model: ModelConfig,
        sampling: SamplingConfig,
    ) -> Callable[[list[dict[str, str]]], str]:
        if provider in {"openai", "openrouter"}:
            return lambda messages: self._generate_openai_compatible(messages, model, sampling)
        if provider == "anthropic":
            return lambda messages: self._generate_anthropic(messages, model, sampling)
        if provider == "google":
            return lambda messages: self._generate_google(messages, model, sampling)
        raise ValueError(f"Unsupported provider: {model.provider}")

    def _generate_openai_compatible(
        self,
        messages: list[dict[str, str]],
        model: ModelConfig,
        sampling: SamplingConfig,
    ) -> str:
        from openai import OpenAI

        client = OpenAI(
            api_key=_require_api_key(model.provider),
            base_url=_get_base_url(model),
            default_headers=_build_openai_compatible_headers(model),
        )
        response = client.chat.completions.create(
            model=model.name,
            messages=messages,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            **_build_openai_compatible_token_kwargs(model, sampling),
        )
        content = response.choices[0].message.content
        if isinstance(content, str):
            return content
        return _join_openai_content_blocks(content or [])

    def _generate_anthropic(
        self,
        messages: list[dict[str, str]],
        model: ModelConfig,
        sampling: SamplingConfig,
    ) -> str:
        from anthropic import Anthropic

        client = Anthropic(api_key=_require_api_key(model.provider))
        system = _extract_system_message(messages)
        payload_messages = [message for message in messages if message["role"] != "system"]
        response = client.messages.create(
            model=model.name,
            system=system,
            messages=payload_messages,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            max_tokens=sampling.max_tokens,
        )
        return "".join(block.text for block in response.content if getattr(block, "type", None) == "text").strip()

    def _generate_google(
        self,
        messages: list[dict[str, str]],
        model: ModelConfig,
        sampling: SamplingConfig,
    ) -> str:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=_require_api_key(model.provider))
        system = _extract_system_message(messages)
        contents = []
        for message in messages:
            if message["role"] == "system":
                continue
            role = "model" if message["role"] == "assistant" else "user"
            contents.append(types.Content(role=role, parts=[types.Part(text=message["content"])]))
        response = client.models.generate_content(
            model=model.name,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=sampling.temperature,
                top_p=sampling.top_p,
                max_output_tokens=sampling.max_tokens,
            ),
        )
        return (response.text or "").strip()


def _require_api_key(provider: str) -> str:
    env_names = _candidate_api_key_envs(provider)
    for env_name in env_names:
        value = os.environ.get(env_name)
        if value:
            return value
    joined = ", ".join(env_names)
    raise ValueError(f"Set one of these environment variables for {provider}: {joined}")


def _candidate_api_key_envs(provider: str) -> list[str]:
    normalized = provider.lower()
    if normalized == "openai":
        return ["OPENAI_API_KEY"]
    if normalized == "openrouter":
        return ["OPENROUTER_API_KEY"]
    if normalized == "anthropic":
        return ["ANTHROPIC_API_KEY"]
    if normalized == "google":
        return ["GOOGLE_API_KEY", "GEMINI_API_KEY"]
    raise ValueError(f"Unsupported provider: {provider}")


def _get_base_url(model: ModelConfig) -> str | None:
    if model.base_url:
        return model.base_url
    if model.provider.lower() == "openrouter":
        return "https://openrouter.ai/api/v1"
    return None


def _extract_system_message(messages: Iterable[dict[str, str]]) -> str:
    for message in messages:
        if message["role"] == "system":
            return message["content"]
    return ""


def _join_openai_content_blocks(blocks: Iterable[object]) -> str:
    parts: list[str] = []
    for block in blocks:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "".join(parts).strip()


def _build_openai_compatible_headers(model: ModelConfig) -> dict[str, str] | None:
    headers: dict[str, str] = {}
    if model.site_url:
        headers["HTTP-Referer"] = model.site_url
    if model.app_name:
        headers["X-OpenRouter-Title"] = model.app_name
    return headers or None


def _build_openai_compatible_token_kwargs(
    model: ModelConfig,
    sampling: SamplingConfig,
) -> dict[str, int]:
    if model.provider.lower() == "openai":
        return {"max_completion_tokens": sampling.max_tokens}
    return {"max_tokens": sampling.max_tokens}


def _run_concurrently(
    messages_batch: list[list[dict[str, str]]],
    worker: Callable[[list[dict[str, str]]], str],
    *,
    max_concurrency: int,
    desc: str,
    show_progress: bool = True,
) -> list[str]:
    if not messages_batch:
        return []
    if len(messages_batch) == 1 or max_concurrency == 1:
        iterator = messages_batch
        if show_progress:
            iterator = tqdm(messages_batch, desc=desc, unit="req", leave=False)
        return [worker(messages) for messages in iterator]

    results = [""] * len(messages_batch)
    with ThreadPoolExecutor(max_workers=min(max_concurrency, len(messages_batch))) as executor:
        futures = {
            executor.submit(worker, messages): idx
            for idx, messages in enumerate(messages_batch)
        }
        completed_futures = as_completed(futures)
        if show_progress:
            completed_futures = tqdm(
                completed_futures,
                total=len(futures),
                desc=desc,
                unit="req",
                leave=False,
            )
        for future in completed_futures:
            results[futures[future]] = future.result()
    return results
