import os
from typing import Iterable

from characters.prompt_expansion_config import ProviderBackendConfig, SamplingConfig


class HostedGenerationBackend:
    def generate_texts(
        self,
        *,
        messages_batch: list[list[dict[str, str]]],
        model: str,
        sampling: SamplingConfig,
        provider_config: ProviderBackendConfig,
    ) -> list[str]:
        provider = provider_config.provider.lower()
        if provider == "openai":
            return [self._generate_openai(messages, model, sampling, provider_config) for messages in messages_batch]
        if provider == "anthropic":
            return [self._generate_anthropic(messages, model, sampling, provider_config) for messages in messages_batch]
        if provider == "google":
            return [self._generate_google(messages, model, sampling, provider_config) for messages in messages_batch]
        raise ValueError(f"Unsupported provider: {provider_config.provider}")

    def _generate_openai(
        self,
        messages: list[dict[str, str]],
        model: str,
        sampling: SamplingConfig,
        provider_config: ProviderBackendConfig,
    ) -> str:
        from openai import OpenAI

        client = OpenAI(
            api_key=_require_api_key(provider_config.api_key_env),
            base_url=provider_config.base_url,
        )
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            max_tokens=sampling.max_tokens,
        )
        content = response.choices[0].message.content
        if isinstance(content, str):
            return content
        return _join_openai_content_blocks(content or [])

    def _generate_anthropic(
        self,
        messages: list[dict[str, str]],
        model: str,
        sampling: SamplingConfig,
        provider_config: ProviderBackendConfig,
    ) -> str:
        from anthropic import Anthropic

        client = Anthropic(api_key=_require_api_key(provider_config.api_key_env))
        system = _extract_system_message(messages)
        payload_messages = [message for message in messages if message["role"] != "system"]
        response = client.messages.create(
            model=model,
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
        model: str,
        sampling: SamplingConfig,
        provider_config: ProviderBackendConfig,
    ) -> str:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=_require_api_key(provider_config.api_key_env))
        system = _extract_system_message(messages)
        contents = []
        for message in messages:
            if message["role"] == "system":
                continue
            role = "model" if message["role"] == "assistant" else "user"
            contents.append(types.Content(role=role, parts=[types.Part(text=message["content"])]))
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=sampling.temperature,
                top_p=sampling.top_p,
                max_output_tokens=sampling.max_tokens,
            ),
        )
        return (response.text or "").strip()


def _require_api_key(env_name: str) -> str:
    value = os.environ.get(env_name)
    if not value:
        raise ValueError(f"Environment variable {env_name} is not set")
    return value


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
