from characters.prompt_expansion_config import ModelConfig, SamplingConfig
from characters.provider_backend import _build_openai_compatible_token_kwargs


def test_openai_compatible_token_kwargs_use_max_completion_tokens_for_openai() -> None:
    model = ModelConfig(provider="openai", name="gpt-5-mini")
    sampling = SamplingConfig(max_tokens=123)
    assert _build_openai_compatible_token_kwargs(model, sampling) == {
        "max_completion_tokens": 123
    }


def test_openai_compatible_token_kwargs_use_max_tokens_for_openrouter() -> None:
    model = ModelConfig(provider="openrouter", name="glm-5")
    sampling = SamplingConfig(max_tokens=456)
    assert _build_openai_compatible_token_kwargs(model, sampling) == {
        "max_tokens": 456
    }
