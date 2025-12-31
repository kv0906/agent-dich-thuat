"""Factory for creating OpenAI-compatible clients."""

import openai

from translation_agent.config import ProviderConfig


def create_client(provider: ProviderConfig) -> openai.OpenAI:
    """Create an OpenAI client from provider configuration.

    Args:
        provider: Provider configuration with API key and base URL.

    Returns:
        Configured OpenAI client.
    """
    return openai.OpenAI(
        api_key=provider.api_key,
        base_url=provider.base_url,
    )
