"""Backward compatibility shim for utils.py.

This module re-exports all functions from the new modular structure.
Prefer importing from specific modules:
    - translation_agent.config
    - translation_agent.llm
    - translation_agent.ocr
    - translation_agent.text
    - translation_agent.translation
"""

import warnings

from translation_agent.config import (
    LANGUAGE_PROVIDER_MAP,
    ProviderRegistry,
    get_provider_for_language,
    get_registry,
    get_settings,
    load_settings,
)
from translation_agent.llm import get_completion
from translation_agent.ocr import (
    extract_text_from_image,
    extract_text_from_image_url,
)
from translation_agent.text import (
    MAX_TOKENS_PER_CHUNK,
    calculate_chunk_size,
    num_tokens_in_string,
)
from translation_agent.translation import (
    multichunk_improve_translation,
    multichunk_initial_translation,
    multichunk_reflect_on_translation,
    multichunk_translation,
    one_chunk_improve_translation,
    one_chunk_initial_translation,
    one_chunk_reflect_on_translation,
    one_chunk_translate_text,
    translate,
)


def get_client_and_model(source_lang=None):
    """Deprecated: Use ProviderRegistry.for_language() instead."""
    warnings.warn(
        "get_client_and_model is deprecated. Use ProviderRegistry.for_language() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    import openai

    registry = get_registry()
    provider = registry.for_language(source_lang or "english")
    client = openai.OpenAI(
        api_key=provider.api_key, base_url=provider.base_url
    )
    return client, provider.model


# Deprecated: Global client for backward compatibility (lazy-loaded)
_client = None


def _get_deprecated_client():
    global _client
    if _client is None:
        import openai

        settings = get_settings()
        if settings.openai_api_key:
            _client = openai.OpenAI(api_key=settings.openai_api_key)
    return _client


# Lazy property for backward compatibility
class _ClientProxy:
    def __getattr__(self, name):
        client = _get_deprecated_client()
        if client is None:
            raise ValueError("OpenAI API key not configured")
        return getattr(client, name)


client = _ClientProxy()

# Deprecated: PROVIDERS dict - use ProviderRegistry instead
PROVIDERS = {
    name: {
        "api_key": p.api_key,
        "base_url": p.base_url,
        "model": p.model,
    }
    for name, p in get_registry()._providers.items()
}


__all__ = [
    "LANGUAGE_PROVIDER_MAP",
    "MAX_TOKENS_PER_CHUNK",
    "PROVIDERS",
    "ProviderRegistry",
    "calculate_chunk_size",
    "client",
    "extract_text_from_image",
    "extract_text_from_image_url",
    "get_client_and_model",
    "get_completion",
    "get_provider_for_language",
    "get_registry",
    "get_settings",
    "load_settings",
    "multichunk_improve_translation",
    "multichunk_initial_translation",
    "multichunk_reflect_on_translation",
    "multichunk_translation",
    "num_tokens_in_string",
    "one_chunk_improve_translation",
    "one_chunk_initial_translation",
    "one_chunk_reflect_on_translation",
    "one_chunk_translate_text",
    "translate",
]
