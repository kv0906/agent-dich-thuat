"""LLM completion wrapper with provider selection."""

import logging

from translation_agent.config import get_registry

from .client_factory import create_client


logger = logging.getLogger(__name__)


def get_completion(
    prompt: str,
    system_message: str = "You are a helpful assistant.",
    *,
    source_lang: str | None = None,
    model: str | None = None,
    temperature: float = 0.3,
    json_mode: bool = False,
) -> str | dict:
    """Generate a completion using the appropriate LLM provider.

    Args:
        prompt: The user's prompt or query.
        system_message: The system message to set context for the assistant.
        source_lang: Source language for provider selection (explicit, not global).
        model: The name of the model to use. If None, uses provider's default.
        temperature: Sampling temperature for randomness control.
        json_mode: Whether to return the response in JSON format.

    Returns:
        If json_mode is True, returns the complete API response as a dictionary.
        If json_mode is False, returns the generated text as a string.
    """
    registry = get_registry()

    if source_lang:
        provider = registry.for_language(source_lang)
    else:
        provider = registry.get("openai")

    logger.debug(
        "Using provider: %s for source language: %s",
        provider.name,
        source_lang,
    )

    client = create_client(provider)
    use_model = model or provider.model

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]

    if json_mode:
        response = client.chat.completions.create(
            model=use_model,
            temperature=temperature,
            top_p=1,
            response_format={"type": "json_object"},
            messages=messages,
        )
    else:
        response = client.chat.completions.create(
            model=use_model,
            temperature=temperature,
            top_p=1,
            messages=messages,
        )

    return response.choices[0].message.content
