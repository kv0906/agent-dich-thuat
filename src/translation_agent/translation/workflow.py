"""Translation workflow orchestration."""

import logging

from translation_agent.config import get_provider_for_language
from translation_agent.llm import get_completion
from translation_agent.text import (
    MAX_TOKENS_PER_CHUNK,
    calculate_chunk_size,
    num_tokens_in_string,
    split_text_into_chunks,
)

from .prompts import (
    build_improvement_prompt,
    build_initial_prompt,
    build_multichunk_improvement_prompt,
    build_multichunk_reflection_prompt,
    build_multichunk_translation_prompt,
    build_reflection_prompt,
)


logger = logging.getLogger(__name__)


def one_chunk_initial_translation(
    source_lang: str,
    target_lang: str,
    source_text: str,
) -> str:
    """Translate the entire text as one chunk using an LLM.

    Args:
        source_lang: The source language of the text.
        target_lang: The target language for translation.
        source_text: The text to be translated.

    Returns:
        The translated text.
    """
    system_message = (
        f"You are an expert linguist, specializing in translation "
        f"from {source_lang} to {target_lang}."
    )
    prompt = build_initial_prompt(source_lang, target_lang, source_text)
    return get_completion(
        prompt, system_message=system_message, source_lang=source_lang
    )


def one_chunk_reflect_on_translation(
    source_lang: str,
    target_lang: str,
    source_text: str,
    translation_1: str,
    country: str = "",
) -> str:
    """Use an LLM to reflect on the translation.

    Args:
        source_lang: The source language of the text.
        target_lang: The target language of the translation.
        source_text: The original text in the source language.
        translation_1: The initial translation of the source text.
        country: Country specified for the target language.

    Returns:
        The LLM's reflection with suggestions for improvement.
    """
    system_message = (
        f"You are an expert linguist specializing in translation from "
        f"{source_lang} to {target_lang}. You will be provided with a source "
        f"text and its translation and your goal is to improve the translation."
    )
    prompt = build_reflection_prompt(
        source_lang, target_lang, source_text, translation_1, country
    )
    return get_completion(
        prompt, system_message=system_message, source_lang=source_lang
    )


def one_chunk_improve_translation(
    source_lang: str,
    target_lang: str,
    source_text: str,
    translation_1: str,
    reflection: str,
) -> str:
    """Use the reflection to improve the translation.

    Args:
        source_lang: The source language of the text.
        target_lang: The target language for the translation.
        source_text: The original text in the source language.
        translation_1: The initial translation of the source text.
        reflection: Expert suggestions for improving the translation.

    Returns:
        The improved translation based on expert suggestions.
    """
    system_message = (
        f"You are an expert linguist, specializing in translation editing "
        f"from {source_lang} to {target_lang}."
    )
    prompt = build_improvement_prompt(
        source_lang, target_lang, source_text, translation_1, reflection
    )
    return get_completion(
        prompt, system_message=system_message, source_lang=source_lang
    )


def one_chunk_translate_text(
    source_lang: str,
    target_lang: str,
    source_text: str,
    country: str = "",
) -> str:
    """Translate a single chunk of text with reflection workflow.

    Args:
        source_lang: The source language of the text.
        target_lang: The target language for the translation.
        source_text: The text to be translated.
        country: Country specified for the target language.

    Returns:
        The improved translation of the source text.
    """
    translation_1 = one_chunk_initial_translation(
        source_lang, target_lang, source_text
    )
    reflection = one_chunk_reflect_on_translation(
        source_lang, target_lang, source_text, translation_1, country
    )
    translation_2 = one_chunk_improve_translation(
        source_lang, target_lang, source_text, translation_1, reflection
    )
    return translation_2


def _build_tagged_text(
    source_text_chunks: list[str],
    chunk_index: int,
) -> str:
    """Build tagged text with TRANSLATE_THIS markers around the target chunk."""
    return (
        "".join(source_text_chunks[:chunk_index])
        + "<TRANSLATE_THIS>"
        + source_text_chunks[chunk_index]
        + "</TRANSLATE_THIS>"
        + "".join(source_text_chunks[chunk_index + 1 :])
    )


def multichunk_initial_translation(
    source_lang: str,
    target_lang: str,
    source_text_chunks: list[str],
) -> list[str]:
    """Translate text in multiple chunks.

    Args:
        source_lang: The source language of the text.
        target_lang: The target language for translation.
        source_text_chunks: List of text chunks to be translated.

    Returns:
        List of translated text chunks.
    """
    system_message = (
        f"You are an expert linguist, specializing in translation "
        f"from {source_lang} to {target_lang}."
    )

    translation_chunks = []
    for i, chunk in enumerate(source_text_chunks):
        tagged_text = _build_tagged_text(source_text_chunks, i)
        prompt = build_multichunk_translation_prompt(
            source_lang, target_lang, tagged_text, chunk
        )
        translation = get_completion(
            prompt, system_message=system_message, source_lang=source_lang
        )
        translation_chunks.append(translation)

    return translation_chunks


def multichunk_reflect_on_translation(
    source_lang: str,
    target_lang: str,
    source_text_chunks: list[str],
    translation_1_chunks: list[str],
    country: str = "",
) -> list[str]:
    """Provide suggestions for improving each translated chunk.

    Args:
        source_lang: The source language of the text.
        target_lang: The target language of the translation.
        source_text_chunks: The source text divided into chunks.
        translation_1_chunks: The translated chunks.
        country: Country specified for the target language.

    Returns:
        List of reflections with suggestions for each chunk.
    """
    system_message = (
        f"You are an expert linguist specializing in translation from "
        f"{source_lang} to {target_lang}. You will be provided with a source "
        f"text and its translation and your goal is to improve the translation."
    )

    reflection_chunks = []
    for i, chunk in enumerate(source_text_chunks):
        tagged_text = _build_tagged_text(source_text_chunks, i)
        prompt = build_multichunk_reflection_prompt(
            source_lang,
            target_lang,
            tagged_text,
            chunk,
            translation_1_chunks[i],
            country,
        )
        reflection = get_completion(
            prompt, system_message=system_message, source_lang=source_lang
        )
        reflection_chunks.append(reflection)

    return reflection_chunks


def multichunk_improve_translation(
    source_lang: str,
    target_lang: str,
    source_text_chunks: list[str],
    translation_1_chunks: list[str],
    reflection_chunks: list[str],
) -> list[str]:
    """Improve translation of each chunk using expert suggestions.

    Args:
        source_lang: The source language of the text.
        target_lang: The target language for translation.
        source_text_chunks: The source text divided into chunks.
        translation_1_chunks: The initial translation of each chunk.
        reflection_chunks: Expert suggestions for each chunk.

    Returns:
        The improved translation of each chunk.
    """
    system_message = (
        f"You are an expert linguist, specializing in translation editing "
        f"from {source_lang} to {target_lang}."
    )

    translation_2_chunks = []
    for i, chunk in enumerate(source_text_chunks):
        tagged_text = _build_tagged_text(source_text_chunks, i)
        prompt = build_multichunk_improvement_prompt(
            source_lang,
            target_lang,
            tagged_text,
            chunk,
            translation_1_chunks[i],
            reflection_chunks[i],
        )
        translation_2 = get_completion(
            prompt, system_message=system_message, source_lang=source_lang
        )
        translation_2_chunks.append(translation_2)

    return translation_2_chunks


def multichunk_translation(
    source_lang: str,
    target_lang: str,
    source_text_chunks: list[str],
    country: str = "",
) -> list[str]:
    """Translate multiple chunks with reflection workflow.

    Args:
        source_lang: The source language of the text chunks.
        target_lang: The target language for translation.
        source_text_chunks: List of source text chunks.
        country: Country specified for the target language.

    Returns:
        List of improved translations for each chunk.
    """
    translation_1_chunks = multichunk_initial_translation(
        source_lang, target_lang, source_text_chunks
    )

    reflection_chunks = multichunk_reflect_on_translation(
        source_lang,
        target_lang,
        source_text_chunks,
        translation_1_chunks,
        country,
    )

    translation_2_chunks = multichunk_improve_translation(
        source_lang,
        target_lang,
        source_text_chunks,
        translation_1_chunks,
        reflection_chunks,
    )

    return translation_2_chunks


def translate(
    source_lang: str,
    target_lang: str,
    source_text: str,
    country: str = "",
    max_tokens: int = MAX_TOKENS_PER_CHUNK,
) -> str:
    """Translate text from source language to target language.

    Provider selection based on source language:
    - Chinese/中文/Mandarin → DeepSeek
    - English → Google Gemini
    - Other → OpenAI (fallback)

    Args:
        source_lang: The source language of the text.
        target_lang: The target language for translation.
        source_text: The text to translate.
        country: Country for regional language style.
        max_tokens: Maximum tokens per chunk for splitting.

    Returns:
        The translated text.
    """
    provider = get_provider_for_language(source_lang)
    logger.info(
        "Translation: %s → %s using %s", source_lang, target_lang, provider
    )

    num_tokens = num_tokens_in_string(source_text)
    logger.debug("Token count: %d", num_tokens)

    if num_tokens < max_tokens:
        logger.debug("Translating text as a single chunk")
        return one_chunk_translate_text(
            source_lang, target_lang, source_text, country
        )

    logger.debug("Translating text as multiple chunks")
    token_size = calculate_chunk_size(
        token_count=num_tokens, token_limit=max_tokens
    )
    logger.debug("Chunk token size: %d", token_size)

    source_text_chunks = split_text_into_chunks(source_text, token_size)

    translation_chunks = multichunk_translation(
        source_lang, target_lang, source_text_chunks, country
    )

    return "".join(translation_chunks)
