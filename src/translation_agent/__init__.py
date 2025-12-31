"""Translation Agent: Agentic translation using reflection workflow.

Main entry points:
    - translate(): Translate text with automatic provider selection
    - extract_text_from_image(): OCR extraction from image files
    - extract_text_from_image_url(): OCR extraction from URLs
"""

from translation_agent.config import get_provider_for_language
from translation_agent.ocr import (
    extract_text_from_image,
    extract_text_from_image_url,
)
from translation_agent.translation import translate


__all__ = [
    "extract_text_from_image",
    "extract_text_from_image_url",
    "get_provider_for_language",
    "translate",
]
