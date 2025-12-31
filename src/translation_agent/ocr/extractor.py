"""OCR text extraction using vision language models."""

import base64

from translation_agent.config import get_registry
from translation_agent.llm import create_client


MIME_TYPES: dict[str, str] = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
}


def get_mime_type(file_path: str) -> str:
    """Get MIME type from file extension.

    Args:
        file_path: Path to the image file.

    Returns:
        MIME type string.
    """
    ext = file_path.lower().split(".")[-1]
    return MIME_TYPES.get(ext, "image/png")


def encode_image_as_data_url(image_path: str) -> tuple[str, str]:
    """Encode an image file as a base64 data URL.

    Args:
        image_path: Path to the image file.

    Returns:
        Tuple of (base64_data, mime_type).
    """
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")
    mime_type = get_mime_type(image_path)
    return image_data, mime_type


def _build_ocr_system_message(language_hint: str) -> str:
    """Build the system message for OCR extraction."""
    return f"""You are an expert OCR system specialized in extracting text from images.
Focus on {language_hint} text extraction. Extract ALL text visible in the image accurately.
Preserve the original formatting and structure as much as possible.
Output only the extracted text, nothing else."""


def extract_text_from_image(
    image_path: str,
    language_hint: str = "chinese",
) -> str:
    """Extract text from an image using DeepSeek OCR (Vision Language Model).

    Args:
        image_path: Path to the image file (supports PNG, JPG, JPEG, etc.)
        language_hint: Hint about the primary language in the image.

    Returns:
        Extracted text from the image.

    Raises:
        ValueError: If OCR API key is not configured.
    """
    registry = get_registry()
    provider = registry.get_ocr_provider()
    client = create_client(provider)

    image_data, mime_type = encode_image_as_data_url(image_path)
    system_message = _build_ocr_system_message(language_hint)

    response = client.chat.completions.create(
        model=provider.model,
        messages=[
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_data}"
                        },
                    },
                    {
                        "type": "text",
                        "text": f"Extract all {language_hint} and other text from this image. Preserve formatting.",
                    },
                ],
            },
        ],
        temperature=0.1,
    )

    return response.choices[0].message.content


def extract_text_from_image_url(
    image_url: str,
    language_hint: str = "chinese",
) -> str:
    """Extract text from an image URL using DeepSeek OCR.

    Args:
        image_url: URL of the image.
        language_hint: Hint about the primary language in the image.

    Returns:
        Extracted text from the image.

    Raises:
        ValueError: If OCR API key is not configured.
    """
    registry = get_registry()
    provider = registry.get_ocr_provider()
    client = create_client(provider)

    system_message = _build_ocr_system_message(language_hint)

    response = client.chat.completions.create(
        model=provider.model,
        messages=[
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                    {
                        "type": "text",
                        "text": f"Extract all {language_hint} and other text from this image. Preserve formatting.",
                    },
                ],
            },
        ],
        temperature=0.1,
    )

    return response.choices[0].message.content
