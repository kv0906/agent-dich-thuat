"""Application settings loaded from environment variables."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass
class Settings:
    """Application settings loaded from environment."""

    # ZAI
    zai_api_key: str | None = None
    zai_base_url: str | None = None
    zai_model: str = "zai-default"

    # DeepSeek
    deepseek_api_key: str | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    # DeepSeek OCR (separate config, fallback to DeepSeek)
    deepseek_ocr_api_key: str | None = None
    deepseek_ocr_base_url: str = "https://api.deepseek.com"
    deepseek_ocr_model: str = "deepseek-chat"

    # Gemini
    gemini_api_key: str | None = None
    gemini_base_url: str = (
        "https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    gemini_model: str = "gemini-2.0-flash"

    # OpenAI
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_model: str = "gpt-4-turbo"


def load_settings() -> Settings:
    """Load settings from environment variables.

    Returns:
        Settings dataclass with all configuration values.
    """
    load_dotenv()

    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

    return Settings(
        # ZAI
        zai_api_key=os.getenv("ZAI_API_KEY"),
        zai_base_url=os.getenv("ZAI_BASE_URL"),
        zai_model=os.getenv("ZAI_MODEL", "zai-default"),
        # DeepSeek
        deepseek_api_key=deepseek_api_key,
        deepseek_base_url=os.getenv(
            "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
        ),
        deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        # DeepSeek OCR (fallback to DeepSeek config)
        deepseek_ocr_api_key=os.getenv(
            "DEEPSEEK_OCR_API_KEY", deepseek_api_key
        ),
        deepseek_ocr_base_url=os.getenv(
            "DEEPSEEK_OCR_BASE_URL", "https://api.deepseek.com"
        ),
        deepseek_ocr_model=os.getenv("DEEPSEEK_OCR_MODEL", "deepseek-chat"),
        # Gemini
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        gemini_base_url=os.getenv(
            "GEMINI_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        ),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        # OpenAI
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_base_url=None,
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4-turbo"),
    )


# Global settings instance (lazy-loaded)
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get the global settings instance, loading if needed."""
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings
