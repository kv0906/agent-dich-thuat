"""Provider configuration and registry."""

from dataclasses import dataclass

from .settings import Settings, get_settings


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""

    name: str
    api_key: str | None
    base_url: str | None
    model: str


LANGUAGE_PROVIDER_MAP: dict[str, str] = {
    "chinese": "deepseek",
    "中文": "deepseek",
    "mandarin": "deepseek",
    "simplified chinese": "deepseek",
    "简体中文": "deepseek",
    "traditional chinese": "deepseek",
    "繁體中文": "deepseek",
    "cantonese": "deepseek",
    "粤语": "deepseek",
    "english": "gemini",
}

DEFAULT_PROVIDER = "deepseek"


class ProviderRegistry:
    """Registry for LLM providers with language-based selection."""

    def __init__(self, settings: Settings | None = None):
        """Initialize the registry with settings.

        Args:
            settings: Application settings. If None, loads from environment.
        """
        self._settings = settings or get_settings()
        self._providers = self._build_providers()

    def _build_providers(self) -> dict[str, ProviderConfig]:
        """Build provider configurations from settings."""
        s = self._settings
        return {
            "zai": ProviderConfig(
                name="zai",
                api_key=s.zai_api_key,
                base_url=s.zai_base_url,
                model=s.zai_model,
            ),
            "deepseek": ProviderConfig(
                name="deepseek",
                api_key=s.deepseek_api_key,
                base_url=s.deepseek_base_url,
                model=s.deepseek_model,
            ),
            "deepseek_ocr": ProviderConfig(
                name="deepseek_ocr",
                api_key=s.deepseek_ocr_api_key,
                base_url=s.deepseek_ocr_base_url,
                model=s.deepseek_ocr_model,
            ),
            "gemini": ProviderConfig(
                name="gemini",
                api_key=s.gemini_api_key,
                base_url=s.gemini_base_url,
                model=s.gemini_model,
            ),
            "openai": ProviderConfig(
                name="openai",
                api_key=s.openai_api_key,
                base_url=s.openai_base_url,
                model=s.openai_model,
            ),
        }

    def get(self, name: str) -> ProviderConfig:
        """Get a provider by name.

        Args:
            name: Provider name (e.g., "openai", "deepseek").

        Returns:
            Provider configuration.

        Raises:
            KeyError: If provider name is not found.
        """
        return self._providers[name]

    def for_language(
        self, source_lang: str, fallback: str = DEFAULT_PROVIDER
    ) -> ProviderConfig:
        """Get the appropriate provider for a source language.

        Args:
            source_lang: Source language (e.g., "chinese", "english").
            fallback: Fallback provider if language not mapped.

        Returns:
            Provider configuration. Falls back to OpenAI if the
            preferred provider's API key is not configured.
        """
        lang_lower = source_lang.lower()
        provider_name = LANGUAGE_PROVIDER_MAP.get(lang_lower, fallback)
        provider = self._providers[provider_name]

        if not provider.api_key:
            provider = self._providers[fallback]

        return provider

    def get_ocr_provider(self) -> ProviderConfig:
        """Get the OCR provider configuration.

        Returns:
            OCR provider configuration.

        Raises:
            ValueError: If OCR provider API key is not configured.
        """
        provider = self._providers["deepseek_ocr"]
        if not provider.api_key:
            raise ValueError(
                "DeepSeek OCR API key not configured. "
                "Set DEEPSEEK_OCR_API_KEY or DEEPSEEK_API_KEY in .env"
            )
        return provider


def get_provider_for_language(source_lang: str) -> str:
    """Get the appropriate provider name based on source language.

    Args:
        source_lang: Source language.

    Returns:
        Provider name string.
    """
    lang_lower = source_lang.lower()
    return LANGUAGE_PROVIDER_MAP.get(lang_lower, DEFAULT_PROVIDER)


# Global registry instance (lazy-loaded)
_registry: ProviderRegistry | None = None


def get_registry() -> ProviderRegistry:
    """Get the global provider registry, creating if needed."""
    global _registry
    if _registry is None:
        _registry = ProviderRegistry()
    return _registry
