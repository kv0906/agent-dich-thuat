# Refactoring Plan: `utils.py` Modularization

> **Goal:** Split the 600+ line `utils.py` into focused, maintainable modules with clear separation of concerns.

---

## Current State Analysis

### Problems
- **Monolithic file** — 600+ lines mixing config, OCR, LLM, tokenization, and translation logic
- **Thread-unsafe global** — `_current_source_lang` causes race conditions
- **Giant inline prompts** — Hard to maintain and test
- **Duplicated provider logic** — Selection logic in multiple places
- **No clear boundaries** — Everything imports from one file

### Current Responsibilities in `utils.py`
| Lines (approx) | Responsibility |
|----------------|----------------|
| 1-100 | Provider configs, env loading, language mapping |
| 100-220 | OCR functions (image text extraction) |
| 220-280 | `get_completion()` LLM wrapper |
| 280-400 | Token counting, text chunking |
| 400-600+ | Translation workflow (initial, reflect, improve) |

---

## Target Structure

```
src/translation_agent/
├── config/
│   ├── __init__.py
│   ├── settings.py        # Environment loading, Settings dataclass
│   └── providers.py       # ProviderConfig, registry, language mapping
├── llm/
│   ├── __init__.py
│   ├── client_factory.py  # OpenAI client creation
│   └── completion.py      # get_completion() wrapper
├── ocr/
│   ├── __init__.py
│   └── extractor.py       # Image text extraction functions
├── text/
│   ├── __init__.py
│   ├── tokenization.py    # Token counting utilities
│   └── chunking.py        # Text splitting/chunking
├── translation/
│   ├── __init__.py
│   ├── prompts.py         # All prompt templates
│   └── workflow.py        # Translation orchestration
└── utils.py               # Backward-compat re-exports (thin shim)
```

---

## Phase 1: Text Utilities (No Dependencies)

**Why first:** Pure utility functions with no external dependencies. Easy to extract and test.

### Phase 1.1: Create `text/tokenization.py`
- [ ] Create `src/translation_agent/text/` directory
- [ ] Create `src/translation_agent/text/__init__.py`
- [ ] Create `src/translation_agent/text/tokenization.py`
- [ ] Move `num_tokens_in_string()` function
- [ ] Move `MAX_TOKENS_PER_CHUNK` constant
- [ ] Add imports: `tiktoken`
- [ ] Test: Verify token counting works

**Functions to move:**
```python
MAX_TOKENS_PER_CHUNK = 1000
def num_tokens_in_string(string: str, encoding_name: str = "cl100k_base") -> int
```

### Phase 1.2: Create `text/chunking.py`
- [ ] Create `src/translation_agent/text/chunking.py`
- [ ] Move `calculate_chunk_size()` function
- [ ] Move text splitting logic (LangChain splitter wrapper)
- [ ] Import `MAX_TOKENS_PER_CHUNK` from `tokenization.py`
- [ ] Test: Verify chunk size calculation

**Functions to move:**
```python
def calculate_chunk_size(token_count: int, token_limit: int) -> int
# Any text splitting helpers using RecursiveCharacterTextSplitter
```

### Phase 1.3: Update `text/__init__.py`
- [ ] Re-export public API:
  ```python
  from .tokenization import num_tokens_in_string, MAX_TOKENS_PER_CHUNK
  from .chunking import calculate_chunk_size
  ```

---

## Phase 2: Configuration & Providers

**Why second:** Foundation for LLM and OCR modules. No runtime dependencies on other modules.

### Phase 2.1: Create `config/settings.py`
- [ ] Create `src/translation_agent/config/` directory
- [ ] Create `src/translation_agent/config/__init__.py`
- [ ] Create `src/translation_agent/config/settings.py`
- [ ] Create `Settings` dataclass with all API keys and URLs
- [ ] Move `load_dotenv()` call
- [ ] Add `load_settings()` function
- [ ] Test: Verify env loading works

**New code structure:**
```python
from dataclasses import dataclass
from typing import Optional
import os
from dotenv import load_dotenv

@dataclass
class Settings:
    # ZAI
    zai_api_key: Optional[str] = None
    zai_base_url: Optional[str] = None
    zai_model: str = "zai-default"
    
    # DeepSeek
    deepseek_api_key: Optional[str] = None
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    
    # ... other providers

def load_settings() -> Settings:
    load_dotenv()
    return Settings(
        zai_api_key=os.getenv("ZAI_API_KEY"),
        # ...
    )
```

### Phase 2.2: Create `config/providers.py`
- [ ] Create `src/translation_agent/config/providers.py`
- [ ] Create `ProviderConfig` dataclass
- [ ] Move `PROVIDERS` dict (transform to registry)
- [ ] Move `LANGUAGE_PROVIDER_MAP` dict
- [ ] Create `ProviderRegistry` class
- [ ] Add `get_provider_for_language()` function
- [ ] **Remove `_current_source_lang` global** — make explicit
- [ ] Test: Verify provider selection logic

**New code structure:**
```python
from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class ProviderConfig:
    name: str
    api_key: Optional[str]
    base_url: Optional[str]
    model: str

LANGUAGE_PROVIDER_MAP: Dict[str, str] = {
    "chinese": "deepseek",
    "中文": "deepseek",
    # ...
}

class ProviderRegistry:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._providers = self._build_providers()
    
    def get(self, name: str) -> ProviderConfig: ...
    def for_language(self, source_lang: str) -> ProviderConfig: ...
```

### Phase 2.3: Update `config/__init__.py`
- [ ] Re-export public API:
  ```python
  from .settings import Settings, load_settings
  from .providers import ProviderConfig, ProviderRegistry, LANGUAGE_PROVIDER_MAP
  ```

---

## Phase 3: LLM Client Layer

**Why third:** Depends on config module. Foundation for OCR and translation.

### Phase 3.1: Create `llm/client_factory.py`
- [ ] Create `src/translation_agent/llm/` directory
- [ ] Create `src/translation_agent/llm/__init__.py`
- [ ] Create `src/translation_agent/llm/client_factory.py`
- [ ] Move OpenAI client creation logic
- [ ] Create `create_client(provider: ProviderConfig) -> openai.OpenAI`
- [ ] Remove duplicate client creation code
- [ ] Test: Verify client creation

**New code structure:**
```python
import openai
from ..config import ProviderConfig

def create_client(provider: ProviderConfig) -> openai.OpenAI:
    return openai.OpenAI(
        api_key=provider.api_key,
        base_url=provider.base_url,
    )
```

### Phase 3.2: Create `llm/completion.py`
- [ ] Create `src/translation_agent/llm/completion.py`
- [ ] Move `get_completion()` function
- [ ] Update signature to accept `source_lang` explicitly (not global)
- [ ] Add proper error handling
- [ ] Remove `icecream` debug calls (use logging)
- [ ] Test: Verify completion works

**Updated signature:**
```python
def get_completion(
    prompt: str,
    system_message: str = "You are a helpful assistant.",
    *,
    source_lang: Optional[str] = None,  # Explicit, not global!
    model: Optional[str] = None,
    temperature: float = 0.3,
    json_mode: bool = False,
) -> Union[str, dict]:
```

### Phase 3.3: Update `llm/__init__.py`
- [ ] Re-export public API:
  ```python
  from .client_factory import create_client
  from .completion import get_completion
  ```

---

## Phase 4: OCR Module

**Why fourth:** Depends on LLM client. Self-contained feature.

### Phase 4.1: Create `ocr/extractor.py`
- [ ] Create `src/translation_agent/ocr/` directory
- [ ] Create `src/translation_agent/ocr/__init__.py`
- [ ] Create `src/translation_agent/ocr/extractor.py`
- [ ] Move `get_ocr_client()` (refactor to use ProviderRegistry)
- [ ] Move `extract_text_from_image()`
- [ ] Move `extract_text_from_image_url()`
- [ ] Extract helper: `encode_image_as_data_url()`
- [ ] Extract helper: `get_mime_type()`
- [ ] Test: Verify OCR extraction works

**Functions to move:**
```python
def get_mime_type(file_path: str) -> str
def encode_image_as_data_url(image_path: str) -> tuple[str, str]
def get_ocr_client() -> tuple[openai.OpenAI, str]
def extract_text_from_image(image_path: str, language_hint: str = "chinese") -> str
def extract_text_from_image_url(image_url: str, language_hint: str = "chinese") -> str
```

### Phase 4.2: Update `ocr/__init__.py`
- [ ] Re-export public API:
  ```python
  from .extractor import extract_text_from_image, extract_text_from_image_url
  ```

---

## Phase 5: Translation Module

**Why fifth:** Core business logic. Depends on all previous modules.

### Phase 5.1: Create `translation/prompts.py`
- [ ] Create `src/translation_agent/translation/` directory
- [ ] Create `src/translation_agent/translation/__init__.py`
- [ ] Create `src/translation_agent/translation/prompts.py`
- [ ] Extract `INITIAL_TRANSLATION_PROMPT` template
- [ ] Extract `REFLECTION_PROMPT` template
- [ ] Extract `IMPROVEMENT_PROMPT` template
- [ ] Create builder functions for each prompt
- [ ] Test: Verify prompt building

**New code structure:**
```python
INITIAL_TRANSLATION_PROMPT = """Your task is to provide a professional translation...
<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>
..."""

REFLECTION_PROMPT = """Your task is to carefully read a source text...
..."""

IMPROVEMENT_PROMPT = """Your task is to carefully read, then improve...
..."""

def build_initial_prompt(source_lang: str, target_lang: str, ...) -> str: ...
def build_reflection_prompt(...) -> str: ...
def build_improvement_prompt(...) -> str: ...
```

### Phase 5.2: Create `translation/workflow.py`
- [ ] Create `src/translation_agent/translation/workflow.py`
- [ ] Move `one_chunk_initial_translation()` (if exists)
- [ ] Move `one_chunk_reflect_on_translation()` (if exists)
- [ ] Move `one_chunk_improve_translation()` (if exists)
- [ ] Move `multichunk_initial_translation()`
- [ ] Move `multichunk_reflect_on_translation()`
- [ ] Move `multichunk_improve_translation()`
- [ ] Move `multichunk_translation()`
- [ ] Move `translate()` main entry point
- [ ] Import prompts from `prompts.py`
- [ ] Import chunking from `text/`
- [ ] Import completion from `llm/`
- [ ] Test: Verify full translation workflow

### Phase 5.3: Update `translation/__init__.py`
- [ ] Re-export public API:
  ```python
  from .workflow import translate, multichunk_translation
  from .prompts import (
      INITIAL_TRANSLATION_PROMPT,
      REFLECTION_PROMPT,
      IMPROVEMENT_PROMPT,
  )
  ```

---

## Phase 6: Backward Compatibility Layer

**Why last:** After all modules exist, create thin shim for existing imports.

### Phase 6.1: Update `utils.py` as Compatibility Shim
- [ ] Remove all implementations from `utils.py`
- [ ] Add re-exports from new modules:
  ```python
  # Backward compatibility - prefer importing from specific modules
  from translation_agent.config import load_settings, ProviderRegistry
  from translation_agent.llm import get_completion
  from translation_agent.ocr import extract_text_from_image, extract_text_from_image_url
  from translation_agent.text import num_tokens_in_string, calculate_chunk_size
  from translation_agent.translation import translate, multichunk_translation
  
  # Deprecated: will be removed in v2.0
  MAX_TOKENS_PER_CHUNK = 1000
  client = None  # Deprecated global
  ```
- [ ] Add deprecation warnings for globals
- [ ] Test: Verify existing imports still work

### Phase 6.2: Update Other Files
- [ ] Find all files importing from `utils.py`
- [ ] Update imports to use new module paths (optional but recommended)
- [ ] Update `__init__.py` in package root if needed

---

## Phase 7: Cleanup & Documentation

### Phase 7.1: Remove Dead Code
- [ ] Remove `_current_source_lang` global completely
- [ ] Remove any unused helper functions
- [ ] Remove `icecream` imports (replace with `logging`)

### Phase 7.2: Add Type Hints
- [ ] Add full type hints to all new modules
- [ ] Consider adding `py.typed` marker

### Phase 7.3: Add Module Docstrings
- [ ] Add docstrings to each `__init__.py`
- [ ] Add docstrings to each module explaining purpose

### Phase 7.4: Update Documentation
- [ ] Update README.md with new import paths
- [ ] Add migration guide for existing users

---

## Testing Strategy

### Unit Tests (Add as You Go)
```
tests/
├── test_text/
│   ├── test_tokenization.py
│   └── test_chunking.py
├── test_config/
│   ├── test_settings.py
│   └── test_providers.py
├── test_llm/
│   └── test_completion.py
├── test_ocr/
│   └── test_extractor.py
└── test_translation/
    ├── test_prompts.py
    └── test_workflow.py
```

### Smoke Test
- [ ] Run existing examples after each phase
- [ ] Verify translation output hasn't changed

---

## Execution Checklist

| Phase | Description | Status |
|-------|-------------|--------|
| 1.1 | `text/tokenization.py` | ⬜ |
| 1.2 | `text/chunking.py` | ⬜ |
| 1.3 | `text/__init__.py` | ⬜ |
| 2.1 | `config/settings.py` | ⬜ |
| 2.2 | `config/providers.py` | ⬜ |
| 2.3 | `config/__init__.py` | ⬜ |
| 3.1 | `llm/client_factory.py` | ⬜ |
| 3.2 | `llm/completion.py` | ⬜ |
| 3.3 | `llm/__init__.py` | ⬜ |
| 4.1 | `ocr/extractor.py` | ⬜ |
| 4.2 | `ocr/__init__.py` | ⬜ |
| 5.1 | `translation/prompts.py` | ⬜ |
| 5.2 | `translation/workflow.py` | ⬜ |
| 5.3 | `translation/__init__.py` | ⬜ |
| 6.1 | `utils.py` compat shim | ⬜ |
| 6.2 | Update other files | ⬜ |
| 7.1 | Remove dead code | ⬜ |
| 7.2 | Add type hints | ⬜ |
| 7.3 | Add docstrings | ⬜ |
| 7.4 | Update docs | ⬜ |

---

## Notes

- **Always run tests** after completing each sub-phase
- **Commit after each phase** to enable easy rollback
- **Keep `utils.py` working** until Phase 6 is complete
- Total estimated effort: 4-6 hours for experienced developer
