# AGENTS.md - Translation Agent

## Commands
- **Install**: `poetry install --with dev,test,app,eval`
- **Lint/Format**: `ruff check . --fix && ruff format .`
- **Type check**: `mypy src`
- **Run all tests**: `pytest tests/`
- **Run single test**: `pytest tests/test_agent.py::test_name -v`

## Architecture
```
src/translation_agent/
├── config/          # Settings (env loading) and ProviderRegistry
├── llm/             # OpenAI client factory and get_completion()
├── ocr/             # Image text extraction (DeepSeek VLM)
├── text/            # Tokenization and chunking utilities
├── translation/     # Prompts and workflow orchestration
└── utils.py         # Backward-compat shim (deprecated)
```

## Code Style
- Python 3.9+, Poetry for dependency management
- Formatter: Ruff (line-length=79, indent=4, double quotes)
- Linting: Ruff with isort, flake8-bugbear, pyupgrade, pep8-naming
- Type hints: Use modern syntax (`list[str]`, `str | None`)
- Imports: Absolute imports from `translation_agent.*`
