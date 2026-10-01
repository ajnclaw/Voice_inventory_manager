# config.py

import os
from pathlib import Path


def _load_dotenv(path):
    """
    Minimal, dependency-free .env loader: KEY=value per line, tolerant of
    surrounding whitespace around '=', blank lines, and '#' comments.
    Never overwrites a variable already set in the real environment.
    """
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


PROJECT_ROOT = Path.cwd()

INVENTORY_DB = PROJECT_ROOT / "inventory.db"

_load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# LLM provider selection -- same provider-agnostic setup as ~/ai-agent's
# llm_client.py: every provider here exposes an OpenAI-compatible
# /chat/completions endpoint, so switching is just picking a different
# base_url/api_key/model triple, never a code change. Set LLM_PROVIDER in
# .env to one of the keys below; override the model with LLM_MODEL.
# ---------------------------------------------------------------------------

LLM_PROVIDERS = {
    "ollama": {
        "base_url": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        "api_key": os.environ.get("OLLAMA_API_KEY", "ollama"),
        "default_model": "qwen2.5:1.5b",
    },
    "deepseek": {
        "base_url": os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        "api_key": os.environ.get("DEEPSEEK_API", ""),
        "default_model": "deepseek-chat",
    },
    "openrouter": {
        "base_url": os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        "api_key": os.environ.get("OPENROUTER_API", ""),
        "default_model": "openai/gpt-4o-mini",
    },
}

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openrouter")

if LLM_PROVIDER not in LLM_PROVIDERS:
    raise ValueError(
        f"Unknown LLM_PROVIDER {LLM_PROVIDER!r} in .env -- expected one of "
        f"{sorted(LLM_PROVIDERS)}"
    )

_active_provider = LLM_PROVIDERS[LLM_PROVIDER]

LLM_BASE_URL = _active_provider["base_url"]
LLM_API_KEY = _active_provider["api_key"]

DEFAULT_MODEL = os.environ.get("LLM_MODEL", _active_provider["default_model"])

DEFAULT_MAX_ITERATIONS = 10
