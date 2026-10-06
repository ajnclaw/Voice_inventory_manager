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

# Shared-secret auth for agent_server.py. Unlike ~/ai-agent's server
# (only ever reachable through an adb-reverse tunnel to one specific,
# physically-connected phone -- private by construction), this one is
# meant to sit behind a public EC2/ECS endpoint, reachable by anyone who
# finds the URL. There is no separate per-user login here -- one shared
# token for the one shop owner -- but agent_server.py refuses to even
# start without one set, so "forgot to configure auth" can't silently
# mean "wide open on the internet." See agent_server.py's check.
AGENT_API_TOKEN = os.environ.get("AGENT_API_TOKEN", "")

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
    "agentrouter": {
        "base_url": os.environ.get("AGENTROUTER_BASE_URL", "https://agentrouter.org/v1"),
        "api_key": os.environ.get("AGENTROUTER_API", ""),
        "default_model": "openai/gpt-4o-mini",
    },
    # Google's own OpenAI-compatibility endpoint -- same /chat/completions
    # shape as every other provider here, so no new SDK dependency (the
    # `google-genai` package the quickstart sample uses is a separate,
    # native client this codebase deliberately doesn't need).
    "gemini": {
        "base_url": os.environ.get(
            "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai"
        ),
        "api_key": os.environ.get("GEMINI_API_KEY", ""),
        "default_model": "gemini-flash-lite-latest",
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
