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

# "Sign in with Google" for the web UI, layered on top of the token above
# (which keeps working for scripts, curl and local dev). All three must be
# set for it to turn on -- an allowlist is mandatory, never "any Google
# account", since a public sign-in button with no allowlist would let
# anyone on the internet in. Fails closed: a missing piece disables Google
# sign-in rather than weakening it.
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
ALLOWED_GOOGLE_EMAILS = {
    email.strip().lower()
    for email in os.environ.get("ALLOWED_GOOGLE_EMAILS", "").split(",")
    if email.strip()
}
# Signs the session cookie. Generate with:
#   python3 -c "import secrets; print(secrets.token_urlsafe(48))"
SESSION_SECRET = os.environ.get("SESSION_SECRET", "")
SESSION_TTL_SECONDS = int(os.environ.get("SESSION_TTL_DAYS", "30")) * 24 * 3600
GOOGLE_AUTH_ENABLED = bool(
    GOOGLE_CLIENT_ID and ALLOWED_GOOGLE_EMAILS and SESSION_SECRET
)


def parse_email_set(raw):
    return {email.strip().lower() for email in (raw or "").split(",") if email.strip()}


def admin_emails(raw, allowed):
    """
    Who may open the admin dashboard: the emails in ADMIN_GOOGLE_EMAILS
    that are ALSO on the sign-in allowlist. An admin who can't sign in
    can't be an admin, so an address listed only as admin is dropped
    rather than trusted. Empty/unset means nobody is an admin (fails
    closed), and the shared bearer token is never an admin either --
    it is one secret held by a phone and scripts, not a person.
    """
    return parse_email_set(raw) & set(allowed)


ADMIN_GOOGLE_EMAILS = admin_emails(
    os.environ.get("ADMIN_GOOGLE_EMAILS", ""), ALLOWED_GOOGLE_EMAILS
)
ADMIN_ENABLED = GOOGLE_AUTH_ENABLED and bool(ADMIN_GOOGLE_EMAILS)

# A "session" is a run of one user's activity with no gap longer than
# this. The signed cookie itself lives for SESSION_TTL_DAYS, far too long
# to mean "one sitting", so sessions are tracked server-side (see
# agent/sessions.py).
SESSION_IDLE_SECONDS = int(os.environ.get("SESSION_IDLE_MINUTES", "30")) * 60

# Timezone the dashboard buckets days and hours in (the shop's, not UTC).
ADMIN_TIMEZONE = os.environ.get("ADMIN_TIMEZONE", "Asia/Kolkata")

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
