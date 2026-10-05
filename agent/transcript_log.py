# transcript_log.py
#
# A single, continuous, append-only record of every actual conversation
# turn -- what the owner said, what came back, which model answered it,
# and what it cost. This is deliberately separate from logs/<run_id>.json
# (the detailed technical trace: every individual LLM call, every tool
# call, full request/response content for one /chat turn): that's built
# for debugging one request; this is built for "show me the whole
# conversation" or "what did we spend this week" without having to open
# dozens of per-request files and reconstruct it yourself.
#
# Plain JSONL (one JSON object per line) rather than a single JSON
# array -- appends are O(1) and never require reading/rewriting the
# whole file, and `tail -f transcripts.jsonl` or any line-oriented tool
# works on it directly.

import json
from datetime import datetime, timezone

from .config import PROJECT_ROOT

TRANSCRIPT_PATH = PROJECT_ROOT / "transcripts.jsonl"


def append_turn(user_input, reply, status, model, prompt_tokens, completion_tokens, cost_usd):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_input": user_input,
        "reply": reply,
        "status": status,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_usd": round(cost_usd, 6) if cost_usd is not None else None,
    }

    with open(TRANSCRIPT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def read_all():
    if not TRANSCRIPT_PATH.exists():
        return []

    entries = []

    with open(TRANSCRIPT_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                entries.append(json.loads(line))

    return entries
