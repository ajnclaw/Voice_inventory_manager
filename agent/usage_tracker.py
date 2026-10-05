# usage_tracker.py
#
# Cumulative LLM usage/cost, persisted to SQLite rather than kept in a
# process-memory counter -- "how much have we spent so far" needs to
# survive a server restart (which this project has been through many
# times in one evening alone), not reset to zero every time.

import sqlite3
from datetime import datetime, timezone

from .config import PROJECT_ROOT

USAGE_DB = PROJECT_ROOT / "usage.db"


def _connect():
    conn = sqlite3.connect(USAGE_DB)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS llm_calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            component TEXT NOT NULL,
            model TEXT NOT NULL,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            cost_usd REAL,
            created_at TEXT NOT NULL
        )
        """
    )
    return conn


def record_call(run_id, component, model, prompt_tokens, completion_tokens, cost_usd):
    with _connect() as conn:
        conn.execute(
            "INSERT INTO llm_calls "
            "(run_id, component, model, prompt_tokens, completion_tokens, cost_usd, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                run_id, component, model, prompt_tokens, completion_tokens,
                cost_usd, datetime.now(timezone.utc).isoformat(),
            ),
        )


def totals(since_iso=None):
    with _connect() as conn:
        if since_iso:
            rows = conn.execute(
                "SELECT * FROM llm_calls WHERE created_at >= ?", (since_iso,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM llm_calls").fetchall()

    by_model = {}

    for row in rows:
        entry = by_model.setdefault(
            row["model"], {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}
        )
        entry["calls"] += 1
        entry["prompt_tokens"] += row["prompt_tokens"] or 0
        entry["completion_tokens"] += row["completion_tokens"] or 0
        entry["cost_usd"] += row["cost_usd"] or 0.0

    for entry in by_model.values():
        entry["cost_usd"] = round(entry["cost_usd"], 6)

    return {
        "total_calls": len(rows),
        "total_prompt_tokens": sum(r["prompt_tokens"] or 0 for r in rows),
        "total_completion_tokens": sum(r["completion_tokens"] or 0 for r in rows),
        "total_cost_usd": round(sum(r["cost_usd"] or 0.0 for r in rows), 6),
        "by_model": by_model,
    }
