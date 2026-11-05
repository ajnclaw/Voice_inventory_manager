# monitoring.py
#
# System-health signals that don't belong in the owner-facing chat UI
# at all -- auth failures, aggregate request failure rate, and bulk-
# import extraction quality (how often a preview gets rejected rather
# than saved). These answer "is something systematically wrong" in a
# way individual request logs don't surface on their own, since
# noticing a pattern across hundreds of log entries by eye doesn't
# scale. Read back via agent_server.py's GET /stats, and, in much more
# detail and per user, by the admin dashboard (agent/admin_metrics.py).
#
# Every row that can be tied to a person carries user_email and, where
# there is one, session_id. Rows written before those columns existed
# read NULL there ("unattributed"); the columns are added to existing
# databases in place, never by rebuilding a table (see db_migrate.py).

import json
import sqlite3
from datetime import datetime, timezone

from .config import PROJECT_ROOT
from .db_migrate import ensure_columns

MONITORING_DB = PROJECT_ROOT / "monitoring.db"

# Stored text is clipped: this is an audit trail, not a copy of the
# conversation, and one runaway tool error must not bloat the file.
MAX_ARGUMENTS_CHARS = 500
MAX_ERROR_CHARS = 300


def _connect():
    conn = sqlite3.connect(MONITORING_DB)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS auth_failures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL,
            remote_addr TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS request_outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            endpoint TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS import_outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            outcome TEXT NOT NULL CHECK(outcome IN ('confirmed', 'cancelled')),
            item_count INTEGER,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            user_email TEXT NOT NULL,
            started_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            remote_addr TEXT,
            user_agent TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sign_ins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT,
            outcome TEXT NOT NULL,
            remote_addr TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tool_calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            user_email TEXT,
            session_id TEXT,
            endpoint TEXT,
            tool_name TEXT NOT NULL,
            success INTEGER NOT NULL,
            denied INTEGER NOT NULL DEFAULT 0,
            error TEXT,
            duration_seconds REAL,
            item TEXT,
            arguments TEXT,
            result_count INTEGER,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            user_email TEXT,
            session_id TEXT,
            rating INTEGER NOT NULL CHECK(rating IN (-1, 1)),
            comment TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(run_id, user_email)
        )
        """
    )
    ensure_columns(conn, "auth_failures", {"status_code": "INTEGER", "email": "TEXT"})
    ensure_columns(
        conn,
        "request_outcomes",
        {
            "user_email": "TEXT",
            "session_id": "TEXT",
            "run_id": "TEXT",
            "duration_seconds": "REAL",
        },
    )
    ensure_columns(conn, "import_outcomes", {"user_email": "TEXT", "session_id": "TEXT"})
    return conn


def _now():
    return datetime.now(timezone.utc).isoformat()


def _clip(text, limit):
    if text is None:
        return None

    text = str(text)

    return text if len(text) <= limit else text[: limit - 3] + "..."


def record_auth_failure(path, remote_addr=None, status_code=401, email=None):
    with _connect() as conn:
        conn.execute(
            "INSERT INTO auth_failures (path, remote_addr, created_at, status_code, email) "
            "VALUES (?, ?, ?, ?, ?)",
            (path, remote_addr, _now(), status_code, email),
        )


def record_request_outcome(
    endpoint, status, user_email=None, session_id=None, run_id=None, duration_seconds=None
):
    with _connect() as conn:
        conn.execute(
            "INSERT INTO request_outcomes "
            "(endpoint, status, created_at, user_email, session_id, run_id, duration_seconds) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (endpoint, status, _now(), user_email, session_id, run_id, duration_seconds),
        )


def record_import_outcome(outcome, item_count, user_email=None, session_id=None):
    """outcome: 'confirmed' or 'cancelled' -- the direct signal of
    whether a bulk-import extraction was actually good enough to use."""
    with _connect() as conn:
        conn.execute(
            "INSERT INTO import_outcomes (outcome, item_count, created_at, user_email, session_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (outcome, item_count, _now(), user_email, session_id),
        )


def record_sign_in(email, outcome, remote_addr=None):
    """outcome: 'ok', 'blocked' (verified Google account, not on the
    allowlist) or 'invalid' (the Google token itself didn't verify)."""
    with _connect() as conn:
        conn.execute(
            "INSERT INTO sign_ins (email, outcome, remote_addr, created_at) VALUES (?, ?, ?, ?)",
            (email, outcome, remote_addr, _now()),
        )


def record_tool_call(
    tool_name,
    success,
    run_id=None,
    user_email=None,
    session_id=None,
    endpoint=None,
    denied=False,
    error=None,
    duration_seconds=None,
    item=None,
    arguments=None,
    result_count=None,
):
    """
    One row per tool execution, whichever door it came through (a chat
    turn, the inventory table, a photo upload, an import) -- `endpoint`
    says which, so "edited by chat" and "edited from the table" can be
    told apart. result_count is the length of a list result, which is
    what makes a successful-but-empty search visible.
    """
    if arguments is not None and not isinstance(arguments, str):
        arguments = json.dumps(arguments, default=str, ensure_ascii=False)

    with _connect() as conn:
        conn.execute(
            "INSERT INTO tool_calls "
            "(run_id, user_email, session_id, endpoint, tool_name, success, denied, "
            "error, duration_seconds, item, arguments, result_count, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id, user_email, session_id, endpoint, tool_name,
                1 if success else 0, 1 if denied else 0,
                _clip(error, MAX_ERROR_CHARS), duration_seconds,
                _clip(item, 200), _clip(arguments, MAX_ARGUMENTS_CHARS),
                result_count, _now(),
            ),
        )


def record_feedback(run_id, rating, user_email=None, session_id=None, comment=None):
    """A thumbs up (1) / down (-1) on one reply. One per user per reply:
    tapping again replaces the earlier rating instead of stacking."""
    with _connect() as conn:
        conn.execute(
            "INSERT INTO feedback (run_id, user_email, session_id, rating, comment, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(run_id, user_email) DO UPDATE SET "
            "rating = excluded.rating, comment = excluded.comment, created_at = excluded.created_at",
            (run_id, user_email, session_id, rating, _clip(comment, 500), _now()),
        )


def stats(since_iso=None):
    with _connect() as conn:
        where = "WHERE created_at >= ?" if since_iso else ""
        params = (since_iso,) if since_iso else ()

        auth_failure_count = conn.execute(
            f"SELECT COUNT(*) as c FROM auth_failures {where}", params
        ).fetchone()["c"]

        recent_auth_failures = conn.execute(
            f"SELECT path, remote_addr, created_at FROM auth_failures {where} "
            f"ORDER BY created_at DESC LIMIT 20",
            params,
        ).fetchall()

        outcome_rows = conn.execute(
            f"SELECT endpoint, status, COUNT(*) as c FROM request_outcomes {where} "
            f"GROUP BY endpoint, status",
            params,
        ).fetchall()

        import_rows = conn.execute(
            f"SELECT outcome, COUNT(*) as c FROM import_outcomes {where} GROUP BY outcome",
            params,
        ).fetchall()

    by_endpoint = {}
    for row in outcome_rows:
        entry = by_endpoint.setdefault(row["endpoint"], {})
        entry[row["status"]] = row["c"]

    total_requests = sum(sum(v.values()) for v in by_endpoint.values())
    total_failed = sum(v.get("failed", 0) for v in by_endpoint.values())
    failure_rate = (total_failed / total_requests) if total_requests else None

    import_counts = {row["outcome"]: row["c"] for row in import_rows}
    total_imports = sum(import_counts.values())
    cancelled = import_counts.get("cancelled", 0)
    import_reject_rate = (cancelled / total_imports) if total_imports else None

    return {
        "auth_failures": {
            "total": auth_failure_count,
            "recent": [dict(row) for row in recent_auth_failures],
        },
        "requests": {
            "by_endpoint": by_endpoint,
            "total": total_requests,
            "total_failed": total_failed,
            "failure_rate": round(failure_rate, 4) if failure_rate is not None else None,
        },
        "bulk_import": {
            "by_outcome": import_counts,
            "total": total_imports,
            "reject_rate": round(import_reject_rate, 4) if import_reject_rate is not None else None,
        },
    }
