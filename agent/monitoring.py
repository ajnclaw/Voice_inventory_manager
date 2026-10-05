# monitoring.py
#
# System-health signals that don't belong in the owner-facing chat UI
# at all -- auth failures, aggregate request failure rate, and bulk-
# import extraction quality (how often a preview gets rejected rather
# than saved). These answer "is something systematically wrong" in a
# way individual request logs don't surface on their own, since
# noticing a pattern across hundreds of log entries by eye doesn't
# scale. Read back via agent_server.py's GET /stats -- a backend-only
# endpoint, never called from web/index.html.

import sqlite3
from datetime import datetime, timezone

from .config import PROJECT_ROOT

MONITORING_DB = PROJECT_ROOT / "monitoring.db"


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
    return conn


def _now():
    return datetime.now(timezone.utc).isoformat()


def record_auth_failure(path, remote_addr=None):
    with _connect() as conn:
        conn.execute(
            "INSERT INTO auth_failures (path, remote_addr, created_at) VALUES (?, ?, ?)",
            (path, remote_addr, _now()),
        )


def record_request_outcome(endpoint, status):
    with _connect() as conn:
        conn.execute(
            "INSERT INTO request_outcomes (endpoint, status, created_at) VALUES (?, ?, ?)",
            (endpoint, status, _now()),
        )


def record_import_outcome(outcome, item_count):
    """outcome: 'confirmed' or 'cancelled' -- the direct signal of
    whether a bulk-import extraction was actually good enough to use."""
    with _connect() as conn:
        conn.execute(
            "INSERT INTO import_outcomes (outcome, item_count, created_at) VALUES (?, ?, ?)",
            (outcome, item_count, _now()),
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
