# sessions.py
#
# Who is using the app, and which "sitting" a request belongs to.
#
# A session is one user's run of activity with no gap longer than
# SESSION_IDLE_SECONDS (config.py). It is tracked server-side rather than
# in the signed cookie because the cookie lives for weeks -- far too long
# to mean "one sitting" -- and because the same logic then also covers the
# shared bearer token, which has no cookie at all.
#
# The server is single-threaded (see agent_server.py), so the
# read-then-write in touch() cannot race with itself.

import secrets
from datetime import datetime, timedelta, timezone

from . import monitoring
from .config import SESSION_IDLE_SECONDS

# The label recorded for requests authenticated by the shared AGENT_API_TOKEN
# (the phone app, scripts). They have no email; this stands in as one
# identity so their activity is still grouped rather than "unattributed".
BEARER_USER = "bearer-token"


def _parse(timestamp):
    return datetime.fromisoformat(timestamp)


def touch(user_email, now=None, remote_addr=None, user_agent=None, idle_seconds=None):
    """
    Returns the id of this user's current session, extending it, or
    starting a new one if their last activity was longer ago than the idle
    limit. `now` is injectable so the idle rule is testable.
    """
    now = now or datetime.now(timezone.utc)
    idle = timedelta(seconds=SESSION_IDLE_SECONDS if idle_seconds is None else idle_seconds)

    with monitoring._connect() as conn:
        row = conn.execute(
            "SELECT id, last_seen_at FROM sessions WHERE user_email = ? "
            "ORDER BY last_seen_at DESC LIMIT 1",
            (user_email,),
        ).fetchone()

        if row is not None and now - _parse(row["last_seen_at"]) <= idle:
            conn.execute(
                "UPDATE sessions SET last_seen_at = ? WHERE id = ?",
                (now.isoformat(), row["id"]),
            )
            return row["id"]

        session_id = "s_" + secrets.token_hex(6)
        conn.execute(
            "INSERT INTO sessions (id, user_email, started_at, last_seen_at, remote_addr, user_agent) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                session_id, user_email, now.isoformat(), now.isoformat(),
                remote_addr, (user_agent or "")[:200] or None,
            ),
        )

        return session_id
