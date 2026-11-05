# admin_metrics.py
#
# Everything the admin dashboard (web/admin.html) shows, computed from the
# data the app already records: transcripts.jsonl (every turn), usage.db
# (every LLM call), monitoring.db (sessions, tool calls, request outcomes,
# imports, sign-ins, feedback, auth failures) and, read-only, inventory.db
# (data health). Nothing here writes to any of them.
#
# Rows from before attribution existed have no user; they are kept and
# shown under "(unattributed)" rather than dropped. Sessions for turns that
# have no session id are inferred from gaps in the timeline, and flagged as
# inferred.
#
# Several metrics are HEURISTICS over free text (a reply "that says it
# couldn't find something", a message "that looks like a correction"). They
# are deliberately simple, listed in one place below, and meant to surface
# turns worth a human look -- not to be exact counts.

import hashlib
import json
import re
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime, timedelta, timezone

from . import config, inventory_db, monitoring, transcript_log, usage_tracker
from . import logger as run_logger_module
from .sessions import BEARER_USER

UNATTRIBUTED_KEY = "_unattributed"

# --- heuristics over free text ------------------------------------------

# Reply says it could not find / do what was asked (English + Hinglish).
NOT_FOUND_RE = re.compile(
    r"nahi\s+mil(a|i|e)\b|nahin\s+mil|not\s+found"
    r"|(couldn'?t|could\s+not|can'?t|cannot|unable\s+to|wasn'?t\s+able\s+to)"
    r"\s+(find|locate|do|finish|complete|record|update|process)"
    r"|no\s+(such\s+item|match|results?)|doesn'?t\s+exist|does\s+not\s+exist"
    r"|something\s+went\s+wrong|nahi\s+kar\s+(paya|pa)|samajh\s+nahi"
    r"|didn'?t\s+understand",
    re.I,
)

# Reply asks the user which item / what they meant.
CLARIFY_RE = re.compile(
    r"which\s+(one|item|product|of)|kaun\s?sa|kaun\s?si|kaunsa|kon\s?sa|konsa"
    r"|kis\s?(item|wala|cheez)|did\s+you\s+mean|do\s+you\s+mean|aapka\s+matlab"
    r"|(please|could\s+you)\s+(specify|clarify)|more\s+than\s+one|multiple\s+(items|matches)"
    r"|ek\s+se\s+(zyada|jyada)",
    re.I,
)

# The user's NEXT message backs out of or fixes what was just done.
CORRECTION_RE = re.compile(
    r"^\s*(no|nope|nahi|nahin|galat|wrong|arre|oops|sorry|undo|actually|wait)\b"
    r"|\bi\s+meant\b|\bmera\s+matlab\b|\bnot\s+that\s+one\b|\b(wo|woh|vo|ye|yeh)\s+nahi\b"
    r"|\bgalat\b|\bwrong\s+(item|one|number|quantity)\b|\bundo\b|\brevert\b"
    r"|\bcancel\s+(that|it|kar)\b|\bulta\b|\bcorrect\s+(it|that|kar)\b",
    re.I,
)

# How soon after a reply a follow-up counts as a correction of it.
CORRECTION_CHAT_WINDOW = timedelta(minutes=5)
CORRECTION_EDIT_WINDOW = timedelta(minutes=10)

# A turn is "slow" at or above the window's p95, but never below this.
SLOW_FLOOR_SECONDS = 8.0

# Tools that change stock or catalog data.
WRITE_TOOLS = {
    "record_sale", "record_restock", "adjust_stock", "add_item",
    "rename_item", "set_category", "set_price", "set_image",
}

# Flag key -> (label shown to the admin, weight in the "worst turns" score).
FLAGS = {
    "failed": ("Request failed", 5),
    "thumbs_down": ("Thumbs down", 4),
    "tool_error": ("Tool call failed", 3),
    "corrected": ("User corrected it", 3),
    "not_found": ("Said it couldn't find/do it", 2),
    "empty_search": ("Search found nothing", 2),
    "clarify": ("Asked which one", 1),
    "slow": ("Slow", 1),
}

MAX_TEXT = 2000
MAX_SESSIONS_PER_USER = 200
WORST_TURNS = 25


# --- small helpers --------------------------------------------------------


def _tz():
    from zoneinfo import ZoneInfo

    try:
        return ZoneInfo(config.ADMIN_TIMEZONE)
    except Exception:
        return timezone.utc


def _parse(timestamp):
    try:
        parsed = datetime.fromisoformat(timestamp)
    except (TypeError, ValueError):
        return None

    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _local(timestamp):
    parsed = _parse(timestamp)
    return parsed.astimezone(_tz()) if parsed else None


def _day(timestamp):
    local = _local(timestamp)
    return local.date().isoformat() if local else None


def _rate(part, whole):
    return round(part / whole, 4) if whole else None


def _round(value, digits=2):
    return round(value, digits) if value is not None else None


def _clip(text, limit=MAX_TEXT):
    text = text or ""
    return text if len(text) <= limit else text[: limit - 1] + "…"


def percentile(values, p):
    """Linear-interpolated percentile (p in 0..100); None for no data."""
    values = sorted(v for v in values if v is not None)

    if not values:
        return None

    if len(values) == 1:
        return values[0]

    position = (len(values) - 1) * p / 100
    low = int(position)
    high = min(low + 1, len(values) - 1)

    return values[low] + (values[high] - values[low]) * (position - low)


def _dist(values):
    values = [v for v in values if v is not None]

    return {
        "count": len(values),
        "p50": _round(percentile(values, 50)),
        "p95": _round(percentile(values, 95)),
        "avg": _round(sum(values) / len(values)) if values else None,
        "max": _round(max(values)) if values else None,
    }


def user_label(user):
    if user is None:
        return "(unattributed)"

    if user == BEARER_USER:
        return "Shared access token"

    return user


def user_key(user):
    return UNATTRIBUTED_KEY if user is None else user


def user_from_key(key):
    return None if key == UNATTRIBUTED_KEY else key


def _window(days, now):
    """(since_iso or None, start local date or None)."""
    if not days:
        return None, None

    since = now - timedelta(days=days)

    return since.isoformat(), since.astimezone(_tz()).date()


# --- loading --------------------------------------------------------------


class Data:
    """One window's worth of rows, already normalised. `subset(user)` gives
    the same shape for one person, so every section function works for both
    the whole-shop overview and a single user's page."""

    def __init__(self):
        self.turns = []        # chat + upload turns, oldest first
        self.llm = []          # llm_calls rows
        self.tools = []        # tool_calls rows
        self.requests = []     # request_outcomes rows
        self.imports = []      # import_outcomes rows
        self.sessions = []     # sessions-table rows (overlapping the window)
        self.feedback = []     # feedback rows
        self.sign_ins = []
        self.auth_failures = []
        self.first_seen = {}   # user -> earliest activity ever (all time)

    def subset(self, user):
        sub = Data()
        sub.turns = [t for t in self.turns if t["user"] == user]
        sub.llm = [r for r in self.llm if r["user_email"] == user]
        sub.tools = [r for r in self.tools if r["user_email"] == user]
        sub.requests = [r for r in self.requests if r["user_email"] == user]
        sub.imports = [r for r in self.imports if r["user_email"] == user]
        sub.sessions = [r for r in self.sessions if r["user_email"] == user]
        sub.feedback = [r for r in self.feedback if r["user_email"] == user]
        sub.sign_ins = [r for r in self.sign_ins if r["email"] == user]
        sub.first_seen = {user: self.first_seen.get(user)}

        return sub


def _read_turns():
    """transcripts.jsonl, tolerating a corrupt or half-written line."""
    path = transcript_log.TRANSCRIPT_PATH

    if not path.exists():
        return []

    turns = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                raw = json.loads(line)
            except ValueError:
                continue

            if not isinstance(raw, dict) or not raw.get("timestamp"):
                continue

            user_input = raw.get("user_input") or ""

            turns.append({
                "ts": raw["timestamp"],
                "user": raw.get("user") or None,
                "session_id": raw.get("session_id"),
                "run_id": raw.get("run_id"),
                "kind": raw.get("kind")
                or ("upload" if user_input.startswith("[upload:") else "chat"),
                "input": user_input,
                "reply": raw.get("reply") or "",
                "status": raw.get("status"),
                "model": raw.get("model"),
                "prompt_tokens": raw.get("prompt_tokens") or 0,
                "completion_tokens": raw.get("completion_tokens") or 0,
                "cost_usd": raw.get("cost_usd"),
                "latency": raw.get("latency_seconds"),
                "input_mode": raw.get("input_mode"),
            })

    turns.sort(key=lambda t: t["ts"])

    return turns


def _query(conn, sql, params=()):
    return [dict(row) for row in conn.execute(sql, params)]


def load(days, now):
    since, _ = _window(days, now)
    where = "WHERE created_at >= ?" if since else ""
    params = (since,) if since else ()
    data = Data()

    all_turns = _read_turns()
    data.turns = [t for t in all_turns if since is None or t["ts"] >= since]

    with closing(usage_tracker._connect()) as conn:
        data.llm = _query(
            conn, f"SELECT * FROM llm_calls {where} ORDER BY created_at", params
        )

    with closing(monitoring._connect()) as conn:
        data.tools = _query(
            conn, f"SELECT * FROM tool_calls {where} ORDER BY id", params
        )
        data.requests = _query(
            conn, f"SELECT * FROM request_outcomes {where} ORDER BY id", params
        )
        data.imports = _query(
            conn, f"SELECT * FROM import_outcomes {where} ORDER BY id", params
        )
        data.feedback = _query(
            conn, f"SELECT * FROM feedback {where} ORDER BY id", params
        )
        data.sign_ins = _query(
            conn, f"SELECT * FROM sign_ins {where} ORDER BY id", params
        )
        data.auth_failures = _query(
            conn, f"SELECT * FROM auth_failures {where} ORDER BY id", params
        )
        data.sessions = _query(
            conn,
            "SELECT * FROM sessions "
            + ("WHERE last_seen_at >= ? " if since else "")
            + "ORDER BY started_at",
            params,
        )

        # Earliest activity per user, ever -- decides "new vs returning".
        for row in conn.execute(
            "SELECT user_email, MIN(started_at) AS first FROM sessions GROUP BY user_email"
        ):
            data.first_seen[row["user_email"]] = row["first"]

    for turn in all_turns:
        known = data.first_seen.get(turn["user"])

        if known is None or turn["ts"] < known:
            data.first_seen[turn["user"]] = turn["ts"]

    return data


# --- per-turn flags -------------------------------------------------------


def _tools_by_run(tool_rows):
    by_run = defaultdict(list)

    for row in tool_rows:
        if row["run_id"]:
            by_run[row["run_id"]].append(row)

    return by_run


def _detect_corrections(turns, tool_rows):
    """
    Sets turn["corrected"] to a short reason for each chat turn the user
    appears to have fixed straight afterwards:
      - their next message (within minutes) reads like "no, I meant ...";
      - their next turn adjusts stock on an item this turn just sold/restocked;
      - they edited that item's stock or price in the table within minutes.
    """
    by_user = defaultdict(list)

    for turn in turns:
        if turn["kind"] == "chat":
            by_user[turn["user"]].append(turn)

    tools_by_run = _tools_by_run(tool_rows)

    for user_turns in by_user.values():
        for previous, following in zip(user_turns, user_turns[1:]):
            gap = _parse(following["ts"]) - _parse(previous["ts"])

            if gap > CORRECTION_CHAT_WINDOW:
                continue

            if CORRECTION_RE.search(following["input"]):
                previous["corrected"] = "next message reads like a correction"
                continue

            sold = {
                (r["item"] or "").lower()
                for r in tools_by_run.get(previous["run_id"], [])
                if r["tool_name"] in ("record_sale", "record_restock") and r["success"]
            }
            adjusted = {
                (r["item"] or "").lower()
                for r in tools_by_run.get(following["run_id"], [])
                if r["tool_name"] == "adjust_stock" and r["success"]
            }

            if sold & adjusted:
                previous["corrected"] = "stock adjusted right after"

    table_edits = [
        r for r in tool_rows
        if (r["endpoint"] or "") in ("table_adjust_stock", "table_set_price") and r["success"]
    ]

    for edit in table_edits:
        edit_at = _parse(edit["created_at"])
        item = (edit["item"] or "").lower()

        if not item or edit_at is None:
            continue

        for turn in reversed(by_user.get(edit["user_email"], [])):
            turn_at = _parse(turn["ts"])

            if turn_at > edit_at:
                continue

            if edit_at - turn_at > CORRECTION_EDIT_WINDOW:
                break

            items_touched = " ".join(
                (r["item"] or "") for r in tools_by_run.get(turn["run_id"], [])
            )
            blob = f"{turn['input']} {turn['reply']} {items_touched}".lower()

            if item in blob:
                turn["corrected"] = "edited in the table right after"
                break


def annotate(data):
    """Adds flags/score/tools/rating to every turn in `data`. Run once on
    the full window, before `subset`, so per-user pages see the same flags
    (and the same 'slow' threshold) as the overview."""
    tools_by_run = _tools_by_run(data.tools)

    ratings = {}
    for row in data.feedback:
        ratings.setdefault(row["run_id"], []).append(row["rating"])

    latencies = [t["latency"] for t in data.turns if t["kind"] == "chat" and t["latency"]]
    p95 = percentile(latencies, 95)
    slow_at = max(SLOW_FLOOR_SECONDS, p95) if p95 is not None else None

    for turn in data.turns:
        turn.pop("corrected", None)

    _detect_corrections(data.turns, data.tools)

    for turn in data.turns:
        run_tools = tools_by_run.get(turn["run_id"], [])
        rating = ratings.get(turn["run_id"])
        flags = []

        if turn["status"] == "failed":
            flags.append("failed")

        if rating and min(rating) < 0:
            flags.append("thumbs_down")

        if any(not r["success"] for r in run_tools):
            flags.append("tool_error")

        if turn.get("corrected"):
            flags.append("corrected")

        if turn["kind"] == "chat":
            if NOT_FOUND_RE.search(turn["reply"]):
                flags.append("not_found")

            if any(r["tool_name"] == "search_items" and r["result_count"] == 0 for r in run_tools):
                flags.append("empty_search")

            if CLARIFY_RE.search(turn["reply"]):
                flags.append("clarify")

        if slow_at is not None and (turn["latency"] or 0) >= slow_at:
            flags.append("slow")

        turn["flags"] = flags
        turn["score"] = sum(FLAGS[f][1] for f in flags)
        turn["tools"] = run_tools
        turn["rating"] = (
            None if not rating else (-1 if min(rating) < 0 else 1)
        )


# --- sessions -------------------------------------------------------------


def build_sessions(data, idle_seconds=None):
    """
    Recorded sessions (sessions table) with their turns attached, plus
    inferred ones for turns that carry no session id (history from before
    sessions existed): consecutive turns of one user with no gap longer
    than the idle limit form one inferred session.
    """
    idle = timedelta(seconds=config.SESSION_IDLE_SECONDS if idle_seconds is None else idle_seconds)
    sessions = {}

    for row in data.sessions:
        sessions[row["id"]] = {
            "id": row["id"],
            "user": row["user_email"],
            "started_at": row["started_at"],
            "last_seen_at": row["last_seen_at"],
            "inferred": False,
            "turns": [],
        }

    loose = defaultdict(list)

    for turn in data.turns:
        sid = turn["session_id"]

        if sid:
            session = sessions.setdefault(sid, {
                "id": sid, "user": turn["user"], "started_at": turn["ts"],
                "last_seen_at": turn["ts"], "inferred": False, "turns": [],
            })
            session["turns"].append(turn)
        else:
            loose[turn["user"]].append(turn)

    for user, user_turns in loose.items():
        current = None

        for turn in user_turns:
            if current and _parse(turn["ts"]) - _parse(current["last_seen_at"]) <= idle:
                current["turns"].append(turn)
                current["last_seen_at"] = turn["ts"]
                continue

            digest = hashlib.sha1(f"{user}|{turn['ts']}".encode()).hexdigest()[:8]
            current = {
                "id": f"i_{digest}", "user": user, "started_at": turn["ts"],
                "last_seen_at": turn["ts"], "inferred": True, "turns": [turn],
            }
            sessions[current["id"]] = current

    for session in sessions.values():
        stamps = [t["ts"] for t in session["turns"]] + [session["started_at"], session["last_seen_at"]]
        session["started_at"] = min(stamps)
        session["last_seen_at"] = max(stamps)

    return sorted(sessions.values(), key=lambda s: s["started_at"])


# --- sections -------------------------------------------------------------


def _quality(data):
    chat = [t for t in data.turns if t["kind"] == "chat"]
    failed = [t for t in data.turns if t["status"] == "failed"]
    flagged = Counter(f for t in data.turns for f in t["flags"])

    by_tool = defaultdict(lambda: {"calls": 0, "failed": 0, "denied": 0, "errors": Counter()})

    for row in data.tools:
        entry = by_tool[row["tool_name"]]
        entry["calls"] += 1

        if not row["success"]:
            entry["failed"] += 1
            entry["errors"][row["error"] or "(no message)"] += 1

        if row["denied"]:
            entry["denied"] += 1

    tool_rows = [
        {
            "tool": name,
            "calls": entry["calls"],
            "failed": entry["failed"],
            "denied": entry["denied"],
            "failure_rate": _rate(entry["failed"], entry["calls"]),
            "top_error": entry["errors"].most_common(1)[0][0] if entry["errors"] else None,
        }
        for name, entry in by_tool.items()
    ]
    tool_rows.sort(key=lambda r: (-r["failed"], -r["calls"]))

    total_tool_calls = sum(r["calls"] for r in tool_rows)
    total_tool_failed = sum(r["failed"] for r in tool_rows)

    request_stats = defaultdict(Counter)

    for row in data.requests:
        request_stats[row["endpoint"]][row["status"]] += 1

    requests_by_endpoint = [
        {
            "endpoint": endpoint,
            "total": sum(counts.values()),
            "failed": counts["failed"],
            "rejected": counts["rejected"],
            "failure_rate": _rate(counts["failed"], sum(counts.values())),
        }
        for endpoint, counts in request_stats.items()
    ]
    requests_by_endpoint.sort(key=lambda r: -r["total"])
    request_total = sum(r["total"] for r in requests_by_endpoint)
    request_failed = sum(r["failed"] for r in requests_by_endpoint)

    searched = Counter(
        (r["item"] or "").strip().lower()
        for r in data.tools
        if r["tool_name"] == "search_items" and r["result_count"] == 0 and r["item"]
    )

    ups = sum(1 for r in data.feedback if r["rating"] > 0)
    downs = sum(1 for r in data.feedback if r["rating"] < 0)

    corrected = [t for t in chat if "corrected" in t["flags"]]

    # Highest score first; among equals, the most recent first.
    worst = sorted(
        sorted((t for t in data.turns if t["score"] > 0), key=lambda t: t["ts"], reverse=True),
        key=lambda t: -t["score"],
    )[:WORST_TURNS]

    return {
        "chat_turns": len(chat),
        "failed_turns": len(failed),
        "failure_rate": _rate(len(failed), len(data.turns)),
        "request_failure_rate": _rate(request_failed, request_total),
        "requests_by_endpoint": requests_by_endpoint,
        "not_found": {
            "count": flagged["not_found"],
            "rate": _rate(flagged["not_found"], len(chat)),
        },
        "clarification": {
            "count": flagged["clarify"],
            "rate": _rate(flagged["clarify"], len(chat)),
        },
        "corrections": {
            "count": len(corrected),
            "rate": _rate(len(corrected), len(chat)),
            "reasons": dict(Counter(t["corrected"] for t in corrected)),
        },
        "empty_searches": {
            "count": sum(searched.values()),
            "top_queries": [{"query": q, "count": c} for q, c in searched.most_common(10)],
        },
        "tool_failures": {
            "calls": total_tool_calls,
            "failed": total_tool_failed,
            "rate": _rate(total_tool_failed, total_tool_calls),
            "by_tool": tool_rows,
        },
        "feedback": {
            "up": ups,
            "down": downs,
            "satisfaction": _rate(ups, ups + downs),
            "coverage": _rate(ups + downs, len(chat)),
        },
        "flag_counts": {
            key: {"label": FLAGS[key][0], "count": flagged[key]} for key in FLAGS
        },
        "worst_turns": [_turn_summary(t) for t in worst],
    }


def _turn_summary(turn, include_tools=False):
    summary = {
        "ts": turn["ts"],
        "run_id": turn["run_id"],
        "user": turn["user"],
        "user_key": user_key(turn["user"]),
        "user_label": user_label(turn["user"]),
        "session_id": turn["session_id"],
        "kind": turn["kind"],
        "input": _clip(turn["input"]),
        "reply": _clip(turn["reply"]),
        "status": turn["status"],
        "prompt_tokens": turn["prompt_tokens"],
        "completion_tokens": turn["completion_tokens"],
        "cost_usd": turn["cost_usd"],
        "latency": turn["latency"],
        "input_mode": turn["input_mode"],
        "flags": [{"key": f, "label": FLAGS[f][0]} for f in turn["flags"]],
        "score": turn["score"],
        "rating": turn["rating"],
        "corrected_reason": turn.get("corrected"),
    }

    if include_tools:
        summary["tools"] = [
            {
                "name": r["tool_name"],
                "ok": bool(r["success"]),
                "item": r["item"],
                "error": r["error"],
                "seconds": r["duration_seconds"],
                "results": r["result_count"],
                "endpoint": r["endpoint"],
            }
            for r in turn["tools"]
        ]
    else:
        summary["tool_names"] = [r["tool_name"] for r in turn["tools"]]

    return summary


def _daily_series(data, start_date, today):
    """Local-date buckets from start_date to today with zeros filled in, so
    a quiet day is a visible gap rather than a missing point."""
    buckets = {}

    def bucket(day):
        return buckets.setdefault(day, {
            "date": day, "cost_usd": 0.0, "calls": 0, "prompt_tokens": 0,
            "completion_tokens": 0, "turns": 0, "users": set(),
        })

    for row in data.llm:
        day = _day(row["created_at"])
        if day:
            b = bucket(day)
            b["cost_usd"] += row["cost_usd"] or 0.0
            b["calls"] += 1
            b["prompt_tokens"] += row["prompt_tokens"] or 0
            b["completion_tokens"] += row["completion_tokens"] or 0

    for turn in data.turns:
        day = _day(turn["ts"])
        if day:
            b = bucket(day)
            b["turns"] += 1
            b["users"].add(turn["user"])

    if not buckets and start_date is None:
        return []

    first = start_date or min(datetime.fromisoformat(d).date() for d in buckets)
    series = []
    day = first

    while day <= today:
        b = bucket(day.isoformat())
        series.append({
            "date": b["date"],
            "cost_usd": round(b["cost_usd"], 6),
            "calls": b["calls"],
            "prompt_tokens": b["prompt_tokens"],
            "completion_tokens": b["completion_tokens"],
            "turns": b["turns"],
            "users": len(b["users"]),
        })
        day += timedelta(days=1)

    return series


def _group_cost(rows, key_fn, label_fn=lambda k: k):
    groups = defaultdict(lambda: {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0})

    for row in rows:
        g = groups[key_fn(row)]
        g["calls"] += 1
        g["prompt_tokens"] += row["prompt_tokens"] or 0
        g["completion_tokens"] += row["completion_tokens"] or 0
        g["cost_usd"] += row["cost_usd"] or 0.0

    total = sum(g["cost_usd"] for g in groups.values())

    out = [
        {
            "key": key,
            "label": label_fn(key),
            **g,
            "cost_usd": round(g["cost_usd"], 6),
            "share": _rate(g["cost_usd"], total),
        }
        for key, g in groups.items()
    ]
    out.sort(key=lambda g: -g["cost_usd"])

    return out


def _cost(data, sessions, start_date, today, now):
    total = sum(r["cost_usd"] or 0.0 for r in data.llm)
    prompt = sum(r["prompt_tokens"] or 0 for r in data.llm)
    completion = sum(r["completion_tokens"] or 0 for r in data.llm)
    series = _daily_series(data, start_date, today)

    last7 = series[-7:]
    prev7 = series[-14:-7]
    last7_avg = sum(d["cost_usd"] for d in last7) / len(last7) if last7 else None
    prev7_avg = sum(d["cost_usd"] for d in prev7) / len(prev7) if prev7 else None
    change = (
        (last7_avg - prev7_avg) / prev7_avg
        if last7_avg is not None and prev7_avg
        else None
    )

    month_start = now.astimezone(_tz()).date().replace(day=1).isoformat()
    month_to_date = sum(d["cost_usd"] for d in series if d["date"] >= month_start)

    chat_turns = [t for t in data.turns if t["kind"] == "chat"]
    sessions_with_turns = [s for s in sessions if s["turns"]]

    return {
        "total_usd": round(total, 6),
        "calls": len(data.llm),
        "unpriced_calls": sum(1 for r in data.llm if r["cost_usd"] is None),
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "tokens_total": prompt + completion,
        "output_share": _rate(completion, prompt + completion),
        "avg_per_turn_usd": _round(total / len(chat_turns), 6) if chat_turns else None,
        "avg_per_session_usd": (
            _round(sum(t["cost_usd"] or 0 for s in sessions_with_turns for t in s["turns"])
                   / len(sessions_with_turns), 6)
            if sessions_with_turns else None
        ),
        "by_day": series,
        "by_user": _group_cost(
            data.llm, lambda r: user_key(r["user_email"]),
            lambda k: user_label(user_from_key(k)),
        ),
        "by_component": _group_cost(data.llm, lambda r: r["component"]),
        "by_model": _group_cost(data.llm, lambda r: r["model"]),
        "trend": {
            "last7_avg_daily_usd": _round(last7_avg, 6),
            "prev7_avg_daily_usd": _round(prev7_avg, 6),
            "change": _round(change, 4),
        },
        "projection": {
            "avg_daily_usd": _round(last7_avg, 6),
            "monthly_usd": _round(last7_avg * 30, 4) if last7_avg is not None else None,
            "basis_days": len(last7),
            "month_to_date_usd": round(month_to_date, 6),
        },
    }


def _usage(data, sessions, known_users, today):
    users_active = {t["user"] for t in data.turns} | {s["user"] for s in sessions}

    active_days = defaultdict(set)      # user -> local days active
    day_users = defaultdict(set)
    day_sessions = Counter()
    week_users = defaultdict(set)

    def mark(user, timestamp):
        day = _day(timestamp)

        if day is None:
            return

        active_days[user].add(day)
        day_users[day].add(user)
        monday = datetime.fromisoformat(day).date()
        monday -= timedelta(days=monday.weekday())
        week_users[monday.isoformat()].add(user)

    for turn in data.turns:
        mark(turn["user"], turn["ts"])

    for session in sessions:
        mark(session["user"], session["started_at"])
        mark(session["user"], session["last_seen_at"])
        day = _day(session["started_at"])

        if day:
            day_sessions[day] += 1

    turn_days = Counter(_day(t["ts"]) for t in data.turns)
    by_day = [
        {"date": day, "users": len(day_users[day]), "turns": turn_days.get(day, 0),
         "sessions": day_sessions.get(day, 0)}
        for day in sorted(set(day_users) | set(turn_days) - {None})
    ]

    hours = Counter()
    weekdays = Counter()

    for turn in data.turns:
        local = _local(turn["ts"])

        if local:
            hours[local.hour] += 1
            weekdays[local.weekday()] += 1

    sessions_with_turns = [s for s in sessions if s["turns"]]
    per_session = [len(s["turns"]) for s in sessions_with_turns]
    minutes = [
        (_parse(s["last_seen_at"]) - _parse(s["started_at"])).total_seconds() / 60
        for s in sessions_with_turns
    ]

    returning = [u for u, days in active_days.items() if len(days) >= 2]
    window_start = min((d for days in active_days.values() for d in days), default=None)
    new_users = [
        u for u in users_active
        if data.first_seen.get(u) and window_start
        and (_day(data.first_seen[u]) or "") >= window_start
    ]

    return {
        "known_users": known_users,
        "active_users": len(users_active),
        "sessions": len(sessions),
        "sessions_with_turns": len(sessions_with_turns),
        "inferred_sessions": sum(1 for s in sessions if s["inferred"]),
        "turns": len(data.turns),
        "chat_turns": sum(1 for t in data.turns if t["kind"] == "chat"),
        "upload_turns": sum(1 for t in data.turns if t["kind"] == "upload"),
        "turns_without_session": sum(1 for t in data.turns if not t["session_id"]),
        "turns_per_session": {
            "avg": _round(sum(per_session) / len(per_session)) if per_session else None,
            "median": _round(percentile(per_session, 50)),
            "max": max(per_session) if per_session else None,
        },
        "session_minutes": {
            "avg": _round(sum(minutes) / len(minutes)) if minutes else None,
            "median": _round(percentile(minutes, 50)),
        },
        "returning_users": {
            "count": len(returning),
            "rate": _rate(len(returning), len(users_active)),
            "definition": "active on 2 or more different days in this window",
        },
        "new_users": len(new_users),
        "active_by_day": by_day,
        "active_by_week": [
            {"week_start": week, "users": len(users)} for week, users in sorted(week_users.items())
        ],
        "peak_hours": [{"hour": h, "turns": hours.get(h, 0)} for h in range(24)],
        "peak_weekdays": [{"weekday": d, "turns": weekdays.get(d, 0)} for d in range(7)],
    }


def _speed(data):
    chat_latencies = [t["latency"] for t in data.turns if t["kind"] == "chat" and t["latency"] is not None]

    by_endpoint = defaultdict(list)
    for row in data.requests:
        if row["duration_seconds"] is not None:
            by_endpoint[row["endpoint"]].append(row["duration_seconds"])

    llm_stage = defaultdict(list)
    for row in data.llm:
        if row["duration_seconds"] is not None:
            llm_stage[row["component"]].append(row["duration_seconds"])

    tool_stage = defaultdict(list)
    for row in data.tools:
        if row["duration_seconds"] is not None:
            tool_stage[row["tool_name"]].append(row["duration_seconds"])

    slowest = sorted(
        (t for t in data.turns if t["latency"] is not None),
        key=lambda t: -t["latency"],
    )[:10]

    return {
        "chat_latency": _dist(chat_latencies),
        "by_endpoint": sorted(
            ({"endpoint": e, **_dist(v)} for e, v in by_endpoint.items()),
            key=lambda r: -r["count"],
        ),
        "llm_stages": sorted(
            ({"stage": s, **_dist(v)} for s, v in llm_stage.items()), key=lambda r: -r["count"]
        ),
        "tool_stages": sorted(
            ({"stage": s, **_dist(v)} for s, v in tool_stage.items()), key=lambda r: -r["count"]
        ),
        "slowest": [_turn_summary(t) for t in slowest],
        "coverage": {
            "turns_with_latency": len(chat_latencies),
            "chat_turns": sum(1 for t in data.turns if t["kind"] == "chat"),
        },
    }


def _adoption(data):
    outcomes = Counter((r["endpoint"], r["status"]) for r in data.requests)
    confirmed = [r for r in data.imports if r["outcome"] == "confirmed"]
    cancelled = [r for r in data.imports if r["outcome"] == "cancelled"]

    def successful(endpoint_prefix, tool):
        return sum(
            1 for r in data.tools
            if r["success"] and r["tool_name"] == tool
            and (r["endpoint"] or "").startswith(endpoint_prefix)
        )

    modes = Counter(t["input_mode"] or "unknown" for t in data.turns if t["kind"] == "chat")

    tools_used = Counter(r["tool_name"] for r in data.tools)
    tool_ok = Counter(r["tool_name"] for r in data.tools if r["success"])

    return {
        "imports": {
            "confirmed": len(confirmed),
            "cancelled": len(cancelled),
            "reject_rate": _rate(len(cancelled), len(confirmed) + len(cancelled)),
            "items_confirmed": sum(r["item_count"] or 0 for r in confirmed),
            "uploads": outcomes[("upload", "completed")] + outcomes[("upload", "failed")],
            "uploads_failed": outcomes[("upload", "failed")],
        },
        "photo_uploads": {
            "completed": outcomes[("inventory_set_image", "completed")],
            "failed": outcomes[("inventory_set_image", "failed")],
        },
        "edits": {
            "stock_from_table": successful("table_", "adjust_stock"),
            "price_from_table": successful("table_", "set_price"),
            "stock_from_chat": successful("chat", "adjust_stock"),
            "price_from_chat": successful("chat", "set_price"),
            "sales_from_chat": successful("chat", "record_sale"),
            "restocks_from_chat": successful("chat", "record_restock"),
        },
        "new_items": {
            "accepted": outcomes[("item_confirm_add", "completed")],
            "failed": outcomes[("item_confirm_add", "failed")],
            "rejected": outcomes[("item_reject_add", "rejected")],
        },
        "input_mode": {
            "voice": modes["voice"],
            "text": modes["text"],
            "unknown": modes["unknown"],
        },
        "tools_used": [
            {"tool": tool, "calls": n, "success_rate": _rate(tool_ok[tool], n)}
            for tool, n in tools_used.most_common()
        ],
    }


def _data_health(data, days, now):
    path = inventory_db.INVENTORY_DB

    if not path.exists():
        return {"available": False}

    since, _ = _window(days, now)

    try:
        with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
            conn.row_factory = sqlite3.Row
            columns = {r[1] for r in conn.execute("PRAGMA table_info(items)")}
            items = _query(conn, "SELECT * FROM items ORDER BY name")
            ledger = _query(
                conn,
                "SELECT t.*, i.name AS item_name FROM transactions t "
                "LEFT JOIN items i ON i.id = t.item_id ORDER BY t.created_at",
            )
    except sqlite3.Error:
        return {"available": False}

    def blank(item, column):
        value = item.get(column)
        return column not in columns or value is None or (isinstance(value, str) and not value.strip())

    def missing(column):
        names = [i["name"] for i in items if blank(i, column)]

        return {"count": len(names), "pct": _rate(len(names), len(items)), "examples": names[:10]}

    low = [
        i for i in items
        if i.get("reorder_threshold") is not None
        and (i.get("current_quantity") or 0) <= i["reorder_threshold"]
    ]

    in_window = [t for t in ledger if since is None or t["created_at"] >= since]
    by_type = Counter(t["type"] for t in in_window)
    by_day = defaultdict(Counter)

    for t in in_window:
        day = _day(t["created_at"])
        if day:
            by_day[day][t["type"]] += 1

    cutoff = (now - timedelta(days=90)).isoformat()
    recently_moved = {t["item_id"] for t in ledger if t["created_at"] >= cutoff}

    complete = [
        i for i in items
        if not blank(i, "sale_price") and not blank(i, "image_path") and not blank(i, "category")
    ]

    return {
        "available": True,
        "items": len(items),
        "missing": {
            "sale_price": missing("sale_price"),
            "cost_price": missing("cost_price"),
            "photo": missing("image_path"),
            "category": missing("category"),
            "reorder_threshold": missing("reorder_threshold"),
        },
        "complete_pct": _rate(len(complete), len(items)),
        "stock": {
            "low": len(low),
            "out_of_stock": sum(1 for i in items if (i.get("current_quantity") or 0) <= 0),
            "negative": sum(1 for i in items if (i.get("current_quantity") or 0) < 0),
            "total_units": sum(i.get("current_quantity") or 0 for i in items),
            "low_items": [
                {"name": i["name"], "quantity": i["current_quantity"], "threshold": i["reorder_threshold"]}
                for i in sorted(low, key=lambda i: i["current_quantity"] or 0)[:15]
            ],
        },
        "ledger": {
            "transactions": len(in_window),
            "by_type": dict(by_type),
            "adjustments": by_type["adjustment"],
            "last_at": ledger[-1]["created_at"] if ledger else None,
            "by_day": [
                {"date": d, "sale": c["sale"], "restock": c["restock"], "adjustment": c["adjustment"]}
                for d, c in sorted(by_day.items())
            ],
            "stale_items_90d": sum(1 for i in items if i["id"] not in recently_moved),
        },
    }


def _security(data):
    failures = data.auth_failures
    status_of = lambda r: r.get("status_code") or 401  # noqa: E731 -- legacy rows are all 401s

    by_ip = defaultdict(lambda: {"count": 0, "last": None})
    by_path = Counter()
    by_status = Counter()
    by_day = Counter()

    for row in failures:
        entry = by_ip[row["remote_addr"] or "unknown"]
        entry["count"] += 1
        entry["last"] = max(filter(None, [entry["last"], row["created_at"]]))
        by_path[row["path"]] += 1
        by_status[str(status_of(row))] += 1
        day = _day(row["created_at"])
        if day:
            by_day[day] += 1

    ips = sorted(
        ({"ip": ip, **v} for ip, v in by_ip.items()), key=lambda r: -r["count"]
    )

    sign_in_counts = Counter(r["outcome"] for r in data.sign_ins)
    per_email = defaultdict(lambda: {"ok": 0, "blocked": 0, "last": None})

    for row in data.sign_ins:
        if row["email"] and row["outcome"] in ("ok", "blocked"):
            e = per_email[row["email"]]
            e[row["outcome"]] += 1
            e["last"] = max(filter(None, [e["last"], row["created_at"]]))

    return {
        "auth_failures": len(failures),
        "by_status": dict(by_status),
        "failure_rate": _rate(len(failures), len(failures) + len(data.requests)),
        "failure_rate_note": "auth failures as a share of auth failures plus recorded requests",
        "by_ip": ips[:10],
        "suspicious_ips": [r["ip"] for r in ips if r["count"] >= 10],
        "by_path": [{"path": p, "count": c} for p, c in by_path.most_common(10)],
        "by_day": [{"date": d, "count": c} for d, c in sorted(by_day.items())],
        "sign_ins": {
            "ok": sign_in_counts["ok"],
            "blocked": sign_in_counts["blocked"],
            "invalid": sign_in_counts["invalid"],
        },
        "blocked_emails": sorted(
            ({"email": e, **v} for e, v in per_email.items() if v["blocked"]),
            key=lambda r: -r["blocked"],
        ),
        "recent": [
            {
                "created_at": r["created_at"], "path": r["path"], "ip": r["remote_addr"],
                "status": status_of(r), "email": r.get("email"),
            }
            for r in reversed(failures[-15:])
        ],
    }


# --- user rows ------------------------------------------------------------


def _user_rows(data, sessions, allowed_emails, admin_emails):
    users = (
        {t["user"] for t in data.turns}
        | {r["user_email"] for r in data.llm}
        | {r["user_email"] for r in data.tools}
        | {s["user"] for s in sessions}
        | {r["email"] for r in data.sign_ins if r["outcome"] == "ok"}
        | set(allowed_emails)
    )
    users = {u for u in users if u != ""}

    rows = []

    for user in users:
        sub = data.subset(user)
        user_sessions = [s for s in sessions if s["user"] == user]
        stamps = (
            [t["ts"] for t in sub.turns]
            + [s["last_seen_at"] for s in user_sessions]
            + [r["created_at"] for r in sub.llm]
        )
        cost = sum(r["cost_usd"] or 0.0 for r in sub.llm)
        chat = [t for t in sub.turns if t["kind"] == "chat"]
        flagged = [t for t in sub.turns if t["score"] > 0]

        rows.append({
            "key": user_key(user),
            "email": user,
            "label": user_label(user),
            "is_admin": user in admin_emails,
            "is_shared_token": user == BEARER_USER,
            "never_active": not stamps,
            "sessions": len(user_sessions),
            "turns": len(sub.turns),
            "failed_turns": sum(1 for t in sub.turns if t["status"] == "failed"),
            "flagged_turns": len(flagged),
            "flagged_rate": _rate(len(flagged), len(sub.turns)),
            "llm_calls": len(sub.llm),
            "prompt_tokens": sum(r["prompt_tokens"] or 0 for r in sub.llm),
            "completion_tokens": sum(r["completion_tokens"] or 0 for r in sub.llm),
            "cost_usd": round(cost, 6),
            "cost_per_turn_usd": _round(cost / len(chat), 6) if chat else None,
            "p50_latency": _round(percentile([t["latency"] for t in chat if t["latency"] is not None], 50)),
            "thumbs_down": sum(1 for r in sub.feedback if r["rating"] < 0),
            "last_seen": max(stamps) if stamps else None,
        })

    rows.sort(key=lambda r: (-r["turns"], -r["cost_usd"], r["label"]))

    return rows


# --- public API -----------------------------------------------------------


def _prepare(days, now):
    now = now or datetime.now(timezone.utc)
    data = load(days, now)
    annotate(data)
    sessions = build_sessions(data)
    _, start_date = _window(days, now)
    today = now.astimezone(_tz()).date()

    return now, data, sessions, start_date, today


def _window_info(days, now):
    since, _ = _window(days, now)

    return {"days": days, "since": since, "timezone": str(config.ADMIN_TIMEZONE)}


def overview(days=30, now=None, allowed_emails=(), admin_emails=()):
    now, data, sessions, start_date, today = _prepare(days, now)
    users = _user_rows(data, sessions, set(allowed_emails), set(admin_emails))
    quality = _quality(data)
    cost = _cost(data, sessions, start_date, today, now)
    usage = _usage(data, sessions, len(users), today)
    speed = _speed(data)

    return {
        "generated_at": now.isoformat(),
        "window": _window_info(days, now),
        "headline": {
            "failure_rate": quality["failure_rate"],
            "not_found_rate": quality["not_found"]["rate"],
            "tool_failure_rate": quality["tool_failures"]["rate"],
            "satisfaction": quality["feedback"]["satisfaction"],
            "total_cost_usd": cost["total_usd"],
            "projected_monthly_usd": cost["projection"]["monthly_usd"],
            "users": len(users),
            "active_users": usage["active_users"],
            "sessions": usage["sessions"],
            "turns": usage["turns"],
            "llm_calls": cost["calls"],
            "tokens": cost["tokens_total"],
            "p95_latency": speed["chat_latency"]["p95"],
        },
        "quality": quality,
        "cost": cost,
        "usage": usage,
        "speed": speed,
        "adoption": _adoption(data),
        "data_health": _data_health(data, days, now),
        "security": _security(data),
        "users": users,
    }


def user_detail(key, days=30, now=None, allowed_emails=(), admin_emails=()):
    user = user_from_key(key)
    now, data, sessions, start_date, today = _prepare(days, now)
    sub = data.subset(user)
    user_sessions = [s for s in sessions if s["user"] == user]
    quality = _quality(sub)
    cost = _cost(sub, user_sessions, start_date, today, now)
    usage = _usage(sub, user_sessions, 1, today)
    speed = _speed(sub)
    adoption = _adoption(sub)

    session_rows = []

    for session in sorted(user_sessions, key=lambda s: s["started_at"], reverse=True)[:MAX_SESSIONS_PER_USER]:
        turns = session["turns"]
        minutes = (_parse(session["last_seen_at"]) - _parse(session["started_at"])).total_seconds() / 60

        session_rows.append({
            "id": session["id"],
            "inferred": session["inferred"],
            "started_at": session["started_at"],
            "last_seen_at": session["last_seen_at"],
            "minutes": round(minutes, 1),
            "turn_count": len(turns),
            "failed": sum(1 for t in turns if t["status"] == "failed"),
            "flagged": sum(1 for t in turns if t["score"] > 0),
            "cost_usd": round(sum(t["cost_usd"] or 0.0 for t in turns), 6),
            "turns": [_turn_summary(t, include_tools=True) for t in turns],
        })

    stamps = [t["ts"] for t in sub.turns] + [s["last_seen_at"] for s in user_sessions]

    return {
        "generated_at": now.isoformat(),
        "window": _window_info(days, now),
        "user": {
            "key": user_key(user),
            "email": user,
            "label": user_label(user),
            "is_admin": user in set(admin_emails),
            "is_shared_token": user == BEARER_USER,
            "allowed": user in set(allowed_emails),
            "first_seen": data.first_seen.get(user),
            "last_seen": max(stamps) if stamps else None,
        },
        "totals": {
            "turns": usage["turns"],
            "sessions": usage["sessions"],
            "cost_usd": cost["total_usd"],
            "llm_calls": cost["calls"],
            "prompt_tokens": cost["prompt_tokens"],
            "completion_tokens": cost["completion_tokens"],
            "avg_per_turn_usd": cost["avg_per_turn_usd"],
            "failure_rate": quality["failure_rate"],
            "flagged_turns": sum(1 for t in sub.turns if t["score"] > 0),
            "active_days": len({_day(t["ts"]) for t in sub.turns}),
            "latency": speed["chat_latency"],
        },
        "quality": quality,
        "cost": cost,
        "usage": usage,
        "speed": speed,
        "adoption": adoption,
        "sessions": session_rows,
        "sessions_shown": len(session_rows),
        "sessions_total": len(user_sessions),
    }


def turn_detail(run_id):
    """The full step-by-step trace of one run, for reviewing a bad turn:
    each LLM call and tool call with its timing. The (large) request
    messages are left out -- they hold the whole conversation context."""
    path = run_logger_module.LOG_DIR / f"{run_id}.json"

    if not path.is_file():
        return None

    try:
        trace = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None

    steps = []
    previous = 0.0

    for event in trace.get("events", []):
        seconds = round(max(0.0, event.get("elapsed_seconds", 0) - previous), 3)
        previous = event.get("elapsed_seconds", previous)
        step = {"type": event.get("type"), "seconds": seconds}

        if event["type"] == "llm_call":
            step.update({
                "component": event.get("component"),
                "model": event.get("model"),
                "prompt_tokens": event.get("prompt_tokens"),
                "completion_tokens": event.get("completion_tokens"),
                "cost_usd": event.get("cost_usd"),
                "content": _clip(event.get("content"), 1500),
                "tool_calls": event.get("tool_calls"),
            })
        elif event["type"] == "tool_call":
            step.update({
                "tool": event.get("tool_name"),
                "arguments": event.get("arguments"),
                "success": event.get("success"),
                "error": event.get("error"),
                "output": _clip(json.dumps(event.get("output"), default=str, ensure_ascii=False), 800),
            })
        else:
            step["message"] = _clip(event.get("message"), 500)

        steps.append(step)

    return {
        "run_id": trace.get("run_id"),
        "user_email": trace.get("user_email"),
        "session_id": trace.get("session_id"),
        "endpoint": trace.get("endpoint"),
        "user_input": _clip(trace.get("user_input")),
        "status": trace.get("status"),
        "duration_seconds": trace.get("duration_seconds"),
        "prompt_tokens": trace.get("total_prompt_tokens"),
        "completion_tokens": trace.get("total_completion_tokens"),
        "cost_usd": trace.get("total_cost_usd"),
        "steps": steps,
    }
