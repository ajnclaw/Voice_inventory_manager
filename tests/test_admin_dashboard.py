"""
Tests for the admin dashboard and the per-user attribution behind it:
schema migrations of existing databases, sessions, the metric heuristics and
roll-ups, and the admin-only HTTP routes.

Everything runs against throwaway databases (see _isolation.py): the real
inventory.db, usage.db, monitoring.db, transcripts.jsonl and logs/ are never
touched. The metric tests seed rows with explicit timestamps against a fixed
"now", so they do not depend on the day they run.

Run from the project root:
    .venv/bin/python3 -m unittest discover -s tests -v
"""

import http.client
import json
import shutil
import sqlite3
import sys
import tempfile
import threading
import unittest
from contextlib import closing
from datetime import datetime, timedelta, timezone
from http.server import HTTPServer
from pathlib import Path
from unittest import mock

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from _isolation import data_patches
from agent import admin_metrics, config, google_auth, inventory_db, monitoring, sessions, usage_tracker
from agent.db_migrate import ensure_columns
from agent.llm_client import AttrDict
from agent.logger import RunLogger

ALICE = "alice@example.com"   # admin
BOB = "bob@example.com"       # allowed, not admin
CAROL = "carol@example.com"   # allowed, never active
BEARER = sessions.BEARER_USER
NOW = datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc)
SECRET = "test-secret-not-used-anywhere-real"
TOKEN = "test-bearer-token"


def at(day, hour, minute=0, second=0):
    return datetime(2026, 11, day, hour, minute, second, tzinfo=timezone.utc).isoformat()


def rows(db_path, sql, params=()):
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(sql, params)]


class TempDataCase(unittest.TestCase):
    """Each test gets empty stores in a fresh temp dir."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="shop-test-"))
        self.patches = data_patches(self.tmp)
        for patch in self.patches:
            patch.start()
        self.usage_db = self.tmp / "usage.db"
        self.monitoring_db = self.tmp / "monitoring.db"
        self.transcript = self.tmp / "transcripts.jsonl"
        self.inventory = self.tmp / "inventory.db"

    def tearDown(self):
        for patch in reversed(self.patches):
            patch.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)


# --- migrations -------------------------------------------------------------


class MigrationTests(TempDataCase):
    def test_ensure_columns_adds_only_missing_ones_and_is_repeatable(self):
        with closing(sqlite3.connect(":memory:")) as conn:
            conn.execute("CREATE TABLE t (a TEXT)")
            conn.execute("INSERT INTO t VALUES ('keep me')")
            ensure_columns(conn, "t", {"a": "TEXT", "b": "INTEGER"})
            ensure_columns(conn, "t", {"a": "TEXT", "b": "INTEGER"})

            self.assertEqual([r[1] for r in conn.execute("PRAGMA table_info(t)")], ["a", "b"])
            self.assertEqual(conn.execute("SELECT a, b FROM t").fetchall(), [("keep me", None)])

    def test_existing_usage_db_keeps_its_rows_and_gains_columns(self):
        with closing(sqlite3.connect(self.usage_db)) as conn:
            conn.execute(
                "CREATE TABLE llm_calls (id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, "
                "component TEXT NOT NULL, model TEXT NOT NULL, prompt_tokens INTEGER, "
                "completion_tokens INTEGER, cost_usd REAL, created_at TEXT NOT NULL)"
            )
            conn.execute(
                "INSERT INTO llm_calls (run_id, component, model, prompt_tokens, completion_tokens, "
                "cost_usd, created_at) VALUES ('old', 'responder', 'm', 10, 5, 0.5, '2026-01-01T00:00:00+00:00')"
            )
            conn.commit()

        usage_tracker.record_call("new", "responder", "m", 1, 1, 0.1, user_email=ALICE, session_id="s1", duration_seconds=1.5)

        stored = rows(self.usage_db, "SELECT * FROM llm_calls ORDER BY id")
        self.assertEqual(len(stored), 2)
        self.assertEqual((stored[0]["run_id"], stored[0]["cost_usd"], stored[0]["user_email"]), ("old", 0.5, None))
        self.assertEqual((stored[1]["user_email"], stored[1]["session_id"], stored[1]["duration_seconds"]), (ALICE, "s1", 1.5))
        self.assertEqual(usage_tracker.totals()["total_calls"], 2)

    def test_existing_monitoring_db_keeps_its_rows_and_gains_columns_and_tables(self):
        with closing(sqlite3.connect(self.monitoring_db)) as conn:
            conn.execute("CREATE TABLE auth_failures (id INTEGER PRIMARY KEY AUTOINCREMENT, path TEXT NOT NULL, remote_addr TEXT, created_at TEXT NOT NULL)")
            conn.execute("CREATE TABLE request_outcomes (id INTEGER PRIMARY KEY AUTOINCREMENT, endpoint TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL)")
            conn.execute("CREATE TABLE import_outcomes (id INTEGER PRIMARY KEY AUTOINCREMENT, outcome TEXT NOT NULL CHECK(outcome IN ('confirmed','cancelled')), item_count INTEGER, created_at TEXT NOT NULL)")
            conn.execute("INSERT INTO auth_failures (path, remote_addr, created_at) VALUES ('/chat', '1.1.1.1', '2026-01-01T00:00:00+00:00')")
            conn.execute("INSERT INTO request_outcomes (endpoint, status, created_at) VALUES ('chat', 'failed', '2026-01-01T00:00:00+00:00')")
            conn.execute("INSERT INTO import_outcomes (outcome, item_count, created_at) VALUES ('confirmed', 4, '2026-01-01T00:00:00+00:00')")
            conn.commit()

        monitoring.record_request_outcome("chat", "completed", user_email=ALICE, session_id="s1", run_id="r", duration_seconds=2.0)

        old = rows(self.monitoring_db, "SELECT * FROM request_outcomes ORDER BY id")
        self.assertEqual((old[0]["status"], old[0]["user_email"]), ("failed", None))
        self.assertEqual((old[1]["user_email"], old[1]["duration_seconds"]), (ALICE, 2.0))
        self.assertEqual(rows(self.monitoring_db, "SELECT item_count FROM import_outcomes")[0]["item_count"], 4)
        self.assertIsNone(rows(self.monitoring_db, "SELECT status_code FROM auth_failures")[0]["status_code"])

        tables = {r["name"] for r in rows(self.monitoring_db, "SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertTrue({"sessions", "sign_ins", "tool_calls", "feedback"} <= tables)
        self.assertEqual(monitoring.stats()["requests"]["total"], 2)

    def test_old_transcript_lines_show_up_as_unattributed(self):
        self.transcript.write_text(
            json.dumps({"timestamp": at(1, 10), "user_input": "old question", "reply": "old answer",
                        "status": "completed", "model": "m", "prompt_tokens": 1, "completion_tokens": 1,
                        "cost_usd": 0.01}) + "\n" + "not json at all\n",
            encoding="utf-8",
        )

        result = admin_metrics.overview(days=30, now=NOW)
        legacy = next(u for u in result["users"] if u["key"] == admin_metrics.UNATTRIBUTED_KEY)

        self.assertEqual(legacy["label"], "(unattributed)")
        self.assertEqual(legacy["turns"], 1)


class ConfigTests(unittest.TestCase):
    def test_admins_must_also_be_on_the_allowlist(self):
        allowed = {ALICE, BOB}
        self.assertEqual(config.admin_emails(" Alice@Example.com , stranger@example.com", allowed), {ALICE})

    def test_unset_or_empty_means_nobody_is_admin(self):
        self.assertEqual(config.admin_emails("", {ALICE}), set())
        self.assertEqual(config.admin_emails(None, {ALICE}), set())
        self.assertEqual(config.admin_emails(ALICE, set()), set())


# --- sessions ---------------------------------------------------------------


class SessionTests(TempDataCase):
    def test_activity_within_the_idle_limit_stays_in_one_session(self):
        first = sessions.touch(ALICE, now=NOW, idle_seconds=1800)
        again = sessions.touch(ALICE, now=NOW + timedelta(minutes=29), idle_seconds=1800)
        self.assertEqual(first, again)

    def test_gap_longer_than_the_idle_limit_starts_a_new_session(self):
        first = sessions.touch(ALICE, now=NOW, idle_seconds=1800)
        later = sessions.touch(ALICE, now=NOW + timedelta(minutes=31), idle_seconds=1800)
        self.assertNotEqual(first, later)
        self.assertEqual(len(rows(self.monitoring_db, "SELECT * FROM sessions")), 2)

    def test_idle_clock_runs_from_the_last_activity_not_the_start(self):
        first = sessions.touch(ALICE, now=NOW, idle_seconds=1800)
        sessions.touch(ALICE, now=NOW + timedelta(minutes=25), idle_seconds=1800)
        third = sessions.touch(ALICE, now=NOW + timedelta(minutes=50), idle_seconds=1800)
        self.assertEqual(first, third)

    def test_users_never_share_a_session(self):
        a = sessions.touch(ALICE, now=NOW)
        b = sessions.touch(BOB, now=NOW)
        self.assertNotEqual(a, b)


# --- heuristics and statistics ----------------------------------------------


class HeuristicTests(unittest.TestCase):
    def test_not_found_replies(self):
        for text in [
            "xyz gasket nahi mila catalog mein", "Item not found", "I couldn't find that item",
            "I wasn't able to finish looking into that -- could you rephrase or try again?",
            "Something went wrong: timeout", "no matching items, doesn't exist",
        ]:
            self.assertTrue(admin_metrics.NOT_FOUND_RE.search(text), text)

        for text in ["oil filter ka stock 5 hai", "3 filter bik gaye, stock 2", "stock 0 hai"]:
            self.assertFalse(admin_metrics.NOT_FOUND_RE.search(text), text)

    def test_clarification_replies(self):
        for text in ["Kaun sa filter? oil ya air", "Which one do you mean?", "kaunsa wala chahiye", "Did you mean the 12mm bolt?"]:
            self.assertTrue(admin_metrics.CLARIFY_RE.search(text), text)

        self.assertFalse(admin_metrics.CLARIFY_RE.search("oil filter 5 hai"))

    def test_correction_messages(self):
        for text in ["nahi 2 sold the filters", "No, I meant the air filter", "galat item tha", "undo that", "mera matlab 12mm tha"]:
            self.assertTrue(admin_metrics.CORRECTION_RE.search(text), text)

        for text in ["how many oil filters", "sold 3 bolts", "air filter ka rate?"]:
            self.assertFalse(admin_metrics.CORRECTION_RE.search(text), text)

    def test_percentile(self):
        self.assertIsNone(admin_metrics.percentile([], 50))
        self.assertEqual(admin_metrics.percentile([7], 95), 7)
        self.assertEqual(admin_metrics.percentile([1, 2, 3, 4], 50), 2.5)
        self.assertAlmostEqual(admin_metrics.percentile([1, 1.5, 1.5, 2, 2.5, 3, 4, 12], 95), 9.2)


# --- metrics over a seeded shop ----------------------------------------------


class SeededMetricsTests(TempDataCase):
    """
    Alice (admin): five chat turns in one session -- a good answer, a sale, a
    user correction of that sale, a thumbs-down "not found", and a failure.
    Bob: a clarification, then a table edit of the same item.
    The shared token: one turn. Legacy rows: three turns, no user at all.
    """

    def setUp(self):
        super().setUp()
        self.seed()

    def turn(self, ts, user, session, run, text, reply, status="completed", latency=2.0,
             cost=0.001, mode=None, kind="chat"):
        entry = {
            "timestamp": ts, "user_input": text, "reply": reply, "status": status,
            "model": "openai/gpt-4o-mini", "prompt_tokens": 100, "completion_tokens": 20,
            "cost_usd": cost,
        }
        if user is not None or session or run:
            entry.update({"user": user, "session_id": session, "run_id": run,
                          "latency_seconds": latency, "input_mode": mode, "kind": kind})
        with open(self.transcript, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def llm(self, ts, user, session, run, cost, model="openai/gpt-4o-mini", duration=1.0, component="responder"):
        with closing(usage_tracker._connect()) as conn:
            conn.execute(
                "INSERT INTO llm_calls (run_id, component, model, prompt_tokens, completion_tokens, cost_usd, "
                "created_at, user_email, session_id, duration_seconds) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (run, component, model, 100, 20, cost, ts, user, session, duration),
            )
            conn.commit()

    def tool(self, ts, user, session, run, name, ok=True, item=None, count=None, endpoint="chat", error=None, seconds=0.1):
        with closing(monitoring._connect()) as conn:
            conn.execute(
                "INSERT INTO tool_calls (run_id, user_email, session_id, endpoint, tool_name, success, denied, "
                "error, duration_seconds, item, arguments, result_count, created_at) VALUES (?,?,?,?,?,?,0,?,?,?,?,?,?)",
                (run, user, session, endpoint, name, 1 if ok else 0, error, seconds, item, "{}", count, ts),
            )
            conn.commit()

    def insert(self, table, **columns):
        with closing(monitoring._connect()) as conn:
            conn.execute(
                f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))})",
                tuple(columns.values()),
            )
            conn.commit()

    def seed(self):
        for sid, user, start, last in [
            ("s_a1", ALICE, at(3, 4, 0), at(3, 4, 20)),
            ("s_b1", BOB, at(4, 5, 0), at(4, 5, 12)),
            ("s_t1", BEARER, at(4, 6, 0), at(4, 6, 0)),
        ]:
            self.insert("sessions", id=sid, user_email=user, started_at=start, last_seen_at=last)

        a = ("s_a1", ALICE)
        # Alice
        self.turn(at(3, 4, 0, 30), ALICE, "s_a1", "R1", "oil filter kitna hai", "oil filter ka stock 5 hai", latency=2.0, cost=0.001, mode="voice")
        self.turn(at(3, 4, 5), ALICE, "s_a1", "R2", "sold 3 filters", "3 oil filter bik gaye, stock 2", latency=4.0, cost=0.002, mode="text")
        self.turn(at(3, 4, 6), ALICE, "s_a1", "R3", "nahi 2 sold the filters", "Theek hai, stock 3 kar diya", latency=2.5, cost=0.0015, mode="text")
        self.turn(at(3, 4, 15), ALICE, "s_a1", "R4", "xyz gasket hai?", "xyz gasket nahi mila catalog mein", latency=3.0, cost=0.001, mode="text")
        self.turn(at(3, 4, 20), ALICE, "s_a1", "R5", "update price", "Something went wrong: boom", status="failed", latency=12.0, cost=0.0005, mode="text")
        # Bob
        self.turn(at(4, 5, 0), BOB, "s_b1", "R6", "filter", "Kaun sa filter? oil ya air", latency=1.5, cost=None, mode="text")
        self.turn(at(4, 5, 10), BOB, "s_b1", "R7", "oil", "oil filter 5 hai", latency=1.5, cost=0.001, mode="text")
        # Shared token
        self.turn(at(4, 6, 0), BEARER, "s_t1", "R8", "stock?", "sab theek hai", latency=1.0, cost=0.001)
        # Legacy: no user, no session; the third is hours after the first two.
        self.turn(at(1, 10, 0), None, None, None, "old q1", "old a1", cost=0.001)
        self.turn(at(1, 10, 10), None, None, None, "old q2", "old a2", cost=0.001)
        self.turn(at(2, 10, 0), None, None, None, "old q3", "old a3", cost=0.0)

        self.llm(at(3, 4, 0, 30), ALICE, "s_a1", "R1", 0.001, duration=1.5)
        self.llm(at(3, 4, 5), ALICE, "s_a1", "R2", 0.002, duration=3.5)
        self.llm(at(3, 4, 6), ALICE, "s_a1", "R3", 0.0015, duration=2.0)
        self.llm(at(3, 4, 15), ALICE, "s_a1", "R4", 0.001, duration=2.5)
        self.llm(at(3, 4, 20), ALICE, "s_a1", "R5", 0.0005, duration=11.0)
        self.llm(at(4, 5, 0), BOB, "s_b1", "R6", None, model="gemini-flash-lite-latest", duration=1.0)
        self.llm(at(4, 5, 10), BOB, "s_b1", "R7", 0.001, duration=1.0)
        self.llm(at(4, 6, 0), BEARER, "s_t1", "R8", 0.001, duration=0.8)
        self.llm(at(1, 10, 0), None, None, "L0", 0.002, component="compactor")  # run_id is required, user is not

        self.tool(at(3, 4, 0, 30), ALICE, "s_a1", "R1", "search_items", item="oil filter", count=1)
        self.tool(at(3, 4, 0, 31), ALICE, "s_a1", "R1", "check_stock", item="Oil Filter")
        self.tool(at(3, 4, 5), ALICE, "s_a1", "R2", "record_sale", item="Oil Filter")
        self.tool(at(3, 4, 6), ALICE, "s_a1", "R3", "adjust_stock", item="Oil Filter")
        self.tool(at(3, 4, 15), ALICE, "s_a1", "R4", "search_items", item="xyz gasket", count=0)
        self.tool(at(3, 4, 20), ALICE, "s_a1", "R5", "set_price", ok=False, item="Nothing", error="Item not found")
        # Bob edits the item from the table two minutes after his chat turn about it.
        self.tool(at(4, 5, 12), BOB, "s_b1", "R9", "adjust_stock", item="Oil Filter", endpoint="table_adjust_stock")

        self.insert("feedback", run_id="R1", user_email=ALICE, session_id="s_a1", rating=1, created_at=at(3, 4, 1))
        self.insert("feedback", run_id="R4", user_email=ALICE, session_id="s_a1", rating=-1, created_at=at(3, 4, 16))

        self.insert("import_outcomes", outcome="confirmed", item_count=3, created_at=at(3, 5), user_email=ALICE)
        self.insert("import_outcomes", outcome="cancelled", item_count=2, created_at=at(3, 6), user_email=BOB)
        for _ in range(2):
            self.insert("request_outcomes", endpoint="inventory_set_image", status="completed",
                        created_at=at(3, 7), user_email=ALICE, duration_seconds=0.3)
        self.insert("request_outcomes", endpoint="chat", status="failed", created_at=at(3, 4, 20),
                    user_email=ALICE, session_id="s_a1", run_id="R5", duration_seconds=12.0)
        self.insert("request_outcomes", endpoint="chat", status="completed", created_at=at(3, 4, 0, 30),
                    user_email=ALICE, session_id="s_a1", run_id="R1", duration_seconds=2.0)

        for _ in range(12):
            self.insert("auth_failures", path="/chat", remote_addr="1.2.3.4", created_at=at(3, 8))
        self.insert("auth_failures", path="/admin/api/overview", remote_addr="5.6.7.8", created_at=at(4, 8),
                    status_code=403, email=BOB)
        for _ in range(2):
            self.insert("sign_ins", email=ALICE, outcome="ok", created_at=at(3, 3, 59))
        for _ in range(3):
            self.insert("sign_ins", email="evil@example.com", outcome="blocked", remote_addr="9.9.9.9", created_at=at(4, 9))
        self.insert("sign_ins", email=None, outcome="invalid", created_at=at(4, 9))

        # Inventory: one complete item, one low-stock item missing price/category/photo, one out of stock.
        inventory_db.add_item("Oil Filter", category="Filters", cost_price=100, sale_price=150, reorder_threshold=3,
                              image_path="product_images/a.jpg")
        inventory_db.add_item("Air Cleaner", cost_price=380, reorder_threshold=2)
        inventory_db.add_item("Bolt", category="Hardware", sale_price=5)
        with closing(sqlite3.connect(self.inventory)) as conn:
            conn.execute("UPDATE items SET current_quantity = 5 WHERE name = 'Oil Filter'")
            conn.execute("UPDATE items SET current_quantity = 1 WHERE name = 'Air Cleaner'")
            ids = {r[1]: r[0] for r in conn.execute("SELECT id, name FROM items")}
            for item, kind, delta, ts in [
                ("Oil Filter", "sale", -3, at(3, 4, 5)),
                ("Oil Filter", "restock", 10, at(1, 9)),
                ("Bolt", "adjustment", -1, at(2, 9)),
            ]:
                conn.execute(
                    "INSERT INTO transactions (item_id, type, quantity_delta, created_at) VALUES (?,?,?,?)",
                    (ids[item], kind, delta, ts),
                )
            conn.commit()

        self.result = admin_metrics.overview(
            days=30, now=NOW, allowed_emails={ALICE, BOB, CAROL}, admin_emails={ALICE}
        )

    # -- quality ------------------------------------------------------------

    def test_failure_and_reply_gap_rates(self):
        q = self.result["quality"]
        self.assertEqual(q["chat_turns"], 11)
        self.assertEqual(q["failed_turns"], 1)
        self.assertAlmostEqual(q["failure_rate"], 1 / 11, places=3)
        self.assertEqual(q["not_found"]["count"], 2)       # "nahi mila" + "Something went wrong"
        self.assertEqual(q["clarification"]["count"], 1)   # "Kaun sa filter?"
        self.assertEqual(q["request_failure_rate"], 0.25)  # 1 failed of 4 recorded requests

    def test_request_failure_rate_counts_recorded_requests(self):
        # 2 photo uploads + 1 failed chat + 1 completed chat.
        by = {r["endpoint"]: r for r in self.result["quality"]["requests_by_endpoint"]}
        self.assertEqual(by["chat"]["total"], 2)
        self.assertEqual(by["chat"]["failed"], 1)
        self.assertEqual(by["inventory_set_image"]["failed"], 0)

    def test_tool_failure_rate_by_tool(self):
        tools = self.result["quality"]["tool_failures"]
        self.assertEqual(tools["calls"], 7)
        self.assertEqual(tools["failed"], 1)
        self.assertAlmostEqual(tools["rate"], 1 / 7, places=3)
        worst = tools["by_tool"][0]
        self.assertEqual((worst["tool"], worst["failed"], worst["top_error"]), ("set_price", 1, "Item not found"))

    def test_user_corrections_are_detected_both_ways(self):
        q = self.result["quality"]["corrections"]
        self.assertEqual(q["count"], 2)
        self.assertEqual(
            q["reasons"],
            {"next message reads like a correction": 1, "edited in the table right after": 1},
        )

    def test_searches_that_found_nothing_are_listed(self):
        empty = self.result["quality"]["empty_searches"]
        self.assertEqual(empty["count"], 1)
        self.assertEqual(empty["top_queries"], [{"query": "xyz gasket", "count": 1}])

    def test_thumbs(self):
        fb = self.result["quality"]["feedback"]
        self.assertEqual((fb["up"], fb["down"], fb["satisfaction"]), (1, 1, 0.5))

    def test_worst_turns_are_ranked_and_explained(self):
        worst = self.result["quality"]["worst_turns"]
        self.assertEqual([t["run_id"] for t in worst[:2]], ["R5", "R4"])
        self.assertEqual({f["key"] for f in worst[0]["flags"]}, {"failed", "tool_error", "not_found", "slow"})
        self.assertEqual({f["key"] for f in worst[1]["flags"]}, {"thumbs_down", "not_found", "empty_search"})
        self.assertTrue(all(t["score"] > 0 for t in worst))

    # -- cost ---------------------------------------------------------------

    def test_cost_totals_and_per_user_split(self):
        cost = self.result["cost"]
        self.assertAlmostEqual(cost["total_usd"], 0.010, places=6)
        self.assertEqual(cost["calls"], 9)
        self.assertEqual(cost["unpriced_calls"], 1)
        self.assertEqual((cost["prompt_tokens"], cost["completion_tokens"]), (900, 180))

        by_user = {r["label"]: r["cost_usd"] for r in cost["by_user"]}
        self.assertAlmostEqual(by_user[ALICE], 0.006, places=6)
        self.assertAlmostEqual(by_user[BOB], 0.001, places=6)
        self.assertAlmostEqual(by_user["Shared access token"], 0.001, places=6)
        self.assertAlmostEqual(by_user["(unattributed)"], 0.002, places=6)

    def test_cost_by_component_and_model(self):
        cost = self.result["cost"]
        self.assertEqual({r["key"] for r in cost["by_component"]}, {"responder", "compactor"})
        self.assertEqual({r["key"] for r in cost["by_model"]}, {"openai/gpt-4o-mini", "gemini-flash-lite-latest"})

    def test_daily_series_has_no_gaps_and_projection_uses_the_last_week(self):
        cost = self.result["cost"]
        days = [d["date"] for d in cost["by_day"]]
        self.assertEqual(len(days), 31)  # 30-day window + today
        self.assertEqual(days[-1], "2026-11-04")
        self.assertEqual(days, sorted(set(days)))

        last7 = sum(d["cost_usd"] for d in cost["by_day"][-7:])
        self.assertAlmostEqual(cost["projection"]["monthly_usd"], round(last7 / 7 * 30, 4), places=4)
        self.assertAlmostEqual(cost["projection"]["month_to_date_usd"], 0.010, places=6)

    # -- usage --------------------------------------------------------------

    def test_users_sessions_and_turns(self):
        usage = self.result["usage"]
        self.assertEqual(usage["turns"], 11)
        self.assertEqual(usage["sessions"], 5)            # 3 recorded + 2 inferred
        self.assertEqual(usage["inferred_sessions"], 2)   # legacy turns split by the long gap
        self.assertEqual(usage["active_users"], 4)        # alice, bob, shared token, unattributed
        self.assertEqual(usage["turns_per_session"]["avg"], 2.2)
        self.assertEqual(usage["turns_without_session"], 3)
        self.assertEqual(usage["returning_users"]["count"], 1)  # only the legacy rows span two days

    def test_user_list_includes_everyone_with_access(self):
        users = {u["email"]: u for u in self.result["users"]}
        self.assertEqual(set(users), {ALICE, BOB, CAROL, BEARER, None})
        self.assertTrue(users[CAROL]["never_active"])
        self.assertTrue(users[ALICE]["is_admin"])
        self.assertTrue(users[BEARER]["is_shared_token"])
        self.assertEqual((users[ALICE]["turns"], users[ALICE]["sessions"]), (5, 1))
        self.assertEqual(users[ALICE]["failed_turns"], 1)

    def test_peak_hours_use_the_shop_timezone(self):
        # Alice's five turns are 04:00-04:20 UTC = 09:30-09:50 in Asia/Kolkata.
        peak = max(self.result["usage"]["peak_hours"], key=lambda h: h["turns"])
        self.assertEqual((peak["hour"], peak["turns"]), (9, 5))

    # -- speed --------------------------------------------------------------

    def test_latency_percentiles_and_stages(self):
        speed = self.result["speed"]
        self.assertEqual(speed["chat_latency"]["p50"], 2.25)
        self.assertEqual(speed["chat_latency"]["p95"], 9.2)
        self.assertEqual(speed["slowest"][0]["run_id"], "R5")
        self.assertEqual({s["stage"] for s in speed["tool_stages"]}, {"search_items", "check_stock", "record_sale", "adjust_stock", "set_price"})
        responder = next(s for s in speed["llm_stages"] if s["stage"] == "responder")
        self.assertEqual(responder["count"], 8)
        self.assertEqual(responder["max"], 11.0)

    # -- adoption -----------------------------------------------------------

    def test_adoption(self):
        a = self.result["adoption"]
        self.assertEqual((a["imports"]["confirmed"], a["imports"]["cancelled"], a["imports"]["reject_rate"]), (1, 1, 0.5))
        self.assertEqual(a["photo_uploads"]["completed"], 2)
        self.assertEqual(a["edits"]["stock_from_table"], 1)
        self.assertEqual(a["edits"]["stock_from_chat"], 1)
        self.assertEqual(a["edits"]["sales_from_chat"], 1)
        self.assertEqual(a["input_mode"], {"voice": 1, "text": 6, "unknown": 4})

    # -- data health --------------------------------------------------------

    def test_data_health(self):
        h = self.result["data_health"]
        self.assertTrue(h["available"])
        self.assertEqual(h["items"], 3)
        self.assertEqual(h["missing"]["sale_price"]["examples"], ["Air Cleaner"])
        self.assertEqual(h["missing"]["photo"]["count"], 2)
        self.assertEqual(h["missing"]["category"]["examples"], ["Air Cleaner"])
        self.assertAlmostEqual(h["complete_pct"], 1 / 3, places=3)
        self.assertEqual(h["stock"]["low"], 1)
        self.assertEqual(h["stock"]["low_items"][0]["name"], "Air Cleaner")
        self.assertEqual(h["stock"]["out_of_stock"], 1)
        self.assertEqual(h["ledger"]["transactions"], 3)
        self.assertEqual(h["ledger"]["adjustments"], 1)
        self.assertEqual(h["ledger"]["stale_items_90d"], 1)  # Air Cleaner never moved

    def test_data_health_never_writes_to_inventory(self):
        before = self.inventory.read_bytes()
        admin_metrics.overview(days=30, now=NOW)
        self.assertEqual(self.inventory.read_bytes(), before)

    # -- security -----------------------------------------------------------

    def test_security(self):
        s = self.result["security"]
        self.assertEqual(s["auth_failures"], 13)
        self.assertEqual(s["by_status"], {"401": 12, "403": 1})  # old rows have no status and count as 401
        self.assertEqual(s["suspicious_ips"], ["1.2.3.4"])
        self.assertEqual(s["by_ip"][0], {"ip": "1.2.3.4", "count": 12, "last": at(3, 8)})
        self.assertEqual(s["sign_ins"], {"ok": 2, "blocked": 3, "invalid": 1})
        self.assertEqual(s["blocked_emails"][0]["email"], "evil@example.com")
        self.assertEqual(s["recent"][0]["status"], 403)

    # -- per-user page ------------------------------------------------------

    def test_user_detail_lists_every_session_and_turn(self):
        detail = admin_metrics.user_detail(ALICE, days=30, now=NOW, allowed_emails={ALICE}, admin_emails={ALICE})
        self.assertEqual(detail["totals"]["turns"], 5)
        self.assertAlmostEqual(detail["totals"]["cost_usd"], 0.006, places=6)
        self.assertEqual(detail["totals"]["sessions"], 1)
        self.assertEqual(detail["sessions_total"], 1)

        session = detail["sessions"][0]
        self.assertEqual((session["id"], session["turn_count"], session["failed"]), ("s_a1", 5, 1))

        turns = {t["run_id"]: t for t in session["turns"]}
        self.assertEqual(turns["R2"]["input"], "sold 3 filters")
        self.assertEqual(turns["R2"]["tools"][0]["name"], "record_sale")
        self.assertEqual(turns["R2"]["latency"], 4.0)
        self.assertIn("corrected", {f["key"] for f in turns["R2"]["flags"]})
        self.assertEqual(turns["R5"]["status"], "failed")
        self.assertFalse(turns["R5"]["tools"][0]["ok"])
        self.assertEqual(turns["R4"]["rating"], -1)

    def test_user_detail_excludes_other_users(self):
        detail = admin_metrics.user_detail(BOB, days=30, now=NOW)
        self.assertEqual(detail["totals"]["turns"], 2)
        self.assertNotIn("R1", [t["run_id"] for s in detail["sessions"] for t in s["turns"]])
        self.assertEqual(detail["quality"]["clarification"]["count"], 1)

    def test_unattributed_history_gets_inferred_sessions(self):
        detail = admin_metrics.user_detail(admin_metrics.UNATTRIBUTED_KEY, days=30, now=NOW)
        self.assertEqual(detail["user"]["label"], "(unattributed)")
        self.assertEqual(detail["totals"]["turns"], 3)
        self.assertEqual(len(detail["sessions"]), 2)
        self.assertTrue(all(s["inferred"] for s in detail["sessions"]))

    def test_unknown_user_is_an_empty_page_not_an_error(self):
        detail = admin_metrics.user_detail("nobody@example.com", days=30, now=NOW)
        self.assertEqual(detail["totals"]["turns"], 0)
        self.assertEqual(detail["sessions"], [])

    def test_window_filters_old_activity(self):
        recent = admin_metrics.overview(days=1, now=NOW)
        self.assertEqual(recent["usage"]["turns"], 3)  # Bob's two and the shared token's, all on Nov 4
        self.assertAlmostEqual(recent["cost"]["total_usd"], 0.002, places=6)

        everything = admin_metrics.overview(days=0, now=NOW)
        self.assertEqual(everything["usage"]["turns"], 11)

    def test_an_empty_shop_does_not_break_the_dashboard(self):
        for path in (self.transcript, self.usage_db, self.monitoring_db, self.inventory):
            path.unlink(missing_ok=True)

        result = admin_metrics.overview(days=30, now=NOW)
        self.assertEqual(result["usage"]["turns"], 0)
        self.assertIsNone(result["quality"]["failure_rate"])
        self.assertEqual(result["cost"]["total_usd"], 0)
        self.assertFalse(result["data_health"]["available"])


class RunLoggerAttributionTests(TempDataCase):
    def fake_response(self, content="hello"):
        return AttrDict(
            message=AttrDict(content=content, tool_calls=[]),
            usage={"prompt_tokens": 1000, "completion_tokens": 200},
        )

    def test_llm_and_tool_calls_carry_the_user_and_session(self):
        run = RunLogger("hi", user=ALICE, session_id="s_x", endpoint="chat")
        run.log_tool_call("search_items", {"query": "oil"}, {"success": True, "output": [{"name": "a"}, {"name": "b"}]})
        run.log_tool_call("set_price", {"item": "Oil Filter", "sale_price": 9}, {"success": False, "error": "boom"})
        run.log_llm_call("responder", "openai/gpt-4o-mini", self.fake_response(), request_messages=[])
        run.finalize("completed")

        llm = rows(self.usage_db, "SELECT * FROM llm_calls")[0]
        self.assertEqual((llm["user_email"], llm["session_id"], llm["run_id"]), (ALICE, "s_x", run.run_id))
        self.assertIsNotNone(llm["duration_seconds"])

        tools = rows(self.monitoring_db, "SELECT * FROM tool_calls ORDER BY id")
        self.assertEqual([(t["tool_name"], t["success"], t["result_count"], t["endpoint"], t["user_email"]) for t in tools],
                         [("search_items", 1, 2, "chat", ALICE), ("set_price", 0, None, "chat", ALICE)])
        self.assertEqual(tools[1]["item"], "Oil Filter")
        self.assertEqual(tools[1]["error"], "boom")

        trace = json.loads((self.tmp / "logs" / f"{run.run_id}.json").read_text())
        self.assertEqual((trace["user_email"], trace["session_id"], trace["endpoint"]), (ALICE, "s_x", "chat"))

    def test_a_logger_without_a_user_still_works(self):
        run = RunLogger("hi")
        run.log_llm_call("responder", "openai/gpt-4o-mini", self.fake_response(), request_messages=[])
        self.assertIsNone(rows(self.usage_db, "SELECT user_email FROM llm_calls")[0]["user_email"])

    def test_monitoring_failure_cannot_break_a_tool_call(self):
        run = RunLogger("hi", user=ALICE)
        with mock.patch.object(monitoring, "record_tool_call", side_effect=sqlite3.OperationalError("locked")):
            run.log_tool_call("check_stock", {"item": "x"}, {"success": True, "output": {}})

    def test_turn_detail_reads_the_trace_without_the_request_messages(self):
        run = RunLogger("hi", user=ALICE, session_id="s_x", endpoint="chat")
        run.log_llm_call("responder", "openai/gpt-4o-mini", self.fake_response("SECRET-REPLY"),
                         request_messages=[{"role": "user", "content": "the whole conversation context"}])
        run.log_tool_call("check_stock", {"item": "x"}, {"success": True, "output": {"qty": 1}})
        run.finalize("completed")

        detail = admin_metrics.turn_detail(run.run_id)
        self.assertEqual(detail["user_email"], ALICE)
        self.assertEqual([s["type"] for s in detail["steps"]], ["llm_call", "tool_call"])
        self.assertNotIn("whole conversation context", json.dumps(detail))
        self.assertIsNone(admin_metrics.turn_detail("20250101T000000Z_deadbeef"))
        self.assertIsNone(admin_metrics.turn_detail("../../etc/passwd"))


# --- HTTP -------------------------------------------------------------------


class AdminServerTests(unittest.TestCase):
    ALLOWED = {ALICE, BOB}
    ADMINS = {ALICE}
    GOOGLE_ON = True

    @classmethod
    def setUpClass(cls):
        import agent_server

        cls.agent_server = agent_server
        cls.tmp = Path(tempfile.mkdtemp(prefix="shop-test-"))
        cls.patches = data_patches(cls.tmp) + [
            mock.patch.object(agent_server, "AGENT_API_TOKEN", TOKEN),
            mock.patch.object(agent_server, "GOOGLE_AUTH_ENABLED", cls.GOOGLE_ON),
            mock.patch.object(agent_server, "GOOGLE_CLIENT_ID", "client-id.apps.googleusercontent.com"),
            mock.patch.object(agent_server, "ALLOWED_GOOGLE_EMAILS", cls.ALLOWED),
            mock.patch.object(agent_server, "ADMIN_GOOGLE_EMAILS", cls.ADMINS),
            mock.patch.object(agent_server, "SESSION_SECRET", SECRET),
        ]
        for patch in cls.patches:
            patch.start()

        cls.server = HTTPServer(("127.0.0.1", 0), agent_server.Handler)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        for patch in reversed(cls.patches):
            patch.stop()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        self.agent_server.histories.clear()
        for name in ("monitoring.db", "usage.db", "transcripts.jsonl", "inventory.db"):
            (self.tmp / name).unlink(missing_ok=True)
        for trace in (self.tmp / "logs").glob("*.json"):
            trace.unlink()

    # -- helpers ------------------------------------------------------------

    def cookie(self, email):
        return {"Cookie": f"shop_session={google_auth.make_session(email, SECRET, 3600)}"}

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        data = json.dumps(body).encode() if body is not None else None
        conn.request(method, path, body=data, headers=headers or {})
        response = conn.getresponse()
        raw = response.read()
        conn.close()
        try:
            return response, json.loads(raw) if raw else None
        except ValueError:
            return response, raw.decode("utf-8", "replace")

    def write(self, method, path, email, body=None, csrf=True):
        headers = {**self.cookie(email), "Content-Type": "application/json"}
        if csrf:
            headers["X-Requested-With"] = "shop-ui"
        return self.request(method, path, body or {}, headers)

    def bearer(self):
        return {"Authorization": f"Bearer {TOKEN}"}

    def db(self, name, sql, params=()):
        return rows(self.tmp / name, sql, params)

    def chat(self, email, message, input_mode=None, seen=None):
        agent_server = self.agent_server

        def fake_respond(user_input, history=None, **_):
            logger = agent_server.responder.logger
            logger.log_llm_call(
                "responder", "openai/gpt-4o-mini",
                AttrDict(message=AttrDict(content="done", tool_calls=[]),
                         usage={"prompt_tokens": 500, "completion_tokens": 50}),
                request_messages=[],
            )
            logger.log_tool_call("search_items", {"query": "oil"}, {"success": True, "output": [{"name": "Oil Filter"}]})
            if seen is not None:
                seen.append(list(history or []))
            return {"reply": f"echo: {user_input}", "image_path": None, "items": None, "pending_new_item": None}

        body = {"message": message}
        if input_mode:
            body["input_mode"] = input_mode

        headers = {**(self.cookie(email) if email else self.bearer()),
                   "Content-Type": "application/json", "X-Requested-With": "shop-ui"}

        with mock.patch.object(agent_server.responder, "respond", side_effect=fake_respond), \
                mock.patch.object(agent_server, "compact_history", side_effect=lambda h, **kw: h):
            return self.request("POST", "/chat", body, headers)

    # -- who may open the admin pages --------------------------------------

    def test_admin_page_and_api_status_codes(self):
        for path in ("/admin", "/admin/api/overview", f"/admin/api/user?email={ALICE}"):
            response, _ = self.request("GET", path)
            self.assertEqual(response.status, 401, f"{path} signed out")

            response, _ = self.request("GET", path, headers=self.bearer())
            self.assertEqual(response.status, 403, f"{path} with only the shared token")

            response, _ = self.request("GET", path, headers={"Authorization": "Bearer wrong"})
            self.assertEqual(response.status, 401, f"{path} wrong token")

            response, _ = self.request("GET", path, headers=self.cookie(BOB))
            self.assertEqual(response.status, 403, f"{path} signed in, not admin")

            response, _ = self.request("GET", path, headers=self.cookie(ALICE))
            self.assertEqual(response.status, 200, f"{path} admin")

    def test_refusals_for_the_page_are_html_and_for_the_api_are_json(self):
        response, body = self.request("GET", "/admin", headers=self.cookie(BOB))
        self.assertIn("text/html", response.getheader("Content-Type"))
        self.assertIn("Admins only", body)

        response, body = self.request("GET", "/admin/api/overview", headers=self.cookie(BOB))
        self.assertEqual(body, {"error": "forbidden"})

    def test_admin_page_is_html_with_no_caching(self):
        response, body = self.request("GET", "/admin", headers=self.cookie(ALICE))
        self.assertIn("text/html", response.getheader("Content-Type"))
        self.assertEqual(response.getheader("Cache-Control"), "no-store, must-revalidate")
        self.assertIn("<html", body)

    def test_a_forged_cookie_is_not_an_admin(self):
        forged = google_auth.make_session(ALICE, "some-other-secret", 3600)
        response, _ = self.request("GET", "/admin/api/overview", headers={"Cookie": f"shop_session={forged}"})
        self.assertEqual(response.status, 401)

    def test_admin_removed_from_the_allowlist_loses_access_immediately(self):
        headers = self.cookie(ALICE)
        with mock.patch.object(self.agent_server, "ALLOWED_GOOGLE_EMAILS", {BOB}):
            response, _ = self.request("GET", "/admin/api/overview", headers=headers)
        self.assertEqual(response.status, 401)

    def test_with_no_admins_configured_nobody_gets_in(self):
        with mock.patch.object(self.agent_server, "ADMIN_GOOGLE_EMAILS", set()):
            for headers in (self.cookie(ALICE), self.cookie(BOB), self.bearer()):
                response, _ = self.request("GET", "/admin/api/overview", headers=headers)
                self.assertEqual(response.status, 403)

    def test_with_google_sign_in_off_the_cookie_is_ignored_and_the_token_is_not_admin(self):
        with mock.patch.object(self.agent_server, "GOOGLE_AUTH_ENABLED", False):
            response, _ = self.request("GET", "/admin/api/overview", headers=self.cookie(ALICE))
            self.assertEqual(response.status, 401)
            response, _ = self.request("GET", "/admin/api/overview", headers=self.bearer())
            self.assertEqual(response.status, 403)

    def test_refused_admin_requests_leave_a_trail(self):
        self.request("GET", "/admin/api/overview", headers=self.cookie(BOB))
        self.request("GET", "/admin/api/overview")
        failures = self.db("monitoring.db", "SELECT status_code, email FROM auth_failures ORDER BY id")
        self.assertEqual([(f["status_code"], f["email"]) for f in failures], [(403, BOB), (401, None)])

    def test_me_reports_admin_only_to_admins(self):
        _, bob = self.request("GET", "/auth/me", headers=self.cookie(BOB))
        _, alice = self.request("GET", "/auth/me", headers=self.cookie(ALICE))
        self.assertEqual(bob, {"signed_in": True, "email": BOB})
        self.assertEqual(alice, {"signed_in": True, "email": ALICE, "admin": True})

    def test_cross_user_endpoints_are_closed_to_non_admins(self):
        for path in ("/usage", "/transcript", "/stats"):
            response, _ = self.request("GET", path, headers=self.cookie(BOB))
            self.assertEqual(response.status, 403, path)
            for headers in (self.cookie(ALICE), self.bearer()):
                response, _ = self.request("GET", path, headers=headers)
                self.assertEqual(response.status, 200, path)

    def test_admin_api_validates_its_arguments(self):
        headers = self.cookie(ALICE)
        self.assertEqual(self.request("GET", "/admin/api/user", headers=headers)[0].status, 400)
        self.assertEqual(self.request("GET", "/admin/api/turn?run_id=../../x", headers=headers)[0].status, 400)
        self.assertEqual(self.request("GET", "/admin/api/turn?run_id=20250101T000000Z_deadbeef", headers=headers)[0].status, 404)
        self.assertEqual(self.request("GET", "/admin/api/nothing", headers=headers)[0].status, 404)
        # A junk window falls back to the default instead of failing.
        response, body = self.request("GET", "/admin/api/overview?days=abc", headers=headers)
        self.assertEqual((response.status, body["window"]["days"]), (200, 30))
        response, body = self.request("GET", "/admin/api/overview?days=0", headers=headers)
        self.assertEqual(body["window"]["days"], 0)

    def test_viewing_the_dashboard_is_not_shop_activity(self):
        self.request("GET", "/admin", headers=self.cookie(ALICE))
        self.request("GET", "/admin/api/overview", headers=self.cookie(ALICE))
        self.assertEqual(self.db("monitoring.db", "SELECT * FROM sessions"), [])

    # -- attribution --------------------------------------------------------

    def test_a_chat_turn_is_attributed_everywhere(self):
        response, body = self.chat(ALICE, "oil filter kitna hai", input_mode="voice")
        self.assertEqual(response.status, 200)
        run_id = body["run_id"]

        session = self.db("monitoring.db", "SELECT * FROM sessions")
        self.assertEqual([s["user_email"] for s in session], [ALICE])
        session_id = session[0]["id"]

        turn = json.loads((self.tmp / "transcripts.jsonl").read_text().splitlines()[0])
        self.assertEqual((turn["user"], turn["session_id"], turn["run_id"], turn["input_mode"], turn["kind"]),
                         (ALICE, session_id, run_id, "voice", "chat"))
        self.assertGreaterEqual(turn["latency_seconds"], 0)

        llm = self.db("usage.db", "SELECT * FROM llm_calls")[0]
        self.assertEqual((llm["user_email"], llm["session_id"], llm["run_id"]), (ALICE, session_id, run_id))

        tool = self.db("monitoring.db", "SELECT * FROM tool_calls")[0]
        self.assertEqual((tool["user_email"], tool["run_id"], tool["endpoint"]), (ALICE, run_id, "chat"))

        outcome = self.db("monitoring.db", "SELECT * FROM request_outcomes")[0]
        self.assertEqual((outcome["user_email"], outcome["session_id"], outcome["run_id"], outcome["status"]),
                         (ALICE, session_id, run_id, "completed"))

        trace = json.loads((self.tmp / "logs" / f"{run_id}.json").read_text())
        self.assertEqual(trace["user_email"], ALICE)

    def test_the_shared_token_is_its_own_identity(self):
        self.chat(None, "stock?")
        turn = json.loads((self.tmp / "transcripts.jsonl").read_text().splitlines()[0])
        self.assertEqual(turn["user"], BEARER)

    def test_bad_input_mode_is_recorded_as_unknown(self):
        self.chat(ALICE, "hello", input_mode="telepathy")
        turn = json.loads((self.tmp / "transcripts.jsonl").read_text().splitlines()[0])
        self.assertIsNone(turn["input_mode"])

    def test_each_user_has_their_own_conversation_history(self):
        seen = []
        self.chat(ALICE, "alice one", seen=seen)
        self.chat(BOB, "bob one", seen=seen)
        self.chat(ALICE, "alice two", seen=seen)

        self.assertEqual(seen[0], [])
        self.assertEqual(seen[1], [])  # Bob does not inherit Alice's context
        self.assertEqual([t["user_input"] for t in seen[2]], ["alice one"])

    def test_clearing_history_only_clears_the_callers_own(self):
        self.chat(ALICE, "alice one")
        self.chat(BOB, "bob one")
        self.write("DELETE", "/history", ALICE)
        self.assertNotIn(ALICE, self.agent_server.histories)
        self.assertIn(BOB, self.agent_server.histories)

    def test_two_users_get_two_sessions(self):
        self.chat(ALICE, "a")
        self.chat(BOB, "b")
        self.assertEqual(sorted(s["user_email"] for s in self.db("monitoring.db", "SELECT * FROM sessions")), [ALICE, BOB])

    def test_table_edit_and_import_cancel_are_attributed(self):
        inventory_db.add_item("Oil Filter", initial_quantity=5)
        response, _ = self.write("POST", "/inventory/adjust-stock", BOB,
                                 {"item": "Oil Filter", "new_quantity": 7, "reason": "miscount"})
        self.assertEqual(response.status, 200)
        tool = self.db("monitoring.db", "SELECT * FROM tool_calls")[0]
        self.assertEqual((tool["tool_name"], tool["endpoint"], tool["user_email"], tool["item"]),
                         ("adjust_stock", "table_adjust_stock", BOB, "Oil Filter"))

        self.write("POST", "/import/cancel", BOB, {"item_count": 4})
        imported = self.db("monitoring.db", "SELECT * FROM import_outcomes")[0]
        self.assertEqual((imported["outcome"], imported["user_email"], imported["item_count"]), ("cancelled", BOB, 4))

    def test_sign_in_outcomes_are_recorded(self):
        def sign_in(email):
            with mock.patch.object(self.agent_server, "verify_google_credential", return_value=email):
                return self.request("POST", "/auth/google", {"credential": "x.y.z"}, {"Content-Type": "application/json"})

        self.assertEqual(sign_in(ALICE)[0].status, 200)
        self.assertEqual(sign_in("evil@example.com")[0].status, 403)
        with mock.patch.object(self.agent_server, "verify_google_credential", side_effect=ValueError("bad")):
            self.request("POST", "/auth/google", {"credential": "x.y.z"}, {"Content-Type": "application/json"})

        outcomes = [(r["email"], r["outcome"]) for r in self.db("monitoring.db", "SELECT * FROM sign_ins ORDER BY id")]
        self.assertEqual(outcomes, [(ALICE, "ok"), ("evil@example.com", "blocked"), (None, "invalid")])

        failures = self.db("monitoring.db", "SELECT status_code, email FROM auth_failures ORDER BY id")
        self.assertEqual([(f["status_code"], f["email"]) for f in failures], [(403, "evil@example.com"), (401, None)])
        self.assertEqual([s["user_email"] for s in self.db("monitoring.db", "SELECT * FROM sessions")], [ALICE])

    # -- feedback (a cookie-authenticated write) ----------------------------

    def test_feedback_follows_the_same_csrf_rule_as_other_writes(self):
        run_id = "20260101T000000Z_abcdef12"
        response, _ = self.write("POST", "/feedback", BOB, {"run_id": run_id, "rating": "down"}, csrf=False)
        self.assertEqual(response.status, 401)
        self.assertEqual(self.db("monitoring.db", "SELECT * FROM feedback"), [])

        response, _ = self.write("POST", "/feedback", BOB, {"run_id": run_id, "rating": "down"})
        self.assertEqual(response.status, 200)

        response, _ = self.request("POST", "/feedback", {"run_id": run_id, "rating": "up"},
                                   {**self.bearer(), "Content-Type": "application/json"})
        self.assertEqual(response.status, 200)  # a bearer caller needs no CSRF header

    def test_voting_again_replaces_the_earlier_vote(self):
        run_id = "20260101T000000Z_abcdef12"
        self.write("POST", "/feedback", BOB, {"run_id": run_id, "rating": "down"})
        self.write("POST", "/feedback", BOB, {"run_id": run_id, "rating": "up"})
        stored = self.db("monitoring.db", "SELECT user_email, rating FROM feedback")
        self.assertEqual(stored, [{"user_email": BOB, "rating": 1}])

    def test_feedback_input_is_validated(self):
        for body in [{"run_id": "nope", "rating": "up"}, {"rating": "up"}, {"run_id": ["x"], "rating": "up"}]:
            self.assertEqual(self.write("POST", "/feedback", BOB, body)[0].status, 400, body)
        for rating in ("meh", 5, None, ["up"], {"a": 1}):
            body = {"run_id": "20260101T000000Z_abcdef12", "rating": rating}
            self.assertEqual(self.write("POST", "/feedback", BOB, body)[0].status, 400, body)

    def test_the_dashboard_reflects_real_traffic_end_to_end(self):
        self.chat(ALICE, "oil filter kitna hai", input_mode="text")
        self.chat(BOB, "air filter?", input_mode="voice")

        response, body = self.request("GET", "/admin/api/overview?days=1", headers=self.cookie(ALICE))
        self.assertEqual(response.status, 200)
        self.assertEqual(body["usage"]["turns"], 2)
        self.assertEqual(body["usage"]["sessions"], 2)
        self.assertEqual(body["adoption"]["input_mode"]["voice"], 1)
        self.assertEqual({u["email"] for u in body["users"] if u["turns"]}, {ALICE, BOB})
        self.assertGreater(body["cost"]["total_usd"], 0)

        response, detail = self.request("GET", f"/admin/api/user?email={ALICE}&days=1", headers=self.cookie(ALICE))
        self.assertEqual(detail["sessions"][0]["turns"][0]["input"], "oil filter kitna hai")
        self.assertEqual(detail["sessions"][0]["turns"][0]["tools"][0]["name"], "search_items")

        run_id = detail["sessions"][0]["turns"][0]["run_id"]
        response, trace = self.request("GET", f"/admin/api/turn?run_id={run_id}", headers=self.cookie(ALICE))
        self.assertEqual(response.status, 200)
        self.assertEqual(trace["user_email"], ALICE)


if __name__ == "__main__":
    unittest.main()
