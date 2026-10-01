"""
agent_server.py

HTTP server for the inventory agent -- a single conversational
responder over inventory tools (see agent/tools.py). No file/shell/
video/memory-fact tools anywhere in reach, and deliberately no
planner/executor/multi-step-plan machinery either: every real request
here is a single resolve-then-act tool-calling turn the responder's own
loop already handles directly (it can call several tools in sequence
within one turn). See the project README for the full reasoning.

Runs with AGENT_AUTO_APPROVE=1 by default: there's no terminal for a
phone call to answer a y/N prompt on, and unlike a slow/irreversible
action, recording a sale/restock/correction IS this agent's routine
job -- see agent/approval_manager.py.

Single-threaded on purpose, same reasoning as ~/ai-agent's server:
Responder.respond() mutates shared component state (the logger) on
every call, so concurrent requests would race. Requests queue instead
-- fine for one shop owner talking to it one turn at a time.

Listens on 0.0.0.0 (not 127.0.0.1) since this is meant to sit behind a
tunnel or reverse proxy reachable from the owner's phone over the
network, not only from localhost on the same machine.

Usage:
    python agent_server.py       # listens on 0.0.0.0:8766, run from
                                  # the project root (cwd determines
                                  # config.PROJECT_ROOT / where
                                  # inventory.db and logs/ are created)
"""

import hmac
import os

os.environ.setdefault("AGENT_AUTO_APPROVE", "1")

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from agent.compactor import compact_history
from agent.config import AGENT_API_TOKEN, DEFAULT_MODEL, LLM_PROVIDER, PROJECT_ROOT
from agent.logger import RunLogger
from agent.responder import Responder

HOST = "0.0.0.0"
PORT = 8766

KEEP_RECENT_TURNS = 5
COMPACT_TRIGGER_TURNS = 10

# The frontend's own HTML/JS is served publicly, with no auth check --
# a browser has to be able to load the page before it can possibly send
# the token back on an API call, so gating the page itself would be
# circular. The token lives only in the browser's localStorage after
# the owner pastes it in once (see web/index.html); it is never baked
# into this file on disk. Everything that actually touches inventory
# data (/chat, /health, DELETE /history) still requires it.
WEB_INDEX = (PROJECT_ROOT / "web" / "index.html").read_text(encoding="utf-8")

responder = Responder()
history = []


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")

        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self):
        """
        Shared-secret check against AGENT_API_TOKEN (see config.py).
        Expects `Authorization: Bearer <token>`. Every endpoint goes
        through this, including /health -- there's no "harmless" public
        endpoint here, and uniform is easier to reason about than
        remembering which routes are exempt. hmac.compare_digest avoids
        leaking the token one byte at a time via a timing side channel.
        """
        header = self.headers.get("Authorization", "")
        prefix = "Bearer "

        if not header.startswith(prefix):
            return False

        provided = header[len(prefix):].strip()

        return hmac.compare_digest(provided, AGENT_API_TOKEN)

    def _require_auth(self):
        if self._authorized():
            return True

        self._send_json({"error": "unauthorized"}, status=401)
        return False

    def _send_html(self, body_text, status=200):
        body = body_text.encode("utf-8")

        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send_html(WEB_INDEX)
            return

        if not self._require_auth():
            return

        if self.path == "/health":
            self._send_json({"status": "ok", "turns": len(history)})
        else:
            self._send_json({"error": "not found"}, status=404)

    def do_POST(self):
        if not self._require_auth():
            return

        if self.path != "/chat":
            self._send_json({"error": "not found"}, status=404)
            return

        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"

        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            self._send_json({"error": "invalid JSON body"}, status=400)
            return

        user_input = (body.get("message") or "").strip()

        if not user_input:
            self._send_json({"error": "missing 'message'"}, status=400)
            return

        global history

        run_logger = RunLogger(user_input)
        responder.set_logger(run_logger)

        try:
            reply = responder.respond(user_input, history=history)
            status = "completed"
        except Exception as exc:
            reply = f"Something went wrong: {exc}"
            status = "failed"

        run_logger.finalize(status)

        history.append({"user_input": user_input, "reply": reply})
        history = compact_history(
            history,
            keep_recent=KEEP_RECENT_TURNS,
            trigger_at=COMPACT_TRIGGER_TURNS,
        )

        self._send_json({"reply": reply, "status": status})

    def do_DELETE(self):
        if not self._require_auth():
            return

        if self.path == "/history":
            global history
            history = []
            self._send_json({"status": "cleared"})
        else:
            self._send_json({"error": "not found"}, status=404)

    def log_message(self, format, *args):
        # Quiet by default -- comment out to see request logs.
        pass


def main():
    if not AGENT_API_TOKEN:
        raise SystemExit(
            "Refusing to start: AGENT_API_TOKEN is not set.\n"
            "This server is meant to sit behind a public EC2/ECS endpoint "
            "-- without a token, every /chat, /health, and /history "
            "request would be answered to anyone on the internet, "
            "including the inventory-changing ones. Set AGENT_API_TOKEN "
            "in .env (see .env.example) before running this."
        )

    server = HTTPServer((HOST, PORT), Handler)

    print(f"Inventory agent listening on http://{HOST}:{PORT}")
    print(f"LLM provider: {LLM_PROVIDER} (model: {DEFAULT_MODEL})")
    print('POST /chat {"message": "..."} -> {"reply": "...", "status": "..."}')
    print("DELETE /history -> clears conversation history")
    print("AGENT_AUTO_APPROVE=" + os.environ.get("AGENT_AUTO_APPROVE", "0"))

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
