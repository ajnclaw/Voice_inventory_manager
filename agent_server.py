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

import os

os.environ.setdefault("AGENT_AUTO_APPROVE", "1")

import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from agent.compactor import compact_history
from agent.config import DEFAULT_MODEL, LLM_PROVIDER
from agent.logger import RunLogger
from agent.responder import Responder

HOST = "0.0.0.0"
PORT = 8766

KEEP_RECENT_TURNS = 5
COMPACT_TRIGGER_TURNS = 10

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

    def do_GET(self):
        if self.path == "/health":
            self._send_json({"status": "ok", "turns": len(history)})
        else:
            self._send_json({"error": "not found"}, status=404)

    def do_POST(self):
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
