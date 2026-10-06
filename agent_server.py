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
    .venv/bin/python3 agent_server.py   # listens on 0.0.0.0:8766, run
                                  # from the project root (cwd
                                  # determines config.PROJECT_ROOT /
                                  # where inventory.db and logs/ are
                                  # created). Needs the venv (not bare
                                  # system python3) since bulk import
                                  # depends on PyMuPDF -- see
                                  # requirements.txt.
"""

import base64
import hmac
import os

os.environ.setdefault("AGENT_AUTO_APPROVE", "1")

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from agent.compactor import compact_history
from agent.config import AGENT_API_TOKEN, DEFAULT_MODEL, LLM_PROVIDER, PROJECT_ROOT
from agent.import_extractor import (
    apply_import_items,
    extract_from_image,
    extract_from_pdf,
    extract_from_text,
)
from agent.inventory_db import inventory_overview
from agent.logger import RunLogger
from agent import monitoring
from agent.responder import Responder
from agent.transcript_log import append_turn, read_all as read_transcript
from agent import usage_tracker

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

IMAGE_DIR = PROJECT_ROOT / "product_images"

IMAGE_CONTENT_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}

responder = Responder()
history = []


def _serialize_item_list(items):
    """
    Turns a multi-item tool result (search_items/list_inventory/
    list_low_stock, see responder.py's last_item_list) into what the
    web UI's chat-reply table needs: image_path -> image_url (same
    "/product_images/<filename>" shape as the single-item image_url
    field, so the frontend's existing authenticated-image fetch works
    unchanged), sorted by category so related items group together
    the way the owner actually asked for, rather than whatever order
    the tool happened to return them in.
    """
    if not items:
        return None

    def sort_key(item):
        return (
            (item.get("category") or "").lower(),
            (item.get("name") or "").lower(),
        )

    serialized = []

    for item in sorted(items, key=sort_key):
        image_path = item.get("image_path")
        serialized.append(
            {
                "name": item.get("name"),
                "category": item.get("category"),
                "current_quantity": item.get("current_quantity"),
                "cost_price": item.get("cost_price"),
                "image_url": f"/product_images/{image_path.split('/')[-1]}"
                if image_path
                else None,
            }
        )

    return serialized


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

        # Silent before this -- a wrong/missing token just got a 401
        # with no record anywhere. Harmless while this only sits behind
        # a private adb tunnel, but once it's a public EC2 endpoint,
        # someone probing it with bad tokens should leave a trail.
        monitoring.record_auth_failure(self.path, self.client_address[0])
        self._send_json({"error": "unauthorized"}, status=401)
        return False

    def _send_html(self, body_text, status=200):
        body = body_text.encode("utf-8")

        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        # Without this, mobile Chrome has repeatedly served a stale
        # cached copy of this page after a real update on the server --
        # confirmed twice during development. This is a single-owner
        # tool mid-active-development, not a high-traffic site; the
        # cost of never caching the page is irrelevant next to the
        # cost of the owner not seeing a fix that's actually live.
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path, content_type):
        body = path.read_bytes()

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send_html(WEB_INDEX)
            return

        if not self._require_auth():
            return

        if self.path.startswith("/product_images/"):
            self._handle_product_image()
            return

        if self.path == "/health":
            self._send_json({"status": "ok", "turns": len(history)})
        elif self.path == "/inventory":
            # Dedicated structured view, bypassing the LLM entirely --
            # a real table + totals is more reliably "human readable"
            # than hoping a chat reply formats a long list nicely, and
            # it's free (no LLM call) and always exactly up to date.
            self._send_json(inventory_overview())
        elif self.path == "/usage":
            # Cumulative LLM spend across this deployment's whole
            # lifetime (persisted in usage.db, survives restarts) --
            # see agent/usage_tracker.py. Numbers are an estimate from
            # a local pricing table, not pulled live from the provider.
            self._send_json(usage_tracker.totals())
        elif self.path == "/transcript":
            # The continuous saved conversation log -- see
            # agent/transcript_log.py. Every /chat and /upload turn,
            # in order, with model/tokens/cost per turn.
            self._send_json({"turns": read_transcript()})
        elif self.path == "/stats":
            # Backend-only system-health view: auth failures, request
            # failure rate, bulk-import accept/reject rate. Deliberately
            # never called from web/index.html -- this is for checking
            # on the system, not something the shop owner's app shows.
            self._send_json(monitoring.stats())
        else:
            self._send_json({"error": "not found"}, status=404)

    def do_POST(self):
        if not self._require_auth():
            return

        if self.path == "/chat":
            self._handle_chat()
        elif self.path == "/upload":
            self._handle_upload()
        elif self.path == "/import/confirm":
            self._handle_import_confirm()
        elif self.path == "/import/cancel":
            self._handle_import_cancel()
        elif self.path == "/inventory/adjust-stock":
            self._handle_inventory_tool("adjust_stock")
        elif self.path == "/inventory/set-price":
            self._handle_inventory_tool("set_price")
        else:
            self._send_json({"error": "not found"}, status=404)

    def _read_json_body(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw)

    def _handle_chat(self):
        try:
            body = self._read_json_body()
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

        image_path = None
        items = None

        try:
            result = responder.respond(user_input, history=history)
            reply = result["reply"]
            image_path = result.get("image_path")
            items = result.get("items")
            status = "completed"
        except Exception as exc:
            reply = f"Something went wrong: {exc}"
            status = "failed"

        run_logger.finalize(status)
        monitoring.record_request_outcome("chat", status)

        # The continuous, human-readable conversation record (separate
        # from the detailed per-request trace run_logger just wrote) --
        # see transcript_log.py. A turn can involve more than one LLM
        # call (tool-calling loop iterations), so these are the run's
        # summed totals, not a single call's.
        append_turn(
            user_input,
            reply,
            status,
            DEFAULT_MODEL,
            run_logger.trace["total_prompt_tokens"],
            run_logger.trace["total_completion_tokens"],
            run_logger.trace["total_cost_usd"],
        )

        history.append({"user_input": user_input, "reply": reply})
        history = compact_history(
            history,
            keep_recent=KEEP_RECENT_TURNS,
            trigger_at=COMPACT_TRIGGER_TURNS,
            logger=run_logger,
        )

        self._send_json(
            {
                "reply": reply,
                "status": status,
                "image_url": f"/product_images/{image_path.split('/')[-1]}"
                if image_path
                else None,
                "items": _serialize_item_list(items),
            }
        )

    def _handle_product_image(self):
        """
        Serves a product picture (see agent/pdf_images.py -- saved
        under product_images/ during a catalog import, path stored on
        the item as image_path). Auth-gated like everything else;
        since a plain <img src> can't send an Authorization header,
        the web UI fetches this with a real authenticated request and
        turns the response into a blob URL instead -- see
        web/index.html's loadAuthenticatedImage().

        Filename only, no path separators -- this serves exclusively
        out of IMAGE_DIR, never anything else on disk.
        """
        filename = self.path[len("/product_images/"):]

        if "/" in filename or "\\" in filename or ".." in filename:
            self._send_json({"error": "invalid filename"}, status=400)
            return

        path = IMAGE_DIR / filename

        if not path.is_file():
            self._send_json({"error": "not found"}, status=404)
            return

        content_type = IMAGE_CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")
        self._send_file(path, content_type)

    def _handle_upload(self):
        """
        Reads a file (text/CSV, PDF, or photo) and extracts candidate
        inventory actions from it -- does NOT write anything to
        inventory.db. See agent/import_extractor.py. The owner reviews
        the returned items in the web UI and POSTs them to
        /import/confirm to actually apply them.
        """
        try:
            body = self._read_json_body()
        except json.JSONDecodeError:
            self._send_json({"error": "invalid JSON body"}, status=400)
            return

        filename = (body.get("filename") or "").lower()
        mime_type = body.get("mime_type") or ""
        content_b64 = body.get("content_base64") or ""
        # Optional: "buy" or "sale", set by the owner in the web UI
        # before picking the file. When present this overrides the
        # model's own judgment about transaction direction entirely --
        # see import_extractor.py's HINT_INSTRUCTIONS.
        hint = body.get("hint") or None

        if not content_b64:
            self._send_json({"error": "missing 'content_base64'"}, status=400)
            return

        try:
            raw_bytes = base64.b64decode(content_b64)
        except Exception:
            self._send_json({"error": "invalid base64 content"}, status=400)
            return

        run_logger = RunLogger(f"[upload: {filename or mime_type or 'unnamed file'}]")

        try:
            if filename.endswith(".pdf") or mime_type == "application/pdf":
                result = extract_from_pdf(raw_bytes, logger=run_logger, hint=hint)
            elif mime_type.startswith("image/") or filename.endswith(
                (".jpg", ".jpeg", ".png", ".webp", ".heic")
            ):
                result = extract_from_image(
                    raw_bytes, mime_type or "image/jpeg", logger=run_logger, hint=hint
                )
            else:
                # Plain text/CSV -- anything else falls through here too,
                # which just means the model sees raw bytes decoded as
                # text and likely returns an empty/explained result
                # rather than crashing.
                text = raw_bytes.decode("utf-8", errors="replace")
                result = extract_from_text(text, logger=run_logger, hint=hint)
        except Exception as exc:
            run_logger.log_event(
                "upload_failed", f"Extraction raised: {exc}", level="error"
            )
            run_logger.finalize("failed")
            monitoring.record_request_outcome("upload", "failed")
            self._send_json({"error": f"Couldn't read that file: {exc}"}, status=500)
            return

        run_logger.log_event(
            "extraction_result",
            f"Extracted {len(result.get('items', []))} item(s): {result.get('summary', '')}",
        )
        run_logger.finalize("completed")
        monitoring.record_request_outcome("upload", "completed")

        append_turn(
            f"[upload: {filename or mime_type or 'unnamed file'}]",
            result.get("summary", ""),
            "completed",
            DEFAULT_MODEL,
            run_logger.trace["total_prompt_tokens"],
            run_logger.trace["total_completion_tokens"],
            run_logger.trace["total_cost_usd"],
        )

        self._send_json(result)

    def _handle_import_confirm(self):
        """
        Actually applies a list of extracted items (as returned by
        /upload, after the owner has reviewed/confirmed them in the web
        UI) -- the only place in the import flow that writes real
        inventory data.
        """
        try:
            body = self._read_json_body()
        except json.JSONDecodeError:
            self._send_json({"error": "invalid JSON body"}, status=400)
            return

        items = body.get("items")

        if not isinstance(items, list) or not items:
            self._send_json({"error": "missing/empty 'items' list"}, status=400)
            return

        run_logger = RunLogger("[bulk import confirm]")
        responder.tool_manager.set_logger(run_logger)

        results = apply_import_items(items, responder.tool_manager)

        run_logger.finalize("completed")
        monitoring.record_import_outcome("confirmed", len(items))

        self._send_json({"results": results})

    def _handle_inventory_tool(self, tool_name):
        """
        Direct stock/price edits from the inventory table view -- same
        "bypass the LLM entirely" reasoning as GET /inventory itself:
        this is a structured, deterministic edit the owner is making
        right there in the table, not a natural-language request that
        needs resolving, so routing it through a whole chat turn would
        just be slower and less certain for no benefit. Goes through
        the same tool_manager as the chat flow (not inventory_db
        directly), so it gets the same logging and stays the single
        choke point for every inventory-changing action.
        """
        try:
            body = self._read_json_body()
        except json.JSONDecodeError:
            self._send_json({"error": "invalid JSON body"}, status=400)
            return

        item = (body.get("item") or "").strip()

        if not item:
            self._send_json({"error": "missing 'item'"}, status=400)
            return

        arguments = {"item": item}

        if tool_name == "adjust_stock":
            arguments["new_quantity"] = body.get("new_quantity")
            arguments["reason"] = body.get("reason")
        elif tool_name == "set_price":
            arguments["cost_price"] = body.get("cost_price")
            arguments["sale_price"] = body.get("sale_price")

        run_logger = RunLogger(f"[inventory table: {tool_name}]")
        responder.tool_manager.set_logger(run_logger)

        result = responder.tool_manager.execute(tool_name, arguments)

        run_logger.finalize("completed" if result.get("success") else "failed")
        monitoring.record_request_outcome(
            f"inventory_{tool_name}", "completed" if result.get("success") else "failed"
        )

        if result.get("success"):
            self._send_json({"output": result.get("output")})
        else:
            self._send_json({"error": result.get("error") or "failed"}, status=400)

    def _handle_import_cancel(self):
        """
        Fire-and-forget signal from the web UI: the owner reviewed an
        extracted preview and rejected it rather than saving it. This
        is the single most direct quality measurement the bulk-import
        feature has (see monitoring.py) -- it's never surfaced back to
        the owner, purely a backend record of how often extraction is
        actually good enough to use.
        """
        try:
            body = self._read_json_body()
        except json.JSONDecodeError:
            body = {}

        item_count = body.get("item_count")
        monitoring.record_import_outcome("cancelled", item_count)
        self._send_json({"status": "noted"})

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
    print('POST /chat {"message": "..."} -> {"reply", "status", "image_url"?, "items"?}')
    print('POST /upload {"filename", "mime_type", "content_base64", "hint"?} -> extracted items (no writes)')
    print('POST /import/confirm {"items": [...]} -> applies them, returns per-item results')
    print("GET /inventory -> full catalog + summary totals (table view)")
    print('POST /inventory/adjust-stock {"item", "new_quantity", "reason"} -> direct stock edit')
    print('POST /inventory/set-price {"item", "cost_price"?, "sale_price"?} -> direct price edit')
    print("GET /usage -> cumulative LLM calls/tokens/estimated cost, all time")
    print("GET /transcript -> saved conversation log (every /chat and /upload turn)")
    print("GET /stats -> backend-only: auth failures, failure rate, import reject rate")
    print("DELETE /history -> clears conversation history")
    print("AGENT_AUTO_APPROVE=" + os.environ.get("AGENT_AUTO_APPROVE", "0"))

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
