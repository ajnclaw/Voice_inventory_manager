import json
import logging
import time
import uuid
from datetime import datetime, timezone

from .config import PROJECT_ROOT
from .pricing import estimate_cost
from . import usage_tracker


LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def _new_run_id():
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_{uuid.uuid4().hex[:8]}"


def _sanitize_for_log(messages):
    """
    The full request -- every system/user/tool message, exactly as
    sent -- gets logged for real auditability (see log_llm_call). The
    one thing deliberately stripped out is a vision call's base64
    image data: a single photo upload can be a few MB of base64, and
    writing that into every trace file would bloat logs/ fast for zero
    benefit (the image itself isn't useful to review later the way the
    text around it is). Everything else is kept verbatim.
    """
    if not messages:
        return []

    sanitized = []

    for message in messages:
        content = message.get("content")

        if isinstance(content, list):
            new_content = []

            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    new_content.append({"type": "image_url", "image_url": "[omitted: base64 image data]"})
                else:
                    new_content.append(part)

            sanitized.append({**message, "content": new_content})
        else:
            sanitized.append(message)

    return sanitized


class RunLogger:
    """
    Emits human-readable lines to the console, and builds a structured
    JSON trace of one /chat turn, written to logs/<run_id>.json once the
    turn finishes. Every tool call that changes inventory ends up in
    here -- this doubles as an audit log of who-did-what-when, on top
    of the transactions table itself.

    Simpler than ~/ai-agent's RunLogger: there's no Task/AgentState
    concept here (no planner), so finalize() just takes a plain status
    string instead of serializing a task list.
    """

    def __init__(self, user_input):
        self.run_id = _new_run_id()
        self.started_at = time.time()

        self.logger = logging.getLogger(f"agent.{self.run_id}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"
                )
            )
            self.logger.addHandler(handler)

        self.trace = {
            "run_id": self.run_id,
            "user_input": user_input,
            "events": [],
            "status": None,
            "duration_seconds": None,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_cost_usd": 0.0,
        }

    def log_event(self, event_type, message, level="info", **fields):
        log_fn = getattr(self.logger, level, self.logger.info)
        log_fn(message)

        self.trace["events"].append({
            "type": event_type,
            "elapsed_seconds": round(time.time() - self.started_at, 3),
            "message": message,
            **fields,
        })

    def log_tool_call(self, tool_name, arguments, result):
        self.logger.info(
            "[Tool] %s(%s) -> success=%s",
            tool_name, arguments, result.get("success"),
        )

        self.trace["events"].append({
            "type": "tool_call",
            "elapsed_seconds": round(time.time() - self.started_at, 3),
            "tool_name": tool_name,
            "arguments": arguments,
            "success": result.get("success"),
            "output": result.get("output"),
            "error": result.get("error"),
            "denied": result.get("denied", False),
        })

    def log_llm_call(self, component, model, response, request_messages=None):
        message = response.message
        content = message.content or ""

        tool_calls = [
            {
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments,
            }
            for tool_call in (message.tool_calls or [])
        ]

        usage = getattr(response, "usage", None) or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        cost_usd = estimate_cost(model, prompt_tokens, completion_tokens)

        preview = content if len(content) <= 300 else content[:300] + "..."
        tool_call_names = [tc["name"] for tc in tool_calls]
        cost_text = f"${cost_usd:.5f}" if cost_usd is not None else "cost unknown"

        self.logger.info(
            "[LLM/%s] %s -> %s%s (tokens: %s in / %s out, %s)",
            component,
            model,
            preview or "(no content)",
            f" tool_calls={tool_call_names}" if tool_calls else "",
            prompt_tokens if prompt_tokens is not None else "?",
            completion_tokens if completion_tokens is not None else "?",
            cost_text,
        )

        self.trace["events"].append({
            "type": "llm_call",
            "elapsed_seconds": round(time.time() - self.started_at, 3),
            "component": component,
            "model": model,
            "request_messages": _sanitize_for_log(request_messages),
            "content": content,
            "tool_calls": tool_calls,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": cost_usd,
        })

        self.trace["total_prompt_tokens"] += prompt_tokens or 0
        self.trace["total_completion_tokens"] += completion_tokens or 0
        self.trace["total_cost_usd"] += cost_usd or 0.0

        # Persisted separately from the per-run trace file so "how much
        # have we spent, ever" survives across runs and restarts -- see
        # usage_tracker.py.
        usage_tracker.record_call(
            self.run_id, component, model, prompt_tokens, completion_tokens, cost_usd
        )

    def finalize(self, status):
        self.trace["status"] = status
        self.trace["duration_seconds"] = round(time.time() - self.started_at, 2)
        self.trace["total_cost_usd"] = round(self.trace["total_cost_usd"], 6)

        path = LOG_DIR / f"{self.run_id}.json"
        path.write_text(
            json.dumps(self.trace, indent=2, default=str),
            encoding="utf-8",
        )

        self.logger.info(
            "Run finished with status '%s'. Trace written to %s",
            status, path,
        )

        return path
