"""
Prints a per-stage timing breakdown of /chat traces from logs/.

Usage (from the project root):
    python3 scripts/trace_summary.py            # newest trace
    python3 scripts/trace_summary.py -n 5       # newest 5
    python3 scripts/trace_summary.py <run_id>   # one specific run

Each event in a trace records elapsed_seconds at the moment it
finished, so a stage's duration is the gap since the previous event.
This covers the server side only (LLM calls + tool calls). Browser
speech-to-text before the request and text-to-speech after the reply
happen client-side and never show up here.
"""

import argparse
import json
import sys
from pathlib import Path

# Run as a script, sys.path[0] is scripts/, not the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.trace_timing import stage_durations  # noqa: E402

# The server writes logs/ under its working directory, so read from the
# same place rather than relative to this script.
LOG_DIR = Path.cwd() / "logs"


def summarize(path):
    trace = json.loads(path.read_text(encoding="utf-8"))
    print(f"\n{trace['run_id']}  \"{trace['user_input']}\"")
    print(f"  status={trace['status']}  total={trace['duration_seconds']}s  "
          f"tokens={trace['total_prompt_tokens']} in / {trace['total_completion_tokens']} out")

    for event, gap in stage_durations(trace["events"]):
        if event["type"] == "llm_call":
            calls = [c["name"] for c in event.get("tool_calls", [])]
            detail = (
                f"{event['component']} {event['model']} "
                f"({event['prompt_tokens']} in / {event['completion_tokens']} out)"
                + (f" -> tool_calls={calls}" if calls else " -> reply")
            )
        elif event["type"] == "tool_call":
            detail = f"{event['tool_name']}({event['arguments']}) success={event['success']}"
        else:
            detail = event.get("message", "")

        print(f"  +{gap:6.2f}s  {event['type']:<9}  {detail}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", nargs="?")
    parser.add_argument("-n", type=int, default=1, help="how many newest traces")
    args = parser.parse_args()

    if args.run_id:
        paths = [LOG_DIR / f"{args.run_id}.json"]
    else:
        paths = sorted(LOG_DIR.glob("*.json"))[-args.n:]

    for path in paths:
        summarize(path)


if __name__ == "__main__":
    main()
