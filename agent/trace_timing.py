# trace_timing.py
#
# Per-stage timing from a run trace (logs/<run_id>.json, see logger.py).
# Each event records elapsed_seconds at the moment it FINISHED, so a
# stage's duration is the gap since the previous event. That one rule is
# shared by scripts/trace_summary.py (human-readable breakdown) and by the
# logger itself, which stores the same number on each LLM/tool call row so
# the admin dashboard can report latency without re-reading trace files.


def stage_gap(elapsed_seconds, previous_elapsed_seconds):
    return round(max(0.0, elapsed_seconds - previous_elapsed_seconds), 3)


def stage_durations(events):
    """[(event, seconds)] in order, using the gap-since-previous rule."""
    previous = 0.0
    stages = []

    for event in events:
        stages.append((event, stage_gap(event["elapsed_seconds"], previous)))
        previous = event["elapsed_seconds"]

    return stages
