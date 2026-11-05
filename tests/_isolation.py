"""
Points every on-disk store the app writes at a throwaway directory, so a test
can run the real server code without touching the shop's real inventory.db,
usage.db, monitoring.db, transcripts.jsonl or logs/.

    cls.tmp = Path(tempfile.mkdtemp(prefix="shop-test-"))
    patches = data_patches(cls.tmp)        # then .start() each, .stop() at the end
"""

from pathlib import Path
from unittest import mock


def data_patches(tmp):
    from agent import inventory_db, logger, monitoring, transcript_log, usage_tracker

    tmp = Path(tmp)
    project = Path(__file__).resolve().parent.parent

    # Guard: never let a suite point at (and so write) the real data.
    assert tmp.resolve() != project and project not in tmp.resolve().parents

    (tmp / "logs").mkdir(parents=True, exist_ok=True)

    return [
        mock.patch.object(inventory_db, "INVENTORY_DB", tmp / "inventory.db"),
        mock.patch.object(usage_tracker, "USAGE_DB", tmp / "usage.db"),
        mock.patch.object(monitoring, "MONITORING_DB", tmp / "monitoring.db"),
        mock.patch.object(transcript_log, "TRANSCRIPT_PATH", tmp / "transcripts.jsonl"),
        mock.patch.object(logger, "LOG_DIR", tmp / "logs"),
    ]
