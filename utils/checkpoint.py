"""Checkpoint utilities for the screening pipeline."""

import json
import logging
from pathlib import Path

log = logging.getLogger("pipeline")

CHECKPOINT_PATH = Path(__file__).parent.parent / "data" / "checkpoints" / "checkpoint.json"


def save(records: list):
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    log.info("Checkpoint saved: %d records", len(records))


def load() -> list:
    if not CHECKPOINT_PATH.exists():
        return []
    try:
        records = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        if isinstance(records, list):
            log.info("Checkpoint loaded: %d records", len(records))
            return records
    except Exception as exc:
        log.warning("Could not read checkpoint, starting fresh: %s", exc)  # corrupted file
    return []


def clear():
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        log.info("Checkpoint cleared")
