"""
debugger.py – Structured logging & debug utilities for eThute Lenna.
"""

import logging
import sys
from datetime import datetime


class DebugLogger:
    """
    Wraps Python's logging module with app-friendly defaults.

    Usage:
        debug = DebugLogger(level=logging.DEBUG)
        logger = debug.get_logger(__name__)
        logger.info("Something happened")
        logger.error("Something broke", exc_info=True)
    """

    LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    DATE_FORMAT = "%H:%M:%S"

    def __init__(self, level: int = logging.INFO):
        self.level = level
        self._configure_root()

    def _configure_root(self):
        root = logging.getLogger()
        root.setLevel(self.level)

        # Remove existing handlers to avoid duplicate output
        root.handlers.clear()

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(self.level)
        handler.setFormatter(
            logging.Formatter(self.LOG_FORMAT, datefmt=self.DATE_FORMAT)
        )
        root.addHandler(handler)

    def get_logger(self, name: str) -> logging.Logger:
        return logging.getLogger(name)


class EventTracker:
    """
    Lightweight in-memory event log shown in the UI debug panel.
    Keeps the last N events.

    Usage:
        tracker = EventTracker(max_events=50)
        tracker.log("INDEX", "Started PDF indexing")
        tracker.log("ERROR", str(e))
        print(tracker.recent(10))
    """

    def __init__(self, max_events: int = 100):
        self.max_events = max_events
        self._events: list[str] = []

    def log(self, category: str, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] [{category}] {message}"
        self._events.append(entry)
        if len(self._events) > self.max_events:
            self._events.pop(0)

    def recent(self, n: int = 20) -> list[str]:
        return self._events[-n:]

    def clear(self):
        self._events.clear()

    def all(self) -> list[str]:
        return list(self._events)


# ── Common debug helpers ──────────────────────────────────────────

def log_chain_step(logger: logging.Logger, step: str, data: str, max_len: int = 120):
    """Log a RAG pipeline step with truncation."""
    preview = data[:max_len] + ("..." if len(data) > max_len else "")
    logger.debug(f"[{step}] {preview}")


def assert_not_none(value, name: str):
    """Raise a clear error if a critical value is None."""
    if value is None:
        raise ValueError(f"Expected '{name}' to be set but got None.")
    return value
