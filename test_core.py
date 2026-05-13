"""
tests/test_core.py – Unit tests for eThute Lenna core logic.

Run with:
    pytest tests/ -v
"""

import pytest
import sys
import os

# Add app to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from config import AppConfig
from debugger import DebugLogger, EventTracker, assert_not_none
import logging


# ── Config Tests ──────────────────────────────────────────────────

class TestAppConfig:
    def test_default_model_in_available(self):
        assert AppConfig.DEFAULT_MODEL in AppConfig.AVAILABLE_MODELS

    def test_chunk_size_positive(self):
        assert AppConfig.CHUNK_SIZE > 0

    def test_chunk_overlap_less_than_size(self):
        assert AppConfig.CHUNK_OVERLAP < AppConfig.CHUNK_SIZE

    def test_retrieval_k_positive(self):
        assert AppConfig.RETRIEVAL_K > 0

    def test_rag_prompt_has_placeholders(self):
        assert "{context}" in AppConfig.RAG_PROMPT
        assert "{question}" in AppConfig.RAG_PROMPT

    def test_max_chars_reasonable(self):
        assert 100 < AppConfig.MAX_CHARS <= 10_000


# ── DebugLogger Tests ─────────────────────────────────────────────

class TestDebugLogger:
    def test_returns_logger(self):
        dl = DebugLogger(level=logging.DEBUG)
        logger = dl.get_logger("test")
        assert isinstance(logger, logging.Logger)

    def test_logger_name(self):
        dl = DebugLogger()
        logger = dl.get_logger("mymodule")
        assert logger.name == "mymodule"


# ── EventTracker Tests ────────────────────────────────────────────

class TestEventTracker:
    def test_log_and_recent(self):
        tracker = EventTracker(max_events=10)
        tracker.log("TEST", "hello")
        recent = tracker.recent(5)
        assert len(recent) == 1
        assert "TEST" in recent[0]
        assert "hello" in recent[0]

    def test_max_events_respected(self):
        tracker = EventTracker(max_events=5)
        for i in range(10):
            tracker.log("STEP", f"event {i}")
        assert len(tracker.all()) == 5

    def test_clear(self):
        tracker = EventTracker()
        tracker.log("A", "something")
        tracker.clear()
        assert tracker.all() == []

    def test_recent_returns_last_n(self):
        tracker = EventTracker()
        for i in range(20):
            tracker.log("N", str(i))
        assert len(tracker.recent(5)) == 5
        assert "19" in tracker.recent(1)[0]


# ── Assert Not None Tests ─────────────────────────────────────────

class TestAssertNotNone:
    def test_passes_non_none(self):
        result = assert_not_none("hello", "my_var")
        assert result == "hello"

    def test_raises_on_none(self):
        with pytest.raises(ValueError, match="my_var"):
            assert_not_none(None, "my_var")

    def test_passes_zero(self):
        result = assert_not_none(0, "zero_val")
        assert result == 0

    def test_passes_empty_string(self):
        result = assert_not_none("", "empty")
        assert result == ""


# ── Prompt Sanity Check ───────────────────────────────────────────

class TestRAGPrompt:
    def test_prompt_renders_with_placeholders(self):
        filled = AppConfig.RAG_PROMPT.format(
            context="Some document text here.",
            question="What is this about?"
        )
        assert "Some document text here." in filled
        assert "What is this about?" in filled
