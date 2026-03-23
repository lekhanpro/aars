"""Tests for pipeline trace."""

from __future__ import annotations

from src.pipeline.trace import TraceRecorder, TraceStore


class TestTraceRecorder:
    def test_record_step(self):
        trace = TraceRecorder()
        trace.record("planning", 150.5, strategy="vector")
        assert len(trace.steps) == 1
        assert trace.steps[0].step == "planning"
        assert trace.steps[0].duration_ms == 150.5
        assert trace.steps[0].details["strategy"] == "vector"

    def test_finalize(self):
        trace = TraceRecorder()
        trace.record("step1", 100.0)
        trace.record("step2", 200.0)
        trace.add_tokens(500)
        result = trace.finalize()
        assert result.trace_id == trace.trace_id
        assert len(result.steps) == 2
        assert result.total_tokens == 500
        assert result.total_duration_ms > 0

    def test_token_counting(self):
        trace = TraceRecorder()
        trace.add_tokens(100)
        trace.add_tokens(200)
        assert trace.total_tokens == 300

    def test_api_call_counting(self):
        trace = TraceRecorder()
        trace.add_api_call()
        trace.add_api_call()
        assert trace.total_api_calls == 2


class TestTraceStore:
    def setup_method(self):
        TraceStore.clear()

    def test_store_and_get(self):
        trace = TraceRecorder()
        result = trace.finalize()
        retrieved = TraceStore.get(result.trace_id)
        assert retrieved is not None
        assert retrieved.trace_id == result.trace_id

    def test_get_missing(self):
        assert TraceStore.get("nonexistent") is None

    def test_clear(self):
        trace = TraceRecorder()
        result = trace.finalize()
        TraceStore.clear()
        assert TraceStore.get(result.trace_id) is None
