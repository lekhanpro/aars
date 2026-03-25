"""Pipeline execution trace recording and storage."""

from __future__ import annotations

import time
import uuid
from threading import Lock

import structlog

from src.api.schemas import PipelineTrace, TraceStep

logger = structlog.get_logger()


class TraceRecorder:
    """Records steps during a single pipeline execution."""

    def __init__(self) -> None:
        self.trace_id = str(uuid.uuid4())
        self.steps: list[TraceStep] = []
        self.total_tokens = 0
        self.total_api_calls = 0
        self._start_time = time.monotonic()

    def record(self, step: str, duration_ms: float, **details: object) -> None:
        self.steps.append(TraceStep(step=step, duration_ms=round(duration_ms, 2), details=details))

    def add_tokens(self, tokens: int) -> None:
        self.total_tokens += tokens

    def add_api_call(self) -> None:
        self.total_api_calls += 1

    def finalize(self) -> PipelineTrace:
        elapsed_ms = (time.monotonic() - self._start_time) * 1000
        recorded_ms = sum(step.duration_ms for step in self.steps)
        total_ms = max(elapsed_ms, recorded_ms)
        trace = PipelineTrace(
            trace_id=self.trace_id,
            steps=self.steps,
            total_duration_ms=round(total_ms, 2),
            total_tokens=self.total_tokens,
            total_api_calls=self.total_api_calls,
        )
        TraceStore.store(trace)
        return trace


class TraceStore:
    """In-memory store for pipeline traces (bounded)."""

    _traces: dict[str, PipelineTrace] = {}
    _lock = Lock()
    _max_traces = 1000

    @classmethod
    def store(cls, trace: PipelineTrace) -> None:
        with cls._lock:
            if len(cls._traces) >= cls._max_traces:
                oldest = next(iter(cls._traces))
                del cls._traces[oldest]
            cls._traces[trace.trace_id] = trace

    @classmethod
    def get(cls, trace_id: str) -> PipelineTrace | None:
        with cls._lock:
            return cls._traces.get(trace_id)

    @classmethod
    def clear(cls) -> None:
        with cls._lock:
            cls._traces.clear()
