"""Debug and trace endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.api.schemas import PipelineTrace
from src.pipeline.trace import TraceStore

router = APIRouter()


@router.get("/debug/trace/{trace_id}", response_model=PipelineTrace)
async def get_trace(trace_id: str) -> PipelineTrace:
    """Get pipeline execution trace by ID."""
    trace = TraceStore.get(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail=f"Trace '{trace_id}' not found")
    return trace
