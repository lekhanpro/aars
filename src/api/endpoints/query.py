"""Main query endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Request

from src.api.schemas import ErrorResponse, QueryRequest, QueryResponse
from src.pipeline.orchestrator import PipelineOrchestrator

router = APIRouter()


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={500: {"model": ErrorResponse}},
)
async def query(request: Request, body: QueryRequest) -> QueryResponse:
    """Main RAG pipeline — query → answer."""
    orchestrator = PipelineOrchestrator(
        llm_client=request.app.state.llm_client,
        chroma_client=request.app.state.chroma_client,
        settings=request.app.state.settings,
    )
    return await orchestrator.run(body)
