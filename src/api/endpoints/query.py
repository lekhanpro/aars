"""Main query endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Request

from src.api.schemas import ErrorResponse, QueryRequest, QueryResponse

router = APIRouter()


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={500: {"model": ErrorResponse}},
)
async def query(request: Request, body: QueryRequest) -> QueryResponse:
    """Main RAG pipeline — query → answer."""
    return await request.app.state.orchestrator.run(body)
