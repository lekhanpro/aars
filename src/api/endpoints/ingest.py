"""Document ingestion endpoint."""

from __future__ import annotations

from fastapi import APIRouter, File, Form, Request, UploadFile

from src.api.schemas import ErrorResponse, IngestResponse
from src.ingestion.pipeline import IngestionPipeline

router = APIRouter()


@router.post(
    "/ingest",
    response_model=IngestResponse,
    responses={500: {"model": ErrorResponse}},
)
async def ingest(
    request: Request,
    file: UploadFile = File(...),
    collection: str = Form(default="default"),
    chunk_size: int | None = Form(default=None),
    chunk_overlap: int | None = Form(default=None),
) -> IngestResponse:
    """Upload and index documents."""
    pipeline = IngestionPipeline(
        chroma_client=request.app.state.chroma_client,
        settings=request.app.state.settings,
    )
    return await pipeline.ingest(
        file=file,
        collection=collection,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
