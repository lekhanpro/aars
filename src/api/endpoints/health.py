"""Health and collections endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request

from src.api.schemas import CollectionInfo, CollectionsResponse, HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    chroma_client = getattr(request.app.state, "chroma_client", None)
    chroma_connected = False
    if chroma_client:
        try:
            chroma_client.heartbeat()
            chroma_connected = True
        except Exception:
            pass
    return HealthResponse(status="healthy", version="0.1.0", chromadb_connected=chroma_connected)


@router.get("/collections", response_model=CollectionsResponse)
async def list_collections(request: Request) -> CollectionsResponse:
    chroma_client = getattr(request.app.state, "chroma_client", None)
    if not chroma_client:
        return CollectionsResponse(collections=[])
    collections = chroma_client.list_collections()
    return CollectionsResponse(
        collections=[
            CollectionInfo(name=c.name, document_count=c.count()) for c in collections
        ]
    )


@router.delete("/collections/{name}")
async def delete_collection(name: str, request: Request) -> dict:
    chroma_client = getattr(request.app.state, "chroma_client", None)
    if not chroma_client:
        return {"error": "ChromaDB not connected"}
    chroma_client.delete_collection(name)
    return {"message": f"Collection '{name}' deleted"}
