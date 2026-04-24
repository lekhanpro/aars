"""Document listing endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from src.api.schemas import ErrorResponse

router = APIRouter()


@router.get(
    "/documents/{collection}",
    responses={500: {"model": ErrorResponse}},
)
async def list_documents(request: Request, collection: str, limit: int = 100) -> dict[str, Any]:
    """List documents in a collection."""
    chroma_client = request.app.state.chroma_client
    if chroma_client is None:
        return {"collection": collection, "documents": [], "error": "ChromaDB not connected"}

    try:
        col = chroma_client.get_collection(collection)
        result = col.peek(limit=limit)
        docs = []
        ids = result.get("ids", [])
        documents = result.get("documents", [])
        metadatas = result.get("metadatas", [])
        for i, doc_id in enumerate(ids):
            docs.append({
                "id": doc_id,
                "content": documents[i][:200] if i < len(documents) and documents[i] else "",
                "metadata": metadatas[i] if i < len(metadatas) else {},
            })
        return {"collection": collection, "count": col.count(), "documents": docs}
    except Exception as e:
        return {"collection": collection, "documents": [], "error": str(e)}
