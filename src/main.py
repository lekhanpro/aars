"""FastAPI application factory with lifespan events."""

from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.logging_config import setup_logging
from config.settings import get_settings

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: startup and shutdown events."""
    settings = get_settings()
    setup_logging(settings.log_level)
    logger.info("aars_startup", version="0.1.0")

    # Store settings on app state
    app.state.settings = settings

    # Initialize ChromaDB client
    try:
        import chromadb

        chroma_client = chromadb.HttpClient(
            host=settings.chroma.host, port=settings.chroma.port
        )
        chroma_client.heartbeat()
        app.state.chroma_client = chroma_client
        logger.info("chromadb_connected", host=settings.chroma.host, port=settings.chroma.port)
    except Exception as e:
        logger.warning("chromadb_unavailable", error=str(e))
        app.state.chroma_client = None

    # Initialize LLM client
    from src.llm.client import LLMClient

    app.state.llm_client = LLMClient(
        api_key=settings.anthropic_api_key, settings=settings.llm
    )

    yield

    logger.info("aars_shutdown")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="AARS — Agentic Adaptive Retrieval System",
        description="Dynamic strategy selection for Retrieval-Augmented Generation",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Import and mount routers
    from src.api.endpoints.debug import router as debug_router
    from src.api.endpoints.health import router as health_router
    from src.api.endpoints.ingest import router as ingest_router
    from src.api.endpoints.query import router as query_router

    app.include_router(health_router, prefix="/api/v1", tags=["health"])
    app.include_router(query_router, prefix="/api/v1", tags=["query"])
    app.include_router(ingest_router, prefix="/api/v1", tags=["ingest"])
    app.include_router(debug_router, prefix="/api/v1", tags=["debug"])

    return app


app = create_app()
