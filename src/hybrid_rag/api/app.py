from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from hybrid_rag import __version__
from hybrid_rag.api.container import Container, build_container
from hybrid_rag.api.routes import router
from hybrid_rag.config import Settings


def create_app(container: Container | None = None) -> FastAPI:
    """App factory. Tests inject a container; production builds one from env settings."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.container = container or build_container(Settings())
        yield

    app = FastAPI(
        title="Production-Grade Hybrid RAG",
        version=__version__,
        summary="Dense + BM25 retrieval fused with RRF, cross-encoder reranking, "
        "and claim-level faithfulness guardrails.",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app
