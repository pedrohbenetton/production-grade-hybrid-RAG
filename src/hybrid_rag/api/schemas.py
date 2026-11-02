"""Public API contract — kept separate from domain models so internals can evolve."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from hybrid_rag.retrieval.hybrid import RetrievalMode, RetrievalResult

Query = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


class RetrieveRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"query": "How does RRF fuse rankings?", "mode": "hybrid_rerank", "k": 5}]
        }
    )

    query: Query
    mode: RetrievalMode = Field(
        "hybrid_rerank",
        description="dense = naive vector RAG; sparse = BM25; hybrid = RRF of both; "
        "hybrid_rerank = RRF + cross-encoder.",
    )
    k: int = Field(5, ge=1, le=50, description="Number of hits to return.")


class Hit(BaseModel):
    rank: int = Field(ge=1)
    chunk_id: str
    doc_id: str
    text: str
    section_path: list[str]
    page: int | None
    score: float = Field(description="Score of the final stage; comparable only within a mode.")


class RetrieveResponse(BaseModel):
    query: str
    mode: RetrievalMode
    hits: list[Hit]
    timings_ms: dict[str, float] = Field(description="Per-stage latency in milliseconds.")

    @classmethod
    def from_result(cls, result: RetrievalResult) -> RetrieveResponse:
        return cls(
            query=result.query,
            mode=result.mode,
            hits=[
                Hit(
                    rank=rank,
                    chunk_id=hit.chunk.chunk_id,
                    doc_id=hit.chunk.doc_id,
                    text=hit.chunk.text,
                    section_path=list(hit.chunk.section_path),
                    page=hit.chunk.page,
                    score=hit.score,
                )
                for rank, hit in enumerate(result.hits, start=1)
            ],
            timings_ms=result.timings_ms,
        )


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ReadyResponse(BaseModel):
    status: Literal["ready"] = "ready"
    chunks: int = Field(ge=0)
