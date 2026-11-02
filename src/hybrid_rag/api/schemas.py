"""Public API contract — kept separate from domain models so internals can evolve."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from hybrid_rag.domain.models import IndexReport, MimeType
from hybrid_rag.generation.service import AnswerResult
from hybrid_rag.guardrails.models import Verdict
from hybrid_rag.guardrails.policy import AbstainReason, PolicyMode
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


class QueryRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "query": "Why use RRF instead of interpolating scores?",
                    "mode": "hybrid_rerank",
                    "k": 5,
                    "policy": "filter",
                }
            ]
        }
    )

    query: Query
    mode: RetrievalMode = Field(
        "hybrid_rerank", description="Retrieval strategy (see /v1/retrieve)."
    )
    k: int = Field(5, ge=1, le=20, description="Passages offered to the generator.")
    policy: PolicyMode | None = Field(
        None,
        description="annotate = return the draft with per-claim verdicts; filter = drop "
        "unsupported sentences; abstain = refuse below the faithfulness threshold. "
        "Server default when omitted.",
    )


class ClaimOut(BaseModel):
    text: str
    citations: list[int]
    verdict: Verdict
    entailment: float = Field(description="Best NLI entailment probability found.")
    evidence: int | None = Field(description="Passage number that supports the claim.")


class CitationOut(BaseModel):
    n: int = Field(description="The [n] marker used in the answer.")
    chunk_id: str
    doc_id: str
    source: str
    section_path: list[str]
    page: int | None
    text: str


class QueryResponse(BaseModel):
    query: str
    mode: RetrievalMode
    answer: str | None = Field(description="Null when the system abstained.")
    abstained: bool
    abstain_reason: AbstainReason | None
    faithfulness: float | None = Field(
        description="Share of claims in the draft entailed by the context (null: no claims)."
    )
    claims: list[ClaimOut] = Field(description="Every claim in the draft, including dropped ones.")
    invalid_citations: list[int] = Field(description="[n] markers that match no passage.")
    citations: list[CitationOut] = Field(description="Passages cited by the returned answer.")
    timings_ms: dict[str, float]

    @classmethod
    def from_result(cls, result: AnswerResult) -> QueryResponse:
        return cls(
            query=result.query,
            mode=result.mode,
            answer=result.answer,
            abstained=result.abstained,
            abstain_reason=result.abstain_reason,
            faithfulness=result.faithfulness,
            claims=[
                ClaimOut(
                    text=v.claim.text,
                    citations=list(v.claim.citations),
                    verdict=v.verdict,
                    entailment=v.entailment,
                    evidence=v.evidence,
                )
                for v in result.claims
            ],
            invalid_citations=list(result.invalid_citations),
            citations=[
                CitationOut(
                    n=c.n,
                    chunk_id=c.chunk_id,
                    doc_id=c.doc_id,
                    source=c.source,
                    section_path=list(c.section_path),
                    page=c.page,
                    text=c.text,
                )
                for c in result.citations
            ],
            timings_ms=result.timings_ms,
        )


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ReadyResponse(BaseModel):
    status: Literal["ready"] = "ready"
    chunks: int = Field(ge=0)


class IngestRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "source_uri": "docs/architecture.md",
                    "content": "# Architecture\n\nHybrid RAG combines dense and sparse retrieval.",
                    "mime_type": "text/markdown",
                }
            ]
        }
    )

    source_uri: Query = Field(
        ...,
        description=(
            "URI or filename of the document (used to derive doc_id and detect "
            "MIME type when omitted)."
        ),
    )
    content: str = Field(
        ...,
        min_length=1,
        description=(
            "Text content of the document (Markdown or HTML). For PDFs, base64-encoded bytes."
        ),
    )
    mime_type: MimeType | None = Field(
        None,
        description=(
            "Document MIME type ('text/markdown', 'text/html', 'application/pdf'). "
            "Inferred from source_uri if omitted."
        ),
    )


class IngestResponse(BaseModel):
    doc_id: str = Field(description="Content-addressed document ID.")
    added: int = Field(ge=0, description="New chunks embedded and indexed.")
    unchanged: int = Field(ge=0, description="Chunks already up-to-date (0 embeddings spent).")
    removed: int = Field(ge=0, description="Stale chunks removed from the indexes.")

    @classmethod
    def from_report(cls, report: IndexReport) -> IngestResponse:
        return cls(
            doc_id=report.doc_id,
            added=report.added,
            unchanged=report.unchanged,
            removed=report.removed,
        )


class DeleteDocumentResponse(BaseModel):
    doc_id: str = Field(description="Identifier of the deleted document.")
    deleted_chunks: int = Field(ge=0, description="Number of chunks removed from both indexes.")
