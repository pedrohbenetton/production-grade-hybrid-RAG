"""Query-time orchestration: dense ∥ sparse → RRF → cross-encoder (§4).

Modes map one-to-one to the benchmark's systems so the API and the benchmark
exercise exactly the same code path.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Literal

from pydantic import BaseModel, ConfigDict

from hybrid_rag.config import RetrievalSettings
from hybrid_rag.domain.models import ScoredChunk
from hybrid_rag.ports import DenseIndex, Embedder
from hybrid_rag.retrieval.fusion import reciprocal_rank_fusion
from hybrid_rag.retrieval.rerank import Reranker
from hybrid_rag.retrieval.sparse import BM25Index

RetrievalMode = Literal["dense", "sparse", "hybrid", "hybrid_rerank"]


class RerankerUnavailableError(RuntimeError):
    pass


class RetrievalResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str
    mode: RetrievalMode
    hits: tuple[ScoredChunk, ...]
    timings_ms: dict[str, float]


class _Stopwatch:
    def __init__(self) -> None:
        self.timings_ms: dict[str, float] = {}

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        start = perf_counter()
        try:
            yield
        finally:
            self.timings_ms[stage] = (perf_counter() - start) * 1000


class HybridRetriever:
    def __init__(
        self,
        embedder: Embedder,
        dense: DenseIndex,
        sparse: BM25Index,
        reranker: Reranker | None = None,
        settings: RetrievalSettings | None = None,
    ) -> None:
        self._embedder = embedder
        self._dense = dense
        self._sparse = sparse
        self._reranker = reranker
        self._settings = settings or RetrievalSettings()

    @property
    def has_reranker(self) -> bool:
        return self._reranker is not None

    def retrieve(
        self, query: str, *, mode: RetrievalMode = "hybrid_rerank", k: int | None = None
    ) -> RetrievalResult:
        if mode == "hybrid_rerank" and self._reranker is None:
            raise RerankerUnavailableError(
                "mode 'hybrid_rerank' needs a reranker; set RAG_RERANK__PROVIDER "
                "or use mode 'hybrid'"
            )
        s = self._settings
        k = k or s.final_k
        # The funnel never narrows below k, so a large k still gets k results.
        fused_n = max(k, s.fused_candidates)
        dense_n = k if mode == "dense" else max(s.dense_candidates, fused_n)
        sparse_n = k if mode == "sparse" else max(s.sparse_candidates, fused_n)

        clock = _Stopwatch()
        start = perf_counter()
        dense_hits: list[ScoredChunk] = []
        sparse_hits: list[ScoredChunk] = []
        if mode != "sparse":
            with clock.measure("embed"):
                vector = self._embedder.embed_query(query)
            with clock.measure("dense"):
                dense_hits = self._dense.search(vector, dense_n)
        if mode != "dense":
            with clock.measure("sparse"):
                sparse_hits = self._sparse.search(query, sparse_n)

        if mode == "dense":
            hits = dense_hits
        elif mode == "sparse":
            hits = sparse_hits
        else:
            with clock.measure("fusion"):
                hits = reciprocal_rank_fusion(
                    [dense_hits, sparse_hits],
                    k=s.rrf_k,
                    weights=(s.dense_weight, s.sparse_weight),
                    limit=k if mode == "hybrid" else fused_n,
                )
            if mode == "hybrid_rerank" and self._reranker is not None:
                with clock.measure("rerank"):
                    hits = self._reranker.rerank(query, hits, top_k=k)

        clock.timings_ms["total"] = (perf_counter() - start) * 1000
        return RetrievalResult(
            query=query, mode=mode, hits=tuple(hits), timings_ms=clock.timings_ms
        )
