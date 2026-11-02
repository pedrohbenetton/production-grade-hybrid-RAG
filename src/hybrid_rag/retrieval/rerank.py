"""Cross-encoder reranking (§4.4): local FlashRank by default, Cohere optional.

Bi-encoders embed query and passage independently; a cross-encoder reads them
together, which is far more precise but O(candidates) per query — so it only
ever sees the fused top-N, never the whole index.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from flashrank import Ranker, RerankRequest

from hybrid_rag.config import RerankSettings
from hybrid_rag.domain.models import ScoredChunk

DEFAULT_FLASHRANK_MODEL = "ms-marco-MiniLM-L-12-v2"
DEFAULT_COHERE_MODEL = "rerank-v3.5"


class Reranker(Protocol):
    def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]: ...


class FlashRankReranker:
    """ONNX cross-encoder on CPU. Scores ``embed_text`` so the breadcrumb is visible."""

    def __init__(
        self,
        model: str = DEFAULT_FLASHRANK_MODEL,
        *,
        cache_dir: str = "./data/models",
        max_length: int = 512,
    ) -> None:
        self._ranker = Ranker(
            model_name=model, cache_dir=cache_dir, max_length=max_length, log_level="WARNING"
        )

    def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        if not candidates:
            return []
        by_id = {c.chunk.chunk_id: c.chunk for c in candidates}
        passages = [{"id": c.chunk.chunk_id, "text": c.chunk.embed_text} for c in candidates]
        results = self._ranker.rerank(RerankRequest(query=query, passages=passages))
        ordered = sorted(results, key=lambda r: (-float(r["score"]), str(r["id"])))
        return [
            ScoredChunk(chunk=by_id[str(r["id"])], score=float(r["score"])) for r in ordered[:top_k]
        ]


class CohereRerankResult(Protocol):
    @property
    def index(self) -> int: ...

    @property
    def relevance_score(self) -> float: ...


class CohereRerankResponse(Protocol):
    @property
    def results(self) -> Sequence[CohereRerankResult]: ...


class CohereClient(Protocol):
    """The slice of ``cohere.ClientV2`` used here (keeps the SDK optional and fakeable)."""

    def rerank(
        self, *, model: str, query: str, documents: Sequence[str], top_n: int
    ) -> CohereRerankResponse: ...


class CohereReranker:
    def __init__(self, client: CohereClient, model: str = DEFAULT_COHERE_MODEL) -> None:
        self._client = client
        self._model = model

    def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        if not candidates:
            return []
        response = self._client.rerank(
            model=self._model,
            query=query,
            documents=[c.chunk.embed_text for c in candidates],
            top_n=top_k,
        )
        return [
            ScoredChunk(chunk=candidates[r.index].chunk, score=float(r.relevance_score))
            for r in response.results
        ]


def make_reranker(settings: RerankSettings) -> Reranker | None:
    if settings.provider == "none":
        return None
    if settings.provider == "flashrank":
        return FlashRankReranker(
            settings.model or DEFAULT_FLASHRANK_MODEL, cache_dir=settings.cache_dir
        )
    if settings.cohere_api_key is None:
        raise ValueError("rerank provider 'cohere' requires RAG_RERANK__COHERE_API_KEY")
    try:
        import cohere
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        raise ImportError("install the optional extra: uv sync --extra cohere") from exc
    client = cohere.ClientV2(api_key=settings.cohere_api_key.get_secret_value())
    return CohereReranker(client, settings.model or DEFAULT_COHERE_MODEL)
