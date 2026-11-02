from collections.abc import Sequence
from dataclasses import dataclass

import pytest
from conftest import FakeEmbedder, make_chunk
from qdrant_client import QdrantClient

from hybrid_rag.config import RetrievalSettings
from hybrid_rag.domain.models import ScoredChunk
from hybrid_rag.retrieval.dense import QdrantDenseIndex
from hybrid_rag.retrieval.fusion import reciprocal_rank_fusion
from hybrid_rag.retrieval.hybrid import HybridRetriever, RerankerUnavailableError
from hybrid_rag.retrieval.sparse import BM25Index

pytestmark = pytest.mark.integration

TEXTS = [
    "Dense retrieval embeds queries and passages into a shared vector space.",
    "BM25 ranks passages with term frequency and inverse document frequency.",
    "Reciprocal rank fusion merges rankings without calibrating scores.",
    "Cross-encoder rerankers jointly score query and passage pairs.",
    "HNSW graphs give approximate nearest neighbour search in vector space.",
    "Stemming maps running and runs to the same retrieval term.",
    "Chunking splits documents into passages before indexing.",
    "Faithfulness checks verify generated claims against retrieved passages.",
]
CORPUS = [make_chunk(text, chunk_id=f"c{i}") for i, text in enumerate(TEXTS)]
SETTINGS = RetrievalSettings(dense_candidates=5, sparse_candidates=5, fused_candidates=4, final_k=3)
QUERY = "how does fusion merge passage rankings"


class LengthReranker:
    """Deterministic stand-in for a cross-encoder: longer text = more relevant."""

    def __init__(self) -> None:
        self.seen: list[list[str]] = []

    def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        self.seen.append([c.chunk.chunk_id for c in candidates])
        ordered = sorted(candidates, key=lambda h: (-len(h.chunk.text), h.chunk.chunk_id))
        return [ScoredChunk(chunk=h.chunk, score=float(len(h.chunk.text))) for h in ordered[:top_k]]


@dataclass
class Setup:
    embedder: FakeEmbedder
    dense: QdrantDenseIndex
    sparse: BM25Index
    reranker: LengthReranker
    retriever: HybridRetriever


@pytest.fixture
def setup(fake_embedder: FakeEmbedder) -> Setup:
    dense = QdrantDenseIndex(QdrantClient(location=":memory:"), "t", fake_embedder.dimension)
    dense.upsert(CORPUS, fake_embedder.embed([c.embed_text for c in CORPUS]))
    sparse = BM25Index()
    sparse.add(CORPUS)
    reranker = LengthReranker()
    retriever = HybridRetriever(fake_embedder, dense, sparse, reranker, SETTINGS)
    return Setup(fake_embedder, dense, sparse, reranker, retriever)


def ids(hits: Sequence[ScoredChunk]) -> list[str]:
    return [h.chunk.chunk_id for h in hits]


def test_dense_mode_is_plain_vector_search(setup: Setup) -> None:
    result = setup.retriever.retrieve(QUERY, mode="dense")
    expected = setup.dense.search(setup.embedder.embed_query(QUERY), k=3)
    assert ids(result.hits) == ids(expected)
    assert set(result.timings_ms) == {"embed", "dense", "total"}


def test_sparse_mode_is_plain_bm25(setup: Setup) -> None:
    result = setup.retriever.retrieve(QUERY, mode="sparse")
    assert ids(result.hits) == ids(setup.sparse.search(QUERY, k=3))
    assert set(result.timings_ms) == {"sparse", "total"}


def test_hybrid_mode_fuses_candidate_lists_with_rrf(setup: Setup) -> None:
    result = setup.retriever.retrieve(QUERY, mode="hybrid")

    dense_hits = setup.dense.search(setup.embedder.embed_query(QUERY), k=5)
    expected = reciprocal_rank_fusion([dense_hits, setup.sparse.search(QUERY, k=5)], limit=3)
    assert ids(result.hits) == ids(expected)
    assert [h.score for h in result.hits] == [h.score for h in expected]


def test_hybrid_rerank_reranks_only_the_fused_top_n(setup: Setup) -> None:
    result = setup.retriever.retrieve(QUERY, mode="hybrid_rerank")

    [candidates] = setup.reranker.seen
    fused = setup.retriever.retrieve(QUERY, mode="hybrid", k=4)
    assert candidates == ids(fused.hits)
    assert len(result.hits) == 3
    assert set(result.timings_ms) == {"embed", "dense", "sparse", "fusion", "rerank", "total"}
    assert all(v >= 0 for v in result.timings_ms.values())


def test_k_overrides_the_default_and_widens_rerank_candidates(setup: Setup) -> None:
    assert len(setup.retriever.retrieve(QUERY, mode="hybrid", k=1).hits) == 1
    setup.retriever.retrieve(QUERY, mode="hybrid_rerank", k=6)
    assert len(setup.reranker.seen[-1]) >= 6


def test_rerank_mode_requires_a_reranker(setup: Setup) -> None:
    retriever = HybridRetriever(setup.embedder, setup.dense, setup.sparse, None, SETTINGS)
    assert not retriever.has_reranker
    with pytest.raises(RerankerUnavailableError):
        retriever.retrieve(QUERY, mode="hybrid_rerank")
    assert retriever.retrieve(QUERY, mode="hybrid").hits


@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid", "hybrid_rerank"])
def test_empty_indexes_return_no_hits(fake_embedder: FakeEmbedder, mode: str) -> None:
    dense = QdrantDenseIndex(QdrantClient(location=":memory:"), "e", fake_embedder.dimension)
    retriever = HybridRetriever(fake_embedder, dense, BM25Index(), LengthReranker(), SETTINGS)
    assert retriever.retrieve(QUERY, mode=mode).hits == ()  # type: ignore[arg-type]
