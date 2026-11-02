from collections.abc import Sequence

import pytest
from conftest import CATS, ROCKETS, FakeEmbedder
from qdrant_client import QdrantClient

from hybrid_rag.config import ChunkingSettings, RetrievalSettings
from hybrid_rag.domain.ids import make_doc_id
from hybrid_rag.domain.models import ScoredChunk
from hybrid_rag.evaluation.beir import BeirDataset, BeirDoc
from hybrid_rag.evaluation.runner import (
    SystemSpec,
    collapse_to_documents,
    doc_id_map,
    index_corpus,
    run_system,
)
from hybrid_rag.indexing import Indexer, rebuild_sparse
from hybrid_rag.ingestion.pipeline import IngestionPipeline
from hybrid_rag.retrieval.dense import QdrantDenseIndex
from hybrid_rag.retrieval.hybrid import HybridRetriever
from hybrid_rag.retrieval.sparse import BM25Index

pytestmark = pytest.mark.integration

DATASET = BeirDataset(
    name="mini",
    corpus={
        # two topics in one document: the chunker splits it, retrieval must still count it once
        "both": BeirDoc(title="Mixed", text=" ".join(CATS + ROCKETS)),
        "gardens": BeirDoc(title="Gardens", text="Tomatoes grow best with steady watering."),
        "rivers": BeirDoc(title="Rivers", text="Rivers carve valleys across many centuries."),
    },
    queries={"q1": "why do cats purr", "q2": "how do rivers carve valleys"},
    qrels={"q1": {"both": 1}, "q2": {"rivers": 1}},
)
CHUNKING = ChunkingSettings(min_tokens=5, max_tokens=40, breakpoint_percentile=80, window=0)
RETRIEVAL = RetrievalSettings(dense_candidates=10, sparse_candidates=10, fused_candidates=10)

Stack = tuple[HybridRetriever, dict[str, str], int]


class ReverseReranker:
    """Puts the lowest-scored candidate first, so reranking visibly changes the order."""

    def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        return list(reversed(candidates))[:top_k]


def spec(mode: str, reranker: str | None = None) -> SystemSpec:
    return SystemSpec(name=mode, label=mode, mode=mode, reranker=reranker)


@pytest.fixture
def stack(fake_embedder: FakeEmbedder) -> Stack:
    dense = QdrantDenseIndex(QdrantClient(location=":memory:"), "mini", fake_embedder.dimension)
    sparse = BM25Index()
    pipeline = IngestionPipeline.default(fake_embedder, CHUNKING)
    n_chunks = index_corpus(DATASET, pipeline, Indexer(fake_embedder, dense, sparse))
    assert rebuild_sparse(dense, BM25Index()) == n_chunks == len(sparse)
    retriever = HybridRetriever(fake_embedder, dense, sparse, ReverseReranker(), RETRIEVAL)
    return retriever, doc_id_map(DATASET), n_chunks


def test_doc_id_map_translates_internal_ids_back_to_beir_ids() -> None:
    mapping = doc_id_map(DATASET)
    assert set(mapping.values()) == {"both", "gardens", "rivers"}
    assert make_doc_id("beir://mini/rivers") in mapping


def test_the_corpus_goes_through_the_real_pipeline_into_both_indexes(stack: Stack) -> None:
    _, _, n_chunks = stack
    assert n_chunks > len(DATASET.corpus)  # the mixed-topic document became several chunks


def test_collapse_keeps_the_first_chunk_per_document_in_rank_order(stack: Stack) -> None:
    retriever, mapping, _ = stack
    hits = retriever.retrieve("cats rockets thrust purr", mode="dense", k=20).hits

    ranking = collapse_to_documents(hits, mapping, top=10)

    assert len(ranking) == len(set(ranking))
    assert ranking[0] == "both"
    assert len(hits) > len(ranking)  # duplicates really were present


@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
def test_systems_rank_the_relevant_document_first_with_per_query_latency(
    stack: Stack, mode: str
) -> None:
    retriever, mapping, _ = stack

    result = run_system(DATASET, retriever, spec(mode), mapping, fetch_k=20, top=10)

    assert result.spec == spec(mode)
    assert result.ranking["q1"][0] == "both"
    assert result.ranking["q2"][0] == "rivers"
    assert set(result.latency_ms) == {"q1", "q2"}
    assert all(ms > 0 for ms in result.latency_ms.values())
    assert all(len(r) <= 10 and len(r) == len(set(r)) for r in result.ranking.values())


def test_rerank_system_uses_the_reranker_on_the_fused_candidates(stack: Stack) -> None:
    retriever, mapping, _ = stack
    hybrid = run_system(DATASET, retriever, spec("hybrid"), mapping, fetch_k=20, top=10)
    reranked = run_system(
        DATASET, retriever, spec("hybrid_rerank", "reverse"), mapping, fetch_k=20, top=10
    )

    assert reranked.ranking["q1"] != hybrid.ranking["q1"]
    assert set(reranked.ranking["q1"]) == set(hybrid.ranking["q1"])


def test_progress_callback_reports_every_query(stack: Stack) -> None:
    retriever, mapping, _ = stack
    seen: list[tuple[int, int]] = []

    run_system(
        DATASET,
        retriever,
        spec("dense"),
        mapping,
        fetch_k=20,
        top=10,
        progress=lambda done, total: seen.append((done, total)),
    )

    assert seen == [(1, 2), (2, 2)]
