"""Indexes a BEIR corpus through the real ingestion pipeline and runs retrieval systems.

Documents are scored, not chunks: a document's rank is the rank of its best chunk, so
chunked and unchunked systems stay comparable with the dataset's document-level qrels.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict

from hybrid_rag.domain.ids import make_doc_id
from hybrid_rag.domain.models import RawDocument, ScoredChunk
from hybrid_rag.evaluation.beir import BeirDataset
from hybrid_rag.indexing import Indexer
from hybrid_rag.ingestion.pipeline import IngestionPipeline
from hybrid_rag.retrieval.hybrid import HybridRetriever, RetrievalMode

Progress = Callable[[int, int], None]


class SystemSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    label: str
    mode: RetrievalMode
    reranker: str | None = None  # reranker model name; None for non-reranking systems


class SystemResult(BaseModel):
    spec: SystemSpec
    ranking: dict[str, list[str]]  # query id → ranked BEIR document ids
    latency_ms: dict[str, float]  # query id → retrieval latency


class BenchmarkResult(BaseModel):
    """Self-contained: the report can be regenerated from this file alone."""

    dataset: str
    n_docs: int
    n_queries: int
    embedding_model: str
    dataset_sha256: str | None
    top_k: int
    qrels: dict[str, dict[str, int]]
    systems: list[SystemResult]


DEFAULT_SYSTEMS = (
    SystemSpec(name="dense", label="Naive RAG (dense only)", mode="dense"),
    SystemSpec(name="sparse", label="BM25 only", mode="sparse"),
    SystemSpec(name="hybrid", label="Hybrid (dense + BM25, RRF)", mode="hybrid"),
    SystemSpec(
        name="hybrid_rerank_minilm",
        label="Hybrid + rerank (MiniLM-L-12)",
        mode="hybrid_rerank",
        reranker="ms-marco-MiniLM-L-12-v2",
    ),
    SystemSpec(
        name="hybrid_rerank_tiny",
        label="Hybrid + rerank (TinyBERT-L-2)",
        mode="hybrid_rerank",
        reranker="ms-marco-TinyBERT-L-2-v2",
    ),
)


def source_uri(dataset: str, doc_id: str) -> str:
    return f"beir://{dataset}/{doc_id}"


def doc_id_map(dataset: BeirDataset) -> dict[str, str]:
    """Internal (hashed) document id → BEIR document id."""
    return {make_doc_id(source_uri(dataset.name, d)): d for d in dataset.corpus}


def to_raw_document(dataset: BeirDataset, doc_id: str) -> RawDocument:
    doc = dataset.corpus[doc_id]
    title = doc.title.strip()
    markdown = f"# {title}\n\n{doc.text.strip()}" if title else doc.text.strip()
    return RawDocument(
        source_uri=source_uri(dataset.name, doc_id),
        mime_type="text/markdown",
        content=markdown.encode(),
    )


def index_corpus(
    dataset: BeirDataset,
    pipeline: IngestionPipeline,
    indexer: Indexer,
    *,
    progress: Progress | None = None,
) -> int:
    """Ingest every document with text; returns the number of chunks added."""
    doc_ids = [d for d, doc in dataset.corpus.items() if doc.text.strip()]
    added = 0
    for done, doc_id in enumerate(doc_ids, start=1):
        processed = pipeline.process(to_raw_document(dataset, doc_id))
        added += indexer.index(processed).added
        if progress:
            progress(done, len(doc_ids))
    return added


def collapse_to_documents(
    hits: Sequence[ScoredChunk], doc_ids: Mapping[str, str], top: int
) -> list[str]:
    """Chunk ranking → document ranking: first (best) chunk wins, order preserved."""
    ranking: list[str] = []
    for hit in hits:
        beir_id = doc_ids[hit.chunk.doc_id]
        if beir_id not in ranking:
            ranking.append(beir_id)
        if len(ranking) == top:
            break
    return ranking


def run_system(
    dataset: BeirDataset,
    retriever: HybridRetriever,
    spec: SystemSpec,
    doc_ids: Mapping[str, str],
    *,
    fetch_k: int,
    top: int,
    progress: Progress | None = None,
) -> SystemResult:
    """Run every query sequentially. ``fetch_k`` > ``top`` leaves room for chunk duplicates."""
    queries = list(dataset.queries.items())
    if queries:  # untimed warm-up: ONNX session start-up must not pollute the first latency
        retriever.retrieve(queries[0][1], mode=spec.mode, k=fetch_k)

    ranking: dict[str, list[str]] = {}
    latency_ms: dict[str, float] = {}
    for done, (query_id, text) in enumerate(queries, start=1):
        result = retriever.retrieve(text, mode=spec.mode, k=fetch_k)
        ranking[query_id] = collapse_to_documents(result.hits, doc_ids, top)
        latency_ms[query_id] = result.timings_ms["total"]
        if progress:
            progress(done, len(queries))
    return SystemResult(spec=spec, ranking=ranking, latency_ms=latency_ms)
