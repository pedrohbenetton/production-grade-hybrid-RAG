"""Composition root: the only place where concrete adapters are chosen."""

from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import QdrantClient

from hybrid_rag.config import Settings
from hybrid_rag.generation.llm import make_llm_client
from hybrid_rag.generation.service import RagService
from hybrid_rag.guardrails.nli import OnnxNLIModel
from hybrid_rag.guardrails.verify import ClaimVerifier
from hybrid_rag.indexing import Indexer, rebuild_sparse
from hybrid_rag.ingestion.pipeline import IngestionPipeline
from hybrid_rag.ports import DenseIndex, Embedder
from hybrid_rag.retrieval.dense import QdrantDenseIndex, make_qdrant_client
from hybrid_rag.retrieval.embeddings import FastEmbedEmbedder
from hybrid_rag.retrieval.hybrid import HybridRetriever
from hybrid_rag.retrieval.rerank import make_reranker
from hybrid_rag.retrieval.sparse import BM25Index


@dataclass(frozen=True)
class Container:
    settings: Settings
    embedder: Embedder
    dense: DenseIndex
    sparse: BM25Index
    retriever: HybridRetriever
    pipeline: IngestionPipeline
    indexer: Indexer
    dense_client: QdrantClient | None = None
    service: RagService | None = None  # None: no LLM key, /v1/query disabled


def _build_service(settings: Settings, retriever: HybridRetriever) -> RagService | None:
    llm = make_llm_client(settings.llm)
    if llm is None:
        return None  # avoid downloading the NLI model for a deployment that cannot generate
    g = settings.guardrails
    nli = OnnxNLIModel(g.nli_repo, g.nli_onnx_file, cache_dir=g.nli_cache_dir)
    verifier = ClaimVerifier(nli, entailment_threshold=g.entailment_threshold)
    return RagService(retriever, llm, verifier, settings=settings.llm, guardrails=g)


def build_container(settings: Settings, *, client: QdrantClient | None = None) -> Container:
    embedder = FastEmbedEmbedder(
        settings.embedding.model,
        batch_size=settings.embedding.batch_size,
        cache_dir=settings.embedding.cache_dir,
    )
    client = client or make_qdrant_client(settings.qdrant)
    dense = QdrantDenseIndex(client, settings.qdrant.collection, embedder.dimension)
    sparse = BM25Index(k1=settings.bm25.k1, b=settings.bm25.b)
    rebuild_sparse(dense, sparse)  # Qdrant is the system of record
    reranker = make_reranker(settings.rerank)
    retriever = HybridRetriever(embedder, dense, sparse, reranker, settings.retrieval)
    return Container(
        settings=settings,
        embedder=embedder,
        dense=dense,
        sparse=sparse,
        retriever=retriever,
        service=_build_service(settings, retriever),
        pipeline=IngestionPipeline.default(embedder, settings.chunking),
        indexer=Indexer(embedder, dense, sparse, batch_size=settings.embedding.batch_size),
        dense_client=client,
    )
