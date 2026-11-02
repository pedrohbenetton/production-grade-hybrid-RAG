"""Composition root: the only place where concrete adapters are chosen."""

from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import QdrantClient

from hybrid_rag.config import Settings
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
    return Container(
        settings=settings,
        embedder=embedder,
        dense=dense,
        sparse=sparse,
        retriever=HybridRetriever(embedder, dense, sparse, reranker, settings.retrieval),
        pipeline=IngestionPipeline.default(embedder, settings.chunking),
        indexer=Indexer(embedder, dense, sparse, batch_size=settings.embedding.batch_size),
        dense_client=client,
    )
