from dataclasses import dataclass
from pathlib import Path

import pytest
from conftest import FakeEmbedder
from qdrant_client import QdrantClient

from hybrid_rag.config import ChunkingSettings
from hybrid_rag.indexing import Indexer, rebuild_sparse
from hybrid_rag.ingestion.pipeline import IngestionPipeline
from hybrid_rag.ingestion.sources import load_file
from hybrid_rag.retrieval.dense import QdrantDenseIndex
from hybrid_rag.retrieval.sparse import BM25Index

pytestmark = pytest.mark.integration

V1 = """# Notes

## Alpha
Alpha section explains orchards, apples and seasonal harvests in detail.

## Beta
Beta section explains bridges, rivers and suspension cable engineering.
"""
V2 = V1.replace("suspension cable engineering", "arched masonry construction")


@dataclass
class Stack:
    embedder: FakeEmbedder
    dense: QdrantDenseIndex
    sparse: BM25Index
    indexer: Indexer
    pipeline: IngestionPipeline


@pytest.fixture
def stack(fake_embedder: FakeEmbedder) -> Stack:
    dense = QdrantDenseIndex(QdrantClient(location=":memory:"), "t", fake_embedder.dimension)
    sparse = BM25Index()
    return Stack(
        embedder=fake_embedder,
        dense=dense,
        sparse=sparse,
        indexer=Indexer(fake_embedder, dense, sparse, batch_size=2),
        pipeline=IngestionPipeline.default(
            fake_embedder, ChunkingSettings(min_tokens=5, max_tokens=60)
        ),
    )


def _ingest(stack: Stack, path: Path, content: str):  # type: ignore[no-untyped-def]
    path.write_text(content, encoding="utf-8")
    processed = stack.pipeline.process(load_file(path))
    return processed, stack.indexer.index(processed)


def test_first_index_adds_everything_to_both_stores(stack: Stack, tmp_path: Path) -> None:
    processed, report = _ingest(stack, tmp_path / "notes.md", V1)

    n = len(processed.chunks)
    assert (report.added, report.unchanged, report.removed) == (n, 0, 0)
    assert report.doc_id == processed.tree.doc_id
    assert stack.dense.count() == len(stack.sparse) == n


def test_unchanged_reindex_is_a_noop_without_embedding_calls(stack: Stack, tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    processed, _ = _ingest(stack, path, V1)
    calls_before = stack.embedder.calls

    _, report = _ingest(stack, path, V1)

    assert (report.added, report.unchanged, report.removed) == (0, len(processed.chunks), 0)
    # The chunker may embed for semantic splitting, but the indexer must not.
    assert stack.embedder.calls == calls_before


def test_edited_section_replaces_only_its_chunk(stack: Stack, tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    _ingest(stack, path, V1)

    _, report = _ingest(stack, path, V2)

    assert (report.added, report.unchanged, report.removed) == (1, 1, 1)
    assert stack.sparse.search("suspension cable", k=5) == []
    [hit] = stack.sparse.search("masonry", k=5)
    assert hit.chunk.section_path == ("Beta",)
    assert stack.dense.count() == len(stack.sparse) == 2


def test_delete_document_clears_both_stores(stack: Stack, tmp_path: Path) -> None:
    processed, _ = _ingest(stack, tmp_path / "notes.md", V1)

    removed = stack.indexer.delete_document(processed.tree.doc_id)

    assert removed == len(processed.chunks)
    assert stack.dense.count() == len(stack.sparse) == 0
    assert stack.indexer.delete_document(processed.tree.doc_id) == 0


def test_sparse_index_can_be_rebuilt_from_the_dense_store(stack: Stack, tmp_path: Path) -> None:
    _ingest(stack, tmp_path / "notes.md", V1)

    rebuilt = BM25Index()
    assert rebuild_sparse(stack.dense, rebuilt) == len(stack.sparse)
    for query in ("apples harvest", "bridges rivers"):
        live = [(h.chunk.chunk_id, h.score) for h in stack.sparse.search(query, k=5)]
        again = [(h.chunk.chunk_id, h.score) for h in rebuilt.search(query, k=5)]
        assert again == live
