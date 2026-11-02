from collections.abc import Sequence
from pathlib import Path

import pytest
from conftest import CATS, ROCKETS, FakeEmbedder, make_chunk
from qdrant_client import QdrantClient

from hybrid_rag.config import QdrantSettings
from hybrid_rag.domain.models import ContextualChunk
from hybrid_rag.retrieval.dense import DimensionMismatchError, QdrantDenseIndex, make_qdrant_client

pytestmark = pytest.mark.integration


@pytest.fixture
def index(fake_embedder: FakeEmbedder) -> QdrantDenseIndex:
    return QdrantDenseIndex(QdrantClient(location=":memory:"), "test", fake_embedder.dimension)


def _add(
    index: QdrantDenseIndex, embedder: FakeEmbedder, chunks: Sequence[ContextualChunk]
) -> None:
    index.upsert(chunks, embedder.embed([c.embed_text for c in chunks]))


def test_search_returns_nearest_with_lossless_payload(
    index: QdrantDenseIndex, fake_embedder: FakeEmbedder
) -> None:
    cats = make_chunk(" ".join(CATS), path=("Pets", "Cats"), position=3)
    rockets = make_chunk(" ".join(ROCKETS))
    _add(index, fake_embedder, [cats, rockets])

    hits = index.search(fake_embedder.embed_query("why do cats purr"), k=2)

    assert hits[0].chunk == cats
    assert hits[0].score > hits[1].score


def test_ids_for_doc_and_delete(index: QdrantDenseIndex, fake_embedder: FakeEmbedder) -> None:
    a = make_chunk("alpha text", doc_id="A")
    b = make_chunk("beta text", doc_id="B")
    _add(index, fake_embedder, [a, b])

    assert index.ids_for_doc("A") == {a.chunk_id}
    index.delete([a.chunk_id])
    assert index.ids_for_doc("A") == set()
    assert index.count() == 1


def test_upsert_is_idempotent(index: QdrantDenseIndex, fake_embedder: FakeEmbedder) -> None:
    chunks = [make_chunk("one thing"), make_chunk("another thing")]
    _add(index, fake_embedder, chunks)
    _add(index, fake_embedder, chunks)
    assert index.count() == 2


def test_iter_chunks_paginates_through_everything(
    fake_embedder: FakeEmbedder,
) -> None:
    index = QdrantDenseIndex(
        QdrantClient(location=":memory:"), "t", fake_embedder.dimension, page_size=7
    )
    chunks = [make_chunk(f"chunk number {i}") for i in range(30)]
    _add(index, fake_embedder, chunks)

    assert {c.chunk_id for c in index.iter_chunks()} == {c.chunk_id for c in chunks}


def test_upsert_rejects_misaligned_vectors(
    index: QdrantDenseIndex, fake_embedder: FakeEmbedder
) -> None:
    with pytest.raises(ValueError, match="vectors"):
        index.upsert([make_chunk("x")], fake_embedder.embed(["x", "y"]))


def test_existing_collection_with_other_dimension_is_rejected() -> None:
    client = QdrantClient(location=":memory:")
    QdrantDenseIndex(client, "shared", 256)
    with pytest.raises(DimensionMismatchError):
        QdrantDenseIndex(client, "shared", 128)


def test_client_factory_memory_and_disk(tmp_path: Path, fake_embedder: FakeEmbedder) -> None:
    memory = make_qdrant_client(QdrantSettings(url=None, path=None))
    QdrantDenseIndex(memory, "m", fake_embedder.dimension)

    disk_settings = QdrantSettings(url=None, path=str(tmp_path / "q"))
    chunk = make_chunk("persisted text")
    disk = make_qdrant_client(disk_settings)
    _add(QdrantDenseIndex(disk, "d", fake_embedder.dimension), fake_embedder, [chunk])
    disk.close()

    reopened = QdrantDenseIndex(make_qdrant_client(disk_settings), "d", fake_embedder.dimension)
    assert [c.chunk_id for c in reopened.iter_chunks()] == [chunk.chunk_id]
