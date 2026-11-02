from collections.abc import Iterator

import pytest
from conftest import FakeEmbedder, make_chunk
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from test_hybrid_retriever import CORPUS, SETTINGS, LengthReranker

from hybrid_rag.api.app import create_app
from hybrid_rag.api.container import Container, build_container
from hybrid_rag.config import (
    ChunkingSettings,
    QdrantSettings,
    RerankSettings,
    Settings,
)
from hybrid_rag.indexing import Indexer
from hybrid_rag.ingestion.pipeline import IngestionPipeline
from hybrid_rag.retrieval.dense import QdrantDenseIndex
from hybrid_rag.retrieval.hybrid import HybridRetriever
from hybrid_rag.retrieval.rerank import Reranker
from hybrid_rag.retrieval.sparse import BM25Index

pytestmark = pytest.mark.integration


def make_container(embedder: FakeEmbedder, reranker: Reranker | None) -> Container:
    dense = QdrantDenseIndex(QdrantClient(location=":memory:"), "api", embedder.dimension)
    dense.upsert(CORPUS, embedder.embed([c.embed_text for c in CORPUS]))
    sparse = BM25Index()
    sparse.add(CORPUS)
    return Container(
        settings=Settings(_env_file=None),
        embedder=embedder,
        dense=dense,
        sparse=sparse,
        retriever=HybridRetriever(embedder, dense, sparse, reranker, SETTINGS),
        pipeline=IngestionPipeline.default(embedder, ChunkingSettings()),
        indexer=Indexer(embedder, dense, sparse),
    )


@pytest.fixture
def client(fake_embedder: FakeEmbedder) -> Iterator[TestClient]:
    with TestClient(create_app(make_container(fake_embedder, LengthReranker()))) as c:
        yield c


def test_health_and_readiness(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json() == {"status": "ready", "chunks": len(CORPUS)}


def test_retrieve_returns_ranked_hits_with_timings(client: TestClient) -> None:
    response = client.post("/v1/retrieve", json={"query": "rank fusion", "mode": "hybrid", "k": 4})

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "hybrid"
    assert [h["rank"] for h in body["hits"]] == [1, 2, 3, 4]
    assert set(body["hits"][0]) == {
        "rank",
        "chunk_id",
        "doc_id",
        "text",
        "section_path",
        "page",
        "score",
    }
    assert "total" in body["timings_ms"]


def test_default_mode_is_hybrid_rerank(client: TestClient) -> None:
    body = client.post("/v1/retrieve", json={"query": "rank fusion"}).json()
    assert body["mode"] == "hybrid_rerank"
    assert len(body["hits"]) == 5


@pytest.mark.parametrize(
    "payload",
    [
        {"query": ""},
        {"query": "   "},
        {"query": "x", "k": 0},
        {"query": "x", "k": 51},
        {"query": "x", "mode": "bogus"},
        {"query": "x" * 2001},
    ],
)
def test_invalid_requests_are_rejected(client: TestClient, payload: dict[str, object]) -> None:
    assert client.post("/v1/retrieve", json=payload).status_code == 422


def test_rerank_mode_without_reranker_is_a_clear_client_error(
    fake_embedder: FakeEmbedder,
) -> None:
    with TestClient(create_app(make_container(fake_embedder, None))) as c:
        response = c.post("/v1/retrieve", json={"query": "x", "mode": "hybrid_rerank"})
        assert response.status_code == 400
        assert "reranker" in response.json()["detail"]
        assert c.post("/v1/retrieve", json={"query": "x", "mode": "hybrid"}).status_code == 200


def test_openapi_documents_the_contract(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert "/v1/retrieve" in schema["paths"]
    request = schema["components"]["schemas"]["RetrieveRequest"]
    assert request["examples"]
    assert set(request["properties"]["mode"]["enum"]) == {
        "dense",
        "sparse",
        "hybrid",
        "hybrid_rerank",
    }


def test_build_container_wires_components_and_rebuilds_bm25(
    monkeypatch: pytest.MonkeyPatch, fake_embedder: FakeEmbedder
) -> None:
    monkeypatch.setattr(
        "hybrid_rag.api.container.FastEmbedEmbedder", lambda *a, **kw: fake_embedder
    )
    settings = Settings(
        _env_file=None,
        qdrant=QdrantSettings(url=None, path=None),
        rerank=RerankSettings(provider="none"),
    )
    container = build_container(settings)
    chunk = make_chunk("persisted in qdrant only")
    container.dense.upsert([chunk], fake_embedder.embed([chunk.embed_text]))

    rebuilt = build_container(settings, client=container.dense_client)

    assert chunk.chunk_id in rebuilt.sparse
    assert not rebuilt.retriever.has_reranker
