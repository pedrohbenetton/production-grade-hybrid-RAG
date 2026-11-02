from collections.abc import Iterator

import pytest
from conftest import FakeEmbedder, make_chunk
from fastapi.testclient import TestClient
from pydantic import SecretStr
from qdrant_client import QdrantClient
from test_hybrid_retriever import CORPUS, SETTINGS, LengthReranker

from hybrid_rag.api.app import create_app
from hybrid_rag.api.container import Container, build_container
from hybrid_rag.config import (
    ChunkingSettings,
    LLMSettings,
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
    assert "/v1/ingest" in schema["paths"]
    assert "/v1/ingest/file" in schema["paths"]
    assert "/v1/documents/{doc_id}" in schema["paths"]
    request = schema["components"]["schemas"]["RetrieveRequest"]
    assert request["examples"]
    assert set(request["properties"]["mode"]["enum"]) == {
        "dense",
        "sparse",
        "hybrid",
        "hybrid_rerank",
    }


def test_ingest_json_markdown_idempotent_and_delete(client: TestClient) -> None:
    doc_payload = {
        "source_uri": "quantum_computing.md",
        "content": (
            "# Quantum Computing\n\n"
            "Superconducting qubits enable quantum supremacy across combinatorial problems.\n\n"
            "## Error Correction\n\n"
            "Surface codes provide fault tolerance against phase and bit flips."
        ),
    }

    initial_chunks = client.get("/ready").json()["chunks"]

    # 1. First ingest: adds new chunks
    res = client.post("/v1/ingest", json=doc_payload)
    assert res.status_code == 200
    data = res.json()
    doc_id = data["doc_id"]
    added = data["added"]
    assert added > 0
    assert data["unchanged"] == 0
    assert data["removed"] == 0
    assert client.get("/ready").json()["chunks"] == initial_chunks + added

    # 2. Re-ingest unchanged: zero embeddings spent, chunks unchanged
    res_again = client.post("/v1/ingest", json=doc_payload)
    assert res_again.status_code == 200
    data_again = res_again.json()
    assert data_again["doc_id"] == doc_id
    assert data_again["added"] == 0
    assert data_again["unchanged"] == added
    assert data_again["removed"] == 0

    # 3. Retrieve: search for the new document
    ret_res = client.post(
        "/v1/retrieve", json={"query": "superconducting qubits quantum supremacy", "k": 3}
    )
    assert ret_res.status_code == 200
    top_hit = ret_res.json()["hits"][0]
    assert top_hit["doc_id"] == doc_id
    assert "Superconducting qubits" in top_hit["text"]

    # 4. Delete document by hashed doc_id
    del_res = client.delete(f"/v1/documents/{doc_id}")
    assert del_res.status_code == 200
    del_data = del_res.json()
    assert del_data["doc_id"] == doc_id
    assert del_data["deleted_chunks"] == added
    assert client.get("/ready").json()["chunks"] == initial_chunks

    # 5. Delete again returns 404
    assert client.delete(f"/v1/documents/{doc_id}").status_code == 404


def test_delete_by_source_uri_fallback(client: TestClient) -> None:
    uri = "folder/subfolder/notes.md"
    client.post(
        "/v1/ingest",
        json={
            "source_uri": uri,
            "content": "# Notes\n\nUnique telemetry data points for rocket trajectory.",
        },
    )

    del_res = client.delete(f"/v1/documents/{uri}")
    assert del_res.status_code == 200
    assert del_res.json()["deleted_chunks"] > 0
    assert client.delete(f"/v1/documents/{uri}").status_code == 404


def test_ingest_file_multipart(client: TestClient) -> None:
    content = b"# Distributed Systems\n\nConsensus protocols like Raft ensure state machine safety."
    res = client.post(
        "/v1/ingest/file",
        files={"file": ("raft.md", content, "text/markdown")},
        data={"source_uri": "raft.md"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["added"] > 0

    # Clean up
    client.delete(f"/v1/documents/{data['doc_id']}")


def test_ingest_validation_errors(client: TestClient) -> None:
    # Unsupported extension
    res = client.post(
        "/v1/ingest",
        json={"source_uri": "archive.zip", "content": "not a supported format"},
    )
    assert res.status_code == 400
    assert "not supported" in res.json()["detail"]

    # Blank content
    res_blank = client.post(
        "/v1/ingest",
        json={"source_uri": "notes.md", "content": "   \n\t   "},
    )
    assert res_blank.status_code == 400

    # Empty file upload
    res_file = client.post(
        "/v1/ingest/file",
        files={"file": ("empty.md", b"", "text/markdown")},
    )
    assert res_file.status_code == 400
    assert "must not be empty" in res_file.json()["detail"]

    # Delete missing
    assert client.delete("/v1/documents/non_existent_doc_id").status_code == 404


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
        llm=LLMSettings(provider="none"),
    )
    container = build_container(settings)
    chunk = make_chunk("persisted in qdrant only")
    container.dense.upsert([chunk], fake_embedder.embed([chunk.embed_text]))

    rebuilt = build_container(settings, client=container.dense_client)

    assert chunk.chunk_id in rebuilt.sparse
    assert not rebuilt.retriever.has_reranker
    assert rebuilt.service is None


def test_build_container_wires_generation_only_when_an_llm_is_configured(
    monkeypatch: pytest.MonkeyPatch, fake_embedder: FakeEmbedder
) -> None:
    built: list[tuple[object, ...]] = []

    class StubNLI:
        def __init__(self, *args: object, **kwargs: object) -> None:
            built.append(args)

    monkeypatch.setattr(
        "hybrid_rag.api.container.FastEmbedEmbedder", lambda *a, **kw: fake_embedder
    )
    monkeypatch.setattr("hybrid_rag.api.container.OnnxNLIModel", StubNLI)
    settings = Settings(
        _env_file=None,
        qdrant=QdrantSettings(url=None, path=None),
        rerank=RerankSettings(provider="none"),
        llm=LLMSettings(api_key=SecretStr("test-key")),
    )

    container = build_container(settings)

    assert container.service is not None
    assert built == [(settings.guardrails.nli_repo, settings.guardrails.nli_onnx_file)]
