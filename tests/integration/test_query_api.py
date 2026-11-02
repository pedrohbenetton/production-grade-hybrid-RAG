from collections.abc import Iterator
from dataclasses import replace

import pytest
from conftest import FakeEmbedder, FakeLLM, SubsetVerifier
from fastapi.testclient import TestClient
from test_api import make_container
from test_hybrid_retriever import LengthReranker

from hybrid_rag.api.app import create_app
from hybrid_rag.config import GuardrailSettings, LLMSettings
from hybrid_rag.generation.llm import LLMError
from hybrid_rag.generation.service import RagService

pytestmark = pytest.mark.integration

QUESTION = "reciprocal rank fusion merges rankings"
GOOD = "Reciprocal rank fusion merges rankings [1]."
BAD = "Rankings come from the moon cheese [1]."


def app_client(embedder: FakeEmbedder, llm: FakeLLM | None) -> TestClient:
    container = make_container(embedder, LengthReranker())
    if llm is not None:
        service = RagService(
            container.retriever,
            llm,
            SubsetVerifier(),
            settings=LLMSettings(),
            guardrails=GuardrailSettings(),
        )
        container = replace(container, service=service)
    return TestClient(create_app(container))


@pytest.fixture
def client(fake_embedder: FakeEmbedder) -> Iterator[TestClient]:
    with app_client(fake_embedder, FakeLLM(f"{GOOD} {BAD}")) as c:
        yield c


def test_query_returns_a_filtered_answer_with_claims_citations_and_timings(
    client: TestClient,
) -> None:
    response = client.post("/v1/query", json={"query": QUESTION, "mode": "hybrid", "k": 3})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == GOOD
    assert body["abstained"] is False
    assert body["abstain_reason"] is None
    assert body["faithfulness"] == 0.5
    assert [(c["verdict"], c["citations"]) for c in body["claims"]] == [
        ("entailed", [1]),
        ("neutral", [1]),
    ]
    [citation] = body["citations"]
    assert citation["n"] == 1
    assert "Reciprocal rank fusion" in citation["text"]
    assert {"retrieval", "generate", "verify", "total"} <= set(body["timings_ms"])


def test_policy_can_be_overridden_per_request(client: TestClient) -> None:
    annotated = client.post("/v1/query", json={"query": QUESTION, "policy": "annotate"}).json()
    refused = client.post(
        "/v1/query", json={"query": QUESTION, "mode": "hybrid", "policy": "abstain"}
    ).json()

    assert annotated["answer"] == f"{GOOD} {BAD}"
    assert refused["abstained"] is True
    assert refused["abstain_reason"] == "low_faithfulness"
    assert refused["answer"] is None


def test_query_is_a_clear_client_error_when_generation_is_not_configured(
    fake_embedder: FakeEmbedder,
) -> None:
    with app_client(fake_embedder, None) as c:
        response = c.post("/v1/query", json={"query": QUESTION})

    assert response.status_code == 400
    assert "ANTHROPIC_API_KEY" in response.json()["detail"]


def test_llm_failures_surface_as_bad_gateway(fake_embedder: FakeEmbedder) -> None:
    with app_client(fake_embedder, FakeLLM(error=LLMError("upstream down"))) as c:
        response = c.post("/v1/query", json={"query": QUESTION})

    assert response.status_code == 502
    assert "upstream down" in response.json()["detail"]


def test_missing_reranker_is_reported_like_on_the_retrieve_endpoint(
    fake_embedder: FakeEmbedder,
) -> None:
    container = make_container(fake_embedder, None)
    service = RagService(
        container.retriever,
        FakeLLM(GOOD),
        SubsetVerifier(),
        settings=LLMSettings(),
        guardrails=GuardrailSettings(),
    )
    with TestClient(create_app(replace(container, service=service))) as c:
        response = c.post("/v1/query", json={"query": QUESTION, "mode": "hybrid_rerank"})

    assert response.status_code == 400
    assert "reranker" in response.json()["detail"]


@pytest.mark.parametrize(
    "payload",
    [
        {"query": ""},
        {"query": "x", "k": 21},
        {"query": "x", "mode": "bogus"},
        {"query": "x", "policy": "bogus"},
    ],
)
def test_invalid_requests_are_rejected(client: TestClient, payload: dict[str, object]) -> None:
    assert client.post("/v1/query", json=payload).status_code == 422


def test_openapi_documents_the_query_contract(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()

    assert "/v1/query" in schema["paths"]
    request = schema["components"]["schemas"]["QueryRequest"]
    assert request["examples"]
    response = schema["components"]["schemas"]["QueryResponse"]
    assert {"answer", "abstained", "faithfulness", "claims", "citations"} <= set(
        response["properties"]
    )
