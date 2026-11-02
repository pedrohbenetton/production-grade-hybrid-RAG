import os
from collections.abc import Sequence
from dataclasses import dataclass, field

import pytest
from conftest import CATS, ROCKETS, make_chunk, ranked
from pydantic import SecretStr

from hybrid_rag.config import RerankSettings
from hybrid_rag.domain.models import ScoredChunk
from hybrid_rag.retrieval.rerank import CohereReranker, make_reranker


@dataclass(frozen=True)
class Result:
    index: int
    relevance_score: float


@dataclass(frozen=True)
class Response:
    results: list[Result]


@dataclass
class FakeCohere:
    calls: list[dict[str, object]] = field(default_factory=list)

    def rerank(self, *, model: str, query: str, documents: Sequence[str], top_n: int) -> Response:
        self.calls.append(
            {"model": model, "query": query, "documents": list(documents), "top_n": top_n}
        )
        return Response(results=[Result(index=2, relevance_score=0.9), Result(0, 0.4)])


def test_cohere_maps_result_indices_back_to_chunks() -> None:
    client = FakeCohere()
    candidates = ranked("a", "b", "c")

    hits = CohereReranker(client, model="rerank-v3.5").rerank("q", candidates, top_k=2)

    assert [(h.chunk.chunk_id, h.score) for h in hits] == [("c", 0.9), ("a", 0.4)]
    [call] = client.calls
    assert call == {
        "model": "rerank-v3.5",
        "query": "q",
        "documents": [c.chunk.embed_text for c in candidates],
        "top_n": 2,
    }


def test_empty_candidates_skip_the_remote_call() -> None:
    client = FakeCohere()
    assert CohereReranker(client).rerank("q", [], top_k=3) == []
    assert client.calls == []


def test_make_reranker_none_and_missing_cohere_key() -> None:
    assert make_reranker(RerankSettings(provider="none")) is None
    with pytest.raises(ValueError, match="COHERE_API_KEY"):
        make_reranker(RerankSettings(provider="cohere"))


def test_make_reranker_builds_cohere_client() -> None:
    pytest.importorskip("cohere", reason="optional extra: uv sync --extra cohere")
    settings = RerankSettings(provider="cohere", cohere_api_key=SecretStr("test-key"))
    assert isinstance(make_reranker(settings), CohereReranker)


@pytest.mark.models
@pytest.mark.skipif(os.environ.get("RUN_MODEL_TESTS") != "1", reason="set RUN_MODEL_TESTS=1")
def test_flashrank_cross_encoder_puts_relevant_passage_first() -> None:
    from hybrid_rag.retrieval.rerank import FlashRankReranker

    reranker = FlashRankReranker()
    candidates = [
        ScoredChunk(chunk=make_chunk(" ".join(ROCKETS), chunk_id="rockets"), score=1.0),
        ScoredChunk(chunk=make_chunk(" ".join(CATS), chunk_id="cats"), score=0.5),
    ]

    hits = reranker.rerank("Why do cats purr?", candidates, top_k=2)

    assert [h.chunk.chunk_id for h in hits] == ["cats", "rockets"]
    assert hits[0].score > hits[1].score
