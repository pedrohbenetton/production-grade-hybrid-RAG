from collections.abc import Sequence

import pytest
from conftest import FakeLLM, SubsetVerifier, make_chunk

from hybrid_rag.config import GuardrailSettings, LLMSettings
from hybrid_rag.domain.models import ScoredChunk
from hybrid_rag.generation.prompts import SYSTEM_PROMPT
from hybrid_rag.generation.service import RagService
from hybrid_rag.guardrails.claims import ABSTAIN_PHRASE
from hybrid_rag.guardrails.policy import GuardrailPolicy
from hybrid_rag.retrieval.hybrid import RetrievalMode, RetrievalResult

RRF = make_chunk(
    "Reciprocal rank fusion merges rankings without calibrating scores.", chunk_id="c1"
)
RERANK = make_chunk("Cross-encoder rerankers jointly score query and passage pairs.", chunk_id="c2")
HITS = [ScoredChunk(chunk=RRF, score=0.9), ScoredChunk(chunk=RERANK, score=0.7)]

GOOD = "Reciprocal rank fusion merges rankings [1]."
BAD = "Rankings come from the moon cheese [1]."


class FakeRetriever:
    def __init__(self, hits: Sequence[ScoredChunk]) -> None:
        self.hits = tuple(hits)
        self.calls: list[tuple[str, RetrievalMode, int]] = []

    def retrieve(self, query: str, *, mode: RetrievalMode, k: int) -> RetrievalResult:
        self.calls.append((query, mode, k))
        return RetrievalResult(
            query=query, mode=mode, hits=self.hits, timings_ms={"embed": 1.0, "total": 2.0}
        )


def build(
    reply: str,
    hits: Sequence[ScoredChunk] = HITS,
    guardrails: GuardrailSettings | None = None,
) -> tuple[RagService, FakeRetriever, FakeLLM, SubsetVerifier]:
    retriever, llm, verifier = FakeRetriever(hits), FakeLLM(reply), SubsetVerifier()
    service = RagService(
        retriever,
        llm,
        verifier,
        settings=LLMSettings(),
        guardrails=guardrails or GuardrailSettings(),
    )
    return service, retriever, llm, verifier


def test_grounded_answer_flows_through_retrieval_generation_and_verification() -> None:
    service, retriever, llm, _ = build(GOOD)

    result = service.answer("what is rrf?", mode="hybrid", k=3)

    assert retriever.calls == [("what is rrf?", "hybrid", 3)]
    assert result.answer == GOOD
    assert not result.abstained
    assert result.abstain_reason is None
    assert result.faithfulness == 1.0
    assert [(c.n, c.chunk_id) for c in result.citations] == [(1, "c1")]
    assert {"embed", "retrieval", "generate", "verify", "total"} <= set(result.timings_ms)
    [(system, prompt, max_tokens)] = llm.calls
    assert system == SYSTEM_PROMPT
    assert max_tokens == LLMSettings().max_tokens
    assert "what is rrf?" in prompt
    assert '<passage id="1"' in prompt
    assert "Cross-encoder rerankers" in prompt


def test_filter_policy_removes_unsupported_claims_and_their_citations() -> None:
    service, *_ = build(f"{GOOD} {BAD}")

    result = service.answer("q")

    assert result.answer == GOOD
    assert result.faithfulness == 0.5
    assert [v.verdict for v in result.claims] == [
        "entailed",
        "neutral",
    ]  # dropped claims stay visible
    assert [c.n for c in result.citations] == [1]


def test_annotate_policy_returns_the_full_draft_with_verdicts() -> None:
    service, *_ = build(f"{GOOD} {BAD}")

    result = service.answer("q", policy=GuardrailPolicy(mode="annotate"))

    assert result.answer == f"{GOOD} {BAD}"
    assert result.faithfulness == 0.5
    assert len(result.claims) == 2


def test_abstain_policy_refuses_a_low_faithfulness_answer_but_still_reports_why() -> None:
    service, *_ = build(f"{GOOD} {BAD}")

    result = service.answer("q", policy=GuardrailPolicy(mode="abstain", min_faithfulness=0.8))

    assert result.abstained
    assert result.abstain_reason == "low_faithfulness"
    assert result.answer is None
    assert result.citations == ()
    assert result.faithfulness == 0.5
    assert len(result.claims) == 2


def test_default_policy_comes_from_settings() -> None:
    settings = GuardrailSettings(policy="abstain", min_faithfulness=0.9)
    service, *_ = build(f"{GOOD} {BAD}", guardrails=settings)

    assert service.answer("q").abstain_reason == "low_faithfulness"


def test_a_model_that_declines_is_reported_without_running_verification() -> None:
    service, _, _, verifier = build(ABSTAIN_PHRASE)

    result = service.answer("q")

    assert result.abstained
    assert result.abstain_reason == "model_abstained"
    assert result.answer is None
    assert result.faithfulness is None
    assert verifier.calls == 0


def test_no_retrieved_context_abstains_without_calling_the_llm() -> None:
    service, _, llm, verifier = build(GOOD, hits=[])

    result = service.answer("q")

    assert result.abstain_reason == "no_relevant_context"
    assert llm.calls == []
    assert verifier.calls == 0


def test_top_hit_below_min_relevance_abstains_before_generation() -> None:
    weak = [ScoredChunk(chunk=RRF, score=0.2)]
    service, _, llm, _ = build(GOOD, hits=weak, guardrails=GuardrailSettings(min_relevance=0.5))

    assert service.answer("q").abstain_reason == "no_relevant_context"
    assert llm.calls == []


def test_min_relevance_is_off_by_default() -> None:
    weak = [ScoredChunk(chunk=RRF, score=0.001)]
    service, _, llm, _ = build(GOOD, hits=weak)

    assert not service.answer("q").abstained
    assert len(llm.calls) == 1


def test_citations_to_missing_passages_are_reported_and_stripped_in_the_filtered_answer() -> None:
    service, *_ = build("Reciprocal rank fusion merges rankings [5].")

    result = service.answer("q")

    assert result.invalid_citations == (5,)
    assert result.answer == "Reciprocal rank fusion merges rankings."
    assert result.citations == ()
    assert result.faithfulness == 1.0  # still verified against all passages


@pytest.mark.parametrize("reply", ["", "   "])
def test_a_blank_completion_is_treated_as_no_supported_claims(reply: str) -> None:
    service, *_ = build(reply)
    assert service.answer("q").abstain_reason == "no_supported_claims"
