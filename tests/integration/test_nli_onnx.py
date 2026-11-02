import os

import pytest
from conftest import make_chunk

from hybrid_rag.domain.models import ScoredChunk
from hybrid_rag.generation.context import build_context
from hybrid_rag.guardrails.claims import extract_claims
from hybrid_rag.guardrails.verify import ClaimVerifier
from hybrid_rag.ingestion.tokens import RegexTokenCounter

pytestmark = [
    pytest.mark.models,
    pytest.mark.skipif(
        os.environ.get("RUN_MODEL_TESTS") != "1", reason="set RUN_MODEL_TESTS=1 (~180 MB download)"
    ),
]

PREMISE = "Reciprocal rank fusion merges several ranked lists without calibrating their scores."


@pytest.fixture(scope="module")
def nli() -> object:
    from hybrid_rag.guardrails.nli import OnnxNLIModel

    return OnnxNLIModel(cache_dir="./data/models")


def test_entailment_is_detected_reliably(nli: object) -> None:
    """Policies only act on ``entailed``. The neutral/contradiction split is NOT asserted:
    SNLI/MNLI-style models label off-topic claims as contradictions (measured)."""
    pairs = [
        (PREMISE, "Reciprocal rank fusion combines multiple rankings."),
        (PREMISE, "Reciprocal rank fusion merges ranked lists."),
        (PREMISE, "Reciprocal rank fusion needs scores to be calibrated first."),
        (PREMISE, "The Eiffel Tower is located in Paris."),
    ]
    scores = nli.predict(pairs)  # type: ignore[attr-defined]

    assert [s.entailment > 0.9 for s in scores] == [True, True, False, False]
    assert max(scores[2].entailment, scores[3].entailment) < 0.5
    for s in scores:
        assert s.entailment + s.neutral + s.contradiction == pytest.approx(1.0, abs=1e-3)


def test_long_premises_are_truncated_instead_of_failing(nli: object) -> None:
    long_premise = " ".join([PREMISE] * 200)
    [score] = nli.predict([(long_premise, "Reciprocal rank fusion combines rankings.")])  # type: ignore[attr-defined]
    assert score.entailment > 0.5


def test_end_to_end_claim_verification_flags_the_hallucination(nli: object) -> None:
    hits = [ScoredChunk(chunk=make_chunk(PREMISE, chunk_id="c1"), score=1.0)]
    context = build_context(hits, token_budget=1000, counter=RegexTokenCounter())
    claims, _ = extract_claims(
        "Reciprocal rank fusion merges ranked lists [1]. It was invented in 1999 by Google [1].", 1
    )

    verdicts = ClaimVerifier(nli).verify(claims, context.passages)  # type: ignore[arg-type]

    assert verdicts[0].verdict == "entailed"
    assert verdicts[1].verdict != "entailed"  # neutral vs contradicted is model-dependent
