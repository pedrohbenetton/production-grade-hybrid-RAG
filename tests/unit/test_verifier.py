from collections.abc import Callable, Sequence

import pytest
from conftest import make_chunk

from hybrid_rag.domain.models import ScoredChunk
from hybrid_rag.generation.context import ContextPassage, build_context
from hybrid_rag.guardrails.models import Claim, NLIScores
from hybrid_rag.guardrails.verify import ClaimVerifier
from hybrid_rag.ingestion.tokens import RegexTokenCounter

ENTAIL = NLIScores(entailment=0.9, neutral=0.05, contradiction=0.05)
NEUTRAL = NLIScores(entailment=0.2, neutral=0.7, contradiction=0.1)
CONTRA = NLIScores(entailment=0.05, neutral=0.15, contradiction=0.8)


class FakeNLI:
    def __init__(self, rule: Callable[[str, str], NLIScores]) -> None:
        self.rule = rule
        self.batches: list[list[tuple[str, str]]] = []

    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[NLIScores]:
        self.batches.append(list(pairs))
        return [self.rule(premise, hypothesis) for premise, hypothesis in pairs]


def passages(*texts: str) -> tuple[ContextPassage, ...]:
    hits = [
        ScoredChunk(chunk=make_chunk(t, chunk_id=f"c{i}"), score=1.0) for i, t in enumerate(texts)
    ]
    return build_context(hits, token_budget=10_000, counter=RegexTokenCounter()).passages


def claim(text: str, *citations: int) -> Claim:
    return Claim(text=text, sentence=text, citations=citations)


def test_a_cited_claim_is_checked_only_against_the_passages_it_cites() -> None:
    nli = FakeNLI(lambda p, h: ENTAIL)
    ctx = passages("alpha text", "beta text", "gamma text")

    [verdict] = ClaimVerifier(nli).verify([claim("beta holds", 2)], ctx)

    assert nli.batches == [[("beta text", "beta holds")]]
    assert (verdict.verdict, verdict.evidence) == ("entailed", 2)


def test_an_uncited_claim_is_checked_against_every_passage_and_reports_the_best_one() -> None:
    nli = FakeNLI(lambda p, h: ENTAIL if p == "beta text" else NEUTRAL)
    ctx = passages("alpha text", "beta text", "gamma text")

    [verdict] = ClaimVerifier(nli).verify([claim("beta holds")], ctx)

    assert len(nli.batches[0]) == 3
    assert (verdict.verdict, verdict.evidence, verdict.entailment) == ("entailed", 2, 0.9)


def test_a_claim_citing_the_wrong_passage_is_not_rescued_by_other_passages() -> None:
    nli = FakeNLI(lambda p, h: ENTAIL if p == "beta text" else NEUTRAL)
    ctx = passages("alpha text", "beta text")

    [verdict] = ClaimVerifier(nli).verify([claim("beta holds", 1)], ctx)

    assert verdict.verdict == "neutral"
    assert verdict.evidence is None


@pytest.mark.parametrize(
    ("scores", "expected"), [(ENTAIL, "entailed"), (NEUTRAL, "neutral"), (CONTRA, "contradicted")]
)
def test_verdict_follows_the_nli_probabilities(scores: NLIScores, expected: str) -> None:
    [verdict] = ClaimVerifier(FakeNLI(lambda p, h: scores)).verify(
        [claim("anything", 1)], passages("a passage")
    )
    assert verdict.verdict == expected


def test_entailment_beats_contradiction_from_another_passage() -> None:
    nli = FakeNLI(lambda p, h: ENTAIL if p == "yes" else CONTRA)
    [verdict] = ClaimVerifier(nli).verify([claim("c")], passages("no", "yes"))
    assert verdict.verdict == "entailed"


def test_threshold_is_configurable() -> None:
    borderline = NLIScores(entailment=0.6, neutral=0.3, contradiction=0.1)
    strict = ClaimVerifier(FakeNLI(lambda p, h: borderline), entailment_threshold=0.7)
    lenient = ClaimVerifier(FakeNLI(lambda p, h: borderline), entailment_threshold=0.5)

    ctx = passages("a passage")
    assert strict.verify([claim("c", 1)], ctx)[0].verdict == "neutral"
    assert lenient.verify([claim("c", 1)], ctx)[0].verdict == "entailed"


def test_all_claims_are_scored_in_a_single_batch_in_order() -> None:
    nli = FakeNLI(lambda p, h: ENTAIL if h == "first" else NEUTRAL)
    ctx = passages("one", "two")

    verdicts = ClaimVerifier(nli).verify([claim("first", 1), claim("second", 2)], ctx)

    assert len(nli.batches) == 1
    assert [v.verdict for v in verdicts] == ["entailed", "neutral"]
    assert [v.claim.text for v in verdicts] == ["first", "second"]


def test_no_claims_means_no_model_call() -> None:
    nli = FakeNLI(lambda p, h: ENTAIL)
    assert ClaimVerifier(nli).verify([], passages("x")) == []
    assert nli.batches == []


def test_a_claim_with_no_premises_is_unsupported() -> None:
    nli = FakeNLI(lambda p, h: ENTAIL)
    [verdict] = ClaimVerifier(nli).verify([claim("orphan")], ())
    assert (verdict.verdict, verdict.entailment) == ("neutral", 0.0)


@pytest.mark.parametrize("threshold", [0.0, -0.1, 1.5])
def test_invalid_threshold_is_rejected(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        ClaimVerifier(FakeNLI(lambda p, h: ENTAIL), entailment_threshold=threshold)
