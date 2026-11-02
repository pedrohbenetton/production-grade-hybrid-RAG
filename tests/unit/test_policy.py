import pytest
from pydantic import ValidationError

from hybrid_rag.guardrails.models import Claim, ClaimVerdict, FaithfulnessReport, Verdict
from hybrid_rag.guardrails.policy import GuardrailPolicy, apply_policy


def verdict(text: str, v: Verdict, *citations: int) -> ClaimVerdict:
    markers = "".join(f"[{n}]" for n in citations)
    # As extract_claims renders it: markers sit before the final period.
    sentence = f"{text.rstrip('.')} {markers}." if markers else text
    claim = Claim(text=text, sentence=sentence, citations=citations)
    evidence = citations[0] if v == "entailed" and citations else None
    return ClaimVerdict(
        claim=claim, verdict=v, entailment=1.0 if v == "entailed" else 0.0, evidence=evidence
    )


DRAFT = "Good claim [1]. Bad claim [2]. Fine claim [1]."
MIXED = FaithfulnessReport(
    claims=(
        verdict("Good claim.", "entailed", 1),
        verdict("Bad claim.", "contradicted", 2),
        verdict("Fine claim.", "entailed", 1),
    )
)


def test_faithfulness_is_the_share_of_entailed_claims() -> None:
    assert MIXED.faithfulness == pytest.approx(2 / 3)
    assert FaithfulnessReport(claims=()).faithfulness is None


def test_annotate_returns_the_draft_untouched_with_every_verdict() -> None:
    decision = apply_policy(DRAFT, MIXED, GuardrailPolicy(mode="annotate"))

    assert decision.answer == DRAFT
    assert not decision.abstained
    assert decision.kept == MIXED.claims


def test_filter_keeps_only_entailed_sentences_in_order() -> None:
    decision = apply_policy(DRAFT, MIXED, GuardrailPolicy(mode="filter"))

    assert decision.answer == "Good claim [1]. Fine claim [1]."
    assert [v.claim.text for v in decision.kept] == ["Good claim.", "Fine claim."]
    assert decision.abstain_reason is None


def test_filter_abstains_when_nothing_is_supported() -> None:
    report = FaithfulnessReport(claims=(verdict("Bad.", "neutral", 1),))
    decision = apply_policy("Bad [1].", report, GuardrailPolicy(mode="filter"))

    assert decision.abstained
    assert decision.abstain_reason == "no_supported_claims"
    assert decision.answer is None
    assert decision.kept == ()


def test_abstain_mode_refuses_below_the_threshold() -> None:
    decision = apply_policy(DRAFT, MIXED, GuardrailPolicy(mode="abstain", min_faithfulness=0.8))

    assert decision.abstained
    assert decision.abstain_reason == "low_faithfulness"
    assert decision.answer is None


def test_abstain_mode_passes_at_or_above_the_threshold() -> None:
    decision = apply_policy(DRAFT, MIXED, GuardrailPolicy(mode="abstain", min_faithfulness=2 / 3))

    assert decision.answer == DRAFT
    assert decision.kept == MIXED.claims


def test_abstain_mode_with_no_claims_refuses() -> None:
    decision = apply_policy("", FaithfulnessReport(claims=()), GuardrailPolicy(mode="abstain"))
    assert decision.abstain_reason == "no_supported_claims"


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_min_faithfulness_must_be_a_probability(value: float) -> None:
    with pytest.raises(ValidationError):
        GuardrailPolicy(min_faithfulness=value)
