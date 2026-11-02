"""What to do with a verified draft (§5.4)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from hybrid_rag.guardrails.models import ClaimVerdict, FaithfulnessReport

PolicyMode = Literal["annotate", "filter", "abstain"]
AbstainReason = Literal[
    "no_relevant_context", "model_abstained", "no_supported_claims", "low_faithfulness"
]


class GuardrailPolicy(BaseModel):
    """annotate: return the draft with verdicts · filter: drop unsupported sentences ·
    abstain: refuse the whole answer when faithfulness < ``min_faithfulness``."""

    model_config = ConfigDict(frozen=True)

    mode: PolicyMode = "filter"
    min_faithfulness: float = Field(0.8, ge=0, le=1)


class PolicyDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    answer: str | None
    abstain_reason: AbstainReason | None = None
    kept: tuple[ClaimVerdict, ...] = ()

    @property
    def abstained(self) -> bool:
        return self.answer is None


def _abstain(reason: AbstainReason) -> PolicyDecision:
    return PolicyDecision(answer=None, abstain_reason=reason)


def apply_policy(draft: str, report: FaithfulnessReport, policy: GuardrailPolicy) -> PolicyDecision:
    if policy.mode == "annotate":
        return PolicyDecision(answer=draft, kept=report.claims)

    if policy.mode == "filter":
        kept = tuple(v for v in report.claims if v.verdict == "entailed")
        if not kept:
            return _abstain("no_supported_claims")
        # Rebuilt from the kept sentences: unsupported text never reaches the user,
        # and invalid [n] markers are already stripped from ``sentence``.
        return PolicyDecision(answer=" ".join(v.claim.sentence for v in kept), kept=kept)

    if report.faithfulness is None:
        return _abstain("no_supported_claims")
    if report.faithfulness < policy.min_faithfulness:
        return _abstain("low_faithfulness")
    return PolicyDecision(answer=draft, kept=report.claims)
