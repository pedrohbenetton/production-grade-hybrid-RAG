from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field

Verdict = Literal["entailed", "neutral", "contradicted"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Claim(_Frozen):
    """One sentence of the draft answer.

    ``text`` is citation-free (what the NLI model sees); ``sentence`` is what is
    shown to users, keeping valid ``[n]`` markers and dropping invalid ones.
    """

    text: str
    sentence: str
    citations: tuple[int, ...] = ()


class NLIScores(_Frozen):
    entailment: float = Field(ge=0, le=1)
    neutral: float = Field(ge=0, le=1)
    contradiction: float = Field(ge=0, le=1)


class ClaimVerdict(_Frozen):
    claim: Claim
    verdict: Verdict
    entailment: float = Field(ge=0, le=1, description="Best entailment probability found.")
    evidence: int | None = Field(None, description="Passage number that supports the claim.")


class FaithfulnessReport(_Frozen):
    claims: tuple[ClaimVerdict, ...]
    invalid_citations: tuple[int, ...] = ()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def faithfulness(self) -> float | None:
        """Share of claims entailed by the context; ``None`` when there are no claims."""
        if not self.claims:
            return None
        return sum(v.verdict == "entailed" for v in self.claims) / len(self.claims)
