"""Claim verification with an NLI model (§5.3).

Premise = a retrieved passage, hypothesis = the claim. A claim is checked against
the passages it cites (strict citation accuracy) or, when it cites none, against
every passage. Passages are scored one by one rather than concatenated: NLI
cross-encoders have a 512-token window that a few chunks would overflow, and a
claim is normally supported by a single passage. The cost is that claims which
need two passages combined are reported as unsupported.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from hybrid_rag.generation.context import ContextPassage
from hybrid_rag.guardrails.models import Claim, ClaimVerdict, NLIScores


class NLIModel(Protocol):
    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[NLIScores]:
        """Score ``(premise, hypothesis)`` pairs."""
        ...


class ClaimVerifier:
    def __init__(self, nli: NLIModel, *, entailment_threshold: float = 0.5) -> None:
        if not 0 < entailment_threshold <= 1:
            raise ValueError("entailment_threshold must be in (0, 1]")
        self._nli = nli
        self._threshold = entailment_threshold

    def verify(
        self, claims: Sequence[Claim], passages: Sequence[ContextPassage]
    ) -> list[ClaimVerdict]:
        if not claims:
            return []
        by_number = {p.n: p for p in passages}
        premises = [
            [by_number[n] for n in claim.citations if n in by_number] or list(passages)
            for claim in claims
        ]
        pairs = [
            (passage.chunk.text, claim.text)
            for claim, group in zip(claims, premises, strict=True)
            for passage in group
        ]
        scores = self._nli.predict(pairs)  # one batch for the whole answer

        verdicts: list[ClaimVerdict] = []
        cursor = 0
        for claim, group in zip(claims, premises, strict=True):
            own = scores[cursor : cursor + len(group)]
            cursor += len(group)
            verdicts.append(self._decide(claim, group, own))
        return verdicts

    def _decide(
        self, claim: Claim, group: Sequence[ContextPassage], own: Sequence[NLIScores]
    ) -> ClaimVerdict:
        if not own:
            return ClaimVerdict(claim=claim, verdict="neutral", entailment=0.0, evidence=None)
        best = max(range(len(own)), key=lambda i: own[i].entailment)
        entailment = own[best].entailment
        if entailment >= self._threshold:
            return ClaimVerdict(
                claim=claim, verdict="entailed", entailment=entailment, evidence=group[best].n
            )
        contradicted = max(s.contradiction for s in own) >= self._threshold
        return ClaimVerdict(
            claim=claim,
            verdict="contradicted" if contradicted else "neutral",
            entailment=entailment,
            evidence=None,
        )
