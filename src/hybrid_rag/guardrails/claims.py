"""Draft answer → claims with citations (§5.3).

Claims are sentences: the generator is prompted to write one claim per sentence,
which makes sentence-level verification a good approximation of atomic claims
without a second LLM call.
"""

from __future__ import annotations

import re

from hybrid_rag.guardrails.models import Claim
from hybrid_rag.ingestion.semantic import split_sentences

ABSTAIN_PHRASE = "I cannot answer this from the provided context."

_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_LIST_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:!?])")
_HAS_WORD = re.compile(r"\w")


def is_abstention(answer: str) -> bool:
    """True only for the exact refusal phrase; refusal plus extra text is verified as usual."""

    def normalize(text: str) -> str:
        return " ".join(text.casefold().split()).rstrip(".!")

    return normalize(answer) == normalize(ABSTAIN_PHRASE)


def _numbers(marker: re.Match[str]) -> list[int]:
    return [int(n) for n in marker.group(1).split(",")]


def _tidy(text: str) -> str:
    return " ".join(_SPACE_BEFORE_PUNCT.sub(r"\1", text).split())


def extract_claims(answer: str, n_passages: int) -> tuple[list[Claim], tuple[int, ...]]:
    """Split ``answer`` into claims; returns them plus the out-of-range citation numbers."""

    def is_valid(n: int) -> bool:
        return 1 <= n <= n_passages

    def keep_valid(marker: re.Match[str]) -> str:
        kept = [n for n in _numbers(marker) if is_valid(n)]
        return f"[{', '.join(map(str, kept))}]" if kept else ""

    claims: list[Claim] = []
    invalid: set[int] = set()
    for line in answer.splitlines():
        for sentence in split_sentences(_LIST_MARKER.sub("", line)):
            numbers = [n for marker in _CITATION.finditer(sentence) for n in _numbers(marker)]
            invalid.update(n for n in numbers if not is_valid(n))
            valid = tuple(dict.fromkeys(n for n in numbers if is_valid(n)))
            text = _tidy(_CITATION.sub("", sentence))
            shown = _tidy(_CITATION.sub(keep_valid, sentence))
            if _HAS_WORD.search(text):
                claims.append(Claim(text=text, sentence=shown, citations=valid))
            elif valid and claims:  # a lone "[1]" after the period cites the previous claim
                previous = claims[-1]
                claims[-1] = Claim(
                    text=previous.text,
                    sentence=f"{previous.sentence} {shown}",
                    citations=tuple(dict.fromkeys((*previous.citations, *valid))),
                )
    return claims, tuple(sorted(invalid))
