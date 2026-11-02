from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from hybrid_rag.domain.models import ContextualChunk, ScoredChunk
from hybrid_rag.generation.context import ContextPassage
from hybrid_rag.guardrails.models import Claim, ClaimVerdict

FIXTURES = Path(__file__).parent / "fixtures"
_WORD = re.compile(r"[a-z]+")

CATS = [
    "Cats purr when they feel content and relaxed.",
    "Domestic cats sleep for most hours of the day.",
    "Many cats enjoy chasing small toys and string.",
]
ROCKETS = [
    "Rockets burn liquid propellant to generate thrust.",
    "Orbital rockets need enormous thrust to escape gravity.",
    "Reusable rockets land vertically after launch.",
]


class FakeEmbedder:
    """Deterministic hashed bag-of-words embedder.

    Texts sharing vocabulary land close together, which is all the semantic
    splitter needs to find topic boundaries — without downloading a model.
    """

    dimension = 256

    def __init__(self) -> None:
        self.calls = 0

    def embed_query(self, text: str) -> NDArray[np.float32]:
        vector: NDArray[np.float32] = self.embed([text])[0]
        return vector

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        self.calls += 1
        out = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in _WORD.findall(text.lower()):
                if len(word) <= 3:
                    continue
                bucket = int.from_bytes(hashlib.md5(word.encode()).digest()[:4], "little")
                out[row, bucket % self.dimension] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return (out / np.where(norms == 0, 1.0, norms)).astype(np.float32)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


def make_chunk(
    text: str,
    *,
    chunk_id: str | None = None,
    doc_id: str = "doc",
    path: tuple[str, ...] = (),
    position: int = 0,
) -> ContextualChunk:
    """Chunk whose ``embed_text`` is exactly ``text`` (no prefix), so BM25
    document lengths in tests are what the test author sees."""
    return ContextualChunk(
        chunk_id=chunk_id or hashlib.sha256(f"{doc_id}|{text}".encode()).hexdigest(),
        doc_id=doc_id,
        text=text,
        section_path=path,
        token_count=len(text.split()),
        position=position,
        context_prefix="",
        embed_text=text,
    )


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


class FakeLLM:
    """Scripted LLM: returns ``reply`` (or raises ``error``) and records every call."""

    def __init__(self, reply: str = "", error: Exception | None = None) -> None:
        self.reply = reply
        self.error = error
        self.calls: list[tuple[str, str, int]] = []

    def complete(self, *, system: str, prompt: str, max_tokens: int) -> str:
        self.calls.append((system, prompt, max_tokens))
        if self.error is not None:
            raise self.error
        return self.reply


class SubsetVerifier:
    """Deterministic verifier: a claim is entailed iff all its words occur in one of
    its premises (the cited passages, or every passage when it cites none)."""

    def __init__(self) -> None:
        self.calls = 0

    def verify(
        self, claims: Sequence[Claim], passages: Sequence[ContextPassage]
    ) -> list[ClaimVerdict]:
        self.calls += 1
        by_n = {p.n: p for p in passages}
        verdicts: list[ClaimVerdict] = []
        for claim in claims:
            premises = [by_n[n] for n in claim.citations if n in by_n] or list(passages)
            words = _WORD.findall(claim.text.lower())
            support = next(
                (
                    p
                    for p in premises
                    if words and set(words) <= set(_WORD.findall(p.chunk.text.lower()))
                ),
                None,
            )
            verdicts.append(
                ClaimVerdict(
                    claim=claim,
                    verdict="entailed" if support else "neutral",
                    entailment=1.0 if support else 0.0,
                    evidence=support.n if support else None,
                )
            )
        return verdicts


def ranked(*chunk_ids: str, scale: float = 1.0) -> list[ScoredChunk]:
    """A ranked list in the given order, with descending (arbitrarily scaled) scores."""
    return [
        ScoredChunk(
            chunk=make_chunk(f"text {cid}", chunk_id=cid), score=scale * (len(chunk_ids) - n)
        )
        for n, cid in enumerate(chunk_ids)
    ]


def build_pdf(path: Path) -> Path:
    """Two-page PDF with a title, numbered sections, and a repeated running footer."""
    import pymupdf

    doc = pymupdf.open()
    pages = [
        [
            (24, "Retrieval Systems Handbook"),
            (16, "1. Dense Retrieval"),
            (11, "Dense retrievers encode queries and passages into vectors."),
            (11, "Similarity is computed with cosine distance over embeddings."),
        ],
        [
            (16, "2. Sparse Retrieval"),
            (11, "BM25 scores documents by term frequency and inverse document frequency."),
        ],
    ]
    for number, lines in enumerate(pages, start=1):
        page = doc.new_page()
        y = 72.0
        for size, text in lines:
            page.insert_text((72, y), text, fontsize=size)
            y += size * 3
        page.insert_text((72, 810), f"Confidential draft - page {number}", fontsize=8)
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    return build_pdf(tmp_path / "handbook.pdf")
