from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

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

    dim = 256

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in _WORD.findall(text.lower()):
                if len(word) <= 3:
                    continue
                bucket = int.from_bytes(hashlib.md5(word.encode()).digest()[:4], "little")
                out[row, bucket % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return (out / np.where(norms == 0, 1.0, norms)).astype(np.float32)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


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
