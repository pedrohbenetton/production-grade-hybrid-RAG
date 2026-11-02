"""Embedding-based semantic splitting for sections that exceed the token budget."""

from __future__ import annotations

import re

import numpy as np

from hybrid_rag.ports import Embedder, TokenCounter

# Split after . ! ? only when the next token looks like a sentence start; keeps
# decimals ("3.5") and lowercase abbreviations ("e.g. this") together.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text.strip()) if s.strip()]


class SemanticSplitter:
    """Cut where adjacent sentence windows are semantically distant.

    1. For each gap between sentences i and i+1, embed the ``window + 1``
       sentences ending at i and the ``window + 1`` sentences starting at i+1
       (TextTiling-style). Gap-centred windows keep the distance peak *on* the
       topic shift; symmetric per-sentence windows smear it one sentence early.
    2. Cosine distance between the left and right block of each gap.
    3. Break where distance > ``breakpoint_percentile`` of this text's distances
       — the threshold adapts to each document instead of being a global constant.
    4. Groups still over ``max_tokens`` fall back to greedy sentence packing, and
       single over-long sentences to word packing, so the budget is a hard cap.
    """

    def __init__(
        self,
        embedder: Embedder,
        counter: TokenCounter,
        *,
        max_tokens: int,
        breakpoint_percentile: float = 90.0,
        window: int = 1,
    ) -> None:
        self._embedder = embedder
        self._counter = counter
        self._max = max_tokens
        self._percentile = breakpoint_percentile
        self._window = window

    def split(self, text: str) -> list[str]:
        sentences = split_sentences(text)
        if len(sentences) <= 1:
            return self._fit([text.strip()])
        return [piece for group in self._semantic_groups(sentences) for piece in self._fit(group)]

    def _semantic_groups(self, sentences: list[str]) -> list[list[str]]:
        w = self._window
        gaps = range(len(sentences) - 1)
        left = [" ".join(sentences[max(0, i - w) : i + 1]) for i in gaps]
        right = [" ".join(sentences[i + 1 : i + w + 2]) for i in gaps]
        vectors = np.asarray(self._embedder.embed(left + right), dtype=np.float64)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.where(norms == 0, 1.0, norms)
        n = len(left)
        distances = 1.0 - np.sum(vectors[:n] * vectors[n:], axis=1)
        threshold = float(np.percentile(distances, self._percentile))

        groups: list[list[str]] = []
        current = [sentences[0]]
        for i, distance in enumerate(distances):
            if distance > threshold:
                groups.append(current)
                current = []
            current.append(sentences[i + 1])
        groups.append(current)
        return groups

    def _fit(self, sentences: list[str]) -> list[str]:
        joined = " ".join(sentences)
        if self._counter.count(joined) <= self._max:
            return [joined]
        pieces: list[str] = []
        buffer: list[str] = []
        for sentence in sentences:
            if self._counter.count(sentence) > self._max:
                if buffer:
                    pieces.append(" ".join(buffer))
                    buffer = []
                pieces.extend(self._pack(sentence.split()))
            elif self._counter.count(" ".join([*buffer, sentence])) <= self._max:
                buffer.append(sentence)
            else:
                pieces.append(" ".join(buffer))
                buffer = [sentence]
        if buffer:
            pieces.append(" ".join(buffer))
        return pieces

    def _pack(self, words: list[str]) -> list[str]:
        pieces: list[str] = []
        buffer: list[str] = []
        for word in words:
            if buffer and self._counter.count(" ".join([*buffer, word])) > self._max:
                pieces.append(" ".join(buffer))
                buffer = []
            buffer.append(word)
        if buffer:
            pieces.append(" ".join(buffer))
        return pieces
