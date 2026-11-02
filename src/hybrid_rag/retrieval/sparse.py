"""Incremental Okapi BM25 over an in-memory inverted index (§4.2).

score(q, d) = Σ_{t ∈ q} idf(t) · tf(t,d)·(k1+1) / (tf(t,d) + k1·(1 − b + b·|d|/avgdl))
idf(t)      = ln(1 + (N − df(t) + 0.5) / (df(t) + 0.5))        (Lucene variant, ≥ 0)

Statistics (N, df, avgdl) are maintained on add/remove and read at query time,
so removing a chunk is exactly equivalent to never having indexed it.
"""

from __future__ import annotations

import heapq
import math
from collections import Counter
from collections.abc import Iterable

from hybrid_rag.domain.models import ContextualChunk, ScoredChunk
from hybrid_rag.retrieval.analyzer import Analyzer


class BM25Index:
    def __init__(
        self, analyzer: Analyzer | None = None, *, k1: float = 1.2, b: float = 0.75
    ) -> None:
        if k1 < 0:
            raise ValueError("k1 must be >= 0")
        if not 0 <= b <= 1:
            raise ValueError("b must be in [0, 1]")
        self._analyzer = analyzer or Analyzer()
        self._k1 = k1
        self._b = b
        self._postings: dict[str, dict[str, int]] = {}  # term -> {chunk_id: tf}
        self._terms: dict[str, tuple[str, ...]] = {}  # chunk_id -> its unique terms
        self._lengths: dict[str, int] = {}
        self._chunks: dict[str, ContextualChunk] = {}
        self._total_length = 0

    def __len__(self) -> int:
        return len(self._chunks)

    def __contains__(self, chunk_id: object) -> bool:
        return chunk_id in self._chunks

    def add(self, chunks: Iterable[ContextualChunk]) -> None:
        """Index ``embed_text``; re-adding an existing ``chunk_id`` replaces it."""
        for chunk in chunks:
            self._remove_one(chunk.chunk_id)
            terms = self._analyzer(chunk.embed_text)
            frequencies = Counter(terms)
            for term, tf in frequencies.items():
                self._postings.setdefault(term, {})[chunk.chunk_id] = tf
            self._terms[chunk.chunk_id] = tuple(frequencies)
            self._lengths[chunk.chunk_id] = len(terms)
            self._chunks[chunk.chunk_id] = chunk
            self._total_length += len(terms)

    def remove(self, chunk_ids: Iterable[str]) -> None:
        for chunk_id in chunk_ids:
            self._remove_one(chunk_id)

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        if k < 1:
            raise ValueError("k must be >= 1")
        n = len(self._chunks)
        if n == 0:
            return []
        avgdl = self._total_length / n or 1.0
        scores: dict[str, float] = {}
        # Sorted unique terms: deterministic float summation order.
        for term in sorted(set(self._analyzer(query))):
            postings = self._postings.get(term)
            if not postings:
                continue
            df = len(postings)
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for chunk_id, tf in postings.items():
                length_norm = 1 - self._b + self._b * self._lengths[chunk_id] / avgdl
                gain = idf * tf * (self._k1 + 1) / (tf + self._k1 * length_norm)
                scores[chunk_id] = scores.get(chunk_id, 0.0) + gain
        top = heapq.nsmallest(k, scores.items(), key=lambda item: (-item[1], item[0]))
        return [ScoredChunk(chunk=self._chunks[cid], score=score) for cid, score in top]

    def _remove_one(self, chunk_id: str) -> None:
        terms = self._terms.pop(chunk_id, None)
        if terms is None:
            return
        for term in terms:
            postings = self._postings[term]
            del postings[chunk_id]
            if not postings:
                del self._postings[term]
        self._total_length -= self._lengths.pop(chunk_id)
        del self._chunks[chunk_id]
