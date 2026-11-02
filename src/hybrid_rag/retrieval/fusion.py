"""Reciprocal Rank Fusion (Cormack, Clarke & Büttcher, SIGIR 2009) — §4.3.

    RRF(d) = Σ_r  w_r / (k + rank_r(d))

Only ranks enter the formula, so retrievers with incomparable score scales
(cosine similarity vs. BM25) fuse without any calibration. ``k`` damps the
advantage of top ranks; 60 is the value from the original paper.
"""

from __future__ import annotations

from collections.abc import Sequence

from hybrid_rag.domain.models import ContextualChunk, ScoredChunk


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[ScoredChunk]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
    limit: int | None = None,
) -> list[ScoredChunk]:
    """Fuse ranked lists; returned ``score`` is the RRF score. Ties break by ``chunk_id``."""
    if k < 0:
        raise ValueError("k must be >= 0")
    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1")
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    elif len(weights) != len(ranked_lists) or any(w < 0 for w in weights):
        raise ValueError("weights must be non-negative, exactly one per ranked list")

    scores: dict[str, float] = {}
    chunks: dict[str, ContextualChunk] = {}
    for hits, weight in zip(ranked_lists, weights, strict=True):
        seen: set[str] = set()
        for rank, hit in enumerate(hits, start=1):
            chunk_id = hit.chunk.chunk_id
            if chunk_id in seen:  # a list votes once per chunk, at its best rank
                continue
            seen.add(chunk_id)
            chunks.setdefault(chunk_id, hit.chunk)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (k + rank)

    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [ScoredChunk(chunk=chunks[cid], score=score) for cid, score in ordered[:limit]]
