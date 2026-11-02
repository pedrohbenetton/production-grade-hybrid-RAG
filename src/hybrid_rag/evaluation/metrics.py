"""Standard IR metrics over document rankings with graded relevance (qrels).

``relevant`` maps document id → relevance grade; grade 0 means "judged, not relevant".
nDCG uses linear gains (the trec_eval / BEIR convention); SciFact judgments are binary.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

METRICS = ("P@1", "P@5", "P@10", "R@10", "MRR@10", "nDCG@10")


def _check_k(k: int) -> None:
    if k < 1:
        raise ValueError("k must be >= 1")


def _is_relevant(relevant: Mapping[str, int], doc_id: str) -> bool:
    return relevant.get(doc_id, 0) > 0


def precision_at_k(ranking: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    """Hits in the top k divided by k (not by the number returned)."""
    _check_k(k)
    return sum(_is_relevant(relevant, d) for d in ranking[:k]) / k


def recall_at_k(ranking: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    _check_k(k)
    total = sum(1 for grade in relevant.values() if grade > 0)
    if total == 0:
        return 0.0
    return sum(_is_relevant(relevant, d) for d in ranking[:k]) / total


def reciprocal_rank(ranking: Sequence[str], relevant: Mapping[str, int], k: int = 10) -> float:
    _check_k(k)
    for rank, doc_id in enumerate(ranking[:k], start=1):
        if _is_relevant(relevant, doc_id):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranking: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    _check_k(k)

    def dcg(gains: Sequence[int]) -> float:
        return sum(g / math.log2(rank + 1) for rank, g in enumerate(gains, start=1))

    ideal = dcg(sorted((g for g in relevant.values() if g > 0), reverse=True)[:k])
    if ideal == 0:
        return 0.0
    return dcg([max(relevant.get(d, 0), 0) for d in ranking[:k]]) / ideal


@dataclass(frozen=True)
class PerQueryMetrics:
    query_ids: list[str]
    values: dict[str, NDArray[np.float64]]  # metric name → one value per query, aligned


def evaluate_run(
    run: Mapping[str, Sequence[str]], qrels: Mapping[str, Mapping[str, int]]
) -> PerQueryMetrics:
    """Score every query that has at least one relevant document, in ``qrels`` order.

    A query absent from ``run`` scores as an empty ranking: failing to answer is a miss,
    not a reason to drop the query from the average.
    """
    query_ids = [q for q, rel in qrels.items() if any(g > 0 for g in rel.values())]
    rows: dict[str, list[float]] = {name: [] for name in METRICS}
    for qid in query_ids:
        ranking, rel = run.get(qid, ()), qrels[qid]
        rows["P@1"].append(precision_at_k(ranking, rel, 1))
        rows["P@5"].append(precision_at_k(ranking, rel, 5))
        rows["P@10"].append(precision_at_k(ranking, rel, 10))
        rows["R@10"].append(recall_at_k(ranking, rel, 10))
        rows["MRR@10"].append(reciprocal_rank(ranking, rel, 10))
        rows["nDCG@10"].append(ndcg_at_k(ranking, rel, 10))
    return PerQueryMetrics(
        query_ids=query_ids,
        values={name: np.asarray(v, dtype=np.float64) for name, v in rows.items()},
    )
