import math
from collections.abc import Callable, Mapping, Sequence

import pytest

from hybrid_rag.evaluation.metrics import (
    METRICS,
    evaluate_run,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

RELEVANT = {"a": 1, "b": 1, "c": 1}
RANKING = ["a", "x", "b", "y"]


def test_precision_divides_by_k_even_when_fewer_results_were_returned() -> None:
    assert precision_at_k(RANKING, RELEVANT, 1) == 1.0
    assert precision_at_k(RANKING, RELEVANT, 2) == 0.5
    assert precision_at_k(RANKING, RELEVANT, 4) == 0.5
    assert precision_at_k(RANKING, RELEVANT, 10) == pytest.approx(0.2)


def test_recall_divides_by_the_number_of_relevant_documents() -> None:
    assert recall_at_k(RANKING, RELEVANT, 4) == pytest.approx(2 / 3)
    assert recall_at_k(RANKING, RELEVANT, 1) == pytest.approx(1 / 3)


def test_reciprocal_rank_uses_the_first_relevant_hit_within_the_cutoff() -> None:
    assert reciprocal_rank(["a", "x"], RELEVANT, 10) == 1.0
    assert reciprocal_rank(["x", "b"], RELEVANT, 10) == 0.5
    assert reciprocal_rank(["x", "y"], RELEVANT, 10) == 0.0
    late = [*"xxxxxxxxxx", "a"]
    assert reciprocal_rank(late, RELEVANT, 10) == 0.0
    assert reciprocal_rank(late, RELEVANT, 11) == pytest.approx(1 / 11)


def test_ndcg_matches_the_hand_computed_binary_case() -> None:
    dcg = 1 / math.log2(2) + 1 / math.log2(4)  # hits at ranks 1 and 3
    ideal = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(RANKING, RELEVANT, 4) == pytest.approx(dcg / ideal)


def test_ndcg_uses_graded_gains() -> None:
    relevant = {"a": 2, "b": 1}
    dcg = 1 / math.log2(2) + 2 / math.log2(3)  # lesser document first
    ideal = 2 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(["b", "a"], relevant, 2) == pytest.approx(dcg / ideal)
    assert ndcg_at_k(["a", "b"], relevant, 2) == pytest.approx(1.0)


def test_a_perfect_ranking_scores_one_everywhere() -> None:
    assert ndcg_at_k(["a", "b", "c"], RELEVANT, 10) == pytest.approx(1.0)
    assert reciprocal_rank(["a", "b", "c"], RELEVANT, 10) == 1.0
    assert recall_at_k(["a", "b", "c"], RELEVANT, 10) == 1.0


def test_zero_scored_qrels_are_not_relevant() -> None:
    assert precision_at_k(["a"], {"a": 0}, 1) == 0.0
    assert ndcg_at_k(["a"], {"a": 0, "b": 1}, 1) == 0.0


@pytest.mark.parametrize("fn", [precision_at_k, recall_at_k, ndcg_at_k])
def test_cutoff_must_be_positive(
    fn: Callable[[Sequence[str], Mapping[str, int], int], float],
) -> None:
    with pytest.raises(ValueError, match="k"):
        fn(["a"], RELEVANT, 0)


def test_evaluate_run_aligns_queries_and_skips_those_without_relevant_documents() -> None:
    qrels = {"q1": {"a": 1}, "q2": {"b": 1}, "q3": {"zzz": 0}}
    run = {"q1": ["a", "x"], "q2": ["x", "b"]}  # q3 has nothing to find; q4 is not in qrels

    result = evaluate_run({**run, "q4": ["a"]}, qrels)

    assert result.query_ids == ["q1", "q2"]
    assert set(result.values) == set(METRICS)
    assert list(result.values["MRR@10"]) == [1.0, 0.5]
    assert list(result.values["P@1"]) == [1.0, 0.0]
    assert list(result.values["R@10"]) == [1.0, 1.0]


def test_a_query_missing_from_the_run_counts_as_an_empty_ranking() -> None:
    result = evaluate_run({}, {"q1": {"a": 1}})
    assert all(list(v) == [0.0] for v in result.values.values())
