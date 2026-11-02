import pytest
from conftest import ranked

from hybrid_rag.domain.models import ScoredChunk
from hybrid_rag.retrieval.fusion import reciprocal_rank_fusion


def ids(hits: list[ScoredChunk]) -> list[str]:
    return [h.chunk.chunk_id for h in hits]


def test_scores_match_hand_computed_rrf() -> None:
    fused = reciprocal_rank_fusion([ranked("x", "y", "z"), ranked("y", "w")], k=60)

    assert ids(fused) == ["y", "x", "w", "z"]
    assert {h.chunk.chunk_id: h.score for h in fused} == pytest.approx(
        {"y": 1 / 62 + 1 / 61, "x": 1 / 61, "w": 1 / 62, "z": 1 / 63}
    )


def test_only_ranks_matter_not_score_scales() -> None:
    """The reason RRF is used: cosine and BM25 scores live on incomparable scales."""
    base = reciprocal_rank_fusion([ranked("x", "y", "z"), ranked("y", "w")])
    rescaled = reciprocal_rank_fusion([ranked("x", "y", "z"), ranked("y", "w", scale=1000.0)])

    assert [(h.chunk.chunk_id, h.score) for h in rescaled] == [
        (h.chunk.chunk_id, h.score) for h in base
    ]


def test_ties_break_by_chunk_id() -> None:
    fused = reciprocal_rank_fusion([ranked("b1", "b2"), ranked("a1", "a2")])
    assert ids(fused) == ["a1", "b1", "a2", "b2"]


def test_weights_scale_each_retriever_contribution() -> None:
    fused = reciprocal_rank_fusion([ranked("x", "y", "z"), ranked("y", "w")], weights=[1.0, 0.0])

    assert ids(fused) == ["x", "y", "z", "w"]
    assert fused[-1].score == 0.0


def test_duplicates_within_one_list_count_once_at_best_rank() -> None:
    [hit, _] = reciprocal_rank_fusion([ranked("a", "b", "a")], k=60)
    assert hit.chunk.chunk_id == "a"
    assert hit.score == pytest.approx(1 / 61)


def test_limit_and_empty_inputs() -> None:
    assert len(reciprocal_rank_fusion([ranked("a", "b", "c")], limit=2)) == 2
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"weights": [1.0]}, "weights"),
        ({"weights": [1.0, -1.0]}, "weights"),
        ({"k": -1}, "k"),
        ({"limit": 0}, "limit"),
    ],
)
def test_invalid_arguments(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        reciprocal_rank_fusion([ranked("a"), ranked("b")], **kwargs)  # type: ignore[arg-type]
