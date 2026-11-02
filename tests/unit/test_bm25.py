import math

import pytest
from conftest import make_chunk

from hybrid_rag.domain.models import ContextualChunk
from hybrid_rag.retrieval.sparse import BM25Index


def _corpus() -> list[ContextualChunk]:
    return [
        make_chunk("apple banana", chunk_id="d1"),
        make_chunk("apple apple cherry", chunk_id="d2"),
        make_chunk("date", chunk_id="d3"),
    ]


def test_scores_match_hand_computed_okapi_bm25() -> None:
    index = BM25Index(k1=1.2, b=0.75)
    index.add(_corpus())

    hits = index.search("apple", k=10)

    # N=3, df(apple)=2, avgdl=(2+3+1)/3=2
    idf = math.log(1 + (3 - 2 + 0.5) / (2 + 0.5))
    d1 = idf * (1 * 2.2) / (1 + 1.2 * (0.25 + 0.75 * 2 / 2))
    d2 = idf * (2 * 2.2) / (2 + 1.2 * (0.25 + 0.75 * 3 / 2))
    assert [h.chunk.chunk_id for h in hits] == ["d2", "d1"]
    assert hits[0].score == pytest.approx(d2)
    assert hits[1].score == pytest.approx(d1)


def test_query_is_analyzed_like_documents() -> None:
    index = BM25Index()
    index.add([make_chunk("Running quickly through fields", chunk_id="r")])
    assert [h.chunk.chunk_id for h in index.search("runs", k=5)] == ["r"]


@pytest.mark.parametrize("query", ["", "the of and", "zebra"])
def test_queries_without_indexed_terms_return_nothing(query: str) -> None:
    index = BM25Index()
    index.add(_corpus())
    assert index.search(query, k=5) == []


def test_remove_is_equivalent_to_never_adding() -> None:
    a, b, c = _corpus()
    removed = BM25Index()
    removed.add([a, b, c])
    removed.remove(["d2"])
    fresh = BM25Index()
    fresh.add([a, c])

    for query in ("apple", "banana cherry", "date apple"):
        got = [(h.chunk.chunk_id, round(h.score, 9)) for h in removed.search(query, k=5)]
        want = [(h.chunk.chunk_id, round(h.score, 9)) for h in fresh.search(query, k=5)]
        assert got == want
    assert len(removed) == 2
    assert "d2" not in removed


def test_removing_unknown_ids_is_a_noop() -> None:
    index = BM25Index()
    index.add(_corpus())
    index.remove(["missing"])
    assert len(index) == 3


def test_re_adding_an_id_replaces_the_previous_version() -> None:
    index = BM25Index()
    index.add([make_chunk("apple", chunk_id="x")])
    index.add([make_chunk("banana", chunk_id="x")])

    assert len(index) == 1
    assert index.search("apple", k=5) == []
    assert [h.chunk.text for h in index.search("banana", k=5)] == ["banana"]


def test_k_limits_results_and_ties_break_by_chunk_id() -> None:
    index = BM25Index()
    index.add([make_chunk("identical words", chunk_id=cid) for cid in ("c", "a", "b")])

    assert [h.chunk.chunk_id for h in index.search("identical", k=2)] == ["a", "b"]


def test_empty_index_and_invalid_params() -> None:
    assert BM25Index().search("anything", k=5) == []
    with pytest.raises(ValueError, match="b"):
        BM25Index(b=1.5)
    with pytest.raises(ValueError, match="k"):
        BM25Index().search("x", k=0)
