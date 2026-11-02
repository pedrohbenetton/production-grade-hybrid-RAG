import pytest

from hybrid_rag.retrieval.analyzer import Analyzer

analyze = Analyzer()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Running Dogs", ["run", "dog"]),
        ("the cat and the hat", ["cat", "hat"]),
        ("BM25 with k1", ["bm25", "k1"]),
        ("ﬁle", ["file"]),  # ligature folded by NFKC
        ("ＡＢＣ", ["abc"]),  # full-width folded by NFKC
        ("", []),
        ("... !!!", []),
    ],
)
def test_analyzer_pipeline(text: str, expected: list[str]) -> None:
    assert analyze(text) == expected


def test_query_and_document_forms_meet_on_the_same_stem() -> None:
    assert analyze("runs") == analyze("running") == ["run"]
