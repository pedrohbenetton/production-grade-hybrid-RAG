from conftest import CATS, ROCKETS, FakeEmbedder

from hybrid_rag.ingestion.semantic import SemanticSplitter, split_sentences
from hybrid_rag.ingestion.tokens import RegexTokenCounter

COUNTER = RegexTokenCounter()


def test_token_counter_counts_words_and_punctuation() -> None:
    assert COUNTER.count("Hello, world!") == 4
    assert COUNTER.count("") == 0


def test_split_sentences_on_terminal_punctuation() -> None:
    assert split_sentences("First one. Second one! Third? Yes.") == [
        "First one.",
        "Second one!",
        "Third?",
        "Yes.",
    ]


def test_split_sentences_keeps_decimals_and_lowercase_continuations() -> None:
    assert split_sentences("Version 3.5 is out. e.g. this stays. Done.") == [
        "Version 3.5 is out. e.g. this stays.",
        "Done.",
    ]


def test_short_text_is_returned_unchanged(fake_embedder: FakeEmbedder) -> None:
    splitter = SemanticSplitter(fake_embedder, COUNTER, max_tokens=100)
    assert splitter.split("Just one sentence.") == ["Just one sentence."]


def test_splits_at_topic_shift(fake_embedder: FakeEmbedder) -> None:
    splitter = SemanticSplitter(
        fake_embedder, COUNTER, max_tokens=500, breakpoint_percentile=80, window=0
    )
    assert splitter.split(" ".join(CATS + ROCKETS)) == [" ".join(CATS), " ".join(ROCKETS)]


def test_groups_over_budget_fall_back_to_sentence_packing(fake_embedder: FakeEmbedder) -> None:
    text = " ".join(CATS + ROCKETS)
    splitter = SemanticSplitter(fake_embedder, COUNTER, max_tokens=12)
    pieces = splitter.split(text)

    assert all(COUNTER.count(p) <= 12 for p in pieces)
    assert " ".join(pieces) == text


def test_single_oversized_sentence_is_hard_split_on_words(fake_embedder: FakeEmbedder) -> None:
    text = " ".join(f"word{i}" for i in range(50))
    splitter = SemanticSplitter(fake_embedder, COUNTER, max_tokens=10)
    pieces = splitter.split(text)

    assert all(COUNTER.count(p) <= 10 for p in pieces)
    assert " ".join(pieces) == text
