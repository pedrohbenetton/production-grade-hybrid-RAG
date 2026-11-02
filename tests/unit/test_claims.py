import pytest

from hybrid_rag.guardrails.claims import ABSTAIN_PHRASE, extract_claims, is_abstention


def pairs(answer: str, n: int = 3) -> list[tuple[str, tuple[int, ...]]]:
    claims, _ = extract_claims(answer, n)
    return [(c.text, c.citations) for c in claims]


def test_each_sentence_becomes_a_claim_with_its_citations() -> None:
    claims, invalid = extract_claims(
        "RRF fuses rankings [1]. It needs no score calibration [1][2].", n_passages=2
    )

    assert [(c.text, c.citations) for c in claims] == [
        ("RRF fuses rankings.", (1,)),
        ("It needs no score calibration.", (1, 2)),
    ]
    assert claims[0].sentence == "RRF fuses rankings [1]."
    assert invalid == ()


def test_comma_separated_and_repeated_markers_are_deduplicated() -> None:
    assert pairs("Both agree [1, 2][2].") == [("Both agree.", (1, 2))]


def test_a_trailing_citation_only_sentence_belongs_to_the_previous_claim() -> None:
    [claim], _ = extract_claims("Dense search embeds text. [1]", n_passages=1)

    assert claim.text == "Dense search embeds text."
    assert claim.citations == (1,)
    assert claim.sentence == "Dense search embeds text. [1]"


def test_citation_only_text_without_a_previous_claim_is_dropped() -> None:
    assert extract_claims("[1]", n_passages=1) == ([], ())


def test_out_of_range_citations_are_reported_and_removed_from_the_claim() -> None:
    claims, invalid = extract_claims("Claim [3]. Other [1][0].", n_passages=2)

    assert invalid == (0, 3)
    assert [(c.text, c.citations, c.sentence) for c in claims] == [
        ("Claim.", (), "Claim."),
        ("Other.", (1,), "Other [1]."),
    ]


def test_list_markers_and_line_breaks_split_claims() -> None:
    answer = "- First fact [1]\n* Second fact [2]\n3. Third fact [3]"
    assert pairs(answer) == [("First fact", (1,)), ("Second fact", (2,)), ("Third fact", (3,))]


@pytest.mark.parametrize("answer", ["", "  \n\n ", "...", "(!!!)"])
def test_text_without_words_yields_no_claims(answer: str) -> None:
    assert extract_claims(answer, 2) == ([], ())


@pytest.mark.parametrize(
    "answer",
    [
        ABSTAIN_PHRASE,
        ABSTAIN_PHRASE.lower(),
        ABSTAIN_PHRASE.rstrip("."),
        f"  {ABSTAIN_PHRASE}\n",
    ],
)
def test_exact_abstention_phrase_is_detected(answer: str) -> None:
    assert is_abstention(answer)


@pytest.mark.parametrize(
    "answer", ["", "RRF fuses rankings [1].", f"{ABSTAIN_PHRASE} But RRF fuses rankings [1]."]
)
def test_anything_else_is_not_an_abstention(answer: str) -> None:
    assert not is_abstention(answer)
