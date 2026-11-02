from pathlib import Path

from conftest import CATS, ROCKETS, FakeEmbedder

from hybrid_rag.config import ChunkingSettings
from hybrid_rag.domain.models import Chunk, DocumentTree, RawDocument
from hybrid_rag.ingestion.chunking import StructuralSemanticChunker
from hybrid_rag.ingestion.parsers import MarkdownParser
from hybrid_rag.ingestion.semantic import split_sentences
from hybrid_rag.ingestion.tokens import RegexTokenCounter
from hybrid_rag.ingestion.tree_builder import TreeBuilder

COUNTER = RegexTokenCounter()


def chunk(
    tree: DocumentTree, embedder: FakeEmbedder, *, min_tokens: int = 5, max_tokens: int = 40
) -> list[Chunk]:
    settings = ChunkingSettings(
        min_tokens=min_tokens, max_tokens=max_tokens, breakpoint_percentile=80
    )
    return StructuralSemanticChunker(embedder, settings, COUNTER).chunk(tree)


def tree_of(*items: tuple[str, int] | tuple[str, str]) -> DocumentTree:
    """("Heading", level) adds a heading; ("kind", "text") adds a block."""
    b = TreeBuilder(fallback_title="Doc")
    for first, second in items:
        if isinstance(second, int):
            b.heading(first, second)
        else:
            b.block(first, second)  # type: ignore[arg-type]
    return b.build(source_uri="file:///doc.md")


def test_each_section_becomes_its_own_chunk(fake_embedder: FakeEmbedder) -> None:
    tree = tree_of(
        ("A", 2),
        ("paragraph", "Alpha section talks about apples and orchards."),
        ("B", 2),
        ("paragraph", "Beta section talks about bridges and rivers."),
    )
    chunks = chunk(tree, fake_embedder)

    assert [c.section_path for c in chunks] == [("A",), ("B",)]
    assert [c.position for c in chunks] == [0, 1]


def test_small_sibling_sections_merge_with_heading_prefixes(fake_embedder: FakeEmbedder) -> None:
    tree = tree_of(("A", 2), ("paragraph", "Tiny one."), ("B", 2), ("paragraph", "Tiny two."))
    chunks = chunk(tree, fake_embedder, min_tokens=20)

    assert len(chunks) == 1
    assert chunks[0].section_path == ()
    assert chunks[0].text == "A\nTiny one.\n\nB\nTiny two."


def test_tiny_sections_do_not_merge_across_unrelated_branches(
    fake_embedder: FakeEmbedder,
) -> None:
    tree = tree_of(
        ("A", 2),
        ("A1", 3),
        ("paragraph", "Tiny one."),
        ("B", 2),
        ("B1", 3),
        ("paragraph", "Tiny two."),
    )
    chunks = chunk(tree, fake_embedder, min_tokens=20)

    assert [c.section_path for c in chunks] == [("A", "A1"), ("B", "B1")]


def test_code_block_is_never_split_even_if_oversized(fake_embedder: FakeEmbedder) -> None:
    code = "\n".join(f"x{i} = compute({i})" for i in range(20))
    tree = tree_of(("Code", 2), ("code", code))
    chunks = chunk(tree, fake_embedder, max_tokens=40)

    assert [c.text for c in chunks] == [code]
    assert chunks[0].token_count > 40


def test_long_paragraph_is_split_at_topic_boundary(fake_embedder: FakeEmbedder) -> None:
    tree = tree_of(("Mixed", 2), ("paragraph", " ".join(CATS + ROCKETS)))
    chunks = chunk(tree, fake_embedder, max_tokens=40)

    assert [c.text for c in chunks] == [" ".join(CATS), " ".join(ROCKETS)]
    assert all(c.section_path == ("Mixed",) for c in chunks)


def test_identical_paragraphs_get_distinct_ids(fake_embedder: FakeEmbedder) -> None:
    long = "Repeated paragraph with enough words to stand alone as a chunk here."
    code = "print('separator')"
    tree = tree_of(("S", 2), ("paragraph", long), ("code", code), ("paragraph", long))
    chunks = chunk(tree, fake_embedder, max_tokens=14)

    texts = [c.text for c in chunks]
    assert texts.count(long) == 2
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_invariants_on_real_markdown(fixtures_dir: Path, fake_embedder: FakeEmbedder) -> None:
    path = fixtures_dir / "sample.md"
    raw = RawDocument(
        source_uri=path.as_uri(), mime_type="text/markdown", content=path.read_bytes()
    )
    tree = MarkdownParser().parse(raw)
    chunks = chunk(tree, fake_embedder, min_tokens=5, max_tokens=60)

    assert [c.position for c in chunks] == list(range(len(chunks)))
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert all(c.doc_id == tree.doc_id for c in chunks)
    for c in chunks:
        assert c.token_count == COUNTER.count(c.text)
        assert c.token_count <= 60
    for _, block in tree.iter_blocks():
        for sentence in split_sentences(block.text):
            assert sum(sentence in c.text for c in chunks) == 1, sentence

    again = chunk(tree, fake_embedder, min_tokens=5, max_tokens=60)
    assert [c.chunk_id for c in again] == [c.chunk_id for c in chunks]
