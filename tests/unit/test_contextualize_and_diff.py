from hybrid_rag.domain.models import Chunk, DocumentTree
from hybrid_rag.ingestion.contextualize import Contextualizer
from hybrid_rag.ingestion.pipeline import diff_chunks
from hybrid_rag.ingestion.tree_builder import TreeBuilder


def _tree() -> DocumentTree:
    b = TreeBuilder(fallback_title="x")
    b.heading("Guide", 1)
    b.block("paragraph", "intro")
    return b.build(source_uri="file:///guide.md")


def _chunk(chunk_id: str, path: tuple[str, ...] = ("Setup", "Linux")) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="d",
        text="Install with apt.",
        section_path=path,
        token_count=4,
        position=0,
    )


class FakeGenerator:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def generate(self, document: DocumentTree, chunk: Chunk) -> str:
        self.calls.append((document.title, chunk.chunk_id))
        return self.reply


def test_breadcrumb_prefix_and_embed_text() -> None:
    [ctx] = Contextualizer().contextualize(_tree(), [_chunk("c1")])

    assert ctx.context_prefix == "Guide > Setup > Linux"
    assert ctx.embed_text == "Guide > Setup > Linux\n\nInstall with apt."
    assert ctx.text == "Install with apt."
    assert ctx.chunk_id == "c1"


def test_root_chunk_breadcrumb_is_the_title() -> None:
    [ctx] = Contextualizer().contextualize(_tree(), [_chunk("c1", path=())])
    assert ctx.context_prefix == "Guide"


def test_llm_context_is_appended_after_breadcrumb() -> None:
    gen = FakeGenerator("  Explains Linux installation.  ")
    [ctx] = Contextualizer(gen).contextualize(_tree(), [_chunk("c1")])

    assert ctx.context_prefix == "Guide > Setup > Linux\nExplains Linux installation."
    assert gen.calls == [("Guide", "c1")]


def test_blank_llm_context_is_ignored() -> None:
    [ctx] = Contextualizer(FakeGenerator("   ")).contextualize(_tree(), [_chunk("c1")])
    assert ctx.context_prefix == "Guide > Setup > Linux"


def test_diff_chunks_computes_add_keep_remove() -> None:
    new = Contextualizer().contextualize(_tree(), [_chunk("b"), _chunk("c")])
    diff = diff_chunks(existing_ids={"a", "b"}, new=new)

    assert [c.chunk_id for c in diff.to_add] == ["c"]
    assert diff.unchanged == frozenset({"b"})
    assert diff.to_remove == frozenset({"a"})
    assert not diff.is_noop


def test_diff_is_noop_when_nothing_changed() -> None:
    new = Contextualizer().contextualize(_tree(), [_chunk("a")])
    assert diff_chunks(existing_ids={"a"}, new=new).is_noop
