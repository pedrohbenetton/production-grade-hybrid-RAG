from pathlib import Path

import pytest

from hybrid_rag.domain.models import DocumentTree, RawDocument, Section
from hybrid_rag.ingestion.parsers import HtmlParser, MarkdownParser, PdfParser, get_parser


def _raw(path: Path, mime: str) -> RawDocument:
    return RawDocument(source_uri=path.as_uri(), mime_type=mime, content=path.read_bytes())


def _section(tree: DocumentTree, *path: str) -> Section:
    node = tree.root
    for heading in path:
        node = next(c for c in node.children if c.heading == heading)
    return node


def _all_text(tree: DocumentTree) -> str:
    return "\n".join(blk.text for _, blk in tree.iter_blocks())


@pytest.mark.parametrize(
    ("mime", "parser_type"),
    [("text/markdown", MarkdownParser), ("text/html", HtmlParser), ("application/pdf", PdfParser)],
)
def test_registry_resolves_parser_by_mime(mime: str, parser_type: type) -> None:
    assert isinstance(get_parser(mime), parser_type)  # type: ignore[arg-type]


class TestMarkdown:
    @pytest.fixture
    def tree(self, fixtures_dir: Path) -> DocumentTree:
        return MarkdownParser().parse(_raw(fixtures_dir / "sample.md", "text/markdown"))

    def test_title_and_intro(self, tree: DocumentTree) -> None:
        assert tree.title == "Hybrid Search Guide"
        assert [b.text for b in tree.root.blocks] == [
            "Hybrid search combines lexical and semantic retrieval."
        ]

    def test_list_is_a_single_block(self, tree: DocumentTree) -> None:
        sparse = _section(tree, "Sparse Retrieval")
        assert [b.kind for b in sparse.blocks] == ["paragraph", "list"]
        assert "saturates with k1" in sparse.blocks[1].text
        assert "controlled by b" in sparse.blocks[1].text

    def test_code_fence_is_kept_intact(self, tree: DocumentTree) -> None:
        dense = _section(tree, "Dense Retrieval")
        code = dense.blocks[1]
        assert code.kind == "code"
        assert "def cosine(a, b):\n    return a @ b" in code.text

    def test_nested_section_with_table(self, tree: DocumentTree) -> None:
        fusion = _section(tree, "Dense Retrieval", "Fusion")
        assert [b.kind for b in fusion.blocks] == ["table"]
        assert "| RRF | No |" in fusion.blocks[0].text


class TestHtml:
    @pytest.fixture
    def tree(self, fixtures_dir: Path) -> DocumentTree:
        return HtmlParser().parse(_raw(fixtures_dir / "sample.html", "text/html"))

    def test_title_and_boilerplate_removed(self, tree: DocumentTree) -> None:
        assert tree.title == "Vector Databases"
        text = _all_text(tree)
        for noise in ("alert(1)", "color: red", "Home", "Copyright"):
            assert noise not in text

    def test_list_rendered_as_markdown(self, tree: DocumentTree) -> None:
        hnsw = _section(tree, "HNSW")
        assert [b.kind for b in hnsw.blocks] == ["paragraph", "list"]
        assert hnsw.blocks[1].text == "- m controls graph degree.\n- ef controls search breadth."

    def test_table_div_and_pre(self, tree: DocumentTree) -> None:
        bench = _section(tree, "Benchmarks")
        assert [b.kind for b in bench.blocks] == ["table", "paragraph", "code"]
        assert bench.blocks[0].text == "| Engine | Recall |\n| --- | --- |\n| Qdrant | 0.98 |"
        assert bench.blocks[1].text == "Loose text inside a div."
        assert bench.blocks[2].text == "client.search(collection, vector)"
