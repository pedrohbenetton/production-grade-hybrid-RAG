from __future__ import annotations

from selectolax.lexbor import LexborHTMLParser as HTMLParser
from selectolax.lexbor import LexborNode as Node

from hybrid_rag.domain.models import DocumentTree, RawDocument
from hybrid_rag.ingestion.parsers.base import decode_text, fallback_title
from hybrid_rag.ingestion.tree_builder import TreeBuilder

_BOILERPLATE = "script, style, noscript, template, svg, nav, header, footer, aside, form"
_HEADINGS = {f"h{n}": n for n in range(1, 7)}
_CONTAINERS = frozenset(
    {"html", "body", "main", "article", "section", "div", "figure", "details", "center"}
)
_LEAF_BLOCKS = frozenset({"p", "ul", "ol", "table", "pre", "blockquote"})
_BLOCK_TAGS = _CONTAINERS | _LEAF_BLOCKS | _HEADINGS.keys()


def _clean(text: str | None) -> str:
    return " ".join((text or "").split())


def _render_list(node: Node, *, ordered: bool) -> str:
    items = [_clean(li.text()) for li in node.iter() if li.tag == "li"]
    return "\n".join(
        f"{i}. {item}" if ordered else f"- {item}" for i, item in enumerate(items, 1) if item
    )


def _render_table(node: Node) -> str:
    rows = [
        [_clean(cell.text()).replace("|", r"\|") for cell in tr.iter() if cell.tag in ("th", "td")]
        for tr in node.css("tr")
    ]
    rows = [r for r in rows if r]
    if not rows:
        return ""
    lines = ["| " + " | ".join(rows[0]) + " |", "| " + " | ".join("---" for _ in rows[0]) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


class HtmlParser:
    def parse(self, raw: RawDocument) -> DocumentTree:
        doc = HTMLParser(decode_text(raw))
        title_node = doc.css_first("title")
        title = _clean(title_node.text()) if title_node is not None else ""
        for node in doc.css(_BOILERPLATE):
            node.decompose()

        builder = TreeBuilder(title or fallback_title(raw.source_uri))
        content = doc.css_first("main") or doc.css_first("article") or doc.body
        if content is not None:
            self._walk(content, builder)
        return builder.build(raw.source_uri)

    def _walk(self, node: Node, builder: TreeBuilder) -> None:
        # Consecutive inline elements and bare text form a single paragraph.
        inline: list[str] = []

        def flush() -> None:
            if inline:
                builder.block("paragraph", _clean(" ".join(inline)))
                inline.clear()

        for child in node.iter(include_text=True):
            tag = child.tag
            if tag not in _BLOCK_TAGS:
                inline.append(child.text() or "")
                continue
            flush()
            if tag in _HEADINGS:
                builder.heading(_clean(child.text()), _HEADINGS[tag])
            elif tag in ("p", "blockquote"):
                builder.block("paragraph", _clean(child.text()))
            elif tag in ("ul", "ol"):
                builder.block("list", _render_list(child, ordered=tag == "ol"))
            elif tag == "table":
                builder.block("table", _render_table(child))
            elif tag == "pre":
                builder.block("code", child.text(deep=True) or "")
            else:
                self._walk(child, builder)
        flush()
