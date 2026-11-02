from __future__ import annotations

from markdown_it import MarkdownIt

from hybrid_rag.domain.models import BlockKind, DocumentTree, RawDocument
from hybrid_rag.ingestion.parsers.base import decode_text, fallback_title
from hybrid_rag.ingestion.tree_builder import TreeBuilder

_MD = MarkdownIt("commonmark").enable("table")

# Container blocks are copied verbatim from source lines (token.map) so lists
# and tables keep their Markdown layout, which LLMs read well.
_VERBATIM: dict[str, BlockKind] = {
    "bullet_list_open": "list",
    "ordered_list_open": "list",
    "table_open": "table",
    "blockquote_open": "paragraph",
    "html_block": "paragraph",
}


class MarkdownParser:
    def parse(self, raw: RawDocument) -> DocumentTree:
        source = decode_text(raw)
        lines = source.splitlines()
        tokens = _MD.parse(source)
        builder = TreeBuilder(fallback_title(raw.source_uri))

        for i, tok in enumerate(tokens):
            if tok.level != 0:
                continue
            if tok.type == "heading_open":
                builder.heading(tokens[i + 1].content, int(tok.tag[1:]))
            elif tok.type == "paragraph_open":
                builder.block("paragraph", tokens[i + 1].content)
            elif tok.type in ("fence", "code_block"):
                builder.block("code", tok.content)
            elif tok.type in _VERBATIM and tok.map is not None:
                start, end = tok.map
                builder.block(_VERBATIM[tok.type], "\n".join(lines[start:end]))

        return builder.build(raw.source_uri)
