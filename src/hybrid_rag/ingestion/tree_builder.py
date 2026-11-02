"""Stack-based builder shared by every parser: heading/block events → DocumentTree."""

from __future__ import annotations

from dataclasses import dataclass, field

from hybrid_rag.domain.ids import make_doc_id
from hybrid_rag.domain.models import Block, BlockKind, DocumentTree, Section


@dataclass
class _Node:
    heading: str
    level: int
    blocks: list[Block] = field(default_factory=list)
    children: list[_Node] = field(default_factory=list)

    def freeze(self) -> Section:
        return Section(
            heading=self.heading,
            level=self.level,
            blocks=tuple(self.blocks),
            children=tuple(c.freeze() for c in self.children),
        )


class TreeBuilder:
    def __init__(self, fallback_title: str) -> None:
        self._root = _Node(heading=fallback_title, level=0)
        self._stack = [self._root]
        # The first H1 is promoted to title only if nothing precedes it.
        self._title_open = True

    def heading(self, text: str, level: int) -> None:
        text = " ".join(text.split())
        if not text:
            return
        level = min(max(level, 1), 6)
        if level == 1 and self._title_open:
            self._root.heading = text
            self._title_open = False
            return
        self._title_open = False
        while self._stack[-1].level >= level:
            self._stack.pop()
        node = _Node(heading=text, level=level)
        self._stack[-1].children.append(node)
        self._stack.append(node)

    def block(self, kind: BlockKind, text: str, page: int | None = None) -> None:
        if not text.strip():
            return
        # Code keeps leading indentation; prose is trimmed.
        text = text.strip("\n").rstrip() if kind == "code" else text.strip()
        self._title_open = False
        self._stack[-1].blocks.append(Block(kind=kind, text=text, page=page))

    def build(self, source_uri: str) -> DocumentTree:
        return DocumentTree(
            doc_id=make_doc_id(source_uri),
            source_uri=source_uri,
            title=self._root.heading,
            root=self._root.freeze(),
        )
