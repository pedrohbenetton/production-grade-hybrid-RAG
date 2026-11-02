"""Typed boundaries between ingestion stages (see docs/ARCHITECTURE.md §3.1)."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, computed_field

MimeType = Literal["text/markdown", "text/html", "application/pdf"]
BlockKind = Literal["paragraph", "list", "table", "code"]
SectionPath = tuple[str, ...]


def _not_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be blank")
    return value


NonBlankStr = Annotated[str, AfterValidator(_not_blank)]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RawDocument(_Frozen):
    source_uri: NonBlankStr
    mime_type: MimeType
    content: bytes = Field(repr=False)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.content).hexdigest()


class Block(_Frozen):
    kind: BlockKind
    text: NonBlankStr
    page: int | None = Field(default=None, ge=1)


class Section(_Frozen):
    heading: str
    level: int = Field(ge=0, le=6)
    blocks: tuple[Block, ...] = ()
    children: tuple[Section, ...] = ()


class DocumentTree(_Frozen):
    """Format-agnostic document: the root section's heading is the title."""

    doc_id: str
    source_uri: str
    title: str
    root: Section

    def iter_blocks(self) -> Iterator[tuple[SectionPath, Block]]:
        """Depth-first, document-order blocks with their heading path (root excluded)."""

        def walk(section: Section, path: SectionPath) -> Iterator[tuple[SectionPath, Block]]:
            for block in section.blocks:
                yield path, block
            for child in section.children:
                yield from walk(child, (*path, child.heading))

        return walk(self.root, ())


class Chunk(_Frozen):
    chunk_id: str
    doc_id: str
    text: NonBlankStr
    section_path: SectionPath = ()
    token_count: int = Field(ge=0)
    position: int = Field(ge=0)
    page: int | None = Field(default=None, ge=1)


class ContextualChunk(Chunk):
    """``text`` goes to the LLM prompt; ``embed_text`` goes to dense and sparse indexes."""

    context_prefix: str
    embed_text: NonBlankStr
