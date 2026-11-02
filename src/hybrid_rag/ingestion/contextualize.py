"""Chunk contextualization (§3.4): breadcrumb always, LLM-written context opt-in."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from hybrid_rag.domain.models import Chunk, ContextualChunk, DocumentTree


class ChunkContextGenerator(Protocol):
    def generate(self, document: DocumentTree, chunk: Chunk) -> str:
        """Short text situating ``chunk`` within ``document`` (Contextual Retrieval)."""
        ...


class Contextualizer:
    def __init__(self, generator: ChunkContextGenerator | None = None) -> None:
        self._generator = generator

    def contextualize(self, tree: DocumentTree, chunks: Sequence[Chunk]) -> list[ContextualChunk]:
        return [self._one(tree, chunk) for chunk in chunks]

    def _one(self, tree: DocumentTree, chunk: Chunk) -> ContextualChunk:
        prefix = " > ".join((tree.title, *chunk.section_path))
        if self._generator is not None:
            extra = self._generator.generate(tree, chunk).strip()
            if extra:
                prefix = f"{prefix}\n{extra}"
        return ContextualChunk(
            **chunk.model_dump(),
            context_prefix=prefix,
            embed_text=f"{prefix}\n\n{chunk.text}",
        )
