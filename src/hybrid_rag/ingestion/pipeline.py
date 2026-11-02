from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Self

from pydantic import BaseModel, ConfigDict

from hybrid_rag.config import ChunkingSettings
from hybrid_rag.domain.models import ContextualChunk, DocumentTree, MimeType, RawDocument
from hybrid_rag.ingestion.chunking import StructuralSemanticChunker
from hybrid_rag.ingestion.contextualize import ChunkContextGenerator, Contextualizer
from hybrid_rag.ingestion.parsers import Parser, get_parser
from hybrid_rag.ports import Embedder


class ProcessedDocument(BaseModel):
    model_config = ConfigDict(frozen=True)

    tree: DocumentTree
    chunks: tuple[ContextualChunk, ...]


@dataclass(frozen=True, slots=True)
class ChunkDiff:
    to_add: tuple[ContextualChunk, ...]
    unchanged: frozenset[str]
    to_remove: frozenset[str]

    @property
    def is_noop(self) -> bool:
        return not self.to_add and not self.to_remove


def diff_chunks(existing_ids: Iterable[str], new: Sequence[ContextualChunk]) -> ChunkDiff:
    """Incremental re-indexing plan: content-addressed IDs make this a set difference."""
    existing = frozenset(existing_ids)
    new_ids = {c.chunk_id for c in new}
    return ChunkDiff(
        to_add=tuple(c for c in new if c.chunk_id not in existing),
        unchanged=existing & new_ids,
        to_remove=existing - new_ids,
    )


class IngestionPipeline:
    def __init__(
        self,
        chunker: StructuralSemanticChunker,
        contextualizer: Contextualizer,
        parser_for: Callable[[MimeType], Parser] = get_parser,
    ) -> None:
        self._chunker = chunker
        self._contextualizer = contextualizer
        self._parser_for = parser_for

    @classmethod
    def default(
        cls,
        embedder: Embedder,
        settings: ChunkingSettings | None = None,
        context_generator: ChunkContextGenerator | None = None,
    ) -> Self:
        return cls(
            StructuralSemanticChunker(embedder, settings or ChunkingSettings()),
            Contextualizer(context_generator),
        )

    def process(self, raw: RawDocument) -> ProcessedDocument:
        tree = self._parser_for(raw.mime_type).parse(raw)
        chunks = self._chunker.chunk(tree)
        return ProcessedDocument(
            tree=tree, chunks=tuple(self._contextualizer.contextualize(tree, chunks))
        )
