"""Two-pass chunking: structural packing, semantic split only where needed (§3.3)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from hybrid_rag.config import ChunkingSettings
from hybrid_rag.domain.ids import make_chunk_id
from hybrid_rag.domain.models import Chunk, DocumentTree, SectionPath
from hybrid_rag.ingestion.semantic import SemanticSplitter
from hybrid_rag.ingestion.tokens import RegexTokenCounter
from hybrid_rag.ports import Embedder, TokenCounter

ATOMIC_KINDS = frozenset({"code", "table"})


@dataclass(frozen=True, slots=True)
class _Unit:
    path: SectionPath
    text: str
    page: int | None
    # A semantic fragment after the first: re-packing it with its predecessor
    # would undo the topic boundary the splitter just found.
    boundary: bool = False


def _common_prefix(paths: Sequence[SectionPath]) -> SectionPath:
    prefix = paths[0]
    for path in paths[1:]:
        n = 0
        while n < min(len(prefix), len(path)) and prefix[n] == path[n]:
            n += 1
        prefix = prefix[:n]
    return prefix


def _related(a: SectionPath, b: SectionPath) -> bool:
    """Siblings, parent/child, or uncle/nephew — close enough to share a chunk."""
    return b[: len(a) - 1] == a[:-1] or a[: len(b) - 1] == b[:-1]


class StructuralSemanticChunker:
    def __init__(
        self,
        embedder: Embedder,
        settings: ChunkingSettings,
        counter: TokenCounter | None = None,
    ) -> None:
        self._settings = settings
        self._counter = counter or RegexTokenCounter()
        self._splitter = SemanticSplitter(
            embedder,
            self._counter,
            max_tokens=settings.max_tokens,
            breakpoint_percentile=settings.breakpoint_percentile,
            window=settings.window,
        )

    def chunk(self, tree: DocumentTree) -> list[Chunk]:
        groups = self._pack(self._units(tree))
        seen: Counter[tuple[SectionPath, str]] = Counter()
        chunks: list[Chunk] = []
        for position, group in enumerate(groups):
            path, text = self._render(group)
            occurrence = seen[path, text]
            seen[path, text] += 1
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(tree.doc_id, path, text, occurrence),
                    doc_id=tree.doc_id,
                    text=text,
                    section_path=path,
                    token_count=self._counter.count(text),
                    position=position,
                    page=next((u.page for u in group if u.page is not None), None),
                )
            )
        return chunks

    def _units(self, tree: DocumentTree) -> Iterator[_Unit]:
        for path, block in tree.iter_blocks():
            fits = self._counter.count(block.text) <= self._settings.max_tokens
            if fits or block.kind in ATOMIC_KINDS:
                yield _Unit(path, block.text, block.page)
                continue
            for i, piece in enumerate(self._splitter.split(block.text)):
                yield _Unit(path, piece, block.page, boundary=i > 0)

    def _pack(self, units: Iterator[_Unit]) -> list[list[_Unit]]:
        groups: list[list[_Unit]] = []
        buffer: list[_Unit] = []
        for unit in units:
            if buffer and self._can_join(buffer, unit):
                buffer.append(unit)
                continue
            if buffer:
                groups.append(buffer)
            buffer = [unit]
        if buffer:
            groups.append(buffer)
        return groups

    def _can_join(self, buffer: list[_Unit], unit: _Unit) -> bool:
        if self._count([*buffer, unit]) > self._settings.max_tokens:
            return False
        if not unit.boundary and all(u.path == unit.path for u in buffer):
            return True
        # Otherwise only absorb into an undersized chunk from a nearby section.
        common = _common_prefix([u.path for u in buffer])
        return self._count(buffer) < self._settings.min_tokens and _related(common, unit.path)

    def _count(self, group: list[_Unit]) -> int:
        return self._counter.count(self._render(group)[1])

    @staticmethod
    def _render(group: list[_Unit]) -> tuple[SectionPath, str]:
        """Mixed-section chunks keep each sub-section's heading inline."""
        common = _common_prefix([u.path for u in group])
        parts: list[str] = []
        previous: SectionPath | None = None
        for unit in group:
            if unit.path != previous and len(unit.path) > len(common):
                parts.append(" > ".join(unit.path[len(common) :]) + "\n" + unit.text)
            else:
                parts.append(unit.text)
            previous = unit.path
        return common, "\n\n".join(parts)
