"""Protocols for pluggable components. Concrete adapters live next to their stage."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from hybrid_rag.domain.models import ContextualChunk, ScoredChunk


class Embedder(Protocol):
    @property
    def dimension(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Embed documents/passages; returns a ``(len(texts), dimension)`` matrix."""
        ...

    def embed_query(self, text: str) -> NDArray[np.float32]:
        """Embed a search query (asymmetric models use a different prompt)."""
        ...


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class DenseIndex(Protocol):
    """Vector store; also the system of record for chunk payloads."""

    def upsert(self, chunks: Sequence[ContextualChunk], vectors: NDArray[np.float32]) -> None: ...

    def delete(self, chunk_ids: Iterable[str]) -> None: ...

    def ids_for_doc(self, doc_id: str) -> set[str]: ...

    def search(self, vector: NDArray[np.float32], k: int) -> list[ScoredChunk]: ...

    def iter_chunks(self) -> Iterator[ContextualChunk]: ...

    def count(self) -> int: ...
