"""Keeps the dense (system of record) and sparse indexes in sync, per document."""

from __future__ import annotations

from itertools import batched

from hybrid_rag.domain.models import IndexReport
from hybrid_rag.ingestion.pipeline import ProcessedDocument, diff_chunks
from hybrid_rag.ports import DenseIndex, Embedder
from hybrid_rag.retrieval.sparse import BM25Index


class Indexer:
    def __init__(
        self, embedder: Embedder, dense: DenseIndex, sparse: BM25Index, *, batch_size: int = 64
    ) -> None:
        self._embedder = embedder
        self._dense = dense
        self._sparse = sparse
        self._batch_size = batch_size

    def index(self, document: ProcessedDocument) -> IndexReport:
        """Embed only new chunks; drop stale ones. Unchanged documents cost zero embeddings."""
        doc_id = document.tree.doc_id
        diff = diff_chunks(self._dense.ids_for_doc(doc_id), document.chunks)

        # Add before remove: a concurrent reader never sees the document vanish.
        for batch in batched(diff.to_add, self._batch_size):
            vectors = self._embedder.embed([chunk.embed_text for chunk in batch])
            self._dense.upsert(batch, vectors)
            self._sparse.add(batch)
        self._dense.delete(diff.to_remove)
        self._sparse.remove(diff.to_remove)

        return IndexReport(
            doc_id=doc_id,
            added=len(diff.to_add),
            unchanged=len(diff.unchanged),
            removed=len(diff.to_remove),
        )

    def delete_document(self, doc_id: str) -> int:
        chunk_ids = self._dense.ids_for_doc(doc_id)
        self._dense.delete(chunk_ids)
        self._sparse.remove(chunk_ids)
        return len(chunk_ids)


def rebuild_sparse(dense: DenseIndex, sparse: BM25Index) -> int:
    """Rebuild BM25 from Qdrant payloads (startup or recovery). O(total tokens)."""
    sparse.add(dense.iter_chunks())
    return len(sparse)
