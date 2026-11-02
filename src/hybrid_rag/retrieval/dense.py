"""Qdrant-backed dense index (§4.1). Payloads hold the full chunk, making Qdrant
the system of record from which the BM25 index can always be rebuilt."""

from __future__ import annotations

import uuid
import warnings
from collections.abc import Iterable, Iterator, Sequence

import numpy as np
from numpy.typing import NDArray
from qdrant_client import QdrantClient, models

from hybrid_rag.config import QdrantSettings
from hybrid_rag.domain.models import ContextualChunk, ScoredChunk

# Qdrant point IDs must be UUIDs or ints; derive them deterministically.
_POINT_NAMESPACE = uuid.UUID("6f1c1d9e-3c0b-4e6a-9a57-2b8f0e1d4c21")


class DimensionMismatchError(RuntimeError):
    pass


def make_qdrant_client(settings: QdrantSettings) -> QdrantClient:
    if settings.url:
        api_key = settings.api_key.get_secret_value() if settings.api_key else None
        return QdrantClient(url=settings.url, api_key=api_key)
    if settings.path:
        return QdrantClient(path=settings.path)
    return QdrantClient(location=":memory:")


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_POINT_NAMESPACE, chunk_id))


class QdrantDenseIndex:
    def __init__(
        self,
        client: QdrantClient,
        collection: str,
        dimension: int,
        *,
        page_size: int = 256,
        hnsw_m: int = 16,
        hnsw_ef_construct: int = 100,
    ) -> None:
        self._client = client
        self._collection = collection
        self._dimension = dimension
        self._page_size = page_size
        self._ensure_collection(hnsw_m, hnsw_ef_construct)

    def upsert(self, chunks: Sequence[ContextualChunk], vectors: NDArray[np.float32]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"got {len(chunks)} chunks but {len(vectors)} vectors")
        if not chunks:
            return
        points = [
            models.PointStruct(
                id=point_id(chunk.chunk_id),
                vector=vector.tolist(),
                payload=chunk.model_dump(mode="json"),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self._client.upsert(self._collection, points=points, wait=True)

    def delete(self, chunk_ids: Iterable[str]) -> None:
        ids: list[models.ExtendedPointId] = [point_id(cid) for cid in chunk_ids]
        if ids:
            self._client.delete(
                self._collection, points_selector=models.PointIdsList(points=ids), wait=True
            )

    def ids_for_doc(self, doc_id: str) -> set[str]:
        doc_filter = models.Filter(
            must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
        )
        return {
            str(record.payload["chunk_id"])
            for record in self._scroll(doc_filter, with_payload=["chunk_id"])
            if record.payload
        }

    def search(self, vector: NDArray[np.float32], k: int) -> list[ScoredChunk]:
        response = self._client.query_points(
            self._collection, query=vector.tolist(), limit=k, with_payload=True
        )
        return [
            ScoredChunk(chunk=ContextualChunk.model_validate(point.payload), score=point.score)
            for point in response.points
        ]

    def iter_chunks(self) -> Iterator[ContextualChunk]:
        for record in self._scroll(None, with_payload=True):
            yield ContextualChunk.model_validate(record.payload)

    def count(self) -> int:
        return self._client.count(self._collection, exact=True).count

    def _scroll(
        self, scroll_filter: models.Filter | None, with_payload: bool | list[str]
    ) -> Iterator[models.Record]:
        offset: models.ExtendedPointId | None = None
        while True:
            records, offset = self._client.scroll(
                self._collection,
                scroll_filter=scroll_filter,
                limit=self._page_size,
                offset=offset,
                with_payload=with_payload,
                with_vectors=False,
            )
            yield from records
            if offset is None:
                return

    def _ensure_collection(self, hnsw_m: int, hnsw_ef_construct: int) -> None:
        if self._client.collection_exists(self._collection):
            vectors = self._client.get_collection(self._collection).config.params.vectors
            size = vectors.size if isinstance(vectors, models.VectorParams) else None
            if size != self._dimension:
                raise DimensionMismatchError(
                    f"collection {self._collection!r} has dimension {size}, "
                    f"embedder produces {self._dimension}; re-index into a new collection"
                )
            return
        self._client.create_collection(
            self._collection,
            vectors_config=models.VectorParams(
                size=self._dimension, distance=models.Distance.COSINE
            ),
            hnsw_config=models.HnswConfigDiff(m=hnsw_m, ef_construct=hnsw_ef_construct),
        )
        with warnings.catch_warnings():
            # Embedded mode ignores payload indexes (and says so); server mode uses it.
            warnings.filterwarnings("ignore", message="Payload indexes have no effect")
            self._client.create_payload_index(
                self._collection,
                field_name="doc_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
