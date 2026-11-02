from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from fastembed import TextEmbedding
from numpy.typing import NDArray


class FastEmbedEmbedder:
    """ONNX (CPU) embeddings via FastEmbed — no torch, no GPU required.

    ``passage_embed`` / ``query_embed`` apply the model's asymmetric prompts
    (e.g. BGE's "Represent this sentence for searching relevant passages: ").
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        *,
        batch_size: int = 64,
        cache_dir: str | None = None,
    ) -> None:
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
        self._dimension = int(TextEmbedding.get_embedding_size(model_name))
        self._batch_size = batch_size

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)
        vectors = self._model.passage_embed(list(texts), batch_size=self._batch_size)
        return np.stack(list(vectors)).astype(np.float32)

    def embed_query(self, text: str) -> NDArray[np.float32]:
        return np.asarray(next(iter(self._model.query_embed(text))), dtype=np.float32)
