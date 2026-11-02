import os

import numpy as np
import pytest
from conftest import CATS, ROCKETS

pytestmark = [
    pytest.mark.models,
    pytest.mark.skipif(
        os.environ.get("RUN_MODEL_TESTS") != "1", reason="set RUN_MODEL_TESTS=1 (~130 MB download)"
    ),
]


def test_bge_small_embeds_and_ranks_by_topic() -> None:
    from hybrid_rag.retrieval.embeddings import FastEmbedEmbedder

    embedder = FastEmbedEmbedder("BAAI/bge-small-en-v1.5")
    passages = embedder.embed([" ".join(CATS), " ".join(ROCKETS)])
    query = embedder.embed_query("why do cats purr?")

    assert embedder.dimension == 384
    assert passages.shape == (2, 384)
    assert passages.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(passages, axis=1), 1.0, atol=1e-3)
    assert float(passages[0] @ query) > float(passages[1] @ query)
    assert embedder.embed([]).shape == (0, 384)
