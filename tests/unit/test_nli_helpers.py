import numpy as np
import pytest

from hybrid_rag.guardrails.nli import label_indices, softmax


def test_softmax_rows_are_probabilities_and_stable_for_large_logits() -> None:
    logits = np.array([[1000.0, 1001.0, 999.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    probs = softmax(logits)

    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    assert probs[0].argmax() == 1
    np.testing.assert_allclose(probs[1], 1 / 3)


def test_label_indices_follow_the_models_own_config() -> None:
    id2label = {"0": "contradiction", "1": "entailment", "2": "neutral"}
    assert label_indices(id2label) == (1, 2, 0)  # (entailment, neutral, contradiction)


def test_label_lookup_is_case_insensitive() -> None:
    assert label_indices({"0": "NEUTRAL", "1": "Entailment", "2": "CONTRADICTION"}) == (1, 0, 2)


def test_models_without_nli_labels_are_rejected() -> None:
    with pytest.raises(ValueError, match="entailment"):
        label_indices({"0": "LABEL_0", "1": "LABEL_1"})
