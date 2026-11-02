"""Local NLI cross-encoder served with ONNX Runtime — no torch required."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from numpy.typing import NDArray
from tokenizers import Tokenizer

from hybrid_rag.guardrails.models import NLIScores

MAX_HYPOTHESIS_CHARS = 1000


def softmax(logits: NDArray[np.float32]) -> NDArray[np.float64]:
    shifted = logits.astype(np.float64) - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    result: NDArray[np.float64] = exp / exp.sum(axis=-1, keepdims=True)
    return result


def label_indices(id2label: Mapping[str, str]) -> tuple[int, int, int]:
    """(entailment, neutral, contradiction) logit positions, read from the model config
    because cross-encoders disagree on label order."""
    by_label = {label.casefold(): int(index) for index, label in id2label.items()}
    try:
        return by_label["entailment"], by_label["neutral"], by_label["contradiction"]
    except KeyError as exc:
        raise ValueError(
            f"model must expose entailment/neutral/contradiction labels, got {sorted(by_label)}"
        ) from exc


class OnnxNLIModel:
    """``cross-encoder/nli-deberta-v3-small`` by default (int8 ONNX, ~180 MB)."""

    def __init__(
        self,
        repo: str = "cross-encoder/nli-deberta-v3-small",
        onnx_file: str = "onnx/model_quint8_avx2.onnx",
        *,
        cache_dir: str | None = None,
        max_length: int = 512,
        batch_size: int = 16,
    ) -> None:
        config = Path(hf_hub_download(repo, "config.json", cache_dir=cache_dir))
        tokenizer_file = hf_hub_download(repo, "tokenizer.json", cache_dir=cache_dir)
        model_file = hf_hub_download(repo, onnx_file, cache_dir=cache_dir)

        self._indices = label_indices(json.loads(config.read_text("utf-8"))["id2label"])
        self._tokenizer = Tokenizer.from_file(tokenizer_file)
        # Only the premise is truncated: the claim must survive intact.
        self._tokenizer.enable_truncation(max_length, strategy="only_first")
        self._tokenizer.enable_padding()
        self._session = ort.InferenceSession(model_file, providers=["CPUExecutionProvider"])
        self._input_names = {i.name for i in self._session.get_inputs()}
        self._batch_size = batch_size

    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[NLIScores]:
        scores: list[NLIScores] = []
        for start in range(0, len(pairs), self._batch_size):
            scores.extend(self._predict_batch(pairs[start : start + self._batch_size]))
        return scores

    def _predict_batch(self, pairs: Sequence[tuple[str, str]]) -> list[NLIScores]:
        encodings = self._tokenizer.encode_batch(
            [(premise, hypothesis[:MAX_HYPOTHESIS_CHARS]) for premise, hypothesis in pairs]
        )
        features = {
            "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
            "token_type_ids": np.array([e.type_ids for e in encodings], dtype=np.int64),
        }
        feeds = {name: array for name, array in features.items() if name in self._input_names}
        logits = np.asarray(self._session.run(None, feeds)[0], dtype=np.float32)
        entail, neutral, contra = self._indices
        probabilities = np.clip(softmax(logits), 0.0, 1.0)
        return [
            NLIScores(
                entailment=float(row[entail]),
                neutral=float(row[neutral]),
                contradiction=float(row[contra]),
            )
            for row in probabilities
        ]
