"""Protocols for pluggable components. Concrete adapters live next to their stage."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Return a ``(len(texts), dim)`` matrix."""
        ...


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...
