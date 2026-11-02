"""Bootstrap confidence intervals: with a few hundred queries, small gaps between
systems are often noise, so every reported difference carries an interval."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _resampled_means(
    values: NDArray[np.float64], n_resamples: int, seed: int
) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_resamples, len(values)))
    means: NDArray[np.float64] = values[indices].mean(axis=1)
    return means


def _interval(means: NDArray[np.float64], confidence: float) -> tuple[float, float]:
    tail = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(means, [tail, 1.0 - tail])
    return float(lo), float(hi)


def _check(confidence: float) -> None:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")


def bootstrap_ci(
    values: NDArray[np.float64],
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """(mean, lower, upper): percentile bootstrap over queries."""
    _check(confidence)
    if len(values) == 0:
        raise ValueError("cannot bootstrap an empty sample")
    lo, hi = _interval(_resampled_means(values, n_resamples, seed), confidence)
    return float(values.mean()), lo, hi


def paired_bootstrap_delta(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """(mean(a − b), lower, upper). Pairing by query removes difficulty variance, so it is
    far more sensitive than comparing two independent intervals."""
    _check(confidence)
    if len(a) != len(b):
        raise ValueError("samples must have the same length")
    return bootstrap_ci(a - b, n_resamples=n_resamples, confidence=confidence, seed=seed)
