import numpy as np
import pytest

from hybrid_rag.evaluation.stats import bootstrap_ci, paired_bootstrap_delta


def test_constant_values_have_a_degenerate_interval() -> None:
    assert bootstrap_ci(np.full(20, 0.4)) == pytest.approx((0.4, 0.4, 0.4))


def test_interval_brackets_the_mean_and_is_reproducible_for_a_seed() -> None:
    values = np.random.default_rng(1).uniform(0, 1, size=200)

    mean, lo, hi = bootstrap_ci(values, seed=7)

    assert mean == pytest.approx(values.mean())
    assert lo < mean < hi
    assert (lo, hi) == bootstrap_ci(values, seed=7)[1:]
    assert (lo, hi) != bootstrap_ci(values, seed=8)[1:]


def test_more_data_gives_a_narrower_interval() -> None:
    rng = np.random.default_rng(3)
    small = bootstrap_ci(rng.uniform(0, 1, 30))
    large = bootstrap_ci(rng.uniform(0, 1, 3000))
    assert (large[2] - large[1]) < (small[2] - small[1])


def test_paired_delta_of_a_constant_gap_is_exact() -> None:
    base = np.random.default_rng(5).uniform(0, 0.5, 50)
    assert paired_bootstrap_delta(base + 0.1, base) == pytest.approx((0.1, 0.1, 0.1))


def test_paired_interval_excludes_zero_for_a_consistent_improvement_only() -> None:
    rng = np.random.default_rng(11)
    base = rng.uniform(0, 1, 300)
    better = base + rng.normal(0.05, 0.05, 300)
    noise = base + rng.normal(0.0, 0.2, 300)

    _, lo, _ = paired_bootstrap_delta(better, base)
    _, lo_noise, hi_noise = paired_bootstrap_delta(noise, base)

    assert lo > 0
    assert lo_noise < 0 < hi_noise


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        bootstrap_ci(np.array([]))
    with pytest.raises(ValueError, match="same length"):
        paired_bootstrap_delta(np.zeros(3), np.zeros(4))
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_ci(np.ones(3), confidence=1.0)
