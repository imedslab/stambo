from typing import Tuple

import numpy as np
import pytest

SEED = 2025


@pytest.fixture
def identical_gaussian_samples() -> Tuple[np.ndarray, np.ndarray]:
    """Generate identical Gaussian samples to stress-test type I errors."""
    rng = np.random.default_rng(SEED)
    sample = rng.normal(loc=0.0, scale=1.0, size=1024)
    return sample, sample.copy()


@pytest.fixture
def grouped_gaussian_samples() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create clustered samples with strong intra-group correlation."""
    rng = np.random.default_rng(SEED)
    group_sizes = np.array([200, 150, 180])
    group_effects = np.array([0.08, -0.07, 0.05])

    sample_1, sample_2, groups = [], [], []

    for gid, (size, effect) in enumerate(zip(group_sizes, group_effects)):
        base = rng.normal(loc=0.0, scale=0.02)
        sample_1.append(base + rng.normal(scale=0.01, size=size))
        sample_2.append(base + effect + rng.normal(scale=0.01, size=size))
        groups.append(np.full(size, gid, dtype=int))

    return (
        np.concatenate(sample_1),
        np.concatenate(sample_2),
        np.concatenate(groups),
    )


@pytest.fixture
def clustered_binary_accuracy_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a deterministic non-iid binary dataset with strong within-group correlation.

    Mirrors the AR(1) block-covariance idea used in notebooks/Two_sample_test.ipynb:
    - variable group sizes
    - within-group AR(1) covariance with high rho
    - two independent correlated draws per group -> two "models"

    Returns:
        preds_1, preds_2, y, groups
    """
    rng = np.random.default_rng(SEED)

    def generate_covariance(size: int, rho: float, sigma: float = 1.0) -> np.ndarray:
        idx = np.arange(size)
        diff = np.abs(np.subtract.outer(idx, idx))
        return (sigma**2) * (rho**diff)

    n_groups = 40
    n_total = 800
    rho = 0.95
    noise_sigma = 1.0

    if n_total < n_groups:
        raise ValueError("n_total must be >= n_groups")

    base = np.ones(n_groups, dtype=int)
    remaining = n_total - n_groups
    probs = rng.dirichlet(np.ones(n_groups) * 0.9)
    extra = rng.multinomial(remaining, probs)
    counts = base + extra
    assert counts.sum() == n_total

    groups = np.concatenate([np.full(counts[i], i, dtype=int) for i in range(n_groups)])

    preds_1 = []
    preds_2 = []
    for i in range(n_groups):
        size = int(counts[i])
        sigma_block = generate_covariance(size, rho, noise_sigma)
        L = np.linalg.cholesky(sigma_block)
        e1 = L @ rng.normal(0.0, 1.0, size=size)
        e2 = L @ rng.normal(0.0, 1.0, size=size)
        preds_1.append((e1 > 0.0).astype(int))
        preds_2.append((e2 > 0.0).astype(int))

    preds_1 = np.concatenate(preds_1)
    preds_2 = np.concatenate(preds_2)
    y = np.ones(n_total, dtype=int)

    return preds_1, preds_2, y, groups

