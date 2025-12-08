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
