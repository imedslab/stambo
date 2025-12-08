import numpy as np

import stambo

SEED = 2025


def test_clustered_bootstrap_reduces_false_positives(grouped_gaussian_samples):
    sample_1, sample_2, groups = grouped_gaussian_samples

    statistics = {"mean": np.mean}

    naive = stambo.two_sample_test(
        sample_1,
        sample_2,
        statistics,
        n_bootstrap=2000,
        seed=SEED,
        silent=True,
    )

    clustered = stambo.two_sample_test(
        sample_1,
        sample_2,
        statistics,
        groups=groups,
        n_bootstrap=2000,
        seed=SEED,
        silent=True,
    )

    naive_p = naive["mean"][0]
    clustered_p = clustered["mean"][0]

    assert naive_p < 0.01
    assert clustered_p > 0.2
    assert clustered_p - naive_p > 0.15
