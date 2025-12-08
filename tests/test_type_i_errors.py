import numpy as np
import pytest

import stambo


def test_two_sample_test_has_no_type_i_errors(identical_gaussian_samples):
    sample_1, sample_2 = identical_gaussian_samples

    results = stambo.two_sample_test(
        sample_1,
        sample_2,
        statistics={"mean": np.mean},
        n_bootstrap=500,
        seed=1337,
        silent=True,
    )

    mean_result = results["mean"]

    # Index 0 -> p-value, 1 -> observed diff, 4 & 7 -> empirical metric per sample.
    assert mean_result[0] == pytest.approx(1.0, rel=0, abs=1e-9)
    assert mean_result[1] == pytest.approx(0.0, abs=1e-9)
    assert mean_result[4] == pytest.approx(mean_result[7], abs=1e-9)
