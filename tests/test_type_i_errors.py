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

    assert mean_result["p_value"] == pytest.approx(1.0, rel=0, abs=1e-9)
    assert mean_result["diff"] == pytest.approx(0.0, abs=1e-9)
    assert mean_result["emp_s1"] == pytest.approx(mean_result["emp_s2"], abs=1e-9)
