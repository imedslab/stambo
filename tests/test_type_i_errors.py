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


def test_two_sample_test_p_value_is_calibrated_for_skewed_statistics():
    r"""General Type-I-error calibration check for the two-tailed p-value.

    ``two_sample_test`` computes a two-tailed, percentile-based bootstrap
    p-value: ``2 * min(P(diff <= 0), P(diff >= 0))``, capped at 1, where
    ``diff`` is the (non-shifted) bootstrap distribution of the difference
    statistic. Both tail masses are computed directly (each with its own
    continuity correction) rather than deriving one from the other via
    ``1 - p``: at a point mass (e.g. identical samples, see
    ``test_two_sample_test_has_no_type_i_errors`` above), both
    ``P(diff <= 0)`` and ``P(diff >= 0)`` equal 1, correctly giving p = 1;
    deriving the second tail as ``1 - P(diff <= 0)`` would instead give
    p = 0 for identical samples, which is what that other test guards
    against.

    This test instead checks general calibration under real (non-
    degenerate) sampling variability: for a statistic with a skewed
    sampling distribution (e.g. the standard deviation, which follows a
    right-skewed, roughly chi-distributed sampling distribution for small
    samples), we draw two independent samples from the same distribution
    many times (true null) and check that the empirical false-positive
    rate stays reasonably close to the nominal alpha.
    """
    rng_seed = 2025
    n_trials = 500
    n_per_sample = 25
    n_bootstrap = 150
    alpha = 0.10
    # Nominal false-positive rate is 10%. The plain percentile method is
    # known to be somewhat biased for skewed statistics at small sample
    # sizes (this is why the docstrings mention BCa as a future
    # improvement); empirically this setting rejects ~11-16% of the time
    # across seeds, so 20% leaves comfortable headroom for Monte Carlo
    # noise while still catching a badly miscalibrated formula.
    max_allowed_rejection_rate = 0.20

    np.random.seed(rng_seed)
    rejections = 0
    for _ in range(n_trials):
        sample_1 = np.random.exponential(1.0, n_per_sample)
        sample_2 = np.random.exponential(1.0, n_per_sample)  # same distribution -> H0 true

        result = stambo.two_sample_test(
            sample_1,
            sample_2,
            statistics={"std": np.std},
            n_bootstrap=n_bootstrap,
            seed=None,
            non_paired=True,
            silent=True,
        )
        if result["std"][0] < alpha:
            rejections += 1

    rejection_rate = rejections / n_trials
    assert rejection_rate < max_allowed_rejection_rate, (
        f"Empirical false-positive rate {rejection_rate:.3f} at alpha={alpha} is too high; "
        "the bootstrap p-value formula looks miscalibrated."
    )
