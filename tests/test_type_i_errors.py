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
    r"""Regression test for a Type-I-error inflation bug.

    ``two_sample_test`` previously computed the right-tailed p-value by
    shifting the bootstrap distribution of the difference by the observed
    effect and then comparing it against the observed effect again, i.e.
    testing ``diff_array >= 2 * observed`` instead of ``diff_array >= 0``.
    That formula only agrees with the (correct) direct percentile p-value
    when the bootstrap distribution of the difference happens to be
    symmetric around the observed effect. For a statistic with a skewed
    sampling distribution (e.g. the standard deviation, which follows a
    right-skewed, roughly chi-distributed sampling distribution for small
    samples), the two formulas disagree, and the shifted version rejects a
    true null hypothesis far more often than the nominal significance
    level allows.

    ``identical_gaussian_samples`` above cannot catch this: identical
    arrays always produce an all-zero bootstrap difference, so both the
    buggy and the fixed formula trivially return a p-value of 1. This test
    instead draws two independent samples from the same (non-degenerate)
    distribution many times and checks that the empirical false-positive
    rate under the true null stays close to the nominal alpha.
    """
    rng_seed = 2025
    n_trials = 500
    n_per_sample = 25
    n_bootstrap = 150
    alpha = 0.10
    # Nominal false-positive rate is 10%; the buggy shifted formula was
    # measured to reject ~17-20% of the time in this setting, so a bound
    # of 16% comfortably separates the fixed and the buggy implementation
    # while leaving headroom for Monte Carlo noise.
    max_allowed_rejection_rate = 0.16

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
