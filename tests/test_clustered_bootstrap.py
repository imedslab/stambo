import numpy as np

import stambo

from stambo._predsamplewrapper import PredSampleWrapper
from stambo.metrics import Accuracy

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

    naive_p = naive["mean"]["p_value"]
    clustered_p = clustered["mean"]["p_value"]

    assert naive_p < 0.01
    assert clustered_p > 0.2
    assert clustered_p - naive_p > 0.15


def test_clustered_bootstrap_with_predsamplewrapper_and_metric_reduces_false_positives(clustered_binary_accuracy_data):
    """
    Follow the same idea as the Two_sample_test notebook:
    generate *within-subject correlated* observations via AR(1) block covariance.

    We create binary "correctness" indicators per model from correlated Gaussian noise.
    Using gt=1 everywhere, Accuracy(pred, gt) reduces to mean(pred == 1), so we can test the
    clustered bootstrap logic with PredSampleWrapper + Metric in a controlled non-iid setting.

    Expectation under the null: the two models have the same accuracy distribution.
    But due to strong within-group correlation, naive resampling (treating observations iid)
    becomes overconfident and can yield a false positive, whereas clustered bootstrap (resampling
    groups) corrects this and yields a non-significant result.
    """
    preds_1, preds_2, y, groups = clustered_binary_accuracy_data

    # Naive: no grouping information
    s1_naive = PredSampleWrapper(preds_1, y, multiclass=False, threshold=0.5, groups=None)
    s2_naive = PredSampleWrapper(preds_2, y, multiclass=False, threshold=0.5, groups=None)

    # Clustered: grouping stored on the wrapper (bootstrap_arrays will use it)
    s1_clustered = PredSampleWrapper(preds_1, y, multiclass=False, threshold=0.5, groups=groups)
    s2_clustered = PredSampleWrapper(preds_2, y, multiclass=False, threshold=0.5, groups=groups)

    statistics = {"Accuracy": Accuracy()}

    naive = stambo.two_sample_test(
        s1_naive,
        s2_naive,
        statistics,
        n_bootstrap=1500,
        seed=SEED,
        silent=True,
    )

    clustered = stambo.two_sample_test(
        s1_clustered,
        s2_clustered,
        statistics,
        n_bootstrap=1500,
        seed=SEED,
        silent=True,
    )

    naive_p = naive["Accuracy"]["p_value"]
    clustered_p = clustered["Accuracy"]["p_value"]

    # With this fixed seed + strong within-group correlation, naive bootstrap becomes overconfident.
    assert naive_p < 0.05
    assert clustered_p > 0.2
