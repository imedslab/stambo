import numpy as np

from stambo._stambo import apply_correction, pairwise_bootstrap_test


def test_apply_correction_returns_results_and_adjusts_in_place():
    # Two p-values -> Holm should adjust the smallest by factor 2.
    results = {
        "mean": {
            "A / B": {"p_value": 0.01, "diff": 0.0},
            "A / C": {"p_value": 0.04, "diff": 0.0},
        }
    }

    out = apply_correction(results)

    # API: return the (mutated) results dict
    assert out is results

    assert results["mean"]["A / B"]["p_value"] == 0.02
    assert results["mean"]["A / C"]["p_value"] == 0.04


def test_pairwise_bootstrap_ci_uses_alpha_as_fraction():
    # Construct a tiny deterministic bootstrap distribution
    # col 0 = model A, col 1 = model B
    sample_1_b = np.array([0.0, 0.0, 0.0, 0.0])
    sample_2_b = np.array([0.0, 1.0, 2.0, 3.0])
    bootstrap_results = {"mean": np.stack([sample_1_b, sample_2_b], axis=1)}

    # samples are only used for empirical stats; keep them simple
    samples = (np.array([0.0, 0.0]), np.array([0.0, 0.0]))

    alpha = 0.05
    res = pairwise_bootstrap_test(
        samples=samples,
        statistics={"mean": np.mean},
        bootstrap_results=bootstrap_results,
        labels=("A", "B"),
        adjusted_p_value=False,
        alpha=alpha,
    )

    label = "A / B"
    diff_array = sample_2_b - sample_1_b
    expected_ci_es = (
        float(np.percentile(diff_array, 100 * alpha / 2.0)),
        float(np.percentile(diff_array, 100 - 100 * alpha / 2.0)),
    )
    expected_ci_s2 = (
        float(np.percentile(sample_2_b, 100 * alpha / 2.0)),
        float(np.percentile(sample_2_b, 100 - 100 * alpha / 2.0)),
    )

    assert np.allclose(res["mean"][label]["ci_es"], expected_ci_es)
    assert np.allclose(res["mean"][label]["ci_s2"], expected_ci_s2)


