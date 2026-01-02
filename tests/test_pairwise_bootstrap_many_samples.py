import numpy as np

from stambo._stambo import bootstrap_arrays, pairwise_bootstrap_test


def test_pairwise_bootstrap_many_samples_holm_adjustment():
    """
    Create 50 paired samples from the same N(0,1) distribution.

    Expect:
    - Exactly N*(N-1)/2 pairwise comparisons.
    - At least one false positive without p-value adjustment (with many comparisons).
    - No significant comparisons after Holm-Bonferroni adjustment (with coarse bootstrap p-value resolution).
    """
    n_models = 50
    n_obs = 20
    n_bootstrap = 300  # coarse p-value grid -> makes Holm adjustment very conservative/stable
    alpha = 0.05

    rng = np.random.default_rng(2026)
    x = rng.normal(loc=0.0, scale=1.0, size=(n_obs, n_models))
    samples = tuple(x[:, i] for i in range(n_models))

    # bootstrap_arrays uses the legacy global RNG (np.random.*), so seed it explicitly.
    np.random.seed(2026)
    bootstrap_results = bootstrap_arrays(
        arrays=samples,
        statistics={"mean": np.mean},
        n_bootstrap=n_bootstrap,
        silent=True,
    )

    unadjusted = pairwise_bootstrap_test(
        bootstrap_results=bootstrap_results,
        samples=samples,
        statistics={"mean": np.mean},
        adjusted_p_value=False,
        alpha=alpha,
    )
    adjusted = pairwise_bootstrap_test(
        bootstrap_results=bootstrap_results,
        samples=samples,
        statistics={"mean": np.mean},
        adjusted_p_value=True,
        alpha=alpha,
    )

    expected_comparisons = n_models * (n_models - 1) // 2
    assert len(unadjusted["mean"]) == expected_comparisons
    assert len(adjusted["mean"]) == expected_comparisons

    pvals_unadj = np.array([unadjusted["mean"][k]["p_value"] for k in unadjusted["mean"]], dtype=float)
    pvals_adj = np.array([adjusted["mean"][k]["p_value"] for k in adjusted["mean"]], dtype=float)

    assert np.all((0.0 <= pvals_unadj) & (pvals_unadj <= 1.0))
    assert np.all((0.0 <= pvals_adj) & (pvals_adj <= 1.0))

    # With many comparisons under the complete null, we should see at least one false positive pre-adjustment.
    assert (pvals_unadj < alpha).sum() >= 1

    # With Holm-Bonferroni and a coarse bootstrap p-value grid, no result should survive adjustment.
    assert (pvals_adj < alpha).sum() == 0

    # Adjustment must not make p-values smaller.
    for label in unadjusted["mean"]:
        assert adjusted["mean"][label]["p_value"] >= unadjusted["mean"][label]["p_value"] - 1e-12


