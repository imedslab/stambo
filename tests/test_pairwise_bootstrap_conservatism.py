import numpy as np

from stambo._stambo import bootstrap_arrays, pairwise_bootstrap_test


def _count_significant(results: dict, alpha: float) -> int:
    pvals = np.array([results["mean"][k]["p_value"] for k in results["mean"]], dtype=float)
    return int((pvals < alpha).sum())


def test_pairwise_bootstrap_conservatism_one_shifted_vs_four_null():
    """
    5 samples, n=50.
    1 sample from N(0.3, 1), 4 samples from N(0, 1).
    Expect exactly 4 significant differences (shifted vs each null).
    """
    n_models = 5
    n_obs = 50
    alpha = 0.05
    n_bootstrap = 400

    rng = np.random.default_rng(2026)
    base = rng.normal(loc=0.0, scale=1.0, size=n_obs)

    # Paired construction (still marginally N(mu, 1)) to make the expected discoveries stable.
    samples = (
        base + 0.3,  # shifted
        base + 0.0,
        base + 0.0,
        base + 0.0,
        base + 0.0,
    )
    assert len(samples) == n_models

    np.random.seed(2026)
    bootstrap_results = bootstrap_arrays(
        arrays=samples,
        statistics={"mean": np.mean},
        n_bootstrap=n_bootstrap,
        silent=True,
    )

    adjusted = pairwise_bootstrap_test(
        bootstrap_results=bootstrap_results,
        samples=samples,
        statistics={"mean": np.mean},
        adjusted_p_value=True,
        alpha=alpha,
    )

    # Total comparisons: 5*4/2 = 10
    assert len(adjusted["mean"]) == (n_models * (n_models - 1) // 2)
    assert _count_significant(adjusted, alpha=alpha) == 4


def test_pairwise_bootstrap_conservatism_two_shifted_vs_three_null():
    """
    5 samples, n=50.
    2 samples from N(0.3, 1), 3 samples from N(0, 1).
    Expect exactly 6 significant differences (each shifted vs each null).
    """
    n_models = 5
    n_obs = 50
    alpha = 0.05
    n_bootstrap = 400

    rng = np.random.default_rng(2027)
    base = rng.normal(loc=0.0, scale=1.0, size=n_obs)

    samples = (
        base + 0.3,  # shifted
        base + 0.3,  # shifted
        base + 0.0,
        base + 0.0,
        base + 0.0,
    )
    assert len(samples) == n_models

    np.random.seed(2027)
    bootstrap_results = bootstrap_arrays(
        arrays=samples,
        statistics={"mean": np.mean},
        n_bootstrap=n_bootstrap,
        silent=True,
    )

    adjusted = pairwise_bootstrap_test(
        bootstrap_results=bootstrap_results,
        samples=samples,
        statistics={"mean": np.mean},
        adjusted_p_value=True,
        alpha=alpha,
    )

    assert len(adjusted["mean"]) == (n_models * (n_models - 1) // 2)
    assert _count_significant(adjusted, alpha=alpha) == 6


