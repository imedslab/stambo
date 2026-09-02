import numpy as np
import pytest

import stambo
from stambo._stambo import _bootstrap_pair_result


def test_pairwise_bootstrap_finds_the_one_shifted_sample():
    """5 paired samples, n=60: one shifted (+0.5) vs. four null (no shift).

    Every comparison touching the shifted sample should be significant; every
    comparison among the four null samples should not.
    """
    n_obs = 60
    alpha = 0.05
    rng = np.random.default_rng(2026)
    base = rng.normal(0.0, 1.0, n_obs)

    samples = (
        base + 0.5,  # shifted
        base + 0.0,
        base + 0.0,
        base + 0.0,
        base + 0.0,
    )
    labels = ("shifted", "n0", "n1", "n2", "n3")

    result = stambo.pairwise_bootstrap_test(
        samples=samples,
        statistics={"mean": np.mean},
        labels=labels,
        n_bootstrap=800,
        seed=2026,
        correction=None,
        silent=True,
    )

    assert len(result["mean"]) == 5 * 4 // 2

    for label, entry in result["mean"].items():
        touches_shifted = "shifted" in label
        if touches_shifted:
            assert entry["p_value"] < alpha, f"{label} should be significant, got p={entry['p_value']}"
        else:
            assert entry["p_value"] >= alpha, f"{label} should not be significant, got p={entry['p_value']}"


def test_pairwise_bootstrap_many_null_samples_holm_removes_false_positives():
    """50 mutually paired samples from the same N(0, 1): a true, complete null.

    Without correction we expect some false positives purely from running many
    comparisons (about 5% of them, at alpha=0.05). Holm-Bonferroni correction
    should control the family-wise error rate down to (in this run) zero.
    """
    n_models = 50
    n_obs = 20
    n_bootstrap = 300
    alpha = 0.05

    rng = np.random.default_rng(2026)
    x = rng.normal(0.0, 1.0, size=(n_obs, n_models))
    samples = tuple(x[:, i] for i in range(n_models))

    unadjusted = stambo.pairwise_bootstrap_test(
        samples=samples, statistics={"mean": np.mean}, n_bootstrap=n_bootstrap,
        seed=2026, correction=None, silent=True,
    )
    adjusted = stambo.pairwise_bootstrap_test(
        samples=samples, statistics={"mean": np.mean}, n_bootstrap=n_bootstrap,
        seed=2026, correction="holm", silent=True,
    )

    expected_comparisons = n_models * (n_models - 1) // 2
    assert len(unadjusted["mean"]) == expected_comparisons
    assert len(adjusted["mean"]) == expected_comparisons

    p_unadjusted = np.array([v["p_value"] for v in unadjusted["mean"].values()])
    p_adjusted = np.array([v["p_value_adjusted"] for v in adjusted["mean"].values()])

    assert np.all((0.0 <= p_unadjusted) & (p_unadjusted <= 1.0))
    assert np.all((0.0 <= p_adjusted) & (p_adjusted <= 1.0))

    # Under a true, complete null with many comparisons, some raw p-values should
    # cross the (uncorrected) 5% line just by chance.
    assert (p_unadjusted < alpha).sum() >= 1

    # Holm-Bonferroni must control the family-wise error rate: no comparison
    # should survive at alpha=0.05 once corrected, in this fixed-seed run.
    assert (p_adjusted < alpha).sum() == 0

    # Correction must never make a p-value smaller than the raw one.
    for label in unadjusted["mean"]:
        raw = unadjusted["mean"][label]["p_value"]
        adj = adjusted["mean"][label]["p_value_adjusted"]
        assert adj >= raw - 1e-12


def test_pairwise_bootstrap_uncorrected_matches_two_sample_test_for_two_samples():
    """For exactly two samples, pairwise_bootstrap_test (uncorrected) should reduce to
    two_sample_test's own math, since there is only a single comparison and Holm
    correction on a single p-value is a no-op."""
    rng = np.random.default_rng(7)
    s1 = rng.normal(0.0, 1.0, 80)
    s2 = rng.normal(0.3, 1.0, 80)

    direct = stambo.two_sample_test(s1, s2, statistics={"mean": np.mean}, n_bootstrap=500, seed=7, silent=True)

    # bootstrap_arrays and two_sample_test's own (no-groups, paired) resampling loop both
    # draw exactly one `np.random.choice(n, n, replace=True)` per iteration, so with the same
    # seed they consume the global RNG identically and should reach the same bootstrap draws.
    boot = stambo.bootstrap_arrays((s1, s2), statistics={"mean": np.mean}, n_bootstrap=500, seed=7, silent=True)
    pairwise = stambo.pairwise_bootstrap_test(
        samples=(s1, s2), statistics={"mean": np.mean}, bootstrap_results=boot,
        correction="holm", silent=True,
    )

    entry = pairwise["mean"]["0 / 1"]
    assert entry["p_value"] == pytest.approx(entry["p_value_adjusted"])  # single comparison: Holm is a no-op
    assert entry["p_value"] == pytest.approx(direct["mean"][0])
    assert entry["diff"] == pytest.approx(direct["mean"][1])


def test_pairwise_bootstrap_respects_groups():
    """Clustered/grouped pairwise comparison should behave like the paired two-sample
    case: a naive (ungrouped) comparison of correlated, clustered samples with no true
    group-level difference should be more prone to false positives than the clustered
    (grouped) one."""
    rng = np.random.default_rng(2025)
    group_sizes = np.array([200, 150, 180])
    group_effects = np.array([0.08, -0.07, 0.05])

    sample_1, sample_2, groups = [], [], []
    for gid, (size, effect) in enumerate(zip(group_sizes, group_effects)):
        base = rng.normal(loc=0.0, scale=0.02)
        sample_1.append(base + rng.normal(scale=0.01, size=size))
        sample_2.append(base + effect + rng.normal(scale=0.01, size=size))
        groups.append(np.full(size, gid, dtype=int))
    sample_1 = np.concatenate(sample_1)
    sample_2 = np.concatenate(sample_2)
    groups = np.concatenate(groups)

    naive = stambo.pairwise_bootstrap_test(
        samples=(sample_1, sample_2), statistics={"mean": np.mean},
        n_bootstrap=2000, seed=2025, correction=None, silent=True,
    )
    clustered = stambo.pairwise_bootstrap_test(
        samples=(sample_1, sample_2), statistics={"mean": np.mean}, groups=groups,
        n_bootstrap=2000, seed=2025, correction=None, silent=True,
    )

    naive_p = naive["mean"]["0 / 1"]["p_value"]
    clustered_p = clustered["mean"]["0 / 1"]["p_value"]

    assert naive_p < 0.01
    assert clustered_p > 0.2


def test_bootstrap_pair_result_identical_arrays_give_p_one():
    diff_array = np.zeros(500)
    boot = np.zeros(500)
    res = _bootstrap_pair_result(diff_array, boot, boot, observed=0.0, emp_1=0.0, emp_2=0.0, alpha=0.05)
    assert res["p_value"] == pytest.approx(1.0)


def test_holm_bonferroni_correction_known_values():
    p_values = np.array([0.01, 0.02, 0.03, 0.5])
    adjusted = stambo.holm_bonferroni_correction(p_values)
    # Sorted p's are already [0.01, 0.02, 0.03, 0.5], factors are [4, 3, 2, 1]:
    # raw*factor = [0.04, 0.06, 0.06, 0.5], already monotone non-decreasing.
    expected = np.array([0.04, 0.06, 0.06, 0.5])
    assert np.allclose(adjusted, expected)


def test_holm_bonferroni_correction_enforces_monotonicity_and_cap():
    # A later (larger) p-value that would adjust to something smaller than an
    # earlier one must be pulled up to match it; anything above 1 is capped.
    p_values = np.array([0.5, 0.5, 0.5])
    adjusted = stambo.holm_bonferroni_correction(p_values)
    assert np.all(adjusted == 1.0)
    assert np.all(np.diff(np.sort(adjusted)) >= -1e-12)


def test_pairwise_bootstrap_test_requires_unique_labels():
    with pytest.raises(AssertionError):
        stambo.pairwise_bootstrap_test(
            samples=(np.array([1.0, 2.0]), np.array([1.0, 2.0])),
            statistics={"mean": np.mean},
            labels=("A", "A"),
            n_bootstrap=1,
            silent=True,
        )


def test_pairwise_bootstrap_test_rejects_unknown_correction():
    with pytest.raises(AssertionError):
        stambo.pairwise_bootstrap_test(
            samples=(np.array([1.0, 2.0]), np.array([1.0, 2.0])),
            statistics={"mean": np.mean},
            correction="bonferroni",
            n_bootstrap=1,
            silent=True,
        )
