import numpy as np
import pytest

import stambo
from stambo.metrics import Metric


class _MulticlassProbe(Metric):
    """Test-only metric that reports whether the sample it was called on was built with
    `multiclass=True`, so we can observe how `compare_models` set up each of its two samples."""

    def __init__(self) -> None:
        Metric.__init__(self, None, int_input=False)

    def __call__(self, sample) -> float:
        return float(sample.multiclass)

    def __str__(self) -> str:
        return "MulticlassProbe"


def test_compare_models_uses_preds_2_own_shape_for_multiclass_detection():
    """Regression test: `compare_models` used to derive `sample_2`'s `multiclass` flag from
    `preds_1.shape` instead of `preds_2.shape`. That's harmless when both prediction arrays
    share the same shape convention, but wrong when, say, model 1's predictions are 1D binary
    scores and model 2's are a 2D (3-class) probability matrix -- with the bug, `sample_2` would
    incorrectly be built with `multiclass=False` (copied from preds_1's 1D shape).
    """
    rng = np.random.default_rng(0)
    n = 100
    preds_1 = rng.uniform(0.0, 1.0, n)  # 1D
    preds_2 = rng.dirichlet(np.ones(3), size=n)  # 2D, 3-class softmax-style probabilities
    y_test = np.argmax(preds_2, axis=1)

    result = stambo.compare_models(y_test, preds_1, preds_2, (_MulticlassProbe(),), n_bootstrap=1, seed=0, silent=True)

    # Index 4 -> empirical value for sample 1, index 7 -> empirical value for sample 2.
    emp_s1, emp_s2 = result["MulticlassProbe"][4], result["MulticlassProbe"][7]
    assert emp_s1 == 0.0, "sample_1 (1D preds_1) must be built with multiclass=False"
    assert emp_s2 == 1.0, "sample_2 (2D preds_2) must be built with multiclass=True, not copied from preds_1's shape"


def test_compare_models_pairwise_smoke_and_holm_conservatism():
    rng = np.random.default_rng(3)
    n = 200
    y_test = rng.integers(0, 2, n)

    # Two similarly-informative models, and one pure-noise model.
    good_1 = np.clip(y_test * 0.6 + rng.normal(0.0, 0.3, n), 0.0, 1.0)
    good_2 = np.clip(y_test * 0.5 + rng.normal(0.0, 0.3, n), 0.0, 1.0)
    noise = rng.uniform(0.0, 1.0, n)

    result = stambo.compare_models_pairwise(
        y_test, (good_1, good_2, noise), ("ROCAUC", "AP"),
        labels=("good_1", "good_2", "noise"), n_bootstrap=500, seed=3, silent=True,
    )

    assert set(result.keys()) == {"ROCAUC", "AP"}
    for stat in result:
        assert set(result[stat].keys()) == {"good_1 / good_2", "good_1 / noise", "good_2 / noise"}
        for comparison, entry in result[stat].items():
            assert 0.0 <= entry["p_value"] <= 1.0
            assert 0.0 <= entry["p_value_adjusted"] <= 1.0
            assert entry["p_value_adjusted"] >= entry["p_value"] - 1e-12
        # Both good models beat the noise model; neither should beat each other.
        assert result[stat]["good_1 / noise"]["p_value_adjusted"] < 0.05
        assert result[stat]["good_2 / noise"]["p_value_adjusted"] < 0.05
        assert result[stat]["good_1 / good_2"]["p_value_adjusted"] >= 0.05


def test_compare_models_pairwise_requires_at_least_two_models():
    y_test = np.array([0, 1, 0, 1])
    preds = np.array([0.1, 0.9, 0.2, 0.8])
    with pytest.raises(AssertionError):
        stambo.compare_models_pairwise(y_test, (preds,), ("ROCAUC",), n_bootstrap=1, silent=True)
