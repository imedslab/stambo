import numpy as np
import pytest

from stambo import PredSampleWrapper, bootstrap_arrays


def test_bootstrap_arrays_rejects_non_array_inputs():
    arrays = (np.array([1, 2, 3], dtype=float), "not-an-array")
    with pytest.raises(ValueError):
        bootstrap_arrays(arrays=arrays, statistics={"mean": np.mean}, n_bootstrap=1, silent=True)


def test_bootstrap_arrays_requires_same_lengths():
    arrays = (np.array([1, 2, 3], dtype=float), np.array([1, 2], dtype=float))
    with pytest.raises(AssertionError, match="same length"):
        bootstrap_arrays(arrays=arrays, statistics={"mean": np.mean}, n_bootstrap=1, silent=True)


def test_bootstrap_arrays_numpy_branch_requires_same_dtype():
    arrays = (np.array([1, 2, 3], dtype=int), np.array([1, 2, 3], dtype=float))
    with pytest.raises(AssertionError, match="same dtype"):
        bootstrap_arrays(arrays=arrays, statistics={"mean": np.mean}, n_bootstrap=1, silent=True)


def test_bootstrap_arrays_predsamples_must_not_mix_with_numpy():
    gt = np.array([0, 1, 0], dtype=int)
    preds = np.array([0.1, 0.9, 0.2], dtype=float)
    w = PredSampleWrapper(preds, gt, multiclass=False, threshold=0.5)

    arrays = (w, np.array([1, 2, 3], dtype=float))
    with pytest.raises(AssertionError, match="PredSampleWrapper"):
        bootstrap_arrays(arrays=arrays, statistics={"mean": np.mean}, n_bootstrap=1, silent=True)


def test_bootstrap_arrays_predsamples_require_same_multiclass_setting():
    gt = np.array([0, 1, 0], dtype=int)
    preds_bin = np.array([0.1, 0.9, 0.2], dtype=float)
    preds_mc = np.array([[0.2, 0.8], [0.9, 0.1], [0.6, 0.4]], dtype=float)
    w_bin = PredSampleWrapper(preds_bin, gt, multiclass=False, threshold=0.5)
    w_mc = PredSampleWrapper(preds_mc, gt, multiclass=True)

    with pytest.raises(AssertionError, match="multiclass"):
        bootstrap_arrays(arrays=(w_bin, w_mc), statistics={"mean": np.mean}, n_bootstrap=1, silent=True)


def test_bootstrap_arrays_predsamples_require_same_threshold():
    gt = np.array([0, 1, 0, 1], dtype=int)
    preds = np.array([0.49, 0.51, 0.2, 0.8], dtype=float)
    w1 = PredSampleWrapper(preds, gt, multiclass=False, threshold=0.5)
    w2 = PredSampleWrapper(preds, gt, multiclass=False, threshold=0.7)

    with pytest.raises(AssertionError, match="threshold"):
        bootstrap_arrays(arrays=(w1, w2), statistics={"mean": np.mean}, n_bootstrap=1, silent=True)


def test_bootstrap_arrays_predsamples_require_same_groups():
    gt = np.array([0, 1, 0, 1], dtype=int)
    preds = np.array([0.49, 0.51, 0.2, 0.8], dtype=float)
    g1 = np.array([0, 0, 1, 1], dtype=int)
    g2 = np.array([0, 1, 0, 1], dtype=int)
    w1 = PredSampleWrapper(preds, gt, groups=g1, multiclass=False, threshold=0.5)
    w2 = PredSampleWrapper(preds, gt, groups=g2, multiclass=False, threshold=0.5)

    with pytest.raises(AssertionError, match="groups"):
        bootstrap_arrays(arrays=(w1, w2), statistics={"mean": np.mean}, n_bootstrap=1, silent=True)


def test_bootstrap_arrays_rejects_conflicting_external_and_embedded_groups():
    gt = np.array([0, 1, 0, 1], dtype=int)
    preds = np.array([0.49, 0.51, 0.2, 0.8], dtype=float)
    embedded_groups = np.array([0, 0, 1, 1], dtype=int)
    external_groups = np.array([0, 1, 0, 1], dtype=int)
    w1 = PredSampleWrapper(preds, gt, groups=embedded_groups, multiclass=False, threshold=0.5)
    w2 = PredSampleWrapper(preds, gt, groups=embedded_groups, multiclass=False, threshold=0.5)

    with pytest.raises(ValueError):
        bootstrap_arrays(arrays=(w1, w2), statistics={"mean": np.mean}, groups=external_groups, n_bootstrap=1, silent=True)


def test_bootstrap_arrays_accepts_valid_numpy_inputs():
    arrays = (np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))
    out = bootstrap_arrays(arrays=arrays, statistics={"mean": np.mean}, n_bootstrap=3, seed=0, silent=True)
    assert out["mean"].shape == (3, 2)


def test_bootstrap_arrays_accepts_valid_predsamples_inputs_and_uses_embedded_groups():
    gt = np.array([0, 1, 0, 1], dtype=int)
    preds1 = np.array([0.49, 0.51, 0.2, 0.8], dtype=float)
    preds2 = np.array([0.6, 0.7, 0.1, 0.9], dtype=float)
    groups = np.array([0, 0, 1, 1], dtype=int)
    w1 = PredSampleWrapper(preds1, gt, groups=groups, multiclass=False, threshold=0.5)
    w2 = PredSampleWrapper(preds2, gt, groups=groups, multiclass=False, threshold=0.5)

    out = bootstrap_arrays(arrays=(w1, w2), statistics={"mean": lambda s: float(np.mean(s.gt))},
                            n_bootstrap=3, seed=0, silent=True)
    assert out["mean"].shape == (3, 2)


def test_predsamplewrapper_getitem_slices_groups():
    gt = np.array([0, 1, 0, 1], dtype=int)
    preds = np.array([0.1, 0.9, 0.2, 0.8], dtype=float)
    groups = np.array([10, 10, 20, 20], dtype=int)
    w = PredSampleWrapper(preds, gt, groups=groups, multiclass=False, threshold=0.5)

    idx = np.array([0, 2, 3])
    sliced = w[idx]
    assert np.array_equal(sliced.groups, groups[idx])


def test_predsamplewrapper_getitem_preserves_none_groups():
    gt = np.array([0, 1, 0, 1], dtype=int)
    preds = np.array([0.1, 0.9, 0.2, 0.8], dtype=float)
    w = PredSampleWrapper(preds, gt, multiclass=False, threshold=0.5)

    sliced = w[np.array([0, 1])]
    assert sliced.groups is None
