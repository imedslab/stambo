import numpy as np

from stambo._predsamplewrapper import PredSampleWrapper
from stambo._stambo import bootstrap_arrays, compare_models
from stambo.metrics import LogOddsRatio


def _log_odds_ratio_from_labels(gt: np.ndarray, pred: np.ndarray) -> float:
    """Log odds ratio matching stambo.metrics.LogOddsRatio (epsilon-stabilized)."""
    gt = np.asarray(gt).astype(int)
    pred = np.asarray(pred).astype(int)
    counts = np.bincount(2 * gt + pred, minlength=4)
    tn, fp, fn, tp = counts
    eps = float(LogOddsRatio.epsilon)
    return float((np.log(tp + eps) + np.log(tn + eps)) - (np.log(fp + eps) + np.log(fn + eps)))


def _odds_from_counts(tp: int, tn: int, fp: int, fn: int, eps: float) -> float:
    """Odds ratio in exponentiated space, matching LogOddsRatio's epsilon stabilization."""
    return float(((tp + eps) * (tn + eps)) / ((fp + eps) * (fn + eps)))


def _pred_from_counts_for_fixed_gt(gt: np.ndarray, tp: int, fp: int) -> np.ndarray:
    """Construct predicted labels for a fixed gt with desired TP and FP counts.

    Assumes gt contains 1s then 0s (but we only rely on the indices).
    """
    gt = np.asarray(gt).astype(int)
    pos_idx = np.where(gt == 1)[0]
    neg_idx = np.where(gt == 0)[0]
    if tp > len(pos_idx):
        raise ValueError("tp exceeds number of positives in gt")
    if fp > len(neg_idx):
        raise ValueError("fp exceeds number of negatives in gt")

    pred = np.zeros_like(gt, dtype=int)
    pred[pos_idx[:tp]] = 1
    pred[neg_idx[:fp]] = 1
    return pred


def test_log_odds_ratio_bootstrap_uses_thresholded_predictions():
    # predictions are *probabilities*; the log-odds ratio must be computed on thresholded labels.
    gt = np.array([1, 1, 1, 0, 0, 0], dtype=int)
    probs = np.array([0.9, 0.8, 0.6, 0.4, 0.2, 0.1], dtype=float)  # threshold=0.5 -> [1,1,1,0,0,0]

    sample = PredSampleWrapper(probs, gt, multiclass=False, threshold=0.5)

    n_bootstrap = 50
    np.random.seed(123)
    out = bootstrap_arrays(
        arrays=(sample,),
        statistics={"LogOddsRatio": LogOddsRatio()},
        n_bootstrap=n_bootstrap,
        silent=True,
    )

    # Re-generate the same bootstrap indices and compute the expected statistic manually.
    np.random.seed(123)
    expected = np.zeros((n_bootstrap, 1), dtype=float)
    for b in range(n_bootstrap):
        ind = np.random.choice(len(sample), len(sample), replace=True)
        expected[b, 0] = _log_odds_ratio_from_labels(gt[ind], sample.predictions_am[ind])

    assert np.allclose(out["LogOddsRatio"], expected)


def test_log_odds_ratio_diff_matches_relative_odds_between_models():
    # Build two models with different confusion tables (via probabilities + threshold).
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=int)

    # model 1 predicted labels (thr=0.5): [1,1,0,0, 1,0,0,0]
    p1 = np.array([0.9, 0.8, 0.1, 0.2, 0.9, 0.1, 0.2, 0.4], dtype=float)
    # model 2 predicted labels (thr=0.5): [1,1,1,0, 0,0,0,0]
    p2 = np.array([0.9, 0.8, 0.7, 0.2, 0.4, 0.1, 0.2, 0.3], dtype=float)

    # We only need empirical values/diff; bootstrap size can be small.
    res = compare_models(
        y_test=y,
        preds_1=p1,
        preds_2=p2,
        metrics=("LogOddsRatio",),
        n_bootstrap=50,
        seed=2027,
        silent=True,
    )

    # Compute expected log ORs on thresholded predictions.
    pred1 = (p1 > 0.5).astype(int)
    pred2 = (p2 > 0.5).astype(int)
    log_or_1 = _log_odds_ratio_from_labels(y, pred1)
    log_or_2 = _log_odds_ratio_from_labels(y, pred2)
    expected_diff = log_or_2 - log_or_1  # log(relative odds)

    assert np.isclose(res["LogOddsRatio"]["emp_s1"], log_or_1)
    assert np.isclose(res["LogOddsRatio"]["emp_s2"], log_or_2)
    assert np.isclose(res["LogOddsRatio"]["diff"], expected_diff)

    # "Relative odds" (odds ratio between models) is exp(log OR2 - log OR1).
    relative_odds = float(np.exp(res["LogOddsRatio"]["diff"]))
    expected_relative_odds = float(np.exp(log_or_2) / np.exp(log_or_1))
    assert np.isclose(relative_odds, expected_relative_odds)


def test_log_odds_ratio_exact_odds_and_relative_odds_for_three_models_exponentiated():
    """
    "Clever" deterministic construction:
    - We pick a fixed gt with P positives and N negatives.
    - For each model, we construct *binary predicted labels* that realize exact (TP,FP) counts.
      This fixes (FN, TN) automatically since gt is fixed.
    - We then assert in *odds space* (exponentiated) that:
        exp(emp_s*) equals the model's odds ratio
        exp(diff) equals the relative odds ratio between models
    - Finally, we assert the test generalizes to multiple samples by checking all pairwise comparisons.
    """
    eps = float(LogOddsRatio.epsilon)

    # Fixed ground-truth: 40 positives, 60 negatives
    p = 40
    n = 60
    gt = np.array([1] * p + [0] * n, dtype=int)

    # Define 3 models by specifying TP and FP counts (with fixed gt these determine FN and TN).
    # All counts are strictly > 0 to avoid degenerate odds.
    models = {
        "A": {"tp": 30, "fp": 5},   # fn=10, tn=55
        "B": {"tp": 20, "fp": 10},  # fn=20, tn=50
        "C": {"tp": 35, "fp": 8},   # fn=5,  tn=52
    }

    samples = []
    expected_odds = {}
    expected_log_odds = {}
    for name, spec in models.items():
        tp = int(spec["tp"])
        fp = int(spec["fp"])
        fn = p - tp
        tn = n - fp
        pred = _pred_from_counts_for_fixed_gt(gt, tp=tp, fp=fp)
        samples.append(PredSampleWrapper(pred, gt, multiclass=False, threshold=0.5))

        expected_odds[name] = _odds_from_counts(tp=tp, tn=tn, fp=fp, fn=fn, eps=eps)
        expected_log_odds[name] = float(np.log(expected_odds[name]))

    # Deterministic bootstrap_results: constant per model, no RNG.
    emp = np.array([expected_log_odds["A"], expected_log_odds["B"], expected_log_odds["C"]], dtype=float)
    bootstrap_results = {"LogOddsRatio": np.tile(emp, (7, 1))}

    from stambo._stambo import pairwise_bootstrap_test

    res = pairwise_bootstrap_test(
        samples=tuple(samples),
        statistics={"LogOddsRatio": LogOddsRatio()},
        bootstrap_results=bootstrap_results,
        labels=("A", "B", "C"),
        adjusted_p_value=False,
        alpha=0.05,
    )

    # Generalizes to >2 samples: 3 models -> 3 pairwise comparisons.
    assert len(res["LogOddsRatio"]) == 3

    # Check each comparison in exponentiated space.
    # Labels follow "i / j" where j is the "improved" (second) model in the code.
    for label, out in res["LogOddsRatio"].items():
        left, right = [s.strip() for s in label.split("/")]
        odds_left = expected_odds[left]
        odds_right = expected_odds[right]

        assert np.isclose(float(np.exp(out["emp_s1"])), odds_left)
        assert np.isclose(float(np.exp(out["emp_s2"])), odds_right)
        assert np.isclose(float(np.exp(out["diff"])), odds_right / odds_left)

