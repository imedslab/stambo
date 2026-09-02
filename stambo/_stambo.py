import numpy as np
from typing import Any, Optional, Dict, Callable, Mapping, Tuple, Union, Sequence
import numpy.typing as npt

from ._utils import pbar
from ._predsamplewrapper import PredSampleWrapper
from .metrics import Metric

from . import metrics as metricslib


def _resolve_metrics(metrics: Tuple[Union[str, Metric], ...]) -> Dict[str, Metric]:
    r"""Turns a tuple of metric names / ``Metric`` instances into a ``{tag: Metric instance}`` dict."""
    metrics_dict = {}
    for metric in metrics:
        if isinstance(metric, Metric):
            metrics_dict[str(metric)] = metric  # note that the object must be instantiated
        elif isinstance(metric, str):
            assert hasattr(metricslib, metric), f"Metric {metric} is not defined"
            metrics_dict[metric] = getattr(metricslib, metric)()
    return metrics_dict


def _bootstrap_pair_result(diff_array: npt.NDArray[np.float64], boot_1: npt.NDArray[np.float64], boot_2: npt.NDArray[np.float64],
                            observed: float, emp_1: float, emp_2: float, alpha: float) -> Dict[str, Any]:
    r"""Two-tailed percentile bootstrap test and confidence intervals for one pairwise comparison.

    This is the single place where the p-value/CI math lives, shared by :func:`two_sample_test`
    and :func:`pairwise_bootstrap_test`, so that a fix (or a bug) only has to happen in one place.

    Args:
        diff_array: Bootstrap distribution of ``boot_2 - boot_1`` (non-shifted).
        boot_1: Bootstrap distribution of the statistic on sample 1.
        boot_2: Bootstrap distribution of the statistic on sample 2.
        observed: The empirical (non-bootstrapped) difference ``emp_2 - emp_1``.
        emp_1: The empirical (non-bootstrapped) statistic for sample 1.
        emp_2: The empirical (non-bootstrapped) statistic for sample 2.
        alpha: A significance level for confidence intervals (from 0 to 1).

    Returns:
        A dict with keys ``p_value``, ``diff``, ``ci_es``, ``ci_s1``, ``ci_s2``, ``emp_s1``, ``emp_s2``.
    """
    n_bootstrap = len(diff_array)
    # Both tails are computed directly (each with its own continuity correction) rather than
    # deriving one from the other via `1 - p`. That matters at the boundary: if the two samples
    # are identical, diff_array is a point mass at 0, and *both* P(diff <= 0) and P(diff >= 0)
    # equal 1, correctly giving a two-tailed p-value of 1. Deriving the left tail as
    # `1 - P(diff <= 0)` would instead (wrongly) give 0 in that case.
    p_right = ((diff_array <= 0).sum() + 1.) / (n_bootstrap + 1)
    p_left = ((diff_array >= 0).sum() + 1.) / (n_bootstrap + 1)
    p_val = min(2. * min(p_right, p_left), 1.0)

    alpha_pct = 100. * alpha
    ci_es = (np.percentile(diff_array, alpha_pct / 2.), np.percentile(diff_array, 100. - alpha_pct / 2.))
    ci_1 = (np.percentile(boot_1, alpha_pct / 2.), np.percentile(boot_1, 100. - alpha_pct / 2.))
    ci_2 = (np.percentile(boot_2, alpha_pct / 2.), np.percentile(boot_2, 100. - alpha_pct / 2.))

    return {
        "p_value": p_val,
        "diff": observed,
        "ci_es": ci_es,
        "ci_s1": ci_1,
        "ci_s2": ci_2,
        "emp_s1": emp_1,
        "emp_s2": emp_2,
    }


def holm_bonferroni_correction(p_values: Union[Sequence[float], npt.NDArray[np.float64]]) -> npt.NDArray[np.float64]:
    r"""Holm-Bonferroni step-down correction for multiple comparisons.

    Given :math:`m` p-values from a family of hypothesis tests, adjusts them so that the
    family-wise error rate (the probability of at least one false positive across the whole
    family) is controlled at the nominal level, while being less conservative than a plain
    Bonferroni correction (dividing every p-value by :math:`m`).

    Sort the p-values ascending, :math:`p_{(1)} \leq \dots \leq p_{(m)}`. The adjusted p-value
    for the :math:`k`-th smallest (1-indexed) is

    .. math::
        \tilde p_{(k)} = \max_{l \leq k} \min\left(1, (m - l + 1) \, p_{(l)}\right),

    i.e. each p-value is multiplied by the number of remaining hypotheses at its rank, capped at
    1, and then enforced to be non-decreasing (monotone) with rank.

    Args:
        p_values: A 1D sequence of raw p-values.

    Returns:
        An array of the same length and order as ``p_values``, with the Holm-adjusted p-values.
    """
    p_values = np.asarray(p_values, dtype=float)
    if p_values.ndim != 1:
        raise ValueError("p_values must be a 1D array-like of p-values")
    m = len(p_values)
    if m == 0:
        return p_values.copy()

    order = np.argsort(p_values, kind="stable")
    p_sorted = p_values[order]

    adjusted_sorted = np.empty(m, dtype=float)
    running_max = 0.0
    for k in range(m):
        adjusted = min(p_sorted[k] * (m - k), 1.0)
        adjusted = max(adjusted, running_max)
        running_max = adjusted
        adjusted_sorted[k] = adjusted

    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def bootstrap_arrays(arrays: Tuple[Union[npt.NDArray[np.int_], npt.NDArray[np.float64], PredSampleWrapper], ...],
                     statistics: Mapping[str, Callable],
                     groups: Optional[npt.NDArray[np.int_]]=None,
                     n_bootstrap: int=5000,
                     seed: Optional[int]=None,
                     silent: bool=False) -> Dict[str, npt.NDArray[np.float64]]:
    r"""Bootstraps a tuple of mutually paired samples (e.g. predictions from :math:`N \geq 2` models
    evaluated on the same test set).

    All samples are resampled with the *same* indices at every bootstrap iteration (a paired
    design), optionally resampling whole groups/clusters instead of individual rows when
    ``groups`` is provided (see :func:`two_sample_test` for the rationale). This is the building
    block used by :func:`pairwise_bootstrap_test`, but it is also useful on its own if you want
    the raw bootstrap distributions for :math:`N` samples at once (e.g. to compute your own
    downstream statistic across all of them jointly).

    Args:
        arrays: A tuple of :math:`N \geq 2` samples to bootstrap together. Every array must have the
            same length, and either all be ``numpy.ndarray`` (with the same dtype) or all be
            ``PredSampleWrapper`` (with the same ``multiclass``, ``threshold``, and ``groups``).
        statistics: A dictionary of statistics to compute on each sample.
        groups: Groups indicating the subject for each measurement. Defaults to None. Must not be
            passed if the samples are ``PredSampleWrapper`` objects that already carry ``groups``.
        n_bootstrap: The number of bootstrap iterations. Defaults to 5000.
        seed: Random seed. Defaults to None.
        silent: Whether to execute the function silently, i.e. not showing the progress bar. Defaults to False.

    Returns:
        A dictionary keyed by statistic tag, each holding an array of shape ``(n_bootstrap, N)``
        with the bootstrap replicates of that statistic for every sample.
    """
    if seed is not None:
        np.random.seed(seed)

    def _same_groups(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> bool:
        if a is None and b is None:
            return True
        if (a is None) != (b is None):
            return False
        assert a is not None and b is not None
        return bool(np.array_equal(a, b))

    assert len(arrays) >= 2, "At least two arrays are required"
    if not all(isinstance(arr, (np.ndarray, PredSampleWrapper)) for arr in arrays):
        raise ValueError("All arrays must be numpy.ndarray or PredSampleWrapper")

    arr_lengths = np.array([len(arr) for arr in arrays])
    assert np.all(arr_lengths == arr_lengths[0]), "All arrays must have the same length"

    first = arrays[0]
    if isinstance(first, PredSampleWrapper):
        assert all(isinstance(arr, PredSampleWrapper) and arr.multiclass == first.multiclass for arr in arrays), \
            "All arrays must be of type PredSampleWrapper, with the same multiclass setting"
        assert all(isinstance(arr, PredSampleWrapper) and arr.threshold == first.threshold for arr in arrays), \
            "All PredSampleWrapper arrays must have the same threshold"
        assert all(isinstance(arr, PredSampleWrapper) and _same_groups(arr.groups, first.groups) for arr in arrays), \
            "All PredSampleWrapper arrays must carry the same groups"
        if groups is not None and first.groups is not None:
            raise ValueError("`groups` was passed explicitly, but the PredSampleWrapper samples already carry `groups`. Pass only one of the two.")
        elif first.groups is not None:
            groups = first.groups
    else:
        assert all(isinstance(arr, np.ndarray) and arr.dtype == first.dtype for arr in arrays), \
            "All arrays must be of type numpy.ndarray, with the same dtype"

    if groups is not None:
        assert len(groups) == arr_lengths[0], "Groups must be of the same length as the samples"
        groups_ids = np.unique(groups)
        group_data = {group_id: np.where(groups == group_id)[0] for group_id in groups_ids}
    else:
        group_data = None

    n_samples = len(arrays)
    result = {s_tag: np.zeros((n_bootstrap, n_samples)) for s_tag in statistics}
    for bootstrap_iter in pbar(range(n_bootstrap), total=n_bootstrap, desc="Bootstrapping", silent=silent):
        if group_data is None:
            ind = np.random.choice(arr_lengths[0], arr_lengths[0], replace=True)
        else:
            groups_ind = np.random.choice(groups_ids, len(groups_ids), replace=True)
            ind = np.concatenate([group_data[group_id] for group_id in groups_ind])

        for sample_idx, arr in enumerate(arrays):
            arr_resampled = arr[ind]
            for s_tag in statistics:
                result[s_tag][bootstrap_iter, sample_idx] = statistics[s_tag](arr_resampled)

    return result


def pairwise_bootstrap_test(samples: Tuple[Union[npt.NDArray[np.int_], npt.NDArray[np.float64], PredSampleWrapper], ...],
                            statistics: Mapping[str, Callable],
                            groups: Optional[npt.NDArray[np.int_]]=None,
                            labels: Optional[Tuple[str, ...]]=None,
                            bootstrap_results: Optional[Dict[str, npt.NDArray[np.float64]]]=None,
                            alpha: float=0.05,
                            n_bootstrap: int=5000,
                            correction: Optional[str]="holm",
                            seed: Optional[int]=None,
                            silent: bool=False) -> Dict[str, Dict[str, Dict[str, float]]]:
    r"""Runs the two-tailed bootstrap test from :func:`two_sample_test` on every pair among
    :math:`N \geq 2` mutually paired samples (e.g. :math:`N` models evaluated on the same test set).

    For :math:`N` samples there are :math:`N (N - 1) / 2` pairwise comparisons. Testing many pairs
    inflates the family-wise Type I error rate (the more pairs you test, the more likely *some*
    pair looks "significant" purely by chance), so by default (``correction="holm"``) a
    Holm-Bonferroni correction (see :func:`holm_bonferroni_correction`) is applied *per statistic*,
    treating the :math:`N (N - 1) / 2` comparisons for that statistic as one family. Pass
    ``correction=None`` to disable this and only get the raw, uncorrected p-values.

    Args:
        samples: A tuple of :math:`N \geq 2` mutually paired samples to compare.
        statistics: A dictionary of statistics to compute on each sample.
        groups: Groups indicating the subject for each measurement. Defaults to None.
        labels: Labels for the samples, used to name each comparison as ``"{label_i} / {label_j}"``.
            Defaults to ``("0", "1", ..., str(N-1))``.
        bootstrap_results: Precomputed output of :func:`bootstrap_arrays` for these exact
            ``samples`` (e.g. to reuse the same bootstrap draws for a corrected and an
            uncorrected call without resampling twice). If None (the default), it is computed
            internally via :func:`bootstrap_arrays`.
        alpha: A significance level for confidence intervals (from 0 to 1). Defaults to 0.05.
        n_bootstrap: The number of bootstrap iterations. Defaults to 5000.
        correction: Multiple-comparison correction to apply to the p-values, or None to disable it.
            Only ``"holm"`` is currently supported. Defaults to ``"holm"``.
        seed: Random seed. Defaults to None.
        silent: Whether to execute the function silently, i.e. not showing the progress bar. Defaults to False.

    Returns:
        A dictionary keyed by statistic tag, then by comparison label (``"{label_i} / {label_j}"``),
        holding a dict with keys ``p_value`` (raw, uncorrected), ``p_value_adjusted`` (the
        Holm-adjusted p-value, or None if ``correction`` is None), ``diff`` (the observed
        effect size), ``ci_es``/``ci_s1``/``ci_s2`` (percentile confidence intervals for the
        effect size and for each of the two samples in the pair), and ``emp_s1``/``emp_s2`` (the
        empirical statistic for each of the two samples in the pair).
    """
    n_samples = len(samples)
    assert n_samples >= 2, "At least two samples are required for a pairwise comparison"
    if labels is None:
        labels = tuple(str(i) for i in range(n_samples))
    assert len(labels) == n_samples, "The number of labels must match the number of samples"
    assert len(set(labels)) == n_samples, "Labels must be unique"
    assert correction in (None, "holm"), f"Unsupported correction method: {correction!r}. Only 'holm' (or None) is currently supported."

    if bootstrap_results is None:
        bootstrap_results = bootstrap_arrays(arrays=samples, statistics=statistics, groups=groups,
                                              n_bootstrap=n_bootstrap, seed=seed, silent=silent)
    else:
        assert len(bootstrap_results) == len(statistics), "The number of bootstrap results must match the number of statistics"
        assert all(s_tag in bootstrap_results for s_tag in statistics), "All statistics must be present in the precomputed bootstrap results"

    result_final: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for s_tag in statistics:
        result_final[s_tag] = {}
        emp = [statistics[s_tag](sample) for sample in samples]
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                label = f"{labels[i]} / {labels[j]}"
                boot_1 = bootstrap_results[s_tag][:, i]
                boot_2 = bootstrap_results[s_tag][:, j]
                diff_array = boot_2 - boot_1
                observed = emp[j] - emp[i]
                pair_res = _bootstrap_pair_result(diff_array, boot_1, boot_2, observed, emp[i], emp[j], alpha=alpha)
                pair_res["p_value_adjusted"] = None
                result_final[s_tag][label] = pair_res

        if correction == "holm":
            comparison_labels = list(result_final[s_tag].keys())
            p_values = np.array([result_final[s_tag][label]["p_value"] for label in comparison_labels])
            p_adjusted = holm_bonferroni_correction(p_values)
            for label, p_adj in zip(comparison_labels, p_adjusted):
                result_final[s_tag][label]["p_value_adjusted"] = float(p_adj)

    return result_final


def two_sample_test(sample_1: Union[npt.NDArray[np.int_], npt.NDArray[np.float64], PredSampleWrapper], 
                    sample_2: Union[npt.NDArray[np.int_], npt.NDArray[np.float64], PredSampleWrapper], 
                    statistics: Mapping[str, Callable], 
                    groups: Optional[npt.NDArray[np.int_]]=None,
                    alpha: float=0.05, 
                    n_bootstrap: int=5000, seed: Optional[int]=None,
                    non_paired: bool=False,
                    silent: bool=False) -> Dict[str, npt.NDArray[np.float64]]:
    r"""Compares whether the empirical difference of statistics computed on two samples is statistically significant or not.

    The hypotheses we test are:

    .. math::

        H_0: f(x_1) = f(x_2) \\
        H_1: f(x_1) \neq f(x_2),

    where :math:`f` is a function of interest, and :math:`x_1` and :math:`x_2` are the samples to be compared.
    Note that the statistics are computed independently, and should thus be treated independently.


    Args:
        sample_1: Sample 1 to be compared.
        sample_2: Sample 2 to be compared.
        groups: Groups indicating the subject for each measurement. Defaults to None.
        statistics: Statistics to compare the samples by.
        alpha: A significance level for confidence intervals (from 0 to 1).
        n_bootstrap: The number of bootstrap iterations. Defaults to 5000.
        non_paired: Whether to use a non-paired design. Defaults to False.
        seed: Random seed. Defaults to None.
        silent: Whether to execute the function silently, i.e. not showing the progress bar. Defaults to False.

    Returns:
        A dictionary containing a tuple with the empirical value of
        the metric, and the p-value. Each entry in the dictionary contains, in order:

            * Two-tailed :math:`p`-value, :math:`p(\texttt{data} \mid H_0)`
            * Observed difference (effect size)
            * CI low (effect size)
            * CI high (effect size)
            * Empirical value (sample 1)
            * CI low (sample 1)
            * CI high (sample 1)
            * Empirical value (sample 2)
            * CI low (sample 2)
            * CI high (sample 2)
    """
    
    if seed is not None:
        np.random.seed(seed)

    # Dict to store the null bootstrap distribution
    result = {s_tag: np.zeros((n_bootstrap, 2)) for s_tag in statistics}
    
    if groups is not None:
        assert len(groups) == len(sample_1), "Groups must be of the same length as the samples"
        assert len(groups) == len(sample_2), "Groups must be of the same length as the samples"
        assert not non_paired, "Paired design must be used when groups are provided"
        
    group_data = {}
    if groups is not None:
        groups_ids = np.unique(groups)
        for group_id in groups_ids:
            group_data[group_id] = {"indices": np.where(groups == group_id)[0]}

    for bootstrap_iter in pbar(range(n_bootstrap), total=n_bootstrap, desc="Bootstrapping", silent=silent):
        # We are here in a paired design, so we need to sample the same indices for both samples
        if groups is None:
            ind = np.random.choice(len(sample_1), len(sample_1), replace=True)
            ind1 = ind
            ind2 = ind
            if non_paired:
                ind2 = np.random.choice(len(sample_2), len(sample_2), replace=True)
        else:
            # When we have groups, we need to sample them with replacement
            groups_ind = np.random.choice(groups_ids, len(groups_ids), replace=True)
            # Once the groups are sampled, we can concatenate the indices
            ind = np.concatenate([group_data[grp]["indices"] for grp in groups_ind])
            ind1 = ind
            ind2 = ind
            
        for s_tag in statistics:
            result[s_tag][bootstrap_iter, 0] = statistics[s_tag](sample_1[ind1])
            result[s_tag][bootstrap_iter, 1] = statistics[s_tag](sample_2[ind2])
    
    result_final = {}
    for s_tag in result:
        emp_s1 = statistics[s_tag](sample_1)
        emp_s2 = statistics[s_tag](sample_2) 

        # Observed difference: Delta
        observed = emp_s2 - emp_s1
        diff_array = result[s_tag][:, 1] - result[s_tag][:, 0]
        # The p-value/CI math (two-tailed percentile bootstrap test) lives in
        # `_bootstrap_pair_result`, shared with `pairwise_bootstrap_test`. See its
        # docstring for why both tails are computed directly rather than via `1 - p`.
        pair_res = _bootstrap_pair_result(diff_array, result[s_tag][:, 0], result[s_tag][:, 1],
                                           observed, emp_s1, emp_s2, alpha=alpha)
        # And we report the p-value, empirical values, as well as the confidence intervals.
        # The format in the documentation.
        result_final[s_tag] = np.array([
            pair_res["p_value"], pair_res["diff"], pair_res["ci_es"][0], pair_res["ci_es"][1],
            pair_res["emp_s1"], pair_res["ci_s1"][0], pair_res["ci_s1"][1],
            pair_res["emp_s2"], pair_res["ci_s2"][0], pair_res["ci_s2"][1],
        ])
    return result_final


def compare_models(y_test: Union[npt.NDArray[np.int_], npt.NDArray[np.float64]], 
                   preds_1: Union[npt.NDArray[np.int_], npt.NDArray[np.float64]], 
                   preds_2: Union[npt.NDArray[np.int_], npt.NDArray[np.float64]], 
                   metrics: Tuple[Union[str, Metric]],
                   groups: Optional[npt.NDArray[np.int_]]=None,
                   alpha: float=0.05, 
                   n_bootstrap: int=5000, 
                   seed: Optional[int]=None, 
                   silent: bool=False) -> Dict[str, npt.NDArray[np.float64]]:
    r"""Compares predictions from two models :math:`f_1(x)` and :math:`f_2(x)` that yield prediction vectors  :math:`\hat y_{1}` and :math:`\hat y_{2}`
    with a two-tailed bootstrap hypothesis test.

    I.e., we state the following null and alternative hypotheses:

    .. math::
        H_0: M(y_{gt}, \hat y_{1}) = M(y_{gt}, \hat y_{2})

        H_1: M(y_{gt}, \hat y_{1}) \neq M(y_{gt}, \hat y_{2}),

    where :math:`M` is a metric, :math:`y_{gt}` is the vector of ground truth labels,
    and :math:`\hat y_{i}, i=1,2` are the vectors of predictions for model 1 and 2, respectively.
    Such kind of testing is performed for every specified metric.

    Since the test is two-tailed, the :math:`p`-value does not depend on which model is passed as model 1 vs. model 2, or on
    whether the metric is defined as more-is-better or less-is-better. The sign of the reported effect size
    (:math:`M(y_{gt}, \hat y_{2}) - M(y_{gt}, \hat y_{1})`) tells you which model scored higher on the metric as passed.

    While the test does return you the :math:`p`-value, one should be careful about its interpretation: the :math:`p`-value
    is the probability of observing the test statistic *at least as extreme* as the one obtained assuming that :math:`H_0` is true (probability of Type II error).
    With large data, even small effects can be statistically significant, so one should consider the effect size.
    
    We compute a standardized effect size using the estimated bootstrap variance.

    Beyond the hypothesis testing, the function also returns confidence intervals per metric, i.e. 
    
    .. math::
        P\left(M(y_{gt,*}, \hat y_*) \in [L_{CI}(\alpha), H_{CI}(\alpha)]\right) = 1 - \alpha,

    where :math:`L` and :math:`H` are the lower and upper bounds of the confidence interval, respectively, and :math:`\alpha` is the significance level,
    and :math:`*` indicates that the metric is computed on infinite data.

    At this moment, 
    the confidence intervals are computed using the simple percentile method. In the future, we will implement the BCa approach, which is more accurate.
    
    Args:
        y_test: Ground truth.
        preds_1: Prediction from model 1.
        preds_2: Prediction from model 2.
        metrics: A set of metrics to call. Here, the user either specifies the metrics available from the stambo library (``stambo.metrics``), or adds an instance of the custom-defined metrics.
        groups: Groups indicating the subject for each measurement. Defaults to None.
        alpha: A significance level for confidence intervals (from 0 to 1). Defaults to 0.05.
        n_bootstrap: The number of bootstrap iterations. Defaults to 5000.
        seed: Random seed. Defaults to None.
        silent: Whether to execute the function silently, i.e. not showing the progress bar. Defaults to False.

    Returns:
        A dictionary containing a tuple with the empirical value of
        the metric, and the two-tailed p-value. The expected format in the output in
        every dict entry is:

            * Two-tailed :math:`p`-value
            * Observed difference (effect size)
            * Effect size CI low
            * Effect size CI high
            * :math:`M(y_{gt}, \hat y_{1})`
            * :math:`M(y_{gt}, \hat y_{1})_{(\alpha / 2)}`
            * :math:`M(y_{gt}, \hat y_{1})_{(1 - \alpha / 2)}`
            * :math:`M(y_{gt}, \hat y_{2})`
            * :math:`M(y_{gt}, \hat y_{2})_{(\alpha / 2)}`
            * :math:`M(y_{gt}, \hat y_{2})_{(1 - \alpha / 2)}`
    """

    # Data samples need to be prepared
    sample_1 = PredSampleWrapper(preds_1, y_test, multiclass=len(preds_1.shape) != 1)
    sample_2 = PredSampleWrapper(preds_2, y_test, multiclass=len(preds_2.shape) != 1)

    metrics_dict = _resolve_metrics(metrics)

    test_results = two_sample_test(sample_1, sample_2, statistics=metrics_dict, groups=groups, alpha=alpha, n_bootstrap=n_bootstrap, seed=seed, non_paired=False, silent=silent)
    output = {}
    for metric in test_results:
        output[metric] = test_results[metric]
    return output


def compare_models_pairwise(y_test: Union[npt.NDArray[np.int_], npt.NDArray[np.float64]],
                            preds: Tuple[Union[npt.NDArray[np.int_], npt.NDArray[np.float64]], ...],
                            metrics: Tuple[Union[str, Metric]],
                            labels: Optional[Tuple[str, ...]]=None,
                            groups: Optional[npt.NDArray[np.int_]]=None,
                            alpha: float=0.05,
                            n_bootstrap: int=5000,
                            correction: Optional[str]="holm",
                            seed: Optional[int]=None,
                            silent: bool=False) -> Dict[str, Dict[str, Dict[str, float]]]:
    r"""Compares predictions from :math:`N \geq 2` models pairwise, i.e. the many-model generalization of
    :func:`compare_models`.

    For :math:`N` models there are :math:`N (N - 1) / 2` pairwise comparisons per metric. This
    function is a thin wrapper that builds a :class:`~stambo.PredSampleWrapper` for each model's
    predictions (exactly like :func:`compare_models` does for two models) and then runs
    :func:`pairwise_bootstrap_test` across all of them. Because testing many pairs inflates the
    family-wise Type I error rate, the p-values are Holm-Bonferroni corrected per metric by
    default (``correction="holm"``); pass ``correction=None`` to disable this.

    Args:
        y_test: Ground truth.
        preds: A tuple of prediction arrays, one per model, all evaluated on the same ``y_test``.
        metrics: A set of metrics to call. Here, the user either specifies the metrics available from the stambo library (``stambo.metrics``), or adds an instance of the custom-defined metrics.
        labels: Labels for the models, used to name each comparison as ``"{label_i} / {label_j}"``.
            Defaults to ``("0", "1", ..., str(N-1))``.
        groups: Groups indicating the subject for each measurement. Defaults to None.
        alpha: A significance level for confidence intervals (from 0 to 1). Defaults to 0.05.
        n_bootstrap: The number of bootstrap iterations. Defaults to 5000.
        correction: Multiple-comparison correction to apply to the p-values, or None to disable it.
            Only ``"holm"`` is currently supported. Defaults to ``"holm"``.
        seed: Random seed. Defaults to None.
        silent: Whether to execute the function silently, i.e. not showing the progress bar. Defaults to False.

    Returns:
        Same format as :func:`pairwise_bootstrap_test`: a dictionary keyed by metric tag, then by
        comparison label, holding a dict with keys ``p_value``, ``p_value_adjusted``, ``diff``,
        ``ci_es``, ``ci_s1``, ``ci_s2``, ``emp_s1``, and ``emp_s2``.
    """
    assert len(preds) >= 2, "At least two models (i.e. two prediction arrays) are required"
    samples = tuple(PredSampleWrapper(p, y_test, multiclass=len(p.shape) != 1) for p in preds)
    metrics_dict = _resolve_metrics(metrics)

    return pairwise_bootstrap_test(samples=samples, statistics=metrics_dict, groups=groups, labels=labels,
                                    alpha=alpha, n_bootstrap=n_bootstrap, correction=correction, seed=seed, silent=silent)
    