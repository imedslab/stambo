import numpy as np
from typing import Optional, Dict, Callable, Tuple, Union
import numpy.typing as npt

from ._utils import pbar
from ._predsamplewrapper import PredSampleWrapper
from .metrics import Metric

from . import metrics as metricslib

def bootstrap_arrays(arrays: tuple[Union[npt.NDArray[int], npt.NDArray[float], PredSampleWrapper]],
                     statistics: Dict[str, Callable],
                     groups: Optional[npt.NDArray[int]]=None,
                     n_bootstrap: int=5000,
                     silent: bool=False) -> tuple[Union[npt.NDArray[int], npt.NDArray[float], PredSampleWrapper]]:
    r"""
    Bootstraps an array of mutually paired samples.
    Perfect if we want to establish how a set of models are related to each other.

    Args:
        arrays: A tuple of arrays to bootstrap. Each array is a sample.
        statistics: A dictionary of statistics to compute.
        groups: Groups indicating the subject for each measurement. Defaults to None.
        n_bootstrap: The number of bootstrap iterations. Defaults to 5000.
        silent: Whether to execute the function silently, i.e. not showing the progress bar. Defaults to False.

    Returns:
        A dictionary of statistics values for each bootstrap iteration and each sample.
    """

    arr_lengths = np.array([len(arr) for arr in arrays])
    assert np.all(arr_lengths == arr_lengths[0]), "All arrays must have the same length"

    if groups is not None:
        groups_ids = np.unique(groups)
        group_data = {}
        for group_id in groups_ids:
            group_data[group_id] = {"indices": np.where(groups == group_id)[0]}
    else:
        group_data = None

    result = {s_tag: np.zeros((n_bootstrap, len(arrays))) for s_tag in statistics}
    for bootstrap_iter in pbar(range(n_bootstrap), total=n_bootstrap, desc="Bootstrapping", silent=silent):
        # We are here in a paired design, so we need to sample the same indices for both samples
        if group_data is None:
            ind = np.random.choice(arr_lengths[0], arr_lengths[0], replace=True)
        else:
            # When we have groups, we need to sample them with replacement
            groups_ind = np.random.choice(groups_ids, len(groups_ids), replace=True)
            # Once the groups are sampled, we can concatenate the indices
            ind = np.concatenate([group_data[grp]["indices"] for grp in groups_ind])
            
        for s_tag in statistics:
            for sample_idx in range(len(arrays)):
                v = statistics[s_tag](arrays[sample_idx][ind])
                result[s_tag][bootstrap_iter, sample_idx] = v

    return result


def compute_bootstrap_model_test(
    bootstrap_results: Dict[str, npt.NDArray[float]],   
    samples: tuple[Union[npt.NDArray[int], npt.NDArray[float], PredSampleWrapper]],
    i: int,
    j: int,
    statistic: str,
    statistics_dict: Dict[str, Callable],
    alpha: float=0.05) -> Dict[str, Tuple[float]]:
    r"""Computes the bootstrap test for a model comparison.
    """
    n_bootstrap_i = len(bootstrap_results[statistic][:, i])
    n_bootstrap_j = len(bootstrap_results[statistic][:, j])
    n_bootstrap = n_bootstrap_i
    # Some sanity checks
    assert n_bootstrap_i == n_bootstrap_j, "The number of bootstrap samples must be the same for both models"
    assert n_bootstrap > 0, "The number of bootstrap samples must be greater than 0"
    assert i < len(samples), "The index i must be less than the number of samples"
    assert j < len(samples), "The index j must be less than the number of samples"
    assert i != j, "The index i and j must be different"
    assert statistic in bootstrap_results, "The statistic must be in the bootstrap results"
    assert statistic in statistics_dict, "The statistic must be in the statistics dictionary"
    sample_1_b = bootstrap_results[statistic][:, i]
    sample_2_b = bootstrap_results[statistic][:, j]

    emp_s1 = statistics_dict[statistic](samples[i])
    emp_s2 = statistics_dict[statistic](samples[j]) 

    # Observed difference: Delta
    observed = emp_s2 - emp_s1
    diff_array = sample_2_b - sample_1_b
    # Generating the null
    # Model 2 != Model 1 is the alternative hypothesis in two-tailed test
    null = diff_array - observed
    p_val_right = ((null >= observed).sum() + 1.) / (n_bootstrap + 1)
    p_val_left = ((null <= observed).sum() + 1.) / (n_bootstrap + 1)
    p_val = 2 * min(p_val_right, p_val_left)
    # Numerical / finite-sample guard: two-tailed doubling can exceed 1.0
    if p_val > 1.0:
        p_val = 1.0
    # Compute the effect size
    # We also want to compute the confidence intervals
    # In this version of STAMBO, we use the simple percentile method
    ci_es = (np.percentile(diff_array, alpha / 2.), np.percentile(diff_array, 100 - alpha / 2.))
    ci_s1 = (np.percentile(sample_1_b, alpha / 2.), np.percentile(sample_1_b, 100 - alpha / 2.))
    ci_s2 = (np.percentile(sample_2_b, alpha / 2.), np.percentile(sample_2_b, 100 - alpha / 2.))

    return {
        "p_value": p_val,
        "diff": observed,
        "ci_es": ci_es,
        "ci_s1": ci_s1, 
        "ci_s2": ci_s2,
        "emp_s1": emp_s1,
        "emp_s2": emp_s2,
        "statistic": statistic
    }

def pairwise_bootstrap_test(
    bootstrap_results: Dict[str, npt.NDArray[float]],   
    samples: tuple[Union[npt.NDArray[int], npt.NDArray[float], PredSampleWrapper]],
    statistics: Dict[str, Callable],
    labels: Optional[Tuple[str, str]]=None,
    adjusted_p_value: bool=False,
    alpha: float=0.05) -> Dict[str, Tuple[float]]:
    r"""
        Performs a pairwise bootstrap test to compare the statistics of the bootstrap results.
        Note: if N samples are compared, there will be N*(N-1)/2 comparisons.
        When N samples are compared, the p-values are adjusted using the Bonforroni-Holm correction.

        Args:
            bootstrap_results: A dictionary of bootstrap results.
            samples: A tuple of samples.
            statistics: A dictionary of statistics to compute.
            labels: A tuple of labels for the samples. Defaults to None.
            adjusted_p_value: Whether to adjust the p-value for multiple testing. Defaults to True. 
            alpha: A significance level for confidence intervals (from 0 to 1). Defaults to 0.05.

        Returns:
            A dictionary of statistics values for each bootstrap iteration and each sample. 
            If adjusted_p_value is True, the p-values are adjusted.
    """
    result_final = {}
    arr_lengths = np.array([len(arr) for arr in samples])
    if labels is None:
        labels = [f"Model {i}" for i in range(len(samples))]
    assert len(labels) == len(samples), "The number of labels must be the same as the number of samples"
    assert np.all(arr_lengths == arr_lengths[0]), "All arrays must have the same length"
    p_val_array = []
    comparisons_array = []
    s_tags_array = []
    # Going over statistics
    for s_tag in bootstrap_results:
        result_final[s_tag] = {}
        for i in range(len(samples)):
            for j in range(i+1, len(samples)):
                # We always take the second model as the improved
                label = f"{labels[i]} / {labels[j]}"

                result_final[s_tag][label] = {}
                
                # And we report the p-value, empirical values, as well as the confidence intervals. 
                # The format in the documentation.
                b_res = compute_bootstrap_model_test(
                    bootstrap_results=bootstrap_results, 
                    samples=samples, 
                    i=i, j=j, 
                    statistic=s_tag, 
                    statistics_dict=statistics, 
                    alpha=alpha
                )

                result_final[s_tag][label]["p_value"] = b_res["p_value"]
                result_final[s_tag][label]["diff"] = b_res["diff"]
                result_final[s_tag][label]["ci_es"] = b_res["ci_es"]
                result_final[s_tag][label]["ci_s1"] = b_res["ci_s1"]
                result_final[s_tag][label]["ci_s2"] = b_res["ci_s2"]
                result_final[s_tag][label]["emp_s1"] = b_res["emp_s1"]
                result_final[s_tag][label]["emp_s2"] = b_res["emp_s2"]

                p_val_array.append(b_res["p_value"])
                comparisons_array.append(label)
                s_tags_array.append(s_tag)

    if adjusted_p_value:
        # Holm-Bonferroni step-down correction across *all* performed tests.
        # We adjust and write back into the nested dict structure under the "p_value" key.
        if len(p_val_array) > 1:
            p_vals = np.asarray(p_val_array, dtype=float)
            comparisons = np.asarray(comparisons_array, dtype=str)
            s_tags = np.asarray(s_tags_array, dtype=str)

            m = len(p_vals)
            order = np.argsort(p_vals)  # increasing p-values

            p_sorted = p_vals[order]
            comparisons_sorted = comparisons[order]
            s_tags_sorted = s_tags[order]

            # Holm adjusted p-values: p_(k) * (m - k), with monotonicity enforcement.
            p_adj_sorted = np.empty_like(p_sorted)
            running_max = 0.0
            for k in range(m):
                factor = (m - k)
                adj = p_sorted[k] * factor
                if adj > 1.0:
                    adj = 1.0
                if adj < running_max:
                    adj = running_max
                running_max = adj
                p_adj_sorted[k] = adj

            for k in range(m):
                result_final[s_tags_sorted[k]][comparisons_sorted[k]]["p_value"] = float(p_adj_sorted[k])
    return result_final

def two_sample_test(sample_1: Union[npt.NDArray[int], npt.NDArray[float], PredSampleWrapper], 
                    sample_2: Union[npt.NDArray[int], npt.NDArray[float], PredSampleWrapper], 
                    statistics: Dict[str, Callable], 
                    groups: Optional[npt.NDArray[int]]=None,
                    alpha: float=0.05, 
                    n_bootstrap: int=5000, seed: int=None, 
                    non_paired: bool=False,
                    silent: bool=False) -> Dict[str, Tuple[float]]:
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

            * Two-tailed :math:`p(H_0 \mid \texttt{data})`
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

    alpha = 100 * alpha


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

    if non_paired:
        # This is a simple case of non-paired, non grouped design
        # We will rarely use it in practice, but it is still a valid test
        for bootstrap_iter in pbar(range(n_bootstrap), total=n_bootstrap, desc="Bootstrapping", silent=silent):
            ind1 = np.random.choice(len(sample_1), len(sample_1), replace=True)
            ind2 = np.random.choice(len(sample_2), len(sample_2), replace=True)
            for s_tag in statistics:
                result[s_tag][bootstrap_iter, 0] = statistics[s_tag](sample_1[ind1])
                result[s_tag][bootstrap_iter, 1] = statistics[s_tag](sample_2[ind2])

    bootstrap_result = bootstrap_arrays(
        arrays=(sample_1, sample_2), 
        statistics=statistics, 
        groups=groups, 
        n_bootstrap=n_bootstrap, 
        silent=silent
    )
    result_final = pairwise_bootstrap_test(
        bootstrap_result, 
        samples=(sample_1, sample_2), 
        statistics=statistics, 
        labels=None, 
        alpha=alpha
    )

    results_return = {}
    # This is a necessary post-processing step
    # The pairwise bootstrap has only two models, so, the labels are not needed
    for s_tag in statistics:
        comaprison_label = list(result_final[s_tag].keys())[0]
        results_return[s_tag] = result_final[s_tag][comaprison_label]
    return results_return


def compare_models(y_test: Union[npt.NDArray[int], npt.NDArray[float]], 
                   preds_1: Union[npt.NDArray[int], npt.NDArray[float]], 
                   preds_2: Union[npt.NDArray[int], npt.NDArray[float]], 
                   metrics: Tuple[Union[str, Metric]],
                   groups: Optional[npt.NDArray[int]]=None,
                   alpha: float=0.05, 
                   n_bootstrap: int=5000, 
                   seed: Optional[int]=None, 
                   silent: bool=False) -> Dict[str, Tuple[float]]:
    r"""Compares predictions from two models :math:`f_1(x)` and :math:`f_2(x)` that yield prediction vectors  :math:`\hat y_{1}` and :math:`\hat y_{2}` 
    with a two-tailed bootstrap hypothesis test. Note: you must make sure that the metric is defined as more is better (e.g. accuracy, AUC, and others).
    
    I.e., we state the following null and alternative hypotheses:

    .. math::
        H_0: M(y_{gt}, \hat y_{1}) = M(y_{gt}, \hat y_{2})

        H_1: M(y_{gt}, \hat y_{2}) \neq M(y_{gt}, \hat y_{1}),

    where :math:`M` is a metric, :math:`y_{gt}` is the vector of ground truth labels, 
    and :math:`\hat y_{i}, i=1,2` are the vectors of predictions for model 1 and 2, respectively. 
    Such kind of testing is performed for every specified metric. 
    
    By default, the function assumes that the metrics are defined as more is better (e.g. accuracy, AUC, and others). 
    If you work with metrics that are defined as less is better, just swap the models (:math:`\hat y_{1}` and :math:`\hat y_{2}`) in the function call.
    
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
    sample_2 = PredSampleWrapper(preds_2, y_test, multiclass=len(preds_1.shape) != 1)

    metrics_dict = {}
    for metric in metrics:
        if isinstance(metric, Metric):
            metrics_dict[str(metric)] = metric # note that the object must be instantiated
        elif isinstance(metric, str):
            assert hasattr(metricslib, metric), f"Metric {metric} is not defined"
            metrics_dict[metric] = getattr(metricslib, metric)()

    test_results = two_sample_test(sample_1, sample_2, statistics=metrics_dict, groups=groups, alpha=alpha, n_bootstrap=n_bootstrap, seed=seed, non_paired=False, silent=silent)
    output = {}
    for metric in test_results:
        output[metric] = test_results[metric]
    return output
    