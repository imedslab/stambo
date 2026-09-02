import numpy as np
from typing import Optional, Dict, Callable, Tuple, Union
import numpy.typing as npt

from ._utils import pbar
from ._predsamplewrapper import PredSampleWrapper
from .metrics import Metric

from . import metrics as metricslib

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
        # Bootstrap percentile test: treat the (non-shifted) bootstrap
        # distribution of the difference statistic as an approximation of its
        # sampling distribution. This is the same distribution used below for
        # the percentile confidence intervals, so the reported p-value and
        # CIs stay consistent with each other (the (1 - alpha) CI excludes 0
        # iff the two-tailed p-value < alpha).
        #
        # We compute both one-tailed tail masses directly (each with its own
        # continuity correction) rather than deriving one from the other via
        # `1 - p`. That matters at the boundary: if the two samples are
        # identical, diff_array is a point mass at 0, and *both*
        # P(diff <= 0) and P(diff >= 0) equal 1 (0 satisfies both `<=` and
        # `>=`), correctly giving a two-tailed p-value of 1. Deriving the
        # left tail as `1 - P(diff <= 0)` would instead (wrongly) give 0 in
        # that case.
        #
        # Note: an earlier version of this function instead shifted the
        # bootstrap distribution by `observed` before comparing it against
        # `observed` again (i.e. testing diff_array >= 2 * observed). That
        # formula is only equivalent to the direct percentile method below
        # when the bootstrap distribution of the difference happens to be
        # symmetric around `observed`, and inflates the false-positive rate
        # otherwise (e.g. for skewed/bounded metrics or small samples). See
        # tests in tests/test_type_i_errors.py for a calibration check.
        p_right = ((diff_array <= 0).sum() + 1.) / (n_bootstrap + 1)
        p_left = ((diff_array >= 0).sum() + 1.) / (n_bootstrap + 1)
        p_val = min(2. * min(p_right, p_left), 1.0)
        # Compute the effect size
        # We also want to compute the confidence intervals
        # In this version of STAMBO, we use the simple percentile method
        ci_es = (np.percentile(diff_array, alpha / 2.), np.percentile(diff_array, 100 - alpha / 2.))
        ci_s1 = (np.percentile(result[s_tag][:, 0], alpha / 2.), np.percentile(result[s_tag][:, 0], 100 - alpha / 2.))
        ci_s2 = (np.percentile(result[s_tag][:, 1], alpha / 2.), np.percentile(result[s_tag][:, 1], 100 - alpha / 2.))
        # And we report the p-value, empirical values, as well as the confidence intervals. 
        # The format in the documentation.
        result_final[s_tag] = [p_val, observed, ci_es[0], ci_es[1], emp_s1, ci_s1[0], ci_s1[1], emp_s2, ci_s2[0], ci_s2[1]]
        result_final[s_tag] = np.array(result_final[s_tag])
    return result_final


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
    