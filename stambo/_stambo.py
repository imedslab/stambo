import numpy as np
from typing import Optional, Dict, Callable, Tuple, Union
import numpy.typing as npt

from ._utils import pbar
from ._predsamplewrapper import PredSampleWrapper
from .metrics import Metric

from . import metrics as metricslib

def two_sample_test(sample_1: Union[npt.NDArray[np.int64], npt.NDArray[np.float64], PredSampleWrapper], 
                    sample_2: Union[npt.NDArray[np.int64], npt.NDArray[np.float64], PredSampleWrapper], 
                    statistics: Dict[str, Callable], 
                    groups: Optional[npt.NDArray[np.int64]]=None,
                    alpha: float=0.05, 
                    n_bootstrap: int=10000, seed: int=None, 
                    non_paired: bool=False,
                    silent: bool=False) -> Dict[str, Tuple[float]]:
    """Compares whether the empirical difference of statistics computed own two samples is statistically significant or not.
    Note that the statistics are computed independently, and should thus be treated independently.

    Args:
        sample_1 (Union[npt.NDArray[np.int64], npt.NDArray[np.float64]): Sample 1 to be compared
        sample_2 (Union[npt.NDArray[np.int64], npt.NDArray[np.float64]): Sample 2 to be compared
        groups (Optional[npt.NDArray[np.int64]]): Groups indicating the subject for each measurement.
        statistics (Dict[str, Callable]): Statistics to compare the samples by.
        alpha (float, optional): A significance level for confidence intervals (from 0 to 1).
        n_bootstrap (int, optional): The number of bootstrap iterations. Defaults to 10000.
        non_paired (bool, optional): Whether to use a non-paired design. Defaults to False.
        seed (int, optional): Random seed. Defaults to None.
        silent (bool, optional): Whether to execute the function silently, i.e. not showing the progress bar. Defaults to False.

    Returns:
        Dict[Tuple[float]]: A dictionary containing a tuple with the empirical value of the metric, and the p-value. 
                            The expected format in the output in every dict entry is: 
                            
                            * two-sided p-value
                            * observed difference (effect size)
                            * CI low (effect size)
                            * CI high (effect size)
                            * empirical value (sample 1), 
                            * CI low (sample 1)
                            * CI high (sample 1)
                            * empirical value (sample 2), 
                            * CI low (sample 2)
                            * CI high (sample 2).

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
            groups_ind = np.random.choice(groups_ids, len(sample_1), replace=True)
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

        observed = emp_s2 - emp_s1
        # Generaing the null
        null = result[s_tag][:, 1] - result[s_tag][:, 0] - observed
        cdf_left = ((null <= -observed).sum() + 1.) / (n_bootstrap + 1)
        p_val = 2 * min(cdf_left, 1 - cdf_left)
        # Compute the standardized effect size
        es_bs = result[s_tag][:, 1] - result[s_tag][:, 0]
        # We also want to compute the confidence intervals
        # In this version of STAMBO, we use the simple percentile method
        # In the future, we will implement the BCa approach
        ci_es = (np.percentile(es_bs, alpha / 2.), np.percentile(es_bs, 100 - alpha / 2.))
        ci_s1 = (np.percentile(result[s_tag][:, 0], alpha / 2.), np.percentile(result[s_tag][:, 0], 100 - alpha / 2.))
        ci_s2 = (np.percentile(result[s_tag][:, 1], alpha / 2.), np.percentile(result[s_tag][:, 1], 100 - alpha / 2.))
        # And we report the p-value, empirical values, as well as the confidence intervals. 
        # The the format in the documentation.
        result_final[s_tag] = [p_val, observed, ci_es[0], ci_es[1], emp_s1, ci_s1[0], ci_s1[1], emp_s2, ci_s2[0], ci_s2[1]]
        result_final[s_tag] = np.array(result_final[s_tag])
    return result_final


def compare_models(y_test: Union[npt.NDArray[np.int64], npt.NDArray[np.float64]], 
                   preds_1: Union[npt.NDArray[np.int64], npt.NDArray[np.float64]], 
                   preds_2: Union[npt.NDArray[np.int64], npt.NDArray[np.float64]], 
                   metrics: Tuple[Union[str, Metric]],
                   groups: Optional[npt.NDArray[np.int64]]=None,
                   alpha: Optional[float]=0.05, 
                   n_bootstrap: int=10000, seed: int=None, 
                   silent: bool=False) -> Dict[str, Tuple[float]]:
    """Compares predictions from two models :math:`f_1(x)` and :math:`f_2(x)` that yield prediction vectors  :math:`\hat y_{1}` and :math:`\hat y_{2}` 
    with a two-tailed bootstrap hypothesis test.
    
    I.e., we state the following null and alternative hypotheses:

    .. math::
        H_0: M(y_{gt}, \hat y_{1}) = M(y_{gt}, \hat y_{2})

        H_1: M(y_{gt}, \hat y_{1}) != M(y_{gt}, \hat y_{2}),

    where :math:`M` is a metric, :math:`y_{gt}` is the vector of ground truth labels, 
    and :math:`\hat y_{i}, i=1,2` are the vectors of predictions for model 1 and 2, respectively. 
    Such kind of testing is performed for every specified metric. 
    
    By default, the function assumes that the metrics are defined as more is better (e.g. accuracy, AUC, and others). 
    If you work with metrics that are defined as less is better, just swap the models (preds_1 and preds_2) in the function call.
    
    While the test does return you the :math:`p`-value, one should be careful about its interpretation: the :math:`p`-value 
    is the probability of observing the test statistic *at least as extreme* as the one obtained assuming that:math:`H_0` is true (probability of Type II error).
    With large data, even small effects can be statistically significant, so one should consider the effect size. 
    
    We compute a standardized effect size using the estimated bootstrap variance.

    Beyond the hypothesis testing, the function also returns confidence intervals per metric, i.e. :math:`[M(y_{gt}, \hat y)_{(\\alpha / 2)}, M(y_{gt}, \hat y)_{(1 - \\alpha / 2)}]`. At this moment, 
    the confidence intervals are computed using the simple percentile method. In the future, we will implement the BCa approach, which is more accurate.
    
    Args:
        y_test (Union[npt.NDArray[np.int64], npt.NDArray[np.float64]]): Ground truth
        preds_1 (Union[npt.NDArray[np.int64], npt.NDArray[np.float64]]): Prediction from model 1
        preds_2 (Union[npt.NDArray[np.int64], npt.NDArray[np.float64]]): Prediction from model 2
        metrics (Tuple[Union[str, Metric]]): A set of metrics to call. Here, the user either specifies the metrics available from the stambo library (``stambo.metrics``), or adds an instance of the custom-defined metrics.
        groups (Optional[npt.NDArray[np.int64]]): Groups indicating the subject for each measurement.
        alpha (float, optional): A significance level for confidence intervals (from 0 to 1).
        n_bootstrap (int, optional): The number of bootstrap iterations. Defaults to 10000.
        seed (int, optional): Random seed. Defaults to None.
        silent (bool, optional): Whether to execute the function silently, i.e. not showing the progress bar. Defaults to False.

    Returns:
        Dict[Tuple[float]]: A dictionary containing a tuple with the empirical value of the metric, and the left-sided p-value. 
                            The expected format in the output in every dict entry is:

                            * Two sided :math:`p`-value
                            * Observed difference (effect size)
                            * Effect size CI low
                            * Effect size CI high
                            * :math:`M(y_{gt}, \hat y_{1})`
                            * :math:`M(y_{gt}, \hat y_{1})_{(\\alpha / 2)}`
                            * :math:`M(y_{gt}, \hat y_{1})_{(1 - \\alpha / 2)}`
                            * :math:`M(y_{gt}, \hat y_{1})`
                            * :math:`M(y_{gt}, \hat y_{2})_{(\\alpha / 2)}`
                            * :math:`M(y_{gt}, \hat y_{2})_{(1 - \\alpha / 2)}`
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
    