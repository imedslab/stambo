import numpy as np
from typing import Optional, Dict, Callable, Tuple, Union
import numpy.typing as npt

from ._utils import pbar
from ._predsamplewrapper import PredSampleWrapper
from .metrics import Metric

from . import metrics as metricslib

def two_sample_test(sample_1: Union[npt.NDArray[np.int64], npt.NDArray[np.float64], PredSampleWrapper], 
                    sample_2: Union[npt.NDArray[np.int64], npt.NDArray[np.float64], PredSampleWrapper], 
                    statistics: Dict[str, Callable]=None, 
                    alpha: float=0.05, 
                    two_tailed: bool = True,
                    n_bootstrap: int=10000, seed: int=None, 
                    silent: bool=False) -> Dict[str, Tuple[float]]:
    """Compares whether the empirical difference of statistics computed own two samples is statistically significant or not.
    Note that the statistics are computed independently, and should thus be treated independently.

    Args:
        sample_1 (Union[npt.NDArray[np.int64], npt.NDArray[np.float64]): Sample 1 to be comapred
        sample_2 (Union[npt.NDArray[np.int64], npt.NDArray[np.float64]): Sample 2 to be comapred
        statistics (Dict[str, Callable]): Statistics to compare the samples by.
        alpha (float, optional): A significance level for confidence intervals (from 0 to 1).
        n_bootstrap (int, optional): The number of bootstrap iterations. Defaults to 10000.
        seed (int, optional): _description_. Random seed. Defaults to None.
        silent (bool, optional): Whether to execute the function silently, i.e. not showing the progress bar. Defaults to False.

    Returns:
        Dict[Tuple[float]]: A dictionary containing a tuple with the empirical value of the metric, and the p-value. 
                            The expected format in the output in every dict entry is: 
                            
                            * p-value
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

    for bootstrap_iter in pbar(range(n_bootstrap), total=n_bootstrap, desc="Bootstrapping", silent=silent):
        ind = np.random.choice(len(sample_1), len(sample_1), replace=True)

        for s_tag in statistics:
            v1 = statistics[s_tag](sample_1[ind])
            v2 = statistics[s_tag](sample_2[ind])
            result[s_tag][bootstrap_iter, 0] = v1
            result[s_tag][bootstrap_iter, 1] = v2
    
    result_final = {}
    for s_tag in result:
        emp_s1 = statistics[s_tag](sample_1)
        emp_s2 = statistics[s_tag](sample_2) 
        emp_diff = emp_s2 - emp_s1
        # The null distribution for the standard error
        # Note that the null distribution is a coming from bootstrap
        # We shift the bootstrap distribution by the empirical difference
        null = result[s_tag][:, 1] - result[s_tag][:, 0] - emp_diff
        # Bootstrap checks whether the empirical difference is within the margins of the standard error
        left_sided_p_val = ((null <= emp_diff).sum() + 1.) / (n_bootstrap + 1)
        right_sided_p_val = ((null >= emp_diff).sum() + 1.) / (n_bootstrap + 1)
        p_val = 2 * min(left_sided_p_val, right_sided_p_val)
        # We also want to compute the confidence intervals
        # In this version of STAMBO, we use the simple percentile method
        # In the future, we will implement the BCa approach
        ci_s1 = (np.percentile(result[s_tag][:, 0], alpha / 2.), np.percentile(result[s_tag][:, 0], 100 - alpha / 2.))
        ci_s2 = (np.percentile(result[s_tag][:, 1], alpha / 2.), np.percentile(result[s_tag][:, 1], 100 - alpha / 2.))
        # And we report the p-value, empirical values, as well as the confidence intervals. 
        # The the format in the documentation.
        result_final[s_tag] = (p_val, emp_s1, ci_s1[0], ci_s1[1], emp_s2, ci_s2[0], ci_s2[1])
    return result_final


def compare_models(y_test: Union[npt.NDArray[np.int64], npt.NDArray[np.float64]], 
                   preds_1: Union[npt.NDArray[np.int64], npt.NDArray[np.float64]], 
                   preds_2: Union[npt.NDArray[np.int64], npt.NDArray[np.float64]], 
                   metrics: Tuple[Union[str, Metric]],
                   alpha: Optional[float]=0.05,
                   two_tailed: bool=True,
                   n_bootstrap: int=10000, seed: int=None, 
                   silent: bool=False) -> Dict[str, Tuple[float]]:
    """Compares predictions from two models :math:`f_1(x)` and :math:`f_1(x)` that yield prediction vectors  :math:`\hat y_{1}` and :math:`\hat y_{2}` 
    with a one-tailed bootstrap hypothesis test. I.e., we state the following null and alternative hypotheses:

    .. math::
        H_0: M(y_{gt}, \hat y_{1}) = M(y_{gt}, \hat y_{2})

        H_1: M(y_{gt}, \hat y_{1}) < M(y_{gt}, \hat y_{2}),

    where :math:`M` is a metric, :math:`y_{gt}` is the vector of ground truth labels, 
    and :math:`\hat y_{i}, i=1,2` are the vectors of predictions for model 1 and 2, respectively. 
    Such kind of testing is performed for every specified metric.
    
    Note that while the test does return you the :math:`p`-value, one should be careful about its interpretation: the :math:`p`-value 
    is the probability of observing the test statistic *at least as extreme* as the one obtained assuming that:math:`H_0` is true. 
    That is: what is the probability of one model being better than the other, given that when we evaluate them on larger data they would be the same.

    Beyond the hypothesis testing, the function also returns confidence intervals per metric, i.e. :math:`[M(y_{gt}, \hat y)_{(\\alpha / 2)}, M(y_{gt}, \hat y)_{(1 - \\alpha / 2)}]`.

    Args:
        y_test (Union[npt.NDArray[np.int64], npt.NDArray[np.float64]]): Ground truth
        preds_1 (Union[npt.NDArray[np.int64], npt.NDArray[np.float64]]): Prediction from model 1
        preds_2 (Union[npt.NDArray[np.int64], npt.NDArray[np.float64]]): Prediction from model 2
        metrics (Tuple[Union[str, Metric]]): A set of metrics to call. Here, the user either specifies the metrics available from the stambo library (``stambo.metrics``), or adds an instance of the custom-defined metrics.
        alpha (float, optional): A significance level for confidence intervals (from 0 to 1).
        n_bootstrap (int, optional): The number of bootstrap iterations. Defaults to 10000.
        seed (int, optional): Random seed. Defaults to None.
        silent (bool, optional): Whether to execute the function silently, i.e. not showing the progress bar. Defaults to False.

    Returns:
        Dict[Tuple[float]]: A dictionary containing a tuple with the empirical value of the metric, and the p-value. 
                            The expected format in the output in every dict entry is:

                            * :math:`p`-value
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

    return two_sample_test(sample_1, sample_2, statistics=metrics_dict, alpha=alpha,
                           two_tailed=two_tailed, n_bootstrap=n_bootstrap, 
                           seed=seed, silent=silent)
    

