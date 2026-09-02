__version__ = "0.1.6"

__all__ = [
    "metrics", "synthetic", "compare_models", "compare_models_pairwise", "two_sample_test",
    "pairwise_bootstrap_test", "bootstrap_arrays", "holm_bonferroni_correction",
    "to_latex", "pairwise_to_latex", "PredSampleWrapper",
]

from . import metrics, synthetic
from ._stambo import (
    compare_models,
    compare_models_pairwise,
    two_sample_test,
    pairwise_bootstrap_test,
    bootstrap_arrays,
    holm_bonferroni_correction,
)
from ._utils import to_latex, pairwise_to_latex
from ._predsamplewrapper import PredSampleWrapper
