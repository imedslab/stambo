__version__ = "0.1.6"

__all__ = [
    "metrics", "synthetic", "compare_models", "two_sample_test", "to_latex", "PredSampleWrapper",
    "bootstrap_arrays", "pairwise_bootstrap_test"
]

from . import metrics, synthetic
from ._stambo import compare_models, two_sample_test
from ._utils import to_latex
from ._predsamplewrapper import PredSampleWrapper
from ._stambo import bootstrap_arrays, pairwise_bootstrap_test
