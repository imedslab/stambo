__version__ = "0.1.5"

__all__ = ["metrics", "synthetic", "compare_models", "two_sample_test", "to_latex", "PredSampleWrapper"]

from . import metrics, synthetic
from ._stambo import compare_models, two_sample_test
from ._utils import to_latex
from ._predsamplewrapper import PredSampleWrapper
