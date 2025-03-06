__version__ = "0.1.3"

__all__ = ["metrics", "synthetic", "compare_models", "two_sample_test", "to_latex"]

from . import metrics, synthetic
from ._stambo import compare_models, two_sample_test
from ._utils import to_latex