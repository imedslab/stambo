from typing import Union, Tuple, Optional
import numpy as np
import numpy.typing as npt


PredGtType = Union[npt.NDArray[np.float64], npt.NDArray[np.int_]]
PredTuple = Tuple[float, int, Union[float, int]]
IndexType = Union[int, npt.NDArray[np.int_]]


class PredSampleWrapper:
    r"""Wraps predictions and targets in one object.

        Args:
            predictions: Model predictions to wrap.
            gt: Ground-truth labels.
            groups: Optional groups indicating the subject for each measurement. When several
                ``PredSampleWrapper`` samples are compared together (e.g. via
                :func:`stambo.pairwise_bootstrap_test`), all of them must carry the same ``groups``.
                Defaults to None.
            multiclass: Whether the predictions correspond to a multiclass classifier. Defaults to True.
            threshold: Threshold to apply to binary predictions when ``multiclass`` is False. Defaults to 0.5.
            cached_am: Optional cached argmax / thresholded predictions to reuse.
    """
    def __init__(self, predictions: PredGtType,
                 gt: PredGtType, groups: Optional[npt.NDArray[np.int_]]=None, multiclass: bool=True, threshold: Optional[float]=0.5,
                 cached_am: Optional[npt.NDArray[np.int_]]=None):

        self.multiclass = multiclass
        self.groups = groups
        self.predictions = predictions
        self.threshold = threshold
        # Re-using thresholded / argmax values if they are available already when we subsample the data
        self.predictions_am: np.ndarray
        if cached_am is not None:
            self.predictions_am = cached_am
        elif self.multiclass:
            self.predictions_am = np.argmax(predictions, axis=1)
        else:
            if threshold is None or not isinstance(threshold, float):
                raise ValueError(f"The threshold must not be None, and be of type `float`. Found: {threshold}")
            self.predictions_am = self.predictions > threshold
        self.gt = gt

    def __getitem__(self, idx: IndexType) -> Union[PredTuple, "PredSampleWrapper"]:
        r"""Give access to the predictions and the ground truth by index or a set of indices.

        Args:
            idx: Single index or collection of indices.

        Returns:
            Either a tuple containing the predictions, argmaxed predictions, and ground truth
            for a single index, or a new ``PredSampleWrapper`` restricted to the provided indices.
        """

        if isinstance(idx, int):
            return self.predictions[idx], self.predictions_am[idx], self.gt[idx]
        return PredSampleWrapper(self.predictions[idx], self.gt[idx], groups=None if self.groups is None else self.groups[idx],
                                 multiclass=self.multiclass, threshold=self.threshold, cached_am=self.predictions_am[idx])
    
    def __len__(self):
        return self.predictions.shape[0]
