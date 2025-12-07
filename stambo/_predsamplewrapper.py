from typing import Union, Iterable, Tuple, Optional, TypeVar
import numpy as np
import numpy.typing as npt


PredGtType = npt.NDArray[Union[float, int]]
PredTuple = Tuple[float, int, Union[float, int]]
IndexType = Union[int, Iterable[int], npt.NDArray[int]]
PredSampleWrapperType = TypeVar("PredSampleWrapperType", bound="PredSampleWrapper")


class PredSampleWrapper:
    r"""Wraps predictions and targets in one object.

        Args:
            predictions: Model predictions to wrap.
            gt: Ground-truth labels.
            multiclass: Whether the predictions correspond to a multiclass classifier. Defaults to True.
            threshold: Threshold to apply to binary predictions when ``multiclass`` is False. Defaults to 0.5.
            cached_am: Optional cached argmax / thresholded predictions to reuse.
    """
    def __init__(self: PredSampleWrapperType, predictions: PredGtType, 
                 gt: PredGtType, multiclass: bool=True, threshold: Optional[float]=0.5,
                 cached_am: Optional[npt.NDArray[int]]=None):

        self.multiclass = multiclass
        self.predictions = predictions
        self.predictions_am = None
        self.threshold = threshold
        # Re-using thresholded / argmax values if they are available already when we subsample the data
        if cached_am is None:
            if self.multiclass:
                self.predictions_am = np.argmax(predictions, axis=1)
            else:
                if threshold is None or not isinstance(threshold, float):
                    raise ValueError(f"The threshold must not be None, and be of type `float`. Found: {threshold}")
                self.predictions_am = self.predictions > threshold
        else:
            self.predictions_am = cached_am
        self.gt = gt

    def __getitem__(self: PredSampleWrapperType, idx: IndexType) -> Union[PredTuple, PredSampleWrapperType]:
        r"""Give access to the predictions and the ground truth by index or a set of indices.

        Args:
            idx: Single index or collection of indices.

        Returns:
            Either a tuple containing the predictions, argmaxed predictions, and ground truth
            for a single index, or a new ``PredSampleWrapper`` restricted to the provided indices.
        """

        if isinstance(idx, int):
            return self.predictions[idx], self.predictions_am[idx], self.gt[idx]
        return PredSampleWrapper(self.predictions[idx], self.gt[idx], multiclass=self.multiclass, 
                                 threshold=self.threshold, cached_am=self.predictions_am[idx])
    
    def __len__(self):
        return self.predictions.shape[0]
