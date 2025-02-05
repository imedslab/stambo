from typing import Union, Iterable, Tuple, Optional, TypeVar
import numpy as np
import numpy.typing as npt


PredGtType = npt.NDArray[Union[np.float64, np.int_]]
PredTuple = Tuple[np.float64, np.int_, Union[np.float64, np.int_]]
IndexType = Union[int, Iterable[int], npt.NDArray[np.int_]]
PredSampleWrapperType = TypeVar("PredSampleWrapperType", bound="PredSampleWrapper")


class PredSampleWrapper:
    def __init__(self: PredSampleWrapperType, predictions: PredGtType, 
                 gt: PredGtType, multiclass: bool=True, threshold: Optional[float]=0.5,
                 cached_am: Optional[npt.NDArray[np.int_]]=None):
        """Wraps predictions and targets in one object.

        Args:
            predictions (npt.NDArray[Union[np.float64, np.int64]): _description_
            gt (npt.NDArray[Union[np.float64 np.int64]]): _description_
            multiclass (bool, optional): Whether it is a multiclass classifier's sample. Defaults to True.
            threshold (Optional[float]): Whether to apply the threshold to predictions in the case when we deal with the binary classification. Defaults to 0.5.
        """

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
        """Give access to the predictions and the ground truth by index or a set of indices.

        Args:
            idx (Union[int, Iterable[int], npt.NDArray[np.int_]]): Index / indices.

        Returns:
            Tuple[Union[npt.NDArray[np.int_], npt.NDArray[np.float64]], Union[npt.NDArray[np.int_], npt.NDArray[np.float64]]]: A pair of predictions
        """

        if isinstance(idx, int):
            return self.predictions[idx], self.predictions_am[idx], self.gt[idx]
        return PredSampleWrapper(self.predictions[idx], self.gt[idx], multiclass=self.multiclass, 
                                 threshold=self.threshold, cached_am=self.predictions_am[idx])
    
    def __len__(self):
        return self.predictions.shape[0]
