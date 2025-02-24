from abc import ABC, abstractmethod

import pandas as pd
import pickle
import numpy as np
import xarray as xr


class PostProcessingQuantileStrategy(ABC):
    """Class representing a postprocessing step for forecasts"""

    def __init__(self):
        self.name = self.__class__.__name__.replace("Strategy", "").lower()

    @abstractmethod
    def process_prediction(self, y_pred: pd.DataFrame) -> pd.DataFrame:
        pass

class NoPostProcessingQuantileStrategy(PostProcessingQuantileStrategy):
    """Default post-processing strategy for quantile forecasts with no processing"""

    def process_prediction(self, y_pred: pd.DataFrame) -> pd.DataFrame:
        return y_pred


# class QuantileSortingStrategy(PostProcessingQuantileStrategy):
#     """Sort quantile predictions in ascending order"""


#     def sort_row_np(self,row):
#         sorted_row = np.sort(row)
#         return pd.Series(sorted_row, index=row.index)

#     def process_prediction(self, y_pred: pd.DataFrame) -> pd.DataFrame:
#         column_names = y_pred.columns
#         y_pred = y_pred.apply(self.sort_row_np, axis=1)
#         # y_pred = y_pred.apply(lambda x: x.sort_values(ignore_index=True), axis=1)
#         y_pred.columns = column_names


#         return y_pred
    
class QuantileSortingStrategy(PostProcessingQuantileStrategy):
    """Sort quantile predictions in ascending order"""

    def sort_row_np(self, row: np.ndarray) -> np.ndarray:
        return np.sort(row)

    def process_prediction(self, y_pred):  # Remove type hints for flexibility
        if isinstance(y_pred, pd.DataFrame):
            return self.process_prediction_dataframe(y_pred)
        elif isinstance(y_pred, xr.DataArray):
            return self.process_prediction_dataarray(y_pred)
        else:
            raise ValueError("Unsupported input type: {}".format(type(y_pred)))

    def process_prediction_dataframe(self, y_pred: pd.DataFrame) -> pd.DataFrame:
        column_names = y_pred.columns
        y_pred = y_pred.apply(lambda x: x.sort_values(ignore_index=True), axis=1)
        y_pred.columns = column_names
        return y_pred

    def process_prediction_dataarray(self, y_pred: xr.DataArray) -> xr.DataArray:
        sorted_data = np.apply_along_axis(self.sort_row_np, axis=-1, arr=y_pred.values)
        sorted_y_pred = xr.DataArray(
            sorted_data,
            dims=y_pred.dims,
            coords=y_pred.coords,
        )
        return sorted_y_pred
