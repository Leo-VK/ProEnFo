from abc import ABC, abstractmethod

import pandas as pd
import pickle
import numpy as np


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


class QuantileSortingStrategy(PostProcessingQuantileStrategy):
    """Sort quantile predictions in ascending order"""


    def sort_row_np(self,row):
        sorted_row = np.sort(row)
        return pd.Series(sorted_row, index=row.index)

    def process_prediction(self, y_pred: pd.DataFrame) -> pd.DataFrame:
        column_names = y_pred.columns
        y_pred = y_pred.apply(self.sort_row_np, axis=1)
        # y_pred = y_pred.apply(lambda x: x.sort_values(ignore_index=True), axis=1)
        y_pred.columns = column_names


        return y_pred
