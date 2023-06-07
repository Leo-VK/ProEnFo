from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class PostProcessingValueStrategy(ABC):
    """Class representing a postprocessing step for forecasts"""

    def __init__(self):
        self.name = self.__class__.__name__.replace("Strategy", "").lower()

    @abstractmethod
    def process_prediction(self, y_pred: pd.DataFrame) -> pd.DataFrame:
        pass


class NoPostProcessingValueStrategy(PostProcessingValueStrategy):
    """Default post-processing strategy for forecasts with no processing"""

    def process_prediction(self, y_pred: pd.DataFrame) -> pd.DataFrame:
        return y_pred


class ValueClippingStrategy(PostProcessingValueStrategy):
    """Clip predictions at given boundaries"""

    def __init__(self, lower: Optional[float] = None, upper: Optional[float] = None):
        super().__init__()
        self.lower = lower
        self.upper = upper

    def process_prediction(self, y_pred: pd.DataFrame) -> pd.DataFrame:
        return y_pred.clip(self.lower, self.upper)
