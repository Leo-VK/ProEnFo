from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf

from feature.time_constant import TimeConstant
from typing import List


class FeatureLagSelectionStrategy(ABC):
    """Class representing a univariate feature selection for lags"""

    def __init__(self, number_of_lags: int):
        self.number_of_lags = number_of_lags
        self.name = self.__class__.__name__.replace("Strategy", "").lower()

    @abstractmethod
    def select_features(self, data: pd.Series, horizon: int) -> List[int]:
        pass


class NoFeatureLagSelectionStrategy(FeatureLagSelectionStrategy):
    """Default feature selection strategy choosing no lags"""

    def __init__(self):
        super().__init__(number_of_lags=0)

    def select_features(self, data: pd.Series, horizon: int) -> List[int]:
        return []


class ManualStrategy(FeatureLagSelectionStrategy):
    """Choose manually predefined feature lags"""

    def __init__(self, lags: List[int]):
        super().__init__(number_of_lags=len(lags))
        self.lags = lags

    def select_features(self, data: pd.Series, horizon: int) -> List[int]:
        if horizon < 1:
            raise ValueError('horizon is < 1!')

        manual_features = []
        for lag in self.lags:
            if lag >= horizon:
                manual_features.append(lag)
            else:
                raise ValueError('manual lag < horizon!')

        return manual_features[:self.number_of_lags]


class RecentStrategy(FeatureLagSelectionStrategy):
    """Choose N most recent feature lags"""
    def __init__(self, number_of_lags: int):
        super().__init__(number_of_lags=number_of_lags)

    def select_features(self, data: pd.Series, horizon: int) -> List[int]:
        if horizon < 1:
            raise ValueError('horizon is < 1!')

        recent_features = [lag for lag in range(horizon, horizon + self.number_of_lags)]

        return recent_features


class AutoCorrelationStrategy(FeatureLagSelectionStrategy):
    """Choose N most autocorrelated feature lags"""

    def __init__(self, number_of_lags: int):
        super().__init__(number_of_lags=number_of_lags)

    def select_features(self, data: pd.Series, horizon: int) -> List[int]:
        if horizon < 1:
            raise ValueError('horizon is < 1!')

        # Calculate autocorrelation values
        MEASUREMENTS_PER_WEEK = TimeConstant.SECONDS_PER_WEEK // data.index.freq.delta.seconds
        acf_values = acf(data, nlags=MEASUREMENTS_PER_WEEK + 1, fft=True)  # Ensure dependency near week look-back

        # Extract requested lags
        acf_features = _extract_lags(acf_values, horizon, self.number_of_lags)

        return acf_features


class PartialAutoCorrelationStrategy(FeatureLagSelectionStrategy):
    """Choose N most partially autocorrelated feature lags"""

    def __init__(self, number_of_lags: int):
        super().__init__(number_of_lags=number_of_lags)

    def select_features(self, data: pd.Series, horizon: int) -> List[int]:
        if horizon < 1:
            raise ValueError('horizon is < 1!')

        # Calculate autocorrelation values
        MEASUREMENTS_PER_WEEK = TimeConstant.SECONDS_PER_WEEK // data.index.freq.delta.seconds
        acf_values = pacf(data, nlags=MEASUREMENTS_PER_WEEK + 1)  # Ensure dependency near week look-back

        # Extract requested lags
        acf_features = _extract_lags(acf_values, horizon, self.number_of_lags)

        return acf_features


def _extract_lags(acf_values: np.ndarray, horizon: int, number_of_lags: int) -> List[int]:
    """Extracts lags in descending order considering horizon"""

    # Negation to emulate descending order
    acf_lags = np.argsort(-np.abs(acf_values))

    # Sort after extracting most important lags!
    acf_features = np.sort(acf_lags[acf_lags >= horizon][:number_of_lags]).tolist()

    return acf_features
