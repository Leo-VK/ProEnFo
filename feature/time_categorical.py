from abc import ABC, abstractmethod
from typing import Optional, Callable

import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

from feature.time_constant import TimeConstant


class TimeCategorical(ABC):
    """Class representing time categorical features"""

    def __init__(self, transformer: Optional[Callable] = None):
        self.transformer = transformer
        self.name = self.__class__.__name__ if transformer is None else f"{transformer.__name__.replace('_transformer', '')}({self.__class__.__name__})"

    @abstractmethod
    def calculate_feature(self, datetime_series: pd.Series) -> pd.Series:
        pass


class Season(TimeCategorical):
    """Season of the year in [0, ..., 3]"""

    def __init__(self, seasonality: Optional[list[int]] = None):
        super().__init__()
        self.seasonality = seasonality if not None else 2 * [0] + 3 * [1] + 3 * [2] + 3 * [3] + 1 * [0]
        self.seasonality_per_month = {i: j for i, j in zip(range(1, 13), self.seasonality)}

    def calculate_feature(self, datetime_series: pd.Series) -> pd.Series:
        return datetime_series.dt.month.replace(self.seasonality_per_month)


class Month(TimeCategorical):
    """Month of the year in [1, ..., 12]"""

    def calculate_feature(self, datetime_series: pd.Series) -> pd.Series:
        return datetime_series.dt.month


class Week(TimeCategorical):
    """Week of the year in [1, ..., 52] or [1, ..., 53] depending on the year"""

    def calculate_feature(self, datetime_series: pd.Series) -> pd.Series:
        return datetime_series.dt.isocalendar().week


class Weekend(TimeCategorical):
    """Weekday as 0, weekend as 1"""

    def calculate_feature(self, datetime_series: pd.Series) -> pd.Series:
        return datetime_series.dt.dayofweek.apply(lambda x: 1 if x == 5 or x == 6 else 0)


class Day(TimeCategorical):
    """Day of month in [1, ..., 31]"""

    def calculate_feature(self, datetime_series: pd.Series) -> pd.Series:
        return datetime_series.dt.day


class Hour(TimeCategorical):
    """Hour of day in [0, ..., 23]"""

    def calculate_feature(self, datetime_series: pd.Series) -> pd.Series:
        return datetime_series.dt.hour


class Minute(TimeCategorical):
    """Minute of hour in [0, ..., 59]"""

    def calculate_feature(self, datetime_series: pd.Series) -> pd.Series:
        return datetime_series.dt.minute


def sin_transformer():
    """Apply sine transform to TimeCategorical"""
    return FunctionTransformer(
        lambda x: np.sin(2 * np.pi * x / (TimeConstant.SECONDS_PER_DAY // x.index.freq.delta.seconds)))


def cos_transformer():
    """Apply cosine transform to TimeCategorical"""
    return FunctionTransformer(
        lambda x: np.cos(2 * np.pi * x / (TimeConstant.SECONDS_PER_DAY // x.index.freq.delta.seconds)))


def add_datetime_features(data: pd.DataFrame, feature_strategies: list[TimeCategorical]) -> pd.DataFrame:
    """Add datetime features in ascending order"""
    for strategy in reversed(feature_strategies):
        feature = strategy.calculate_feature(data.index.to_series())
        if feature.sum() == 0:
            raise ValueError("Resolution does not support this TimeCategorical")
        if strategy.transformer is not None:
            feature = strategy.transformer().fit_transform(feature)
        data.insert(loc=0, column=strategy.name, value=feature)

    return data
