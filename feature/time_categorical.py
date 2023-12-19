from abc import ABC, abstractmethod
from typing import Optional, Callable

import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

from feature.time_constant import TimeConstant

from typing import List,Dict

class TimeCategorical(ABC):
    """Class representing time categorical features"""

    def __init__(self, transformer_dict: Dict = None):
        if transformer_dict is not None:
            self.transformer = transformer_dict['transformer']
        else:
            self.transformer = None
        # self.name = self.__class__.__name__ if transformer is None else f"{type(transformer).__name__.replace('_transformer', '')}({self.__class__.__name__})"
        self.name = self.__class__.__name__ if transformer_dict is None else self.__class__.__name__+transformer_dict['name']
    @abstractmethod
    def calculate_feature(self, datetime_series: pd.Series) -> pd.Series:
        pass


class Season(TimeCategorical):
    """Season of the year in [0, ..., 3]"""

    def __init__(self, seasonality: Optional[List[int]] = None):
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
    
class Weekday(TimeCategorical):
    """Day of month in [1, ..., 31]"""

    def calculate_feature(self, datetime_series: pd.Series) -> pd.Series:
        return datetime_series.dt.weekday


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
    return {'name':'sin_transformer','transformer':FunctionTransformer(
        lambda x: np.sin(2 * np.pi * x / (TimeConstant.SECONDS_PER_DAY // x.index.freq.delta.seconds)))}


def cos_transformer():
    """Apply cosine transform to TimeCategorical"""
    return {'name':'cos_transformer','transformer':FunctionTransformer(
        lambda x: np.cos(2 * np.pi * x / (TimeConstant.SECONDS_PER_DAY // x.index.freq.delta.seconds)))}


def add_datetime_features(data: pd.DataFrame, feature_strategies: List[TimeCategorical]) -> pd.DataFrame:
    """Add datetime features in ascending order"""
    for strategy in reversed(feature_strategies):
        feature = strategy.calculate_feature(data.index.to_series())
        if feature.sum() == 0:
            raise ValueError("Resolution does not support this TimeCategorical")
        if strategy.transformer is not None:
            feature = strategy.transformer.fit_transform(feature)
        data.insert(loc=data.shape[1], column=strategy.name, value=feature)

    return data


def add_datetime_features_with_lag(data: pd.DataFrame, feature_strategies: List[TimeCategorical], target_lag: List[int], freq: str = 'H') -> pd.DataFrame:
    """Add datetime features with and without target lags"""
    original_index = data.index.to_series()
    
    # Add non-lagged features
    for strategy in feature_strategies:
        feature = strategy.calculate_feature(original_index)
        if feature.sum() == 0:
            raise ValueError("Resolution does not support this TimeCategorical")
        if strategy.transformer is not None:
            feature = strategy.transformer.fit_transform(feature)
        
        column_name = f"{strategy.name}"
        data.insert(loc=data.shape[1], column=column_name, value=feature)
    
    # Add lagged features
    for lag in target_lag:
        lagged_index = original_index.shift(lag)  # Shift the index backward by 'lag' units
        
        for strategy in feature_strategies:
            feature = strategy.calculate_feature(lagged_index)  # Calculate the feature using the lagged index
            if feature.sum() == 0:
                raise ValueError("Resolution does not support this TimeCategorical")
            if strategy.transformer is not None:
                feature = strategy.transformer.fit_transform(feature)
            
            feature.index = original_index  # Set the feature index back to the original index
            column_name = f"{strategy.name}_lag{lag}"
            data.insert(loc=data.shape[1], column=column_name, value=feature)

    return data
