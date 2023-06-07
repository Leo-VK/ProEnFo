from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from feature.feature_lag_selection import FeatureLagSelectionStrategy
from feature.time_categorical import add_datetime_features, Month, Day, Hour
from feature.time_lag import lag_target, rename_lag_series


class FeatureExternalSelectionStrategy(ABC):
    """Class representing a univariate feature selection for lags"""

    def __init__(self, external_names: list[str]):
        self.external_names = external_names
        self.name = self.__class__.__name__.replace("Strategy", "").lower()

    @abstractmethod
    def select_features(self, data: pd.DataFrame, horizon: int) -> tuple[
        pd.DataFrame, dict[str, list[int]]]:
        pass


class NoExternalSelectionStrategy(FeatureExternalSelectionStrategy):
    """Default feature selection strategy choosing no external features"""

    def __init__(self):
        super().__init__(list())

    def select_features(self, data: pd.DataFrame, horizon: int) -> tuple[
        pd.DataFrame, dict[str, list[int]]]:
        return pd.DataFrame(), {}


class TaosVanillaStrategy(FeatureExternalSelectionStrategy):
    """Choose features according to Tao's vanilla benchmark
    (T. Hong, "Short term electric load forecasting", North Carolina State University)"""

    def __init__(self, temperature_column_name: str):
        super().__init__([temperature_column_name])
        self.temperature_column_name = temperature_column_name

    def select_features(self, data: pd.DataFrame, horizon: int) -> tuple[
        pd.DataFrame, dict[str, list[int]]]:
        if horizon < 1:
            raise ValueError('horizon is < 1!')

        features = pd.DataFrame({"Trend": np.arange(len(data))}, index=data.index)
        hour, day, month = Hour(), Day(), Month()
        time_categoricals = [hour, day, month]
        features = add_datetime_features(features, time_categoricals)
        features[f"{day.name}*{hour.name}"] = features[day.name] * features[hour.name]
        try:
            temperature = data[self.temperature_column_name]
            features[f"{self.temperature_column_name}^2"] = temperature.pow(2)
            features[f"{self.temperature_column_name}^3"] = temperature.pow(3)
            for name in [month.name, hour.name]:
                features[f"{name}*{self.temperature_column_name}"] = features[name] * temperature
                features[f"{name}*{self.temperature_column_name}^2"] = features[name] * temperature.pow(2)
                features[f"{name}*{self.temperature_column_name}^3"] = features[name] * temperature.pow(3)
        except KeyError:
            raise KeyError("Temperature column name is not congruent")

        return features, {}


class LagStrategy(FeatureExternalSelectionStrategy):
    """Choose features according to Tao's vanilla benchmark
    (T. Hong, "Short term electric load forecasting", North Carolina State University)"""

    def __init__(self, lag_strategy_by_name: dict[str, FeatureLagSelectionStrategy]):
        super().__init__(list(lag_strategy_by_name.keys()))
        self.lag_strategy_by_name = lag_strategy_by_name

    def select_features(self, data: pd.DataFrame, horizon: int) -> tuple[
        pd.DataFrame, dict[str, list[int]]]:
        if horizon < 1:
            raise ValueError('horizon is < 1!')

        columns = list(self.lag_strategy_by_name.keys())
        features = data[columns]
        lags = {}
        for name, strategy in self.lag_strategy_by_name.items():
            lags[name] = strategy.select_features(data[name], horizon)
            features = lag_target(features, name, lags[name])
            features = features.drop(columns=name)  # Drop original feature

        return features, lags


class ZeroLagStrategy(FeatureExternalSelectionStrategy):
    """Use zero lag external features, be cautious about this class!"""

    def __init__(self, external_names: list[str]):
        super().__init__(external_names)

    def select_features(self, data: pd.DataFrame, horizon: int) -> tuple[
        pd.DataFrame, dict[str, list[int]]]:
        if horizon < 1:
            raise ValueError('horizon is < 1!')

        zero_lag = 0
        features, lags = {}, {}
        for name in self.external_names:
            lags[name] = [zero_lag]
            zero_lag_feature = data[name].pipe(rename_lag_series, zero_lag)
            features[zero_lag_feature.name] = zero_lag_feature

        return pd.DataFrame(features), lags


def add_modified_external_features(data: pd.DataFrame, external_features: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([data, external_features], axis=1)


def remove_original_external_columns(data: pd.DataFrame, original_external_feature_names: list[str]) -> pd.DataFrame:
    return data.drop(columns=original_external_feature_names)
