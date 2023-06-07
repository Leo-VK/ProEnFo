from abc import ABC, abstractmethod
from typing import List

import pandas as pd
from pandas import DataFrame
from statsmodels.tsa.seasonal import STL

from utils.statsmodelstools import MSTL


class TimeStationarizationStrategy(ABC):
    """Class representing a stationarization function for a dataseries"""

    def __init__(self, apply_forecast: bool = False):
        self.name = self.__class__.__name__.replace("Strategy", "").lower()
        self.apply_forecast = apply_forecast

    @abstractmethod
    def make_stationary(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def invert_stationary(self, data: pd.DataFrame) -> pd.DataFrame:
        pass


class NoStationarizationStrategy(TimeStationarizationStrategy):
    """Default stationarization strategy with no stationarization"""

    def make_stationary(self, data: pd.DataFrame) -> DataFrame:
        return data

    def invert_stationary(self, data: pd.DataFrame) -> pd.DataFrame:
        return data


class DifferencingStrategy(TimeStationarizationStrategy):
    """Time differencing to stabilize mean"""

    def __init__(self, periods: List[int], fill_nan: int = 0, apply_forecast: bool = False):
        super().__init__(apply_forecast)
        self.periods = periods
        self.fill_nan = fill_nan
        self.histories: dict[str, dict[int, pd.DataFrame]]

    def make_stationary(self, data: pd.DataFrame) -> pd.DataFrame:
        histories = {}
        for column in data.columns:
            histories[column] = {}
            for period in self.periods:
                histories[column][period] = data
                data = data.diff(periods=period).iloc[period:]
        self.histories = histories

        return data

    def invert_stationary(self, data: pd.DataFrame) -> pd.DataFrame:
        for column in data.columns:
            for period in self.periods[-1::-1]:
                data = data + self.histories[column][period].shift(period)
        return data.fillna(self.fill_nan)


class LOESSStrategy(TimeStationarizationStrategy):
    """Locally Estimated Scatterplot Smoothing (LOESS) for season and trend removal"""

    def __init__(self, period: int, apply_forecast: bool = False):
        super().__init__(apply_forecast)
        self.period = period
        self.seasonal: dict[str, pd.Series]
        self.trend: dict[str, pd.Series]

    def make_stationary(self, data: pd.DataFrame) -> pd.DataFrame:
        seasonal, trend = {}, {}
        for column in data.columns:
            series = data[column]
            model = STL(series, period=self.period)
            res = model.fit()
            seasonal[column] = res.seasonal
            trend[column] = res.trend
            data.loc[:, column] = res.resid
        self.seasonal = seasonal
        self.trend = trend

        return data

    def invert_stationary(self, data: pd.DataFrame) -> pd.DataFrame:
        for column in data.columns:
            seasonal = extrapolate_seasonal(self.seasonal[column], data[column], self.period)
            trend = extrapolate_trend(self.trend[column], data[column])
            data.loc[:, column] += (seasonal + trend)[data.index]
        return data


class MLOESSStrategy(TimeStationarizationStrategy):
    """Multiple Seasonal-Trend decomposition using LOESS for season and trend removal"""

    def __init__(self, periods: List[int], apply_forecast: bool = False):
        super().__init__(apply_forecast)
        self.periods = periods
        self.seasonals: dict[str, pd.DataFrame]
        self.trend: dict[str, pd.Series]

    def make_stationary(self, data: pd.DataFrame) -> pd.DataFrame:
        seasonals, trend = {}, {}
        for column in data.columns:
            model = MSTL(data[column], periods=self.periods)
            res = model.fit()
            seasonals[column] = res.seasonal
            trend[column] = res.trend
            data.loc[:, column] = res.resid
        self.seasonals = seasonals
        self.trend = trend

        return data

    def invert_stationary(self, data: pd.DataFrame) -> pd.DataFrame:
        for column in data.columns:
            seasonals = {}
            for period in self.periods:
                seasonals[period] = extrapolate_seasonal(self.seasonals[column][f"seasonal_{period}"],
                                                              data[column], period)
            trend = extrapolate_trend(self.trend[column], data[column])
            data.loc[:, column] += (pd.DataFrame(seasonals).sum(axis="columns") + trend)[data.index]
        return data


def extrapolate_seasonal(seasonal: pd.Series, series: pd.Series, period: int) -> pd.Series:
    """ Naive Seasonal model
    (https://www.statsmodels.org/dev/generated/statsmodels.tsa.forecasting.stl.STLForecast.html)"""
    combined = seasonal.combine(series, func=lambda x, y: x)  # Add index of series
    if len(combined) != len(seasonal):
        N = len(seasonal) - 1
        for h in range(1, len(series) + 1):
            k = int(period - h + period * ((h - 1) // period))
            combined.iloc[N + h] = combined.iloc[N - k]
    return combined


def extrapolate_trend(trend: pd.Series, series: pd.Series) -> pd.Series:
    return trend.combine(series, func=lambda x, y: x).interpolate(method="slinear", fill_value="extrapolate")
