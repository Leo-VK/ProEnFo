import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from preprocessing.quantile_format import check_quantile_list

from typing import List, Tuple


class Persistence:
    """Naive persistence benchmark given as last seen historical value"""

    def build_benchmark(self, y_train: pd.Series, y_test: pd.Series, horizon: int) -> pd.DataFrame:
        pred = y_test.shift(periods=horizon)
        pred.iloc[0:horizon] = y_train.iloc[-horizon:]
        return pred


class Climatology:
    """Climatology benchmark given as historical mean value"""

    def build_benchmark(self, y_train: pd.Series, y_test: pd.Series, horizon: int) -> pd.DataFrame:
        pred = pd.concat([y_train, y_test]).shift(periods=horizon)
        return pred.expanding(1).mean().loc[y_test.index]


class Drift:
    """Naive drift benchmark given as current value plus historical mean value"""

    def build_benchmark(self, y_train: pd.Series, y_test: pd.Series, horizon: int) -> pd.DataFrame:
        pred = pd.concat([y_train, y_test]).shift(periods=horizon).iloc[horizon:]
        pred += horizon * pred.diff(1).fillna(0).expanding(1).mean()
        return pred.loc[y_test.index]


class Seasonal:
    """Naive seasonal benchmark given as last seasonal value"""

    def __init__(self, period: int):
        self.period = period

    def build_benchmark(self, y_train: pd.Series, y_test: pd.Series, horizon: int) -> pd.DataFrame:
        shift = int(self.period - horizon + self.period * ((horizon - 1) // self.period))
        pred = pd.concat([y_train, y_test]).shift(periods=shift)
        return pred.loc[y_test.index]


class ExponentialSmoothing:
    """Exponential smoothing benchmark given as weighted historical values"""

    def __init__(self, alpha: float, adjust: bool = False):
        self.alpha = alpha
        self.adjust = adjust

    def build_benchmark(self, y_train: pd.Series, y_test: pd.Series, horizon: int) -> pd.DataFrame:
        pred = pd.concat([y_train, y_test]).shift(periods=horizon)
        return pred.ewm(alpha=self.alpha, adjust=self.adjust).mean().loc[y_test.index]


class MovingAverage:
    """Moving average benchmark given as sliding window mean"""

    def __init__(self, window_size: int):
        self.window_size = window_size

    def build_benchmark(self, y_train: pd.Series, y_test: pd.Series, horizon: int) -> pd.DataFrame:
        pred = pd.concat([y_train, y_test]).shift(periods=horizon)
        return pred.rolling(window=self.window_size, min_periods=1).mean().loc[y_test.index]


class ExpandingARIMA:
    """Exponential smoothing benchmark given as weighted historical values"""

    def __init__(self, autoregressive_order: int, differencing_order: int, moving_average_order: int):
        self.autoregressive_order = autoregressive_order
        self.differencing_order = differencing_order
        self.moving_average_order = moving_average_order

    def build_benchmark(self, y_train: pd.Series, y_test: pd.Series, horizon: int) -> pd.DataFrame:
        def arima_forecast(series: pd.Series):
            horizon_index = len(series) + horizon
            model = ARIMA(series, order=(self.autoregressive_order, self.differencing_order, self.moving_average_order))
            model_fit = model.fit()
            return model_fit.predict(start=horizon_index, end=horizon_index, typ='levels').iloc[0]

        y = pd.concat([y_train, y_test]).shift(periods=horizon)

        return y.expanding(len(y_train) - horizon).apply(arima_forecast).loc[y_test.index]


class ConditionalErrorPersistence:
    """Persistence model with conditional error quantiles from historical error values"""

    def __init__(self, quantiles: List[float]):
        self.quantiles = check_quantile_list(quantiles)

    def build_benchmark(self, y_train: pd.Series, y_test: pd.Series, horizon: int) -> pd.DataFrame:
        y = pd.concat([y_train, y_test])
        pred = y.shift(periods=horizon)
        error = y - pred
        distribution = {}
        for q in self.quantiles:
            distribution[q] = pred + error.expanding(1).quantile(q)
        return pd.DataFrame(distribution).loc[y_test.index]
    

class ConditionalErrorARIMA:
    """Persistence model with conditional error quantiles from historical error values"""

    def __init__(self, autoregressive_order: int, differencing_order: int, moving_average_order: int,quantiles: List[float]):
        self.autoregressive_order = autoregressive_order
        self.differencing_order = differencing_order
        self.moving_average_order = moving_average_order
        self.quantiles = check_quantile_list(quantiles)

    def build_benchmark(self, y_train: pd.Series, y_test: pd.Series, horizon: int) -> pd.DataFrame:
        def arima_forecast(series: pd.Series):
            horizon_index = len(series) + horizon
            model = ARIMA(series, order=(self.autoregressive_order, self.differencing_order, self.moving_average_order))
            model_fit = model.fit()
            return model_fit.predict(start=horizon_index, end=horizon_index, typ='levels').iloc[0]
        y = pd.concat([y_train, y_test]).shift(periods=horizon)
        pred = y.expanding(len(y_train) - horizon).apply(arima_forecast).loc[y_test.index]
        error = y - pred
        distribution = {}
        for q in self.quantiles:
            distribution[q] = pred + error.expanding(1).quantile(q)
        return pd.DataFrame(distribution).loc[y_test.index]


class ExpandingQuantiles:
    """Expanding quantile benchmark given as historical quantile"""

    def __init__(self, quantiles: List[float]):
        self.quantiles = check_quantile_list(quantiles)

    def build_benchmark(self, y_train: pd.Series, y_test: pd.Series, horizon: int) -> pd.DataFrame:
        pred = pd.concat([y_train, y_test]).shift(periods=horizon)
        distribution = {}
        for q in self.quantiles:
            distribution[q] = pred.expanding(1).quantile(q).loc[y_test.index]
        return pd.DataFrame(distribution)


class MovingQuantiles:
    """Moving quantile benchmark given as sliding window quantile"""

    def __init__(self, window_size: int, quantiles: List[float]):
        self.window_size = window_size
        self.quantiles = check_quantile_list(quantiles)

    def build_benchmark(self, y_train: pd.Series, y_test: pd.Series, horizon: int) -> pd.DataFrame:
        pred = pd.concat([y_train, y_test]).shift(periods=horizon)
        distribution = {}
        for q in self.quantiles:
            distribution[q] = pred.rolling(window=self.window_size, min_periods=1).quantile(q).loc[y_test.index]
        return pd.DataFrame(distribution)
