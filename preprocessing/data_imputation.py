from typing import Literal, Union, Callable

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.impute import KNNImputer, SimpleImputer


def last_value_imputation(data: pd.DataFrame):
    """Impute with last seen value"""
    check_interpolation_need(data)
    return data.fillna(method="ffill")


def interpolate(data: pd.DataFrame,
                method: Literal["linear", "time", "index", "values", "pad", "krogh", "from_derivatives",
                                "nearest", "zero", "slinear", "quadratic", "cubic", "spline", "barycentric",
                                "polynomial"] = "linear",
                direction: Literal["forward", "backward", "both"] = "both") -> pd.DataFrame:
    """Interpolate with several function options"""
    check_interpolation_need(data)
    return data.interpolate(method=method, limit_direction=direction)


def statistical_imputation(data: pd.DataFrame,
                           strategy: Literal["mean", "median", "most_frequent", "constant"] = 'mean') -> pd.DataFrame:
    """Impute with statistical measures"""
    check_interpolation_need(data)
    imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
    imputed_data = imputer.fit_transform(data)
    return pd.DataFrame(imputed_data, index=data.index, columns=data.columns)


def k_nearest_neighbors_imputation(data: pd.DataFrame,
                                   n_neighbors: int = 2,
                                   weights: Union[
                                       Literal["uniform", "distance"], Callable] = "uniform") -> pd.DataFrame:
    """Impute with K-Nearest Neighbors algorithm"""
    check_interpolation_need(data)
    imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
    imputed_data = imputer.fit_transform(data)
    return pd.DataFrame(imputed_data, index=data.index, columns=data.columns)


def arima_imputation(data: pd.DataFrame,
                     autoregressive_order: int,
                     differencing_order: int,
                     moving_average_order: int) -> pd.DataFrame:
    """Impute with ARIMA model based on Kalman Filter"""
    check_interpolation_need(data)
    imputed_data = {}
    for column in data.columns:
        imputer = ARIMA(data[column], order=(autoregressive_order, differencing_order, moving_average_order))
        imputer_fit = imputer.fit()
        imputed_data[column] = np.squeeze(imputer_fit.filter_results.smoothed_forecasts)
    return pd.DataFrame(imputed_data, index=data.index, columns=data.columns)


def check_interpolation_need(data: pd.DataFrame):
    if data.isna().sum().sum() == 0:
        raise ValueError("There is nothing to interpolate")