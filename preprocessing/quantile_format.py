import pandas as pd
from pandas import Series


def check_quantile_list(quantiles: list[float]) -> list[float]:
    """Checks quantile list for monotonicity and value range"""
    if not all(i < j for i, j in zip(quantiles, quantiles[1:])):
        raise ValueError("Quantiles are not in ascending order")
    if not all(0 < i < 1 for i in quantiles):
        raise ValueError("Quantiles are not in value range (0,1)")
    if not all(round(i, 2) == i for i in quantiles):
        raise ValueError("Quantiles below two digit precision are not supported")
    return quantiles


def check_prediction_interval(prediction_interval: list[float]) -> list[float]:
    """Checks quantile list for monotonicity and value range"""
    if not len(prediction_interval) == 2:
        raise ValueError("Prediction interval has not length 2")
    if not all(0 < i < 1 for i in prediction_interval):
        raise ValueError("Prediction interval are not in value range (0,1)")
    if not prediction_interval[0] < prediction_interval[1]:
        raise ValueError("Prediction interval lower bound is greater than upper bound")
    return prediction_interval


def split_prediction_interval_symmetrically(lower_bounds: pd.Series, upper_bounds: pd.Series) -> tuple[Series, Series]:
    """Checks symmetric of prediction intervals given"""
    if not lower_bounds.is_monotonic_increasing or not upper_bounds.is_monotonic_increasing:
        raise ValueError("Prediction interval is not monotonic")
    if not len(lower_bounds) == len(upper_bounds):
        raise ValueError("Prediction interval halfs are not of same length")
    if not (1 - upper_bounds.sort_values(ascending=False)).round(2).tolist() == lower_bounds.round(2).tolist():
        raise ValueError("Prediction interval halfs are not symmetric")
    return lower_bounds, upper_bounds
