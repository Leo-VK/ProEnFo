import numpy as np
import pandas as pd
from typing import List, Tuple

"""
Quantile weightings
("Comparing Density Forecasts Using Threshold-and Quantile-Weighted Scoring Rules", T. Gneiting et. al)
"""


def uniform_quantile_weighting(quantiles: List[float]) -> pd.Series:
    return pd.Series(1, index=quantiles)


def center_quantile_weighting(quantiles: List[float]) -> pd.Series:
    q = pd.Series(quantiles, index=quantiles)
    return q * (1 - q)


def left_tail_quantile_weighting(quantiles: List[float]) -> pd.Series:
    q = pd.Series(quantiles, index=quantiles)
    return (1 - q) ** 2


def right_tail_quantile_weighting(quantiles: List[float]) -> pd.Series:
    q = pd.Series(quantiles, index=quantiles)
    return q ** 2


def two_tailed_quantile_weighting(quantiles: List[float]) -> pd.Series:
    q = pd.Series(quantiles, index=quantiles)
    return (2 * q - 1) ** 2


"""
Sample weightings
"""


def uniform_sample_weighting(y_true: pd.Series) -> pd.Series:
    """Weight samples equally"""
    return pd.Series(1, index=y_true.index)


def linear_time_weighting(y_true: pd.Series) -> pd.Series:
    """Weight linearly with increasing time"""
    w = np.arange(1, len(y_true) + 1)
    return pd.Series(w, index=y_true.index)


def activity_time_weighting(y_true: pd.Series) -> pd.Series:
    """Weight according to day/night consumer activity"""
    w = ~y_true.index.to_series().dt.hour.isin([i for i in range(7)])  # Inactivity from 0am to 6am
    return 1 * w  # Convert to int


def load_time_weighting(y_true: pd.Series) -> pd.Series:
    """Weight according to load hour times"""
    values_per_day = y_true.index.to_series().groupby(pd.Grouper(freq="D")).count().mode().iloc[0]
    x = np.linspace(0, 4, values_per_day - int(values_per_day // 4))  # Account for 6 hours of inactivity
    load_weight = -0.25 * (x - 2) ** 2 + 1  # Parabolic formula
    inactivity_weight = np.zeros(int(values_per_day // 4))
    daily_w = np.concatenate([inactivity_weight, load_weight])  # Construct daily weight
    w = y_true.to_frame().eval("""Hour = index.dt.hour 
    Minute = index.dt.minute * 0.01
    Second = index.dt.second * 0.001""").drop(columns=y_true.name)  # Prepare time identifier
    w = w.loc[:, (w != 0).any(axis=0)].sum(axis=1)  # Make time identifier
    return w.replace(w.unique(), daily_w)  # Map values


def sample_level_weighting(y_true: pd.Series) -> pd.Series:
    """Weight according to relative value importance"""
    return y_true


def scaled_error_weighting(y_true: pd.Series) -> pd.Series:
    """Weighting used for scaled error metrics ("Another look at measures of forecast accuracy", R. J. Hyndman)"""
    return y_true.diff(1).iloc[1:].mean()
