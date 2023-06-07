from typing import Optional

import numpy as np
import pandas as pd


def value_match_cleaning(data: pd.DataFrame, value: float):
    """Flag matching values with NaN"""
    return data.mask(data == value)


def threshold_cleaning(data: pd.DataFrame,
                       lower_bound: Optional[float] = None,
                       upper_bound: Optional[float] = None) -> pd.DataFrame:
    """Flag values outside of bounds with NaN"""
    if lower_bound is not None:
        data = data.mask(data < lower_bound, other=np.nan)
    if upper_bound is not None:
        data = data.mask(data > upper_bound, other=np.nan)
    return data


def interquantile_range_cleaning(data: pd.DataFrame,
                                 lower_quantile: float = 0.25,
                                 upper_quantile: float = 0.75,
                                 outlier_factor: float = 1.5) -> pd.DataFrame:
    """Flag values outside of interquantile range with NaN"""
    Q1 = data.quantile(lower_quantile)
    Q3 = data.quantile(upper_quantile)
    IQR = Q3 - Q1
    outlier_mask = (data < Q1 - outlier_factor * IQR) | (data > Q3 + outlier_factor * IQR)
    return data.mask(outlier_mask, other=np.nan)


def zero_gap_cleaning(data: pd.DataFrame, n_zeros: int = 24) -> pd.DataFrame:
    """Flag zero gaps longer than a threshold length"""
    rolling_sum = data.rolling(window=n_zeros).sum()
    outlier_mask = rolling_sum == 0
    for column in data.columns:
        for outlier_index in np.flatnonzero([outlier_mask[column]]):
            left_bound = outlier_mask.index[outlier_index - n_zeros + 1]
            right_bound = outlier_mask.index[outlier_index]
            outlier_mask.loc[left_bound:right_bound, column] = True
    outlier_mask.iloc[:n_zeros] = False  # Disregard first n_zero values
    return data.mask(outlier_mask, other=np.nan)


def hampel_filter(data: pd.DataFrame, window_size: int, threshold: float = 3) -> pd.DataFrame:
    """Flag values according to hampel identifier"""
    rolling_median = data.rolling(window_size).median()
    standard_deviation = np.abs(rolling_median - data)
    median_absolute_deviation = standard_deviation.rolling(window_size).median()
    outlier_mask = standard_deviation > 1.4826 * threshold * median_absolute_deviation
    return data.mask(outlier_mask, other=np.nan)
