import warnings

import pandas as pd
from typing import List,Dict


def rename_lag_series(column: pd.Series, lag: int):
    """Rename column with {name}_t-{lag} pattern"""
    column.name = f'{column.name}_t-{lag}'
    return column


def lag_target(data: pd.DataFrame, target: str, time_lags: List[int]) -> pd.DataFrame:
    """Lag a target column of dataframe for given lag list"""
    if 0 in time_lags:
        raise ValueError('A zero time lag is given')

    # Add a lagged column for every lag
    for lag in time_lags:
        data = data.join(data[target].shift(lag).pipe(rename_lag_series, lag))

    return data


def remove_lag_interval(data: pd.DataFrame,
                        horizon: int,
                        target_lags: List[int],
                        external_lags_by_name: Dict[str, List[int]]) -> pd.DataFrame:
    """Remove NaN intervals introduced by lagging of columns"""
    target_time_drop = max(target_lags) if target_lags else horizon

    flattened_external_lags = [x for l in external_lags_by_name.values() for x in l if l]
    external_time_drop = max(flattened_external_lags) if flattened_external_lags else horizon

    time_drop = max(target_time_drop, external_time_drop)
    return data.drop(index=data.index[range(time_drop)])
