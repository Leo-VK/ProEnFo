import warnings

import pandas as pd
from typing import List,Dict


# def rename_lag_series(column: pd.Series, lag: int):
#     """Rename column with {name}_t-{lag} pattern"""
#     column.name = f'{column.name}_t-{lag}'
#     return column

def rename_lag_series(series: pd.Series, lag: int) -> pd.Series:
    if lag > 0:
        series.name = f"{series.name}_t-{lag}"
    else:
        series.name = f"{series.name}_t+{-lag}"
    return series


def lag_target(data: pd.DataFrame, target: str, time_lags: List[int]) -> pd.DataFrame:
    """Lag a target column of dataframe for given lag list"""
    if 0 in time_lags:
        raise ValueError('A zero time lag is given')

    # Add a lagged column for every lag
    for lag in time_lags:
        data = data.join(data[target].shift(lag).pipe(rename_lag_series, lag))

    return data

def pred_target(data: pd.DataFrame, target: str, time_preds: List[int]) -> pd.DataFrame:
    """Lag a target column of dataframe for given lag list"""
    if 0 in time_preds:
        raise ValueError('A zero time lag is given')

     # Add a pred column for every pred
    for pred in time_preds:
        data = data.join(data[target].shift(-pred).pipe(rename_lag_series, -pred))

    return data

# def lag_pred_target(data: pd.DataFrame, target: str, time_lags: List[int], time_preds: List[int]) -> pd.DataFrame:
#     """Lag (move backward) and pred (move forward) a target column of dataframe for given lag and pred lists"""

#     if 0 in time_lags or 0 in time_preds:
#         raise ValueError('A zero time lag or pred is given')

#     # Add a lagged column for every lag
#     for lag in time_lags:
#         data = data.join(data[target].shift(lag).pipe(rename_lag_series, lag))

#     # Add a pred column for every pred
#     for pred in time_preds:
#         data = data.join(data[target].shift(-pred).pipe(rename_lag_series, -pred))

#     return data

def lag_pred_target(data: pd.DataFrame, targets: List[str], time_lags: List[int], time_preds: List[int]) -> pd.DataFrame:
    """Lag (move backward) and pred (move forward) target columns of dataframe for given lag and pred lists"""

    if 0 in time_lags or 0 in time_preds:
        raise ValueError('A zero time lag or pred is given')

    # Add a lagged column for every lag and target
    for target in targets:
        for lag in time_lags:
            data = data.join(data[target].shift(lag).pipe(rename_lag_series, lag))

    # Add a pred column for every pred and target
    for target in targets:
        for pred in time_preds:
            data = data.join(data[target].shift(-pred).pipe(rename_lag_series, -pred))

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


# def remove_lag_pred_interval(data: pd.DataFrame,
#                               horizon: int,
#                               target_lags: List[int],
#                               target_preds: List[int],
#                               external_lags_by_name: Dict[str, List[int]]) -> pd.DataFrame:
#     """Remove NaN intervals introduced by lagging and prediction of columns"""
#     target_time_drop = max(target_lags) if target_lags else horizon
#     target_time_pred = max(target_preds) if target_preds else 0

#     flattened_external_lags = [x for l in external_lags_by_name.values() for x in l if l]
#     external_time_drop = max(flattened_external_lags) if flattened_external_lags else horizon

#     time_drop_start = max(target_time_drop, external_time_drop)
#     time_drop_end = target_time_pred

#     data = data.drop(index=data.index[range(time_drop_start)])  # Drop NaNs at the beginning
#     data = data.drop(index=data.index[-time_drop_end:])  # Drop NaNs at the end

#     return data


def remove_lag_pred_interval(data: pd.DataFrame,
                              horizon: int,
                              target_lags: List[int],
                              target_preds: List[int],
                              external_lags_by_name: Dict[str, List[int]],
                              external_preds_by_name: Dict[str, List[int]]) -> pd.DataFrame:
    """Remove NaN intervals introduced by lagging and prediction of columns"""
    target_time_drop = max(target_lags) if target_lags else horizon
    target_time_pred = max(target_preds) if target_preds else 0

    flattened_external_lags = [x for l in external_lags_by_name.values() for x in l if l]
    external_time_drop = max(flattened_external_lags) if flattened_external_lags else horizon

    flattened_external_preds = [x for l in external_preds_by_name.values() for x in l if l]
    external_time_pred = max(flattened_external_preds) if flattened_external_preds else 0

    time_drop_start = max(target_time_drop, external_time_drop)
    time_drop_end = max(target_time_pred, external_time_pred)

    data = data.drop(index=data.index[range(time_drop_start)])  # Drop NaNs at the beginning
    data = data.drop(index=data.index[-time_drop_end:])  # Drop NaNs at the end

    return data
