import datetime
from typing import Literal, Union

import pandas as pd

from typing import List, Dict

import matplotlib.pyplot as plt


def set_datetimeindex(data: pd.DataFrame,
                      datetime_resolution: datetime.timedelta,
                      current_timestamp_column_name: str,
                      timestamp_format: str) -> pd.DataFrame:
    """Set datetimeindex correctly based on current timestamp column"""
    data['datetime_index'] = pd.to_datetime(data[current_timestamp_column_name], format=timestamp_format)
    data = data.set_index('datetime_index')
    data = data.sort_index()
    try:
        data.index.freq = f"{datetime_resolution.seconds}S"
    except ValueError:
        raise ValueError("Either inconsistent datetime resolution or missing timestamps")
    return data.drop(current_timestamp_column_name, axis="columns")


def check_datetimeindex(data: pd.DataFrame) -> bool:
    """Checks dataframe index for datetime-type and frequency"""
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Index is not of DateTimeIndex type")
    if not data.index.freq:
        raise ValueError("Index frequency is not set")
    return True


def find_missing_datetime(timestamp_series: pd.Series,
                          timestamp_format: str,
                          datetime_resolution: datetime.timedelta,
                          datetime_start: datetime.datetime,
                          datetime_end: datetime.date,
                          inclusive: Union[
                              Literal["left", "right", "both", "neither"], None] = "left") -> pd.DatetimeIndex:
    """Find missing datetime indices"""
    dt_target = pd.DatetimeIndex(pd.to_datetime(timestamp_series, format=timestamp_format))
    dt_ref = pd.date_range(start=datetime_start, end=datetime_end, freq=str(datetime_resolution.seconds) + "S",
                           inclusive=inclusive)
    return dt_ref.difference(dt_target)


def fill_missing_datetime(data: pd.DataFrame,
                          missing_datetimes: pd.DatetimeIndex,
                          timestamp_column_name: str) -> pd.DataFrame:
    if missing_datetimes.empty:
        raise ValueError("No missing datetimes provided")
    missing_df = pd.DataFrame(data=None, index=missing_datetimes).reset_index(names=timestamp_column_name)
    return pd.concat([data, missing_df])


def check_data_feature_alignment(data: pd.DataFrame, target: str, external_names: List[str]) -> bool:
    column_names = data.columns.tolist()
    if target not in column_names:
        raise KeyError(f"Target {target} is not present in data")
    column_names.remove(target)
    for name in column_names:
        if name not in external_names:
            raise KeyError(f"{name} is present in the data, but not listed as external feature")
    return True


def check_missing_values(data: pd.DataFrame):
    if data.isna().sum().sum() != 0:
        raise ValueError("There are missing values")


def check_dublicated_columns(data: pd.DataFrame):
    if data.columns.duplicated().sum() > 0:
        raise ValueError("There are duplicated columns")
    for first_column in data.columns:
        for second_column in data.columns.drop(first_column):
            duplicated = data[first_column].equals(data[second_column])
            if duplicated:
                raise ValueError("There are duplicated columns")
