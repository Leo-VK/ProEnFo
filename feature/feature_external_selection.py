from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from feature.feature_lag_selection import FeatureLagSelectionStrategy
from feature.time_categorical import add_datetime_features, Month, Day, Hour
from feature.time_lag import lag_target, rename_lag_series

from typing import List,Dict,Tuple


class FeatureExternalSelectionStrategy(ABC):
    """Class representing a univariate feature selection for lags"""

    def __init__(self, external_names: List[str]):
        self.external_names = external_names
        self.name = self.__class__.__name__.replace("Strategy", "").lower()

    @abstractmethod
    def select_features(self, data: pd.DataFrame, horizon: int) -> Tuple[
        pd.DataFrame, Dict[str, List[int]]]:
        pass


class NoExternalSelectionStrategy(FeatureExternalSelectionStrategy):
    """Default feature selection strategy choosing no external features"""

    def __init__(self):
        super().__init__(list())

    def select_features(self, data: pd.DataFrame, horizon: int) -> Tuple[
        pd.DataFrame, Dict[str, List[int]]]:
        return pd.DataFrame(), {}


# class TaosVanillaStrategy(FeatureExternalSelectionStrategy):
#     """Choose features according to Tao's vanilla benchmark
#     (T. Hong, "Short term electric load forecasting", North Carolina State University)"""

#     def __init__(self, temperature_column_name: str):
#         super().__init__([temperature_column_name])
#         self.temperature_column_name = temperature_column_name

#     def select_features(self, data: pd.DataFrame, horizon: int) -> Tuple[
#         pd.DataFrame, Dict[str, List[int]]]:
#         if horizon < 1:
#             raise ValueError('horizon is < 1!')

#         features = pd.DataFrame({"Trend": np.arange(len(data))}, index=data.index)
#         hour, day, month = Hour(), Day(), Month()
#         time_categoricals = [hour, day, month]
#         features = add_datetime_features(features, time_categoricals)
#         features[f"{day.name}*{hour.name}"] = features[day.name] * features[hour.name]
#         try:
#             temperature = data[self.temperature_column_name]
#             features[f"{self.temperature_column_name}^2"] = temperature.pow(2)
#             features[f"{self.temperature_column_name}^3"] = temperature.pow(3)
#             for name in [month.name, hour.name]:
#                 features[f"{name}*{self.temperature_column_name}"] = features[name] * temperature
#                 features[f"{name}*{self.temperature_column_name}^2"] = features[name] * temperature.pow(2)
#                 features[f"{name}*{self.temperature_column_name}^3"] = features[name] * temperature.pow(3)
#         except KeyError:
#             raise KeyError("Temperature column name is not congruent")

#         return features, {}



class TaosVanillaStrategy(FeatureExternalSelectionStrategy):
    """Choose features according to Tao's vanilla benchmark
    (T. Hong, "Short term electric load forecasting", North Carolina State University)"""

    def __init__(self, temperature_column_name: str,load_name='load'):
        super().__init__(temperature_column_name)
        self.temperature_column_name = temperature_column_name[0]
        self.load_name = load_name


    def temp_function(self, df, temp, prefix=''):
        t = temp
        t2 = temp**2
        t3 = temp**3
        t_m_columns = []
        t2_m_columns = []
        t3_m_columns = []
        t_h_columns = []
        t2_h_columns = []
        t3_h_columns = []
        for i in range(12):
            tm_name = f't{prefix}_m_{i+1}'
            t2m_name = f't2{prefix}_m_{i+1}'
            t3m_name = f't3{prefix}_m_{i+1}'
            t_m_columns.append(tm_name)
            t2_m_columns.append(t2m_name)
            t3_m_columns.append(t3m_name)

        for i in range(24):
            th_name = f't{prefix}_h_{i}'
            t2h_name = f't2{prefix}_h_{i}'
            t3h_name = f't3{prefix}_h_{i}'
            t_h_columns.append(th_name)
            t2_h_columns.append(t2h_name)
            t3_h_columns.append(t3h_name)
        enc_t_h = pd.get_dummies(df['hour'], prefix='h')
        enc_t2_h = pd.get_dummies(df['hour'], prefix='h')
        enc_t3_h = pd.get_dummies(df['hour'], prefix='h')

        enc_t_m = pd.get_dummies(df['month'], prefix='m')
        enc_t2_m = pd.get_dummies(df['month'], prefix='m')
        enc_t3_m = pd.get_dummies(df['month'], prefix='m')

        for i,row in enc_t_m.iterrows():
            enc_t_m.loc[i] = row.values * temp[i]
            enc_t2_m.loc[i] = row.values * t2[i]
            enc_t3_m.loc[i] = row.values * t3[i]

        for i,row in enc_t_h.iterrows():
            enc_t_h.loc[i] = row.values * temp[i]
            enc_t2_h.loc[i] = row.values * t2[i]
            enc_t3_h.loc[i] = row.values * t3[i]
            

        enc_t_m.columns = t_m_columns
        enc_t2_m.columns = t2_m_columns
        enc_t3_m.columns = t3_m_columns

        enc_t_h.columns = t_h_columns
        enc_t2_h.columns = t2_h_columns
        enc_t3_h.columns = t3_h_columns
        
        temp_df = pd.DataFrame(np.hstack((np.reshape(t,(-1,1)),np.reshape(t2,(-1,1)), np.reshape(t3,(-1,1)))), columns=[f't{prefix}',f't2{prefix}',f't3{prefix}'])
        temp_df = pd.concat([temp_df, enc_t_m, enc_t2_m, enc_t3_m, enc_t_h, enc_t2_h, enc_t3_h], axis=1)
        
        return temp_df

    def select_features(self, data: pd.DataFrame, horizon: int) -> Tuple[pd.DataFrame, Dict[str, List[int]]]:
        year = []
        month = []
        weekday = []
        day = []
        hour = []
        temp = []
        value = []
        for i in range(len(data)):
            year.append(data.index[i].year)
            month.append(data.index[i].month)
            weekday.append(data.index[i].weekday())
            day.append(data.index[i].day)
            hour.append(data.index[i].hour)
            temp.append(data[self.temperature_column_name][i])
            value.append(data[self.load_name][i])
        df = pd.concat([pd.Series(year), pd.Series(month), pd.Series(weekday), pd.Series(day), pd.Series(hour), pd.Series(temp), pd.Series(value)], axis=1)
        df.columns = ['year','month','weekday','day','hour','temp','value']
        trend = range(len(df))
        df['trend'] = trend

        # construct month, weekday, hour onehot encoding feature
        df = pd.concat([df, pd.get_dummies(df['month'], prefix='m')], axis=1)
        df = pd.concat([df, pd.get_dummies(df['weekday'], prefix='w')], axis=1)
        df = pd.concat([df, pd.get_dummies(df['hour'], prefix='h')], axis=1)

        # construct weekday_hour onehot encoding feature
        columns = []
        for i in range(7):
            for j in range(24):
                name = f'w_{i}_h_{j}'
                columns.append(name)

        enc_w_h = np.zeros((len(df), 168))
        for i, row in df.iterrows():
            w = int(row['weekday'])
            h = int(row['hour'])
            enc_w_h[i, 24 * w + h] = 1

        enc_w_h = pd.DataFrame(enc_w_h, columns=columns)
        df = pd.concat([df, enc_w_h], axis=1)

        # construct temperature feature
        temp = np.array(df['temp'])
        df.drop('temp', axis=1, inplace=True)
        temp_df = self.temp_function(df, temp)

        df = pd.concat([df, temp_df], axis=1)
        value = df['value']
        df.drop('value', axis=1, inplace=True)
        df = pd.concat([df, value], axis=1)
        drop_col = ['day', 'year', 'month', 'weekday', 'hour', 'trend','value']
        df.drop(drop_col, axis=1, inplace=True)
        return df, {}


class LagStrategy(FeatureExternalSelectionStrategy):
    """Choose features according to Tao's vanilla benchmark
    (T. Hong, "Short term electric load forecasting", North Carolina State University)"""

    def __init__(self, lag_strategy_by_name: Dict[str, FeatureLagSelectionStrategy]):
        super().__init__(List(lag_strategy_by_name.keys()))
        self.lag_strategy_by_name = lag_strategy_by_name

    def select_features(self, data: pd.DataFrame, horizon: int) -> Tuple[
        pd.DataFrame, Dict[str, List[int]]]:
        if horizon < 1:
            raise ValueError('horizon is < 1!')

        columns = List(self.lag_strategy_by_name.keys())
        features = data[columns]
        lags = {}
        for name, strategy in self.lag_strategy_by_name.items():
            lags[name] = strategy.select_features(data[name], horizon)
            features = lag_target(features, name, lags[name])
            features = features.drop(columns=name)  # Drop original feature

        return features, lags


class ZeroLagStrategy(FeatureExternalSelectionStrategy):
    """Use zero lag external features, be cautious about this class!"""

    def __init__(self, external_names: List[str]):
        super().__init__(external_names)

    def select_features(self, data: pd.DataFrame, horizon: int) -> Tuple[
        pd.DataFrame, Dict[str, List[int]]]:
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
    index = data.index
    external_features.index = index
    return pd.concat([data, external_features], axis=1)


def remove_original_external_columns(data: pd.DataFrame, original_external_feature_names: List[str]) -> pd.DataFrame:
    return data.drop(columns=original_external_feature_names)
