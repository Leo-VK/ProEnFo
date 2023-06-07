import unittest

import pandas as pd

import feature.feature_lag_selection as fls
import feature.feature_external_selection as fes


class TestFeatureExternalSelection(unittest.TestCase):
    def test_feature_selection_zero_lag_strategy_only_return_specified_column(self):
        horizon = 1
        first_column_name = "a"
        first_column_values = [1, 2, 3]
        zero_lag_column_name = "temperature"
        zero_lag_column_values = [10, 20, 30]
        data = pd.DataFrame({first_column_name: first_column_values, zero_lag_column_name: zero_lag_column_values})

        strategy = fes.ZeroLagStrategy([zero_lag_column_name])
        features, lags = strategy.select_features(data, horizon)

        actual_zero_lag_series = features[features.columns[0]].tolist()
        actual_zero_lags = lags[zero_lag_column_name]
        self.assertEqual(1, features.columns.size)
        self.assertEqual(1, len(lags))
        self.assertEqual(zero_lag_column_values, actual_zero_lag_series)
        self.assertEqual([0], actual_zero_lags)


    def test_feature_selection_lag_strategy_only_return_specified_column(self):
        horizon = 1
        first_column_name = "a"
        first_column_values = [1, 2, 3]
        lag_column_name = "temperature"
        lag_column_values = [10, 20, 30]
        lag = 1
        data = pd.DataFrame({first_column_name: first_column_values, lag_column_name: lag_column_values})

        strategy = fes.LagStrategy({lag_column_name: fls.ManualStrategy([lag])})
        features, lags = strategy.select_features(data, horizon)

        actual_lag_series = features[features.columns[0]].tolist()
        actual_lags = lags[lag_column_name]
        self.assertEqual(1, features.columns.size)
        self.assertEqual(1, len(lags))
        self.assertEqual(lag_column_values[:-lag], actual_lag_series[lag:])
        self.assertEqual([lag], actual_lags)