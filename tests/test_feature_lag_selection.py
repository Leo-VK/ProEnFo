import datetime
import unittest
import numpy as np
import pandas as pd
import feature.feature_lag_selection as fls


class TestFeatureLagSelection(unittest.TestCase):

    def test_feature_selection_with_horizon_zero_throws_error(self):
        horizon = 0
        number_of_lags = 6
        resolution = datetime.timedelta(minutes=60)
        data_dummy = pd.Series(dtype=float)

        strategies = [fls.RecentStrategy(number_of_lags), fls.AutoCorrelationStrategy(number_of_lags)]
        for strategy in strategies:
            self.assertRaises(ValueError, lambda: strategy.select_features(data_dummy, horizon))

    def test_feature_selection_with_recent_strategy_and_horizon_three(self):
        horizon = 3
        number_of_lags = 6
        data_dummy = pd.Series(dtype=float)

        strategy = fls.RecentStrategy(number_of_lags)
        features = strategy.select_features(data_dummy, horizon)
        expected_features = [3, 4, 5, 6, 7, 8]

        self.assertEqual(expected_features, features)

    def test_feature_selection_helper_method_extract_lags(self):
        horizon = 3
        number_of_lags = 4
        correlation_values = np.array([1.0, 0.3, -0.9, -0.5, 0.6, 0.7, 0.1, -0.8, -0.2, 0.4])

        lags = fls._extract_lags(correlation_values, horizon, number_of_lags)
        expected_lags = [3, 4, 5, 7]

        self.assertEqual(expected_lags, lags)


if __name__ == '__main__':
    unittest.main()
