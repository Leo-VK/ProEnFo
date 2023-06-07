import unittest
import numpy as np
import pandas as pd
from feature.time_lag import lag_target


class TestTimeLag(unittest.TestCase):

    def test_lag_target_with_lag_two_result_is_lagged_by_two(self):
        target = 'series'
        time_lag = 2
        series = pd.DataFrame([i for i in range(10)], columns=[target])

        lagged_series = lag_target(series, target, [time_lag])

        target_series = lagged_series[f'{target}_t-{time_lag}'].values.tolist()
        reference_series = [np.nan, np.nan, 0, 1, 2, 3, 4, 5, 6, 7]

        self.assertEqual(reference_series[time_lag:], target_series[time_lag:])

    def test_lag_target_with_no_lag_leads_to_exception(self):
        target = 'series'
        time_lag = 0
        series = pd.DataFrame([i for i in range(10)], columns=[target])

        self.assertRaises(ValueError, lambda: lag_target(series, target, [time_lag]))

    def test_lag_target_with_two_lags_result_contains_both(self):
        target = 'series'
        time_lags = [2, 4]
        series = [i for i in range(10)]
        series_df = pd.DataFrame(series, columns=[target])

        lagged_series = lag_target(series_df, target, time_lags)

        reference_series_one = [np.nan, np.nan, 0, 1, 2, 3, 4, 5, 6, 7]
        reference_series_two = [np.nan, np.nan, np.nan, np.nan, 0, 1, 2, 3, 4, 5]
        reference_df = pd.DataFrame([series, reference_series_one, reference_series_two],
                                    index=[target,
                                           f'{target}_t-{time_lags[0]}',
                                           f'{target}_t-{time_lags[1]}']).transpose()
        actual_length = (lagged_series.iloc[max(time_lags):] == reference_df.iloc[max(time_lags):]).sum().sum()
        expected_length = (len(series) - max(time_lags)) * (len(time_lags) + 1)
        self.assertEqual(expected_length, actual_length)


if __name__ == '__main__':
    unittest.main()
