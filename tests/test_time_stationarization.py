import unittest

import pandas as pd

import feature.time_stationarization as ts


class TestTimeStationarization(unittest.TestCase):

    def test_no_stationarization(self):
        target = 'series'
        sequence = [1, 2, 3, 7, 8, 9, 4, 5, 10, 6]

        series = pd.DataFrame(sequence, columns=[target])
        strategy = ts.NoStationarizationStrategy()
        stationary_series = strategy.make_stationary(series)

        self.assertEqual(sequence, stationary_series[target].tolist())

        inverted_series = strategy.invert_stationary(stationary_series)

        self.assertEqual(sequence, inverted_series[target].tolist())

    def test_differencing_one_period(self):
        target = 'series'
        period = 1
        sequence = [1, 2, 3, 7, 8, 9, 4, 5, 10, 6]

        series = pd.DataFrame(sequence, columns=[target])
        strategy = ts.DifferencingStrategy(periods=[period], fill_nan=-10)
        stationary_series = strategy.make_stationary(series)
        reference_series = [(2 - 1), (3 - 2), (7 - 3), (8 - 7), (9 - 8), (4 - 9), (5 - 4), (10 - 5), (6 - 10)]

        self.assertEqual(reference_series, stationary_series[target].tolist())

        inverted_series = strategy.invert_stationary(stationary_series)

        self.assertEqual([-10, 2, 3, 7, 8, 9, 4, 5, 10, 6], inverted_series[target].tolist())

    def test_differencing_seasonal_period(self):
        target = 'series'
        period = 3
        sequence = [1, 2, 3, 7, 8, 9, 4, 5, 10, 6]

        series = pd.DataFrame(sequence, columns=[target])
        strategy = ts.DifferencingStrategy(periods=[period], fill_nan=-10)
        stationary_series = strategy.make_stationary(series)
        reference_series = [(7 - 1), (8 - 2), (9 - 3), (4 - 7), (5 - 8), (10 - 9), (6 - 4)]

        self.assertEqual(reference_series, stationary_series[target].tolist())

        inverted_series = strategy.invert_stationary(stationary_series)

        self.assertEqual([-10, -10, -10, 7, 8, 9, 4, 5, 10, 6], inverted_series[target].tolist())

    def test_two_differencing_seasonal_period(self):
        target = 'series'
        periods = [2, 4]
        sequence = [1, 2, 3, 7, 8, 9, 4, 5, 10, 6]

        series = pd.DataFrame(sequence, columns=[target])
        strategy = ts.DifferencingStrategy(periods=periods, fill_nan=-10)
        stationary_series = strategy.make_stationary(series)
        pre_reference_series = [(3 - 1), (7 - 2), (8 - 3), (9 - 7), (4 - 8), (5 - 9), (10 - 4), (6 - 5)]
        reference_series = [(4 - 8) - (3 - 1), (5 - 9) - (7 - 2), (10 - 4) - (8 - 3), (6 - 5) - (9 - 7)]

        self.assertEqual(reference_series, stationary_series[target].tolist())

        inverted_series = strategy.invert_stationary(stationary_series)

        self.assertEqual([-10, -10, -10, -10, -10, -10, 4, 5, 10, 6], inverted_series[target].tolist())

    def test_loess_strategy_with_sequence(self):
        target = 'series'
        sequence = [1, 2, 3, 7, 8, 9, 4, 5, 10, 6]
        period = 3

        series = pd.DataFrame(sequence, columns=[target])
        strategy = ts.LOESSStrategy(period=period)
        stationary_series = strategy.make_stationary(series)
        inverted_series = strategy.invert_stationary(stationary_series)

        self.assertEqual(sequence, inverted_series[target].round(12).tolist())

    def test_loess_strategy_with_extrapolation_on_test_sequence(self):
        target = 'series'
        sequence = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        test_sequence = [2, 4, 6]
        period = 3

        series = pd.DataFrame(sequence, index=[i for i in range(10)], columns=[target])
        strategy = ts.LOESSStrategy(period=period)
        _ = strategy.make_stationary(series)
        inverted_series = strategy.invert_stationary(pd.DataFrame(test_sequence,
                                                                  index=[10, 11, 12],
                                                                  columns=[target]))

        self.assertEqual([2+11, 4+12, 6+13], inverted_series[target].round(12).tolist())

    def test_multiloess_strategy_with_sequence(self):
        target = 'series'
        sequence = [1, 2, 3, 7, 8, 9, 4, 5, 10, 6]
        periods = [2, 4]

        series = pd.DataFrame(sequence, columns=[target])
        strategy = ts.MLOESSStrategy(periods=periods)
        stationary_series = strategy.make_stationary(series)
        inverted_series = strategy.invert_stationary(stationary_series)

        self.assertEqual(sequence, inverted_series[target].round(12).tolist())

    def test_mloess_strategy_with_extrapolation_on_test_sequence(self):
        target = 'series'
        sequence = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        test_sequence = [2, 4, 6]
        periods = [2, 4]

        series = pd.DataFrame(sequence, index=[i for i in range(10)], columns=[target])
        strategy = ts.MLOESSStrategy(periods=periods)
        _ = strategy.make_stationary(series)
        inverted_series = strategy.invert_stationary(pd.DataFrame(test_sequence,
                                                                  index=[10, 11, 12],
                                                                  columns=[target]))

        self.assertEqual([2+11, 4+12, 6+13], inverted_series[target].round(12).tolist())