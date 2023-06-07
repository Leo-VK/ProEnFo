import unittest

import numpy as np
import pandas as pd

import models.benchmark as benchmark


class TestBenchmark(unittest.TestCase):

    def test_persistence_benchmark(self):
        y = pd.Series(np.arange(1, 11))
        horizon = 3

        model = benchmark.Persistence()
        forecast = model.build_benchmark(y_train=y.iloc[0:5],
                                         y_test=y.iloc[5:],
                                         horizon=horizon)

        persistence_reference = np.array([3, 4, 5, 6, 7])

        self.assertEqual(persistence_reference.tolist(), forecast.tolist())

    def test_climatology_benchmark(self):
        y = pd.Series(np.arange(1, 11))
        horizon = 3

        model = benchmark.Climatology()
        forecast = model.build_benchmark(y_train=y.iloc[0:5],
                                         y_test=y.iloc[5:],
                                         horizon=horizon)

        climatology_reference = np.array([6 / 3, 10 / 4, 15 / 5, 21 / 6, 28 / 7])

        self.assertEqual(climatology_reference.tolist(), forecast.tolist())

    def test_drift_benchmark(self):
        y = pd.Series(np.arange(1, 11))
        horizon = 3

        model = benchmark.Drift()
        forecast = model.build_benchmark(y_train=y.iloc[0:5],
                                         y_test=y.iloc[5:],
                                         horizon=horizon)

        drift_reference = np.array([n + horizon * ((n - 1) / i) for i, n in enumerate(range(3, 8), start=3)])

        self.assertEqual(drift_reference.tolist(), forecast.tolist())

    def test_seasonal_benchmark(self):
        y = pd.Series(np.arange(1, 11))
        horizon = 3
        period = 2

        model = benchmark.Seasonal(period=period)
        forecast = model.build_benchmark(y_train=y.iloc[0:5],
                                         y_test=y.iloc[5:],
                                         horizon=horizon)

        seasonal_reference = np.array([5, 6, 7, 8, 9])

        self.assertEqual(seasonal_reference.tolist(), forecast.tolist())

    def test_exponential_smoothing_benchmark(self):
        y = pd.Series(np.arange(1, 11))
        horizon = 3
        smoothing = 0.5

        model = benchmark.ExponentialSmoothing(alpha=smoothing)
        forecast = model.build_benchmark(y_train=y.iloc[0:5],
                                         y_test=y.iloc[5:],
                                         horizon=horizon)

        moving_average_reference = np.array([9 / 4, 25 / 8, 65 / 16, 161 / 32, 385 / 64])

        self.assertEqual(moving_average_reference.tolist(), forecast.tolist())

    def test_moving_average_benchmark(self):
        y = pd.Series(np.arange(1, 11))
        horizon = 3
        window_size = 3

        model = benchmark.MovingAverage(window_size=window_size)
        forecast = model.build_benchmark(y_train=y.iloc[0:5],
                                         y_test=y.iloc[5:],
                                         horizon=horizon)

        moving_average_reference = np.array([6 / 3, 9 / 3, 12 / 3, 15 / 3, 18 / 3])

        self.assertEqual(moving_average_reference.tolist(), forecast.tolist())

    def test_expanding_arima_benchmark(self):
        y = pd.Series(np.arange(1, 11))
        horizon = 3

        model = benchmark.ExpandingARIMA(0, 1, 0)
        forecast = model.build_benchmark(y_train=y.iloc[0:5],
                                         y_test=y.iloc[5:],
                                         horizon=horizon)

        expanding_arima_reference = np.array([3, 4, 5, 6, 7])

        self.assertEqual(expanding_arima_reference.tolist(), forecast.round(8).tolist())

    def test_conditional_error_persistence_benchmark(self):
        y = pd.Series([1, 3, 5, 7, 2, 4, 6, 8, 10, 9])
        horizon = 3
        quantiles = [0.25, 0.75]

        model = benchmark.ConditionalErrorPersistence(quantiles=quantiles)
        forecast = model.build_benchmark(y_train=y.iloc[0:5],
                                         y_test=y.iloc[5:],
                                         horizon=horizon)

        error_quantile_reference = {
            0.25: [4.0, 6.0, 1.0, 3.0, 5.0],
            0.75: [7.5, 7.75, 8.0, 10.0, 12.0]
        }

        for q in quantiles:
            self.assertEqual(error_quantile_reference[q], forecast[q].tolist())

    def test_expanding_quantile_benchmark(self):
        y = pd.Series(np.arange(1, 11))
        horizon = 3
        quantiles = [0.25, 0.75]

        model = benchmark.ExpandingQuantiles(quantiles=quantiles)
        forecast = model.build_benchmark(y_train=y.iloc[0:5],
                                         y_test=y.iloc[5:],
                                         horizon=horizon)

        expanding_quantile_reference = {
            0.25: [1.5, 1.75, 2.0, 2.25, 2.5],
            0.75: [2.5, 3.25, 4.0, 4.75, 5.5]
        }

        for q in quantiles:
            self.assertEqual(expanding_quantile_reference[q], forecast[q].tolist())

    def test_moving_quantile_benchmark(self):
        y = pd.Series(np.arange(1, 11))
        horizon = 3
        window_size = 3
        quantiles = [0.25, 0.75]

        model = benchmark.MovingQuantiles(window_size=window_size, quantiles=quantiles)
        forecast = model.build_benchmark(y_train=y.iloc[0:5],
                                         y_test=y.iloc[5:],
                                         horizon=horizon)

        moving_quantile_reference = {
            0.25: [1.5, 2.5, 3.5, 4.5, 5.5],
            0.75: [2.5, 3.5, 4.5, 5.5, 6.5]
        }

        for q in quantiles:
            self.assertEqual(moving_quantile_reference[q], forecast[q].tolist())


if __name__ == '__main__':
    unittest.main()
