import unittest

import numpy as np
import pandas as pd
import feature.feature_transformation as ft
import feature.time_stationarization as ts
from feature.transformation_chain import apply_transformations_if_requested, invert_transformations_if_requested


class TestTimeLag(unittest.TestCase):
    def test_series_of_transformation_applied_transformation_flag_false(self):
        target = 'series'
        sequence = [1, 2, 3, 7, 8, 9, 4, 5, 10, 6]
        base, period = '2', 6
        series = pd.DataFrame(sequence, columns=[target])
        strategies = (ft.LogStrategy(base=base), ts.DifferencingStrategy(periods=[period]))

        series, transformed_series = apply_transformations_if_requested(series, strategies)
        reference_series = [np.log2(4) - np.log2(1), np.log2(5) - np.log2(2), np.log2(10) - np.log2(3),
                            np.log2(6) - np.log2(7)]
        self.assertEqual(reference_series, transformed_series[target].tolist())
        self.assertEqual(sequence, series[target].tolist())

        inverted_series = invert_transformations_if_requested(transformed_series, target, strategies)
        self.assertNotEqual(sequence, inverted_series[target].round(12).tolist())

    def test_series_of_transformation_applied_transformation_flag_true_for_first(self):
        target = 'series'
        sequence = [1, 2, 3, 7, 8, 9, 4, 5, 10, 6]
        base, period = '2', 6
        series = pd.DataFrame(sequence, columns=[target])
        strategies = (ft.LogStrategy(base=base, apply_forecast=True),
                      ts.DifferencingStrategy(periods=[period], apply_forecast=False))

        series, transformed_series = apply_transformations_if_requested(series, strategies)
        first_reference_series = [np.log2(x) for x in sequence]
        second_reference_series = [np.log2(4) - np.log2(1), np.log2(5) - np.log2(2), np.log2(10) - np.log2(3),
                                   np.log2(6) - np.log2(7)]
        self.assertEqual(second_reference_series, transformed_series[target].tolist())
        self.assertEqual(first_reference_series, series[target].tolist())

        inverted_series = invert_transformations_if_requested(transformed_series, target, strategies)
        self.assertNotEqual(sequence, inverted_series[target].round(12).tolist())

    def test_series_of_transformation_applied_transformation_flag_true_for_second(self):
        target = 'series'
        sequence = [1, 2, 3, 7, 8, 9, 4, 5, 10, 6]
        base, period = '2', 6
        series = pd.DataFrame(sequence, columns=[target])
        strategies = (ft.LogStrategy(base=base, apply_forecast=False),
                      ts.DifferencingStrategy(periods=[period], apply_forecast=True))

        series, transformed_series = apply_transformations_if_requested(series, strategies)
        first_reference_series = [4 - 1, 5 - 2, 10 - 3, 6 - 7]
        second_reference_series = [np.log2(4) - np.log2(1), np.log2(5) - np.log2(2), np.log2(10) - np.log2(3),
                                   np.log2(6) - np.log2(7)]
        self.assertEqual(second_reference_series, transformed_series[target].tolist())
        self.assertEqual(first_reference_series, series[target].tolist())

        inverted_series = invert_transformations_if_requested(transformed_series, target, strategies)
        self.assertNotEqual(sequence, inverted_series[target].round(12).tolist())

    def test_series_of_transformation_applied_transformation_flag_true_for_all(self):
        target = 'series'
        sequence = [1, 2, 3, 7, 8, 9, 4, 5, 10, 6]
        base, period = '2', 6
        series = pd.DataFrame(sequence, columns=[target])
        strategies = (ft.LogStrategy(base=base, apply_forecast=True),
                      ts.DifferencingStrategy(periods=[period], fill_nan=-10, apply_forecast=True))

        series, transformed_series = apply_transformations_if_requested(series, strategies)
        reference_series = [np.log2(4) - np.log2(1), np.log2(5) - np.log2(2), np.log2(10) - np.log2(3),
                            np.log2(6) - np.log2(7)]
        self.assertEqual(reference_series, transformed_series[target].tolist())
        self.assertEqual(reference_series, series[target].tolist())

        inverted_series = invert_transformations_if_requested(transformed_series, target, strategies)
        self.assertEqual([np.exp2(-10) for _ in range(period)]+[4, 5, 10, 6], inverted_series[target].round(12).tolist())
