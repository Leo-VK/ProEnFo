import unittest

import pandas as pd
import feature.feature_transformation as ft


class TestFeatureTransformations(unittest.TestCase):

    def test_no_stationarization_with_one_sequence(self):
        target = 'series'
        sequence = [1, 2, 3, 7, 8, 9, 4, 5, 10, 6]

        series = pd.DataFrame(sequence, columns=[target])
        strategy = ft.NoTransformationStrategy()
        transformed_series = strategy.transform_data(series)

        self.assertEqual(sequence, transformed_series[target].tolist())

        inverted_series = strategy.inverse_transform_data(transformed_series)

        self.assertEqual(sequence, inverted_series[target].tolist())

    def test_boxcox_transformation_with_one_sequence(self):
        target = 'series'
        sequence = [1, 2, 3, 7, 8, 9, 4, 5, 10, 6]

        series = pd.DataFrame(sequence, columns=[target])
        strategy = ft.BoxCoxStrategy()
        transformed_series = strategy.transform_data(series)
        inverted_series = strategy.inverse_transform_data(transformed_series)

        self.assertEqual(sequence, inverted_series[target].round(12).tolist())

    def test_log_transformation_with_one_sequence(self):
        target = 'series'
        sequence = [1, 2, 3, 7, 8, 9, 4, 5, 10, 6]

        series = pd.DataFrame(sequence, columns=[target])
        strategy = ft.LogStrategy()
        transformed_series = strategy.transform_data(series)
        inverted_series = strategy.inverse_transform_data(transformed_series)

        self.assertEqual(sequence, inverted_series[target].round(12).tolist())

