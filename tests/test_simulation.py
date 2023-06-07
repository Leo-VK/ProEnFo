import unittest

import numpy as np
import pandas as pd

from utils.simulation import SwingingDoorCompression


class TestDataSimulation(unittest.TestCase):
    def test_swinging_door_compression_with_default_index(self):
        series = [5, 6.5, 8.5, 7, 6, 5.25]
        compressor = SwingingDoorCompression(threshold=2)

        compressed_series = compressor.compress(pd.Series(series))
        expected_series = [5, np.nan, np.nan, np.nan, 6, np.nan]
        self.assertTrue(np.array_equal(expected_series, compressed_series, equal_nan=True))

    def test_swinging_door_compression_with_datetime_index_unused(self):
        series = [5, 6.5, 8.5, 7, 6, 5.25]
        index = pd.DatetimeIndex([f"2022-04-03 0{i}:00:00" for i in range(6)])
        compressor = SwingingDoorCompression(threshold=2, use_datetime=False)

        compressed_series = compressor.compress(pd.Series(series, index=index))
        expected_series = [5, np.nan, np.nan, np.nan, 6, np.nan]
        self.assertTrue(np.array_equal(expected_series, compressed_series, equal_nan=True))

    def test_swinging_door_compression_with_datetime_index_used(self):
        series = [5, 6.5, 8.5, 7, 6, 5.25]
        index = pd.DatetimeIndex([f"2022-04-03 0{i}:00:00" for i in range(6)])
        compressor = SwingingDoorCompression(threshold=2, use_datetime=True)

        compressed_series = compressor.compress(pd.Series(series, index=index))
        expected_series = [5, np.nan, np.nan, np.nan, 6, np.nan]
        self.assertTrue(np.array_equal(expected_series, compressed_series, equal_nan=True))

    def test_swinging_door_compression_with_no_treshold(self):
        series = [5, 6.5, 8.5, 7, 6, 5.25]
        compressor = SwingingDoorCompression(threshold=0)

        compressed_series = compressor.compress(pd.Series(series))
        expected_series = [5, 6.5, 8.5, 7, 6, 5.25]
        self.assertTrue(np.array_equal(expected_series, compressed_series, equal_nan=True))

    def test_swinging_door_compression_with_large_threshold(self):
        series = [5, 6.5, 8.5, 7, 6, 5.25]
        compressor = SwingingDoorCompression(threshold=10)

        compressed_series = compressor.compress(pd.Series(series))
        expected_series = [5, np.nan, np.nan, np.nan, np.nan, np.nan]
        self.assertTrue(np.array_equal(expected_series, compressed_series, equal_nan=True))