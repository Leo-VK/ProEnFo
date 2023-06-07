import unittest

import pandas as pd

from preprocessing import data_cleaning


class TestDataCleaning(unittest.TestCase):
    def test_value_match_cleaning_two_columns_with_no_matching_values(self):
        value_to_match = -3
        list_one = [hour for hour in range(24)]
        list_two = 10 * [1] + 5 * [2] + 9 * [3]
        data = pd.DataFrame({"one": list_one, "two": list_two})

        data = data_cleaning.value_match_cleaning(data, value_to_match)

        self.assertTrue(data["one"].notna().all())
        self.assertTrue(data["two"].notna().all())

    def test_value_match_cleaning_two_columns_with_matching_values(self):
        value_to_match = -3
        list_one = [hour for hour in range(24)]
        list_one[5] = value_to_match
        list_two = 10 * [1] + 5 * [value_to_match] + 9 * [3]
        data = pd.DataFrame({"one": list_one, "two": list_two})

        data = data_cleaning.value_match_cleaning(data, value_to_match)

        self.assertTrue(data.iloc[5:6]["one"].isna().all())
        self.assertTrue(data.iloc[10:15]["two"].isna().all())
        self.assertEqual(1, data["one"].isna().sum())
        self.assertEqual(5, data["two"].isna().sum())

    def test_threshold_cleaning_two_columns_with_no_matching_values(self):
        lower_bound, upper_bound = 0, 24
        list_one = [hour for hour in range(24)]
        list_two = 10 * [1] + 5 * [2] + 9 * [3]
        data = pd.DataFrame({"one": list_one, "two": list_two})

        data = data_cleaning.threshold_cleaning(data, lower_bound, upper_bound)

        self.assertTrue(data["one"].notna().all())
        self.assertTrue(data["two"].notna().all())

    def test_threshold_cleaning_two_columns_with_matching_values(self):
        lower_bound, upper_bound = 5, 10
        list_one = [hour for hour in range(24)]
        list_two = 10 * [1] + 5 * [7] + 9 * [3]
        data = pd.DataFrame({"one": list_one, "two": list_two})

        data = data_cleaning.threshold_cleaning(data, lower_bound, upper_bound)

        self.assertTrue(data.iloc[:5]["one"].isna().all())
        self.assertTrue(data.iloc[11:]["one"].isna().all())
        self.assertTrue(data.iloc[:9]["two"].isna().all())
        self.assertTrue(data.iloc[15:]["two"].isna().all())
        self.assertEqual(18, data["one"].isna().sum())
        self.assertEqual(19, data["two"].isna().sum())

    def test_interquantile_cleaning_two_columns_with_no_matching_values(self):
        lower_quantile, upper_quantile = 0.25, 0.75
        list_one = [hour for hour in range(24)]
        list_two = 10 * [1] + 5 * [2] + 9 * [3]
        data = pd.DataFrame({"one": list_one, "two": list_two})

        data = data_cleaning.interquantile_range_cleaning(data, lower_quantile, upper_quantile)

        self.assertTrue(data["one"].notna().all())
        self.assertTrue(data["two"].notna().all())

    def test_interquantile_cleaning_two_columns_with_matching_values(self):
        lower_quantile, upper_quantile = 0.25, 0.75
        list_one = [hour for hour in range(24)]
        list_one[5] = -50
        list_two = 10 * [1] + 5 * [100] + 9 * [3]
        data = pd.DataFrame({"one": list_one, "two": list_two})

        data = data_cleaning.interquantile_range_cleaning(data, lower_quantile, upper_quantile)

        self.assertTrue(data.iloc[5:6]["one"].isna().all())
        self.assertTrue(data.iloc[10:15]["two"].isna().all())
        self.assertEqual(1, data["one"].isna().sum())
        self.assertEqual(5, data["two"].isna().sum())

    def test_zero_gap_cleaning_two_columns_with_no_matching_values(self):
        n_zeros = 3
        list_one = [1, 2, 3, 4] + 6 * [1] + 14 * [10]
        list_two = 10 * [1] + 5 * [1] + 9 * [3]
        data = pd.DataFrame({"one": list_one, "two": list_two})

        data = data_cleaning.zero_gap_cleaning(data, n_zeros)

        self.assertTrue(data["one"].notna().all())
        self.assertTrue(data["two"].notna().all())

    def test_zero_gap_cleaning_two_columns_with_matching_values(self):
        n_zeros = 3
        list_one = [1, 2, 3, 4] + 6 * [0] + 14 * [10]
        list_two = 10 * [1] + 5 * [0] + 9 * [3]
        data = pd.DataFrame({"one": list_one, "two": list_two})

        data = data_cleaning.zero_gap_cleaning(data, n_zeros)

        self.assertTrue(data.iloc[4:10]["one"].isna().all())
        self.assertTrue(data.iloc[10:15]["two"].isna().all())
        self.assertEqual(6, data["one"].isna().sum())
        self.assertEqual(5, data["two"].isna().sum())

    def test_zero_gap_cleaning_two_columns_ignore_short_zero_values(self):
        n_zeros = 3
        list_one = [1, 2, 3, 4] + 2 * [0] + 18 * [10]
        list_two = 10 * [1] + 2 * [0] + 12 * [3]
        data = pd.DataFrame({"one": list_one, "two": list_two})

        data = data_cleaning.zero_gap_cleaning(data, n_zeros)

        self.assertTrue(data["one"].notna().all())
        self.assertTrue(data["two"].notna().all())

    def test_zero_gap_cleaning_two_columns_ignore_first_zero_values(self):
        n_zeros = 3
        list_one = [1, 2, 3, 4] + 6 * [0] + 14 * [10]
        list_two = 10 * [0] + 5 * [0] + 9 * [3]
        data = pd.DataFrame({"one": list_one, "two": list_two})

        data = data_cleaning.zero_gap_cleaning(data, n_zeros)

        self.assertTrue(data.iloc[4:10]["one"].isna().all())
        self.assertTrue(data.iloc[:3]["two"].notna().all())
        self.assertTrue(data.iloc[10:15]["two"].isna().all())
        self.assertEqual(6, data["one"].isna().sum())
        self.assertEqual(12, data["two"].isna().sum())

    def test_hampel_cleaning_two_columns_with_no_matching_values(self):
        window_size = 5
        list_one = [hour for hour in range(24)]
        list_two = [hour for hour in range(-24, 0)]
        data = pd.DataFrame({"one": list_one, "two": list_two})

        data = data_cleaning.hampel_filter(data, window_size)

        self.assertTrue(data["one"].notna().all())
        self.assertTrue(data["two"].notna().all())

    def test_hampel_cleaning_two_columns_with_matching_values(self):
        window_size = 5
        list_one = [hour for hour in range(24)]
        list_one[15] = 50
        list_two = 10 * [1] + 5 * [2] + 9 * [3]
        data = pd.DataFrame({"one": list_one, "two": list_two})

        data = data_cleaning.hampel_filter(data, window_size)

        self.assertTrue(data.iloc[15:16]["one"].isna().all())
        self.assertTrue(data.iloc[10:12]["two"].isna().all())
        self.assertTrue(data.iloc[15:17]["two"].isna().all())
        self.assertEqual(1, data["one"].isna().sum())
        self.assertEqual(4, data["two"].isna().sum())
