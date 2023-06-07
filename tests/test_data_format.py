import unittest
import datetime as dt
import preprocessing.data_format as data_format

import pandas as pd


class TestDataFormat(unittest.TestCase):
    def test_setting_datetimeindex_with_no_missing_datetimes(self):
        resolution = dt.timedelta(minutes=60)
        values = [hour for hour in range(24)]
        timestamp_format = "%Y-%m-%d %H:%M:%S"
        timestamps = [f"2016-01-01 {hour}:00:00" for hour in range(24)]
        data = pd.DataFrame({"value": values, "timestamp": timestamps})

        data = data_format.set_datetimeindex(data,
                                             resolution,
                                             "timestamp",
                                             timestamp_format)

        self.assertTrue(data.index.size > 0)
        self.assertRaises(KeyError, lambda: data["timestamp"])

    def test_setting_datetimeindex_with_two_missing_datetimes(self):
        remove_five, remove_fifteen = 5, 14
        resolution = dt.timedelta(minutes=60)
        values = [hour for hour in range(24)]
        values.pop(remove_five)
        values.pop(remove_fifteen)
        timestamp_format = "%Y-%m-%d %H:%M:%S"
        timestamps = [f"2016-01-01 {hour}:00:00" for hour in range(24)]
        timestamps.pop(remove_five)
        timestamps.pop(remove_fifteen)
        data = pd.DataFrame({"value": values, "timestamp": timestamps})

        self.assertRaises(ValueError,
                          lambda: data_format.set_datetimeindex(data, resolution, "timestamp", timestamp_format))

    def test_setting_datetimeindex_with_inconsistent_resolution(self):
        resolution = dt.timedelta(minutes=30)
        values = [hour for hour in range(24)]
        timestamp_format = "%Y-%m-%d %H:%M:%S"
        timestamps = [f"2016-01-01 {hour}:00:00" for hour in range(24)]
        data = pd.DataFrame({"value": values, "timestamp": timestamps})

        self.assertRaises(ValueError,
                          lambda: data_format.set_datetimeindex(data, resolution, "timestamp", timestamp_format))

    def test_missing_datetime_search_with_no_missing_datetimes(self):
        resolution = dt.timedelta(minutes=60)
        start = dt.datetime(day=1, month=1, year=2016)
        end = dt.datetime(day=2, month=1, year=2016)
        timestamp_format = "%Y-%m-%d %H:%M:%S"
        timestamps = [f"2016-01-01 {hour}:00:00" for hour in range(24)]
        timestamp_series = pd.Series(timestamps)

        missing_datetime = data_format.find_missing_datetime(timestamp_series,
                                                             timestamp_format,
                                                             resolution,
                                                             start,
                                                             end)

        expected_missing_datetime_number = 0
        actual_missing_datetimes_number = missing_datetime.size
        self.assertEqual(expected_missing_datetime_number, actual_missing_datetimes_number)

    def test_missing_datetime_search_with_two_missing_datetimes(self):
        remove_five, remove_fifteen = 5, 14
        resolution = dt.timedelta(minutes=60)
        start = dt.datetime(day=1, month=1, year=2016)
        end = dt.datetime(day=2, month=1, year=2016)
        timestamp_format = "%Y-%m-%d %H:%M:%S"
        timestamps = [f"2016-01-01 {hour}:00:00" for hour in range(24)]
        timestamps.pop(remove_five)
        timestamps.pop(remove_fifteen)
        timestamp_series = pd.Series(timestamps)

        missing_datetime = data_format.find_missing_datetime(timestamp_series,
                                                             timestamp_format,
                                                             resolution,
                                                             start,
                                                             end)

        expected_missing_datetimes = pd.to_datetime(["2016-01-01 05:00:00", "2016-01-01 15:00:00"])
        actual_missing_datetimes = missing_datetime.sort_values()
        for expected, actual in zip(expected_missing_datetimes, actual_missing_datetimes):
            self.assertEqual(expected, actual)
