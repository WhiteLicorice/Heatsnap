"""
test_get_utc_datetime.py

Unit testing suite for the Skyfinder date parser.
Verifies MATLAB serial conversion, timezone math, and edge cases.
"""

import unittest
import datetime
import pytz
import math
from extract_working_dataset import get_utc_datetime

class TestSkyfinderDateParsing(unittest.TestCase):

    def test_matlab_serial_conversion_no_time(self):
        """
        Test Case: Jan 1, 2013 (Midnight)
        MATLAB Serial: 735235.0
        """
        serial = 735235.0
        expected = datetime.datetime(2013, 1, 1, 0, 0, 0, tzinfo=pytz.utc)
        
        result = get_utc_datetime(serial, float('nan'), float('nan'), 0)
        self.assertEqual(result, expected)
        print(f"[PASS] Serial {serial} -> {result}")

    def test_leap_year_day(self):
        """
        Test Case: Feb 29, 2012 (Leap Day)
        """
        serial = 734928.0
        expected = datetime.datetime(2012, 2, 29, 0, 0, 0, tzinfo=pytz.utc)
        
        result = get_utc_datetime(serial, float('nan'), float('nan'), 0)
        self.assertEqual(result, expected)
        print(f"[PASS] Leap Day Serial {serial} -> {result}")

    def test_day_after_leap_day(self):
        """
        Test Case: Mar 1, 2012
        Serial: 734929.0
        """
        serial = 734929.0 
        expected = datetime.datetime(2012, 3, 1, 0, 0, 0, tzinfo=pytz.utc)
        
        result = get_utc_datetime(serial, float('nan'), float('nan'), 0)
        self.assertEqual(result, expected)
        print(f"[PASS] Post-Leap Day Serial {serial} -> {result}")

    def test_matlab_serial_with_fractional_time(self):
        """
        Test Case: Jan 1, 2013 at Noon (12:00)
        MATLAB Serial: 735235.5
        """
        serial = 735235.5
        expected = datetime.datetime(2013, 1, 1, 12, 0, 0, tzinfo=pytz.utc)
        
        result = get_utc_datetime(serial, float('nan'), float('nan'), 0)
        self.assertEqual(result, expected)
        print(f"[PASS] Fractional Serial {serial} -> {result}")

    def test_standard_string_format(self):
        """Test Case: '20130630' with explicit Hour=15, Min=53"""
        date_str = "20130630"
        hour = 15
        minute = 53
        tz = 0
        expected = datetime.datetime(2013, 6, 30, 15, 53, 0, tzinfo=pytz.utc)
        
        result = get_utc_datetime(date_str, hour, minute, tz)
        self.assertEqual(result, expected)
        print(f"[PASS] String {date_str} {hour}:{minute} -> {result}")

    def test_timezone_math(self):
        """Test Case: Timezone handling (EST = -5)"""
        date_str = "20130101"
        hour = 12
        minute = 0
        tz = -5.0
        # 12:00 Local - (-5) = 17:00 UTC
        expected = datetime.datetime(2013, 1, 1, 17, 0, 0, tzinfo=pytz.utc)
        
        result = get_utc_datetime(date_str, hour, minute, tz)
        self.assertEqual(result, expected)
        print(f"[PASS] TZ Math: 12:00 Local (TZ -5) -> {result}")

    def test_24_hour_rollover(self):
        """Test Case: Jan 1, 24:00 -> Jan 2, 00:00"""
        date_str = "20130101"
        hour = 24
        minute = 0
        tz = 0
        expected = datetime.datetime(2013, 1, 2, 0, 0, 0, tzinfo=pytz.utc)
        
        result = get_utc_datetime(date_str, hour, minute, tz)
        self.assertEqual(result, expected)
        print(f"[PASS] 24:00 Rollover -> {result}")

    def test_invalid_24_hour_time(self):
        """Test Case: Invalid '24:15' time -> None"""
        date_str = "20130101"
        hour = 24
        minute = 15 
        tz = 0
        
        result = get_utc_datetime(date_str, hour, minute, tz)
        self.assertIsNone(result)
        print(f"[PASS] Invalid 24:15 -> Returned None (Correct)")

if __name__ == '__main__':
    print("Running Verification Tests for Skyfinder Date Logic...")
    print("-" * 50)
    unittest.main(argv=['first-arg-is-ignored'], exit=False)