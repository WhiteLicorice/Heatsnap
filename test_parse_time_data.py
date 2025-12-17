"""
test_parse_time_data.py

Unit testing suite for the Skyfinder date parser (parse_time_data).
Verifies MATLAB serial conversion, timezone math, and edge cases.
"""

import unittest
import datetime
import pytz
import math
from extract_working_dataset import parse_time_data

class TestSkyfinderDateParsing(unittest.TestCase):

    def test_matlab_serial_conversion_no_time(self):
        """
        Test Case: Jan 1, 2013 (Midnight)
        MATLAB Serial: 735235.0
        """
        serial = 735235.0
        expected_utc = datetime.datetime(2013, 1, 1, 0, 0, 0, tzinfo=pytz.utc)
        
        # Result is (utc_dt, local_hour, local_min, tz_offset)
        utc_dt, local_h, local_m, offset = parse_time_data(serial, float('nan'), float('nan'), 0)
        
        self.assertEqual(utc_dt, expected_utc)
        self.assertEqual(local_h, 0)
        self.assertEqual(local_m, 0)

    def test_leap_year_day(self):
        """
        Test Case: Feb 29, 2012 (Leap Day)
        """
        serial = 734928.0
        expected_utc = datetime.datetime(2012, 2, 29, 0, 0, 0, tzinfo=pytz.utc)
        
        utc_dt, _, _, _ = parse_time_data(serial, float('nan'), float('nan'), 0)
        self.assertEqual(utc_dt, expected_utc)

    def test_day_after_leap_day(self):
        """
        Test Case: Mar 1, 2012
        Serial: 734929.0
        """
        serial = 734929.0 
        expected_utc = datetime.datetime(2012, 3, 1, 0, 0, 0, tzinfo=pytz.utc)
        
        utc_dt, _, _, _ = parse_time_data(serial, float('nan'), float('nan'), 0)
        self.assertEqual(utc_dt, expected_utc)

    def test_matlab_serial_with_fractional_time(self):
        """
        Test Case: Jan 1, 2013 at Noon (12:00)
        MATLAB Serial: 735235.5
        """
        serial = 735235.5
        expected_utc = datetime.datetime(2013, 1, 1, 12, 0, 0, tzinfo=pytz.utc)
        
        utc_dt, local_h, local_m, _ = parse_time_data(serial, float('nan'), float('nan'), 0)
        
        self.assertEqual(utc_dt, expected_utc)
        self.assertEqual(local_h, 12)
        self.assertEqual(local_m, 0)

    def test_standard_string_format(self):
        """Test Case: '20130630' with explicit Hour=15, Min=53"""
        date_str = "20130630"
        hour = 15
        minute = 53
        tz = 0
        expected_utc = datetime.datetime(2013, 6, 30, 15, 53, 0, tzinfo=pytz.utc)
        
        utc_dt, local_h, local_m, _ = parse_time_data(date_str, hour, minute, tz)
        
        self.assertEqual(utc_dt, expected_utc)
        self.assertEqual(local_h, 15)
        self.assertEqual(local_m, 53)

    def test_timezone_math(self):
        """
        Test Case: Timezone handling (EST = -5)
        Local: 12:00
        UTC:   17:00
        """
        date_str = "20130101"
        hour = 12
        minute = 0
        tz = -5.0
        
        # 12:00 Local - (-5) = 17:00 UTC
        expected_utc = datetime.datetime(2013, 1, 1, 17, 0, 0, tzinfo=pytz.utc)
        
        utc_dt, local_h, local_m, offset = parse_time_data(date_str, hour, minute, tz)
        
        self.assertEqual(utc_dt, expected_utc)
        self.assertEqual(local_h, 12)  # Should verify Local Time is preserved
        self.assertEqual(offset, -5.0)

    def test_24_hour_rollover(self):
        """Test Case: Jan 1, 24:00 -> Jan 2, 00:00 Local"""
        date_str = "20130101"
        hour = 24
        minute = 0
        tz = 0
        
        # The function rolls 24:00 to 00:00 of the NEXT day
        expected_utc = datetime.datetime(2013, 1, 2, 0, 0, 0, tzinfo=pytz.utc)
        
        utc_dt, local_h, local_m, _ = parse_time_data(date_str, hour, minute, tz)
        
        self.assertEqual(utc_dt, expected_utc)
        self.assertEqual(local_h, 0) # Local hour should be 0, not 24

    def test_invalid_24_hour_time(self):
        """Test Case: Invalid '24:15' time -> Should fail"""
        date_str = "20130101"
        hour = 24
        minute = 15 
        tz = 0
        
        result = parse_time_data(date_str, hour, minute, tz)
        # Expecting tuple of Nones
        self.assertEqual(result, (None, None, None, None))

    def test_historical_dates(self):
        """
        Verify offset holds for pre-2000 dates (e.g., 1950).
        MATLAB Jan 1, 1950 = 712224
        """
        serial = 712224.0
        expected_utc = datetime.datetime(1950, 1, 1, 0, 0, 0, tzinfo=pytz.utc)
        
        utc_dt, _, _, _ = parse_time_data(serial, float('nan'), float('nan'), 0)
        self.assertEqual(utc_dt, expected_utc)
        
if __name__ == '__main__':
    print("Running Verification Tests for Skyfinder Date Logic...")
    print("-" * 50)
    unittest.main(argv=['first-arg-is-ignored'], exit=False)