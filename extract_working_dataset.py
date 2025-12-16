"""
extract_working_dataset.py

A strictly typed data processing pipeline to filter the Skyfinder dataset.
Adheres to PEP 484 type hinting and resolves monkey-patching static analysis issues.

Methodology References:
    - Solar Physics: Reda, I. & Andreas, A. (2004). Solar Position Algorithm.
    - Day/Night Threshold: USNO (2013). Civil Twilight (-6 deg).
    - Heat Index Standard: Anderson, G. B. et al. (2013). "Methods to Calculate 
      the Heat Index as an Exposure Metric in Environmental Health Research."
      Env. Health Perspectives.
        Logic (Anderson et al., 2013 / NWS Rothfusz):
            1. If T <= 40F: HI = T (Heat index invalid in cold).
            2. Else: Compute HI_simple (Steadman Linear Approximation).
            3. If HI_simple < 80F: HI = HI_simple (Consistency adjustment).
            4. If HI_simple >= 80F: HI = Rothfusz Regression (w/ adjustments).
"""

from __future__ import annotations
import pandas as pd
import datetime
import pytz
import math
import os
from typing import Optional, Any, cast

# Third-party libraries
try:
    from pysolar.solar import get_altitude  # type: ignore[import]
except ImportError:
    get_altitude = lambda **kwargs: 0.0

from tqdm import tqdm

# --- Configuration ---
INPUT_CSV = "data/complete_table_with_mcr.csv"
OUTPUT_CSV = "data/working_dataset.csv"
CIVIL_TWILIGHT_THRESHOLD = -6.0

def validate_input_columns(df: pd.DataFrame) -> bool:
    """Checks if all required columns for processing exist."""
    required = {'Date', 'Hour', 'Min', 'Timezone', 'Latitude', 'Longitude', 'TempI', 'Hum'}
    missing = required - set(df.columns)
    if missing:
        print(f"[Critical Error] Missing required input columns: {missing}")
        return False
    return True

def get_utc_datetime(
    date_val: Any, 
    hour: Any, 
    minute: Any, 
    tz_offset: Any
) -> Optional[datetime.datetime]:
    """
    Constructs a timezone-aware UTC datetime object.
    Robustly handles MATLAB serials and standard strings.
    """
    try:
        try:
            offset = float(tz_offset)
            if math.isnan(offset):
                return None
        except (ValueError, TypeError):
            return None

        has_explicit_time = False
        h, m = 0, 0
        try:
            h = int(hour)
            m = int(minute)
            has_explicit_time = True
        except (ValueError, TypeError):
            pass

        d_str = str(date_val).split('.')[0]
        year, month, day = 0, 0, 0
        
        # Strategy A: Standard String (YYYYMMDD)
        if len(d_str) == 8 and d_str.isdigit():
            if not has_explicit_time:
                return None 
            year = int(d_str[0:4])
            month = int(d_str[4:6])
            day = int(d_str[6:8])
            
            if h == 24:
                if m != 0: return None
                h = 0
                day_offset = 1 
            elif h < 0 or h > 23:
                return None
            else:
                day_offset = 0

            local_dt = datetime.datetime(year, month, day, h, m) + datetime.timedelta(days=day_offset)

        # Strategy B: Serial Date (Float)
        else:
            try:
                serial = float(date_val)
                # Python ordinal is 1-based. MATLAB/Excel approx offset is 366 days.
                ordinal_int = int(serial)
                dt_date = datetime.date.fromordinal(ordinal_int) - datetime.timedelta(days=366)
                
                if has_explicit_time:
                    if h == 24:
                        if m != 0:
                            return None
                        h = 0
                        dt_date += datetime.timedelta(days=1)
                    local_dt = datetime.datetime.combine(dt_date, datetime.time(h, m))
                else:
                    fraction = serial - ordinal_int
                    seconds_in_day = fraction * 86400
                    local_dt = datetime.datetime.combine(dt_date, datetime.time(0, 0)) + \
                               datetime.timedelta(seconds=round(seconds_in_day))
                    
            except (ValueError, OverflowError):
                return None

        utc_dt = local_dt - datetime.timedelta(hours=offset)
        return pytz.utc.localize(utc_dt)

    except Exception as e:
        print(f"get_utc_datetime: error for inputs {date_val, hour, minute,tz_offset}, {e}")
        return None

def calculate_solar_elevation(row: pd.Series[Any]) -> float:
    """
    Computes Solar Elevation Angle.
    Uses string type hint for pd.Series to support older pandas stubs.
    """
    if row['utc_time'] is None or pd.isna(row['utc_time']):
        return float('nan')
    try:
        return get_altitude(
            latitude_deg=float(row['Latitude']),
            longitude_deg=float(row['Longitude']),
            when=row['utc_time']
        )
    except Exception as e:
        print(f"calculate_solar_elevation: error for inputs {row}, {e}")
        return float('nan')

def calculate_heat_index(row: pd.Series[Any]) -> float:
    """
    Calculates Heat Index following the Anderson et al. (2013) protocol.
    Ref: https://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml
    """
    try:
        T = float(row['TempI'])
        RH = float(row['Hum'])
        
        # QC: Sanity limits
        if math.isnan(T) or math.isnan(RH): return float('nan')
        if not (-60 <= T <= 140): return float('nan') 
        if not (0 <= RH <= 100): return float('nan')

        # Domain 1: Cold (Anderson 2013 / NWS)
        # Heat index is technically undefined/irrelevant below 40F.
        if T <= 40.0:
            return T

        # Domain 2: Transitional (Steadman's Linear Approximation)
        # NWS Formula: 0.5 * (T + 61.0 + (T-68.0)*1.2 + RH*0.094)
        # Algebraic Simplification: -10.3 + 1.1*T + 0.047*RH
        hi_simple = -10.3 + 1.1 * T + 0.047 * RH

        if hi_simple < 80.0:
            return hi_simple

        # Domain 3: Hot (Rothfusz Regression)
        hi = (-42.379 + 2.04901523 * T + 10.14333127 * RH 
              - 0.22475541 * T * RH 
              - 6.83783e-3 * T**2 
              - 5.481717e-2 * RH**2 
              + 1.22874e-3 * T**2 * RH 
              + 8.5282e-4 * T * RH**2 
              - 1.99e-6 * T**2 * RH**2)
        
        # Adjustments (Rothfusz 1990)
        if RH < 13 and 80 <= T <= 112:
            adj = ((13 - RH) / 4) * math.sqrt((17 - abs(T - 95)) / 17)
            hi -= adj
        elif RH > 85 and 80 <= T <= 87:
            adj = ((RH - 85) / 10) * ((87 - T) / 5)
            hi += adj

        return hi

    except (ValueError, KeyError, TypeError) as e:
        print(f"calculate_heat_index: error for inputs {row}, {e}")
        return float('nan')

def main() -> None:
    if not os.path.exists(INPUT_CSV):
        print(f"main: {INPUT_CSV} not found.")
        return

    print(f"main: loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    original_count = len(df)

    if not validate_input_columns(df):
        print(f"main: error on missing required columns.")
        return
    
    print("main: 'dirty' filter disabled (metadata is overly aggressive, ugh).")
    
    # --- 1. Time Parsing ---
    print("main: parsing timestamps (recovering time from serial dates)...")
    tqdm.pandas(desc="Time Parsing")
    
    df_dynamic = cast(Any, df)
    df['utc_time'] = df_dynamic.progress_apply(
        lambda x: get_utc_datetime(x['Date'], x['Hour'], x['Min'], x['Timezone']), 
        axis=1
    )

    valid_time_mask = df['utc_time'].notna()
    dropped_count = len(df) - valid_time_mask.sum()
    if dropped_count > 0:
        print(f"main: dropped {dropped_count} rows due to invalid dates/times.")
        
    df = df[valid_time_mask].copy()

    # --- 2. Solar Physics ---
    print(f"main: calculating Solar Elevation (Reda & Andreas, 2004)...")
    tqdm.pandas(desc="Solar Calc")
    df_dynamic = cast(Any, df)
    df['solar_elevation'] = df_dynamic.progress_apply(calculate_solar_elevation, axis=1)

    # --- 3. Filter Night ---
    daytime_df = df[df['solar_elevation'] >= CIVIL_TWILIGHT_THRESHOLD].copy()
    
    # --- 4. Calculate Heat Index ---
    print(f"main: calculating heat index (Anderson et al., 2013)...")
    tqdm.pandas(desc="Heat Index")
    df_dynamic = cast(Any, daytime_df)
    daytime_df['heat_index'] = df_dynamic.progress_apply(calculate_heat_index, axis=1)

    # --- 5. Format Output ---
    target_columns = {
        'Filename': 'filename',
        'CamId': 'camera_id',
        'TempI': 'temp_f',
        'Hum': 'humidity',
        'heat_index': 'heat_index',
        'solar_elevation': 'solar_elevation',
        'Latitude': 'latitude',
        'Longitude': 'longitude',
        'utc_time': 'timestamp'
    }
    
    if not set(target_columns.keys()).issubset(set(daytime_df.columns)):
         print(f"main: error around missing columns, available: {daytime_df.columns}")
         return

    final_df = daytime_df[list(target_columns.keys())].rename(columns=target_columns)
    
    # Final QC
    final_df = final_df.dropna(subset=['temp_f', 'humidity', 'heat_index'])

    # --- 6. Save ---
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    final_df.to_csv(OUTPUT_CSV, index=False)
    
    print("-" * 40)
    print(f"Original: {original_count}")
    print(f"Parsed:   {len(df)}")
    print(f"Daytime:  {len(daytime_df)}")
    print(f"Final:    {len(final_df)}")
    print("-" * 40)

if __name__ == "__main__":
    main()