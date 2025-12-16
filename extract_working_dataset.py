"""
extract_working_dataset.py
A data processing pipeline to filter the Skyfinder dataset.
Now robustly handles Excel/MATLAB serial dates by extracting time from the 
decimal component when the explicit 'Hour' column is NaN.

References:
    - USNO (2013): Civil Twilight Definition (-6 deg)
    - Reda & Andreas (2004): Solar Position Algorithm
    - Mihail, Workman, Bessinger, & Jacobs (2016): SkyFinder
"""

from __future__ import annotations
import pandas as pd
import datetime
import pytz
import math
import os
from typing import Optional, Any, cast

# Third-party libraries that may not have stubs installed by default
try:
    from pysolar.solar import get_altitude  # type: ignore[import]
except ImportError:
    # Fallback for type checking context if library missing
    get_altitude = lambda **kwargs: 0.0

from tqdm import tqdm

# --- Configuration ---
INPUT_CSV = "data/complete_table_with_mcr.csv"
OUTPUT_CSV = "data/working_dataset.csv"
CIVIL_TWILIGHT_THRESHOLD = -6.0

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
        # --- 1. Parse Timezone ---
        try:
            offset = float(tz_offset)
            if math.isnan(offset):
                return None
        except (ValueError, TypeError):
            return None

        # --- 2. Determine Date & Time Strategy ---
        has_explicit_time = False
        h, m = 0, 0
        try:
            h = int(hour)
            m = int(minute)
            has_explicit_time = True
        except (ValueError, TypeError):
            pass

        # --- 3. Parse Date (and Time if needed) ---
        d_str = str(date_val).split('.')[0]
        year, month, day = 0, 0, 0
        
        # Strategy A: Standard String (YYYYMMDD)
        if len(d_str) == 8 and d_str.isdigit():
            if not has_explicit_time: 
                return None 
            year = int(d_str[0:4])
            month = int(d_str[4:6])
            day = int(d_str[6:8])
            
            # Handle 24:00 wrap
            if h == 24:
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
                        h = 0
                        dt_date += datetime.timedelta(days=1)
                    local_dt = datetime.datetime.combine(dt_date, datetime.time(h, m))
                else:
                    # RECOVERY: Extract time from the serial fraction
                    fraction = serial - ordinal_int
                    seconds_in_day = fraction * 86400
                    local_dt = datetime.datetime.combine(dt_date, datetime.time(0, 0)) + \
                               datetime.timedelta(seconds=round(seconds_in_day))
                    
            except (ValueError, OverflowError):
                return None

        # --- 4. Convert to UTC ---
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

def main() -> None:
    if not os.path.exists(INPUT_CSV):
        print(f"[Error] {INPUT_CSV} not found.")
        return

    print(f"[Info] Loading {INPUT_CSV}...")
    # Use generic read_csv, typing inferred by pandas stubs
    df = pd.read_csv(INPUT_CSV)
    original_count = len(df)
    
    print("[Info] NOTE: 'Dirty' filter disabled to preserve dataset size.")
    
    # --- 2. Time Parsing ---
    print("[Info] Parsing timestamps (recovering time from serial dates)...")
    
    # Initialize tqdm for pandas
    tqdm.pandas(desc="Time Parsing")
    
    # MYPY FIX: Cast df to Any to allow calling the dynamic 'progress_apply' method
    # This prevents the "Series[Any] not callable" error without suppressing legitimate errors elsewhere.
    df_dynamic = cast(Any, df)
    
    df['utc_time'] = df_dynamic.progress_apply(
        lambda x: get_utc_datetime(x['Date'], x['Hour'], x['Min'], x['Timezone']), 
        axis=1
    )

    valid_time_mask = df['utc_time'].notna()
    dropped_dates = len(df) - valid_time_mask.sum()
    if dropped_dates > 0:
        print(f"[Warning] Dropped {dropped_dates} rows.")
    df = df[valid_time_mask].copy()

    # --- 3. Solar Physics ---
    print(f"[Info] Calculating Solar Elevation...")
    tqdm.pandas(desc="Solar Calc")
    
    # Refresh the dynamic cast for the new dataframe slice
    df_dynamic = cast(Any, df)
    df['solar_elevation'] = df_dynamic.progress_apply(calculate_solar_elevation, axis=1)

    # --- 4. Filter Night ---
    daytime_df = df[df['solar_elevation'] >= CIVIL_TWILIGHT_THRESHOLD].copy()
    
    # --- 5. Format Output ---
    target_columns = {
        'Filename': 'filename',
        'CamId': 'camera_id',
        'TempI': 'temp_f',
        'Hum': 'humidity',
        'solar_elevation': 'solar_elevation',
        'Latitude': 'latitude',
        'Longitude': 'longitude',
        'utc_time': 'timestamp'
    }
    
    available_cols = set(daytime_df.columns)
    required_cols = set(target_columns.keys())
    if not required_cols.issubset(available_cols):
        missing = required_cols - available_cols
        print(f"[Error] Missing columns: {missing}")
        return

    final_df = daytime_df[list(target_columns.keys())].rename(columns=target_columns)
    final_df = final_df.dropna(subset=['temp_f', 'humidity'])

    # --- 6. Save ---
    os.makedirs('dataset', exist_ok=True)
    final_df.to_csv(OUTPUT_CSV, index=False)
    
    print("-" * 40)
    print(f"Original: {original_count}")
    print(f"Parsed:   {len(df)}")
    print(f"Final:    {len(final_df)} (Daytime Only)")
    print("-" * 40)

if __name__ == "__main__":
    main()