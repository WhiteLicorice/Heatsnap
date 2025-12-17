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
import math
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

# Third-party libraries
try:
    from pysolar.solar import get_altitude  # type: ignore[import]
except ImportError:
    # Fallback to prevent crash if library is missing (though it should be installed)
    print("Module 'pysolar' not found. Solar calculations will fail.")
    get_altitude = lambda **kwargs: 0.0

# --- Configuration ---
INPUT_CSV = Path("data/complete_table_with_mcr.csv")
OUTPUT_CSV = Path("data/working_dataset.csv")

CIVIL_TWILIGHT_THRESHOLD = -6.0

def validate_input_columns(df: pd.DataFrame) -> bool:
    """
    Checks if explicit temporal columns exist. 
    We intentionally ignore 'Date' (float) as it represents file artifact time?
    """
    required = {
        'Year', 'Month', 'Day', 'Hour', 'Min', 'Timezone', 
        'Latitude', 'Longitude', 'TempI', 'Hum'
    }
    missing = required - set(df.columns)
    if missing:
        print(f"main: missing required input columns {missing}")
        return False
    return True

def calculate_solar_elevation(row: pd.Series) -> float:
    """
    Computes Solar Elevation Angle using the UTC timestamp.
    """
    if pd.isna(row['utc_time']):
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

def calculate_heat_index(row: pd.Series) -> float:
    """
    Calculates Heat Index following the Anderson et al. (2013) protocol.
    Ref: https://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml
    """
    try:
        T = float(row['TempI'])
        RH = float(row['Hum'])
        
        # QC: Sanity limits
        if math.isnan(T) or math.isnan(RH): return float('nan')
        # Sanity check: Earth temps only
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
    if not INPUT_CSV.exists():
        print(f"main: {INPUT_CSV} not found.")
        return

    print(f"main: loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    original_count = len(df)

    if not validate_input_columns(df):
        print("main: error on missing required columns.")
        return
    
    # Initialize tqdm for pandas
    tqdm.pandas()
    
    # --- 1. Normalize Time (Vectorized) ---
    print("main: standardizing timestamps to UTC using explicit columns...")
    
    # 1A. Construct Local Timestamp from explicit columns
    # This ignores the 'Date' serial column entirely.
    time_cols = ['Year', 'Month', 'Day', 'Hour', 'Min']
    
    # Ensure inputs are standard numeric (handle any stray strings)
    for c in time_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        
    # We rename columns temporarily for pd.to_datetime to recognize them
    df['local_timestamp'] = pd.to_datetime(df[time_cols].rename(columns={
        'Year': 'year', 'Month': 'month', 'Day': 'day', 'Hour': 'hour', 'Min': 'minute'
    }), errors='coerce')
    
    # 1B. Create UTC Timestamp (Local - TZ Offset)
    # The 'Timezone' column is the offset in hours (e.g., +1, -5).
    df['tz_delta'] = pd.to_timedelta(df['Timezone'], unit='h')
    df['utc_time'] = df['local_timestamp'] - df['tz_delta']
    
    # Add UTC timezone info for pysolar compatibility
    df['utc_time'] = df['utc_time'].dt.tz_localize('UTC')

    # Drop invalid times
    valid_time_mask = df['utc_time'].notna()
    dropped_count = len(df) - valid_time_mask.sum()
    if dropped_count > 0:
        print(f"main: dropped {dropped_count} rows due to invalid dates/times.")
    df = df[valid_time_mask].copy()

    # --- 2. Solar Physics ---
    print("main: calculating solar elevation...")
    df['solar_elevation'] = df.progress_apply(calculate_solar_elevation, axis=1) # type: ignore

    # --- 3. Filter Night ---
    daytime_df = df[df['solar_elevation'] >= CIVIL_TWILIGHT_THRESHOLD].copy()
    
    # --- 4. Calculate Heat Index ---
    print("main: calculating heat index...")
    daytime_df['heat_index'] = daytime_df.progress_apply(calculate_heat_index, axis=1) # type: ignore

    # --- 5. Extract Cyclic Features ---
    # We use the UTC timestamp for consistency, but Local Hour for daily cycle.
    daytime_df['month'] = daytime_df['utc_time'].dt.month
    daytime_df['local_hour'] = daytime_df['local_timestamp'].dt.hour
    
    # Month (1-12) -> sin/cos
    daytime_df['sin_month'] = np.sin(2 * np.pi * (daytime_df['month'] - 1) / 12)
    daytime_df['cos_month'] = np.cos(2 * np.pi * (daytime_df['month'] - 1) / 12)
    
    # Hour (0-23) -> sin/cos
    daytime_df['sin_hour'] = np.sin(2 * np.pi * daytime_df['local_hour'] / 24)
    daytime_df['cos_hour'] = np.cos(2 * np.pi * daytime_df['local_hour'] / 24)

    # --- 6. Format Output ---
    # Note: We need to make sure we map the original columns 'Day' and 'Min' 
    # to the final output.
    target_columns = {
        'Filename': 'filename',
        'CamId': 'camera_id',
        'TempI': 'temp_f',
        'Hum': 'humidity',
        'heat_index': 'heat_index',
        'solar_elevation': 'solar_elevation',
        'Latitude': 'latitude',
        'Longitude': 'longitude',
        'utc_time': 'timestamp',
        'month': 'month',
        'Day': 'day',
        'local_hour': 'hour',      # 0-23 Local
        'Min': 'minute',
        'sin_month': 'sin_month',
        'cos_month': 'cos_month',
        'sin_hour': 'sin_hour',
        'cos_hour': 'cos_hour'
    }
    
    final_df = daytime_df[list(target_columns.keys())].rename(columns=target_columns)
    
    # Final QC
    final_df = final_df.dropna(subset=['temp_f', 'humidity', 'heat_index', 'month', 'hour'])

    # Integer casting
    final_df['camera_id'] = final_df['camera_id'].astype(int)
    final_df['month'] = final_df['month'].astype(int)
    final_df['day'] = final_df['day'].astype(int)
    final_df['hour'] = final_df['hour'].astype(int)
    final_df['minute'] = final_df['minute'].astype(int)

    # --- 7. Save ---
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT_CSV, index=False)
    
    print("-" * 40)
    print(f"Original: {original_count}")
    print(f"Parsed:   {len(df)}")
    print(f"Daytime:  {len(daytime_df)}")
    print(f"Final:    {len(final_df)}")
    print("-" * 40)
    print(f"Saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()