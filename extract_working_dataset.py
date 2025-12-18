"""
extract_working_dataset.py

A strictly typed data processing pipeline to filter the Skyfinder dataset.
Adheres to PEP 484 type hinting and research-grade documentation standards.

Methodology References:
    - Solar Physics: Reda, I. & Andreas, A. (2004). Solar Position Algorithm for 
      Solar Radiation Applications. National Renewable Energy Laboratory (NREL).
    - Day/Night Threshold: U.S. Naval Observatory (USNO) (2013). Definition of 
      Civil Twilight (Sun at -6 degrees relative to horizon).
    - Heat Index Standard: Anderson, G. B., Bell, M. L., & Peng, R. D. (2013). 
      "Methods to Calculate the Heat Index as an Exposure Metric in Environmental 
      Health Research." Environmental Health Perspectives.
    - Regression Source: Rothfusz, L. P. (1990). "The Heat Index Equation." 
      NWS Technical Attachment (SR 90-23).
    - Baseline Theory: Steadman, R. G. (1979). "The Assessment of Sultriness." 
      Journal of Applied Meteorology.
"""

from __future__ import annotations
import math
from pathlib import Path
from typing import Final, Dict, Any

import pandas as pd
import numpy as np
from tqdm import tqdm

# Third-party libraries
try:
    from pysolar.solar import get_altitude  # type: ignore[import]
except ImportError:
    raise ImportError(
        "Module 'pysolar' not found. This script requires pysolar for "
        "accurate solar elevation filtering (Reda & Andreas, 2004)."
    )

# --- Configuration & Bounds ---
INPUT_CSV: Final[Path] = Path("data/complete_table_with_mcr.csv")
OUTPUT_CSV: Final[Path] = Path("data/working_dataset.csv")

# USNO (2013) Standard: Civil twilight is defined as solar elevation >= -6 degrees.
CIVIL_TWILIGHT_THRESHOLD: Final[float] = -6.0

# Physical constraints for Earth-based observations to filter sensor errors/sentinels
LAT_BOUNDS: Final[tuple[float, float]] = (-90.0, 90.0)
LON_BOUNDS: Final[tuple[float, float]] = (-180.0, 180.0)
TEMP_BOUNDS: Final[tuple[float, float]] = (-60.0, 140.0)  # Fahrenheit (Strict Earth limits)
HUM_BOUNDS: Final[tuple[float, float]] = (0.0, 100.0)     # Relative Humidity %


def calculate_heat_index(temp_f: float, hum: float) -> float:
    """
    Calculates the Heat Index (Apparent Temperature) using the multi-stage logic 
    standardized by Anderson et al. (2013) based on NWS protocols.

    The Heat Index (HI) is an index that combines air temperature and relative 
    humidity to posit a human-perceived equivalent temperature.

    Args:
        temp_f: Ambient temperature in degrees Fahrenheit.
        hum: Relative humidity expressed as a percentage (0-100).

    Returns:
        The calculated Heat Index. Returns the ambient temperature if HI is 
        irrelevant (cold) or np.nan if inputs are out of physical bounds.
    """
    # Validation against physical bounds (Filters sentinel values like -9999)
    if not (TEMP_BOUNDS[0] <= temp_f <= TEMP_BOUNDS[1]): return np.nan
    if not (HUM_BOUNDS[0] <= hum <= HUM_BOUNDS[1]): return np.nan

    # 1. Cold Domain: HI is not defined/meaningful for temperatures <= 40F.
    # Ref: Anderson et al. (2013) / NWS Protocol.
    if temp_f <= 40.0:
        return temp_f

    # 2. Simple Domain: Steadman's Linear Approximation.
    # Used when the simple calculation yields a value below 80F.
    # Formula: HI = 0.5 * {T + 61.0 + [(T-68.0)*1.2] + (RH*0.094)}
    # Constants Origin: Steadman (1979) baseline apparent temperature.
    hi_simple: float = -10.3 + (1.1 * temp_f) + (0.047 * hum)

    if hi_simple < 80.0:
        return hi_simple

    # 3. Hot Domain: Rothfusz Regression (Rothfusz, 1990).
    # This polynomial is a multi-parameter fit to Steadman's original tables.
    # The coefficients are "magic numbers" derived from the NWS regression analysis.
    hi: float = (
        -42.379
        + (2.04901523 * temp_f)
        + (10.14333127 * hum)
        - (0.22475541 * temp_f * hum)
        - (6.83783e-3 * temp_f**2)
        - (5.481717e-2 * hum**2)
        + (1.22874e-3 * temp_f**2 * hum)
        + (8.5282e-4 * temp_f * hum**2)
        - (1.99e-6 * temp_f**2 * hum**2)
    )
    
    # 4. Adjustment for Dry/Hot conditions (Rothfusz, 1990).
    # Applied if RH < 13% and Temp is between 80F and 112F.
    if hum < 13.0 and 80.0 <= temp_f <= 112.0:
        adjustment = ((13.0 - hum) / 4.0) * math.sqrt((17.0 - abs(temp_f - 95.0)) / 17.0)
        hi -= adjustment
        
    # 5. Adjustment for Humid/Hot conditions (Rothfusz, 1990).
    # Applied if RH > 85% and Temp is between 80F and 87F.
    elif hum > 85.0 and 80.0 <= temp_f <= 87.0:
        adjustment = ((hum - 85.0) / 10.0) * ((87.0 - temp_f) / 5.0)
        hi += adjustment

    return hi


def main() -> None:
    """
    Executes the data extraction pipeline:
    1. Loads raw Skyfinder/MCR table data.
    2. Performs strict sanitization of sensor and GPS outliers.
    3. Normalizes 24-hour time formats and converts to UTC.
    4. Applies Solar Position Algorithm (SPA) to isolate daytime samples.
    5. Calculates Heat Index targets for the supervised regression task.
    6. Formats and exports the final working dataset.
    """
    if not INPUT_CSV.exists():
        print(f"Error: {INPUT_CSV} not found.")
        return

    print(f"main: loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    initial_count: int = len(df)

    # --- 1. Strict Sanitization ---
    # Removes sentinel values (e.g., -9999) and sensor errors using physical bounds.
    print("main: sanitizing raw sensor and coordinate data...")
    valid_mask = (
        df['TempI'].between(*TEMP_BOUNDS) &
        df['Hum'].between(*HUM_BOUNDS) &
        df['Latitude'].between(*LAT_BOUNDS) &
        df['Longitude'].between(*LON_BOUNDS)
    )
    df = df[valid_mask].copy()
    print(f"main: dropped {initial_count - len(df)} rows due to invalid data.")

    # --- 2. Temporal Standardization ---
    tqdm.pandas()
    # Skyfinder uses separate columns for date/time components.
    time_cols: Dict[str, str] = {
        'Year': 'year', 'Month': 'month', 'Day': 'day', 
        'Hour': 'hour', 'Min': 'minute'
    }
    
    # Handle '24:00' timestamps which are invalid in Python's datetime.
    df.loc[df['Hour'] == 24, 'Hour'] = 0
    
    df['local_time'] = pd.to_datetime(
        df[list(time_cols.keys())].rename(columns=time_cols), 
        errors='coerce'
    )
    
    # Correct for Timezone to obtain UTC. Required for SPA (Reda & Andreas, 2004).
    df['utc_time'] = (
        df['local_time'] - pd.to_timedelta(df['Timezone'], unit='h')
    ).dt.tz_localize('UTC')
    
    df = df.dropna(subset=['utc_time'])

    # --- 3. Solar Physics Filtering ---
    # Filters for daytime only (Solar elevation >= -6 degrees).
    print("main: calculating solar elevation (SPA)...")
    df['solar_elevation'] = df.progress_apply(
        lambda r: get_altitude(r['Latitude'], r['Longitude'], r['utc_time']), 
        axis=1
    ) # type: ignore[attr-defined, operator]
    
    df = df[df['solar_elevation'] >= CIVIL_TWILIGHT_THRESHOLD].copy()

    # --- 4. Target Generation ---
    # Logic based on Anderson et al. (2013).
    print("main: calculating heat index (Rothfusz regression)...")
    df['heat_index'] = df.progress_apply(
        lambda r: calculate_heat_index(r['TempI'], r['Hum']), 
        axis=1
    ) # type: ignore[attr-defined, operator]

    # --- 5. Feature Extraction & Final Formatting ---
    df['day_of_year'] = df['utc_time'].dt.dayofyear
    
    # Explicit mapping for the PyDataset to ingest
    final_cols: Dict[str, str] = {
        'Filename': 'filename', 
        'CamId': 'camera_id', 
        'Latitude': 'latitude',
        'Longitude': 'longitude', 
        'solar_elevation': 'solar_elevation',
        'day_of_year': 'day_of_year', 
        'Hour': 'hour', 
        'heat_index': 'heat_index'
    }
    
    final_df = df[list(final_cols.keys())].rename(columns=final_cols).dropna()

    # Enforce integral types for discrete features
    final_df['camera_id'] = final_df['camera_id'].astype(int)
    final_df['day_of_year'] = final_df['day_of_year'].astype(int)
    final_df['hour'] = final_df['hour'].astype(int)

    # --- 6. Export ---
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT_CSV, index=False)
    
    print("-" * 30)
    print("Final Dataset Summary...")
    print(f"Total processed samples: {len(final_df)}")
    print(f"Yield: {(len(final_df)/initial_count)*100:.2f}% of raw input")
    print(f"Saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()