"""
create_splits.py

Site-Stratified Dataset Splitter.
Uses a two-stage stratified group split to ensure that rare meteorological events 
(NWS Category 3 & 4) are represented in Train, Validation, and Test sets, while 
maintaining strict camera-site isolation to prevent spatial data leakage.

LITERATURE CITATIONS:
- Data Leakage Prevention: Kaufman, S., et al. (2012). "Leakage in Data Mining: 
  Formulation, Detection, and Avoidance." ACM TKDD. https://doi.org/10.1145/2330667.2330670
- Group Stratification: Sechidis, K., et al. (2011). "On the Stratification of Multi-label Data." 
  ECML PKDD. (Adapted for group-based constraints).
"""

from __future__ import annotations
from pathlib import Path
from typing import Final, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  # type: ignore
from utils.nws_labels import get_nws_label

# --- Configuration & Constants ---
INPUT_CSV: Final[Path] = Path("data/clean_dataset.csv")
OUTPUT_DIR: Final[Path] = Path("data/splits")
TEST_SIZE: Final[float] = 0.15
VAL_SIZE: Final[float] = 0.15
RANDOM_STATE: Final[int] = 42

def perform_stratified_group_split(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Executes a two-stage split that preserves site-independence and label balance.

    This function groups the data by camera_id, identifies the maximum NWS risk 
    level recorded by that site, and uses that level as the stratification 
    key for splitting.

    Args:
        df: The cleaned dataset containing 'camera_id' and 'heat_index'.

    Returns:
        A tuple containing (train_df, val_df, test_df).
    """
    # 1. Pre-calculate NWS labels for stratification
    df['nws_label'] = df['heat_index'].apply(get_nws_label)
    
    # 2. Extract Site-Level Metadata
    # We stratify on the *Maximum Risk* a camera witnessed to ensure 
    # extreme event sites are distributed across all splits.
    camera_stats = df.groupby('camera_id')['nws_label'].max().reset_index()
    camera_stats.rename(columns={'nws_label': 'max_risk'}, inplace=True)

    # 3. Stage 1: Isolate Test Cameras
    train_val_cams, test_cams = train_test_split(
        camera_stats, 
        test_size=TEST_SIZE, 
        stratify=camera_stats['max_risk'], 
        random_state=RANDOM_STATE
    )

    # 4. Stage 2: Isolate Validation Cameras from Training pool
    # Adjust val_size to be relative to the remaining 85% of cameras
    relative_val_size: float = VAL_SIZE / (1.0 - TEST_SIZE)
    
    train_cams, val_cams = train_test_split(
        train_val_cams, 
        test_size=relative_val_size, 
        stratify=train_val_cams['max_risk'], 
        random_state=RANDOM_STATE
    )

    # 5. Map Site IDs back to the Full Image Dataset
    train_df = df[df['camera_id'].isin(train_cams['camera_id'])].copy()
    val_df = df[df['camera_id'].isin(val_cams['camera_id'])].copy()
    test_df = df[df['camera_id'].isin(test_cams['camera_id'])].copy()

    return train_df, val_df, test_df

def main() -> None:
    """Main execution routine for generating site-stratified splits."""
    if not INPUT_CSV.exists():
        print(f"main: {INPUT_CSV} not found. Ensure dataset preparation is complete.")
        return

    print(f"main: Loading {INPUT_CSV}...")
    df: pd.DataFrame = pd.read_csv(INPUT_CSV)
    
    # Execute split logic
    train_df, val_df, test_df = perform_stratified_group_split(df)

    # Save to disk
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, data in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out_path = OUTPUT_DIR / f"{name}.csv"
        data.to_csv(out_path, index=False)
        print(f"main: Saved {len(data)} rows to {out_path}")

    # --- Full Verification Report ---
    print("\n" + "═"*90)
    print(f"{'SPLIT':<10} | {'IMAGES':<8} | {'CAMS':<6} | {'CAT 0':<6} | {'CAT 1':<6} | {'CAT 2':<6} | {'CAT 3':<6} | {'CAT 4':<6}")
    print("-" * 90)
    for name, d in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        # Count all categories 0 through 4
        counts = [int((d['nws_label'] == i).sum()) for i in range(5)]
        cam_count: int = int(d['camera_id'].nunique())
        
        print(f"{name:<10} | {len(d):<8} | {cam_count:<6} | "
              f"{counts[0]:<6} | {counts[1]:<6} | {counts[2]:<6} | {counts[3]:<6} | {counts[4]:<6}")
    print("═"*90)

if __name__ == "__main__":
    main()