"""
create_splits.py

Splits the cleaned Skyfinder dataset into Train, Validation, and Test sets.
Uses 'GroupShuffleSplit' around the 'camera_id' column to ensure site-stratified splits.
This prevents the model from memorizing static backgrounds by ensuring images from
a specific camera appear in only one split (Train, Val, or Test).

Output:
    - data/splits/train.csv
    - data/splits/val.csv
    - data/splits/test.csv
"""

from pathlib import Path
from typing import Set, Tuple
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit # type: ignore

# --- Configuration ---
INPUT_CSV = Path("data/clean_dataset.csv")
OUTPUT_DIR = Path("data/splits")

# Split Ratios (approximate, as we split by groups/cameras, not exact row counts)
TEST_SIZE = 0.15  # 15% of cameras for Test
VAL_SIZE = 0.15   # 15% of cameras for Validation
# Remainder (70%) is for Train

def validate_no_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
    """
    Verifies that no camera ID appears in more than one split.
    
    Args:
        train_df (pd.DataFrame): Training data.
        val_df (pd.DataFrame): Validation data.
        test_df (pd.DataFrame): Test data.
        
    Returns:
        bool: True if splits are disjoint (valid), False if leakage is detected.
    """
    train_cams: Set[int] = set(train_df['camera_id'].unique())
    val_cams: Set[int] = set(val_df['camera_id'].unique())
    test_cams: Set[int] = set(test_df['camera_id'].unique())
    
    # Check intersection between any pair
    leakage = (train_cams & val_cams) | (train_cams & test_cams) | (val_cams & test_cams)
    
    if leakage:
        print(f"main: data leak detected, overlapping cameras {leakage}")
        return False
        
    print("main: integrity check passed, no overlapping cameras between sets")
    return True

def perform_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Executes the two-stage GroupShuffleSplit to separate Test, Validation, and Train sets.
    
    Args:
        df (pd.DataFrame): The complete cleaning dataset.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, val_df, test_df)
    """
    groups = df['camera_id']

    # --- Step 1: Split off the Test Set ---
    splitter_test = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=69)
    train_val_idx, test_idx = next(splitter_test.split(df, groups=groups))
    
    train_val_df = df.iloc[train_val_idx].copy()
    test_df = df.iloc[test_idx].copy()

    # --- Step 2: Split the remaining (Train+Val) into Train and Val ---
    # Adjust val_size relative to the remaining data
    # We want 15% of TOTAL to be Val. We just removed 15%. 
    # Remaining is 85%. 0.15 / 0.85 ~= 0.176
    relative_val_size = VAL_SIZE / (1.0 - TEST_SIZE)
    
    # We define new groups based on the subset
    tv_groups = train_val_df['camera_id']
    splitter_val = GroupShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=69)
    
    train_idx, val_idx = next(splitter_val.split(train_val_df, groups=tv_groups))
    
    train_df = train_val_df.iloc[train_idx].copy()
    val_df = train_val_df.iloc[val_idx].copy()
    
    return train_df, val_df, test_df

def main() -> None:
    """
    Main execution routine for generating dataset splits.
    """
    if not INPUT_CSV.exists():
        print(f"main: {INPUT_CSV} not found, run verify_pictures_integrity.py first")
        return

    print(f"main: loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    
    n_cameras = df['camera_id'].nunique()
    print(f"main: dataset contains {len(df)} images from {n_cameras} unique cameras")

    # Perform Splitting
    train_df, val_df, test_df = perform_split(df)

    # --- Reporting & Saving ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
    val_df.to_csv(OUTPUT_DIR / "val.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)
    
    print(f"main: {'-' * 30}")
    print("main: split complete (site-stratified)")
    print(f"main: {'-' * 30}")
    print(f"main: train {len(train_df):5d} images ({train_df['camera_id'].nunique()} cameras)")
    print(f"main: val   {len(val_df):5d} images ({val_df['camera_id'].nunique()} cameras)")
    print(f"main: test  {len(test_df):5d} images ({test_df['camera_id'].nunique()} cameras)")
    print(f"main: {'-' * 30}")
    
    validate_no_leakage(train_df, val_df, test_df)

if __name__ == "__main__":
    main()