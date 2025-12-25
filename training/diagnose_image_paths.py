"""
diagnose_image_paths.py
Diagnoses why 'verify_pictures_integrity.py' is failing to find files that should exist.
"""
from pathlib import Path
import pandas as pd
import os

# --- Configuration ---
# Must match your previous script
IMAGE_ROOT_DIR = Path("data/skyfinder_images")
INPUT_CSV = Path("data/working_dataset.csv")

def main():
    print(f"main: loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    
    # 1. Check Data Types
    sample_id = df['camera_id'].iloc[0]
    print(f"main: 'camera_id' type in Pandas is: {type(sample_id)}")
    print(f"main: example raw value: {sample_id}")
    
    # 2. Find the first missing file to investigate
    print("\nmain: Searching for the first 'missing' file...")
    
    missing_example = None
    
    for _, row in df.iterrows():
        # Replicate the exact logic from your verification script
        camera_dir = str(row['camera_id'])
        fname = str(row['filename'])
        full_path = IMAGE_ROOT_DIR / camera_dir / fname
        
        if not full_path.exists():
            missing_example = (row, full_path, camera_dir)
            break
    
    if not missing_example:
        print("main: strangely, I cannot find any missing files now.")
        return

    # 3. Analyze the failure
    row, full_path, camera_dir_str = missing_example
    parent_dir = full_path.parent
    
    print(f"\n{'!'*10} FAILURE ANALYSIS {'!'*10}")
    print(f"The script failed to find:\n-> {full_path}")
    print(f"\nThis resolves to absolute path:\n-> {full_path.resolve()}")
    
    # 4. Check the directory structure
    print(f"\n--- Checking Parent Directory: {parent_dir} ---")
    if parent_dir.exists():
        print("main: the camera folder EXISTS.")
        print("Contents (first 5 files):")
        try:
            files = [f.name for f in parent_dir.iterdir() if f.is_file()]
            for f in files[:5]:
                print(f"   - {f}")
            if not files:
                print("   (Folder is empty)")
                
            # Check for nesting
            subdirs = [f.name for f in parent_dir.iterdir() if f.is_dir()]
            if subdirs:
                print(f"\nmain: I found subdirectories! Are images nested? {subdirs[:3]}")
        except Exception as e:
            print(f"Error reading folder: {e}")
    else:
        print("main: the camera folder DOES NOT EXIST.")
        print(f"I looked for: '{camera_dir_str}'")
        
        # Check if a similar folder exists (e.g., 10066 vs 10066.0)
        if IMAGE_ROOT_DIR.exists():
            potential_matches = list(IMAGE_ROOT_DIR.glob(f"*{str(int(float(row['camera_id'])))}*"))
            if potential_matches:
                print(f"\nDid you mean one of these? {potential_matches}")

if __name__ == "__main__":
    main()