"""
extract_clean_dataset.py

Validates the integrity of the Skyfinder image dataset.
Iterates through the working CSV, checks if physical image files exist,
and verifies they are not corrupt JPEGs using Pillow.

Output:
    - data/clean_dataset.csv: A filtered CSV containing only valid image entries.
    - data/bad_files.txt: A log of missing or corrupt files.
"""

from pathlib import Path
from typing import Tuple, List
import pandas as pd
from PIL import Image
from tqdm import tqdm

# --- Configuration ---
# Using pathlib handles OS-specific separators automatically (Windows/Linux)
IMAGE_ROOT_DIR = Path("data/skyfinder_images")
INPUT_CSV = Path("data/working_dataset.csv")
OUTPUT_CSV = Path("data/clean_dataset.csv")
BAD_FILES_LOG = Path("data/bad_files.txt")

def verify_image(path: Path) -> Tuple[bool, str]:
    """
    Tries to open an image file to verify it is not corrupt.

    Args:
        path (Path): The full path to the image file.

    Returns:
        Tuple[bool, str]: A tuple containing:
            - bool: True if valid, False otherwise.
            - str: Status message ("OK" or error reason).
    """
    try:
        # Check existence first
        if not path.exists():
            return (False, "Missing (File not found)")

        # Verify it's a valid image structure (headers, etc.)
        # Image.verify() checks for truncated files or corrupt headers 
        # without the overhead of decoding pixel data.
        with Image.open(path) as img:
            img.verify()
        
        return (True, "OK")

    except Exception as e:
        return (False, f"Corrupt ({str(e)})")

def main() -> None:
    """
    Main execution routine for dataset integrity verification.
    """
    if not INPUT_CSV.exists():
        print(f"main: {INPUT_CSV} not found. Run extract_working_dataset.py first.")
        return

    print(f"main: loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    
    valid_rows = []
    issues: List[Tuple[str, str]] = []

    print(f"main: verifying {len(df)} images in '{IMAGE_ROOT_DIR}'...")
    
    # Iterate through the dataset using tqdm for progress tracking
    for _, row in tqdm(df.iterrows(), total=len(df)):
        
        # --- PATH CONSTRUCTION ---
        # Structure: data/skyfinder_images/{camera_id}/{filename}
        camera_dir = str(row['camera_id'])
        fname = str(row['filename'])
        
        # Pathlib allows simpler joining with "/" operator
        full_path = IMAGE_ROOT_DIR / camera_dir / fname
        
        # Verify
        is_valid, reason = verify_image(full_path)
        
        if is_valid:
            valid_rows.append(row)
        else:
            issues.append((str(full_path), reason))

    # --- REPORTING ---
    print(f"\nmain: {'=' * 30}")
    print("main: integrity check complete")
    print(f"main: {'=' * 30}")
    print(f"main: total scanned {len(df)}")
    print(f"main: valid images {len(valid_rows)}")
    print(f"main: issues found {len(issues)}")
    
    if issues:
        # Save bad files log
        with open(BAD_FILES_LOG, "w") as f:
            for path_str, reason in issues:
                f.write(f"{path_str},{reason}\n")
                
        print(f"\nmain: {len(issues)} images were missing or corrupt.")
        print(f"main: First failure example at {issues[0]}")
        print(f"main: Full list of bad files saved to {BAD_FILES_LOG}")
    else:
        print("\nmain: Dataset is 100% intact.")

    # Always save the clean CSV (even if identical, ensures consistency for next steps)
    clean_df = pd.DataFrame(valid_rows)
    clean_df.to_csv(OUTPUT_CSV, index=False)
    print(f"main: clean index saved to {OUTPUT_CSV}")
    print(f"main: use {OUTPUT_CSV} for all future steps.")

if __name__ == "__main__":
    main()