"""
diagnose_data.py
Audit the raw Skyfinder dataset to understand the behavior of extract_working_dataset.py.
"""
import pandas as pd
import os

INPUT_CSV = "data/complete_table_with_mcr.csv"

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"[Error] Could not find {INPUT_CSV}")
        return

    print(f"[Info] Inspecting {INPUT_CSV}...")
    
    # Load all data as strings first to avoid pandas type inference masking issues
    df = pd.read_csv(INPUT_CSV, dtype=str)
    
    total_rows = len(df)
    print(f"\nTotal Rows in CSV: {total_rows}")
    print("=" * 40)

    # 1. CHECK THE 'DIRTY' FLAG
    if 'dirty' in df.columns:
        # Convert to numeric, errors='coerce' turns non-numbers to NaN
        dirty_vals = pd.to_numeric(df['dirty'], errors='coerce').fillna(0)
        clean_count = (dirty_vals == 0).sum()
        print(f"[Filter 1] Dirty Flag:")
        print(f"  - Total Clean (dirty=0): {clean_count}")
        print(f"  - Total Dirty (dirty=1): {total_rows - clean_count}")
    else:
        print("[Filter 1] 'dirty' column NOT FOUND (Skipping filter)")

    # 2. CHECK CRITICAL COLUMNS FOR MISSING VALUES
    print("\n[Filter 2] Missing Data Check:")
    critical_cols = ['Date', 'Timezone', 'Latitude', 'Longitude', 'TempI', 'Hum']
    for col in critical_cols:
        if col in df.columns:
            missing = df[col].isna().sum()
            print(f"  - {col}: {missing} missing values")
        else:
            print(f"  - {col}: COLUMN MISSING!")

    # 3. INSPECT DATE FORMATS
    print("\n[Filter 3] Date Format Inspection (First 10 non-null):")
    if 'Date' in df.columns:
        samples = df['Date'].dropna().head(10).tolist()
        print(f"  - Raw Samples: {samples}")
    
    # 4. INSPECT TIMEZONE
    print("\n[Filter 4] Timezone Inspection (First 10 non-null):")
    if 'Timezone' in df.columns:
        samples = df['Timezone'].dropna().head(10).tolist()
        print(f"  - Raw Samples: {samples}")

    print("=" * 40)
    print("Please copy and paste this output in the chat.")

if __name__ == "__main__":
    main()