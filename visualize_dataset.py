import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- Configuration ---
INPUT_CSV = Path("data/clean_dataset.csv")
SAVE_DIR = Path("analysis/detailed_plots")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

def main():
    if not INPUT_CSV.exists():
        print(f"main: {INPUT_CSV} not found. Check your file path.")
        return

    df = pd.read_csv(INPUT_CSV)
    sns.set_style("whitegrid")
    print(f"Loaded {len(df)} samples. Generating...")

    # 0. Descriptive Statistics
    stats = df.describe()
    stats.to_csv(SAVE_DIR / "0_descriptives.csv")
    #print("\n--- Summary Statistics ---")
    #print(stats[['solar_elevation', 'day_of_year', 'hour', 'heat_index']])
    
    # 1. Target Distribution: Heat Index
    plt.figure(figsize=(10, 6))
    sns.histplot(df['heat_index'], kde=True, color='crimson')
    plt.title("Target Distribution: Heat Index")
    plt.xlabel("Heat Index (°F)")
    plt.savefig(SAVE_DIR / "1_heat_index_distribution.png", dpi=300)
    plt.close()

    # 2. Solar Elevation vs Heat Index (Hexbin)
    plt.figure(figsize=(10, 8))
    hb = plt.hexbin(df['solar_elevation'], df['heat_index'], gridsize=35, cmap='YlOrRd')
    plt.colorbar(hb, label='Sample Density')
    plt.title("Physical Correlation: Solar Elevation vs Heat Index")
    plt.xlabel("Solar Elevation (Degrees)")
    plt.ylabel("Heat Index (°F)")
    plt.savefig(SAVE_DIR / "2_solar_vs_heat_hexbin.png", dpi=300)
    plt.close()

    # 3. Temporal Distribution (Samples per Hour)
    plt.figure(figsize=(12, 6))
    sns.countplot(x='hour', data=df, palette='viridis', hue='hour', legend=False)
    plt.title("Data Density by Hour of Day")
    plt.xlabel("Hour (24h Format)")
    plt.ylabel("Sample Count")
    plt.savefig(SAVE_DIR / "3_samples_per_hour.png", dpi=300)
    plt.close()

    # 4. Seasonal Distribution (Samples per Day of Year)
    plt.figure(figsize=(12, 6))
    sns.histplot(df['day_of_year'], bins=36, color='teal', kde=False)
    plt.title("Data Density across Calendar Year")
    plt.xlabel("Day of Year (1-366)")
    plt.savefig(SAVE_DIR / "4_seasonal_distribution.png", dpi=300)
    plt.close()

    # 5. Feature Correlation Heatmap
    plt.figure(figsize=(10, 8))
    cols = ['latitude', 'longitude', 'solar_elevation', 'day_of_year', 'hour', 'heat_index']
    corr = df[cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True)
    plt.title("Metadata Correlation Matrix")
    plt.savefig(SAVE_DIR / "5_feature_correlation.png", dpi=300)
    plt.close()

    # 6. Heat Index Variance by Hour (Boxplot)
    plt.figure(figsize=(12, 7))
    sns.boxplot(x='hour', y='heat_index', data=df, palette='magma', hue='hour', legend=False)
    plt.title("Daily Heat Index Variance (Diurnal Cycle)")
    plt.xlabel("Hour of Day")
    plt.ylabel("Heat Index (°F)")
    plt.savefig(SAVE_DIR / "6_hourly_heat_variance.png", dpi=300)
    plt.close()

    #print(f"Success: 6 plots saved to {SAVE_DIR}/")

if __name__ == "__main__":
    main()