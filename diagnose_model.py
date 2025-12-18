"""
diagnose_model.py

Comprehensive Evaluation Suite for Heatsnap.

This script loads a trained model checkpoint and performs a "Post-Mortem" analysis 
to generate publication-quality statistics and figures. It does NOT retrain the model.

--- METRICS EXPLAINED ---

1. RMSE (Root Mean Squared Error):
   - What it is: The standard deviation of the prediction errors.
   - Purpose: Penalizes large outliers heavily. The standard for safety-critical apps.
   - Goal: Lower is better (e.g., < 10 degrees).

2. MAE (Mean Absolute Error):
   - What it is: The average absolute difference between predicted and actual values.
   - Purpose: Represents the "expected error" for a single prediction. easier for humans to understand.
   - Goal: Lower is better.

3. MBE (Mean Bias Error):
   - What it is: The average direction of the error (Predicted - Actual).
   - Purpose: Reveals systematic bias. 
     * Positive (+) = Model consistently overestimates heat.
     * Negative (-) = Model consistently underestimates heat.
   - Goal: Closer to 0.0 is better.

4. R² (Coefficient of Determination):
   - What it is: The proportion of variance in the dependent variable predictable from the independent variable.
   - Purpose: Measures "Goodness of Fit."
   - Goal: Closer to 1.0 is better (e.g., > 0.8 is strong correlation).

5. Pearson Correlation (r):
   - What it is: Measures the linear correlation between prediction and truth.
   - Purpose: Shows if the model "learned the pattern" even if the absolute numbers are off.
   - Goal: Closer to 1.0 is better.

6. Willmott's Index of Agreement (d):
   - What it is: A standardized measure of the degree of model prediction error.
   - Purpose: Specifically designed for environmental/meteorological models to be more robust than R².
   - Goal: Closer to 1.0 is better (e.g., > 0.9 is excellent).

7. Scatter Index (SI):
   - What it is: RMSE normalized by the mean of the actual data (%).
   - Purpose: Allows comparison of error rates across different datasets/units.
   - Goal: Lower is better (< 10% is high precision, < 20% is acceptable).

8. Tolerance Accuracy (±3° / ±5°):
   - What it is: The percentage of predictions that fall within X degrees of the truth.
   - Purpose: Practical usability metric. "How often is the model 'right enough'?"
   - Goal: Higher is better (e.g., > 80%).

--- FIGURES GENERATED ---

1. Scatter Plot (Actual vs. Predicted):
   - Visual proof of correlation. Points should cluster tightly around the diagonal red line.
   
2. Error Histogram (Residual Distribution):
   - Checks for Gaussian distribution of errors. A "bell curve" centered at 0 means the model is healthy and unbiased.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, cast
from collections.abc import Sized
from datetime import datetime
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# Custom Modules
from objects.SkyFinderDataset import SkyfinderDataset, get_transforms
from objects.SkyFinderModel import SkyFinderModel

# --- Configuration ---
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = Path("checkpoints/best_model.pth")

# Create timestamped results directory
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(f"results/{TIMESTAMP}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def calculate_willmott_d(actuals: np.ndarray, preds: np.ndarray) -> float:
    """
    Calculates Willmott's Index of Agreement (d).
    Range: 0.0 (mismatch) to 1.0 (perfect match).
    """
    numerator = np.sum((preds - actuals) ** 2)
    mean_actual = np.mean(actuals)
    denominator = np.sum((np.abs(preds - mean_actual) + np.abs(actuals - mean_actual)) ** 2)
    if denominator == 0:
        return 0.0
    return float(1 - (numerator / denominator))

def get_regime_stats(actuals: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    """Calculates MAE and RMSE for a specific array subset, ensuring JSON compatibility."""
    if len(actuals) == 0: 
        return {"mae": 0.0, "rmse": 0.0, "count": 0}
    
    errors = predictions - actuals
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    
    return {
        "mae": float(mae), 
        "rmse": float(rmse), 
        "count": int(len(actuals))
    }

def evaluate_metrics(loader: DataLoader[Any], model: nn.Module, device: str) -> None:
    """Runs inference and calculates publication-grade metrics with regime breakdown."""
    model.eval()
    all_preds, all_targets = [], []
    
    print(f"evaluate_metrics: processing {len(cast(Sized, loader.dataset))} images...")
    
    with torch.no_grad():
        for (images, metadata), targets in tqdm(loader, desc="Inference"):
            images, metadata = images.to(device), metadata.to(device)
            outputs = model(images, metadata) 
            
            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())
    
    # Convert to flattened numpy arrays
    predictions = torch.cat(all_preds).view(-1).numpy()
    actuals = torch.cat(all_targets).view(-1).numpy()
    
    # --- 1. Regime Analysis ---
    hot_mask = actuals > 80.0
    hot_stats = get_regime_stats(actuals[hot_mask], predictions[hot_mask])
    cold_stats = get_regime_stats(actuals[~hot_mask], predictions[~hot_mask])
    
    # --- 2. Global Metrics ---
    # We cast to float() to avoid NumPy JSON serialization errors
    mae = float(np.mean(np.abs(predictions - actuals)))
    rmse = float(np.sqrt(np.mean((predictions - actuals)**2)))
    
    ss_res = np.sum((actuals - predictions)**2)
    ss_tot = np.sum((actuals - np.mean(actuals))**2)
    r2 = float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0
    
    pearson_r = float(np.corrcoef(predictions, actuals)[0, 1])
    willmott_d = calculate_willmott_d(actuals, predictions)
    
    # --- 3. Report Generation ---
    report = {
        "metadata": {
            "timestamp": TIMESTAMP,
            "checkpoint": str(CHECKPOINT_PATH)
        },
        "global": {
            "mae": mae, 
            "rmse": rmse, 
            "r2": r2, 
            "pearson_r": pearson_r, 
            "willmott_d": willmott_d
        },
        "hot_regime": hot_stats,
        "cold_regime": cold_stats
    }
    
    # Save Metrics JSON
    metrics_path = OUTPUT_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(report, f, indent=4)

    # CLI Feedback
    print("\n" + "="*50)
    print(f"HEATSNAP DIAGNOSTIC: {TIMESTAMP}")
    print("="*50)
    print(f"GLOBAL MAE:          {mae:.2f}°F")
    print(f"GLOBAL R²:           {r2:.4f}")
    print("-" * 50)
    print(f"HOT REGIME (>80F) MAE:  {hot_stats['mae']:.2f}°F  (n={hot_stats['count']})")
    print(f"COLD REGIME (<80F) MAE: {cold_stats['mae']:.2f}°F (n={cold_stats['count']})")
    print("="*50)

    # --- 4. Generate Plots ---
    plot_results(actuals, predictions, r2, mae, hot_mask)

def plot_results(actuals, predictions, r2, mae, hot_mask):
    plt.figure(figsize=(15, 6))

    # --- Plot 1: Scatter (Colored by Regime) ---
    plt.subplot(1, 2, 1)
    plt.scatter(actuals[~hot_mask], predictions[~hot_mask], alpha=0.3, s=8, c='skyblue', label='Cold (<80°F)')
    plt.scatter(actuals[hot_mask], predictions[hot_mask], alpha=0.5, s=12, c='orangered', label='Hot (>80°F)')
    
    lims = [min(actuals.min(), predictions.min()), max(actuals.max(), predictions.max())]
    plt.plot(lims, lims, 'k--', alpha=0.7, label='Ideal')
    plt.title(f"Actual vs Predicted (R²={r2:.3f})")
    plt.xlabel("Ground Truth Heat Index (°F)")
    plt.ylabel("Predicted Heat Index (°F)")
    plt.legend()
    plt.grid(True, alpha=0.2)

    # --- Plot 2: Error Distribution ---
    plt.subplot(1, 2, 2)
    errors = predictions - actuals
    plt.hist(errors[~hot_mask], bins=50, color='skyblue', alpha=0.6, label='Cold Errors')
    plt.hist(errors[hot_mask], bins=50, color='orangered', alpha=0.6, label='Hot Errors')
    plt.axvline(0, color='black', lw=1)
    plt.title(f"Error Distribution (Total MAE={mae:.2f} °F)")
    plt.xlabel("Error (Predicted - Actual)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.2)

    save_path = OUTPUT_DIR / "diagnostic_plots.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\nplot_results: evaluation complete, results in {OUTPUT_DIR}")
    plt.close()

def main():
    if not CHECKPOINT_PATH.exists():
        print(f"main: no model at {CHECKPOINT_PATH}, run train.py first")
        return

    # Loading validation data
    val_dataset = SkyfinderDataset(csv_path="data/splits/val.csv", transform=get_transforms('val'))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Loading model
    model = SkyFinderModel(pretrained=False)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.to(DEVICE)
    
    evaluate_metrics(val_loader, model, DEVICE)

if __name__ == "__main__":
    main()