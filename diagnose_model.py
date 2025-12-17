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
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

def calculate_willmott_d(actuals: np.ndarray, preds: np.ndarray) -> float:
    """
    Calculates Willmott's Index of Agreement (d).
    Range: 0.0 (mismatch) to 1.0 (perfect match).
    """
    numerator = np.sum((preds - actuals) ** 2)
    
    mean_actual = np.mean(actuals)
    denominator = np.sum(
        (np.abs(preds - mean_actual) + np.abs(actuals - mean_actual)) ** 2
    )
    
    if denominator == 0:
        return 0.0
    return 1 - (numerator / denominator)

def evaluate_metrics(
    loader: DataLoader[Any], 
    model: nn.Module, 
    device: str
) -> None:
    """
    Runs inference and calculates publication-grade metrics.
    """
    model.eval()
    
    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    
    dataset_len = len(cast(Sized, loader.dataset))
    print(f"eval: Processing {dataset_len} images...")
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Evaluating")
        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            
            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())
    
    # Flatten arrays
    predictions = torch.cat(all_preds).view(-1).numpy()
    actuals = torch.cat(all_targets).view(-1).numpy()
    
    # --- 1. Basic Error Metrics ---
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))
    mbe = np.mean(predictions - actuals) # Mean Bias Error
    
    # --- 2. Advanced Correlation Metrics ---
    
    # R2 (Coefficient of Determination)
    target_mean = np.mean(actuals)
    ss_tot = np.sum((actuals - target_mean) ** 2)
    ss_res = np.sum((actuals - predictions) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Pearson Correlation (r)
    # [0, 1] is correlation matrix. [0, 1] is the r value.
    pearson_r = np.corrcoef(predictions, actuals)[0, 1]
    
    # Willmott's Index of Agreement (d)
    willmott_d = calculate_willmott_d(actuals, predictions)
    
    # Scatter Index (SI)
    scatter_index = (rmse / target_mean) * 100
    
    # Tolerance Accuracy
    errors = np.abs(predictions - actuals)
    acc_3deg = np.mean(errors <= 3.0) * 100
    acc_5deg = np.mean(errors <= 5.0) * 100
    
    print("\n" + "="*50)
    print("FINAL PUBLICATION METRICS")
    print("="*50)
    print(f"RMSE (Root Mean Sq Error):    {rmse:.4f}")
    print(f"MAE  (Mean Abs Error):        {mae:.4f}")
    print(f"MBE  (Mean Bias Error):       {mbe:.4f}")
    print(f"SI   (Scatter Index):         {scatter_index:.2f}%")
    print("-" * 50)
    print(f"R^2  (Coeff. Determination):  {r2:.4f}")
    print(f"r    (Pearson Correlation):   {pearson_r:.4f}")
    print(f"d    (Willmott Index):        {willmott_d:.4f}")
    print("-" * 50)
    print(f"Accuracy (±3° tolerance):     {acc_3deg:.2f}%")
    print(f"Accuracy (±5° tolerance):     {acc_5deg:.2f}%")
    print("="*50)

    # --- 3. Generate Plots ---
    plot_results(actuals, predictions, r2, mae, mbe)

def plot_results(
    actuals: np.ndarray, 
    predictions: np.ndarray, 
    r2: float, 
    mae: float,
    mbe: float
) -> None:
    plt.figure(figsize=(14, 6))

    # --- Plot 1: Scatter ---
    plt.subplot(1, 2, 1)
    plt.scatter(actuals, predictions, alpha=0.4, s=10, c='blue', edgecolors='none', label='Data')
    
    min_val = min(actuals.min(), predictions.min())
    max_val = max(actuals.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit')
    
    plt.title(f"Predicted vs Actual (R²={r2:.3f})")
    plt.xlabel("Actual Heat Index")
    plt.ylabel("Predicted Heat Index")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # --- Plot 2: Residuals ---
    plt.subplot(1, 2, 2)
    residuals = predictions - actuals
    plt.hist(residuals, bins=40, color='purple', alpha=0.7, edgecolor='black')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=1)
    plt.axvline(x=mbe, color='red', linestyle='--', linewidth=2, label=f'Bias ({mbe:.2f})')
    
    plt.title(f"Error Distribution (MAE={mae:.2f})")
    plt.xlabel("Error (Predicted - Actual)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = RESULTS_DIR / "evaluation_plots.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\n[Success] Plots saved to {save_path}")
    plt.close()

def main() -> None:
    if not CHECKPOINT_PATH.exists():
        print(f"Error: Could not find checkpoint at {CHECKPOINT_PATH}")
        return

    print("eval: Loading validation data...")
    val_transforms = get_transforms('val')
    val_dataset = SkyfinderDataset(csv_path="data/splits/val.csv", transform=val_transforms)
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4
    )
    
    print(f"eval: Loading model from {CHECKPOINT_PATH}...")
    model = SkyFinderModel(pretrained=False)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.to(DEVICE)
    
    evaluate_metrics(val_loader, model, DEVICE)

if __name__ == "__main__":
    main()