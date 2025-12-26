"""
train.py

Main training pipeline for the Heatsnap Categorical Risk model.
Pivots from continuous regression to NWS-aligned classification to handle 
imbalanced data and improve safety-critical reliability.

LITERATURE CITATIONS:
- Weather Classification Strategy: Chu, W. T., et al. (2017). "Camera as weather sensor." 
  J. Vis. Commun. Image Represent. https://doi.org/10.1016/j.jvcir.2017.03.016
- Cost-Sensitive Learning: Sun, Y., et al. (2007). "Cost-sensitive learning for 
  imbalanced classification." IEEE ICDM. https://ieeexplore.ieee.org/document/4470208
- Imbalance Mitigation: He, H., & Garcia, E. A. (2009). "Learning from Imbalanced Data." 
  IEEE Trans. Knowl. Data Eng. https://ieeexplore.ieee.org/document/4781574
- Dataset Foundation: La Place, C., et al. (2018). "Segmenting Sky Pixels in Images." 
  arXiv:1712.09161. https://arxiv.org/abs/1712.09161
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, cast, Dict, List
from collections.abc import Sized
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import pandas as pd # type: ignore
import numpy as np
from tqdm import tqdm # type: ignore

# Custom Modules
from objects.SkyFinderDataset import SkyfinderDataset, get_transforms
from objects.SkyFinderModel import SkyFinderModel

# --- Configuration & Hyperparameters ---
# Batch Size: 32 is a standard balance for GPU memory (VRAM) vs training stability.
# If you get "CUDA Out of Memory", reduce this to 16.
BATCH_SIZE: int = 32
# Learning Rate: 1e-4 is the standard for fine-tuning EfficientNets.
# Too high (1e-3) destroys pretrained weights while too low (1e-5) takes forever.
LEARNING_RATE: float = 1e-4
NUM_EPOCHS: int = 20
NUM_CLASSES: int = 5 
WEIGHT_DECAY: float = 1e-3
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_CSV: Path = Path("data/splits/train.csv")
VAL_CSV: Path = Path("data/splits/val.csv")
CHECKPOINT_DIR: Path = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)
CHECKPOINT_PATH: Path = CHECKPOINT_DIR / "best_categorical_model.pth"

# NWS Thresholds & Bin Centers for "Virtual MAE" calculation
# Reference: NWS Heat Index Safety Guidelines. https://www.weather.gov/safety/heat-index
THRESHOLDS: List[int] = [80, 91, 104, 125]
BIN_CENTERS: torch.Tensor = torch.tensor([70.0, 85.5, 97.0, 114.5, 130.0])

def get_nws_label(hi: float) -> int:
    """
    Maps continuous Heat Index to NWS Risk Categories.
    
    Categories: 0 (Safe), 1 (Caution), 2 (Ex. Caution), 3 (Danger), 4 (Ex. Danger).
    """
    for i, threshold in enumerate(THRESHOLDS):
        if hi < threshold:
            return i
    return len(THRESHOLDS)

def calculate_virtual_mae(logits: torch.Tensor, real_hi: torch.Tensor) -> float:
    """
    Measures 'off-ness' in degrees Fahrenheit by comparing predicted bin 
    centers to ground truth heat index values.
    
    Enables comparison between classification and regression baselines.
    """
    _, preds = torch.max(logits, 1)
    centers: torch.Tensor = BIN_CENTERS.to(logits.device)
    pred_temps: torch.Tensor = centers[preds]
    return float(torch.abs(pred_temps - real_hi.view(-1)).mean().item())

def get_class_weights(csv_path: str | Path) -> torch.Tensor:
    """
    Calculates inverse frequency weights to penalize misclassification of rare 'Hot' samples.
    
    Citation: Sun, Y., et al. (2007). "Cost-sensitive learning for imbalanced classification."
    https://ieeexplore.ieee.org/document/4470208
    """
    df: pd.DataFrame = pd.read_csv(csv_path)
    labels: List[int] = [get_nws_label(float(x)) for x in df['heat_index']]
    counts: np.ndarray[Any, np.dtype[np.int64]] = np.bincount(labels, minlength=NUM_CLASSES)
    
    weights: np.ndarray[Any, np.dtype[np.float64]] = counts.sum() / (NUM_CLASSES * np.maximum(counts, 1))
    
    # Safety Override: Double the penalty for missing 'Danger' categories (Class 3 & 4)
    weights[3] *= 2.0
    weights[4] *= 2.0
    
    return torch.tensor(weights, dtype=torch.float).to(DEVICE)

def get_balanced_sampler(csv_path: str | Path) -> WeightedRandomSampler:
    """
    Ensures every batch contains a balanced representation of all 5 NWS categories.
    
    Citation: He, H., & Garcia, E. A. (2009). "Learning from Imbalanced Data."
    https://ieeexplore.ieee.org/document/4781574
    """
    df: pd.DataFrame = pd.read_csv(csv_path)
    labels: np.ndarray[Any, np.dtype[np.int64]] = np.array([get_nws_label(float(x)) for x in df['heat_index']])
    class_counts: np.ndarray[Any, np.dtype[np.int64]] = np.bincount(labels, minlength=NUM_CLASSES)
    
    weights: np.ndarray[Any, np.dtype[np.float64]] = 1. / np.maximum(class_counts, 1)
    sample_weights: torch.Tensor = torch.from_numpy(weights[labels]).double()
    
    # .tolist() converts the tensor to a Sequence[float] that mypy accepts
    return WeightedRandomSampler(sample_weights.tolist(), len(sample_weights), replacement=True)

def train_one_epoch(
    model: nn.Module, 
    loader: DataLoader[Any], 
    criterion: nn.Module, 
    optimizer: optim.Optimizer, 
    device: str
) -> Dict[str, float]:
    """Executes one full pass of training over the balanced dataset."""
    model.train()
    running_loss: float = 0.0
    running_mae: float = 0.0
    correct: int = 0
    total: int = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for (images, metadata), hi_targets in pbar:
        images, metadata = images.to(device), metadata.to(device)
        hi_targets = hi_targets.to(device).float()
        
        # Quantize targets on the fly
        class_labels: torch.Tensor = torch.tensor(
            [get_nws_label(float(x.item())) for x in hi_targets]
        ).long().to(device)
        
        optimizer.zero_grad()
        logits: torch.Tensor = model(images, metadata)
        loss: torch.Tensor = criterion(logits, class_labels)
        loss.backward()
        
        # Gradient Clipping: Prevents instability from rare high-loss samples
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += float(loss.item() * images.size(0))
        running_mae += calculate_virtual_mae(logits, hi_targets) * images.size(0)
        
        _, predicted = logits.max(1)
        total += int(class_labels.size(0))
        correct += int(predicted.eq(class_labels).sum().item())
        
        pbar.set_postfix({'vMAE': f"{calculate_virtual_mae(logits, hi_targets):.1f}°F"})
    
    dataset_size: int = len(cast(Sized, loader.dataset))
    return {
        "loss": running_loss / dataset_size, 
        "acc": 100.0 * correct / total, 
        "mae": running_mae / dataset_size
    }

def validate(
    model: nn.Module, 
    loader: DataLoader[Any], 
    criterion: nn.Module, 
    device: str
) -> Dict[str, float]:
    """Evaluates the model on unseen data using categorical metrics and Virtual MAE."""
    model.eval()
    running_loss: float = 0.0
    running_mae: float = 0.0
    correct: int = 0
    total: int = 0
    
    with torch.no_grad():
        for (images, metadata), hi_targets in tqdm(loader, desc="Validating", leave=False):
            images, metadata = images.to(device), metadata.to(device)
            hi_targets = hi_targets.to(device).float()
            
            class_labels: torch.Tensor = torch.tensor(
                [get_nws_label(float(x.item())) for x in hi_targets]
            ).long().to(device)
            
            logits: torch.Tensor = model(images, metadata)
            loss: torch.Tensor = criterion(logits, class_labels)
            
            running_loss += float(loss.item() * images.size(0))
            running_mae += calculate_virtual_mae(logits, hi_targets) * images.size(0)
            
            _, predicted = logits.max(1)
            total += int(class_labels.size(0))
            correct += int(predicted.eq(class_labels).sum().item())
            
    dataset_size: int = len(cast(Sized, loader.dataset))
    return {
        "loss": running_loss / dataset_size, 
        "acc": 100.0 * correct / total, 
        "mae": running_mae / dataset_size
    }

def main() -> None:
    """Main execution loop for model training and validation."""
    tqdm.write(f"Starting training on {DEVICE}...")
    
    # Training uses the Weighted Sampler; Validation uses standard Sequential
    train_loader: DataLoader[Any] = DataLoader(
        SkyfinderDataset(TRAIN_CSV, transform=get_transforms('train')),
        batch_size=BATCH_SIZE, 
        sampler=get_balanced_sampler(TRAIN_CSV), 
        num_workers=8
    )
    val_loader: DataLoader[Any] = DataLoader(
        SkyfinderDataset(VAL_CSV, transform=get_transforms('val')),
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=8
    )
    
    # Initialize 5-output model for NWS classification
    model: nn.Module = SkyFinderModel(pretrained=True, num_outputs=NUM_CLASSES).to(DEVICE)
    
    # Cost-Sensitive Loss Initialization
    class_weights: torch.Tensor = get_class_weights(TRAIN_CSV)
    criterion: nn.Module = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer: optim.Optimizer = optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY
    )
    
    # Scheduler monitors Validation Loss for learning rate reduction
    scheduler: optim.lr_scheduler.ReduceLROnPlateau = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=2
    )
    
    best_val_mae: float = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        start_time: float = time.time()
        
        train_metrics: Dict[str, float] = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_metrics: Dict[str, float] = validate(model, val_loader, criterion, DEVICE)
        
        # Step scheduler based on validation loss
        scheduler.step(val_metrics['loss'])
        
        duration: float = time.time() - start_time
        status: str = (f"Epoch {epoch+1:02d} | "
                       f"Train Acc: {train_metrics['acc']:.1f}% | "
                       f"Val Acc: {val_metrics['acc']:.1f}% | "
                       f"Val vMAE: {val_metrics['mae']:.1f}°F | {duration:.1f}s")
        tqdm.write(status)
        
        # Checkpoint based on Validation Virtual MAE (Parity Metric)
        if val_metrics['mae'] < best_val_mae:
            best_val_mae = val_metrics['mae']
            checkpoint_path: Path = CHECKPOINT_PATH
            torch.save(model.state_dict(), checkpoint_path)
            tqdm.write(f"--- Checkpoint Saved (Best Val vMAE: {best_val_mae:.2f}°F) ---")

if __name__ == "__main__":
    main()