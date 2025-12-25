"""
train.py

Main training pipeline for the Heatsnap regression model.
Implements WeightedRandomSampler to combat class imbalance and regression-to-the-mean.

LITERATURE CITATIONS:
- He, H., & Garcia, E. A. (2009). "Learning from Imbalanced Data." 
  IEEE Trans. Knowl. Data Eng. https://ieeexplore.ieee.org/document/4781574
- La Place, C., et al. (2018). "Segmenting Sky Pixels in Images." 
  arXiv:1712.09161. https://arxiv.org/abs/1712.09161
- Huber, P. J. (1964). "Robust Estimation of a Location Parameter."
  Ann. Math. Statist. https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-35/issue-1/Robust-Estimation-of-a-Location-Parameter/10.1214/aoms/1177703732.full
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, cast
from collections.abc import Sized
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from tqdm import tqdm

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
WEIGHT_DECAY: float = 1e-3
HOT_THRESHOLD: float = 80.0  # Threshold for "Extreme Heat" regime

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR: Path = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)
CHECKPOINT_PATH: Path = CHECKPOINT_DIR / "best_model.pth"

def get_balanced_sampler(csv_path: str | Path) -> WeightedRandomSampler:
    """
    Creates a sampler that oversamples the minority 'Hot' regime samples.
    
    This addresses the generalization ceiling identified in diagnostic tests.
    By balancing the training batch, we prevent the model from defaulting to 
    global mean predictions (~59°F).
    """
    df: pd.DataFrame = pd.read_csv(csv_path)
    heat_indices: np.ndarray = df['heat_index'].values
    
    # Define binary classes for sampling logic
    labels: np.ndarray = (heat_indices > HOT_THRESHOLD).astype(int)
    class_counts: np.ndarray = np.bincount(labels) # [count_cold, count_hot]
    
    # Weight per class is inverse frequency
    class_weights: np.ndarray = 1. / class_counts
    
    # Map weight back to every specific sample
    sample_weights: torch.Tensor = torch.from_numpy(class_weights[labels]).double()
    
    # Replacement=True allows the rare 'Hot' samples to be picked multiple times per epoch
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

def train_one_epoch(
    model: nn.Module, 
    loader: DataLoader[Any], 
    criterion: nn.Module, 
    optimizer: optim.Optimizer, 
    device: str
) -> float:
    """
    Executes one full pass (epoch) of training over the dataset.
    
    Args:
        model (nn.Module): The neural network.
        loader (DataLoader): Iterator for training data.
        criterion (nn.Module): Loss function (Huber).
        optimizer (optim.Optimizer): Optimization algorithm (AdamW).
        device (str): Computation device ('cuda' or 'cpu').
        
    Returns:
        float: The average loss for this epoch.
    """
    model.train()
    running_loss: float = 0.0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    
    for inputs, targets in pbar:
        images, metadata = inputs
        images, metadata = images.to(device), metadata.to(device)
        targets = targets.to(device).float().unsqueeze(1)
        
        # Forward Pass
        outputs: torch.Tensor = model(images, metadata)
        
        # Loss Calculation (reduction='none' for manual weighting)
        base_losses: torch.Tensor = criterion(outputs, targets) 
        
        # Triple Weight for Hot Regime (Safety-Critical Weighting)
        weights: torch.Tensor = torch.where(targets > HOT_THRESHOLD, 3.0, 1.0)
        weighted_loss: torch.Tensor = (base_losses * weights).mean()
        
        # Backward Pass
        optimizer.zero_grad()
        weighted_loss.backward()
        
        # Gradient Clipping
        # This prevents the "bouncing" by capping the maximum change (gradient) 
        # to 1.0. If a batch is weird, we limit the damage it can do.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += weighted_loss.item() * images.size(0)
        pbar.set_postfix({'loss': f"{weighted_loss.item():.4f}"})
    
    return running_loss / len(cast(Sized, loader.dataset))

def validate(
    model: nn.Module, 
    loader: DataLoader[Any], 
    criterion: nn.Module, 
    device: str
) -> float:
    """
    Evaluates the model on unseen validation data.
    Gradient calculation is disabled to save memory and speed up computation.
    
    Args:
        model (nn.Module): The neural network.
        loader (DataLoader): Iterator for validation data.
        criterion (nn.Module): Loss function.
        device (str): Computation device.
        
    Returns:
        float: The average validation loss.
    """
    model.eval()
    running_loss: float = 0.0
    
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Validating", leave=False):
            images, metadata = inputs
            images, metadata = images.to(device), metadata.to(device)
            targets = targets.to(device).float().unsqueeze(1)
            
            outputs: torch.Tensor = model(images, metadata)
            loss: torch.Tensor = criterion(outputs, targets).mean()
            running_loss += loss.item() * images.size(0)
            
    return running_loss / len(cast(Sized, loader.dataset))

def main() -> None:
    tqdm.write(f"main: using device {DEVICE}")
    
    train_csv: str = "data/splits/train.csv"
    val_csv: str = "data/splits/val.csv"

    # Initialize Datasets
    train_dataset = SkyfinderDataset(csv_path=train_csv, transform=get_transforms('train'))
    val_dataset = SkyfinderDataset(csv_path=val_csv, transform=get_transforms('val'))
    
    # Initialize Sampler for Imbalance Correction
    sampler: WeightedRandomSampler = get_balanced_sampler(train_csv)
    
    # Initialize Loaders
    # Note: shuffle must be False when using a sampler.
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        sampler=sampler, 
        num_workers=4, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    model: nn.Module = SkyFinderModel(pretrained=True).to(DEVICE)
    
    # Huber Loss is robust to outliers
    # reduction='none' is mandatory for manual weighting
    criterion: nn.Module = nn.HuberLoss(delta=1.0, reduction='none')
    optimizer: optim.Optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Fix loss oscillation via Plateau Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    
    best_val_loss: float = float('inf')
    
    tqdm.write(f"main: training for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        start_time: float = time.time()
        
        train_loss: float = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss: float = validate(model, val_loader, criterion, DEVICE)
        
        scheduler.step(val_loss)
        
        duration: float = time.time() - start_time
        status: str = f"Epoch {epoch+1:02d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | {duration:.1f}s"
        tqdm.write(status)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            tqdm.write(f"--- Checkpoint Saved (Val Loss: {val_loss:.4f}, improved) ---")

if __name__ == "__main__":
    main()