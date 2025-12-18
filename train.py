"""
train.py

Main training pipeline for the Skyfinder Heat Index regression model.
Orchestrates the data loading, model initialization, training loop, and validation.
  - Uses ReduceLROnPlateau Scheduler to fix loss oscillation.
  - Uses Weight Decay for regularization.
  - Uses Gradient Clipping to prevent exploding gradients.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, cast
from collections.abc import Sized
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Custom Modules
from objects.SkyFinderDataset import SkyfinderDataset, get_transforms
from objects.SkyFinderModel import SkyFinderModel

# --- Configuration & Hyperparameters ---

# Batch Size: 32 is a standard balance for GPU memory (VRAM) vs training stability.
# If you get "CUDA Out of Memory", reduce this to 16.
BATCH_SIZE = 32

# Learning Rate: 1e-4 is the standard for fine-tuning EfficientNets.
# Too high (1e-3) destroys pretrained weights while too low (1e-5) takes forever.
LEARNING_RATE = 1e-4

# Adjust epochs as needed...
NUM_EPOCHS = 20
WEIGHT_DECAY = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)
CHECKPOINT_PATH = CHECKPOINT_DIR / "best_model.pth"

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
    running_loss = 0.0
    
    # We use tqdm.write() instead of print() to avoid breaking the bars
    pbar = tqdm(loader, desc="Training", leave=False)
    
    for inputs, targets in pbar:
        images, metadata = inputs
        images, metadata = images.to(device), metadata.to(device)
        targets = targets.to(device).unsqueeze(1)
        
        # 1. Forward Pass
        outputs = model(images, metadata)
        
        # 2. Weighted Loss Calculation
        # 'criterion' must have reduction='none' for this to work
        base_losses = criterion(outputs, targets) 
        
        # Priority Weighting: Make errors > 80F three times more "painful"
        # This combats the "Mean Regression" seen in descriptives.csv
        weights = torch.where(targets > 80.0, 3.0, 1.0)
        weighted_loss = (base_losses * weights).mean()
        
        # 3. Backward Pass
        optimizer.zero_grad()
        weighted_loss.backward()
        
        # Gradient Clipping
        # This prevents the "bouncing" by capping the maximum change (gradient) 
        # to 1.0. If a batch is weird, we limit the damage it can do.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Update metrics
        running_loss += weighted_loss.item() * images.size(0)
        pbar.set_postfix({'loss': f"{weighted_loss.item():.4f}"})
    
    dataset_len = len(cast(Sized, loader.dataset))
    return running_loss / dataset_len

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
    running_loss = 0.0
    
    # Validation usually stays unweighted for an "honest" metric
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Validating", leave=False):
            images, metadata = inputs
            images, metadata = images.to(device), metadata.to(device)
            targets = targets.to(device).unsqueeze(1)
            
            outputs = model(images, metadata)
            # Use .mean() here because criterion is in reduction='none' mode
            loss = criterion(outputs, targets).mean()
            running_loss += loss.item() * images.size(0)
            
    dataset_len = len(cast(Sized, loader.dataset))
    return running_loss / dataset_len

def main() -> None:
    # Use tqdm.write for all prints during the main loop
    tqdm.write(f"main: using device {DEVICE}")
    
    train_dataset = SkyfinderDataset(csv_path="data/splits/train.csv", transform=get_transforms('train'))
    val_dataset = SkyfinderDataset(csv_path="data/splits/val.csv", transform=get_transforms('val'))
    
    # num_workers=4 is usually the sweet spot for balance
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    model = SkyFinderModel(pretrained=True).to(DEVICE)
    
    # reduction='none' is mandatory for manual weighting
    criterion = nn.HuberLoss(delta=1.0, reduction='none')
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    
    best_val_loss = float('inf')
    
    tqdm.write(f"main: training for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss = validate(model, val_loader, criterion, DEVICE)
        
        scheduler.step(val_loss)
        
        duration = time.time() - start_time
        status = f"Epoch {epoch+1:02d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | {duration:.1f}s"
        tqdm.write(status)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            tqdm.write(f"--- Checkpoint Saved (Val Loss: {val_loss:.4f}, improved) ---")

if __name__ == "__main__":
    main()