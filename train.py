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
        criterion (nn.Module): Loss function (MSELoss).
        optimizer (optim.Optimizer): Optimization algorithm (AdamW).
        device (str): Computation device ('cuda' or 'cpu').
        
    Returns:
        float: The average loss for this epoch.
    """
    model.train()
    running_loss = 0.0
    
    # Progress bar for training
    pbar = tqdm(loader, desc="Training", leave=False)
    
    for images, targets in pbar:
        images, targets = images.to(device), targets.to(device)
        
        # Forward pass
        # Model outputs [Batch, 1], targets are [Batch].
        # We un-squeeze targets to match output shape: [Batch, 1]
        outputs = model(images)
        loss = criterion(outputs, targets.unsqueeze(1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient Clipping
        # This prevents the "bouncing" by capping the maximum change (gradient) 
        # to 1.0. If a batch is weird, we limit the damage it can do.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
        # Update progress bar with current loss
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    # Cast dataset to Sized to satisfy mypy (DataLoader.dataset is Optional[Dataset])
    dataset_len = len(cast(Sized, loader.dataset))
    epoch_loss = running_loss / dataset_len
    return epoch_loss

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
        criterion (nn.Module): Loss function (MSELoss).
        device (str): Computation device.
        
    Returns:
        float: The average validation loss.
    """
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validating", leave=False)
        for images, targets in pbar:
            images, targets = images.to(device), targets.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, targets.unsqueeze(1))
            
            running_loss += loss.item() * images.size(0)
            
    dataset_len = len(cast(Sized, loader.dataset))
    epoch_loss = running_loss / dataset_len
    return epoch_loss

def main() -> None:
    """
    Main execution routine:
    1. Loads datasets.
    2. Initializes model, loss, and optimizer.
    3. Runs training loop.
    4. Saves best model based on validation loss.
    """
    print(f"main: using device {DEVICE}")
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    
    # 1. Prepare Data
    train_transforms = get_transforms('train') 
    val_transforms = get_transforms('val')     
    
    train_dataset = SkyfinderDataset(csv_path="data/splits/train.csv", transform=train_transforms)
    val_dataset = SkyfinderDataset(csv_path="data/splits/val.csv", transform=val_transforms)
    
    # num_workers=4 allows parallel loading of images (speeds up training significantly)
    # pin_memory=True speeds up transfer from CPU RAM to GPU VRAM
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    # --- 2. Prepare Model ---
    # Load EfficientNetV2 with ImageNet weights
    model = SkyFinderModel(pretrained=True).to(DEVICE)
    
    # --- 3. Setup Training ---
    # HuberLoss is robust against outliers (bad sensor data)
    # We have strict delta at 1.0 because heat index needs precision
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=2, verbose=True
    )
    
    best_val_loss = float('inf')
    
    # 4. Training Loop
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss = validate(model, val_loader, criterion, DEVICE)
        
        scheduler.step(val_loss)
        
        print(f"main: train_loss (Huber) is {train_loss:.4f} | val_loss (Huber) is {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"main: validation loss improved, saved to {CHECKPOINT_PATH}")

if __name__ == "__main__":
    main()