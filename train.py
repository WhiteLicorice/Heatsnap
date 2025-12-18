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
    
    # Progress bar for training
    pbar = tqdm(loader, desc="Training", leave=False)
    
    # Unpack the tuple: ((images, metadata), targets)
    # metadata now contains [SinMonth, CosMonth, SinHour, CosHour, Lat, Lon]
    for inputs, targets in pbar:
        images, metadata = inputs
        
        # Move all tensors to the computation device
        images = images.to(device)
        metadata = metadata.to(device)
        targets = targets.to(device).unsqueeze(1) # Match output shape [Batch, 1]
        
        # Forward pass: Model handles 5->7 feature expansion internally
        outputs = model(images, metadata)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient Clipping
        # This prevents the "bouncing" by capping the maximum change (gradient) 
        # to 1.0. If a batch is weird, we limit the damage it can do.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
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
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validating", leave=False)
        for inputs, targets in pbar:
            images, metadata = inputs
            images, metadata = images.to(device), metadata.to(device)
            targets = targets.to(device).unsqueeze(1)
            
            outputs = model(images, metadata)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * images.size(0)
            
    dataset_len = len(cast(Sized, loader.dataset))
    return running_loss / dataset_len

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
    train_dataset = SkyfinderDataset(csv_path="data/splits/train.csv", transform=get_transforms('train'))
    val_dataset = SkyfinderDataset(csv_path="data/splits/val.csv", transform=get_transforms('val'))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
    
    # 2. Initialize Model
    # Physics branch now expects raw 5 features [day, hour, lat, lon, elev]
    model = SkyFinderModel(pretrained=True).to(DEVICE)
    
    # --- 3. Setup Training ---
    # HuberLoss is robust against outliers (bad sensor data)
    # We have strict delta at 1.0 because heat index needs precision
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=2,
    )
    
    best_val_loss = float('inf')
    
    # 4. Training Loop
    
    print(f"main: training for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss = validate(model, val_loader, criterion, DEVICE)
        
        scheduler.step(val_loss)
        
        duration = time.time() - start_time
        print(f"Epoch {epoch+1:02d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | {duration:.1f}s")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"main: validation loss improved, saved to {CHECKPOINT_PATH}")

if __name__ == "__main__":
    main()