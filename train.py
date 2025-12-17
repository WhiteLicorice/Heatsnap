"""
train.py

Main training pipeline for the Skyfinder Heat Index regression model.
Orchestrates the data loading, model initialization, training loop, and validation.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, cast
from collections.abc import Sized

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from objects.SkyFinderDataset import SkyfinderDataset, get_transforms
from objects.SkyFinderModel import SkyFinderModel

# --- Configuration & Hyperparameters ---

# Batch Size: 32 is a standard balance for GPU memory (VRAM) vs training stability.
# If you get "CUDA Out of Memory", reduce this to 16.
BATCH_SIZE = 32

# Learning Rate: 1e-4 is the standard for fine-tuning EfficientNets.
# Too high (1e-3) destroys pretrained weights while too low (1e-5) takes forever.
LEARNING_RATE = 1e-4

# Epochs: 10 is usually enough for fine-tuning since the base model (EfficientNet)
# already knows how to "see" images.
NUM_EPOCHS = 10

# Device Selection: Auto-detects NVIDIA GPU.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Checkpoint Directory
SAVE_DIR = Path("checkpoints")
SAVE_DIR.mkdir(exist_ok=True)


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
    
    # tqdm provides a progress bar in the terminal
    pbar = tqdm(loader, desc="Training", unit="batch")
    
    for images, targets in pbar:
        # Move data to GPU
        images = images.to(device)
        
        # Reshape targets to [Batch_Size, 1] to match model output
        targets = targets.to(device).unsqueeze(1) 
        
        # Zero gradients from previous step
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        # Backward pass (calculate gradients)
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Accumulate loss (item() gets float from tensor)
        running_loss += loss.item() * images.size(0)
        
        # Update progress bar with current batch loss
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
    # We cast the dataset to Sized to satisfy type checkers that len() is valid.
    # PyTorch Datasets are not guaranteed to be Sized (e.g. Iterables), but ours is.
    dataset = cast(Sized, loader.dataset)
    epoch_loss = running_loss / len(dataset)
    
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
        for images, targets in tqdm(loader, desc="Validating", unit="batch"):
            images = images.to(device)
            targets = targets.to(device).unsqueeze(1)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * images.size(0)
            
    # Calculate average loss over the entire dataset
    dataset = cast(Sized, loader.dataset)
    epoch_loss = running_loss / len(dataset)
    
    return epoch_loss


def main() -> None:
    """
    Main execution routine:
    1. Loads datasets.
    2. Initializes model, loss, and optimizer.
    3. Runs training loop.
    4. Saves best model based on validation loss.
    """
    logging.basicConfig(level=logging.INFO, format='main: %(message)s')
    print(f"main: using device {DEVICE}")
    
    # --- 1. Prepare Data ---
    train_ds = SkyfinderDataset(
        csv_path="data/splits/train.csv", 
        transform=get_transforms('train')
    )
    val_ds = SkyfinderDataset(
        csv_path="data/splits/val.csv", 
        transform=get_transforms('val')
    )
    
    # num_workers=4 allows parallel loading of images (speeds up training significantly)
    # pin_memory=True speeds up transfer from CPU RAM to GPU VRAM
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    # --- 2. Prepare Model ---
    # Load EfficientNetV2 with ImageNet weights
    model = SkyFinderModel(pretrained=True).to(DEVICE)
    
    # --- 3. Setup Training ---
    # MSELoss is standard for regression tasks (penalizes large errors heavily)
    criterion = nn.MSELoss()
    
    # AdamW is the preferred optimizer for Transformers and Modern CNNs
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    best_val_loss = float('inf')
    
    # --- 4. Training Loop ---
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss = validate(model, val_loader, criterion, DEVICE)
        
        print(f"main: train_loss (MSE): {train_loss:.4f} | val_loss (MSE): {val_loss:.4f}")
        
        # RMSE (Root Mean Squared Error) puts the error back into original units (degrees)
        rmse = val_loss**0.5
        print(f"main: RMSE (approx error): {rmse:.2f} degrees")
        
        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = SAVE_DIR / "best_model.pth"
            torch.save(model.state_dict(), save_path)
            print(f"main: validation loss improved, saved to {save_path}")

if __name__ == "__main__":
    main()