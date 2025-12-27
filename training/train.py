"""
train.py

Main training pipeline for the Heatsnap Categorical Risk model.
Pivots from continuous regression to NWS-aligned classification to handle 
imbalanced data and improve safety-critical reliability.

ON DATASET LIMITATIONS:
Categories 3 (Danger) and 4 (Extreme Danger) are merged into a single 'High Risk'
class (Class 3) due to data scarcity in the >125°F range.

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
from typing import Any, Final, cast, Dict, List, Optional
from collections.abc import Sized
import time
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import pandas as pd # type: ignore
import numpy as np
from tqdm import tqdm # type: ignore
from sklearn.metrics import precision_recall_fscore_support # type: ignore

# Custom Modules
from objects.SkyFinderDataset import SkyfinderDataset, get_transforms
from objects.SkyFinderModel import SkyFinderModel
from objects.FocalLoss import FocalLoss
from utils.nws_labels import get_nws_label, BIN_CENTERS, NUMBER_OF_CLASSES
from utils.logs import log_and_print

# --- Configuration & Hyperparameters ---
# Batch Size: 32 is a standard balance for GPU memory (VRAM) vs training stability.
# If you get "CUDA Out of Memory", reduce this to 16.
BATCH_SIZE: int = 32
# Learning Rate: 1e-4 is the standard for fine-tuning EfficientNets.
# Too high (1e-3) destroys pretrained weights while too low (1e-5) takes forever.
LEARNING_RATE: float = 1e-4
NUM_EPOCHS: int = 20
NUM_CLASSES: int = NUMBER_OF_CLASSES
WEIGHT_DECAY: float = 1e-3
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_CSV: Path = Path("data/splits/train.csv")
VAL_CSV: Path = Path("data/splits/val.csv")
CHECKPOINT_DIR: Path = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)
CHECKPOINT: Path = CHECKPOINT_DIR / "best_categorical_model.pth"

TRAIN_LOGS_BASE: Final[Path] = Path("logs")

BIN_CENTERS_AS_TENSOR: torch.Tensor = torch.tensor(BIN_CENTERS)
WEIGHT_SAFE: Final[float] = 1.0
WEIGHT_CAUTION: Final[float] = 5.0
WEIGHT_EXT_CAUTION: Final[float] = 10.0
WEIGHT_DANGER: Final[float] = 15.0

def calculate_virtual_mae(logits: torch.Tensor, real_hi: torch.Tensor) -> float:
    """
    Measures 'off-ness' in degrees Fahrenheit by comparing predicted bin 
    centers to ground truth heat index values.
    """
    _, preds = torch.max(logits, 1)
    centers: torch.Tensor = BIN_CENTERS_AS_TENSOR.to(logits.device)
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
    
    weights[0] *= WEIGHT_SAFE
    weights[1] *= WEIGHT_CAUTION
    weights[2] *= WEIGHT_EXT_CAUTION
    weights[3] *= WEIGHT_DANGER
    
    return torch.tensor(weights, dtype=torch.float).to(DEVICE)

def get_balanced_sampler(csv_path: str | Path) -> WeightedRandomSampler:
    """
    Ensures every batch contains a balanced representation of all 4 NWS categories.
    
    Citation: He, H., & Garcia, E. A. (2009). "Learning from Imbalanced Data."
    https://ieeexplore.ieee.org/document/4781574
    """
    df: pd.DataFrame = pd.read_csv(csv_path)
    labels: np.ndarray = np.array([get_nws_label(float(x)) for x in df['heat_index']])
    class_counts: np.ndarray = np.bincount(labels, minlength=NUM_CLASSES)
    
    # Square root smoothing often works better for very rare classes than 1/N...
    weights: np.ndarray = 1. / np.sqrt(np.maximum(class_counts, 1))
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
        
        # Mapping HI to labels (Done on-device where possible is faster, 
        # but for complex logic like NWS labels, this list-comp is safe)
        labels = torch.tensor([get_nws_label(x.item()) for x in hi_targets], device=device).long()
        
        optimizer.zero_grad()
        logits: torch.Tensor = model(images, metadata)
        loss: torch.Tensor = criterion(logits, labels)
        loss.backward()
        
        # Gradient Clipping: Prevents instability from rare high-loss samples
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += float(loss.item() * images.size(0))
        running_mae += calculate_virtual_mae(logits, hi_targets) * images.size(0)
        _, predicted = logits.max(1)
        total += int(labels.size(0))
        correct += int(predicted.eq(labels).sum().item())
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
    device: str,
    log_path: Optional[Path] = None
) -> Dict[str, float]:
    """Evaluates the model with an F2-Score to prioritize Safety (Recall)."""
    model.eval()
    running_loss, running_mae = 0.0, 0.0
    correct, total = 0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for (images, metadata), hi_targets in tqdm(loader, desc="Validating", leave=False):
            images, metadata = images.to(device), metadata.to(device)
            hi_targets = hi_targets.to(device).float()
            
            labels = torch.tensor(
                [get_nws_label(float(x.item())) for x in hi_targets], 
                device=device
            ).long()
            
            logits = model(images, metadata)
            loss = criterion(logits, labels)
            
            batch_size = images.size(0)
            running_loss += float(loss.item() * batch_size)
            running_mae += calculate_virtual_mae(logits, hi_targets) * batch_size
            
            _, predicted = logits.max(1)
            total += int(labels.size(0))
            correct += int(predicted.eq(labels).sum().item())
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    precision, recall, f2, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=range(NUM_CLASSES), beta=2.0, zero_division=0
    )
    
    # Atomic logging of the Performance Table
    table = [
        "\n" + "═"*75,
        f"{'CATEGORY':<15} | {'PRECISION':<10} | {'RECALL (POD)':<12} | {'F2-SAFETY':<10} | {'SAMPLES':<8}",
        "-" * 75
    ]
    nws_names = ["Safe", "Caution", "Ex. Caution", "DANGER/EXT"]
    for i in range(NUM_CLASSES):
        table.append(f"{nws_names[i]:<15} | {precision[i]*100:>8.1f}% | {recall[i]*100:>10.1f}% | {f2[i]*100:>8.1f}% | {int(support[i]):<8}")
    table.append("═"*75)
    log_and_print("\n".join(table), log_path)

    dataset_size: int = len(cast(Sized, loader.dataset))
    return {
        "loss": running_loss / dataset_size, 
        "acc": 100.0 * correct / total, 
        "mae": running_mae / dataset_size,
        "macro_f2": float(np.mean(f2)) * 100,
        "macro_recall": float(np.mean(recall)) * 100
    }

def main() -> None:
    now = datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=8)))
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S") 
    training_dir = TRAIN_LOGS_BASE / timestamp
    training_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = training_dir / "logs.md"

    log_and_print(f"Starting training on {DEVICE}...", log_file_path)
    
    train_loader = DataLoader(
        SkyfinderDataset(TRAIN_CSV, transform=get_transforms('train')),
        batch_size=BATCH_SIZE,
        sampler=get_balanced_sampler(TRAIN_CSV),
        num_workers=8
    )
    val_loader = DataLoader(
        SkyfinderDataset(VAL_CSV, transform=get_transforms('val')),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8
    )
    
    model: nn.Module = SkyFinderModel(pretrained=True, num_outputs=NUM_CLASSES).to(DEVICE)
    class_weights: torch.Tensor = get_class_weights(TRAIN_CSV)
    criterion: nn.Module = FocalLoss(weight=class_weights, gamma=2)
    optimizer: optim.Optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=2
    )
    
    best_f2: float = 0.0
    
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)    
        val_metrics = validate(model, val_loader, criterion, DEVICE, log_path=log_file_path)
        
        scheduler.step(val_metrics['loss'])
        duration = time.time() - start_time
        status = (f"Epoch {epoch+1:02d} | Val Acc: {val_metrics['acc']:.1f}% | "
                  f"Safety (F2): {val_metrics['macro_f2']:.1f}% | "
                  f"Val vMAE: {val_metrics['mae']:.1f}°F | Time: {duration:.1f}s")
        
        log_and_print(status, log_file_path)
        
        if val_metrics['macro_f2'] > best_f2:
            best_f2 = val_metrics['macro_f2']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_f2': best_f2,
                'val_mae': val_metrics['mae']
            }, CHECKPOINT)
            log_and_print(f"--- Safety Checkpoint Saved (Best Macro F2: {best_f2:.2f}%) ---", log_file_path)

if __name__ == "__main__":
    main()