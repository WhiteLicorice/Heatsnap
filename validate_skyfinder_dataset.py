"""
validate_skyfinder_dataset.py

Standalone script to verify the SkyfinderDataset class.
Checks:
    1. File loading.
    2. Tensor shapes.
    3. Normalization statistics.
"""

from __future__ import annotations
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from objects.SkyFinderDataset import get_transforms, SkyfinderDataset

def main() -> None:
    """
    Executes a sanity check on the SkyfinderDataset.
    Verifies data loading, tensor shapes, and batch statistics.
    """
    print("main: running dataset sanity check...")
    
    train_csv = Path("data/splits/train.csv")
    if not train_csv.exists():
        print(f"main: {train_csv} not found. Run create_splits.py first.")
        return

    # 1. Initialize
    tsfm = get_transforms('train')
    ds: SkyfinderDataset = SkyfinderDataset(csv_path=train_csv, transform=tsfm)
    
    print(f"main: loaded {len(ds)} items from training split")
    
    # 2. Check Single Item
    img: torch.Tensor
    target: torch.Tensor
    img, target = ds[0]
    
    print(f"main: item[0] shape: {img.shape} (Expect [3, 224, 224])")
    print(f"main: item[0] label: {target.item():.2f} (Heat Index)")
    
    # 3. Check Batch Statistics
    print("main: checking batch statistics (first 32 images)...")
    loader: DataLoader = DataLoader(ds, batch_size=32, shuffle=True)
    
    # Fetch one batch
    batch_imgs: torch.Tensor
    batch_targets: torch.Tensor
    batch_imgs, batch_targets = next(iter(loader))
    
    # Calculate stats across the batch (N, C, H, W) -> mean over (N, H, W)
    mean: torch.Tensor = batch_imgs.mean(dim=[0, 2, 3])
    std: torch.Tensor = batch_imgs.std(dim=[0, 2, 3])
    
    print(f"main: batch mean: {mean.numpy()} (Should be approx 0.0)")
    print(f"main: batch std:  {std.numpy()}  (Should be approx 1.0)")
    
    print("main: sanity check passed")

if __name__ == "__main__":
    main()