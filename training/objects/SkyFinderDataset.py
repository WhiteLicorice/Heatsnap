"""
SkyFinderDataset.py

PyTorch Dataset implementation for the Skyfinder dataset.
Handles image loading, tensor normalization, and target extraction.

Current Configuration:
    - Augmentation: ENABLED (RandomFlip, Rotation, ColorJitter).
    - Preprocessing: Resize(256) -> CenterCrop(224) -> ImageNet Norm.
    - Metadata: Returns Cyclical Time (Sin/Cos) + Location (Lat/Lon).
"""

from __future__ import annotations
from typing_extensions import override
from pathlib import Path
import random
from typing import Tuple, Optional, Literal, Union, cast

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms  # type: ignore
from PIL import Image, ImageFile

# --- Configuration ---

# Allow loading of truncated images.
# This prevents crashes when an image file is missing a few bytes at the end.
# Hopefully, these few bytes won't cook our deep learning model.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# All pre-trained models expect input images normalized in the same way,
# i.e. mini-batches of 3-channel RGB images of shape (3 x H x W),
# where H and W are expected to be at least 224.
# The images have to be loaded in to a range of [0, 1] and then normalized
# using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
# From: https://pytorch.org/hub/pytorch_vision_fcn_resnet101/
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGE_ROOT = Path("data/skyfinder_images")

class SkyfinderDataset(Dataset):
    """
    PyTorch Dataset for loading Skyfinder images, Metadata, and Heat Index targets.
    Inherits from torch.utils.data.Dataset.
    """

    @override
    def __init__(
        self, 
        csv_path: Union[Path, str], 
        transform: Optional[transforms.Compose] = None,
        image_root: Path = IMAGE_ROOT
    ) -> None:
        """
        Initialize the dataset by loading the metadata CSV.

        Args:
            csv_path (Union[Path, str]): Path to the split CSV (train/val/test).
            transform (Optional[transforms.Compose]): Torchvision transforms pipeline.
            image_root (Path): Root directory containing camera folders.

        Raises:
            FileNotFoundError: If the provided CSV path does not exist.
        """
        self.csv_path = Path(csv_path)
        self.image_root = image_root
        self.transform = transform
        
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found at {self.csv_path}")
            
        self.df = pd.read_csv(self.csv_path)
        
        # Ensure camera_id matches folder names
        self.df['camera_id'] = self.df['camera_id'].astype(int).astype(str)
        self.df['filename'] = self.df['filename'].astype(str)

    @override
    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.df)
    
    @override
    def __getitem__(self, idx: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Retrieves the image, metadata, and target at the specified index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            Tuple[Tuple[Tensor, Tensor], Tensor]: 
                - inputs: ((image, metadata))
                - target: Heat Index value
        """
        row = self.df.iloc[idx]
        img_path = self.image_root / str(row['camera_id']) / str(row['filename'])
        
        try:
            with Image.open(img_path) as img:
                image = self.transform(img.convert("RGB"))
        except Exception as e:
            raise FileNotFoundError(f"SkyFinderDataset: error loading  {img_path}, {e}.")

        # raw_meta indices: 0:day_of_year, 1:hour, 2:lat, 3:lon, 4:elevation
        metadata = torch.tensor([
            float(row['day_of_year']),
            float(row['hour']),
            float(row['latitude']),
            float(row['longitude']),
            float(row['solar_elevation'])
        ], dtype=torch.float32)

        target = torch.tensor(float(row['heat_index']), dtype=torch.float32)
        return (image, metadata), target

def get_transforms(split: Literal['train', 'val', 'test']) -> transforms.Compose:
    """
    Returns the preprocessing pipeline for a given split.
    
    Args:
        split: 'train' gets heavy augmentation. 'val'/'test' get deterministic resize.
    """
    if split == 'train':
        # --- AUGMENTATION PIPELINE ---
        # 1. RandomResizedCrop: Randomly zoom into the sky (forces looking at details)
        # 2. RandomHorizontalFlip: Clouds don't care about left/right
        # 3. ColorJitter: Changes brightness/contrast to simulate different lighting conditions
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        # --- VALIDATION/TEST PIPELINE (As Is) ---
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])