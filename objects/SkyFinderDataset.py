"""
SkyFinderDataset.py

PyTorch Dataset implementation for the Skyfinder dataset.
Handles image loading, tensor normalization, and target extraction.

Current Configuration:
    - Augmentation: DISABLED (Baseline mode).
    - Preprocessing: Resize(256) -> CenterCrop(224) -> ImageNet Norm.
    - Robustness: Tolerates truncated/slightly corrupt images.
"""

from __future__ import annotations
from pathlib import Path
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
    PyTorch Dataset for loading Skyfinder images and Heat Index targets.
    Inherits from torch.utils.data.Dataset.
    """

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
        
        # Ensure camera_id matches folder names (integers as strings)
        self.df['camera_id'] = self.df['camera_id'].astype(int).astype(str)
        self.df['filename'] = self.df['filename'].astype(str)

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves the image and target at the specified index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - image: The preprocessed image tensor (C, H, W).
                - target: The Heat Index value as a float tensor.

        Raises:
            OSError, FileNotFoundError: If the image file cannot be opened.
        """
        row = self.df.iloc[idx]
        
        img_path = self.image_root / row['camera_id'] / row['filename']
        
        try:
            with Image.open(img_path) as img:
                # Always convert to RGB (handles occasional RGBA or Grayscale images)
                image_pil = img.convert("RGB")
                
                if self.transform:
                    # Apply transform pipeline. 
                    # We cast to torch.Tensor because generic transforms can return Any,
                    # but we know our pipeline ends with ToTensor + Normalize.
                    image_t = self.transform(image_pil)
                    image = cast(torch.Tensor, image_t)
                else:
                    # Fallback: Convert to tensor manually if no transform provided
                    # to match the return signature (though normally we always provide transforms)
                    image = transforms.functional.to_tensor(image_pil)
                    
        except (OSError, FileNotFoundError) as e:
            # We catch the error to print context, but we re-raise it because
            # PyTorch's DataLoader usually handles occasional skipped errors 
            # if we wanted to filter them out, but better to fix the root cause 
            print(f"Error loading {img_path}: {e}")
            raise e

        # Target: Heat Index (Float)
        # We return it as a tensor of shape (1,) or scalar depending on loss function needs.
        target = torch.tensor(float(row['heat_index']), dtype=torch.float32)
        
        return image, target

def get_transforms(split: Literal['train', 'val', 'test']) -> transforms.Compose:
    """
    Returns the preprocessing pipeline for a given split.
    Currently identical for all splits (NO AUGMENTATION).

    Args:
        split (Literal['train', 'val', 'test']): The dataset split (unused logic currently).

    Returns:
        transforms.Compose: The composition of image transformations.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])