"""
SkyFinderDataset.py

PyTorch Dataset implementation for the Skyfinder dataset.
Extracts images and heat index targets, serving both regression and categorical workflows.

LITERATURE CITATIONS:
- Dataset Reference: La Place, C., et al. (2018). "Segmenting Sky Pixels in Images." 
  arXiv:1712.09161. https://arxiv.org/abs/1712.09161
- Input Normalization: PyTorch Vision Models Documentation. 
  https://pytorch.org/vision/stable/models.html
- Image Augmentation Strategy: Chu, W. T., et al. (2017). "Camera as weather sensor."
  J. Vis. Commun. Image Represent. https://doi.org/10.1016/j.jvcir.2017.03.016
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional, Literal, Union, List, Any
from typing_extensions import TypeAlias

import pandas as pd
import torch # type: ignore
from torch.utils.data import Dataset
from torchvision import transforms  # type: ignore
from PIL import Image, ImageFile

# --- Type Aliases for Mypy ---
# Returns: ((ImageTensor, MetadataTensor), HeatIndexTarget)
DatasetItem: TypeAlias = Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]

# --- Configuration ---

# Enables loading of files with missing end-of-file markers, common in large webcam crawls.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# All pre-trained models expect input images normalized in the same way,
# i.e. mini-batches of 3-channel RGB images of shape (3 x H x W),
# where H and W are expected to be at least 224.
# The images have to be loaded in to a range of [0, 1] and then normalized
# using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
# From: https://pytorch.org/hub/pytorch_vision_fcn_resnet101/
IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]

IMAGE_ROOT: Path = Path("data/skyfinder_images")

class SkyfinderDataset(Dataset[DatasetItem]):
    """
    PyTorch Dataset for loading Skyfinder images and associated meteorological metadata.
    
    Targets are returned as raw floats to maintain compatibility with both 
    continuous regression and categorical quantization pipelines.
    """

    def __init__(
        self, 
        csv_path: Union[Path, str], 
        transform: Optional[transforms.Compose] = None,
        image_root: Path = IMAGE_ROOT
    ) -> None:
        """
        Args:
            csv_path: Path to the split-specific CSV (train/val/test).
            transform: A torchvision.transforms pipeline.
            image_root: The filesystem root where camera subfolders reside.
        """
        self.csv_path: Path = Path(csv_path)
        self.image_root: Path = image_root
        self.transform: Optional[transforms.Compose] = transform
        
        if not self.csv_path.exists():
            raise FileNotFoundError(f"SkyFinder metadata CSV not found at: {self.csv_path}")
            
        self.df: pd.DataFrame = pd.read_csv(self.csv_path)
        
        # Consistent ID formatting to ensure Path resolution matches directory naming.
        self.df['camera_id'] = self.df['camera_id'].astype(int).astype(str)

    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> DatasetItem:
        """
        Returns:
            A tuple containing ((Image, Metadata), Target).
            - Image: Normalized RGB Tensor [3, 224, 224]
            - Metadata: Tensor of environmental features [5]
            - Target: Scalar tensor of the Heat Index
        """
        row: pd.Series[Any] = self.df.iloc[idx]
        img_path: Path = self.image_root / str(row['camera_id']) / str(row['filename'])
        
        # Load and convert to RGB (standard for ImageNet-pretrained models)
        with Image.open(img_path) as img:
            rgb_img: Image.Image = img.convert("RGB")
            if self.transform:
                image_tensor: torch.Tensor = self.transform(rgb_img)
            else:
                image_tensor = transforms.ToTensor()(rgb_img)

        # Vectorized metadata for the tabular branch of the model
        metadata_tensor: torch.Tensor = torch.tensor([
            float(row['day_of_year']),
            float(row['hour']),
            float(row['latitude']),
            float(row['longitude']),
            float(row['solar_elevation'])
        ], dtype=torch.float32)

        target_tensor: torch.Tensor = torch.tensor(float(row['heat_index']), dtype=torch.float32)
        
        return (image_tensor, metadata_tensor), target_tensor

def get_transforms(split: Literal['train', 'val', 'test']) -> transforms.Compose:
    """
    Constructs the image preprocessing and augmentation pipeline.
    
    For 'train': Includes stochastic augmentations to prevent overfitting to 
    specific camera perspectives.
    
    For 'val'/'test': Deterministic resizing and cropping for consistent evaluation.
    """
    if split == 'train':
        # Citation: Chu et al. (2017) recommend color jitter to simulate 
        # atmospheric variance (haze, glare).
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        # Standard 256->224 resizing strategy for ImageNet-trained backbones.
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])