"""
SkyFinderModel.py

Defines the regression model for the Skyfinder dataset.
Uses EfficientNetV2-Small as the backbone with a modified regression head.
Implements a fusion architecture to combine Visual features with Time metadata.
"""

from typing import cast

import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights  # type: ignore

class SkyFinderModel(nn.Module):
    """
    EfficientNetV2-Small based model for regressing Heat Index from sky images AND metadata.
    
    This model:
      1. Processes the image via EfficientNetV2 (Visual Branch).
      2. Processes Month/Hour via a small MLP (Time Branch).
      3. Concatenates both feature vectors.
      4. Regresses the final Heat Index.
    """

    def __init__(self, pretrained: bool = True) -> None:
        """
        Initialize the model architecture.

        Args:
            pretrained (bool): If True, loads weights pre-trained on ImageNet-1K. 
                               If False, initializes random weights. Default: True.
        """
        super().__init__()
        
        # 1. Load Weights (or None)
        weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        
        # 2. Instantiate Backbone (Visual Branch)
        self.backbone = efficientnet_v2_s(weights=weights)
        
        # We remove the original classifier entirely.
        # EfficientNetV2-S outputs a feature vector of size 1280.
        self.backbone.classifier = nn.Identity()
        
        # 3. Instantiate Metadata Encoder (Time Branch)
        # Input: 2 features (Month Normalized, Hour Normalized)
        # Output: 64 features
        self.meta_mlp = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        
        # 4. The Fusion Head
        # Input: 1280 (Visual) + 64 (Time) = 1344 features
        self.head = nn.Sequential(
            nn.Linear(1280 + 64, 512),
            nn.SiLU(), # Swish activation (modern standard)
            nn.Dropout(p=0.3),
            nn.Linear(512, 128),
            nn.SiLU(),
            nn.Linear(128, 1) # Final prediction
        )
        
        # Initialize the head weights
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, image: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the network.

        Args:
            image (torch.Tensor): Input batch of images.
                                  Shape: [Batch_Size, 3, Height, Width]
            metadata (torch.Tensor): Input batch of time data.
                                     Shape: [Batch_Size, 2] -> (Month, Hour)

        Returns:
            torch.Tensor: Predicted Heat Index values.
                          Shape: [Batch_Size, 1]
        """
        # 1. Get Visual Features [Batch, 1280]
        img_feats = self.backbone(image)
        
        # 2. Get Metadata Features [Batch, 64]
        meta_feats = self.meta_mlp(metadata)
        
        # 3. Fuse (Concatenate) [Batch, 1344]
        combined = torch.cat((img_feats, meta_feats), dim=1)
        
        # 4. Predict
        out = self.head(combined)
        return out