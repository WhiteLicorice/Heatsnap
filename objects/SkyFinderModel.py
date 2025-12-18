"""
SkyFinderModel.py

Architecture: 
    - Visual: EfficientNetV2-S (Tan & Le, 2021).
    - Physics: MLP-based fusion head.
"""

from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn as nn
import math
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

class SkyFinderModel(nn.Module):
    """
    EfficientNetV2-Small based model for regressing Heat Index from sky images AND metadata.
    
    This model:
      1. Processes the image via EfficientNetV2 (Visual Branch).
      2. Processes Time/Location via a small MLP (Physics Branch).
      3. Concatenates both feature vectors.
      4. Regresses the final Heat Index.
    """

    @override
    def __init__(self, pretrained: bool = True) -> None:
        """
        Initialize the model architecture.

        Args:
            pretrained (bool): If True, loads weights pre-trained on ImageNet-1K. 
                               If False, initializes random weights. Default: True.
        """
        super().__init__()
        # 1. Visual Branch
        weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_v2_s(weights=weights)
        self.backbone.classifier = nn.Identity() 
        
        # 2. Physics Branch (Enhanced with LayerNorm)
        self.meta_mlp = nn.Sequential(
            nn.Linear(7, 32),
            nn.LayerNorm(32), # Stabilizes fusion with visual features
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        
        # 3. Fusion Head
        self.head = nn.Sequential(
            nn.Linear(1280 + 64, 512),
            nn.SiLU(), 
            nn.Dropout(p=0.3),
            nn.Linear(512, 1) 
        )

    @override
    def forward(self, image: torch.Tensor, raw_meta: torch.Tensor) -> torch.Tensor:
        """
        raw_meta: [Batch, 5] -> [day_of_year, hour, lat, lon, elevation]
        """
        # Ensure input is 2D [Batch, Features]
        if raw_meta.dim() == 1:
            raw_meta = raw_meta.unsqueeze(0)

        # Internal Transformation
        day, hour = raw_meta[:, 0], raw_meta[:, 1]
        lat, lon, elev = raw_meta[:, 2]/90.0, raw_meta[:, 3]/180.0, raw_meta[:, 4]/90.0
        
        d_sin, d_cos = torch.sin(2*math.pi*day/366.0), torch.cos(2*math.pi*day/366.0)
        h_sin, h_cos = torch.sin(2*math.pi*hour/24.0), torch.cos(2*math.pi*hour/24.0)
        
        processed_meta = torch.stack([d_sin, d_cos, h_sin, h_cos, lat, lon, elev], dim=1)
        
        img_feats = self.backbone(image)
        meta_feats = self.meta_mlp(processed_meta)
        
        return self.head(torch.cat((img_feats, meta_feats), dim=1))