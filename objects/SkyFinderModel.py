"""
SkyFinderModel.py

Architecture: 
    - Visual: EfficientNetV2-S (Tan & Le, 2021).
    - Physics: MLP-based metadata branch (PINN).
    - Fusion: Gated Multimodal Unit (Arevalo et al., 2017).
"""

from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn as nn
import math
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

class SkyFinderModel(nn.Module):
    """
    EfficientNetV2-Small based model for regressing Heat Index from sky images and metadata.
    Uses a Gated Multimodal Unit (GMU) to dynamically weight visual vs. physical inputs.

    References:
        1. Tan, M., & Le, Q. (2021). EfficientNetV2: Smaller Models and Faster Training. 
           International Conference on Machine Learning (ICML).
           Manuscript: https://arxiv.org/abs/2104.00298
        
        2. Arevalo, J., Solorio, T., Montes-y-Gómez, M., & Hernández, A. M. (2017). 
           Gated Multimodal Units for Information Fusion. ICLR Workshop.
           Manuscript: https://arxiv.org/abs/1702.01992
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
        
        # 1. Visual Branch (EfficientNetV2-S)
        weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_v2_s(weights=weights)
        self.backbone.classifier = nn.Identity() 

        # 2. Physics Branch (MLP)
        # Projects 7 cyclical/spatial features to 1280 to match EfficientNet output
        self.meta_mlp = nn.Sequential(
            nn.Linear(7, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 1280),
            nn.SiLU()
        )
        
        # 3. Gated Multimodal Fusion (GMU) logic
        # Learns the 'z' gate which modulates the influence of each modality.
        self.gate = nn.Sequential(
            nn.Linear(1280 + 1280, 512),
            nn.ReLU(),
            nn.Linear(512, 1280),
            nn.Sigmoid()
        )
        
        # 4. Final Regression Head
        self.head = nn.Sequential(
            nn.Linear(1280, 512),
            nn.SiLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 1) 
        )

    @override
    def forward(self, image: torch.Tensor, raw_meta: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: [Batch, 3, 224, 224]
            raw_meta: [Batch, 5] -> [day_of_year, hour, lat, lon, elevation]
        """
        if raw_meta.dim() == 1:
            raw_meta = raw_meta.unsqueeze(0)

        # 1. Cyclical and Spatial Feature Engineering
        day, hour = raw_meta[:, 0], raw_meta[:, 1]
        lat, lon, elev = raw_meta[:, 2]/90.0, raw_meta[:, 3]/180.0, raw_meta[:, 4]/90.0
        
        d_sin, d_cos = torch.sin(2*math.pi*day/366.0), torch.cos(2*math.pi*day/366.0)
        h_sin, h_cos = torch.sin(2*math.pi*hour/24.0), torch.cos(2*math.pi*hour/24.0)
        
        processed_meta = torch.stack([d_sin, d_cos, h_sin, h_cos, lat, lon, elev], dim=1)
        
        # 2. Extract Features
        v_feats = self.backbone(image)         # Visual representation
        p_feats = self.meta_mlp(processed_meta) # Physical/Contextual representation
        
        # 3. Gated Fusion (GMU)
        # We calculate a per-channel gate value between 0 and 1
        combined = torch.cat([v_feats, p_feats], dim=1)
        z = self.gate(combined)
        
        # Modality blending: z*Visual + (1-z)*Physics
        fused_feats = (z * v_feats) + ((1.0 - z) * p_feats)
        
        # 4. Final Output (Heat Index Regression)
        return self.head(fused_feats)