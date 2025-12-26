"""
SkyFinderModel.py

Architecture: 
    - Visual: EfficientNetV2-S (Tan & Le, 2021).
    - Physics: MLP-based metadata branch.
    - Fusion: Gated Multimodal Unit (Arevalo et al., 2017).

LITERATURE CITATIONS:
- Backbone: Tan, M., & Le, Q. (2021). "EfficientNetV2: Smaller Models and Faster Training." 
  ICML. https://arxiv.org/abs/2104.00298
- Fusion Architecture: Arevalo, J., et al. (2017). "Gated Multimodal Units for Information Fusion." 
  ICLR Workshop. https://arxiv.org/abs/1702.01992
"""

from __future__ import annotations
import math
from typing import Final

import torch # type: ignore
import torch.nn as nn
from typing_extensions import override
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights # type: ignore

# Constants for mpy adherence and architectural clarity
VISUAL_DIM: Final[int] = 1280
META_INPUT_DIM: Final[int] = 7  # d_sin, d_cos, h_sin, h_cos, lat, lon, elev

class SkyFinderModel(nn.Module):
    """
    Multimodal Categorical Risk Assessment model.
    Uses a gated fusion mechanism to integrate visual cues with physical metadata.
    References:
        1. Tan, M., & Le, Q. (2021). EfficientNetV2: Smaller Models and Faster Training. 
           International Conference on Machine Learning (ICML).
           Manuscript: https://arxiv.org/abs/2104.00298
        
        2. Arevalo, J., Solorio, T., Montes-y-Gómez, M., & Hernández, A. M. (2017). 
           Gated Multimodal Units for Information Fusion. ICLR Workshop.
           Manuscript: https://arxiv.org/abs/1702.01992
    """

    @override
    def __init__(self, pretrained: bool = True, num_outputs: int = 5) -> None:
        """
        Initialize the model architecture.

        Args:
            pretrained (bool): If True, loads weights pre-trained on ImageNet-1K. 
                               If False, initializes random weights. Default: True.
            num_outputs (int): Number of categories in the output. Default: 5.
        """

        super().__init__()
        
        # 1. Visual Branch (EfficientNetV2-S)
        # EfficientNetV2-S is optimized for faster training and parameter efficiency.
        # Citation: Tan & Le (2021). https://arxiv.org/abs/2104.00298
        
        weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_v2_s(weights=weights)
        self.backbone.classifier = nn.Identity() 

        # 2. Physics Branch (MLP)
        # Maps 7 engineered features to the same latent dimension as the visual backbone.
        self.meta_mlp = nn.Sequential(
            nn.Linear(META_INPUT_DIM, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, VISUAL_DIM),
            nn.SiLU()
        )
        
        # 3. Gated Multimodal Fusion (GMU)
        # Learns a gate 'z' to weigh the importance of pixels vs. physics.
        # Citation: Arevalo et al. (2017). https://arxiv.org/abs/1702.01992
        
        self.gate = nn.Sequential(
            nn.Linear(VISUAL_DIM + VISUAL_DIM, 512),
            nn.ReLU(),
            nn.Linear(512, VISUAL_DIM),
            nn.Sigmoid()
        )
        
        # 4. Categorical Classification Head
        # Output maps to NWS Heat Risk categories.
        self.head = nn.Sequential(
            nn.Linear(VISUAL_DIM, 512),
            nn.SiLU(),
            nn.Dropout(p=0.4),
            nn.Linear(512, num_outputs) 
        )

    @override
    def forward(self, image: torch.Tensor, raw_meta: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with on-the-fly feature engineering.
        
        Args:
            image: Batch of images [B, 3, 224, 224]
            raw_meta: Batch of raw metadata [B, 5] (day, hour, lat, lon, elev)
        """
        if raw_meta.dim() == 1:
            raw_meta = raw_meta.unsqueeze(0)

        # --- Feature Engineering (Cyclical Time + Spatial Normalization) ---
        day, hour = raw_meta[:, 0], raw_meta[:, 1]
        lat: torch.Tensor = raw_meta[:, 2] / 90.0
        lon: torch.Tensor = raw_meta[:, 3] / 180.0
        elev: torch.Tensor = raw_meta[:, 4] / 90.0
        
        # Encoding time as unit circle coordinates to preserve periodicity.
        d_sin: torch.Tensor = torch.sin(2 * math.pi * day / 366.0)
        d_cos: torch.Tensor = torch.cos(2 * math.pi * day / 366.0)
        h_sin: torch.Tensor = torch.sin(2 * math.pi * hour / 24.0)
        h_cos: torch.Tensor = torch.cos(2 * math.pi * hour / 24.0)
        
        processed_meta = torch.stack([d_sin, d_cos, h_sin, h_cos, lat, lon, elev], dim=1)
        
        # Branch Execution
        v_feats: torch.Tensor = self.backbone(image)
        p_feats: torch.Tensor = self.meta_mlp(processed_meta)
        
        # --- Gated Fusion Logic ---
        # Gate 'z' determines the contribution of each modality.
        combined = torch.cat([v_feats, p_feats], dim=1)
        z = self.gate(combined)
        
        # Modality blending: z*Visual + (1-z)*Physics
        fused_feats = (z * v_feats) + ((1.0 - z) * p_feats)
        
        return self.head(fused_feats)