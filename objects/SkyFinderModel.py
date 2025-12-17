"""
SkyFinderModel.py

Defines the regression model for the Skyfinder dataset.
Uses EfficientNetV2-Small as the backbone with a modified regression head.
"""

from typing import cast

import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights  # type: ignore

class SkyFinderModel(nn.Module):
    """
    EfficientNetV2-Small based model for regressing Heat Index from sky images.
    
    This model loads a pre-trained EfficientNetV2-S (trained on ImageNet) and 
    replaces the final classification layer (1000 classes) with a regression 
    layer (1 scalar output).
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
        # We use the DEFAULT weights (currently IMAGENET1K_V1)
        weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        
        # 2. Instantiate Backbone
        # We load the full model structure
        self.backbone = efficientnet_v2_s(weights=weights)
        
        # 3. Replace the Classifier Head
        # The default classifier in EfficientNetV2 is a Sequential block:
        # (classifier): Sequential(
        #    (0): Dropout(p=0.2, inplace=True)
        #    (1): Linear(in_features=1280, out_features=1000, bias=True)
        # )
        # We preserve the Dropout (0) for regularization but replace the Linear (1).
        
        # Retrieve the existing classifier block (Sequential)
        classifier = self.backbone.classifier
        
        # Get the input features of the existing linear layer (index 1).
        # We use 'cast' to satisfy mypy, ensuring it knows this is a Linear layer.
        original_layer = cast(nn.Linear, classifier[1])
        in_features = original_layer.in_features
        
        # Replace it with a new Linear layer (Output = 1 for Regression)
        # We modify the module in-place.
        classifier[1] = nn.Linear(in_features=in_features, out_features=1)
        
        # 4. Initialize the new layer
        # Xavier/Glorot initialization is standard for linear regression heads 
        # to prevent vanishing/exploding gradients at the start of training.
        new_layer = cast(nn.Linear, classifier[1])
        nn.init.xavier_uniform_(new_layer.weight)
        if new_layer.bias is not None:
            nn.init.zeros_(new_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the network.

        Args:
            x (torch.Tensor): Input batch of images.
                              Shape: [Batch_Size, 3, Height, Width]

        Returns:
            torch.Tensor: Predicted Heat Index values.
                          Shape: [Batch_Size, 1]
        """
        # EfficientNetV2 returns raw logits. 
        # Since we replaced the head, these are now continuous regression values.
        out = self.backbone(x)
        return out