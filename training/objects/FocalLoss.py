from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss implementation for imbalanced safety-critical classification.
    
    This loss function addresses extreme class imbalance by down-weighting 
    easy-to-classify 'Safe' samples and focusing the model's gradient on rare 
    'Danger' or 'Extreme Caution' samples. 
    
    Implementation Logic:
    --------------------
    This is a Softmax-based generalization of the Focal Loss. While the original 
    Lin et al. paper focused on binary detection (Sigmoid), this implementation 
    recovers p_t (the probability of the ground truth class) from the negative 
    log-likelihood provided by PyTorch's `cross_entropy`. This method is 
    numerically stable and standard for multi-class classification.
    
    Formula:
    --------
    FL(p_t) = -(1 - p_t)^gamma * log(p_t)
    
    Literature Citations:
    ---------------------
    - Manuscript: Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). 
      "Focal Loss for Dense Object Detection." ICCV.
      Link: https://arxiv.org/abs/1708.02002
      PDF: https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf
    
    Community Generalizations (Standard Implementations):
    ----------------------------------------------------
    This specific Softmax approach is consistent with the following 
    production-grade AI libraries:
    - Kornia: https://kornia.readthedocs.io/en/latest/losses.html#kornia.losses.focal_loss
    - MONAI (Medical AI): https://docs.monai.io/en/stable/losses.html#focalloss
    - Segmentation Models PyTorch: https://github.com/qubvel/segmentation_models.pytorch
    """
    weight: Optional[Tensor]
    gamma: float

    def __init__(self, weight: Optional[Tensor] = None, gamma: float = 2.0) -> None:
        super().__init__()
        self.weight = weight 
        self.gamma = gamma

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            inputs: Logits from the model (Shape: [N, C])
            targets: Ground truth labels (Shape: [N])
            
        Returns:
            The scalar mean focal loss value.
        """
        # 1. Calculate Cross Entropy without reduction to get per-sample log(pt)
        # weight is passed to F.cross_entropy to provide class-wise scaling (alpha)
        ce_loss: Tensor = F.cross_entropy(
            inputs, 
            targets, 
            reduction='none', 
            weight=self.weight
        )
        
        # 2. Recover pt (probability of the ground truth class)
        # pt = exp(-ce_loss) since ce_loss is -log(pt)
        pt: Tensor = torch.exp(-ce_loss)
        
        # 3. Apply the focal modulating factor (1 - pt)^gamma
        focal_loss: Tensor = ((1 - pt) ** self.gamma) * ce_loss
        
        return focal_loss.mean()