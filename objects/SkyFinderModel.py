"""
SkyFinderModel.py

Multimodal Fusion Network: EfficientNetV2-S + In-Graph Physics Encoding.

Methodology References:
    - Backbone: Tan, M., & Le, Q. V. (2021). "EfficientNetV2: Smaller Models and 
      Faster Training." https://arxiv.org/abs/2104.00298
    - Solar Position: Reda, I. & Andreas, A. (2004). "Solar Position Algorithm 
      for Solar Radiation Applications." NREL Report No. TP-560-34302.
    - Cyclical Encoding: Geman, S., et al. (1992). "Neural Networks and the 
      Bias/Variance Dilemma." (Foundational logic for periodic feature mapping).
"""

from __future__ import annotations
from typing import Any
import keras # type: ignore
import keras_cv # type: ignore

class PhysicsEncoding(keras.layers.Layer):
    """
    Transforms raw spacetime metadata into a feature vector optimized for NNs.
    
    This layer solves the 'continuity problem' in linear time representations 
    by mapping temporal inputs onto a unit circle.
    """
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Fundamental constant for circular mapping
        self.PI: float = 3.141592653589793

    def call(self, inputs: Any) -> Any:
        """
        Args:
            inputs: Tensor [batch, 5] -> [doy, hour, lat, lon, solar]
        """
        doy: Any   = inputs[:, 0]
        hour: Any  = inputs[:, 1]
        lat: Any   = inputs[:, 2]
        lon: Any   = inputs[:, 3]
        solar: Any = inputs[:, 4]

        # 1. Cyclical Time Encoding
        # Magic Number: 365.25 is the mean length of the tropical year (Reda & Andreas, 2004).
        # This ensures leap-year compatibility in long-term datasets.
        sin_doy = keras.ops.sin(2.0 * self.PI * (doy - 1.0) / 365.25)
        cos_doy = keras.ops.cos(2.0 * self.PI * (doy - 1.0) / 365.25)
        
        # Magic Number: 24.0 represents the full diurnal rotation.
        # This prevents a 'cliff' between 23:59 and 00:00.
        sin_hr  = keras.ops.sin(2.0 * self.PI * hour / 24.0)
        cos_hr  = keras.ops.cos(2.0 * self.PI * hour / 24.0)

        # 2. Normalized Spatial/Solar Features
        # Magic Numbers: 90.0 (Lat), 180.0 (Lon). Standard Earth coordinate bounds.
        # Magic Number: 90.0 (Solar). Max solar elevation at zenith.
        # Normalizing to [-1, 1] range prevents metadata from dominating visual gradients.
        n_lat   = lat / 90.0
        n_lon   = lon / 180.0
        n_solar = solar / 90.0

        return keras.ops.stack([
            sin_doy, cos_doy, sin_hr, cos_hr, n_lat, n_lon, n_solar
        ], axis=1)



class SkyFinderModel(keras.Model):
    """
    Late-Fusion Multimodal Regressor.
    
    Architecture:
        - Visual: EfficientNetV2-S (Pretrained on ImageNet).
        - Physics: 2-layer MLP processing cyclical spacetime features.
        - Fusion: Concatenation followed by a SiLU-activated regressor head.
    """
    def __init__(self, augment: bool = True, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.augment: bool = augment
        
        # Feature Extractor: Faster training and smaller footprint than EfficientNetV1.
        self.backbone = keras.applications.EfficientNetV2S(
            weights="imagenet", 
            include_top=False, 
            pooling="avg", 
            include_preprocessing=True
        )
        
        self.physics_branch = keras.Sequential([
            PhysicsEncoding(),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(64, activation="relu")
        ], name="physics_branch")

        # Regressor Head: Uses SiLU (Swish) as it lacks the 'dying ReLU' problem 
        # in regression tasks with wide output ranges (40F to 120F).
        self.regressor = keras.Sequential([
            keras.layers.Dense(256, activation="silu"),
            keras.layers.Dropout(0.3),
            # dtype="float32" ensures the final prediction 
            # and loss calculation are numerically stable.
            keras.layers.Dense(1, activation="linear", dtype="float32") 
        ], name="regressor_head")

        self.aug = keras.Sequential([
            keras_cv.layers.RandomFlip("horizontal"),
            keras_cv.layers.RandomColorJitter(
                value_range=(0, 255), 
                brightness_factor=0.15, # Simulates exposure variance
                contrast_factor=0.1,   # Simulates atmospheric haze/clarity
                saturation_factor=0.1, # Accounts for sensor vibrance
                hue_factor=0.02        # VERY low: only accounts for white-balance tint
            )
        ]) if augment else keras.layers.Identity()

    def call(self, inputs: dict[str, Any], training: bool | None = False) -> Any:
        img_input: Any = inputs["image"]
        meta_input: Any = inputs["metadata"]

        if training:
            img_input = self.aug(img_input)

        img_feats: Any = self.backbone(img_input)
        phy_feats: Any = self.physics_branch(meta_input)
        
        combined: Any = keras.ops.concatenate([img_feats, phy_feats], axis=-1)
        return self.regressor(combined)

    def build_graph(self) -> keras.Model:
        """Functional builder for serialization and TFLite compatibility."""
        inputs = {
            "image": keras.Input(shape=(224, 224, 3), name="image"),
            "metadata": keras.Input(shape=(5,), name="metadata")
        }
        return keras.Model(inputs=inputs, outputs=self.call(inputs))