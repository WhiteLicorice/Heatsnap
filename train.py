"""
train.py

Main training pipeline for the Skyfinder Heat Index regression model.
Orchestrates data ingestion, model compilation, and the training lifecycle.

Methodology References:
    - Loss Function: Huber, P. J. (1964). "Robust Estimation of a Location Parameter." 
      Annals of Mathematical Statistics. (Robustness against sensor outliers).
    - Optimization: Loshchilov, I., & Hutter, F. (2017). "Decoupled Weight Decay 
      Regularization" (AdamW).
    - Regularization: Anderson et al. (2013). Importance of tail-end accuracy in 
      Heat Index exposure metrics (implemented via Weighted Huber Loss).
    - Metrics: Willmott, C. J., & Matsuura, K. (2005). "Advantages of the mean 
      absolute error (MAE) over the root mean square error (RMSE) in assessing 
      average model performance." (Rationale for using both).
"""

from __future__ import annotations
import os

# --- CRITICAL: Set backend BEFORE importing Keras ---
os.environ["KERAS_BACKEND"] = "torch"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 
# Uncomment below to silence lower-level C++ logging if needed
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

from pathlib import Path
from typing import Any

import keras # type: ignore
keras.mixed_precision.set_global_policy("mixed_float16")

# Custom Modules
from objects.SkyFinderModel import SkyFinderModel
from load_skyfinder_dataset import load_skyfinder_dataset

# --- Configuration & Hyperparameters ---
BATCH_SIZE: int = 32
LEARNING_RATE: float = 1e-4
WEIGHT_DECAY: float = 1e-3
NUM_EPOCHS: int = 50
CHECKPOINT_DIR: Path = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

class PredictionLogger(keras.callbacks.Callback):
    def __init__(self, val_ds: Any) -> None:
        super().__init__()
        # We take the sample batch ONCE during initialization.
        # logger_ds is uncached, so this take(1) will not trigger warnings.
        self.sample_batch = next(iter(val_ds.take(1)))

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        inputs, targets = self.sample_batch
        preds = self.model.predict(inputs, verbose=0)
        
        print(f"\n--- Epoch {epoch + 1} Sample Predictions ---")
        for i in range(3): 
            p_val, t_val = float(preds[i][0]), float(targets[i])
            print(f"Sample {i}: Pred: {p_val:.2f}°F | True: {t_val:.2f}°F | Error: {p_val-t_val:+.2f}°F")
        print("-" * 40)
        
class WeightedHuberLoss(keras.losses.Loss):
    """
    Physiological-weighted Huber Loss.
    
    Weights errors more heavily when the true Heat Index is in the 'Caution' 
    or 'Danger' zones (> 90°F) to ensure the model is reliable for public health.
    """
    def __init__(self, delta: float = 1.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.delta = delta
        self.huber = keras.losses.Huber(delta=delta)

    def call(self, y_true: Any, y_pred: Any) -> Any:
        base_loss = self.huber(y_true, y_pred)
        
        # Magic Numbers: 90.0, 10.0 (Ref: Anderson et al., 2013).
        # Increases weight for high-heat events to prioritize safety.
        weights = 1.0 + keras.ops.maximum(0.0, (y_true - 90.0) / 10.0)
        return keras.ops.mean(base_loss * weights)

@keras.saving.register_keras_serializable()
def danger_zone_mae(y_true: Any, y_pred: Any) -> Any:
    """
    Safety Metric: Calculates MAE specifically for high-risk heat scenarios (> 90°F).
    """
    mask = keras.ops.greater(y_true, 90.0)
    abs_error = keras.ops.abs(y_true - y_pred)
    masked_error = keras.ops.where(mask, abs_error, 0.0)
    
    count = keras.ops.sum(keras.ops.cast(mask, "float32"))
    return keras.ops.sum(masked_error) / (count + 1e-7)

def train_model() -> None:
    """
    Initializes the multimodal stack and executes the training loop.
    """
    print("main: Initializing SkyFinder training pipeline...")

    # 1. Data Preparation
    # train_ds uses cache=True for performance. 
    # val_ds and logger_ds use cache=False to prevent "peeking" warnings.
    train_ds = load_skyfinder_dataset("data/splits/train.csv", batch_size=BATCH_SIZE)
    val_ds = load_skyfinder_dataset("data/splits/val.csv", batch_size=BATCH_SIZE, shuffle=False, use_cache=False)
    logger_ds = load_skyfinder_dataset("data/splits/val.csv", batch_size=3, shuffle=True, use_cache=False)
    
    # --- OPTION 2: Pre-warm Training Cache ---
    # We iterate through the dataset once to satisfy the .cache() call.
    # This prevents the Torch backend's shape-inference "peek" from discarding the cache.
    print("main: Pre-warming training cache to achieve max speed...")
    for _ in train_ds:
        pass
    print("main: Cache populated. Epoch 1 will start at full speed.")

    # 2. Model Initialization
    raw_model = SkyFinderModel(augment=True)
    model = raw_model.build_graph()
    
    # 3. Compilation
    # Includes RMSE to penalize large outliers and Danger MAE for safety auditing.
    optimizer = keras.optimizers.AdamW(
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        clipnorm=1.0
    )

    model.compile(
        optimizer=optimizer,
        loss=WeightedHuberLoss(delta=1.0),
        metrics=["mae", keras.metrics.RootMeanSquaredError(name="rmse"), danger_zone_mae]
    )

    # 4. Callbacks Setup
    training_callbacks: list[keras.callbacks.Callback] = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(CHECKPOINT_DIR / "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        PredictionLogger(logger_ds),
        keras.callbacks.TensorBoard(log_dir="results")
    ]

    # 5. Training Execution
    print(f"main: Starting training on {keras.backend.backend()} backend...") 
    
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=NUM_EPOCHS,
        callbacks=training_callbacks,
        verbose=1
    )

    print("-" * 30)
    print(f"Training Complete. Best model saved to: {CHECKPOINT_DIR}")

if __name__ == "__main__":
    train_model()