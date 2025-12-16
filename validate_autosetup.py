"""Module for validating environment setup"""
import os
import sys

# CRITICAL: Set backend BEFORE importing Keras or Keras-CV
os.environ["KERAS_BACKEND"] = "torch"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Reduce TF log noise

print(f"Python Version: {sys.version}")
print("-" * 40)

# --- 1. Validate PyTorch & GPU ---
print("Validating PyTorch (GPU Engine)...")
try:
    import torch
    print(f"PyTorch Version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available:  {cuda_available}")
    
    if cuda_available:
        print(f"GPU Device:      {torch.cuda.get_device_name(0)}")
        # Actual computation test
        x = torch.rand(5, 3).cuda()
        print("GPU Tensor Test: Success")
    else:
        print("WARNING: CUDA is NOT available. Training will be slow (CPU only).")
except ImportError:
    print("ERROR: PyTorch not installed.")

print("-" * 40)

# --- 2. Validate Keras ---
print("Validating Keras 3...")
try:
    import keras
    print(f"Keras Version:   {keras.__version__}")
    print(f"Active Backend:  {keras.backend.backend()}")
    
    if keras.backend.backend() != "torch":
        print("WARNING: Keras is NOT using PyTorch. Check your KERAS_BACKEND setting.")
    else:
        print("Success: Keras is using PyTorch backend.")
except ImportError:
    print("ERROR: Keras not installed.")

print("-" * 40)

# --- 3. Validate KerasCV ---
print("Validating KerasCV...")
try:
    import keras_cv
    print(f"KerasCV Version: {keras_cv.__version__}")
except ImportError:
    print("ERROR: KerasCV not installed.")

print("-" * 40)

# --- 4. Validate Solar Physics ---
print("Validating Science Libs...")
try:
    import pysolar
    import pytz
    print("Pysolar & Pytz:  Installed")
except ImportError:
    print("ERROR: Pysolar or Pytz missing. Solar filtering will fail.")

print("-" * 40)
print("Validation Complete.")