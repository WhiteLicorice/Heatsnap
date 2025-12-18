"""
validate_autosetup.py
Validation script for HeatSnap Environment (Pure PyTorch Architecture)
"""
import sys
import importlib

print(f"Python Version: {sys.version}")
print("-" * 50)

# --- 1. Validate PyTorch & GPU (The Engine) ---
print("Validating PyTorch (Compute Engine)...")
try:
    import torch
    print(f"PyTorch Version: {torch.__version__}")
    
    # Check CUDA (NVIDIA GPU Support)
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available:  {cuda_available}")
    
    if cuda_available:
        print(f"CUDA Version:    {torch.version.cuda}")
        device_count = torch.cuda.device_count()
        print(f"GPU Devices:     {device_count}")
        
        for i in range(device_count):
            print(f"  > GPU {i}:      {torch.cuda.get_device_name(i)}")
            
        # Actual computation test (Matrix Multiplication)
        try:
            x = torch.rand(1000, 1000).cuda()
            y = torch.rand(1000, 1000).cuda()
            z = torch.matmul(x, y)
            print("GPU Tensor Test: Success (Matrix multiplication executed on GPU)")
        except Exception as e:
            print(f"GPU Tensor Test: FAILED. {e}")
            
    else:
        print("WARNING: CUDA is NOT available. Training will be extremely slow (CPU only).")
        print("         Did you install the correct CUDA 12.4 version of PyTorch?")
except ImportError:
    print("CRITICAL ERROR: PyTorch ('torch') is not installed.")

print("-" * 50)

# --- 2. Validate Data Processing & Imaging ---
print("Validating Core Libraries...")
required_libs = [
    ("pandas", "Dataframes"),
    ("PIL", "Pillow (Image Processing)"),
    ("tqdm", "Progress Bars"),
    ("matplotlib", "Plotting"),
    ("numpy", "Numerical Operations")
]

for lib_name, description in required_libs:
    try:
        importlib.import_module(lib_name)
        print(f"  [OK] {lib_name:<12} ({description})")
    except ImportError:
        print(f"  [MISSING] {lib_name} is required for {description}")

print("-" * 50)

# --- 3. Validate Science & Methodology ---
print("Validating Methodology Libraries...")
science_libs = [
    ("sklearn", "Scikit-Learn (Splitting/Metrics)"),
    ("pysolar", "Solar Elevation Calculation"),
    ("pytz", "Timezone Handling")
]

for lib_name, description in science_libs:
    try:
        importlib.import_module(lib_name)
        print(f"  [OK] {lib_name:<12} ({description})")
    except ImportError:
        print(f"  [MISSING] {lib_name} is required for {description}")

print("-" * 50)
print("Environment Validation Complete.")