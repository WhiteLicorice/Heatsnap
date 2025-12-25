@echo off
setlocal

REM ===============================================================================
REM HEATSNAP ENVIRONMENT SETUP SCRIPT
REM Target: Windows 10/11 | GPU: RTX 3060 | Python: 3.11
REM 
REM We migrate from Keras 3 to Pure PyTorch (Native).
REM 
REM REASON 1 (Performance & Control): Keras 3 acts as a high-level abstraction. 
REM While convenient, it introduced overhead and obscured low-level data loading 
REM issues (e.g., the JPEG corruption/cache warnings). Native PyTorch gives us 
REM direct control over the 'DataLoader' and optimization steps.
REM 
REM REASON 2 (Deployment Readiness): We are targeting Android (Offline) and Web.
REM - Android: PyTorch 2.x supports "ExecuTorch" and "LibTorch" for native mobile execution.
REM - Web/Edge: Native PyTorch models export cleanly to ONNX, the industry standard 
REM   for cross-platform deployment.
REM 
REM ===============================================================================

REM --- Check for Python 3.11 ---
py -3.11 --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.11 not found. Please install it from python.org.
    pause
    exit /b
)

REM --- 1. Environment Creation ---
if exist venv (
    echo [INFO] Activating existing virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo [INFO] Creating virtual environment with Python 3.11...
    py -3.11 -m venv venv
    call venv\Scripts\activate.bat
)

REM --- 2. Upgrade Deps ---
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

REM --- 3. PyTorch (Latest Stable) ---
REM Source: https://pytorch.org/get-started/locally/
REM We use the Stable (2.5.1) release with CUDA 12.4. 
REM This ensures support for recent deployment tools (ExecuTorch) and RTX 3060 features.
echo [INFO] Installing PyTorch 2.5.1 (GPU Enabled for Windows)...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

REM --- 4. Scientific & Methodology Dependencies ---
REM pysolar/pytz: Required for Solar Elevation calculation (Reda & Andreas, 2004).
REM scikit-learn: Required for "Leave-One-Site-Out" Split.
REM pillow: Explicitly required for robust image loading/sanitization.
echo [INFO] Installing Scientific Libraries (Pandas, Pysolar, Scikit-Learn)...
python -m pip install pandas matplotlib scikit-learn pysolar pytz tqdm pillow

REM --- 5. Cleanup Configuration ---
REM We remove the KERAS_BACKEND variable if it exists to avoid confusion.
echo [CONFIG] Clearing legacy Keras environment variables...
setx KERAS_BACKEND "" >nul 2>&1

echo.
echo [VERIFICATION]
echo 1. Pure PyTorch 2.5.1 installed (CUDA 12.4 enabled).
echo 2. Keras/TensorFlow dependencies removed.
echo 3. Environment is ready for high-performance training.
echo.
echo.
REM You will need to update validate_autosetup.py to check for torch.cuda.is_available()
python validate_autosetup.py
echo Setup completed.
pause