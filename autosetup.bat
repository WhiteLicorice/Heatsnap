@echo off
setlocal

REM ===============================================================================
REM HEATSNAP ENVIRONMENT SETUP SCRIPT
REM Target: Windows 10/11 | GPU: RTX 3060 | Python: 3.11
REM 
REM CRITICAL ARCHITECTURE DECISION:
REM We are using Keras 3 with the PyTorch backend.
REM 
REM SOURCE 1 (TensorFlow on Windows): "TensorFlow 2.10 was the last TensorFlow release 
REM to support GPU on native Windows. Starting with TensorFlow 2.11, we will need 
REM to install TensorFlow in WSL2." 
REM -> https://www.tensorflow.org/install/pip#windows-native
REM 
REM SOURCE 2 (Keras 3 Backends): "Keras 3 is a multi-backend deep learning framework...
REM We can run Keras workflows on top of JAX, TensorFlow, or PyTorch."
REM -> https://keras.io/getting_started/
REM 
REM CONCLUSION: Since we are on native Windows (not WSL2), using the PyTorch backend 
REM is the only way to get native GPU acceleration for Keras 3 without complex WSL setup.
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

REM --- 3. PyTorch with CUDA 12.1 ---
REM Source: https://pytorch.org/get-started/locally/
REM We select the index-url for CUDA 12.1 to match RTX 3060 drivers.
echo [INFO] Installing PyTorch (GPU Enabled for Windows)...
python -m pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

REM --- 4. Keras 3 ---
REM We install tensorflow-cpu because Keras might check for it, but we won't use it for training.
echo [INFO] Installing Keras 3 and Keras-CV...
python -m pip install tensorflow-cpu==2.16.1
python -m pip install keras==3.0.5
python -m pip install keras-cv==0.8.2

REM --- 5. Scientific & Methodology Dependencies ---
REM pysolar/pytz: Required for Solar Elevation calculation (Reda & Andreas, 2004).
REM scikit-learn: Required for "Leave-One-Site-Out" Split.
echo [INFO] Installing Scientific Libraries (Pandas, Pysolar, Scikit-Learn)...
python -m pip install pandas matplotlib scikit-learn pysolar pytz tqdm

REM --- 6. Configuration ---
REM Setting the environment variable to force Keras to use PyTorch.
REM Source: https://keras.io/getting_started/intro_to_keras_3/#configuring-your-backend
setx KERAS_BACKEND torch
echo [CONFIG] 'KERAS_BACKEND' set to 'torch'.

echo.
echo [VERIFICATION]
echo 1. Keras will use PyTorch backend (GPU enabled).
echo 2. Pysolar is installed for sun-angle filtering.
echo.
echo.
python validate_autosetup.py
echo Setup completed.
pause