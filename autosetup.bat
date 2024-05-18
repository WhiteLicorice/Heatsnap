@echo off

REM Check if venv directory exists
if exist venv (
    echo Activating existing virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo Creating virtual environment...
    py -3.11 -m venv venv

    echo Activating new virtual environment...
    call venv\Scripts\activate.bat
)

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing PyTorch...
python -m pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

echo Installing Keras...
python -m pip install keras==3.0.5

echo Installing Keras-CV...
python -m pip install --upgrade keras-cv~=0.9.0

echo Installing TensorFlow...
python -m pip install tensorflow==2.16.1

echo Installing Pandas...
python -m pip install pandas

echo.
echo Validating installation...
python test.py

echo.
echo Heatsnap environment setup completed.
pause
