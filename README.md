# Heatsnap

This is a repository for Heatsnap, a project that aims to determine the heat index of an outdoor environment out of a photo of its surroundings.

# Environment

We'll be using Keras with PyTorch in the backend. Run `pip install -r requirements` in the project directory. To install manually, follow the steps below.

1. Create a virtual environment, assuming that Python 3.11 is on `PATH`: `py -3.11 -m venv venv` (**Note**: Running `python.exe -m pip install --upgrade pip` may be needed).
2. Install `torch~=2.1.0`: `pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121`.
3. Install `keras~=3.0.0`: `pip install keras==3.0.5` (**Note**: run `pip install packaging` if `packaging` module fails to install).
4. Install `keras-cv for keras~=3.0.0 and torch~=2.1.0`: `pip install --upgrade keras-cv~=0.9.0`.
5. Install `tensorflow~=2.16.1` for `keras-cv`: `pip install tensorflow~=2.16.1`.
6. Install `pandas` separately: `pip install pandas`.
7. Check if the environment has been installed properly by running `python test.py`.

These steps have been automated in `autosetup.bat`, in the event that `pip install -r requirements.txt` does not function correctly.