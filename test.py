import os

print("Validating PyTorch installation...")
import torch
print(torch.__version__)
print(f"CUDA Availability: {torch.cuda.is_available()}")

print("Validating Keras installation...")
#   Declare our Keras backend
os.environ["KERAS_BACKEND"] = "torch"
import keras
print(keras.__version__)

print("Validating TensorFlow installation...")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'   #   Avoid floating-point round-off errors from different computation orders
import tensorflow as tf
print(tf.__version__)

print("Validating KerasCV installation...")
import keras_cv
print(keras_cv.__version__)

print("All KerasPyCV dependencies installed!")