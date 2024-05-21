"""Module for training the models"""

import os

# Neural Network Libraries
os.environ["KERAS_BACKEND"] = "torch"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import keras
import keras_cv
import tensorflow as tf

# Data Science + Python Libraries
import pandas as pd
import matplotlib.pyplot as plt
import time

#   TODO: Define path to dataset and use this global across the script
DATASET = 'data\sky_pictures_dataset_time_ascending_1_year.csv'

# DATA MODEL

def get_model():
    inputs = keras.layers.Input(shape=(128, 128, 3), batch_size=32)
    
    con2dfirst = keras.layers.Conv2D(32, (3, 3), activation="relu")(inputs)
    con2dtd1 = keras.layers.TimeDistributed(con2dfirst)
    mp1 = keras.layers.MaxPooling2D((2, 2))(con2dtd1)
    mptd1 = keras.layers.TimeDistributed(mp1)
    
    con2dsecond = keras.layers.Conv2D(32, (3, 3), activation="relu")(mptd1)
    con2dtd2 = keras.layers.TimeDistributed(con2dsecond)
    mp2 = keras.layers.MaxPooling2D((2, 2))(con2dtd2)
    mptd2 = keras.layers.TimeDistributed(mp2)
    
    f1 = keras.layers.Flatten()(mptd2)
    ftd1 = keras.layers.TimeDistributed(f1)
            
    d1 = keras.layers.Dense(256, activation="relu")(ftd1)
    dtd1 = keras.layers.TimeDistributed(d1)
    
    
    # Concatenation with other information in this line
    
    
    d2 = keras.layers.Dense(128, activation="relu")(dtd1)
    dtd2 = keras.layers.TimeDistributed(d2)
    
    d3 = keras.layers.Dense(64, activation="relu")(dtd2)
    dtd3 = keras.layers.TimeDistributed(d3)
    
    lstm = keras.layers.LSTM(32)(dtd3)
    
    d4 = keras.layers.Dense(1, name="predictions")(lstm)
    model = keras.Model(inputs=inputs, outputs=d4)
    
    return model

# DATA PROCESSING FUNCTIONS

def train_save_model(train_data: pd.DataFrame, val_data: pd.DataFrame) -> None:
    """
    Trains model using train and validation data provided, then saves the model after.

    Parameters:
        train_data (images): The uploaded set of training data.
        val_data (images): The uploaded set of validation data.
    Returns:
        None. It just trains data.
    """
    
    # TODO
    # 1. Data Pipeline
    # 2. Train Model
    # 3. Save model
    return

def test(test_data: pd.DataFrame) -> None:
    """
    Tests model using test data provided.

    Parameters:
        test_data (image): The uploaded set of test data.
    Returns:
        None. It just tests data.
    """
    
    # TODO
    # 3. Check accuracy
    return

def train_val_test_split(data: pd.DataFrame, train_ratio = 0.7, val_ratio = 0.15, test_ratio = 0.15) -> pd.DataFrame:
    """
    Returns train, validation, and test data.

    Parameters:
        data (image): The uploaded set of data.
        train_ratio: The ratio for train data.
        val_ratio: The ratio for validation data.
        test_ratio: The ratio for test data.
    Returns:
        train_data: The train data for training.
        val_data: The validation data to help with training.
        test_data: The test data to check model performance. 
    """
    n = len(data)
    
    train_index = int(n * train_ratio)
    val_index = int(n * val_ratio)
    
    # Slicing
    train_data = data[:train_index]
    val_data = data[train_index:train_index+val_index]
    test_data = data[train_index+val_index:]
    
    return train_data, val_data, test_data
    
# MAIN
def main():
    data = pd.read_csv(DATASET)
    data = data.drop(columns=['TempI', 'Min'])     #   Drop degrees Fahrenheit since we'll be using degrees Celsius, drop Minutes since lots of bad entries
    data = data.loc[data['TempM'] != -9999]        #   Filter against -9999 degrees Celsius entries
    
    #   TODO: train model after reading dataset
    
    pass

if __name__ == "__main__":
    main()