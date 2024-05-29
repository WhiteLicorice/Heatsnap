"""Module for training the models"""

import os
import sys

# Neural Network Libraries
os.environ["KERAS_BACKEND"] = "torch"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from keras.preprocessing.image import load_img, img_to_array
import torch
import keras
import keras_cv
import tensorflow as tf

# Data Science + Python Libraries
import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np

# Import Python custom methods
import sample

from sklearn.preprocessing import OneHotEncoder

from keras.models import load_model

sys.stdout.reconfigure(encoding='utf-8')

#   TODO: Define path to dataset and use this global across the script
DATASET = 'data/sky_pictures_dataset.csv'
BATCH_SIZE = 1

# DATA MODEL

def get_model():
    image_shape = (128, 128, 3)
    additional_data_shape = (3,)
    
    inputs = keras.layers.Input(shape=image_shape, batch_size=BATCH_SIZE)
    
    # First Convolution Layer with MaxPool
    con2dfirst = keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    mp1 = keras.layers.MaxPooling2D((2, 2))(con2dfirst)
    dropout1 = keras.layers.Dropout(0.1)(mp1)
    
    # Second Convolution Layer with MaxPool
    con2dsecond = keras.layers.Conv2D(32, (3, 3), activation='relu')(dropout1)
    mp2 = keras.layers.MaxPooling2D((2, 2))(con2dsecond)
    dropout2 = keras.layers.Dropout(0.1)(mp2)
    
    # Flatten and Dense
    f1 = keras.layers.Flatten()(dropout2)
    d1 = keras.layers.Dense(256, activation='relu')(f1)
    
    # Concatenation with other information in this line
    additional_data = keras.layers.Input(shape=additional_data_shape, batch_size=BATCH_SIZE)
    merged_data = keras.layers.Concatenate()([d1, additional_data])
    dropout3 = keras.layers.Dropout(0.5)(merged_data)
    
    # Two layers of Dense
    d2 = keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.L1L2)(dropout3)
    d3 = keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.L1L2)(d2)
    
    reshaped_data = keras.layers.Reshape((1, 64))(d3)

    # LSTM layer
    lstm = keras.layers.LSTM(64, activation='tanh', recurrent_activation='sigmoid', dropout=0.5, recurrent_dropout=0.1, unroll=False, use_bias=True)(reshaped_data)
    
    # Last Dense layer
    d4 = keras.layers.Dense(9, name="predictions", activation='softmax')(lstm)
    model = keras.Model(inputs=[inputs, additional_data], outputs=d4)
    
    return model

# DATA PROCESSING FUNCTIONS

def train_save_model() -> None:
    """
    Trains model using train and validation data provided, then saves the model after.

    Parameters:
        None. The data can be retrieved by sample.py
    Returns:
        None. It just trains data.
    """
    
    # Switched to OneHotEncoder from OrdinalEncoder
    encoder = OneHotEncoder(categories = [[
    'Extreme Cold Danger',
    'Cold Danger',
    'Extreme Cold Caution',
    'Cold Caution',
    'Safe',
    'Heat Caution',
    'Extreme Heat Caution',
    'Heat Danger',
    'Extreme Heat Danger'
    ]], sparse_output=False, handle_unknown='ignore')
    
    # Retrieve data
    X_train, X_val, X_test, y_train, y_val, y_test = sample.sample()
    
    # Check shapes of the datasets
    print("Shapes of datasets:")
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_val:", y_val.shape)
    print("y_test:", y_test.shape)
    
    # Make them as DataFrames
    y_train = pd.DataFrame(y_train, columns=('TempClass', 'Month', 'Hour', 'Timezone'))
    y_val = pd.DataFrame(y_val, columns=('TempClass', 'Month', 'Hour', 'Timezone'))
    y_test = pd.DataFrame(y_test, columns=('TempClass', 'Month', 'Hour', 'Timezone'))
    
    # Extract TempM only and encode
    y_train_target = y_train[['TempClass']]
    y_val_target = y_val[['TempClass']]
    y_test_target = y_test[['TempClass']]
    
    y_train_target = encoder.fit_transform(y_train_target)
    y_val_target = encoder.transform(y_val_target)
    y_test_target = encoder.transform(y_test_target)

    # Drop the 'TempClass' column
    y_train = y_train.drop(columns=['TempClass'])
    y_val = y_val.drop(columns=['TempClass'])
    y_test = y_test.drop(columns=['TempClass'])
    
    # Making sure all the datatypes are not objects
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_val = y_val.astype(np.float32)
    y_test = y_test.astype(np.float32)
    y_train_target = y_train_target.astype(np.float32)
    y_val_target = y_val_target.astype(np.float32)
    y_test_target = y_test_target.astype(np.float32)
    
    # Adam Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    
    # Retrieve model and train
    model = get_model()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit([X_train, y_train], y_train_target, validation_data=([X_val, y_val], y_val_target), epochs=30, batch_size=BATCH_SIZE)
    
    # Check performance
    loss, accuracy = model.evaluate([X_test, y_test], y_test_target)
    print(f'Categorical Crossentropy Loss: {loss}, Test Accuracy: {accuracy}')
    
    # Plot performance
    plot_accuracy_and_loss(history)  
    
    # Save in Keras format
    model.save('saved_model/h5/Heatsnap.h5')

    return

def plot_accuracy_and_loss(history):
    """
    Plots model accuracy and loss performance using data provided.
    Saves the figures in the data folder.

    Parameters:
        history (pandas Dataframe): The result loss and accuracy data.
    Returns:
        None. It just plots model performance.
    """
    
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('CNN LSTM Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train Data', 'Validation Data'], loc='upper left')
    plt.savefig(f"data/Data Exploration/Accuracy_Regularized_LSTM.jpg")
    plt.clf()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('CNN LSTM Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train Data', 'Validation Data'], loc='upper left')
    plt.savefig(f"data/Data Exploration/Loss_Regularized_LSTM.jpg")

def train__() -> None:
    """
    Trains model using train and validation data provided, then saves the model after.

    Parameters:
        None. The data can be retrieved by sample.py
    Returns:
        None. It just trains data.
    """
    
    # Switched to OneHotEncoder from OrdinalEncoder
    encoder = OneHotEncoder(categories = [[
    'Extreme Cold Danger',
    'Cold Danger',
    'Extreme Cold Caution',
    'Cold Caution',
    'Safe',
    'Heat Caution',
    'Extreme Heat Caution',
    'Heat Danger',
    'Extreme Heat Danger'
    ]], sparse_output=False, handle_unknown='ignore')
    
    # Retrieve data
    X_train, X_val, X_test, y_train, y_val, y_test = sample.sample()
    
    # Check shapes of the datasets
    print("Shapes of datasets:")
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_val:", y_val.shape)
    print("y_test:", y_test.shape)
    
    # Make them as DataFrames
    y_train = pd.DataFrame(y_train, columns=('TempClass', 'Month', 'Hour', 'Timezone'))
    y_val = pd.DataFrame(y_val, columns=('TempClass', 'Month', 'Hour', 'Timezone'))
    y_test = pd.DataFrame(y_test, columns=('TempClass', 'Month', 'Hour', 'Timezone'))
    
    # Extract TempM only and encode
    y_train_target = y_train[['TempClass']]
    y_val_target = y_val[['TempClass']]
    y_test_target = y_test[['TempClass']]
    y_train_target = encoder.fit_transform(y_train_target)
    y_val_target = encoder.transform(y_val_target)
    y_test_target = encoder.transform(y_test_target)

    # Drop the 'TempClass' column
    y_train = y_train.drop(columns=['TempClass'])
    y_val = y_val.drop(columns=['TempClass'])
    y_test = y_test.drop(columns=['TempClass'])
    
    # Making sure all the datatypes are not objects
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_val = y_val.astype(np.float32)
    y_test = y_test.astype(np.float32)
    y_train_target = y_train_target.astype(np.float32)
    y_val_target = y_val_target.astype(np.float32)
    y_test_target = y_test_target.astype(np.float32)
    
    # Adam Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    
    # Retrieve model and train
    model = get_model()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit([X_train, y_train], y_train_target, validation_data=([X_val, y_val], y_val_target), epochs=30, batch_size=BATCH_SIZE)
    
    # Check performance
    loss, accuracy = model.evaluate([X_test, y_test], y_test_target)
    print(f'Categorical Crossentropy Loss: {loss}, Test Accuracy: {accuracy}')
    
    # Plot performance
    plot_accuracy_and_loss(history)  
    
    # Save in Keras format
    model.save('saved_model/h5/Heatsnap.h5')

    return

def plot_accuracy_and_loss(history):
    """
    Plots model accuracy and loss performance using data provided.
    Saves the figures in the data folder.

    Parameters:
        history (pandas Dataframe): The result loss and accuracy data.
    Returns:
        None. It just plots model performance.
    """
    
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('CNN LSTM Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train Data', 'Validation Data'], loc='upper left')
    plt.savefig(f"data/Data Exploration/Accuracy_Regularized_LSTM.jpg")
    plt.clf()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('CNN LSTM Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train Data', 'Validation Data'], loc='upper left')
    plt.savefig(f"data/Data Exploration/Loss_Regularized_LSTM.jpg")
    
def test(image) -> None:
    """
    Tests model using test data provided.

    Parameters:
        test_data (image): The uploaded set of test data.
    Returns:
        None. It just tests data.
    """
    
    # Resize image
    img = load_img(image, target_size=(128, 128))
    
    # Normalize image
    img_array = img_to_array(img) / 255.0
    
    # Load its metadata
    img_features = pd.read_csv("data/try.csv")
    
    # Reshape features
    timezone = np.array(img_features['Timezone']).reshape(-1,1)
    hour = np.array(img_features['Hour']).reshape(-1,1)
    month = np.array(img_features['Month']).reshape(-1,1)
    
    # Concatenate arrays
    y = np.concatenate([month, hour, timezone], axis=1)
    
    # Load Keras model
    model = load_model("saved_model\h5\Heatsnap.h5")
    
    # Multiclass Labels   
    tempClasses = [
    'Extreme Cold Danger',
    'Cold Danger',
    'Extreme Cold Caution',
    'Cold Caution',
    'Safe',
    'Heat Caution',
    'Extreme Heat Caution',
    'Heat Danger',
    'Extreme Heat Danger'
    ]
    
    # Predict data   
    prediction = tempClasses[np.argmax(model.predict([np.expand_dims(img_array, axis=0), y]))]
    
    print(prediction)

    return

def process_data():
    
    # Load dataset
    data = pd.read_csv(DATASET)
    
    # Drop row TempM with NaN values
    data = data.dropna(subset=['TempM'])
    
    # Make TempM integers
    data["TempM"] = data['TempM'].round().astype(int)

    # Categorize TempM
    data['TempClass'] = data['TempM'].apply(categorize_temperature)
    
    # Save new dataset
    data.to_csv('data/classification_data.csv', index=False)
    
# MAIN
def main():
    test_image = "data\IMG_20240529_105051.jpg"
    
    #process_data()
    
    #train_save_model()
    
    test(test_image)

def categorize_temperature(temp):
    if temp <= -60:
        return 'Extreme Cold Danger'
    elif -59 <= temp <= -45:
        return 'Cold Danger'
    elif -44 <= temp <= -25:
        return 'Extreme Cold Caution'
    elif -24 <= temp <= 0:
        return 'Cold Caution'
    elif 1 <= temp <= 27:
        return 'Safe'    
    elif 28 <= temp <= 32:
        return 'Heat Caution'
    elif 33 <= temp <= 41:
        return 'Extreme Heat Caution'
    elif 42 <= temp <= 51:
        return 'Heat Danger'
    elif temp >= 52:
        return 'Extreme Heat Danger'

if __name__ == "__main__":
    main()