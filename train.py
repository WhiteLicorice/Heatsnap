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
import numpy as np

# Import Python custom methods
import sample

from sklearn.preprocessing import OneHotEncoder

#   TODO: Define path to dataset and use this global across the script
DATASET = 'data/sky_pictures_dataset_time_ascending.csv'
BATCH_SIZE = 1

# DATA MODEL

def get_model():
    image_shape = (128, 128, 3)
    additional_data_shape = (3,)
    
    inputs = keras.layers.Input(shape=image_shape, batch_size=BATCH_SIZE)
    
    # First TimeDistributed of Convolution Layer with MaxPool
    con2dfirst = keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    #con2dtd1 = keras.layers.TimeDistributed(con2dfirst)
    mp1 = keras.layers.MaxPooling2D((2, 2))(con2dfirst)
    #mptd1 = keras.layers.TimeDistributed(mp1)
    
    # Second TimeDistributed of Convolution Layer with MaxPool
    con2dsecond = keras.layers.Conv2D(32, (3, 3), activation='relu')(mp1)
    #con2dtd2 = keras.layers.TimeDistributed(con2dsecond)
    mp2 = keras.layers.MaxPooling2D((2, 2))(con2dsecond)
   #mptd2 = keras.layers.TimeDistributed(mp2)
    
    # TimeDistributed of Flatten and Dense
    f1 = keras.layers.Flatten()(mp2)
    #ftd1 = keras.layers.TimeDistributed(f1)
    d1 = keras.layers.Dense(256, activation='relu')(f1)
    #dtd1 = keras.layers.TimeDistributed(d1)
    
    # Concatenation with other information in this line
    additional_data = keras.layers.Input(shape=additional_data_shape, batch_size=BATCH_SIZE)
    merged_data = keras.layers.Concatenate()([d1, additional_data])
    
    # Two layers of TimeDistributed Dense
    d2 = keras.layers.Dense(128, activation='relu')(merged_data)
    #dtd2 = keras.layers.TimeDistributed(d2)
    d3 = keras.layers.Dense(64, activation='relu')(d2)
    #dtd3 = keras.layers.TimeDistributed(d3)
 
    # Last Dense layer
    d4 = keras.layers.Dense(9, name="predictions", activation='softmax')(d3)
    model = keras.Model(inputs=[inputs, additional_data], outputs=d4)
    
    return model

# DATA PROCESSING FUNCTIONS

def train_save_model() -> None:
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
    
    """ encoder = OrdinalEncoder(categories = [[
    'Extreme Cold Danger',
    'Cold Danger',
    'Extreme Cold Caution',
    'Cold Caution',
    'Safe',
    'Heat Caution',
    'Extreme Heat Caution',
    'Heat Danger',
    'Extreme Heat Danger'
    ]]) """
    
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
    
    X_train, X_val, X_test, y_train, y_val, y_test = sample.sample()
    
    #   Check shapes of the datasets
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
    
    # Making sure the datatypes are not objects
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_val = y_val.astype(np.float32)
    y_test = y_test.astype(np.float32)
    y_train_target = y_train_target.astype(np.float32)
    y_val_target = y_val_target.astype(np.float32)
    y_test_target = y_test_target.astype(np.float32)
    
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
    
    model = get_model()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit([X_train, y_train], y_train_target, validation_data=([X_val, y_val], y_val_target), epochs=30, batch_size=BATCH_SIZE)
    
    loss, accuracy = model.evaluate([X_test, y_test], y_test_target)
    print(f'Categorical Crossentropy Loss: {loss}, Test Accuracy: {accuracy}')

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

def process_data():
    
    data = pd.read_csv(DATASET)
    #data = data.drop(columns=['TempI', 'Min'])     #   Drop degrees Fahrenheit since we'll be using degrees Celsius, drop Minutes since lots of bad entries
    #data = data.loc[data['TempM'] != -9999]        #   Filter against -9999 degrees Celsius entries
    #df_no_duplicates = data.drop_duplicates(subset=['Year', 'Day', 'Month'])
    
    data = data.dropna(subset=['TempM'])
    
    data["TempM"] = data['TempM'].round().astype(int)

    data['TempClass'] = data['TempM'].apply(categorize_temperature)

    
    """ data_per_day = 5
    
    df_no_duplicates = (df_no_duplicates.groupby(['Year', 'Month', 'Day'])
               .filter(lambda x: len(x) == data_per_day or len(x) > data_per_day)
               .groupby(['Year', 'Month', 'Day'])
               .apply(lambda x: x.head(data_per_day))
               .reset_index(drop=True)) """
    
    #print(df_no_duplicates.shape)
       
    #df_no_duplicates = df_no_duplicates[(df_no_duplicates['Hour'] >= 10) & (df_no_duplicates['Hour'] <= 11)]
    
    #df_no_duplicates = df_no_duplicates.drop_duplicates(subset=['Year', 'Day', 'Month'])
    
    #df_no_duplicates = df_no_duplicates.sort_values(by=['Year', 'Month', 'Day'])
    
    data.to_csv('data/classification_data.csv', index=False)
    
# MAIN
def main():
    process_data()
    
    train_save_model()
    
    #   TODO: train model after reading dataset
    
    pass

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