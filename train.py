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

# Import Python custom methods
import sample

#   TODO: Define path to dataset and use this global across the script
DATASET = 'data/no_duplicates.csv'

# DATA MODEL

def get_model():
    image_shape = (128, 128, 3)
    additional_data_shape = (3,)
    
    inputs = keras.layers.Input(shape=image_shape, batch_size=1)
    
    # First TimeDistributed of Convolution Layer with MaxPool
    con2dfirst = keras.layers.Conv2D(32, (3, 3), kernel_initializer='glorot_normal', kernel_regularizer=keras.regularizers.l2(0.001), activation='relu')(inputs)
    con2dfirst = keras.layers.Dropout(0.5)(con2dfirst)
    con2dfirst = keras.layers.BatchNormalization()(con2dfirst)
    #con2dtd1 = keras.layers.TimeDistributed(con2dfirst)
    mp1 = keras.layers.MaxPooling2D((2, 2))(con2dfirst)
    #mptd1 = keras.layers.TimeDistributed(mp1)
    
    # Second TimeDistributed of Convolution Layer with MaxPool
    con2dsecond = keras.layers.Conv2D(32, (3, 3), kernel_initializer='glorot_normal', kernel_regularizer=keras.regularizers.l2(0.001), activation='relu')(mp1)
    con2dsecond = keras.layers.Dropout(0.5)(con2dsecond)
    con2dsecond = keras.layers.BatchNormalization()(con2dsecond)
    #con2dtd2 = keras.layers.TimeDistributed(con2dsecond)
    mp2 = keras.layers.MaxPooling2D((2, 2))(con2dsecond)
   #mptd2 = keras.layers.TimeDistributed(mp2)
    
    # TimeDistributed of Flatten and Dense
    f1 = keras.layers.Flatten()(mp2)
    #ftd1 = keras.layers.TimeDistributed(f1)
    d1 = keras.layers.Dense(256, kernel_initializer='glorot_normal', kernel_regularizer=keras.regularizers.l2(0.001), activation='relu')(f1)
    d1 = keras.layers.Dropout(0.5)(d1)
    d1 = keras.layers.BatchNormalization()(d1)
    #dtd1 = keras.layers.TimeDistributed(d1)
    
    # Concatenation with other information in this line
    additional_data = keras.layers.Input(shape=additional_data_shape, batch_size=1)
    merged_data = keras.layers.Concatenate()([d1, additional_data])
    
    # Two layers of TimeDistributed Dense
    d2 = keras.layers.Dense(128, kernel_initializer='glorot_normal', kernel_regularizer=keras.regularizers.l2(0.001), activation='relu')(merged_data)
    d2 = keras.layers.Dropout(0.5)(d2)
    d2 = keras.layers.BatchNormalization()(d2)
    #dtd2 = keras.layers.TimeDistributed(d2)
    d3 = keras.layers.Dense(64, kernel_initializer='glorot_normal', kernel_regularizer=keras.regularizers.l2(0.001), activation='relu')(d2)
    d3 = keras.layers.Dropout(0.5)(d3)
    d3 = keras.layers.BatchNormalization()(d3)
    #dtd3 = keras.layers.TimeDistributed(d3)
    
    reshaped_data = keras.layers.Reshape((1, 64))(d3)

    # LSTM layer
    lstm = keras.layers.LSTM(32, activation='tanh', recurrent_activation='sigmoid', dropout=0, recurrent_dropout=0, unroll=False, use_bias=True)(reshaped_data)
    
    # Last Dense layer
    d4 = keras.layers.Dense(1, name="predictions")(lstm)
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
    
    X_train, X_val, X_test, y_train, y_val, y_test = sample.sample()
    
    #   Check shapes of the datasets
    print("Shapes of datasets:")
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_val:", y_val.shape)
    print("y_test:", y_test.shape)
    
    # Make them DataFrames
    y_train = pd.DataFrame(y_train, columns=('TempM', 'Month', 'Hour', 'Timezone'))
    y_val = pd.DataFrame(y_val, columns=('TempM', 'Month', 'Hour', 'Timezone'))
    y_test = pd.DataFrame(y_test, columns=('TempM', 'Month', 'Hour', 'Timezone'))
    
    # Extract TempM only
    y_train_target = y_train.pop('TempM')
    y_val_target = y_val.pop('TempM')
    y_test_target = y_test.pop('TempM')
    
    model = get_model()
    
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
    
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[keras.metrics.RootMeanSquaredError()])
    
    model.fit([X_train, y_train], y_train_target, validation_data=([X_val, y_val], y_val_target), epochs=30, batch_size=1)
    
    loss, rmse_value = model.evaluate([X_test, y_test], y_test_target)
    print(f'Test Loss (MSE): {loss}, Test RMSE: {rmse_value}')

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
    
# MAIN
def main():
    #data = pd.read_csv(DATASET)
    #data = data.drop(columns=['TempI', 'Min'])     #   Drop degrees Fahrenheit since we'll be using degrees Celsius, drop Minutes since lots of bad entries
    #data = data.loc[data['TempM'] != -9999]        #   Filter against -9999 degrees Celsius entries
    #df_no_duplicates = data.drop_duplicates(subset=['Year', 'Day', 'Month'])

    #df_no_duplicates = data[data['CamId'] == 4801]
    
    #df_no_duplicates.to_csv('data/no_duplicates.csv', index=False)
    
    train_save_model()
    
    #   TODO: train model after reading dataset
    
    pass

if __name__ == "__main__":
    main()