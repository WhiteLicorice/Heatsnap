"""

    A module for collecting the images in the Skyfinder dataset.
    Usage: sample()
    Return: X_train, X_val, X_test, y_train, y_val, y_test
    Where X_train are images as arrays and Y are the features defined in FEATURES
    
"""

import os
import time

import numpy as np
import pandas as pd

os.environ["KERAS_BACKEND"] = "torch"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from keras.preprocessing.image import load_img, img_to_array

from sklearn.model_selection import train_test_split

DIRECTORY = 'data/sky_pictures'
DATASET = 'data/complete_table_with_mcr.csv'
FEATURES = ['Filename', 'CamId', 'TempM', 'Month', 'Hour', 'Timezone']
IMAGE_SIZE = (128, 128)
IMAGES_PER_DIRECTORY = 500

#   Usage: just call sample() directly, unless you want to change the defaults
def sample(directory = DATASET, features = FEATURES, max_images_per_directory = IMAGES_PER_DIRECTORY):
    data = pd.read_csv(directory)
    data = data[features]
    data['UniqueId'] = data['CamId'].astype(str) + '_' + data['Filename']
    
    duplicates = data['UniqueId'].duplicated().sum()
    if duplicates > 0:
        raise Exception("Duplicates found in UniqueId column.")
    
    #camids = data['CamId'].astype(str).tolist()
    unique_ids = data['UniqueId'].tolist()
    
    matched_image_paths = []
    tempM = []
    month = []
    hour = []
    timezone = []

    images_per_directory = {}
    
    for _, row in data.iterrows():
        camid = row['CamId']
        filename = row['Filename']
        unique_id = f"{camid}_{filename}"
        image_path = os.path.join(DIRECTORY, str(camid), filename)
        
        print("Attempting to load image:", image_path)
        
        if os.path.exists(image_path) and unique_id in unique_ids:
            try:
                if camid not in images_per_directory:
                    images_per_directory[camid] = 0
                    
                if images_per_directory[camid] < max_images_per_directory:
                    img = load_img(image_path, target_size=IMAGE_SIZE)
                    img_array = img_to_array(img) / 255.0   #   Normalize image as array
                    matched_image_paths.append(img_array)
                    
                    tempM.append(row['TempM'])
                    month.append(row['Month'])
                    hour.append(row['Hour'])
                    timezone.append(row['Timezone'])
                    images_per_directory[camid] += 1
                    
            except OSError as e:
                print(f"Error loading image: {image_path}")
                print(e)
                continue
    
    images = np.array(matched_image_paths)
    
    tempM = np.array(tempM).reshape(-1, 1)
    month = np.array(month).reshape(-1, 1)
    hour = np.array(hour).reshape(-1, 1)
    timezone = np.array(timezone).reshape(-1, 1)

    #   Concatenate all target values
    y = np.concatenate([tempM, month, hour, timezone], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(images, y, test_size=0.2, random_state=42)
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test
    
#   Unit test for the sample() function
def test_sample():
    start_time = time.time()

    #   Call the sample function to get the datasets
    X_train, X_val, X_test, y_train, y_val, y_test = sample()
    
    print(f"Time taken: {time.time() - start_time}")

    #   Check shapes of the datasets
    print("Shapes of datasets:")
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_val:", y_val.shape)
    print("y_test:", y_test.shape)

    #   Optionally, print sample data points and labels
    print("\nSample data points and labels:")
    print("X_train sample:", X_train[0])  
    print("y_train label:", y_train[0])  
    print("X_val sample:", X_val[0])     
    print("y_val label:", y_val[0])
    print("X_test sample:", X_test[0])
    print("y_test label:", y_test[0]) 
    
    print("Sample ran successfully!")

if __name__ == "__main__":
    test_sample()
    
   