import tensorflow as tf
import keras
import pandas as pd
import matplotlib.pyplot as plt
import time

# DATA PROCESSING FUNCTIONS

def upload_data(csv_file_path: str) -> pd.DataFrame:
    """
    Upload dataset.

    Parameters:
        csv_file_path: the file path to the chosen csv file.
        
    Returns:
        data: the dataset from the chosen csv file.
    """
    file_path = csv_file_path
    data  = pd.read_csv(file_path)
    return data

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

# DATA EXPLORATION

def data_temp_exploration(data):
    y1 = data.iloc[:,10]
    plt.plot(y1, label = data.columns[10])
    
    y2 = data.iloc[:,11]
    plt.plot(y2, label = data.columns[11])
    
    # naming the x axis
    plt.xlabel('Index')
    # naming the y axis
    plt.ylabel('Temp (no unit)')
    # giving a title to my graph
    plt.title('TempM and TempI Graph for 1 year')
    
    # show a legend on the plot
    plt.legend()
    
    # function to show the plot
    plt.show()
    
def data_long_lat_exploration(data):
    y1 = data.iloc[:,3]
    plt.plot(y1, label = data.columns[3])
    
    y2 = data.iloc[:,4]
    plt.plot(y2, label = data.columns[4])
    
    # naming the x axis
    plt.xlabel('Index')
    # naming the y axis
    plt.ylabel('Values')
    # giving a title to my graph
    plt.title('Longitude and Latitude for 1 year')
    
    # show a legend on the plot
    plt.legend()
    
    # function to show the plot
    plt.show()

def data_all_values_exploration(data):
    y1 = data.iloc[:,10]
    plt.plot(y1, label = data.columns[10])
    
    y2 = data.iloc[:,11]
    plt.plot(y2, label = data.columns[11])
    
    y3 = data.iloc[:,3]
    plt.plot(y3, label = data.columns[3])
    
    y4 = data.iloc[:,4]
    plt.plot(y4, label = data.columns[4])
    
    # naming the x axis
    plt.xlabel('Index')
    # naming the y axis
    plt.ylabel('Values')
    # giving a title to my graph
    plt.title('TempM, TempI, Longitude, and Latitude Graph for 1 year')
    
    # show a legend on the plot
    plt.legend()
    
    # function to show the plot
    plt.show()

def data_exploration(data):
    data_all_values_exploration(data)
    
# MAIN

def main():
    start_time = time.time()
    
    # Upload dataset
    csv_file_path = 'data\sky_pictures_dataset_time_ascending_1_year.csv'
    data = upload_data(csv_file_path)
    
    data = data.loc[data['TempM'] > -20]
    
    data_exploration(data)
    
    """ # Split dataset
    train_data, val_data, test_data = train_val_test_split(data)
    
    # Train and save model
    train_save_model(train_data, val_data)
    
    # Test model
    test(test_data) """
    
    elapsed_time = time.time() - start_time
    print("Elapsed time:", elapsed_time, "seconds")
    return

if __name__ == "__main__":
    main()