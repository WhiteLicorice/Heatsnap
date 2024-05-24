"""Module for Exploring the Dataset"""

import pandas as pd
import matplotlib.pyplot as plt

###   TODO: Define the path to the dataset

#   Complete dataset
DATASET = 'data/no_duplicates.csv'

#   One-year dataset
#DATASET = 'data\sky_pictures_dataset_time_ascending_1_year.csv'

#   PLAN: We'll selectively plot temperature against columns that seem useful
def main():
    data = pd.read_csv(DATASET)
    data = data.drop(columns=['TempI', 'Min'])     #   Drop degrees Fahrenheit since we'll be using degrees Celsius, drop Minutes since lots of bad entries
    data = data.loc[data['TempM'] != -9999]        #   Filter against -9999 degrees Celsius entries
    
    
    
    plt_temperature_month(data)
    
    """ plt_longitude_latitude(data)
    plt_temperature_hour(data)
    plt_temperature_month(data)
    plt_temperature_timezone(data) """

#   Ren's Notes: Temperature seems to spike at around 3—4 PM
def plt_temperature_hour(data):
    plt.figure(figsize=(10, 6))
    plt.scatter(data['Hour'], data['TempM'], alpha=0.5)
    plt.xlabel('Hour (24H)')
    plt.ylabel('Temperature (°C)')
    plt.title('Temperature vs Hour of the Day')
    plt.show()
    
def plt_temperature_day(data):
    plt.figure(figsize=(10, 6))
    plt.scatter(data['Day'], data['TempM'], alpha=0.5)
    plt.xlabel('Day')
    plt.ylabel('Temperature (°C)')
    plt.title('Temperature vs Day')
    plt.show()
    
def plt_temperature_day(data):
    plt.figure(figsize=(10, 6))
    plt.scatter(data['Day'], data['TempM'], alpha=0.5)
    plt.xlabel('Day')
    plt.ylabel('Temperature (°C)')
    plt.title('Temperature vs Day')
    plt.show()

#   Ren's Notes: Temperature seems to spike around May to June, then taper off toward July onward
def plt_temperature_month(data):
    plt.figure(figsize=(10, 6))
    plt.scatter(data['Month'], data['TempM'], alpha=0.5)
    plt.xlabel('Month')
    plt.ylabel('TempM (°C)')
    plt.title('Temperature vs Month')
    plt.show()

#   Ren's Notes: Longitude and latitude don't seem to be useful since the hexbin plot is concentrated around sparse coordinates
def plt_longitude_latitude(data):
    plt.figure(figsize=(10, 6))
    plt.hexbin(data['Longitude'], data['Latitude'], gridsize=50, cmap='viridis', mincnt=1)
    plt.colorbar(label='Count')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Hexbin Plot of Longitude and Latitude')
    plt.show()

#   Ren's Notes: Timezone seems useful since there's a trend and few outliers, with hotspots concentrated around -8 to -6 UTC (Pacific & Carribean)
def plt_temperature_timezone(data):
    plt.figure(figsize=(10, 6))
    plt.scatter(data['Timezone'], data['TempM'], alpha=0.5)
    plt.xlabel('Timezone (UTC)')
    plt.ylabel('Temperature (°C)')
    plt.title('Temperature vs Timezone')
    plt.show()

def plt_temperature(data):
    cam_id = data['CamId'].iloc[0]
    
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['TempM'], 'o', alpha=0.5)  # Using 'o' to get scatter plot points
    plt.xlabel(f'Index of {cam_id}')
    plt.ylabel('Temperature (°C)')
    plt.title('Temperature Over Index')
    plt.show()
    
def partition_dataset_by_column(dataframe, column):
    partitioned_list = []
    grouped = dataframe.groupby(column)
    
    for name, group in grouped:
        partitioned_list.append(group)
    
    return partitioned_list
    
# Run the main function
if __name__ == "__main__":
    main()
