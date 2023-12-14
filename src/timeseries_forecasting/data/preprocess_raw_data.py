import pandas as pd
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch


# Function to convert radians to degrees
def radians_to_degrees(radians):
    return math.degrees(float(radians))


def radian_to_degree_per_second(rad_per_second):
    degrees_per_second = rad_per_second * (180.0 / math.pi)
    return degrees_per_second


def apply_degree_conversion(time_series_preprocessed):
    time_series_preprocessed['Angle1'] = time_series_preprocessed['Angle1'].apply(
        lambda x: radians_to_degrees(x) if not pd.isna(x) else x)
    time_series_preprocessed['Angle2'] = time_series_preprocessed['Angle2'].apply(
        lambda x: radians_to_degrees(x) if not pd.isna(x) else x)

    time_series_preprocessed['AngularVel1'] = time_series_preprocessed['AngularVel1'].apply(
        lambda x: radian_to_degree_per_second(x) if not pd.isna(x) else x)
    time_series_preprocessed['AngularVel2'] = time_series_preprocessed['AngularVel2'].apply(
        lambda x: radian_to_degree_per_second(x) if not pd.isna(x) else x)
    return time_series_preprocessed


def plot_time_series(time_series, value1, value2, value3, value4):
    # Plotting the first set of graphs
    plt.figure(figsize=(12, 5), dpi=80, linewidth=10)

    plt.subplot(2, 1, 1)
    plt.plot(time_series['Time'], time_series[value1], label=value1)
    plt.plot(time_series['Time'], time_series[value2], label=value2)
    plt.title('Chaotic pendulum - Angles')
    plt.xlabel('Seconds', fontsize=14)
    plt.ylabel('Degree', fontsize=14)
    plt.legend()

    # Plotting the second set of graphs
    plt.subplot(2, 1, 2)
    plt.plot(time_series['Time'], time_series[value3], label=value3)
    plt.plot(time_series['Time'], time_series[value4], label=value4)
    plt.title('Chaotic pendulum - Angular Velocities')
    plt.xlabel('Seconds', fontsize=14)
    plt.ylabel('Degree', fontsize=14)
    plt.legend()

    # Adjust layout to prevent clipping of titles
    plt.tight_layout()

    # Show the plot
    plt.show()


def cut_irrelevant_data(time_series):
    time_series = time_series.reset_index(drop=True)
    chunk_size = 20

    for i in range(0, len(time_series) - chunk_size + 1, chunk_size):
        chunk = time_series['Angle1'].iloc[i:i + chunk_size]

        # Check if any value in the chunk is under 90 or over -90
        if not any((chunk > 90) | (chunk < -90)):
            # Remove all values after the first 20 values that meet the condition
            index_to_remove = time_series.index[i:i + chunk_size]
            time_series = time_series.drop(index_to_remove)
            print(time_series)
            break  # Exit the loop if the condition is met for this chunk

    print("Remaining data:")
    print(time_series)


def prepare_time_series(time_series: pd.DataFrame):

    plot_time_series(time_series, 'Angle1', 'Angle2', 'AngularVel1', 'AngularVel2')
    # Remove rows with NaN values
    time_series.dropna(inplace=True)

    # Filter rows where both 'AngularVel1' and 'AngularVel2' are not nearly zero
    # time_series_preprocessed = time_series[
    #     (time_series['AngularVel1'] > 0.5) | (time_series['AngularVel1'] < -0.5) |
    #     (time_series['AngularVel2'] > 0.5) | (time_series['AngularVel2'] < -0.5) |
    #     (time_series['Angle1'] > 0.01) | (time_series['Angle1'] < -0.01) |
    #     (time_series['Angle2'] > 0.01) | (time_series['Angle2'] < -0.01)
    #     ]
    # time_series_preprocessed = time_series_preprocessed.iloc[4:]
    # time_series_preprocessed = time_series_preprocessed.iloc[:-700]
    plot_time_series(time_series, 'Angle1', 'Angle2', 'AngularVel1', 'AngularVel2')
    print(len(time_series))
    return time_series


def load_data_from_csv(path):
    return pd.read_csv(path, delimiter=";")


def main():
    path = r'C:\Users\Marco\dev\git\proj-chaotic-pendulum\src\timeseries_forecasting\data\raw\3.csv'

    # Data loading
    data = prepare_time_series(load_data_from_csv(path))

    # Specify the columns you want to drop
    columns_to_drop = ['AngularVel1', 'AngularVel2']

    # Use the drop method to remove the specified columns
    data.drop(columns=columns_to_drop, inplace=True)

    # Convert angles to sine and cosine values
    data['Sin_Angle1'] = np.sin(data['Angle1'])
    data['Cos_Angle1'] = np.cos(data['Angle1'])

    data['Sin_Angle2'] = np.sin(data['Angle2'])
    data['Cos_Angle2'] = np.cos(data['Angle2'])

    # Optionally, you can also calculate the arctangent (angle) using arctan2
    data['Converted_Angle1'] = np.arctan2(data['Sin_Angle1'], data['Cos_Angle1'])
    data['Converted_Angle2'] = np.arctan2(data['Sin_Angle2'], data['Cos_Angle2'])

    plot_time_series(data, 'Angle1', 'Angle2', 'Sin_Angle2', 'Cos_Angle2')


if __name__ == "__main__":

    # main()
    for number in range(1, 94):
        print(number)
        path = fr'C:\Users\Marco\dev\git\proj-chaotic-pendulum\DataRecords\{number}\{number}.csv'

        # Data loading
        data = prepare_time_series(load_data_from_csv(path))
        df = pd.DataFrame(data)

        # Save the DataFrame to a CSV file
        df.to_csv(fr'C:\Users\Marco\dev\git\proj-chaotic-pendulum\src\timeseries_forecasting\data\processed\{number}.csv', index=False)
