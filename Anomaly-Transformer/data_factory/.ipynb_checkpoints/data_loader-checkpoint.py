import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import collections
import numbers
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
import pickle


class PSMSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(data_path + '/train.csv')
        data = data.values[:, 1:]

        data = np.nan_to_num(data)

        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(data_path + '/test.csv')

        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)

        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test

        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values[:, 1:]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/MSL_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/MSL_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/MSL_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMAP_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMAP_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/SMAP_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

class _WACASegLoader(Dataset):
    """ deprecated, do not use """
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        
        # Read raw data
        data = pd.read_csv(data_path + '/train.csv')
        data = data.values[:, [0, 2, 3, 4]]  # Keep sensor, x, y, z columns only
        data = np.nan_to_num(data)

        # Compute min/max for each sensor and feature (x, y, z) from train data 
        self.sensor_min_max = self.compute_sensor_min_max(data)

        # Scale the training data
        scaled_data = self.min_max_scale(data, self.sensor_min_max)
        
        # Split into train and validation sets
        data_len = len(scaled_data)
        self.train = scaled_data[:int(data_len * 0.8)]
        self.val = scaled_data[int(data_len * 0.8):]

        # Read and scale the test data
        test_data = pd.read_csv(data_path + '/test.csv')
        test_data = test_data.values[:, [0, 2, 3, 4]]  # Keep sensor, x, y, z columns only
        test_data = np.nan_to_num(test_data)
        self.test = self.min_max_scale(test_data, self.sensor_min_max)

        # Read test labels
        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values[:, 1:]
        
        print("test:", self.test.shape)
        print("train:", self.train.shape)
        print("val:", self.val.shape)

    def compute_sensor_min_max(self, data):
        """
        Compute the min and max for each feature (x, y, z) for each sensor from the raw data.
        
        Returns:
            dict: Min/max values for each feature for each sensor.
        """
        # Initialize the dictionary for storing min/max values
        sensor_min_max = {}
        # Extract columns: sensor labels, x, y, z (assuming sensor label is in column 1)
        sensor_labels = data[:, 0].astype(int)  # Assuming sensor label is in column 1
        x_values = data[:, 1]
        y_values = data[:, 2]
        z_values = data[:, 3]

        # Compute min/max for each sensor
        for sensor_label in np.unique(sensor_labels):
            sensor_label = int(sensor_label)
            sensor_mask = sensor_labels == sensor_label
            sensor_x = x_values[sensor_mask]
            sensor_y = y_values[sensor_mask]
            sensor_z = z_values[sensor_mask]

            # Store min and max for each feature (x, y, z) for the current sensor
            sensor_min_max[sensor_label] = {
                'x': (sensor_x.min(), sensor_x.max()),
                'y': (sensor_y.min(), sensor_y.max()),
                'z': (sensor_z.min(), sensor_z.max())
            }

        return sensor_min_max

    def min_max_scale(self, data, sensor_min_max):
        """
        Apply min-max scaling to the data using the provided sensor min/max values.
        
        Args:
            data (numpy array): The raw data to scale.
            sensor_min_max (dict): Dictionary containing min/max values for each feature for each sensor.
        
        Returns:
            numpy array: The scaled data.
        """
        scaled_data = data.copy()
        
        # Extract columns: sensor labels, x, y, z (assuming sensor label is in column 0)
        sensor_labels = data[:, 0].astype(int)  # Assuming sensor label is in column 0
        x_values = data[:, 1]
        y_values = data[:, 2]
        z_values = data[:, 3]

        # Apply scaling for each sensor (excluding timestamp)
        for sensor_label in np.unique(sensor_labels):
            print("Unique sensor labels:", np.unique(sensor_labels))
            sensor_label = int(sensor_label)
            # Get min and max for each feature of this sensor
            min_x, max_x = sensor_min_max[sensor_label]['x']
            min_y, max_y = sensor_min_max[sensor_label]['y']
            min_z, max_z = sensor_min_max[sensor_label]['z']

            # Scale the data for the current sensor
            sensor_mask = sensor_labels == sensor_label
            scaled_data[sensor_mask, 1] = (x_values[sensor_mask] - min_x) / (max_x - min_x)
            scaled_data[sensor_mask, 2] = (y_values[sensor_mask] - min_y) / (max_y - min_y)
            scaled_data[sensor_mask, 3] = (z_values[sensor_mask] - min_z) / (max_z - min_z)

        return scaled_data
    
    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'val':
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == 'val':
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == 'test':
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
        
class WACASegLoader(Dataset):
    """
    WACASegLoader for Anomaly Transformer implementation. 
    Datapoints are 6 columns (x_a, y_a, z_a, x_g, y_g, z_g). 
    Uses RobustScaler for training data, and min/max of training data to scale the test data.
    """

    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        print(f"Initialized WACASegLoader with win_size={self.win_size} and step={self.step}")

        # Load training data
        train_data = pd.read_csv(f"{data_path}/train.csv", header=0).values  # Convert to NumPy array

        # Step 1: Compute min/max for each column
        self.sensor_min_max = self.compute_sensor_min_max(train_data)

        # Step 2: Scale the training data using RobustScaler
        scaler = RobustScaler()
        scaled_train_data = scaler.fit_transform(train_data)

        # Split into train and validation sets
        data_len = len(scaled_train_data)
        split_idx = int(data_len * 0.8)
        self.train = scaled_train_data[:split_idx]
        self.val = scaled_train_data[split_idx:]

        # Step 3: Read and scale the test data
        test_data = pd.read_csv(f"{data_path}/test.csv", header=0).values  # Convert to NumPy array
        test_data = np.nan_to_num(test_data)  # Handle NaNs if present
        self.test = self.min_max_scale(test_data, self.sensor_min_max)

        # Read test labels
        self.test_labels = pd.read_csv(f"{data_path}/test_label.csv", header=0).values[:, 1:]  # Skip index column

        print("Test shape:", self.test.shape)
        print("Train shape:", self.train.shape)
        print("Validation shape:", self.val.shape)

    def compute_sensor_min_max(self, data):
        """
        Compute the min and max for each column in the dataset.

        Args:
            data (numpy array): The raw data array with shape (N, 6).

        Returns:
            dict: Min/max values for each column (0-5).
        """
        return {
            i: {"min": np.min(data[:, i]), "max": np.max(data[:, i])} 
            for i in range(data.shape[1])  # Iterate over all 6 columns
        }

    def min_max_scale(self, data, sensor_min_max):
        """
        Apply min-max scaling to each column using the provided min/max values.

        Args:
            data (numpy array): The raw data to scale.
            sensor_min_max (dict): Dictionary containing min/max values for each column.

        Returns:
            numpy array: The scaled data.
        """
        scaled_data = np.copy(data)  # Avoid modifying original data

        for i in range(data.shape[1]):  # Iterate over all 6 columns
            min_val = sensor_min_max[i]["min"]
            max_val = sensor_min_max[i]["max"]

            if max_val - min_val > 0:  # Avoid division by zero
                scaled_data[:, i] = (data[:, i] - min_val) / (max_val - min_val)
            else:
                scaled_data[:, i] = 0  # If min == max, set scaled values to 0

        return scaled_data
    
    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'val':
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index: index+self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == 'val':
            return np.float32(self.val[index: index+self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == 'test':
            return np.float32(self.test[index: index+self.win_size]), np.float32(
                self.test_labels[index: index+self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])        
        

class __WACASegLoader(Dataset):
    """
    WACASegLoader implementation for AnomalyTransformer. Does not include timestamps OR Sensor label. 
    Uses RobustScaler for the training data, and min/max of the training data to scale the test data.
    """
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        
        # Read raw data (lightly cleaned before added to train.csv, data sorted and insignificant sensors removed)
        raw_data = pd.read_csv(data_path + '/train.csv', header=0)
        
        # NOTE: THIS STEP SHOULD MAYBE BE DONE IN THE WACA DATA PROCESSING SCRIPT, NOT HERE
        # First discard 2.5 - 5 seconds from start and end of the timeseries 
        # Corresponds to user putting on/taking off the smartwatch
        data = raw_data.iloc[250:-250].astype("float32").reset_index(drop=True)  # Reset index to keep it clean
        
        # Compute min/max for each sensor and feature (x, y, z) from train data 
        self.sensor_min_max = self.compute_sensor_min_max(data.values[:, [0, 2, 3, 4]]) # disregards timestamps

        # Scale the training data using sklearn RobustScaler 
        ## Scaling should be done on the xyz axes, but not the sensors 
        
        # Step 1:split data into separate dfs for each sensor
        sensor_dfs = {}
        for sensor in data['sensor'].unique():
            sensor_dfs[sensor] = data[data['sensor'] == sensor].copy()

        # Step 2: Scale on each sensor
        for sensor, df_sensor in sensor_dfs.items():
            scaler = RobustScaler()
            # Fit and transform only the x, y, z columns.
            scaled_features = scaler.fit_transform(df_sensor[['x', 'y', 'z']])
            df_sensor[['x', 'y', 'z']] = scaled_features
            sensor_dfs[sensor] = df_sensor

        # Step 3: Concatenate the scaled DataFrames and sort them back into chronological order
        scaled_data = pd.concat(sensor_dfs.values()).sort_values(by="timestamp").reset_index(drop=True)
        scaled_data = scaled_data.drop(columns=["sensor", "timestamp"])
        
        
        # Split into train and validation sets
        data_len = len(scaled_data)
        self.train = scaled_data[:int(data_len * 0.8)]
        self.train = self.train.to_numpy()
        self.val = scaled_data[int(data_len * 0.8):]
        self.val = self.val.to_numpy()
        
        print("train", self.train[:5])
        
        # Read and scale the test data
        test_data = pd.read_csv(data_path + '/test.csv')
        test_data = test_data.values[:, [0, 2, 3, 4]].astype("float32")  # Keep sensor, x, y, z columns only
        test_data = np.nan_to_num(test_data)
        self.test = self.min_max_scale(test_data, self.sensor_min_max)
        self.test = self.test[:, 1:].astype("float32")  # DROP SENSOR
        print("THIS IS SELF test", self.test[:5])

        # Read test labels
        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values[:, 1:]
        
        print("test:", self.test.shape)
        print("train:", self.train.shape)
        print("val:", self.val.shape)

    def compute_sensor_min_max(self, data):
        """
        Compute the min and max for each feature (x, y, z) for each sensor from the raw data.
        
        Returns:
            dict: Min/max values for each feature for each sensor.
        """
        # Initialize the dictionary for storing min/max values
        sensor_min_max = {}
        # Extract columns: sensor labels, x, y, z (assuming sensor label is in column 1)
        sensor_labels = data[:, 0].astype(int)  # Assuming sensor label is in column 1
        x_values = data[:, 1]
        y_values = data[:, 2]
        z_values = data[:, 3]

        # Compute min/max for each sensor
        for sensor_label in np.unique(sensor_labels):
            sensor_label = int(sensor_label)
            sensor_mask = sensor_labels == sensor_label
            sensor_x = x_values[sensor_mask]
            sensor_y = y_values[sensor_mask]
            sensor_z = z_values[sensor_mask]

            # Store min and max for each feature (x, y, z) for the current sensor
            sensor_min_max[sensor_label] = {
                'x': (sensor_x.min(), sensor_x.max()),
                'y': (sensor_y.min(), sensor_y.max()),
                'z': (sensor_z.min(), sensor_z.max())
            }

        return sensor_min_max

    def min_max_scale(self, data, sensor_min_max):
        """
        Apply min-max scaling to the data using the provided sensor min/max values.
        
        Args:
            data (numpy array): The raw data to scale.
            sensor_min_max (dict): Dictionary containing min/max values for each feature for each sensor.
        
        Returns:
            numpy array: The scaled data.
        """
        scaled_data = data.copy()
        
        # Extract columns: sensor labels, x, y, z (assuming sensor label is in column 0)
        sensor_labels = data[:, 0].astype(int)  # Assuming sensor label is in column 0
        x_values = data[:, 1]
        y_values = data[:, 2]
        z_values = data[:, 3]

        # Apply scaling for each sensor (excluding timestamp)
        for sensor_label in np.unique(sensor_labels):
            print("Unique sensor labels:", np.unique(sensor_labels))
            sensor_label = int(sensor_label)
            # Get min and max for each feature of this sensor
            min_x, max_x = sensor_min_max[sensor_label]['x']
            min_y, max_y = sensor_min_max[sensor_label]['y']
            min_z, max_z = sensor_min_max[sensor_label]['z']

            # Scale the data for the current sensor
            sensor_mask = sensor_labels == sensor_label
            scaled_data[sensor_mask, 1] = (x_values[sensor_mask] - min_x) / (max_x - min_x)
            scaled_data[sensor_mask, 2] = (y_values[sensor_mask] - min_y) / (max_y - min_y)
            scaled_data[sensor_mask, 3] = (z_values[sensor_mask] - min_z) / (max_z - min_z)

        return scaled_data
    
    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'val':
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index: index+self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == 'val':
            return np.float32(self.val[index: index+self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == 'test':
            return np.float32(self.test[index: index+self.win_size]), np.float32(
                self.test_labels[index: index+self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
        
        
class SMDSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMD_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMD_test.npy")
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(data_path + "/SMD_test_label.npy")

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


def get_loader_segment(data_path, batch_size, win_size=100, step=100, mode='train', dataset='KDD'):
    if (dataset == 'SMD'):
        dataset = SMDSegLoader(data_path, win_size, step, mode)
    elif (dataset == 'MSL'):
        dataset = MSLSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'SMAP'):
        dataset = SMAPSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'PSM'):
        dataset = PSMSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'WACA'):
        dataset = WACASegLoader(data_path, win_size, step, mode)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    return data_loader
