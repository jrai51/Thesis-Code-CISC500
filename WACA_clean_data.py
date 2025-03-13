import pandas as pd
import numpy as np


def clean_data(filepath):
    column_names = ["sensor", "timestamp", "x", "y", "z"]
    column_types = {
        "sensor": "float32",      # Enforce integer type for sensor
        "timestamp": "float32", # Enforce float type for timestamp
        "x": "float32",         # Enforce float type for x
        "y": "float32",         # Enforce float type for y
        "z": "float32"          # Enforce float type for z
    }
    

    try:
        #load data 
        df = pd.read_csv(filepath, on_bad_lines="skip", header=None, usecols=range(5), names=column_names, dtype=column_types)
        
        # Filter only relevant sensors
        df = df[df['sensor'].isin([10, 4])]
        df = df.dropna()

        # Sort by timestamp
        df = df.sort_values(by=['timestamp'])
        
        # First discard 2.5 - 5 seconds from start and end of the timeseries, Corresponds to user putting on/taking off the smartwatch 
        # (should this be done before or after concatenating sensor axes?)
        df = df.iloc[250:-250]
        
        return df

    except ValueError as e:
        print(f"Error reading file {filepath}: {e}")
        return None
    

def match_accel_gyro_data(accel, gyro):
    """
    Matches accelerometer and gyroscope data by timestamp using nearest neighbor merge.

    Args:
        accel (pd.DataFrame): Accelerometer data.
        gyro (pd.DataFrame): Gyroscope data.

    Returns:
        dict: Matched accelerometer and gyroscope data.
    """

    accel_count = accel.shape[0]
    gyro_count = gyro.shape[0]
    column_names = ['sensor', 'timestamp', 'x', 'y', 'z']

    if accel_count > gyro_count:
        df = pd.merge_asof(accel, gyro, on="timestamp", direction='nearest')
        df = df.dropna()
        accel = df[["sensor_x", "timestamp", "x_x", "y_x", "z_x"]]
        gyro = df[["sensor_y", "timestamp", "x_y", "y_y", "z_y"]]
    else:
        df = pd.merge_asof(gyro, accel, on="timestamp", direction='nearest')
        df = df.dropna()
        gyro = df[["sensor_x", "timestamp", "x_x", "y_x", "z_x"]]
        accel = df[["sensor_y", "timestamp", "x_y", "y_y", "z_y"]]

    accel.columns = column_names
    gyro.columns = column_names
    
    return {'accel': accel, 'gyro': gyro}


def combine_sensor_data(df, print_stats=True):
    """
    Combines accelerometer (sensor 10) and gyroscope (sensor 4) data
    by matching timestamps and merging into a single DataFrame.

    Args:
        df (pd.DataFrame): Cleaned DataFrame from clean_data().
        print_stats (bool): Whether to print sensor and merged dataframe counts.

    Returns:
        pd.DataFrame: DataFrame with combined sensor data (x_a, y_a, z_a, x_g, y_g, z_g).
    """

    # Extract Accelerometer values and sort
    accel = df[df.sensor == 10].sort_values(by=['timestamp'])

    # Extract Gyroscope values and sort
    gyro = df[df.sensor == 4].sort_values(by=['timestamp'])

    if print_stats:
        print(f"Accelerometer count: {accel.shape[0]}, Gyroscope count: {gyro.shape[0]}")

    # Match timestamps between accel and gyro
    result = match_accel_gyro_data(accel, gyro)
    accel, gyro = result['accel'], result['gyro']

    # Rename columns for clarity
    accel = accel.rename(columns={"x": "x_a", "y": "y_a", "z": "z_a"})
    gyro = gyro.rename(columns={"x": "x_g", "y": "y_g", "z": "z_g"})

    # Merge data on timestamps
    merged_df = pd.merge(accel, gyro, on='timestamp', suffixes=('_a', '_g'))
    

    if print_stats:
        print(f"Final merged count: {merged_df.shape[0]}")

    return merged_df[['x_a', 'y_a', 'z_a', 'x_g', 'y_g', 'z_g']] #Not including any other information such as sensor label or timestamp


# def get_valid_users(num_users, typing_task):
#     valid_users = []
#     for user in range(1, 50):
#         current_user = f"user{user}"
#         path_to_train = f"data/WACA/WACA/WACA_dataset/{current_user}_{typing_task}.csv"
#         if not os.path.exists(path_to_train): continue

#         df = clean_data(path_to_train)
#         if df is None:
#             continue

#         sufficient_count = has_sufficient_data(df)
#         if sufficient_count:
#             valid_users.append(current_user)
#         else:
#             print(f"Lacking sufficient sensor data: {current_user}")
#     return valid_users

def save_user_train_data(path_to_train, path_to_csv="data/WACA/WACA/train.csv"):
    """
    path_to_train :  filepath to the user data you training on
    path_to_csv :   name of the file to save the processed test data
    """
    clean_df = clean_data(path_to_train)
    df = combine_sensor_data(clean_df)
    df.to_csv(path_to_csv, index=False)
        
def save_user_test_data(path_to_test, path_to_csv="data/WACA/WACA/test.csv"):
    """
    path_to_test :  filepath to the user data you are testing against
    path_to_csv :   name of the file to save the processed test data
    """
    clean_df = clean_data(path_to_test)
    df = combine_sensor_data(clean_df)
    df.to_csv(path_to_csv, index=False)        

def save_sensor_specific_train_data(path_to_train, sensor_type, path_to_npy="data/WACA/WACA/WACA_train_sensor.npy", path_to_csv="data/WACA/WACA/train_sensor.csv"):
    """
    FOR GYROSCROPE: sensor_type = 4
    FOR ACCELEROMETER: sensor_type = 10    
    """
    
    df = clean_data(path_to_train)
    if df is not None:
        df_sensor = df[df['sensor'] == sensor_type]
        if not df_sensor.empty:
            np.save(path_to_npy, df_sensor.to_numpy())
            df_sensor.to_csv(path_to_csv, index=False)
            
def save_sensor_specific_test_data(path_to_test, sensor_type, 
                                   path_to_npy="data/WACA/WACA/WACA_test_sensor.npy", 
                                   path_to_csv="data/WACA/WACA/test_sensor.csv"):
    """
    Save test data specific to a given sensor type.

    path_to_test :  filepath to the user data you are testing against
    sensor_type  :  sensor type to filter (e.g., 10 for gyroscope, 4 for accelerometer)
    path_to_npy  :  Name of the npy file to save
    path_to_csv  :  Name of the csv file to save
    """
    
    df = clean_data(path_to_test)
    if df is not None:
        df_sensor = df[df['sensor'] == sensor_type]
        if not df_sensor.empty:
            np.save(path_to_npy, df_sensor.to_numpy())
            df_sensor.to_csv(path_to_csv, index=False)

            
        
def save_user_test_label_data(impostor, 
                              path_to_csv="data/WACA/WACA/test.csv", 
                              path_to_test_label="data/WACA/WACA/test_label.csv",
                              timestamps_present=False):
    """
    Save the testing data and create a labels file.

    impostor    :   True if the user is an impostor, False if the user is the authentic user       
    path_to_csv :   Name of the csv file containing the test data
    path_to_test_label : Name of the test label csv file to save
    """

    # Assumes the user test data has already been saved using save_user_test_data

    # Load the saved test data to create labels
    df = pd.read_csv(path_to_csv)
    
    # Create labels based on impostor status
    labels = [1 if impostor else 0] * len(df)
    
    if timestamps_present:
        # Create a DataFrame for labels
        labels_df = pd.DataFrame({
            "timestamp": df["timestamp"],
            "label": labels
        })
        # Save the labels DataFrame to a CSV file
        labels_df.to_csv(path_to_test_label, index=False)
    
    else:
        labels_df = pd.DataFrame({
            "label": labels
        })     

    print(labels_df.head())
    # Save the labels DataFrame to a CSV file
    labels_df.to_csv(path_to_test_label, index=True)

    

        

if __name__ == "__main__":
    path_to_train = "WACA_dataset/user1_2.csv"
    cleaned_data = clean_data(path_to_train)
    merged_df = combine_sensor_data(cleaned_data)
    print(merged_df.head())

  