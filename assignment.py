import numpy as np
import pandas as pd

def create_1d_array():
    """
    Create a 1D NumPy array with values [1, 2, 3, 4, 5]
    Returns:
        numpy.ndarray: 1D array
    """
    return np.array([1, 2, 3, 4, 5])

def create_2d_array():
    """
    Create a 2D NumPy array with shape (3,3) of consecutive integers
    Returns:
        numpy.ndarray: 2D array
    """
    return np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
 
def array_operations(arr):
    """
    Perform basic array operations:
    1. Calculate mean
    2. Calculate standard deviation
    3. Find max value
    Returns:
        tuple: (mean, std_dev, max_value)
    """
    return arr.mean(), arr.std(), arr.max()

def read_csv_file(filepath):
    """
    Read a CSV file using Pandas
    Args:
        filepath (str): Path to CSV file
    Returns:
        pandas.DataFrame: Loaded dataframe
    """
    return pd.read_csv(filepath)

import pandas as pd

def handle_missing_values(df):
    """
    Handle missing values in the DataFrame.
    1. Identify number of missing values.
    2. Fill missing values with appropriate method.
    
    Args:
        df (pandas.DataFrame): Input DataFrame with missing values.
    
    Returns:
        pandas.DataFrame: Cleaned DataFrame with missing values handled.
    """
    # Step 1: Identify number of missing values
    df.isnull().sum()
    
    # Step 2: Fill missing values
    for column in df.columns:
        if df[column].dtype in ['int64', 'float64']:  # Numeric columns
            df[column].fillna(df[column].mean(), inplace=True)
        else:  # Non-numeric columns 
            df[column].fillna(df[column].mode()[0], inplace=True)  # Fill with the most frequent value
    return df
    

def select_data(df):
    """
    Select specific columns and rows from DataFrame
    Returns:
        pandas.DataFrame: Selected data
    """
    return df[['Name', 'Age']]

def rename_columns(df):
    """
    Rename columns of the DataFrame
    Returns:
        pandas.DataFrame: DataFrame with renamed columns
    """
    return df.rename(columns={'Name': 'Full Name', 'Age': 'Years'})
