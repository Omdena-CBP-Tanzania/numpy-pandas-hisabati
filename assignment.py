import numpy as np
import pandas as pd

def create_1d_array(): 
    """
    Create a 1D NumPy array with values [1, 2, 3, 4, 5]
    Returns:
        numpy.ndarray: 1D array
    """
    arr = np.array([1, 2, 3, 4, 5])
    return arr

print(create_1d_array())


def create_2d_array():
    """
    Create a 2D NumPy array with shape (3,3) of consecutive integers
    Returns:
        numpy.ndarray: 2D array
    """
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    return arr

print(create_2d_array())


def array_operations(arr):
    """
    Perform basic array operations:
    1. Calculate mean
    2. Calculate standard deviation
    3. Find max value
    Returns:
        tuple: (mean, std_dev, max_value)
    """
    arr = np.array(arr)
    mean = np.mean(arr)
    std_dev = np.std(arr)
    max_value = np.max(arr)
    return (mean, std_dev, max_value)

print(array_operations([1, 2, 3, 4, 5]))


def read_csv_file(filepath):
    """
    Read a CSV file using Pandas
    Args:
        filepath (str): Path to CSV file
    Returns:
        pandas.DataFrame: Loaded dataframe
    """
    df = pd.read_csv(filepath)
    return df

print(read_csv_file("C:/Users/YYY/Omdena_numpy_pandas_assignment/numpy-pandas-hisabati/data/sample-data.csv"))


def handle_missing_values(df):
    """
    Handle missing values in the DataFrame
    1. Identify number of missing values
    2. Fill missing values with appropriate method
    Returns:
        pandas.DataFrame: Cleaned dataframe
    """
    print("Missing values per column:")
    print(df.isnull().sum())   # identify missing values per column
    
    for column in df.columns:
        if df[column].dtype in ['int64', 'float64']: # check if column is numeric
            df[column] = df[column].fillna(df[column].mean()) # fill missing values with mean

        else: # non-numeric columns
            df[column] = df[column].fillna(df[column].mode()[0]) # fill missing values with mode

    return df

print(handle_missing_values(read_csv_file("C:/Users/YYY/Omdena_numpy_pandas_assignment/numpy-pandas-hisabati/data/sample-data.csv")))


def select_data(df):
    """
    Select specific columns and rows from DataFrame
    Returns:
        pandas.DataFrame: Selected data
    """
    df = df.loc[df['Age'] > 30, ['Name', 'Age', 'Salary']]

    return df

df = read_csv_file("C:/Users/YYY/Omdena_numpy_pandas_assignment/numpy-pandas-hisabati/data/sample-data.csv")
result = select_data(df)
print(result)


def rename_columns(df):
    """
    Rename columns of the DataFrame
    Returns:
        pandas.DataFrame: DataFrame with renamed columns
    """
    df.rename(columns={'Name': 'Full Name', 'Salary': 'Annual Salary'}, inplace=True)
    return df

df = read_csv_file("C:/Users/YYY/Omdena_numpy_pandas_assignment/numpy-pandas-hisabati/data/sample-data.csv")
result = rename_columns(df)
print(result)

