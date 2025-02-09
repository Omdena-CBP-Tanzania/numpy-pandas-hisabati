import numpy as np
import pandas as pd
import assignment
def create_1d_array():

    """
    Create a 1D NumPy array with values [1, 2, 3, 4, 5]
    Returns:
        numpy.ndarray: 1D array
    """
    arr_1d = np.array([1, 2, 3, 4, 5])
    return arr_1d
    print(arr_1d)
    pass

def create_2d_array():
    """
    Create a 2D NumPy array with shape (3,3) of consecutive integers
    Returns:
        numpy.ndarray: 2D array
    """
    array_2D = np.array([[1, 2,3], [4, 5, 6], [7, 8, 9]])
    return array_2D
    print(array_2D)
    pass

def array_operations(arr):
    """
    Perform basic array operations:
    1. Calculate mean
    2. Calculate standard deviation
    3. Find max value
    Returns:
        tuple: (mean, std_dev, max_value)
    """
    array_operations = np.array([2, 3, 4, 5])
    mean = np.mean(array_operations)
    std_dev = np.std(array_operations)
    max_value = np.max(array_operations)
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
    data = pd.read_csv ("C:/Users/emmanueljames/Desktop/Hesabu/numpy-pandas-hisabati/data/sample-data.csv")
    data_frame = pd.read_csv(data)
    print (data)

def handle_missing_values(df):
    """
    Handle missing values in the DataFrame
    1. Identify number of missing values
    2. Fill missing values with appropriate method
    Returns:
        pandas.DataFrame: Cleaned dataframe
    """
    df.isnull().sum()
    for column in df.columns:
        if df[column].dtype in ['int64', 'float']:
            df[column].fillna(df[column].mean()), inplace = True
        else:
            df[column].fillna(df[column].mode()[0], inplace = True)
            return df
    pass

def select_data(df):
    """
    Select specific columns and rows from DataFrame
    Returns:
        pandas.DataFrame: Selected data
    """
    selected_columns = df[['Name' ,'Age','Salary',]]
    print("Selected columns: ",selected_columns.head())
    
    # specific rows selected
    selected_rows = selected_columns.loc[selected_columns['Age']>25]
    return selected_rows

def rename_columns(df):
    """
    Rename columns of the DataFrame
    Returns:
        pandas.DataFrame: DataFrame with renamed columns
    """
    renamed_df = df.rename(columns = columns_rename)
    columns_rename = { 
        "Name": "Office Name", 
        "Salary": "Monthly Payment"}, inplace = True
    return columns_rename


