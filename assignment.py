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
arr_1d = create_1d_array()
print(arr_1d)

def create_2d_array():
    """
    Create a 2D NumPy array with shape (3,3) of consecutive integers
    Returns:
        numpy.ndarray: 2D array
    """
    arr2 = np.array([[1,2,3],[4,5,6],[7,8,9]])
    return arr2
arr_2d = create_2d_array() #calling out the function
print(arr_2d)

def array_operations(arr):
    """
    Perform basic array operations:
    1. Calculate mean
    2. Calculate standard deviation
    3. Find max value
    Returns:
        tuple: (mean, std_dev, max_value)
    """
    #to calculate mean
    mean_value = np.mean(arr)

    #to calculate standard deviation
    std_dev = np.std(arr)

    #to find max value
    max_value = np.max(arr)
    return (mean_value, std_dev, max_value)
arr = np.array([3,5,7])
array_ops = array_operations(arr)
print(array_ops)


def read_csv_file(filepath):
    """
    Read a CSV file using Pandas
    Args:
        filepath (str): Path to CSV file
    Returns:
        pandas.DataFrame: Loaded dataframe
    """
    return pd.read_csv(filepath)

filepath = r"C:\Users\user\Omdena\numpy-pandas-hisabati\data\sample-data.csv"
data_frame = read_csv_file(filepath)
print(data_frame.head())


def handle_missing_values(df):
    """
    Handle missing values in the DataFrame
    1. Identify number of missing values
    2. Fill missing values with appropriate method
    Returns:
        pandas.DataFrame: Cleaned dataframe
    """
    #identify number of missing values
    missing_values_before = df.isna().sum()
    print("Missing value before cleaning is: ", missing_values_before)

    #To fill missing values with mean method
    dataframe_filled = df.fillna(df.select_dtypes(include=['number']).median())


    #to verify again missing value 
    missing_value_after = dataframe_filled.isna().sum()
    print("Missing values after cleaning is: ", missing_value_after)

    return dataframe_filled
filepath = r"C:\Users\user\Omdena\numpy-pandas-hisabati\data\sample-data.csv"
df = pd.read_csv(filepath)

#Handle missing values
cleaned_dataframe = handle_missing_values(df)

# Print the cleaned dataframe
print("\nCleaned DataFrame:")
print(cleaned_dataframe.head())


def select_data(df):
    """
    Select specific columns and rows from DataFrame
    Returns:
        pandas.DataFrame: Selected data
    """
    # Selected specific columns
    selected_columns = df[['Name' ,'Age','Salary',]]
    print("Selected columns: ",selected_columns.head())
    
    #selected specific rows
    selected_rows = selected_columns.loc[selected_columns['Age']>25] #select rows where the Experience > 10 years
    return selected_rows
filepath = r"C:\Users\user\Omdena\numpy-pandas-hisabati\data\sample-data.csv"
df = pd.read_csv(filepath)
#selected data
selected_data = select_data(df)
print("\nSelected Data (Experience > 10) is: \n",selected_data)

def rename_columns(df):
    """
    Rename columns of the DataFrame
    Returns:
        pandas.DataFrame: DataFrame with renamed columns
    """
    column_renames = {
        'Name': 'Employee Name',
        'Salary': 'Monthly Salary',
    }

    # Rename the columns using the .rename() method
    renamed_df = df.rename(columns=column_renames)

    return renamed_df
filepath = r"C:\Users\user\Omdena\numpy-pandas-hisabati\data\sample-data.csv"
df = pd.read_csv(filepath)
renamed_columns = rename_columns(df)
print("\nRenamed Columns: \n",renamed_columns.head())
    
