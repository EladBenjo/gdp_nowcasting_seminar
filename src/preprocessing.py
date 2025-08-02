import os
import pandas as pd
from typing import Dict

def load_csv_folder_as_dataframes(folder_path: str, suffix: str = '_df') -> Dict[str, pd.DataFrame]:
    """
    Loads all CSV files from a given folder into a dictionary of pandas DataFrames.

    Parameters:
        folder_path (str): The path to the folder containing CSV files.
        suffix (str): Optional suffix to append to dataframe names (default is '_df').

    Returns:
        Dict[str, pd.DataFrame]: A dictionary where keys are file-based names and values are DataFrames.
    """
    data_frames = {}

    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df_name = filename.replace('.csv', suffix)
            try:
                df = pd.read_csv(file_path)
                data_frames[df_name] = df
                print(f"[INFO] Loaded {filename} as '{df_name}'")
            except Exception as e:
                print(f"[ERROR] Failed to load {filename}: {e}")

    return data_frames

def info_and_describe(data_frames):
    """
    Prints information and descriptive statistics for each DataFrame in a dictionary.

    Args:
        data_frames (dict): A dictionary where keys are DataFrame names (str) and values are pandas DataFrame objects.

    Returns:
        None

    Side Effects:
        Prints the output of DataFrame.info() and DataFrame.describe() for each DataFrame in the dictionary.
    """
    for df_name, df in data_frames.items():
        print(f"\n--- Info for {df_name} ---")
        df.info()
        print(f"\n--- Description for {df_name} ---")
        print(df.describe())

def clean_month_year(series):
    """
    Parses a pandas Series of date strings into datetime objects.
    Supports formats like 'Jan-96', '25-Apr', and '1-Nov' (assumed 2001).

    Args:
        series (pd.Series): Series of date strings.

    Returns:
        pd.Series: Series of datetime64[ns] objects.
    """
    def parse_date(val):
        if not isinstance(val, str) or '-' not in val:
            return pd.NaT

        parts = val.split('-')

        # Case: "Jan-96"
        if len(parts[1]) == 2 and parts[0].isalpha():
            return pd.to_datetime(val, format='%b-%y', errors='coerce')

        # Case: "25-Apr" → assume fixed year (e.g., 2025)
        elif len(parts[0]) == 2 and parts[0].isdigit():
            return pd.to_datetime(f"{parts[1]}-2025", format='%b-%Y', errors='coerce')

        # Case: "1-Nov", "2-Jan", etc. → assume "Nov 2001", "Jan 2002"
        elif parts[0].isdigit() and len(parts[0]) == 1:
            year = 2000 + int(parts[0])
            return pd.to_datetime(f"{parts[1]}-{year}", format='%b-%Y', errors='coerce')

        return pd.NaT

    return series.apply(parse_date)

def convert_object_columns_to_float(df):
    """
    Converts all columns of type 'object' to 'float', excluding the 'Date' column.

    Args:
        df (pd.DataFrame): Input DataFrame to modify in-place or return modified copy.

    Returns:
        pd.DataFrame: A DataFrame with object columns converted to float where applicable.
    """
    for col in df.columns:
        if col != "Date" and df[col].dtype == 'object':
            try:
                df[col] = df[col].str.replace(",", "")  # Optional: remove thousands separator
                df[col] = df[col].astype(float)
            except Exception as e:
                print(f"Could not convert column '{col}' to float: {e}")
    return df

def clean_percentage_columns(df):
    """
    Removes '%' character from all object columns and converts them to float if possible.

    Args:
        df (pd.DataFrame): DataFrame with potential percentage strings (e.g., '42%').

    Returns:
        pd.DataFrame: Modified DataFrame with cleaned and converted columns.
    """
    for col in df.columns:
        if df[col].dtype == 'object':
            if df[col].str.contains('%').any():
                try:
                    df[col] = df[col].str.replace('%', '', regex=False).astype(float)
                except Exception as e:
                    print(f"Could not convert column '{col}' to float: {e}")
    return df

def print_date_range(df, date_column='Date'):
    """
    Prints the earliest and latest dates in a DataFrame's date column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        date_column (str): The name of the column containing date values.
    """
    if date_column not in df.columns:
        print(f"Error: Date column '{date_column}' not found in the DataFrame.")
        return

    # Ensure the date column is in datetime format
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

    # Drop rows where date conversion failed
    df_cleaned = df.dropna(subset=[date_column])

    if not df_cleaned.empty:
        earliest_date = df_cleaned[date_column].min()
        latest_date = df_cleaned[date_column].max()
        print(f"Earliest date: {earliest_date}")
        print(f"Latest date: {latest_date}")
    else:
        print(f"No valid dates found in the '{date_column}' column.")