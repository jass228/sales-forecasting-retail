"""
@author: Joseph A.
Description: Module for loading and preprocessing sales data.
"""
from pathlib import Path
import pandas as pd

def load_data(filepath: str | Path) -> pd.DataFrame:
    """ Load sales data from a CSV file and perform basic preprocessing.

    Args:
        filepath (str | Path): Path to the CSV file.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    df = pd.read_csv(filepath)

    # Basic preprocessing
    df['date'] = pd.to_datetime(df['date']) # Convert date column to datetime
    # Drop unnecessary columns if they exist
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"]) 

    return df

def get_data_info(df: pd.DataFrame) -> dict:
    """ Get information about the loaded sales data.

    Args:
        df (pd.DataFrame): Preprocessed DataFrame.

    Returns:
        dict: Dictionary containing information about the data.
    """
    return {
        "n_rows": df.shape[0],
        "n_columns": df.shape[1],
        "date_min": df['date'].min(),
        "date_max": df['date'].max(),
        "n_agencies": df['agency'].nunique(),
        "n_skus": df['sku'].nunique(),
        "columns": df.columns.tolist()
    }

def split_train_test(df: pd.DataFrame, test_start_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ Split data into training and testing sets based on date.

    Args:
        df (pd.DataFrame): DataFrame with sales data.
        test_start_date (str): Date string to start the test set (inclusive).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Training and testing DataFrames.
    """
    test_start = pd.to_datetime(test_start_date)

    train_df = df[df['date'] < test_start].copy()
    test_df = df[df['date'] >= test_start].copy()

    return train_df, test_df
