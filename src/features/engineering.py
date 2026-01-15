"""
@author: Joseph A.
Description: Feature engineering module.
"""
import pandas as pd
from src.configs.config import KEY_COLS, TARGET_COL

def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """ Add calendar-based features from a date column.
    """
    df = df.copy()

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter

    return df

def add_lag_features(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    """Add lag features to the DataFrame.
    """
    df = df.copy()
    for lag in lags:
        df[f'{TARGET_COL}_lag_{lag}'] = df.groupby(KEY_COLS)[TARGET_COL].shift(lag)
    return df

def add_rolling_mean_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Add rolling mean features to the DataFrame.
    """
    df = df.copy()
    for window in windows:
        df[f'{TARGET_COL}_rolling_mean_{window}'] = (
            df.groupby(KEY_COLS)[TARGET_COL]
            .shift(1)
            .rolling(window)
            .mean()
        )
    return df

def build_features(df: pd.DataFrame, lags: list[int],
                windows: list[int]) -> pd.DataFrame:
    """Feature engineering pipeline.
    """
    df = df.copy()
    df = add_date_features(df)
    df = add_lag_features(df, lags)
    df = add_rolling_mean_features(df, windows)

    return df
