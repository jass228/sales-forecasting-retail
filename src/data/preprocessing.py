"""
@author: Joseph A.
Description: Data preparation module
"""
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from src.configs.config import KEY_COLS, DATE_COL, TARGET_COL

DROP_COLS = ["Unnamed: 0", "timeseries"]

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess raw data."""
    df = df.copy()

    df = df.drop(columns=DROP_COLS, errors="ignore")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(KEY_COLS + [DATE_COL]).reset_index(drop=True)

    constant_cols = df.columns[df.nunique(dropna=False) <= 1]
    df = df.drop(columns=constant_cols)

    # Check for missing values in target
    n_missing = df[TARGET_COL].isna().sum()
    if n_missing > 0:
        raise ValueError(f"Found {n_missing} missing values in target.")

    # Check for duplicates
    duplicates = df.duplicated(subset=KEY_COLS + [DATE_COL]).sum()
    if duplicates > 0:
        raise ValueError(f"Found {duplicates} duplicated rows.")

    # Reorder columns: date, keys, target, then exogenous
    exog_cols = [c for c in df.columns if c not in KEY_COLS + [DATE_COL, TARGET_COL]]
    ordered_cols = [DATE_COL] + KEY_COLS + [TARGET_COL] + exog_cols
    return df[ordered_cols].copy()

def split_train_test(df: pd.DataFrame,
                    n_val_periods: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ Split data into train and test sets based on temporal split.
    """
    max_date = df[DATE_COL].max()
    cutoff_date = max_date - pd.DateOffset(months=n_val_periods)

    train_df = df[df[DATE_COL] < cutoff_date].copy()
    test_df = df[df[DATE_COL] >= cutoff_date].copy()

    return train_df, test_df

def encode_data(df: pd.DataFrame, encoder: OrdinalEncoder = None
            ) -> tuple[pd.DataFrame, OrdinalEncoder]:
    """Encode categorical key columns.
    """
    df = df.copy()
    if encoder is None:
        encoder = OrdinalEncoder()
        df[KEY_COLS] = encoder.fit_transform(df[KEY_COLS])
    else:
        df[KEY_COLS] = encoder.transform(df[KEY_COLS])

    return df, encoder

def save_data(df: pd.DataFrame, path: str) -> None:
    """Save DataFrame to CSV file.
    """
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
