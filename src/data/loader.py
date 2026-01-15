"""
@author: Joseph A.
Description: Loader data module
"""
from pathlib import Path
import pandas as pd

def load_data(filepath: str | Path) -> pd.DataFrame:
    """Load data from a CSV file.
    """
    df = pd.read_csv(filepath)
    return df
