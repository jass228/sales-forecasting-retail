"""
@author: Joseph A.
Description: Configuration module.
"""
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "raw_data.csv"
PREDICT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "new_data.csv"
MODEL_DIR = PROJECT_ROOT / "models"
HISTORY_PATH = MODEL_DIR / "history.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Data columns
KEY_COLS = ["agency", "sku"]
DATE_COL = "date"
TARGET_COL = "volume"

# Feature engineering
LAGS = [1, 2, 3, 6, 12]
ROLLING_WINDOWS = [3, 6, 12]
HORIZONS = [1, 2, 3, 4]
