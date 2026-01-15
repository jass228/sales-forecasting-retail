"""
@author: Joseph A.
Description: Inference module for sales forecasting.
"""
from pathlib import Path
import joblib
import pandas as pd

#pylint: disable=C0103:invalid-name

def load_model(model_path: str):
    """Load trained model from pickle file.
    """
    filepath = Path(model_path)
    if not filepath.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(filepath)

def predict(model, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Generate predictions using trained model.
    """
    df = df.copy()
    X = df[feature_cols]
    df["prediction"] = model.predict(X)
    return df
