"""
@author: Joseph A.
Description: Model training module.
"""
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

def train_model(model_name: str, random_state: int = 42):
    """ Initialize and return a regression model based on the specified model name.
    """
    if model_name == "random_forest":
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            random_state=random_state,
            n_jobs=-1
        )

    elif model_name == "xgboost":
        model = XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1)

    elif model_name == "lightgbm":
        model = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state
        )
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    return model

def save_model(model, path: str) -> None:
    """Save trained model to pickle file.
    """
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
