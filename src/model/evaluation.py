"""
@author: Joseph A.
Description: Evaluation module.
"""
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error
)

#pylint: disable=C0103:invalid-name

def evaluate_model(model, X_val: pd.DataFrame, y_val: pd.Series) -> dict:
    """Evaluate model performance on validation set.
    """
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = root_mean_squared_error(y_val, y_pred)
    mape = mean_absolute_percentage_error(y_val, y_pred)

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape
    }
