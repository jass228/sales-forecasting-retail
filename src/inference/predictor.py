"""
@author: Joseph A.
Description: Inference module for making predictions with trained models.
"""
from pathlib import Path
import pandas as pd
import joblib
from src.features.engineering import prepare_features, get_feature_columns

# pylint: disable=C0103:invalid-name

def load_model(model_path: Path) -> object:
    """Load a trained model from disk.

    Args:
        model_path (Path): Path to the saved model file.

    Returns:
        object: The loaded model object.
    """
    return joblib.load(model_path)

def load_artifacts(artifacts_path: str | Path) -> dict:
    """Load training artifacts (historical means, encoders).

    Args:
        artifacts_path (Path): Path to the directory containing artifacts.

    Returns:
        dict: Dictionary with historical_means and encoders.
    """
    return joblib.load(artifacts_path)

def save_model(model: object, model_path: str | Path) -> None:
    """Save a trained model to disk.

    Args:
        model (object): Trained model object
        model_path (str | Path): Trained model object
    """
    joblib.dump(model, model_path)

def save_artifacts(
    historical_means: dict,
    encoders: dict,
    artifacts_path: str | Path) -> None:
    """Save training artifacts to disk.

    Args:
        historical_means (dict): Dictionary of historical means DataFrames
        encoders (dict): Dictionary of categorical encoders
        artifacts_path (str | Path): Path to save the artifacts
    """
    artifacts = {
        "historical_means": historical_means,
        "encoders": encoders 
    }
    joblib.dump(artifacts, artifacts_path)

def predict(
    df: pd.DataFrame,
    model: object,
    historical_means: dict,
    encoders: dict  ) -> pd.DataFrame:
    """Make predictions on new data.

    Args:
        df (pd.DataFrame): DataFrame with raw data to predict
        model (object): Trained model object
        historical_means (dict): Historical means from training
        encoders (dict): Categorical encoders from training

    Returns:
        pd.DataFrame: DataFrame with predictions added
    """
    # Prepare features using saved artifacts
    df_prepared, _, _ = prepare_features(df, historical_means, encoders)

    # Get feature columns
    feature_cols = get_feature_columns()
    X = df_prepared[feature_cols]

    # Make predictions
    predictions = model.predict(X)

    # Add predictions to DataFrame
    df_prepared = df_prepared.copy()
    df_prepared['predicted_volume'] = predictions

    return df_prepared

def predict_from_files(
    df: pd.DataFrame,
    model_path: str | Path,
    artifacts_path: str | Path
) -> pd.DataFrame:
    """Make predictions using saved model and artifacts.

    Args:
        df (pd.DataFrame): DataFrame with raw data to predict
        model_path (str | Path): Path to saved model
        artifacts_path (str | Path): Path to saved artifacts

    Returns:
        pd.DataFrame: DataFrame with predictions added
    """
    # Load model and artifacts
    model = load_model(model_path)
    artifacts = load_artifacts(artifacts_path)

    # Make predictions
    return predict(
        df,
        model,
        artifacts["historical_means"],
        artifacts["encoders"]
    )

def generate_forecast(
    agencies: list[str],
    skus: list[str],
    start_date: str,
    end_date: str,
    model: object,
    historical_means: dict,
    encoders: dict) -> pd.DataFrame:
    """Generate forecasts for future dates.

    Args:
        agencies (list[str]): List of agency IDs
        skus (list[str]): List of SKU IDs
        start_date (str): Start date for the forecast
        end_date (str): End date for the forecast
        model (object): Trained model object
        historical_means (dict): Historical means from training
        encoders (dict): Categorical encoders from training

    Returns:
        pd.DataFrame: DataFrame with forecasted volumes
    """
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')

    # Create all combinations of agencies, skus, and dates
    rows =[]
    for date in date_range:
        for agency in agencies:
            for sku in skus:
                rows.append({
                    'agency': agency,
                    'sku': sku,
                    'date': date,
                    # Placeholder values for required columns
                    "volume": 0,
                    "avg_max_temp": 25.0,
                    "price_actual": 1000.0,
                    "discount_in_percent": 5.0
                })

    df = pd.DataFrame(rows)

    # Make predictions
    result = predict(df, model, historical_means, encoders)

    return result[['agency', 'sku', 'date', 'predicted_volume']]
