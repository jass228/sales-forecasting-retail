"""
@author: Joseph A.
Description: Prediction script for sales forecasting.
"""
import pandas as pd
from src.configs.config import (
    DATE_COL, TARGET_COL, KEY_COLS,
    PREDICT_DATA_PATH, MODEL_DIR, OUTPUT_DIR,
    LAGS, ROLLING_WINDOWS, HISTORY_PATH
)
from src.data.loader import load_data
from src.data.preprocessing import encode_data, save_data
from src.features.engineering import build_features
from src.inference.predictor import load_model, predict

#pylint: disable=C0103:invalid-name

EXCLUDE_COLS = [TARGET_COL, DATE_COL]

def fill_missing_exog(df_new: pd.DataFrame, df_history: pd.DataFrame) -> pd.DataFrame:
    """Fill missing exogenous columns with last known values from history."""
    df_new = df_new.copy()

    # Get exogenous columns from history (exclude date, keys, target, _is_new)
    exog_cols = [c for c in df_history.columns
                if c not in KEY_COLS + [DATE_COL, TARGET_COL, "_is_new"]]

    # Get last known values per agency/sku
    df_last = (df_history.sort_values(DATE_COL)
            .groupby(KEY_COLS)[exog_cols]
            .last()
            .reset_index())

    # Fill missing columns in new data
    for col in exog_cols:
        if col not in df_new.columns:
            df_new = df_new.merge(df_last[KEY_COLS + [col]], on=KEY_COLS, how="left")

    return df_new

def main():
    """Main prediction pipeline."""
    # 1. Load historical data and new data
    print("[1/5] Loading data...")
    df_history = load_data(HISTORY_PATH)
    df_history[DATE_COL] = pd.to_datetime(df_history[DATE_COL])

    df_new = load_data(PREDICT_DATA_PATH)
    df_new[DATE_COL] = pd.to_datetime(df_new[DATE_COL])

    # Mark new data rows for filtering later
    df_history["_is_new"] = False
    df_new["_is_new"] = True

    # Add placeholder volume for new data (needed for feature engineering)
    if TARGET_COL not in df_new.columns:
        df_new[TARGET_COL] = 0

    # Fill missing exogenous columns with last known values
    df_new = fill_missing_exog(df_new, df_history)

    # 2. Concatenate history and new data
    print("[2/5] Preparing data with history...")
    df_combined = pd.concat([df_history, df_new], ignore_index=True)
    df_combined = df_combined.sort_values(KEY_COLS + [DATE_COL]).reset_index(drop=True)

    # 3. Feature engineering (lags computed from historical volume)
    print("[3/5] Computing features...")
    df_fe = build_features(df_combined, LAGS, ROLLING_WINDOWS)

    # Filter only new rows (the ones we want to predict)
    df_predict = df_fe[df_fe["_is_new"]].copy()

    # Keep only rows with valid lag features (first horizon has real lags)
    lag_cols = [f"{TARGET_COL}_lag_{lag}" for lag in LAGS]
    rolling_cols = [f"{TARGET_COL}_rolling_mean_{w}" for w in ROLLING_WINDOWS]
    feature_lag_cols = lag_cols + rolling_cols
    df_predict = df_predict.dropna(subset=feature_lag_cols)

    df_predict = df_predict.drop(columns=["_is_new", TARGET_COL])

    if len(df_predict) == 0:
        raise ValueError("No rows to predict after feature engineering.")

    # 4. Encode data
    print("[4/5] Encoding data...")
    encoder = load_model(MODEL_DIR / "encoder.pkl")
    df_encoded, _ = encode_data(df_predict, encoder)

    feature_cols = [c for c in df_encoded.columns if c not in EXCLUDE_COLS]

    # 5. Load model and predict
    print("[5/5] Generating predictions...")
    model = load_model(MODEL_DIR / "best_model.pkl")
    df_pred = predict(model, df_encoded, feature_cols)

    # Clip negative predictions to 0
    df_pred["prediction"] = df_pred["prediction"].clip(lower=0)

    # Decode agency/sku back to original names
    df_pred[KEY_COLS] = encoder.inverse_transform(df_pred[KEY_COLS])

    # Prepare output
    output_cols = [DATE_COL] + KEY_COLS + ["prediction"]
    available_cols = [c for c in output_cols if c in df_pred.columns]
    df_output = df_pred[available_cols].copy()

    # Save predictions
    output_path = OUTPUT_DIR / "predictions.csv"
    save_data(df_output, output_path)

    print("Predictions saved!")

if __name__ == "__main__":
    main()
