"""
@author: Joseph A.
Description: Main training script.
"""
import pandas as pd
from src.configs.config import (
    LAGS, ROLLING_WINDOWS, DATE_COL, TARGET_COL,
    TRAIN_DATA_PATH, MODEL_DIR, PROCESSED_DIR
)
from src.data.loader import load_data
from src.data.preprocessing import preprocess_data, encode_data, split_train_test, save_data
from src.features.engineering import build_features
from src.model.training import train_model, save_model
from src.model.evaluation import evaluate_model

#pylint: disable=C0103:invalid-name

EXCLUDE_COLS = [TARGET_COL, DATE_COL]

def main():
    """Main training pipeline."
    """
    # 1. Load & Preprocess Data
    df_raw = load_data(TRAIN_DATA_PATH)
    df_prepared = preprocess_data(df_raw)

    # Save preprocessed data
    save_data(df_prepared, PROCESSED_DIR / "df_prepared.csv")

    # Save recent history (for inference - needed to compute lags)
    max_lag = max(LAGS + ROLLING_WINDOWS)
    max_date = df_prepared[DATE_COL].max()
    cutoff_date = max_date - pd.DateOffset(months=max_lag)
    df_history = df_prepared[df_prepared[DATE_COL] > cutoff_date].copy()
    save_data(df_history, MODEL_DIR / "history.csv")

    # 2. Feature Engineering
    df_fe = build_features(df_prepared, LAGS, ROLLING_WINDOWS)
    df_fe = df_fe.dropna()
    train_df, test_df = split_train_test(df_fe, 4)

    # Save train and test data
    save_data(train_df, PROCESSED_DIR / "train.csv")
    save_data(test_df, PROCESSED_DIR / "test.csv")

    # 3. Encode data
    train_df_enc, encoder = encode_data(train_df)
    test_df_enc, _ = encode_data(test_df, encoder)

    # 4. Define
    feature_cols = [c for c in train_df.columns if c not in EXCLUDE_COLS]

    X_train = train_df_enc[feature_cols].copy()
    y_train = train_df_enc[TARGET_COL]

    X_test = test_df_enc[feature_cols].copy()
    y_test = test_df_enc[TARGET_COL]

    # 5. Model training and comparison
    MODEL_CANDIDATES = ["random_forest", "xgboost", "lightgbm"]

    results = []
    models = {}

    for model_name in MODEL_CANDIDATES:
        print(f"Training model: {model_name}")
        model = train_model(model_name)
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)

        results.append({
            'model': model_name,
            'mae': metrics['mae'],
            'rmse': metrics['rmse'],
            'mape': metrics['mape']
        })
        models[model_name] = model

    # Create results DataFrame
    results_df = pd.DataFrame(results)
    save_data(results_df, MODEL_DIR / "model_comparison.csv")

    # Select and save best model
    best_model_name = results_df.loc[results_df['mae'].idxmin(), 'model']
    best_model = models[best_model_name]

    save_model(best_model, MODEL_DIR / "best_model.pkl")
    save_model(encoder, MODEL_DIR / "encoder.pkl")
    print("Model and encoder saved !")

if __name__ == "__main__":
    main()
