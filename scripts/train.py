"""
@author: Joseph A.
Description: Main training script.
"""
import argparse
from pathlib import Path
import pandas as pd
from src.data.loader import load_data, get_data_info, split_train_test
from src.features.engineering import prepare_features, get_feature_columns, get_target_column
from src.models.trainer import train_model, get_feature_importance
from src.evaluation.metrics import (
    calculate_all_metrics,
    calculate_baseline_metrics,
    print_evaluation_report
)
from src.inference.predictor import save_model, save_artifacts

def parse_args() -> argparse.Namespace:
    """ Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train a sales forecasting model.")

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the training data CSV file."
    )
    parser.add_argument(
        "--test-date",
        type=str,
        default="2017-07-01",
        help="Start date for the test set (YYYY-MM-DD)."
    )
    parser.add_argument(
        "--model-output", 
        type=str,
        default="models/model.pkl",
        help="Path to save the trained model."
    )
    parser.add_argument(
        "--artifacts-output", 
        type=str,
        default="models/artifacts.pkl",
        help="Path to save training artifacts."
    )

    return parser.parse_args()

def main() -> None:
    """Main training pipeline.
    """
    args = parse_args()

    # =========================================================================
    # 1. LOAD DATA
    # =========================================================================
    print("=" * 60)
    print("1. LOADING DATA")
    print("=" * 60)

    df = load_data(args.data)
    info = get_data_info(df)

    print(f"Rows: {info['n_rows']:,}")
    print(f"Columns: {info['n_columns']}")
    print(f"Date range: {info['date_min']} → {info['date_max']}")
    print(f"Agencies: {info['n_agencies']}")
    print(f"SKUs: {info['n_skus']}")

    # =========================================================================
    # 2. SPLIT TRAIN/TEST
    # =========================================================================
    print("\n" + "=" * 60)
    print("2. TRAIN/TEST SPLIT")
    print("=" * 60)

    train_df, test_df = split_train_test(df, args.test_date)

    print(f"Train: {len(train_df):,} rows ({train_df['date'].min()} → {train_df['date'].max()})")
    print(f"Test:  {len(test_df):,} rows ({test_df['date'].min()} → {test_df['date'].max()})")

    # =========================================================================
    # 3. FEATURE ENGINEERING
    # =========================================================================
    print("\n" + "=" * 60)
    print("3. FEATURE ENGINEERING")
    print("=" * 60)

    # Prepare features on train set (creates historical_means and encoders)
    train_prepared, historical_means, encoders = prepare_features(train_df)

    # Prepare features on test set (uses saved historical_means and encoders)
    test_prepared, _, _ = prepare_features(test_df, historical_means, encoders)

    feature_cols = get_feature_columns()
    target_col = get_target_column()

    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"Target: {target_col}")

    # Prepare X and y
    X_train = train_prepared[feature_cols]
    y_train = train_prepared[target_col]

    X_test = test_prepared[feature_cols]
    y_test = test_prepared[target_col]

    # =========================================================================
    # 4. TRAIN MODEL
    # =========================================================================
    print("\n" + "=" * 60)
    print("4. TRAINING MODEL")
    print("=" * 60)

    model = train_model(X_train, y_train, X_test, y_test)

    print(f"Model trained with {model.best_iteration_} iterations")

    # =========================================================================
    # 5. EVALUATE MODEL
    # =========================================================================
    print("\n" + "=" * 60)
    print("5. EVALUATION")
    print("=" * 60)

    # Model predictions
    y_pred = model.predict(X_test)
    model_metrics = calculate_all_metrics(y_test.values, y_pred)

    # Baseline: historical mean by agency/sku/month
    baseline_pred = test_prepared["mean_volume_agency_sku_month"].values
    baseline_metrics = calculate_baseline_metrics(y_test.values, baseline_pred)

    print_evaluation_report(model_metrics, baseline_metrics)

    # =========================================================================
    # 6. FEATURE IMPORTANCE
    # =========================================================================
    print("\n" + "=" * 60)
    print("6. FEATURE IMPORTANCE")
    print("=" * 60)

    importance_df = get_feature_importance(model, feature_cols)
    print(importance_df.to_string(index=False))

    # =========================================================================
    # 7. SAVE MODEL AND ARTIFACTS
    # =========================================================================
    print("\n" + "=" * 60)
    print("7. SAVING MODEL AND ARTIFACTS")
    print("=" * 60)

    # Create output directories if needed
    Path(args.model_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.artifacts_output).parent.mkdir(parents=True, exist_ok=True)

    save_model(model, args.model_output)
    save_artifacts(historical_means, encoders, args.artifacts_output)

    print(f"Model saved to: {args.model_output}")
    print(f"Artifacts saved to: {args.artifacts_output}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
