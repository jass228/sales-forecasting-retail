"""
@author: Joseph A.
Description:Main prediction script.
"""

import argparse
from pathlib import Path
import pandas as pd
from src.data.loader import load_data
from src.inference.predictor import (
    load_model,
    load_artifacts,
    predict,
    generate_forecast,
)

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate sales predictions")

    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to the input data CSV file",
    )
    parser.add_argument(
        "--forecast",
        action="store_true",
        help="Generate forecast for future dates",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Forecast start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Forecast end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/model.pkl",
        help="Path to the trained model",
    )
    parser.add_argument(
        "--artifacts",
        type=str,
        default="models/artifacts.pkl",
        help="Path to training artifacts",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/predictions.csv",
        help="Path to save predictions",
    )

    return parser.parse_args()


def main() -> None:
    """Main prediction pipeline."""
    args = parse_args()

    # =========================================================================
    # 1. LOAD MODEL AND ARTIFACTS
    # =========================================================================
    print("=" * 60)
    print("1. LOADING MODEL AND ARTIFACTS")
    print("=" * 60)

    model = load_model(args.model)
    artifacts = load_artifacts(args.artifacts)

    historical_means = artifacts["historical_means"]
    encoders = artifacts["encoders"]

    print(f"Model loaded from: {args.model}")
    print(f"Artifacts loaded from: {args.artifacts}")

    # =========================================================================
    # 2. GENERATE PREDICTIONS
    # =========================================================================
    print("\n" + "=" * 60)
    print("2. GENERATING PREDICTIONS")
    print("=" * 60)

    if args.forecast:
        # Forecast mode: generate predictions for future dates
        if not args.start_date or not args.end_date:
            raise ValueError("--start-date and --end-date required for forecast mode")

        # Get all agencies and SKUs from encoders
        agencies = list(encoders["agency"].keys())
        skus = list(encoders["sku"].keys())

        print(f"Forecast mode: {args.start_date} → {args.end_date}")
        print(f"Agencies: {len(agencies)}")
        print(f"SKUs: {len(skus)}")

        result = generate_forecast(
            agencies=agencies,
            skus=skus,
            start_date=args.start_date,
            end_date=args.end_date,
            model=model,
            historical_means=historical_means,
            encoders=encoders,
        )

    else:
        # Prediction mode: predict on existing data
        if not args.data:
            raise ValueError("--data required for prediction mode")

        df = load_data(args.data)
        print(f"Input data: {len(df):,} rows")

        result = predict(df, model, historical_means, encoders)
        result = result[["agency", "sku", "date", "volume", "predicted_volume"]]

    print(f"Predictions generated: {len(result):,} rows")

    # =========================================================================
    # 3. SAVE PREDICTIONS
    # =========================================================================
    print("\n" + "=" * 60)
    print("3. SAVING PREDICTIONS")
    print("=" * 60)

    # Create output directory if needed
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    result.to_csv(args.output, index=False)
    print(f"Predictions saved to: {args.output}")

    # =========================================================================
    # 4. SUMMARY
    # =========================================================================
    print("\n" + "=" * 60)
    print("4. PREDICTION SUMMARY")
    print("=" * 60)

    print("\nPredicted volume statistics:")
    print(f"  Min:  {result['predicted_volume'].min():.2f}")
    print(f"  Max:  {result['predicted_volume'].max():.2f}")
    print(f"  Mean: {result['predicted_volume'].mean():.2f}")
    print(f"  Std:  {result['predicted_volume'].std():.2f}")

    print("\nSample predictions:")
    print(result.head(10).to_string(index=False))

    print("\n" + "=" * 60)
    print("PREDICTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
