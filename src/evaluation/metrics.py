"""
@author: Joseph A.
Description: Metrics module for model evaluation.
"""
import numpy as np

def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """ Calculate Mean Absolute Error (MAE).

    Args:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted target values.

    Returns:
        float: Mean Absolute Error.
    """
    return np.mean(np.abs(y_true - y_pred))

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """ Calculate Root Mean Squared Error (RMSE).

    Args:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted target values.

    Returns:
        float: Root Mean Squared Error.
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """ Calculate Mean Absolute Percentage Error (MAPE).

    Args:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted target values.

    Returns:
        float: Mean Absolute Percentage Error.
    """
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def calculate_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """ Calculate all evaluation metrics.

    Args:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted target values.

    Returns:
        dict: Dictionary containing MAE, RMSE, and MAPE.
    """
    return {
        "mae": calculate_mae(y_true, y_pred),
        "rmse": calculate_rmse(y_true, y_pred),
        "mape": calculate_mape(y_true, y_pred),
    }

def calculate_baseline_metrics(
    y_true: np.ndarray,
    baseline_pred: np.ndarray
) -> dict:
    """ Calculate evaluation metrics for baseline predictions.

    Args:
        y_true (np.ndarray): True target values.
        baseline_pred (np.ndarray): Baseline predicted target values.

    Returns:
        dict: Dictionary with baseline metrics.

    """
    return calculate_all_metrics(y_true, baseline_pred)

def compare_to_baseline(model_metrics: dict, baseline_metrics: dict) -> dict:
    """ Compare model metrics to baseline metrics.

    Args:
        model_metrics (dict): Dictionary with model metrics.
        baseline_metrics (dict): Dictionary with baseline metrics.

    Returns:
        dict: Dictionary with comparison results.
    """
    improvements = {}
    for metric in model_metrics:
        baseline_val = baseline_metrics[metric]
        model_val = model_metrics[metric]

        if baseline_val != 0:
            improvement = (baseline_val - model_val) / baseline_val * 100
            improvements[f"{metric}_improvement_pct"] = improvement
        else:
            improvements[f"{metric}_improvement_pct"] = None

    return improvements

def print_evaluation_report(
    model_metrics: dict,
    baseline_metrics: dict | None = None
) -> None:
    """ Print evaluation report comparing model to baseline.

    Args:
        model_metrics (dict): Dictionary with model metrics.
        baseline_metrics (dict): Dictionary with baseline metrics.
    """
    print("=" * 50)
    print("Model Evaluation Report")
    print("=" * 50)

    print("\nModel Metrics:")
    print(f"    MAE     : {model_metrics['mae']:.2f}")
    print(f"    RMSE    : {model_metrics['rmse']:.2f}")
    print(f"    MAPE    : {model_metrics['mape']:.2f}%")

    if baseline_metrics:
        print("\nBaseline Metrics:")
        print(f"    MAE     : {baseline_metrics['mae']:.2f}")
        print(f"    RMSE    : {baseline_metrics['rmse']:.2f}")
        print(f"    MAPE    : {baseline_metrics['mape']:.2f}%")

        improvements = compare_to_baseline(model_metrics, baseline_metrics)

        print("\nImprovements over Baseline:")
        print(f"    MAE Improvement  : {improvements['mae_improvement_pct']:.1f}%")
        print(f"    RMSE Improvement : {improvements['rmse_improvement_pct']:.1f}%")
        print(f"    MAPE Improvement : {improvements['mape_improvement_pct']:.1f}%")

    print("=" * 50)
