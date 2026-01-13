"""
@author: Joseph A.
Description: Trainer module for model training and evaluation.
"""
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

# pylint: disable=C0103:invalid-name

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame | None = None,
    y_test: pd.Series | None = None,
    params: dict | None = None
) -> lgb.LGBMRegressor:
    """ Train a LightGBM model.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target values.
        X_test (pd.DataFrame | None, optional): Test features. Defaults to None.
        y_test (pd.Series | None, optional): Test target values. Defaults to None.
        params (dict | None, optional): Model parameters. Defaults to None.

    Returns:
        lgb.LGBMRegressor: Trained LightGBM model.
    """
    default_params = {
        'n_estimators': 500,        # number of boosting rounds
        'learning_rate': 0.05,      # step size shrinkage
        'num_leaves': 31,           # maximum number of leaves in one tree
        'max_depth': -1,            # no limit on tree depth
        'min_child_samples': 20,    # minimum number of data in one leaf
        'random_state': 42,         # for reproducibility
        'n_jobs': -1,               # use all available cores
        'verbosity': -1,            # suppress warnings
    }

    if params:
        default_params.update(params)

    model = lgb.LGBMRegressor(**default_params)

    if X_test is not None and y_test is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
    else:
        model.fit(X_train, y_train)

    return model

def cross_validate_model(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    params: dict | None = None
) -> dict:
    """ Cross-validate the model using TimeSeriesSplit.

    Args:
        X (pd.DataFrame): Training features.
        y (pd.Series): Training target values.
        n_splits (int, optional): Number of splits for cross-validation. Defaults to 5.
        params (dict | None, optional): Model parameters. Defaults to None.

    Returns:
        dict: Cross-validation scores.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    scores = {
        "fold": [],
        "train_size": [],
        "test_size": [],
        "mae": [],
        "rmse": [],
    }

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = train_model(X_train, y_train, X_test, y_test, params)
        y_pred = model.predict(X_test)

        mae = abs(y_test - y_pred).mean()
        rmse = ((y_test - y_pred) ** 2).mean() ** 0.5

        scores["fold"].append(fold + 1)
        scores["train_size"].append(len(X_train))
        scores["test_size"].append(len(X_test))
        scores["mae"].append(mae)
        scores["rmse"].append(rmse)

    return scores

def get_feature_importance(
    model: lgb.LGBMRegressor,
    feature_names: list[str]
) -> pd.DataFrame:
    """ Get feature importance from the trained model.

    Args:
        model (lgb.LGBMRegressor): Trained LightGBM model.
        feature_names (list[str]): List of feature names.

    Returns:
        pd.DataFrame: DataFrame with feature names and their importance scores.
    """
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    ).reset_index(drop=True)

    return importance_df
