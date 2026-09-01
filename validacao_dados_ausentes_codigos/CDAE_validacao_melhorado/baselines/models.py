"""Baseline model training and masked-target prediction."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import KNNImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

from baselines.config import HGB_PARAMS, RIDGE_ALPHA_GRID


def fit_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[Ridge, Dict[str, object]]:
    """Select Ridge alpha on validation only, then refit on train."""
    best_alpha = 1.0
    best_mae = float("inf")
    for alpha in RIDGE_ALPHA_GRID:
        model = Ridge(alpha=float(alpha))
        model.fit(x_train, y_train)
        pred_val = model.predict(x_val)
        mae_val = float(mean_absolute_error(y_val, pred_val))
        if mae_val < best_mae:
            best_mae = mae_val
            best_alpha = float(alpha)

    final_model = Ridge(alpha=best_alpha)
    final_model.fit(x_train, y_train)
    config = {
        "model_type": "ridge",
        "alpha_grid": list(RIDGE_ALPHA_GRID),
        "selected_alpha": best_alpha,
        "val_mae_for_selection": best_mae,
    }
    return final_model, config


def fit_hgb(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
) -> Tuple[HistGradientBoostingRegressor, Dict[str, object]]:
    """Train HistGradientBoosting with fixed hyperparameters (train-only fit)."""
    model = HistGradientBoostingRegressor(
        max_depth=int(HGB_PARAMS["max_depth"]),
        learning_rate=float(HGB_PARAMS["learning_rate"]),
        max_iter=int(HGB_PARAMS["max_iter"]),
        min_samples_leaf=int(HGB_PARAMS["min_samples_leaf"]),
        l2_regularization=float(HGB_PARAMS["l2_regularization"]),
        early_stopping=False,
        random_state=int(seed),
    )
    model.fit(x_train, y_train)
    val_pred = model.predict(x_val)
    val_mae = float(mean_absolute_error(y_val, val_pred))
    config = {
        "model_type": "hgb",
        "hyperparameters": dict(HGB_PARAMS),
        "early_stopping": False,
        "val_mae_audit": val_mae,
        "n_iter": int(getattr(model, "n_iter_", HGB_PARAMS["max_iter"])),
    }
    return model, config


def fit_knn_imputer(
    x_train: np.ndarray,
    y_train: np.ndarray,
    k_neighbors: int,
) -> Tuple[KNNImputer, Dict[str, object]]:
    """Fit multivariate KNN-Imputer on scaled [X, y]."""
    xy_train = np.hstack([x_train, y_train.reshape(-1, 1)])
    imputer = KNNImputer(n_neighbors=int(k_neighbors))
    imputer.fit(xy_train)
    config = {
        "model_type": "knn",
        "k_neighbors": int(k_neighbors),
        "input_shape": list(xy_train.shape),
    }
    return imputer, config


def predict_supervised(
    model: object,
    x_test_scaled: np.ndarray,
    scaler_y: object,
) -> np.ndarray:
    """Predict target from auxiliaries only (soft-sensor path)."""
    y_pred_scaled = np.asarray(model.predict(x_test_scaled), dtype=np.float64)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(-1)
    return y_pred.astype(np.float64)


def predict_knn_masked(
    imputer: KNNImputer,
    x_test_scaled: np.ndarray,
    y_test_scaled: np.ndarray,
    miss_mask: np.ndarray,
    scaler_y: object,
) -> np.ndarray:
    """Impute masked target values with KNN on [X, y] matrix."""
    y_with_nan = y_test_scaled.copy()
    y_with_nan[miss_mask] = np.nan
    xy_test = np.hstack([x_test_scaled, y_with_nan.reshape(-1, 1)])
    xy_imputed = imputer.transform(xy_test)
    y_pred_scaled = xy_imputed[:, -1]
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(-1)
    return y_pred.astype(np.float64)
