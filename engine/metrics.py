"""Evaluation metrics for time series forecasting."""
from typing import Dict, Union

import numpy as np
import torch


def _to_numpy(data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Convert torch.Tensor or array-like to a flat/squeezed numpy array."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def compute_rmse(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
) -> float:
    """Compute Root Mean Squared Error (RMSE).

    Args:
        y_true: Ground truth target values.
        y_pred: Predicted values.

    Returns:
        Scalar RMSE value.
    """
    y_t = _to_numpy(y_true).astype(np.float64)
    y_p = _to_numpy(y_pred).astype(np.float64)
    mse = np.mean((y_t - y_p) ** 2)
    return float(np.sqrt(mse))


def compute_mae(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
) -> float:
    """Compute Mean Absolute Error (MAE).

    Args:
        y_true: Ground truth target values.
        y_pred: Predicted values.

    Returns:
        Scalar MAE value.
    """
    y_t = _to_numpy(y_true).astype(np.float64)
    y_p = _to_numpy(y_pred).astype(np.float64)
    return float(np.mean(np.abs(y_t - y_p)))


def compute_safe_mape(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-5,
) -> float:
    """Compute Safe Mean Absolute Percentage Error (MAPE).

    Uses a threshold epsilon in the denominator to avoid division by zero or extreme
    spikes when actual power consumption values are zero or near-zero.

    Args:
        y_true: Ground truth target values.
        y_pred: Predicted values.
        eps: Small positive constant added or thresholded to prevent division by zero.

    Returns:
        Safe MAPE percentage value (0 to 100+).
    """
    y_t = _to_numpy(y_true).astype(np.float64)
    y_p = _to_numpy(y_pred).astype(np.float64)
    denominator = np.maximum(np.abs(y_t), eps)
    absolute_percentage_errors = np.abs((y_t - y_p) / denominator) * 100.0
    return float(np.mean(absolute_percentage_errors))


def compute_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-5,
) -> Dict[str, float]:
    """Compute RMSE, MAE, and safe MAPE in a single call.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        eps: Epsilon for safe MAPE denominator.

    Returns:
        Dictionary with 'rmse', 'mae', and 'mape'.
    """
    return {
        "rmse": compute_rmse(y_true, y_pred),
        "mae": compute_mae(y_true, y_pred),
        "mape": compute_safe_mape(y_true, y_pred, eps=eps),
    }

