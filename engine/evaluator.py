"""Evaluation utilities and naive baseline comparison."""
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from engine.metrics import compute_metrics


def predict_loader(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run model inference over a DataLoader and collect predictions and targets.

    Args:
        model: Trained PyTorch forecasting model.
        dataloader: DataLoader yielding (inputs, targets).
        device: Device to run evaluation on.

    Returns:
        Tuple (all_predictions, all_targets) as 2D numpy arrays.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.inference_mode():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_preds.append(outputs.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())

    preds_arr = np.concatenate(all_preds, axis=0)
    targets_arr = np.concatenate(all_targets, axis=0)
    return preds_arr, targets_arr


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    scaler: Optional[Any] = None,
) -> Dict[str, float]:
    """Evaluate a PyTorch model and return RMSE, MAE, and safe MAPE.

    Args:
        model: Model instance.
        dataloader: Evaluation DataLoader.
        device: Evaluation device.
        scaler: Optional fitted scikit-learn scaler to inverse-transform before evaluation.

    Returns:
        Dictionary with 'rmse', 'mae', 'mape'.
    """
    preds, targets = predict_loader(model, dataloader, device)

    if scaler is not None and hasattr(scaler, "inverse_transform"):
        # Reshape to 2D for scaler
        orig_preds_shape = preds.shape
        orig_targets_shape = targets.shape
        preds = scaler.inverse_transform(preds.reshape(-1, 1)).reshape(orig_preds_shape)
        targets = scaler.inverse_transform(targets.reshape(-1, 1)).reshape(orig_targets_shape)

    return compute_metrics(y_true=targets, y_pred=preds)


def evaluate_naive_baseline(
    dataloader: DataLoader,
    scaler: Optional[Any] = None,
) -> Dict[str, float]:
    """Evaluate a naive persistence baseline (repeats the last observed input value).

    For an input sequence x of shape (batch, seq_len, 1) and target y of shape (batch, horizon),
    the naive forecast repeats the last time step x[:, -1, 0] across all horizon steps.

    Args:
        dataloader: DataLoader providing (inputs, targets).
        scaler: Optional fitted scaler to unscale predictions and targets.

    Returns:
        Dictionary of baseline metrics: 'rmse', 'mae', 'mape'.
    """
    all_naive_preds = []
    all_targets = []

    for inputs, targets in dataloader:
        # inputs: [batch, seq_len, 1], targets: [batch, horizon] or [batch, 1]
        last_known = inputs[:, -1, :].numpy()  # [batch, 1]
        horizon = targets.shape[-1]
        # Repeat last known value across horizon if multi-step
        naive_forecast = np.repeat(last_known, horizon, axis=-1)
        all_naive_preds.append(naive_forecast)
        all_targets.append(targets.numpy())

    preds = np.concatenate(all_naive_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    if scaler is not None and hasattr(scaler, "inverse_transform"):
        orig_preds_shape = preds.shape
        orig_targets_shape = targets.shape
        preds = scaler.inverse_transform(preds.reshape(-1, 1)).reshape(orig_preds_shape)
        targets = scaler.inverse_transform(targets.reshape(-1, 1)).reshape(orig_targets_shape)

    return compute_metrics(y_true=targets, y_pred=preds)


def compare_model_with_baseline(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    scaler: Optional[Any] = None,
) -> Dict[str, Any]:
    """Compare model against naive baseline and calculate improvement percentages.

    Returns:
        Dictionary containing model metrics, baseline metrics, and percentage improvements.
    """
    model_metrics = evaluate_model(model, dataloader, device, scaler=scaler)
    baseline_metrics = evaluate_naive_baseline(dataloader, scaler=scaler)

    rmse_improvement = (
        (baseline_metrics["rmse"] - model_metrics["rmse"]) / baseline_metrics["rmse"]
    ) * 100.0
    mae_improvement = (
        (baseline_metrics["mae"] - model_metrics["mae"]) / baseline_metrics["mae"]
    ) * 100.0

    return {
        "model": model_metrics,
        "naive_baseline": baseline_metrics,
        "improvements": {
            "rmse_reduction_pct": float(rmse_improvement),
            "mae_reduction_pct": float(mae_improvement),
        },
    }

