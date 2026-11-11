"""Plotting utilities for time series forecasting loss curves and predictions."""
from pathlib import Path
from typing import Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np


def plot_losses(
    history: Dict[str, list],
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """Plot training and validation loss curves and learning rate trajectory.

    Args:
        history: Dictionary containing 'train_loss', 'val_loss', and optionally 'Learning_rate'.
        save_path: Optional file path to save the generated figure.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.plot(epochs, history["train_loss"], label="Train Loss (MSE)", color="tab:blue", lw=2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss (MSE)", color="tab:orange", lw=2)
    if "val_rmse" in history and len(history["val_rmse"]) == len(epochs):
        ax1.plot(epochs, history["val_rmse"], label="Val RMSE", color="tab:green", linestyle="--", lw=1.5)
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    plt.title("Training and Validation Performance")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_forecast(
    targets: np.ndarray,
    predictions: np.ndarray,
    num_points: int = 200,
    unit: str = "kW",
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """Plot ground truth versus forecasted values.

    Args:
        targets: True series values array.
        predictions: Model predictions array.
        num_points: Maximum points to display in plot.
        unit: Measurement unit label.
        save_path: Optional file path to save figure.
    """
    targets = np.asarray(targets)[:num_points]
    predictions = np.asarray(predictions)[:num_points]

    plt.figure(figsize=(13, 5))
    plt.plot(targets, label="Ground Truth", color="tab:blue", lw=1.5)
    plt.plot(predictions, label="Prediction", color="tab:orange", linestyle="--", lw=1.5)

    plt.xlabel("Time (Hours)")
    plt.ylabel(f"Global Active Power ({unit})")
    plt.title(f"Forecasting Comparison (First {len(targets)} hours)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()
