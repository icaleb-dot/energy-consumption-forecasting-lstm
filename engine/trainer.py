"""Training and validation loops for PyTorch forecasting models."""
from typing import Any, Dict, Optional

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train the model for a single epoch.

    Args:
        model: Model to train.
        dataloader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to train on.

    Returns:
        Average training loss.
    """
    model.train()
    total_loss = 0.0
    total_samples = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

    return total_loss / total_samples if total_samples > 0 else 0.0


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on dataloader, returning both average loss and RMSE.

    Args:
        model: Model to evaluate.
        dataloader: Validation DataLoader.
        criterion: Loss function.
        device: Device to evaluate on.

    Returns:
        Dictionary with 'loss' and 'rmse'.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_se = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

            # Accumulate squared error for RMSE
            se = (outputs - labels).pow(2).sum().item()
            all_se.append(se)

    mean_loss = total_loss / total_samples if total_samples > 0 else 0.0
    rmse = np.sqrt(sum(all_se) / total_samples) if total_samples > 0 else 0.0

    return {"loss": mean_loss, "rmse": float(rmse)}


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    early_stopping: Optional[Any] = None,
    scheduler: Optional[Any] = None,
    log_to_mlflow: bool = True,
) -> Dict[str, list]:
    """Train and validate model over epochs with early stopping, scheduling, and MLflow logging.

    Args:
        model: PyTorch model.
        train_loader: DataLoader for training.
        val_loader: DataLoader for validation.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to run on.
        epochs: Maximum number of epochs.
        early_stopping: Optional EarlyStopping callback.
        scheduler: Optional learning rate scheduler.
        log_to_mlflow: Whether to log metrics per epoch to active MLflow run.

    Returns:
        Dictionary history containing 'train_loss', 'val_loss', 'val_rmse', and 'Learning_rate'.
    """
    history: Dict[str, list] = {
        "train_loss": [],
        "val_loss": [],
        "val_rmse": [],
        "Learning_rate": [],
    }

    for epoch in range(epochs):
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_eval = evaluate(model, val_loader, criterion, device)
        val_loss = val_eval["loss"]
        val_rmse = val_eval["rmse"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_rmse"].append(val_rmse)
        history["Learning_rate"].append(current_lr)

        print(
            f"Epoch {epoch + 1:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.5f} | "
            f"Val Loss: {val_loss:.5f} | "
            f"Val RMSE: {val_rmse:.5f} | "
            f"LR: {current_lr:.2e}"
        )

        # Log epoch metrics to MLflow if an active run exists
        if log_to_mlflow and mlflow.active_run() is not None:
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_rmse": val_rmse,
                    "lr": current_lr,
                },
                step=epoch + 1,
            )

        if scheduler is not None:
            scheduler.step(val_loss)

        if early_stopping is not None:
            early_stopping(model, val_loss)
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    if early_stopping is not None:
        early_stopping.load_best_model(model)

    return history
