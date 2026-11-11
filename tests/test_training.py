"""Tests for training engine, loops, metrics, and callbacks."""
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from callbacks.early_stopping import EarlyStopping
from datasets.uci_power import UCIPowerDataset
from engine.evaluator import evaluate_model, evaluate_naive_baseline
from engine.metrics import compute_metrics, compute_rmse, compute_safe_mape
from engine.trainer import evaluate, fit, train_one_epoch
from models.lstm import LSTMForecaster


def _create_synthetic_loaders(n_samples: int = 150, seq_len: int = 24, batch_size: int = 16):
    """Generate synthetic sequence loaders for testing."""
    # Sine wave time series
    series = np.sin(np.linspace(0, 10 * np.pi, n_samples)).astype(np.float32).reshape(-1, 1)

    # Windows
    x_list, y_list = [], []
    for i in range(len(series) - seq_len):
        x_list.append(series[i : i + seq_len])
        y_list.append(series[i + seq_len])

    x_arr = np.asarray(x_list, dtype=np.float32)
    y_arr = np.asarray(y_list, dtype=np.float32)

    ds = UCIPowerDataset(x_arr, y_arr)
    train_loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def test_training_reduces_loss():
    """Verify that multiple training epochs decrease model training loss."""
    model = LSTMForecaster(input_size=1, hidden_size=16, output_size=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    device = torch.device("cpu")

    train_loader, _ = _create_synthetic_loaders()

    # Initial loss
    initial_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

    # Train for several epochs
    for _ in range(8):
        final_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

    assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"


def test_fit_history_keys_and_lengths():
    """Verify fit() returns complete history dictionary with all required keys."""
    model = LSTMForecaster(input_size=1, hidden_size=16, output_size=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    device = torch.device("cpu")

    train_loader, val_loader = _create_synthetic_loaders()
    epochs = 3

    history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        log_to_mlflow=False,
    )

    for key in ["train_loss", "val_loss", "val_rmse", "Learning_rate"]:
        assert key in history, f"Missing key {key} in history"
        assert len(history[key]) == epochs


def test_early_stopping_activation_and_recovery():
    """Verify that early stopping activates after patience is exceeded and stores best state."""
    early_stopping = EarlyStopping(patience=2, min_delta=0.0)
    model = LSTMForecaster(input_size=1, hidden_size=16, output_size=1)

    early_stopping(model, 1.0)
    assert not early_stopping.early_stop
    assert early_stopping.best_loss == 1.0

    early_stopping(model, 1.1)
    assert not early_stopping.early_stop

    early_stopping(model, 1.2)
    assert early_stopping.early_stop

    # Test loading best model
    early_stopping.load_best_model(model)
    assert early_stopping.best_model_state is not None


def test_naive_baseline_and_metrics():
    """Verify that the naive baseline and evaluation metrics return valid positive numbers."""
    train_loader, _ = _create_synthetic_loaders()

    metrics = evaluate_naive_baseline(train_loader)
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "mape" in metrics
    assert metrics["rmse"] >= 0.0
    assert metrics["mae"] >= 0.0
    assert metrics["mape"] >= 0.0


def test_safe_mape_zero_handling():
    """Verify that compute_safe_mape does not divide by zero or yield inf/nan."""
    y_true = np.array([0.0, 0.0, 1.0, 2.0], dtype=np.float32)
    y_pred = np.array([0.1, 0.2, 1.1, 1.9], dtype=np.float32)

    mape = compute_safe_mape(y_true, y_pred, eps=1e-5)
    assert not np.isnan(mape)
    assert not np.isinf(mape)
    assert mape > 0
