"""Tests for UCI Household Power Consumption PyTorch Dataset."""
import numpy as np
import pytest
import torch

from datasets.factory import create_dataset
from datasets.uci_power import UCIPowerDataset


def test_uci_power_dataset_shapes_and_types():
    """Verify UCIPowerDataset generates tensors with correct shapes and float32 dtype."""
    num_samples = 50
    seq_len = 24
    features = 1
    horizon = 1

    x_data = np.random.randn(num_samples, seq_len, features).astype(np.float32)
    y_data = np.random.randn(num_samples, horizon).astype(np.float32)

    dataset = UCIPowerDataset(x_data, y_data)
    assert len(dataset) == num_samples

    x_tensor, y_tensor = dataset[0]

    assert isinstance(x_tensor, torch.Tensor)
    assert isinstance(y_tensor, torch.Tensor)
    assert x_tensor.shape == (seq_len, features)
    assert y_tensor.shape == (horizon,)
    assert x_tensor.dtype == torch.float32
    assert y_tensor.dtype == torch.float32


def test_uci_power_dataset_2d_input_expansion():
    """Verify that 2D input (num_samples, seq_len) gets expanded to (num_samples, seq_len, 1)."""
    num_samples = 20
    seq_len = 24
    x_data = np.random.randn(num_samples, seq_len).astype(np.float32)
    y_data = np.random.randn(num_samples).astype(np.float32)

    dataset = UCIPowerDataset(x_data, y_data)
    x_tensor, y_tensor = dataset[0]

    assert x_tensor.shape == (seq_len, 1)
    assert y_tensor.shape == (1,)


def test_uci_power_dataset_factory():
    """Verify dataset instantiation via datasets.factory.create_dataset."""
    num_samples = 15
    seq_len = 24
    x_data = np.random.randn(num_samples, seq_len, 1).astype(np.float32)
    y_data = np.random.randn(num_samples, 1).astype(np.float32)

    ds = create_dataset("uci_power", x=x_data, y=y_data)
    assert isinstance(ds, UCIPowerDataset)
    assert len(ds) == num_samples


def test_uci_power_dataset_mismatched_length_raises():
    """Verify ValueError is raised when x and y sample counts differ."""
    x_data = np.random.randn(10, 24, 1).astype(np.float32)
    y_data = np.random.randn(8, 1).astype(np.float32)

    with pytest.raises(ValueError, match="mismatch"):
        UCIPowerDataset(x_data, y_data)
