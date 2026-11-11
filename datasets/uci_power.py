"""UCI Household Electric Power Consumption PyTorch Dataset."""
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class UCIPowerDataset(Dataset):
    """PyTorch Dataset for windowed UCI Household Power Consumption time series."""

    def __init__(
        self,
        x: Union[np.ndarray, torch.Tensor, str, Path],
        y: Union[np.ndarray, torch.Tensor, None] = None,
    ) -> None:
        """Initialize the dataset from numpy arrays, tensors, or an npz file path.

        Args:
            x: Feature array of shape (N, input_window, features) or path to an .npz file
               containing 'x' and 'y' arrays.
            y: Target array of shape (N, horizon) or (N, 1). If x is a path, y is ignored.
        """
        if isinstance(x, (str, Path)):
            path = Path(x)
            if not path.exists():
                raise FileNotFoundError(f"Processed dataset file not found at: {path}")
            data = np.load(path)
            x_arr = data["x"]
            y_arr = data["y"]
        else:
            if y is None:
                raise ValueError("Target array 'y' must be provided when 'x' is an array/tensor.")
            x_arr = x
            y_arr = y

        if isinstance(x_arr, np.ndarray):
            self.x = torch.from_numpy(x_arr).float()
        else:
            self.x = x_arr.float()

        if isinstance(y_arr, np.ndarray):
            self.y = torch.from_numpy(y_arr).float()
        else:
            self.y = y_arr.float()

        # Ensure correct dimension shapes: [N, seq_len, features] and [N, horizon]
        if self.x.ndim == 2:
            self.x = self.x.unsqueeze(-1)
        if self.y.ndim == 1:
            self.y = self.y.unsqueeze(-1)

        if len(self.x) != len(self.y):
            raise ValueError(
                f"Features count ({len(self.x)}) and targets count ({len(self.y)}) mismatch."
            )

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


def get_uci_power_loaders(
    data_dir: Union[str, Path] = "data/processed",
    batch_size: int = 64,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load train, val, and test data loaders from processed npz files.

    Args:
        data_dir: Directory containing train.npz, val.npz, and test.npz.
        batch_size: Batch size for the data loaders.
        num_workers: Worker processes for data loading.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    processed_dir = Path(data_dir)
    train_path = processed_dir / "train.npz"
    val_path = processed_dir / "val.npz"
    test_path = processed_dir / "test.npz"

    for path in (train_path, val_path, test_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing processed data split at '{path}'. Run the preprocessing pipeline first."
            )

    train_ds = UCIPowerDataset(train_path)
    val_ds = UCIPowerDataset(val_path)
    test_ds = UCIPowerDataset(test_path)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,  # Sequential/time order kept or optionally batched
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader

