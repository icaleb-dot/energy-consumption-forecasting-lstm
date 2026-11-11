"""Dataset factory for time series forecasting datasets."""
from pathlib import Path
from typing import Any, Dict, Tuple, Union

from torch.utils.data import DataLoader, Dataset

from datasets.uci_power import UCIPowerDataset, get_uci_power_loaders

DATASET_REGISTRY = {
    "uci_power": UCIPowerDataset,
    "household_power": UCIPowerDataset,
}


def create_dataset(dataset_name: str, **kwargs: Any) -> Dataset:
    """Create a dataset instance by registered name.

    Args:
        dataset_name: Registered dataset name.
        **kwargs: Arguments passed to dataset constructor.

    Returns:
        Dataset instance.
    """
    normalized_name = dataset_name.lower().strip()
    if normalized_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Available datasets: {list(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[normalized_name](**kwargs)


def create_data_loaders(
    dataset_name: str,
    data_dir: Union[str, Path] = "data/processed",
    batch_size: int = 64,
    num_workers: int = 0,
    **kwargs: Any,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test DataLoaders for the specified dataset.

    Args:
        dataset_name: Name of dataset to instantiate.
        data_dir: Directory containing processed data files.
        batch_size: Batch size for DataLoader.
        num_workers: Number of workers for DataLoader.
        **kwargs: Extra parameters.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    normalized_name = dataset_name.lower().strip()
    if normalized_name in ("uci_power", "household_power"):
        return get_uci_power_loaders(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    raise ValueError(
        f"Unsupported dataset loaders for '{dataset_name}'. Available: ['uci_power', 'household_power']"
    )

