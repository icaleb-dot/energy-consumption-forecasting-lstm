"""Data preprocessing and feature engineering pipeline for UCI Household Power Consumption.

Pipeline steps:
1. Load raw text dataset (separators ';', missing '?').
2. Parse datetime from 'Date' and 'Time' columns.
3. Resample to hourly mean frequency ('1h').
4. Handle missing values:
   - Interpolation strategy: Linear interpolation.
   - Decision rationale: Electric power consumption varies smoothly between adjacent
     hours due to steady household baseline loads and predictable activity cycles.
     Linear interpolation preserves local trend and continuity for short missing intervals
     without injecting sudden step artifacts or distributional bias.
5. Strict temporal split (70% train, 15% val, 15% test) without temporal leakage.
6. Scaler (StandardScaler / MinMaxScaler) fitted strictly on the train set only and saved.
7. Sliding window generation with configurable input_window and forecast horizon.
8. Output arrays saved to data/processed/{train,val,test}.npz for fast PyTorch ingestion.
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_raw_power_data(raw_path: Path, target_col: str = "Global_active_power") -> pd.DataFrame:
    """Load and parse datetime for the UCI power consumption text file.

    Args:
        raw_path: Path to household_power_consumption.txt.
        target_col: Target variable column name.

    Returns:
        DataFrame with DatetimeIndex and the target column.
    """
    logger.info("Loading raw data from %s...", raw_path)
    df = pd.read_csv(
        raw_path,
        sep=";",
        na_values=["?", ""],
        usecols=["Date", "Time", target_col],
        dtype={target_col: "float32"},
        low_memory=False,
    )

    logger.info("Parsing Date and Time timestamps...")
    # Date format is dd/mm/yyyy, Time is hh:mm:ss
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )
    df = df.dropna(subset=["datetime"])
    df = df.set_index("datetime")
    df = df[[target_col]].sort_index()
    return df


def resample_and_impute(
    df: pd.DataFrame,
    target_col: str = "Global_active_power",
    freq: str = "1h",
) -> pd.Series:
    """Resample minute-level series to hourly frequency and interpolate missing values.

    Args:
        df: Input DataFrame with DatetimeIndex.
        target_col: Column to resample.
        freq: Pandas resampling frequency (default '1h').

    Returns:
        Clean continuous hourly pandas Series.
    """
    logger.info("Resampling series to hourly frequency ('%s')...", freq)
    series = df[target_col].resample(freq).mean()

    missing_count = int(series.isna().sum())
    logger.info("Missing values after hourly aggregation: %d (%.2f%%)", missing_count, (missing_count / len(series)) * 100)

    # Imputation: Linear interpolation + edge forward/backward fill
    series = series.interpolate(method="linear").bfill().ffill()
    logger.info("Imputation complete. Total hourly samples: %d", len(series))
    return series


def temporal_train_val_test_split(
    values: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split time series sequentially without shuffle to prevent data leakage.

    Args:
        values: 1D or 2D array of time series values.
        train_ratio: Proportion for training set.
        val_ratio: Proportion for validation set.

    Returns:
        Tuple of (train_split, val_split, test_split).
    """
    n = len(values)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_split = values[:train_end]
    val_split = values[train_end:val_end]
    test_split = values[val_end:]

    return train_split, val_split, test_split


def build_and_fit_scaler(
    train_data: np.ndarray,
    scaler_type: str = "standard",
) -> Tuple[object, np.ndarray]:
    """Fit a scaler strictly on the train partition.

    Args:
        train_data: Train values array of shape (N,) or (N, 1).
        scaler_type: 'standard' for StandardScaler or 'minmax' for MinMaxScaler.

    Returns:
        Fitted scaler instance and scaled train array.
    """
    if train_data.ndim == 1:
        train_data = train_data.reshape(-1, 1)

    if scaler_type.lower() == "minmax":
        scaler = MinMaxScaler(feature_range=(0, 1))
    else:
        scaler = StandardScaler()

    scaler.fit(train_data)
    train_scaled = scaler.transform(train_data)
    return scaler, train_scaled


def create_sliding_windows(
    series: np.ndarray,
    input_window: int = 24,
    horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate rolling windows for forecasting.

    Args:
        series: Scaled 1D or 2D series (N, 1).
        input_window: Number of historical lag steps in each window.
        horizon: Forecast horizon (number of steps to predict ahead).

    Returns:
        Tuple (X, y):
            X: Array of shape (num_windows, input_window, 1)
            y: Array of shape (num_windows, horizon)
    """
    if series.ndim == 1:
        series = series.reshape(-1, 1)

    n_samples = len(series) - input_window - horizon + 1
    if n_samples <= 0:
        raise ValueError(
            f"Series length ({len(series)}) too short for input_window={input_window} and horizon={horizon}"
        )

    x_list = []
    y_list = []

    for i in range(n_samples):
        x_window = series[i : i + input_window]
        y_window = series[i + input_window : i + input_window + horizon]
        x_list.append(x_window)
        y_list.append(y_window.squeeze(-1) if horizon == 1 else y_window)

    x_arr = np.asarray(x_list, dtype=np.float32)
    y_arr = np.asarray(y_list, dtype=np.float32)

    if y_arr.ndim == 1:
        y_arr = y_arr.reshape(-1, 1)

    return x_arr, y_arr


def run_pipeline(params: Dict) -> None:
    """Execute the full data preprocessing and artifact persistence pipeline."""
    dataset_cfg = params.get("dataset", {})
    raw_path = Path(dataset_cfg.get("raw_path", "data/raw/household_power_consumption.txt"))
    target_col = dataset_cfg.get("target_col", "Global_active_power")
    input_window = int(dataset_cfg.get("input_window", 24))
    horizon = int(dataset_cfg.get("horizon", 1))
    train_ratio = float(dataset_cfg.get("train_ratio", 0.70))
    val_ratio = float(dataset_cfg.get("val_ratio", 0.15))
    scaler_type = dataset_cfg.get("scaler_type", "standard")
    processed_dir = Path(dataset_cfg.get("processed_dir", "data/processed"))
    scaler_path = Path(dataset_cfg.get("scaler_path", "artifacts/scaler.joblib"))

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw data file not found at {raw_path}. Run ingest.py first to download it."
        )

    processed_dir.mkdir(parents=True, exist_ok=True)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)

    # 1 & 2: Load and resample
    raw_df = load_raw_power_data(raw_path, target_col=target_col)
    hourly_series = resample_and_impute(raw_df, target_col=target_col, freq="1h")
    values = hourly_series.to_numpy(dtype=np.float32).reshape(-1, 1)

    # 3: Temporal Split
    train_raw, val_raw, test_raw = temporal_train_val_test_split(
        values, train_ratio=train_ratio, val_ratio=val_ratio
    )
    logger.info(
        "Temporal split completed: Train=%d, Val=%d, Test=%d samples",
        len(train_raw),
        len(val_raw),
        len(test_raw),
    )

    # 4: Fit scaler ONLY on train split
    scaler, train_scaled = build_and_fit_scaler(train_raw, scaler_type=scaler_type)
    val_scaled = scaler.transform(val_raw)
    test_scaled = scaler.transform(test_raw)

    joblib.dump(scaler, scaler_path)
    logger.info("Saved fitted scaler (%s) to %s", scaler_type, scaler_path)

    # 5: Create sliding windows
    x_train, y_train = create_sliding_windows(train_scaled, input_window=input_window, horizon=horizon)
    x_val, y_val = create_sliding_windows(val_scaled, input_window=input_window, horizon=horizon)
    x_test, y_test = create_sliding_windows(test_scaled, input_window=input_window, horizon=horizon)

    logger.info("Window shapes - Train: X=%s, y=%s", x_train.shape, y_train.shape)
    logger.info("Window shapes - Val:   X=%s, y=%s", x_val.shape, y_val.shape)
    logger.info("Window shapes - Test:  X=%s, y=%s", x_test.shape, y_test.shape)

    # 6: Save processed npz files
    np.savez_compressed(processed_dir / "train.npz", x=x_train, y=y_train)
    np.savez_compressed(processed_dir / "val.npz", x=x_val, y=y_val)
    np.savez_compressed(processed_dir / "test.npz", x=x_test, y=y_test)

    metadata = {
        "target_col": target_col,
        "input_window": input_window,
        "horizon": horizon,
        "train_samples": int(len(x_train)),
        "val_samples": int(len(x_val)),
        "test_samples": int(len(x_test)),
        "scaler_type": scaler_type,
        "scaler_path": str(scaler_path),
        "mean_train": float(getattr(scaler, "mean_", [0])[0]) if hasattr(scaler, "mean_") else None,
        "scale_train": float(getattr(scaler, "scale_", [1])[0]) if hasattr(scaler, "scale_") else None,
    }
    with open(processed_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Preprocessing complete! Processed data stored in %s", processed_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess UCI Household Power Consumption dataset")
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()

    with open(args.params, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    run_pipeline(params)


if __name__ == "__main__":
    main()

