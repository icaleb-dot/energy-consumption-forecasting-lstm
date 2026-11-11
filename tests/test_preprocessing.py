"""Tests for data preprocessing, temporal splitting, and data leakage prevention."""
import numpy as np
import pandas as pd
import pytest

from preprocess import (
    build_and_fit_scaler,
    create_sliding_windows,
    resample_and_impute,
    temporal_train_val_test_split,
)


def test_temporal_split_order_and_proportions():
    """Verify that train/val/test splits maintain strict chronological ordering without shuffling."""
    n_samples = 1000
    # Strictly increasing sequence so values equal their time index
    series = np.arange(n_samples, dtype=np.float32)

    train, val, test = temporal_train_val_test_split(
        series, train_ratio=0.70, val_ratio=0.15
    )

    # 1. Check sizes
    assert len(train) == 700
    assert len(val) == 150
    assert len(test) == 150
    assert len(train) + len(val) + len(test) == n_samples

    # 2. Check strict temporal order (max of previous split < min of subsequent split)
    assert train[-1] < val[0], "Temporal order violated between train and val"
    assert val[-1] < test[0], "Temporal order violated between val and test"

    # 3. Check exact boundaries
    assert train[0] == 0.0
    assert train[-1] == 699.0
    assert val[0] == 700.0
    assert val[-1] == 849.0
    assert test[0] == 850.0
    assert test[-1] == 999.0


def test_scaler_fitted_strictly_on_train_no_leakage():
    """Verify that the scaler is fitted exclusively on train data without seeing val or test."""
    # Construct train distribution with mean ~ 10, std ~ 2
    np.random.seed(42)
    train_data = np.random.normal(loc=10.0, scale=2.0, size=(1000, 1)).astype(np.float32)

    # Construct val and test with different mean to easily detect leakage
    val_data = np.random.normal(loc=50.0, scale=5.0, size=(300, 1)).astype(np.float32)
    test_data = np.random.normal(loc=100.0, scale=10.0, size=(300, 1)).astype(np.float32)

    scaler, train_scaled = build_and_fit_scaler(train_data, scaler_type="standard")

    # Scaler mean and std must match train_data statistics within numeric precision
    np.testing.assert_allclose(scaler.mean_[0], np.mean(train_data), rtol=1e-4)
    np.testing.assert_allclose(scaler.scale_[0], np.std(train_data), rtol=1e-4)

    # Scaler must NOT have absorbed any information from val or test
    overall_mean = np.mean(np.concatenate([train_data, val_data, test_data]))
    assert abs(scaler.mean_[0] - overall_mean) > 10.0, "Data leakage detected: scaler reflects full dataset mean!"

    # Transformed train split should have mean ~0 and std ~1
    assert abs(np.mean(train_scaled)) < 1e-3
    assert abs(np.std(train_scaled) - 1.0) < 1e-3


def test_create_sliding_windows_shapes_and_offsets():
    """Verify that sliding windows create expected X and y shapes and temporal alignment."""
    # 100 sequential timestamps: [0, 1, 2, ..., 99]
    series = np.arange(100, dtype=np.float32).reshape(-1, 1)
    input_win = 24
    horizon = 3

    x, y = create_sliding_windows(series, input_window=input_win, horizon=horizon)

    expected_samples = 100 - input_win - horizon + 1  # 100 - 24 - 3 + 1 = 74
    assert x.shape == (expected_samples, input_win, 1)
    assert y.shape == (expected_samples, horizon)

    # Check first window alignment
    # x[0] should be 0..23, y[0] should be 24, 25, 26
    np.testing.assert_array_equal(x[0, :, 0], np.arange(0, 24))
    np.testing.assert_array_equal(y[0], np.array([24.0, 25.0, 26.0]))

    # Check last window alignment
    np.testing.assert_array_equal(x[-1, :, 0], np.arange(73, 97))
    np.testing.assert_array_equal(y[-1], np.array([97.0, 98.0, 99.0]))


def test_resample_and_impute_missing_values():
    """Verify hourly resampling and linear interpolation of missing intervals."""
    # Create 120 minutes of data (2 hours) with missing values
    times = pd.date_range("2026-01-01 00:00:00", periods=120, freq="1min")
    values = np.linspace(1.0, 3.0, 120)
    # Introduce NaN gap
    values[30:50] = np.nan

    df = pd.DataFrame({"Global_active_power": values}, index=times)
    series = resample_and_impute(df, target_col="Global_active_power", freq="1h")

    # Should have 2 hourly values
    assert len(series) == 2
    # No NaNs should remain
    assert not series.isna().any()

