"""Evaluation script for test split comparison and metrics.json generation."""
import argparse
import json
import logging
from pathlib import Path
from typing import Dict

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from datasets.factory import create_data_loaders
from datasets.uci_power import UCIPowerDataset
from engine.evaluator import compare_model_with_baseline, predict_loader
from models.factory import create_model
from utils.device import get_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_pipeline(params: Dict, metrics_out_path: Path = Path("metrics.json")) -> Dict:
    """Run model evaluation on test set, compare to naive baseline, and write metrics.json.

    Args:
        params: Configuration dictionary loaded from params.yaml.
        metrics_out_path: Destination path for the output JSON file.

    Returns:
        Evaluation summary dictionary.
    """
    device = get_device()
    logger.info("Evaluating on device: %s", device)

    # Paths
    dataset_cfg = params.get("dataset", {})
    model_cfg = params.get("model", {})
    processed_dir = Path(dataset_cfg.get("processed_dir", "data/processed"))
    test_path = processed_dir / "test.npz"
    scaler_path = Path(dataset_cfg.get("scaler_path", "artifacts/scaler.joblib"))
    checkpoint_path = Path("artifacts/best_model.pth")

    if not test_path.exists():
        raise FileNotFoundError(f"Test split not found at {test_path}. Run preprocessing first.")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}. Train model first.")

    # Data Loader
    test_ds = UCIPowerDataset(test_path)
    batch_size = int(params.get("train", {}).get("batch_size", 64))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Scaler
    scaler = None
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        logger.info("Loaded scaler from %s", scaler_path)

    # Model
    model = create_model(
        model_name=model_cfg.get("type", "lstm"),
        input_size=model_cfg.get("input_size", 1),
        hidden_size=model_cfg.get("hidden_size", 32),
        num_layers=model_cfg.get("num_layers", 1),
        output_size=model_cfg.get("output_size", 1),
        dropout=model_cfg.get("dropout", 0.0),
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        model = checkpoint
    model = model.to(device)
    model.eval()

    # Compare with Naive Baseline (evaluated on real, unscaled scale)
    comparison = compare_model_with_baseline(
        model=model,
        dataloader=test_loader,
        device=device,
        scaler=scaler,
    )

    model_m = comparison["model"]
    base_m = comparison["naive_baseline"]
    imp = comparison["improvements"]

    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS (TEST SET - REAL SCALE [kW])")
    logger.info("-" * 60)
    logger.info("Model (%s):        RMSE=%.4f | MAE=%.4f | MAPE=%.2f%%", model_cfg.get("type"), model_m["rmse"], model_m["mae"], model_m["mape"])
    logger.info("Naive Baseline:     RMSE=%.4f | MAE=%.4f | MAPE=%.2f%%", base_m["rmse"], base_m["mae"], base_m["mape"])
    logger.info("Improvement vs Baseline: RMSE: +%.2f%% | MAE: +%.2f%%", imp["rmse_reduction_pct"], imp["mae_reduction_pct"])
    logger.info("=" * 60)

    # Flatten metrics for DVC / tracking
    flat_metrics = {
        "test_rmse": round(model_m["rmse"], 4),
        "test_mae": round(model_m["mae"], 4),
        "test_mape": round(model_m["mape"], 2),
        "baseline_rmse": round(base_m["rmse"], 4),
        "baseline_mae": round(base_m["mae"], 4),
        "baseline_mape": round(base_m["mape"], 2),
        "improvement_rmse_pct": round(imp["rmse_reduction_pct"], 2),
        "improvement_mae_pct": round(imp["mae_reduction_pct"], 2),
    }

    metrics_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_out_path, "w", encoding="utf-8") as f:
        json.dump(flat_metrics, f, indent=2)
    logger.info("Metrics written to %s", metrics_out_path)

    # Generate and save test forecast plot
    preds, targets = predict_loader(model, test_loader, device)
    if scaler is not None:
        preds = scaler.inverse_transform(preds.reshape(-1, 1)).reshape(preds.shape)
        targets = scaler.inverse_transform(targets.reshape(-1, 1)).reshape(targets.shape)

    plot_samples = min(200, len(targets))
    plt.figure(figsize=(14, 5))
    plt.plot(targets[:plot_samples, 0], label="Ground Truth (Real)", color="tab:blue", linewidth=1.5)
    plt.plot(preds[:plot_samples, 0], label=f"Prediction ({model_cfg.get('type')})", color="tab:orange", linestyle="--", linewidth=1.5)
    plt.title(f"UCI Power Consumption - Real vs Predicted (First {plot_samples} test hours)")
    plt.xlabel("Hour")
    plt.ylabel("Global Active Power (kW)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    plot_file = artifacts_dir / "test_forecast.png"
    plt.savefig(plot_file, dpi=150)
    plt.close()
    logger.info("Forecast plot saved to %s", plot_file)

    return flat_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate forecasting model on test set")
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    parser.add_argument("--output", default="metrics.json", help="Path to output metrics json")
    args = parser.parse_args()

    with open(args.params, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    evaluate_pipeline(params, metrics_out_path=Path(args.output))


if __name__ == "__main__":
    main()

