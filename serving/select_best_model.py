"""Select best model run from MLflow by validation RMSE and export for serving."""
import argparse
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import joblib
import mlflow
import mlflow.pytorch
import pandas as pd
import torch
import yaml

from models.factory import create_model
from tracking.mlflow_utils import setup_mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def select_and_export_best_model(
    params: Dict,
    export_dir: Path = Path("models_export/best_model"),
    register_model_name: Optional[str] = "BestHouseholdPowerForecaster",
) -> Path:
    """Find the best MLflow run by lowest validation RMSE and export artifacts for serving.

    Args:
        params: Configuration dictionary from params.yaml.
        export_dir: Directory where exported production artifacts are saved.
        register_model_name: Optional model registry name.

    Returns:
        Path to the exported best model directory.
    """
    tracking_cfg = params.get("tracking", {})
    tracking_uri = tracking_cfg.get("uri", "sqlite:///mlflow.db")
    experiment_name = tracking_cfg.get("experiment_name", "household_power_forecasting")

    setup_mlflow(tracking_uri=tracking_uri, experiment_name=experiment_name)

    export_dir.mkdir(parents=True, exist_ok=True)

    # 1. Search finished runs in the experiment
    logger.info("Searching MLflow runs in experiment '%s'...", experiment_name)
    try:
        runs = mlflow.search_runs(
            experiment_names=[experiment_name],
            filter_string="attributes.status = 'FINISHED'",
        )
    except Exception as exc:
        logger.warning("MLflow search_runs encountered an error: %s", exc)
        runs = None

    best_run_id = None
    best_metrics = {}
    best_model_type = params.get("model", {}).get("type", "lstm")

    if runs is not None and not runs.empty:
        # Determine sort metric
        metric_col = None
        for candidate in ["metrics.best_val_rmse", "metrics.val_rmse", "metrics.best_val_loss"]:
            if candidate in runs.columns:
                metric_col = candidate
                break

        if metric_col:
            runs_sorted = runs.sort_values(by=metric_col, ascending=True)
            best_run = runs_sorted.iloc[0]
            best_run_id = best_run["run_id"]
            best_metric_val = best_run[metric_col]
            logger.info("Found best run ID: %s (%s = %.5f)", best_run_id, metric_col, best_metric_val)

            # Extract model type from params if present
            for col in ["params.model.type", "params.model_type"]:
                if col in best_run and not pd.isna(best_run[col]):
                    best_model_type = str(best_run[col])
                    break
        else:
            best_run = runs.iloc[0]
            best_run_id = best_run["run_id"]

    # 2. Export model weights
    dest_model_path = export_dir / "model.pth"
    model_loaded = False

    if best_run_id:
        model_uri = f"runs:/{best_run_id}/model"
        try:
            logger.info("Loading model from MLflow: %s", model_uri)
            py_model = mlflow.pytorch.load_model(model_uri, map_location="cpu")
            torch.save(py_model.state_dict(), dest_model_path)
            model_loaded = True
            logger.info("Exported MLflow model state_dict to %s", dest_model_path)

            if register_model_name:
                try:
                    mlflow.register_model(model_uri=model_uri, name=register_model_name)
                    logger.info("Registered model in MLflow Model Registry as '%s'", register_model_name)
                except Exception as reg_exc:
                    logger.info("Model registry registration skipped: %s", reg_exc)
        except Exception as load_exc:
            logger.warning("Could not load model directly from MLflow URI: %s", load_exc)

    # Fallback to local artifact if MLflow loading was not possible
    if not model_loaded:
        local_checkpoint = Path("artifacts/best_model.pth")
        if local_checkpoint.exists():
            shutil.copy(local_checkpoint, dest_model_path)
            logger.info("Exported local checkpoint %s to %s", local_checkpoint, dest_model_path)
        else:
            raise FileNotFoundError(
                "No trained model could be found in MLflow runs or local artifacts/best_model.pth."
            )

    # 3. Export Scaler
    dest_scaler_path = export_dir / "scaler.joblib"
    local_scaler = Path(params.get("dataset", {}).get("scaler_path", "artifacts/scaler.joblib"))
    if local_scaler.exists():
        shutil.copy(local_scaler, dest_scaler_path)
        logger.info("Exported scaler to %s", dest_scaler_path)
    else:
        logger.warning("Scaler file %s not found.", local_scaler)

    # 4. Export Metadata
    metadata = {
        "best_run_id": best_run_id,
        "model_type": best_model_type,
        "input_size": params.get("model", {}).get("input_size", 1),
        "hidden_size": params.get("model", {}).get("hidden_size", 32),
        "num_layers": params.get("model", {}).get("num_layers", 1),
        "output_size": params.get("model", {}).get("output_size", 1),
        "dropout": params.get("model", {}).get("dropout", 0.0),
        "input_window": params.get("dataset", {}).get("input_window", 24),
        "horizon": params.get("dataset", {}).get("horizon", 1),
        "target_col": params.get("dataset", {}).get("target_col", "Global_active_power"),
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(export_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Export complete! Production model and artifacts ready at %s", export_dir)
    return export_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Select best model from MLflow and export for serving")
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    parser.add_argument("--export-dir", default="models_export/best_model", help="Export target directory")
    args = parser.parse_args()

    with open(args.params, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    select_and_export_best_model(params=params, export_dir=Path(args.export_dir))


if __name__ == "__main__":
    main()
