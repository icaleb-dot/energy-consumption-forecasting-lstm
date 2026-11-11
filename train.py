"""Model training entrypoint for time series forecasting with MLflow tracking."""
import argparse
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
import mlflow

from callbacks.early_stopping import EarlyStopping
from datasets.factory import create_data_loaders
from engine.trainer import fit
from models.factory import create_model
from tracking.mlflow_utils import (
    log_artifact_if_exists,
    log_model_compatible,
    log_params_dict,
    setup_mlflow,
)
from utils.device import get_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Fix random seeds across python, numpy, and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_params(path: str = "params.yaml") -> Dict:
    """Load configuration dictionary from YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_single_model(
    model_type: str,
    params: Dict,
    train_loader,
    val_loader,
    device: torch.device,
    save_checkpoint: bool = True,
) -> Dict:
    """Train a single model architecture and record to MLflow."""
    train_cfg = params.get("train", {})
    model_cfg = params.get("model", {}).copy()
    model_cfg["type"] = model_type
    scheduler_cfg = params.get("scheduler", {})
    tracking_cfg = params.get("tracking", {})
    dataset_cfg = params.get("dataset", {})

    batch_size = int(train_cfg.get("batch_size", 64))
    lr = float(train_cfg.get("learning_rate", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    epochs = int(train_cfg.get("epochs", 30))
    patience = int(train_cfg.get("patience", 5))
    min_delta = float(train_cfg.get("min_delta", 1e-4))

    run_name = f"{model_type}-forecaster"

    logger.info("Starting MLflow run: %s", run_name)
    with mlflow.start_run(run_name=run_name):
        # 1. Log all hyperparameters
        full_params = {
            "model": model_cfg,
            "train": train_cfg,
            "scheduler": scheduler_cfg,
            "dataset": {
                "name": dataset_cfg.get("name", "uci_power"),
                "input_window": dataset_cfg.get("input_window", 24),
                "horizon": dataset_cfg.get("horizon", 1),
                "scaler_type": dataset_cfg.get("scaler_type", "standard"),
            },
        }
        log_params_dict(full_params)

        # 2. Instantiate Model
        model = create_model(
            model_name=model_type,
            input_size=int(model_cfg.get("input_size", 1)),
            hidden_size=int(model_cfg.get("hidden_size", 32)),
            num_layers=int(model_cfg.get("num_layers", 1)),
            output_size=int(model_cfg.get("output_size", 1)),
            dropout=float(model_cfg.get("dropout", 0.0)),
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        scheduler = None
        if scheduler_cfg:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=scheduler_cfg.get("mode", "min"),
                factor=float(scheduler_cfg.get("factor", 0.5)),
                patience=int(scheduler_cfg.get("patience", 2)),
            )

        early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

        # 3. Fit
        history = fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=epochs,
            early_stopping=early_stopping,
            scheduler=scheduler,
            log_to_mlflow=True,
        )

        best_val_loss = min(history["val_loss"])
        best_val_rmse = min(history["val_rmse"])
        final_metrics = {
            "best_val_loss": float(best_val_loss),
            "best_val_rmse": float(best_val_rmse),
            "final_train_loss": float(history["train_loss"][-1]),
            "final_val_loss": float(history["val_loss"][-1]),
            "epochs_trained": len(history["train_loss"]),
        }
        mlflow.log_metrics(final_metrics)

        # 4. Save model and scaler artifacts
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        if save_checkpoint:
            checkpoint_path = artifacts_dir / "best_model.pth"
            if early_stopping.best_model_state is not None:
                torch.save(early_stopping.best_model_state, checkpoint_path)
            else:
                torch.save(model.state_dict(), checkpoint_path)
            log_artifact_if_exists(checkpoint_path)

        scaler_path = Path(dataset_cfg.get("scaler_path", "artifacts/scaler.joblib"))
        log_artifact_if_exists(scaler_path)

        # 5. Log PyTorch Model
        log_model_compatible(model, artifact_path="model")

        # 6. Save loss plot
        plt.figure(figsize=(8, 4))
        plt.plot(history["train_loss"], label="Train Loss (MSE)")
        plt.plot(history["val_loss"], label="Val Loss (MSE)")
        plt.title(f"{model_type.upper()} Training History")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        loss_plot_path = artifacts_dir / f"{model_type}_losses.png"
        plt.savefig(loss_plot_path, dpi=150)
        plt.close()
        log_artifact_if_exists(loss_plot_path)

        logger.info(
            "Finished run %s: Best Val Loss=%.5f, Best Val RMSE=%.5f",
            run_name,
            best_val_loss,
            best_val_rmse,
        )
        return final_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train forecasting recurrent models")
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model type: 'rnn', 'lstm', 'gru' or 'all'",
    )
    args = parser.parse_args()

    params = load_params(args.params)
    train_cfg = params.get("train", {})
    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)

    device = get_device()
    logger.info("Using device: %s", device)

    tracking_cfg = params.get("tracking", {})
    setup_mlflow(
        tracking_uri=tracking_cfg.get("uri", "sqlite:///mlflow.db"),
        experiment_name=tracking_cfg.get("experiment_name", "household_power_forecasting"),
    )

    # Prepare DataLoaders
    dataset_cfg = params.get("dataset", {})
    processed_dir = Path(dataset_cfg.get("processed_dir", "data/processed"))
    batch_size = int(train_cfg.get("batch_size", 64))

    train_loader, val_loader, _ = create_data_loaders(
        dataset_name=dataset_cfg.get("name", "uci_power"),
        data_dir=processed_dir,
        batch_size=batch_size,
    )

    chosen_model = args.model or params.get("model", {}).get("type", "lstm")

    if chosen_model.lower() == "all":
        models_to_train = ["rnn", "lstm", "gru"]
        logger.info("Training comparison across all models: %s", models_to_train)
        for m in models_to_train:
            train_single_model(
                model_type=m,
                params=params,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                save_checkpoint=(m == models_to_train[0]),
            )
    else:
        train_single_model(
            model_type=chosen_model,
            params=params,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            save_checkpoint=True,
        )


if __name__ == "__main__":
    main()
