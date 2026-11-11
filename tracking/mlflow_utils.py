"""MLflow tracking helpers and version-compatible logging."""
import inspect
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import mlflow.pytorch
import torch.nn as nn

logger = logging.getLogger(__name__)


def setup_mlflow(
    tracking_uri: Optional[str] = None,
    experiment_name: str = "household_power_forecasting",
) -> str:
    """Configure MLflow tracking URI and active experiment.

    Priority:
    1. tracking_uri parameter passed explicitly
    2. MLFLOW_TRACKING_URI environment variable
    3. Default local SQLite database: sqlite:///mlflow.db

    Args:
        tracking_uri: Optional URI string.
        experiment_name: Name of the MLflow experiment.

    Returns:
        The resolved tracking URI.
    """
    resolved_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(resolved_uri)
    logger.info("MLflow tracking URI set to: %s", resolved_uri)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        logger.info("Created new MLflow experiment '%s' (id: %s)", experiment_name, experiment_id)
    else:
        logger.info("Using existing MLflow experiment '%s'", experiment_name)

    mlflow.set_experiment(experiment_name)
    return resolved_uri


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Recursively flatten nested dictionary for clean MLflow parameter logging."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def log_params_dict(params: Dict[str, Any]) -> None:
    """Log parameters dictionary to MLflow, flattening nested structures."""
    if mlflow.active_run() is None:
        return
    flat = flatten_dict(params)
    # MLflow parameter values truncated if too long
    sanitized = {k: str(v) if not isinstance(v, (int, float, bool)) else v for k, v in flat.items()}
    mlflow.log_params(sanitized)


def log_model_compatible(
    model: nn.Module,
    artifact_path: str = "model",
    registered_model_name: Optional[str] = None,
) -> None:
    """Log PyTorch model to MLflow respecting version differences in argument names.

    Different MLflow versions use 'artifact_path' vs 'name'.
    """
    if mlflow.active_run() is None:
        logger.warning("No active MLflow run. Skipping log_model.")
        return

    sig = inspect.signature(mlflow.pytorch.log_model)
    kwargs: Dict[str, Any] = {"pytorch_model": model}

    if "artifact_path" in sig.parameters:
        kwargs["artifact_path"] = artifact_path
    elif "name" in sig.parameters:
        kwargs["name"] = artifact_path

    if registered_model_name and "registered_model_name" in sig.parameters:
        kwargs["registered_model_name"] = registered_model_name

    try:
        mlflow.pytorch.log_model(**kwargs)
        logger.info("Successfully logged model to MLflow (artifact_path='%s')", artifact_path)
    except Exception as exc:
        logger.warning("Failed to log model via mlflow.pytorch.log_model: %s", exc)


def log_artifact_if_exists(file_path: str | Path, artifact_path: Optional[str] = None) -> None:
    """Log a file artifact to MLflow if it exists on disk."""
    if mlflow.active_run() is None:
        return
    p = Path(file_path)
    if p.exists():
        mlflow.log_artifact(str(p), artifact_path=artifact_path)
        logger.info("Logged artifact to MLflow: %s", p)
    else:
        logger.warning("Artifact not found on disk, skipping log: %s", p)

