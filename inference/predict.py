"""Inference module for household electricity demand forecasting.

Loads the exported production model and scaler to generate real-world unscaled predictions.
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import torch

from models.factory import create_model

logger = logging.getLogger(__name__)


def load_forecaster(
    export_dir: Union[str, Path] = "models_export/best_model",
) -> Tuple[torch.nn.Module, Any, Dict[str, Any]]:
    """Load the exported production model, scaler, and metadata.

    Args:
        export_dir: Directory containing model.pth, scaler.joblib, and metadata.json.

    Returns:
        Tuple of (model, scaler, metadata).
    """
    export_path = Path(export_dir)
    metadata_path = export_path / "metadata.json"
    model_path = export_path / "model.pth"
    scaler_path = export_path / "scaler.joblib"

    if not metadata_path.exists():
        metadata = {
            "model_type": "lstm",
            "input_size": 1,
            "hidden_size": 32,
            "num_layers": 1,
            "output_size": 1,
            "dropout": 0.0,
            "input_window": 24,
            "horizon": 1,
        }
    else:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    model = create_model(
        model_name=metadata.get("model_type", "lstm"),
        input_size=int(metadata.get("input_size", 1)),
        hidden_size=int(metadata.get("hidden_size", 32)),
        num_layers=int(metadata.get("num_layers", 1)),
        output_size=int(metadata.get("output_size", 1)),
        dropout=float(metadata.get("dropout", 0.0)),
    )

    if model_path.exists():
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
    else:
        # Fallback to artifacts if not exported yet
        fallback = Path("artifacts/best_model.pth")
        if fallback.exists():
            state_dict = torch.load(fallback, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"Model file not found at {model_path} or {fallback}")

    model.eval()

    scaler = None
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
    else:
        fallback_scaler = Path("artifacts/scaler.joblib")
        if fallback_scaler.exists():
            scaler = joblib.load(fallback_scaler)

    return model, scaler, metadata


def predict(
    model: torch.nn.Module,
    scaler: Any,
    input_sequence: Union[List[float], np.ndarray],
) -> List[float]:
    """Generate descaled power consumption forecast for the given historical window.

    Args:
        model: Evaluated PyTorch model.
        scaler: Fitted scikit-learn scaler.
        input_sequence: 1D sequence of length equal to input_window in original scale (kW).

    Returns:
        List of descaled forecasted values in kW.
    """
    raw_arr = np.asarray(input_sequence, dtype=np.float32).reshape(-1, 1)

    # 1. Scale input using fitted training scaler
    if scaler is not None and hasattr(scaler, "transform"):
        scaled_input = scaler.transform(raw_arr)
    else:
        scaled_input = raw_arr

    # 2. Reshape to [1, input_window, features]
    x_tensor = torch.from_numpy(scaled_input).float().unsqueeze(0)

    # 3. Model forward pass
    with torch.inference_mode():
        output = model(x_tensor).cpu().numpy()  # [1, horizon]

    # 4. Descale prediction to original units (kW)
    if scaler is not None and hasattr(scaler, "inverse_transform"):
        descaled_output = scaler.inverse_transform(output.reshape(-1, 1)).reshape(output.shape)
    else:
        descaled_output = output

    return descaled_output[0].tolist()


def forecast_multi_step(
    model: torch.nn.Module,
    scaler: Any,
    initial_sequence: Union[List[float], np.ndarray],
    steps: int = 24,
) -> List[float]:
    """Autoregressive multi-step forecasting.

    Args:
        model: Trained model.
        scaler: Fitted scaler.
        initial_sequence: Historical window.
        steps: Total future hours to predict.

    Returns:
        List of predictions for t+1 through t+steps in kW.
    """
    window = list(initial_sequence)
    seq_len = len(initial_sequence)
    predictions = []

    for _ in range(steps):
        current_input = window[-seq_len:]
        step_preds = predict(model, scaler, current_input)
        next_val = step_preds[0]
        predictions.append(float(next_val))
        window.append(next_val)

    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict household active power consumption")
    parser.add_argument("--export-dir", default="models_export/best_model", help="Path to exported model")
    parser.add_argument("--steps", type=int, default=24, help="Number of autoregressive steps")
    args = parser.parse_args()

    model, scaler, metadata = load_forecaster(args.export_dir)
    input_win = int(metadata.get("input_window", 24))
    print(f"Loaded {metadata.get('model_type')} forecaster (input_window={input_win}, horizon={metadata.get('horizon', 1)})")

    # Generate synthetic demo sequence of typical active power values (approx 1.0 - 2.5 kW)
    demo_sequence = [1.2 + 0.4 * np.sin(i / 3.0) for i in range(input_win)]
    print(f"Input historical window ({input_win} hours): {[round(x, 2) for x in demo_sequence]}")

    single_step = predict(model, scaler, demo_sequence)
    print(f"\nNext hour prediction (t+1): {single_step[0]:.4f} kW")

    if args.steps > 1:
        multi_step = forecast_multi_step(model, scaler, demo_sequence, steps=args.steps)
        print(f"\nAutoregressive forecast for next {args.steps} hours (kW):")
        for h, val in enumerate(multi_step, 1):
            print(f"  Hour t+{h:02d}: {val:.4f} kW")


if __name__ == "__main__":
    main()
