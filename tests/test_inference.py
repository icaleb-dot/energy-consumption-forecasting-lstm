"""Tests for inference module and FastAPI endpoints."""
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from sklearn.preprocessing import StandardScaler

from inference.api import app, state
from inference.predict import forecast_multi_step, predict
from models.lstm import LSTMForecaster


@pytest.fixture
def dummy_forecaster():
    """Create a dummy LSTM forecaster with fitted scaler."""
    model = LSTMForecaster(input_size=1, hidden_size=16, num_layers=1, output_size=1)
    model.eval()

    scaler = StandardScaler()
    scaler.fit(np.array([[0.0], [1.0], [2.0], [3.0]]))
    return model, scaler


def test_predict_shape_and_type(dummy_forecaster):
    """Test that predict returns a list of floats matching horizon."""
    model, scaler = dummy_forecaster
    seq = [1.0] * 24

    preds = predict(model, scaler, seq)
    assert isinstance(preds, list)
    assert len(preds) == 1
    assert isinstance(preds[0], float)


def test_forecast_multi_step(dummy_forecaster):
    """Test that multi-step autoregressive forecast produces expected steps."""
    model, scaler = dummy_forecaster
    seq = [1.5] * 24
    steps = 6

    multi_preds = forecast_multi_step(model, scaler, seq, steps=steps)
    assert len(multi_preds) == steps
    assert all(isinstance(v, float) for v in multi_preds)


def test_api_health_endpoint(dummy_forecaster):
    """Test the /health endpoint with TestClient."""
    model, scaler = dummy_forecaster
    state["model"] = model
    state["scaler"] = scaler
    state["metadata"] = {"model_type": "lstm", "input_window": 24, "horizon": 1}

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["model_type"] == "lstm"


def test_api_predict_valid_payload(dummy_forecaster):
    """Test /predict endpoint with a valid 24-step historical sequence."""
    model, scaler = dummy_forecaster
    state["model"] = model
    state["scaler"] = scaler
    state["metadata"] = {"model_type": "lstm", "input_window": 24, "horizon": 1}

    client = TestClient(app)
    payload = {"input_window": [1.2] * 24}
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 1
    assert data["unit"] == "kW"
    assert data["model_type"] == "lstm"


def test_api_predict_invalid_window_length(dummy_forecaster):
    """Test that /predict rejects inputs that do not match the expected window length."""
    model, scaler = dummy_forecaster
    state["model"] = model
    state["scaler"] = scaler
    state["metadata"] = {"model_type": "lstm", "input_window": 24, "horizon": 1}

    client = TestClient(app)
    # Send only 10 items instead of 24
    payload = {"input_window": [1.0] * 10}
    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert "Input window length mismatch" in response.json()["detail"]

