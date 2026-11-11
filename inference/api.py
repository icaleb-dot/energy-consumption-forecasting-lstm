"""FastAPI inference microservice for Household Electric Power Forecasting."""
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from inference.predict import load_forecaster, predict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("inference_api")

# Global model state
state: Dict[str, Any] = {
    "model": None,
    "scaler": None,
    "metadata": {},
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager to load models and scaler into memory on startup."""
    export_dir = os.getenv("MODEL_EXPORT_DIR", "models_export/best_model")
    logger.info("Initializing inference service. Loading artifacts from %s...", export_dir)
    try:
        model, scaler, metadata = load_forecaster(export_dir)
        state["model"] = model
        state["scaler"] = scaler
        state["metadata"] = metadata
        logger.info(
            "Model successfully loaded: type=%s, input_window=%s, horizon=%s",
            metadata.get("model_type"),
            metadata.get("input_window"),
            metadata.get("horizon"),
        )
    except Exception as exc:
        logger.warning(
            "Could not load production model on startup (%s). "
            "Service will report degraded health until model is mounted.",
            exc,
        )
    yield
    logger.info("Shutting down inference service.")


app = FastAPI(
    title="Household Electricity Forecasting API",
    description="Microservice providing real-time forecasting of household active power consumption (kW).",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    """Input payload containing historical power readings."""

    input_window: List[float] = Field(
        ...,
        description="Historical hourly sequence of Global_active_power values in original units (kW).",
        examples=[[1.2, 1.4, 1.1, 0.9, 0.8, 1.3, 2.1, 2.5, 2.8, 2.4, 1.9, 1.5,
                   1.3, 1.1, 1.0, 1.2, 1.8, 2.9, 3.2, 2.7, 2.2, 1.8, 1.5, 1.3]],
    )


class PredictResponse(BaseModel):
    """Forecast response containing unscaled predictions in real units (kW)."""

    predictions: List[float] = Field(..., description="Forecasted power values for upcoming horizon steps.")
    model_type: str = Field(..., description="Recurrent neural network architecture utilized.")
    horizon: int = Field(..., description="Number of forecasted steps ahead.")
    unit: str = Field(default="kW", description="Measurement unit of the forecasted target.")


class HealthResponse(BaseModel):
    """Health check status response."""

    status: str
    model_loaded: bool
    model_type: Optional[str] = None
    input_window: Optional[int] = None
    horizon: Optional[int] = None


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Return health status and loaded model metadata."""
    is_loaded = state["model"] is not None
    meta = state.get("metadata", {})
    return HealthResponse(
        status="ok" if is_loaded else "degraded",
        model_loaded=is_loaded,
        model_type=meta.get("model_type"),
        input_window=meta.get("input_window"),
        horizon=meta.get("horizon"),
    )


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(payload: PredictRequest) -> PredictResponse:
    """Predict future active power consumption given historical input window.

    Validates sequence length, scales input with production scaler, executes model inference,
    and returns predictions inverse-transformed back to kW.
    """
    model = state["model"]
    scaler = state["scaler"]
    metadata = state["metadata"]

    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Forecasting model is not loaded yet. Mount or export best_model first.",
        )

    expected_len = int(metadata.get("input_window", 24))
    if len(payload.input_window) != expected_len:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Input window length mismatch. Expected {expected_len} hourly steps, received {len(payload.input_window)}.",
        )

    try:
        forecast = predict(
            model=model,
            scaler=scaler,
            input_sequence=payload.input_window,
        )
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(exc)}",
        )

    return PredictResponse(
        predictions=[round(float(v), 4) for v in forecast],
        model_type=str(metadata.get("model_type", "unknown")),
        horizon=int(metadata.get("horizon", len(forecast))),
        unit="kW",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)

