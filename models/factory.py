"""Model factory for time series forecasting recurrent networks."""
from typing import Any, Dict, Type

from models.base import BaseNN
from models.gru import GRUForecaster
from models.lstm import LSTMForecaster
from models.rnn import RNNForecaster

MODEL_REGISTRY: Dict[str, Type[BaseNN]] = {
    "rnn": RNNForecaster,
    "lstm": LSTMForecaster,
    "gru": GRUForecaster,
}


def create_model(model_name: str, **kwargs: Any) -> BaseNN:
    """Create a model instance by registered name.

    Args:
        model_name: Name of the model ('rnn', 'lstm', 'gru').
        **kwargs: Arguments passed to the model constructor
                 (e.g., input_size, hidden_size, num_layers, output_size, dropout).

    Returns:
        Instance of BaseNN model.
    """
    key = model_name.lower().strip()
    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Available models: {list(MODEL_REGISTRY.keys())}"
        )
    model_cls = MODEL_REGISTRY[key]
    return model_cls(**kwargs)
