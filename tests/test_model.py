"""Tests for RNN, LSTM, and GRU forecasting models and factory."""
import pytest
import torch

from models.base import BaseNN
from models.factory import MODEL_REGISTRY, create_model
from models.gru import GRUForecaster
from models.lstm import LSTMForecaster
from models.rnn import RNNForecaster


@pytest.mark.parametrize("model_type,model_cls", [
    ("rnn", RNNForecaster),
    ("lstm", LSTMForecaster),
    ("gru", GRUForecaster),
])
def test_model_forward_shapes_and_inheritance(model_type, model_cls):
    """Verify that all recurrent forecasters inherit from BaseNN and produce correct shapes."""
    batch_size = 8
    seq_len = 24
    input_size = 1
    hidden_size = 32
    output_size = 1

    model = create_model(
        model_name=model_type,
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
    )

    assert isinstance(model, BaseNN)
    assert isinstance(model, model_cls)
    assert model.name == model_type

    x = torch.randn(batch_size, seq_len, input_size)
    output = model(x)

    assert output.shape == (batch_size, output_size)


@pytest.mark.parametrize("model_name", ["rnn", "lstm", "gru"])
def test_different_hidden_sizes_and_layers(model_name):
    """Verify models operate properly with varying hidden sizes and multi-layer stacks."""
    for hidden_size in [16, 64]:
        for num_layers in [1, 2]:
            model = create_model(
                model_name=model_name,
                input_size=1,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=1,
                dropout=0.2 if num_layers > 1 else 0.0,
            )
            x = torch.randn(4, 24, 1)
            output = model(x)
            assert output.shape == (4, 1)


@pytest.mark.parametrize("model_name", ["rnn", "lstm", "gru"])
def test_gradient_flow(model_name):
    """Verify that gradients flow back through every parameter during backprop."""
    model = create_model(model_name=model_name, input_size=1, hidden_size=16, output_size=1)
    x = torch.randn(4, 24, 1)
    y = torch.randn(4, 1)

    output = model(x)
    loss = torch.nn.MSELoss()(output, y)
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient computed for {model_name} parameter: {name}"
        assert not torch.all(param.grad == 0), f"Zero gradient for {model_name} parameter: {name}"


def test_factory_invalid_model_name():
    """Verify ValueError is raised for unregistered model names."""
    with pytest.raises(ValueError, match="Unknown model"):
        create_model("transformer_unknown")
