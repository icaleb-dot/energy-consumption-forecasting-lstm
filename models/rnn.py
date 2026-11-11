"""Simple RNN forecasting model inheriting from BaseNN."""
import torch
import torch.nn as nn

from models.base import BaseNN


class RNNForecaster(BaseNN):
    """Simple Recurrent Neural Network (RNN) forecaster."""

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 32,
        num_layers: int = 1,
        output_size: int = 1,
        dropout: float = 0.0,
        name: str = "rnn",
    ) -> None:
        super().__init__(name=name)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout if num_layers > 1 else 0.0

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=self.dropout,
        )
        self.fc = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            Output prediction tensor of shape (batch_size, output_size).
        """
        output, _ = self.rnn(x)
        # Extract last time step: (batch_size, hidden_size)
        last_hidden = output[:, -1, :]
        return self.fc(last_hidden)
