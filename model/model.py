from typing import cast

import torch
from torch import nn


class CreditDefaultPredictor(nn.Module):
    """
    GRU-based model for predicting credit default risk.

    Parameters
    ----------
    hidden_size : int, default=64
        The number of features in the GRU hidden state.
    num_layers : int, default=1
        Number of recurrent layers in the GRU.
    """

    def __init__(self, hidden_size: int = 64, num_layers: int = 1, static_size: int = 5, dropout_prob: float = 0.3):
        super().__init__()

        self.gru = nn.GRU(
            input_size=3,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc1 = nn.Linear(in_features=hidden_size + static_size, out_features=32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(in_features=32, out_features=1)

    def forward(self, x_seq: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Note: To optimize performance, this method does not perform runtime
        shape validation. The caller is responsible for ensuring the input
        tensor satisfies the expected dimensions.

        Parameters
        ----------
        x_seq : torch.Tensor
            Input tensor of shape (batch_size, seq_size, 3).

        x_static : torch.Tensor
            Input tensor of shape (batch_size, static_size).

        Returns
        -------
        torch.Tensor
            Raw prediction logits of shape (batch_size, 1).
        """
        gru_out, _ = self.gru(x_seq)
        last_hidden = gru_out[:, -1, :]

        combined = torch.cat([last_hidden, x_static], dim=1)
        x = self.fc1(combined)
        x = self.relu(x)
        x = self.dropout(x)
        return cast(torch.Tensor, self.fc2(x))
