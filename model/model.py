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

    def __init__(self, hidden_size: int = 64, num_layers: int = 1):
        super().__init__()

        self.gru = nn.GRU(
            input_size=3,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Note: To optimize performance, this method does not perform runtime
        shape validation. The caller is responsible for ensuring the input
        tensor satisfies the expected dimensions.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, 3).

        Returns
        -------
        torch.Tensor
            Raw prediction logits of shape (batch_size, 1).
        """
        output, _ = self.gru(x)
        return cast(torch.Tensor, self.fc(output[:, -1, :]))
