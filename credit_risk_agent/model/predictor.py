"""
High-level domain predictor wrapper combining model evaluation and feature scaling.
"""

import pandas as pd
import torch

from credit_risk_agent.config import SCALER_COLS
from credit_risk_agent.data.standard_scaler import StandardScaler
from credit_risk_agent.model.dataset import prepare_dataset
from credit_risk_agent.model.model import CreditDefaultModel


class CreditRiskPredictor:
    """
    High-level domain predictor encapsulating the PyTorch CreditDefaultModel and StandardScaler.

    Parameters
    ----------
    model : CreditDefaultModel
        Pre-trained PyTorch credit default neural network.
    scaler : StandardScaler
        Fitted StandardScaler instance for normalizing numerical features.
    """

    def __init__(self, model: CreditDefaultModel, scaler: StandardScaler) -> None:
        self.model = model
        self.scaler = scaler
        self.model.eval()

    def predict_pd(self, client_df: pd.DataFrame) -> float:
        """
        Scale client features and compute credit default probability (PD).

        Parameters
        ----------
        client_df : pd.DataFrame
            Unscaled raw client feature DataFrame.

        Returns
        -------
        float
            Probability of default in range [0.0, 1.0].
        """
        scaled_df = self.scaler.transform(client_df, SCALER_COLS)
        dataset = prepare_dataset(scaled_df)

        x_seq = dataset[0][0].unsqueeze(0)
        x_stat = dataset[0][1].unsqueeze(0)

        with torch.no_grad():
            logits = self.model(x_seq, x_stat)
            return float(torch.sigmoid(logits).item())
