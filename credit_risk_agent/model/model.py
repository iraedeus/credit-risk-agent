import pandas as pd
import torch

from credit_risk_agent.config import SCALER_COLS
from credit_risk_agent.data.standard_scaler import StandardScaler
from credit_risk_agent.model.dataset import prepare_dataset
from credit_risk_agent.model.predictor import CreditDefaultPredictor


class CreditRiskModel:
    def __init__(self, predictor: CreditDefaultPredictor, scaler: StandardScaler) -> None:
        self.predictor = predictor
        self.scaler = scaler
        self.predictor.eval()

    def predict_pd(self, client_df: pd.DataFrame) -> float:
        scaled_df = self.scaler.transform(client_df, SCALER_COLS)
        dataset = prepare_dataset(scaled_df)

        x_seq = dataset[0][0].unsqueeze(0)
        x_stat = dataset[0][1].unsqueeze(0)

        with torch.no_grad():
            logits = self.predictor(x_seq, x_stat)
            return float(torch.sigmoid(logits).item())
