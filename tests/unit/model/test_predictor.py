from unittest.mock import MagicMock, patch

import pandas as pd
import torch

from credit_risk_agent.model.predictor import CreditRiskPredictor


class TestCreditRiskPredictor:
    @patch("credit_risk_agent.model.predictor.prepare_dataset")
    def test_predict_pd_calculates_probability(self, mock_prepare_dataset: MagicMock) -> None:
        """Verify predict_pd scales client DataFrame, prepares dataset, and returns sigmoid score."""
        # Arrange
        mock_model = MagicMock()
        mock_model.return_value = torch.tensor([[0.0]])
        mock_scaler = MagicMock()
        mock_df = pd.DataFrame({"client_id": [15]})
        mock_scaler.transform.return_value = mock_df

        dummy_seq = torch.zeros((6, 3))
        dummy_static = torch.zeros((14,))
        mock_prepare_dataset.return_value = [(dummy_seq, dummy_static)]

        predictor = CreditRiskPredictor(mock_model, mock_scaler)

        # Act
        score = predictor.predict_pd(mock_df)

        # Assert
        assert score == 0.5000
        mock_scaler.transform.assert_called_once()
        mock_prepare_dataset.assert_called_once()
        mock_model.assert_called_once()
