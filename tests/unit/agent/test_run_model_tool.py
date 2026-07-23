from unittest.mock import MagicMock, patch

import pandas as pd
import torch

from credit_risk_agent.agent import run_model


class TestRunModelTool:
    @patch("credit_risk_agent.agent.tools.run_model.prepare_dataset")
    @patch("credit_risk_agent.agent.tools.run_model.StandardScaler")
    @patch("credit_risk_agent.agent.tools.run_model.load_and_preprocess_from_db")
    @patch("credit_risk_agent.agent.tools.run_model.CreditDefaultPredictor")
    @patch("torch.load")
    def test_run_model_success(
        self,
        mock_torch_load: MagicMock,
        mock_predictor_cls: MagicMock,
        mock_load_data: MagicMock,
        mock_scaler_cls: MagicMock,
        mock_prepare_dataset: MagicMock,
    ) -> None:
        """Verify successful scoring for an existing client ID."""
        # 1. Arrange
        client_id = 15
        mock_df = pd.DataFrame({"client_id": [15, 20]})
        mock_load_data.return_value = mock_df
        mock_scaler_instance = MagicMock()
        mock_scaler_instance.transform.return_value = mock_df
        mock_scaler_cls.load.return_value = mock_scaler_instance

        dummy_seq = torch.zeros((6, 3))
        dummy_static = torch.zeros((14,))
        mock_prepare_dataset.return_value = [(dummy_seq, dummy_static)]

        mock_model_instance = MagicMock()
        mock_model_instance.return_value = torch.tensor([[0.0]])
        mock_predictor_cls.return_value = mock_model_instance

        # 2. Act
        result = run_model(client_id)

        # 3. Assert
        assert result == "Модель на клиенте с id=15 выдала результат равный 0.5000."
        mock_prepare_dataset.assert_called_once()
        mock_model_instance.assert_called_once()

    @patch("credit_risk_agent.agent.tools.run_model.StandardScaler")
    @patch("credit_risk_agent.agent.tools.run_model.load_and_preprocess_from_db")
    @patch("credit_risk_agent.agent.tools.run_model.CreditDefaultPredictor")
    @patch("torch.load")
    def test_run_model_client_not_found(
        self,
        mock_torch_load: MagicMock,
        mock_predictor_cls: MagicMock,
        mock_load_data: MagicMock,
        mock_scaler_cls: MagicMock,
    ) -> None:
        """Verify tool response when client_id is not found in database."""
        # 1. Arrange
        client_id = 999999
        mock_df = pd.DataFrame({"client_id": [1, 2, 3]})
        mock_load_data.return_value = mock_df
        mock_scaler_instance = MagicMock()
        mock_scaler_instance.transform.return_value = mock_df
        mock_scaler_cls.load.return_value = mock_scaler_instance

        # 2. Act
        result = run_model(client_id)

        # 3. Assert
        assert result == f"Клиент с id={client_id} не был найден в базе."

    @patch("credit_risk_agent.agent.tools.run_model.prepare_dataset")
    @patch("credit_risk_agent.agent.tools.run_model.StandardScaler")
    @patch("credit_risk_agent.agent.tools.run_model.load_and_preprocess_from_db")
    @patch("credit_risk_agent.agent.tools.run_model.CreditDefaultPredictor")
    @patch("torch.load")
    def test_run_model_formatting_precision(
        self,
        mock_torch_load: MagicMock,
        mock_predictor_cls: MagicMock,
        mock_load_data: MagicMock,
        mock_scaler_cls: MagicMock,
        mock_prepare_dataset: MagicMock,
    ) -> None:
        """Verify model prediction score formatting precision (4 decimal places)."""

        # 1. Arrange
        client_id = 42
        mock_df = pd.DataFrame({"client_id": [42]})
        mock_load_data.return_value = mock_df
        mock_scaler_instance = MagicMock()
        mock_scaler_instance.transform.return_value = mock_df
        mock_scaler_cls.load.return_value = mock_scaler_instance

        dummy_seq = torch.zeros((6, 3))
        dummy_static = torch.zeros((14,))
        mock_prepare_dataset.return_value = [(dummy_seq, dummy_static)]

        mock_model_instance = MagicMock()
        # Logit 2.1972246 -> sigmoid = 0.9000
        mock_model_instance.return_value = torch.tensor([[2.1972246]])
        mock_predictor_cls.return_value = mock_model_instance

        # 2. Act
        result = run_model(client_id)

        # 3. Assert
        assert result == "Модель на клиенте с id=42 выдала результат равный 0.9000."
