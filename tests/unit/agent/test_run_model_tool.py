from unittest.mock import MagicMock, patch

from credit_risk_agent.agent import run_model
from credit_risk_agent.services.data_service.exceptions import DataServiceHTTPError


class TestRunModelTool:
    """Test suite for run_model agent evaluation tool."""

    @patch("credit_risk_agent.agent.tools.run_model.load_scaler_from_registry")
    @patch("credit_risk_agent.agent.tools.run_model.load_model_from_registry")
    @patch("credit_risk_agent.agent.tools.run_model.CreditRiskPredictor")
    @patch("credit_risk_agent.agent.tools.run_model.get_data_service_client")
    def test_run_model_success(
        self,
        provider: MagicMock,
        predictor: MagicMock,
        mock_load_model: MagicMock,
        mock_load_scaler: MagicMock,
    ) -> None:
        """Verify successful scoring for an existing client ID via DataServiceClient."""

        mock_full_info = MagicMock()
        mock_client = MagicMock()
        mock_client.get_client.return_value = mock_full_info

        provider.return_value = mock_client
        predictor.return_value.predict_pd.return_value = 0.5

        result = run_model(1)
        assert result == "Модель на клиенте с id=1 выдала результат равный 0.5000."

    @patch("credit_risk_agent.agent.tools.run_model.get_data_service_client")
    def test_run_model_client_not_found(self, provider: MagicMock) -> None:
        """Verify tool response when client_id is not found in DataServiceClient."""

        mock_client = MagicMock()
        mock_client.get_client.return_value = None
        provider.return_value = mock_client

        result = run_model(999)

        assert result == "Клиент с client_id = 999 не был найден в базе данных."

    @patch("credit_risk_agent.agent.tools.run_model.get_data_service_client")
    def test_run_model_service_error(self, provider: MagicMock) -> None:
        """Verify error handling when DataServiceClient raises HTTP error."""

        mock_client = MagicMock()
        mock_client.get_client.side_effect = DataServiceHTTPError(500, "Internal Service Error")
        provider.return_value = mock_client

        result = run_model(1)

        assert "Ошибка Data Service:" in result
