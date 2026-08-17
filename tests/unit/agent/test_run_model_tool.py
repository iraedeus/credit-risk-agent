from unittest.mock import MagicMock, patch

import pytest

from credit_risk_agent.agent import run_model
from credit_risk_agent.schemas.client_schemas import ClientFinancialMetrics, ClientPaymentHistory, ClientProfile
from credit_risk_agent.schemas.enums import Education, Marriage, Sex
from credit_risk_agent.services.data_service.exceptions import DataServiceHTTPError
from credit_risk_agent.services.data_service.schemas import ClientFullInfo
from credit_risk_agent.services.ml_service.exceptions import MLServiceHTTPError
from credit_risk_agent.services.ml_service.schemas import PredictionResponse


@pytest.fixture
def client_full_info() -> ClientFullInfo:
    """Build a fully-populated ClientFullInfo with a consistent client_id of 1."""
    return ClientFullInfo(
        profile=ClientProfile(
            client_id=1,
            limit_bal=300000.0,
            age=35,
            sex=Sex.MALE,
            education=Education.UNIVERSITY,
            marriage=Marriage.MARRIED,
        ),
        history=[
            ClientPaymentHistory(client_id=1, month=month, pay_status=-1, bill_amt=55000.0, pay_amt=18000.0)
            for month in range(1, 7)
        ],
        metrics=ClientFinancialMetrics(
            client_id=1,
            limit_bal=300000.0,
            avg_bill=55000.0,
            avg_utilization=0.5,
            max_utilization=0.5,
            avg_pay=18000.0,
            repayment_rate=0.3,
            max_delay_status=-1,
            delay_months_count=0,
        ),
    )


class TestRunModelTool:
    """Test suite for run_model agent evaluation tool."""

    @patch("credit_risk_agent.agent.tools.run_model.get_ml_service_client")
    @patch("credit_risk_agent.agent.tools.run_model.get_data_service_client")
    def test_run_model_success(
        self,
        data_client_provider: MagicMock,
        ml_client_provider: MagicMock,
        client_full_info: ClientFullInfo,
    ) -> None:
        """Verify successful scoring for an existing client ID via DataServiceClient."""

        data_mock = MagicMock()
        ml_mock = MagicMock()
        data_mock.get_client.return_value = client_full_info
        ml_mock.predict.return_value = PredictionResponse(default_probability=0.5)

        data_client_provider.return_value = data_mock
        ml_client_provider.return_value = ml_mock

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
    def test_run_model_data_service_error(self, provider: MagicMock) -> None:
        """Verify error handling when DataServiceClient raises HTTP error."""

        mock_client = MagicMock()
        mock_client.get_client.side_effect = DataServiceHTTPError(500, "Internal Service Error")
        provider.return_value = mock_client

        result = run_model(1)

        assert "Ошибка Data Service:" in result

    @patch("credit_risk_agent.agent.tools.run_model.get_ml_service_client")
    @patch("credit_risk_agent.agent.tools.run_model.get_data_service_client")
    def test_run_model_ml_service_error(
        self,
        data_client_provider: MagicMock,
        ml_client_provider: MagicMock,
        client_full_info: ClientFullInfo,
    ) -> None:
        """Verify error handling when ML service raises HTTP error."""

        data_mock = MagicMock()
        ml_mock = MagicMock()
        data_mock.get_client.return_value = client_full_info
        ml_mock.predict.side_effect = MLServiceHTTPError(500, "Internal Service Error")

        data_client_provider.return_value = data_mock
        ml_client_provider.return_value = ml_mock

        result = run_model(1)

        assert "Ошибка ML Service:" in result
