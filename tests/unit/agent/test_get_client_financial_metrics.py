from unittest.mock import MagicMock, patch

from credit_risk_agent.agent.tools import get_client_financial_metrics
from credit_risk_agent.services.data_service.exceptions import DataServiceHTTPError
from credit_risk_agent.services.data_service.schemas import ClientFinancialMetrics


class TestGetClientFinancialMetrics:
    def test_get_client_metrics_success(self) -> None:
        """Verify formatted string output of client financial metrics calculation."""

        mock_metrics = ClientFinancialMetrics(
            client_id=101,
            limit_bal=100000.0,
            avg_bill=10000.0,
            avg_pay=5000.0,
            avg_utilization=10.0,
            max_utilization=20.0,
            repayment_rate=50.0,
            max_delay_status=2,
            delay_months_count=3,
        )

        mock_client = MagicMock()
        mock_client.get_client_metrics.return_value = mock_metrics

        with patch(
            "credit_risk_agent.agent.tools.get_client_financial_metrics.get_data_service_client",
            return_value=mock_client,
        ):
            result = get_client_financial_metrics(101)

        assert "Финансовые метрики клиента id=101:" in result
        assert "Кредитный лимит: 100,000.00" in result
        assert "Средний ежемесячный счет (bill_amt): 10,000.00" in result
        assert "Средняя утилизация лимита: 10.0%" in result
        assert "Максимальная утилизация лимита: 20.0%" in result
        assert "Средний ежемесячный платеж (pay_amt): 5,000.00" in result
        assert "- Коэффициент покрытия выставляемых счетов (Repayment Rate): 50.0%" in result
        assert "Максимальный статус просрочки за 6 мес.: 2" in result
        assert "Количество месяцев с просрочкой: 3 из 6" in result

    def test_get_client_metrics_client_not_found(self) -> None:
        """Verify error message when client ID is not found in database."""

        mock_client = MagicMock()
        mock_client.get_client_metrics.return_value = None

        with patch(
            "credit_risk_agent.agent.tools.get_client_financial_metrics.get_data_service_client",
            return_value=mock_client,
        ):
            result = get_client_financial_metrics(999)

        # Assert
        assert result == "Клиент с client_id = 999 не был найден в базе данных."

    def test_get_client_metrics_service_error(self) -> None:
        """Verify database error handling when SQLite raises an exception."""

        mock_client = MagicMock()
        mock_client.get_client_metrics.side_effect = DataServiceHTTPError(500, "Internal Service Error")

        with patch(
            "credit_risk_agent.agent.tools.get_client_financial_metrics.get_data_service_client",
            return_value=mock_client,
        ):
            result = get_client_financial_metrics(1)

        assert "Ошибка Data Service:" in result
