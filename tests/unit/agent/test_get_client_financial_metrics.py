import sqlite3
from unittest.mock import MagicMock, patch

from credit_risk_agent.agent.tools import get_client_financial_metrics


class TestGetClientFinancialMetrics:
    def test_get_client_metrics_success(self) -> None:
        """Verify formatted string output of client financial metrics calculation."""
        # Arrange
        mock_conn = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        # First query: client row (limit_bal, age, default)
        # Second query: metrics (avg_bill, max_bill, avg_pay, sum_pay, sum_bill, max_delay_status, delay_months_count)
        mock_cursor.fetchone.side_effect = [
            (100000.0, 30, 0),
            (10000.0, 20000.0, 5000.0, 30000.0, 60000.0, 2, 3),
        ]

        # Act
        with patch("sqlite3.connect", return_value=mock_conn):
            result = get_client_financial_metrics(101)

        # Assert
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
        # Arrange
        mock_conn = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None

        # Act
        with patch("sqlite3.connect", return_value=mock_conn):
            result = get_client_financial_metrics(999)

        # Assert
        assert result == "Клиент с client_id = 999 не был найден в базе данных."

    def test_get_client_metrics_no_payment_history(self) -> None:
        """Verify message when client exists but payment history is absent."""
        # Arrange
        mock_conn = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.side_effect = [
            (50000.0, 25, 0),
            (None, None, None, None, None, None, None),
        ]

        # Act
        with patch("sqlite3.connect", return_value=mock_conn):
            result = get_client_financial_metrics(102)

        # Assert
        assert result == "Для клиента с client_id=102 отсутствует история платежей."

    def test_get_client_metrics_zero_division_safety(self) -> None:
        """Verify zero division safety when credit limit or total bills are zero."""
        # Arrange
        mock_conn = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.side_effect = [
            (0.0, 40, 1),
            (0.0, 0.0, 0.0, 0.0, 0.0, 0, 0),
        ]

        # Act
        with patch("sqlite3.connect", return_value=mock_conn):
            result = get_client_financial_metrics(103)

        # Assert
        assert "Средняя утилизация лимита: 0.0%" in result
        assert "- Коэффициент покрытия выставляемых счетов (Repayment Rate): 0.0%" in result

    def test_get_client_metrics_database_error(self) -> None:
        """Verify database error handling when SQLite raises an exception."""
        # Arrange
        mock_conn = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = sqlite3.Error("no such table: payment_history")
        mock_conn.cursor.return_value = mock_cursor

        # Act
        with patch("sqlite3.connect", return_value=mock_conn):
            result = get_client_financial_metrics(104)

        # Assert
        assert "Ошибка SQL: no such table: payment_history" in result
