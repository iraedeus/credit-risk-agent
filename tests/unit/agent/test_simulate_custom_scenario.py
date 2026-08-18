from unittest.mock import MagicMock, patch

import pytest

from credit_risk_agent.agent.tools.simulate_custom_scenario import (
    _get_risk_level,
    _update_profile_history_params,
    simulate_custom_scenario,
)
from credit_risk_agent.schemas.client_schemas import (
    ClientFinancialMetrics,
    ClientPaymentHistory,
    ClientProfile,
)
from credit_risk_agent.schemas.enums import Education, Marriage, Sex
from credit_risk_agent.services.data_service.exceptions import DataServiceHTTPError
from credit_risk_agent.services.data_service.schemas import ClientFullInfo
from credit_risk_agent.services.ml_service.exceptions import MLServiceHTTPError
from credit_risk_agent.services.ml_service.schemas import ClientProfileHistory, PredictionResponse


@pytest.fixture
def sample_client_full_info() -> ClientFullInfo:
    """Fixture providing a standard 6-month client profile, payment history, and metrics."""
    return ClientFullInfo(
        profile=ClientProfile(
            client_id=15,
            limit_bal=50000.0,
            age=30,
            sex=Sex.MALE,
            education=Education.UNIVERSITY,
            marriage=Marriage.SINGLE,
        ),
        history=[
            ClientPaymentHistory(client_id=15, month=m, pay_status=1, bill_amt=10000.0, pay_amt=2000.0)
            for m in range(1, 7)
        ],
        metrics=ClientFinancialMetrics(
            client_id=15,
            limit_bal=50000.0,
            avg_bill=10000.0,
            avg_utilization=0.2,
            max_utilization=0.2,
            avg_pay=2000.0,
            repayment_rate=0.2,
            max_delay_status=1,
            delay_months_count=6,
        ),
    )


class TestInternalHelpers:
    """Unit tests for internal helper functions."""

    @pytest.mark.parametrize(
        ("pd_val", "expected_risk"),
        [
            (0.10, "Низкий риск"),
            (0.349, "Низкий риск"),
            (0.35, "Умеренный/Средний риск"),
            (0.549, "Умеренный/Средний риск"),
            (0.55, "Высокий риск"),
            (0.95, "Высокий риск"),
        ],
    )
    def test_get_risk_level(self, pd_val: float, expected_risk: str) -> None:
        """Verify risk level mapping for boundary probability values."""
        assert _get_risk_level(pd_val) == expected_risk

    def test_update_profile_history_params_updates_profile_and_month_one_only(
        self, sample_client_full_info: ClientFullInfo
    ) -> None:
        """Verify feature modifications apply to profile and only month 1 history."""
        base_history = ClientProfileHistory(
            profile=sample_client_full_info.profile,
            history=sample_client_full_info.history,
        )
        updated = _update_profile_history_params(
            base_history,
            {
                "limit_bal": 150000.0,
                "age": 35,
                "sex": Sex.FEMALE,
                "education": Education.GRADUATE_SCHOOL,
                "marriage": Marriage.MARRIED,
                "pay_status": -1,
                "bill_amt": 500.0,
                "pay_amt": 500.0,
            },
        )

        assert updated.profile.limit_bal == 150000.0
        assert updated.profile.age == 35
        assert updated.profile.sex == Sex.FEMALE
        assert updated.profile.education == Education.GRADUATE_SCHOOL
        assert updated.profile.marriage == Marriage.MARRIED

        # Month 1 should reflect the updated values
        assert updated.history[0].pay_status == -1
        assert updated.history[0].bill_amt == 500.0
        assert updated.history[0].pay_amt == 500.0

        # Months 2 to 6 must remain intact
        for h in updated.history[1:]:
            assert h.pay_status == 1
            assert h.bill_amt == 10000.0
            assert h.pay_amt == 2000.0


class TestSimulateCustomScenarioTool:
    """Test suite for simulate_custom_scenario agent evaluation tool."""

    def test_simulate_custom_scenario_invalid_parameters_only(self) -> None:
        """Verify error message when all parameters in params are unknown columns."""
        result = simulate_custom_scenario(15, {"unknown_param": 123})
        assert "Были переданы некорректные параметры" in result

    @patch("credit_risk_agent.agent.tools.simulate_custom_scenario.get_data_service_client")
    def test_simulate_custom_scenario_client_not_found(self, mock_data_provider: MagicMock) -> None:
        """Verify error message when specified client_id is not in DataServiceClient."""
        data_client = MagicMock()
        data_client.get_client.return_value = None
        mock_data_provider.return_value = data_client

        result = simulate_custom_scenario(999, {"limit_bal": 100000})
        assert result == "Клиент с client_id = 999 не был найден в базе данных."

    @patch("credit_risk_agent.agent.tools.simulate_custom_scenario.get_ml_service_client")
    @patch("credit_risk_agent.agent.tools.simulate_custom_scenario.get_data_service_client")
    def test_simulate_custom_scenario_risk_decrease(
        self,
        mock_data_provider: MagicMock,
        mock_ml_provider: MagicMock,
        sample_client_full_info: ClientFullInfo,
    ) -> None:
        """Verify simulation output for decreasing default risk with ignored keys warning."""
        data_client = MagicMock()
        data_client.get_client.return_value = sample_client_full_info
        mock_data_provider.return_value = data_client

        ml_client = MagicMock()
        ml_client.predict.side_effect = [
            PredictionResponse(default_probability=0.50),
            PredictionResponse(default_probability=0.20),
        ]
        mock_ml_provider.return_value = ml_client

        result = simulate_custom_scenario(15, {"limit_bal": 100000, "pay_status": 0, "unknown_field": 42})

        assert "Результаты What-If симуляции для клиента id=15" in result
        assert "Вероятность дефолта (PD): 50.00%" in result
        assert "Категория риска: Умеренный/Средний риск" in result
        assert "- limit_bal: 100000" in result
        assert "- pay_status: 0" in result
        assert "⚠️ Игнорируемые неизвестные ключи: ['unknown_field']" in result
        assert "Симулированный PD: 20.00%" in result
        assert "Категория риска: Низкий риск" in result
        assert "Изменение PD: -30.00% п.п." in result
        assert "Динамика риска: Снижение риска" in result

    @patch("credit_risk_agent.agent.tools.simulate_custom_scenario.get_ml_service_client")
    @patch("credit_risk_agent.agent.tools.simulate_custom_scenario.get_data_service_client")
    def test_simulate_custom_scenario_risk_increase(
        self,
        mock_data_provider: MagicMock,
        mock_ml_provider: MagicMock,
        sample_client_full_info: ClientFullInfo,
    ) -> None:
        """Verify simulation output for increasing default risk with positive delta sign."""
        data_client = MagicMock()
        data_client.get_client.return_value = sample_client_full_info
        mock_data_provider.return_value = data_client

        ml_client = MagicMock()
        ml_client.predict.side_effect = [
            PredictionResponse(default_probability=0.20),
            PredictionResponse(default_probability=0.60),
        ]
        mock_ml_provider.return_value = ml_client

        result = simulate_custom_scenario(15, {"limit_bal": 20000, "pay_status": 2})

        assert "Изменение PD: +40.00% п.п." in result
        assert "Динамика риска: Увеличение риска" in result

    @patch("credit_risk_agent.agent.tools.simulate_custom_scenario.get_ml_service_client")
    @patch("credit_risk_agent.agent.tools.simulate_custom_scenario.get_data_service_client")
    def test_simulate_custom_scenario_no_risk_change(
        self,
        mock_data_provider: MagicMock,
        mock_ml_provider: MagicMock,
        sample_client_full_info: ClientFullInfo,
    ) -> None:
        """Verify simulation output when baseline and simulated PD are identical."""
        data_client = MagicMock()
        data_client.get_client.return_value = sample_client_full_info
        mock_data_provider.return_value = data_client

        ml_client = MagicMock()
        ml_client.predict.side_effect = [
            PredictionResponse(default_probability=0.30),
            PredictionResponse(default_probability=0.30),
        ]
        mock_ml_provider.return_value = ml_client

        result = simulate_custom_scenario(15, {"limit_bal": 50000})

        assert "Изменение PD: 0.00% п.п." in result
        assert "Динамика риска: Без изменений" in result

    @patch("credit_risk_agent.agent.tools.simulate_custom_scenario.get_data_service_client")
    def test_simulate_custom_scenario_data_service_error(self, mock_data_provider: MagicMock) -> None:
        """Verify graceful error reporting when DataServiceClient fails."""
        data_client = MagicMock()
        data_client.get_client.side_effect = DataServiceHTTPError(500, "Internal Server Error")
        mock_data_provider.return_value = data_client

        result = simulate_custom_scenario(15, {"limit_bal": 100000})
        assert "Ошибка Data Service: [500] Internal Server Error" in result

    @patch("credit_risk_agent.agent.tools.simulate_custom_scenario.get_ml_service_client")
    @patch("credit_risk_agent.agent.tools.simulate_custom_scenario.get_data_service_client")
    def test_simulate_custom_scenario_ml_service_error(
        self,
        mock_data_provider: MagicMock,
        mock_ml_provider: MagicMock,
        sample_client_full_info: ClientFullInfo,
    ) -> None:
        """Verify graceful error reporting when MLServiceClient fails."""
        data_client = MagicMock()
        data_client.get_client.return_value = sample_client_full_info
        mock_data_provider.return_value = data_client

        ml_client = MagicMock()
        ml_client.predict.side_effect = MLServiceHTTPError(503, "ML Service Unavailable")
        mock_ml_provider.return_value = ml_client

        result = simulate_custom_scenario(15, {"limit_bal": 100000})
        assert "Ошибка ML Service: [503] ML Service Unavailable" in result
