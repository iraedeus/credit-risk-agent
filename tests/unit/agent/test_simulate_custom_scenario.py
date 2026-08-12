from unittest.mock import MagicMock, patch

import pandas as pd

from credit_risk_agent.agent.tools.simulate_custom_scenario import simulate_custom_scenario
from credit_risk_agent.services.data_service.exceptions import DataServiceHTTPError


class TestSimulateCustomScenarioTool:
    @patch("credit_risk_agent.agent.tools.simulate_custom_scenario.client_full_info_to_df")
    @patch("credit_risk_agent.agent.tools.simulate_custom_scenario.load_model_from_mlflow")
    @patch("credit_risk_agent.agent.tools.simulate_custom_scenario.load_scaler_from_mlflow")
    @patch("credit_risk_agent.agent.tools.simulate_custom_scenario.CreditRiskPredictor")
    @patch("credit_risk_agent.agent.tools.simulate_custom_scenario.get_data_service_client")
    def test_simulate_custom_scenario_success(
        self, provider, predictor, mock_load_model: MagicMock, mock_load_scaler: MagicMock, mock_to_df
    ) -> None:
        """Verify successful simulation output when updating client features."""
        mock_df = pd.DataFrame({"client_id": [15], "limit_bal": [50000], "pay_0": [2]})
        mock_to_df.return_value = mock_df

        mock_client = MagicMock()
        mock_client.get_client.return_value = MagicMock()
        provider.return_value = mock_client

        predictor.return_value.predict_pd.side_effect = [0.5, 0.2]

        result = simulate_custom_scenario(15, {"limit_bal": 100000, "pay_0": 0})

        assert "Результаты What-If симуляции для клиента id=15" in result
        assert "Вероятность дефолта (PD): 50.00%" in result
        assert "Категория риска: Низкий риск" in result
        assert "- limit_bal: 100000" in result
        assert "- pay_0: 0" in result
        assert "Симулированный PD: 20.00%" in result
        assert "Изменение PD: -30.00% п.п." in result
        assert "Динамика риска: Снижение риска" in result

    @patch("credit_risk_agent.agent.tools.simulate_custom_scenario.client_full_info_to_df")
    @patch("credit_risk_agent.agent.tools.simulate_custom_scenario.get_data_service_client")
    def test_simulate_custom_scenario_unknown_keys(
        self,
        provider: MagicMock,
        mock_to_df: MagicMock,
    ) -> None:
        """Verify error message when all parameters in params are unknown columns."""
        mock_df = pd.DataFrame({"client_id": [15], "limit_bal": [50000]})
        mock_to_df.return_value = mock_df

        mock_client = MagicMock()
        mock_client.get_client.return_value = MagicMock()
        provider.return_value = mock_client

        result = simulate_custom_scenario(15, {"wrong_param": 3})
        assert "Ошибка: Ни один из переданных параметров" in result

    @patch("credit_risk_agent.agent.tools.simulate_custom_scenario.get_data_service_client")
    def test_simulate_custom_scenario_client_not_found(self, provider) -> None:
        """Verify error message when specified client_id is not in database."""
        mock_client = MagicMock()
        mock_client.get_client.return_value = None
        provider.return_value = mock_client

        result = simulate_custom_scenario(999, {"limit_bal": 100})

        assert result == "Клиент с client_id = 999 не был найден в базе данных."

    @patch("credit_risk_agent.agent.tools.simulate_custom_scenario.get_data_service_client")
    def test_simulate_custom_scenario_service_error(self, provider) -> None:
        """Verify error message when all parameters in params are unknown columns."""
        mock_client = MagicMock()
        mock_client.get_client.side_effect = DataServiceHTTPError(500, "Internal Service Error")
        provider.return_value = mock_client

        result = simulate_custom_scenario(1, {"limit_bal": 100})

        assert "Ошибка Data Service:" in result
