from unittest.mock import MagicMock, patch

import pandas as pd
import torch

from credit_risk_agent.agent.tools.simulate_custom_scenario import simulate_custom_scenario


class TestSimulateCustomScenarioTool:
    @patch("credit_risk_agent.model.predictor.prepare_dataset")
    @patch("credit_risk_agent.agent.tools.simulate_custom_scenario.load_scaler_from_mlflow")
    @patch("credit_risk_agent.agent.tools.simulate_custom_scenario.load_and_preprocess_from_db")
    @patch("credit_risk_agent.agent.tools.simulate_custom_scenario.load_model_from_mlflow")
    def test_simulate_custom_scenario_success(
        self,
        mock_load_model: MagicMock,
        mock_load_data: MagicMock,
        mock_load_scaler: MagicMock,
        mock_prepare_dataset: MagicMock,
    ) -> None:
        """Verify successful simulation output when updating client features."""
        # 1. Arrange
        client_id = 15
        mock_df = pd.DataFrame({"client_id": [15], "limit_bal": [50000], "pay_0": [2]})
        mock_load_data.return_value = mock_df

        mock_scaler_instance = MagicMock()
        mock_scaler_instance.transform.side_effect = lambda df, cols: df
        mock_load_scaler.return_value = mock_scaler_instance

        dummy_seq = torch.zeros((6, 3))
        dummy_static = torch.zeros((14,))
        mock_prepare_dataset.return_value = [(dummy_seq, dummy_static)]

        mock_model_instance = MagicMock()
        # Logit 0.0 -> sigmoid = 0.5 (old_pd), Logit -1.0 -> sigmoid = 0.2689 (new_pd)
        mock_model_instance.side_effect = [torch.tensor([[0.0]]), torch.tensor([[-1.0]])]
        mock_load_model.return_value = mock_model_instance

        # 2. Act
        result = simulate_custom_scenario(client_id, {"limit_bal": 100000, "pay_0": 0})

        # 3. Assert
        assert "Результаты What-If симуляции для клиента id=15" in result
        assert "50.00%" in result
        assert "26.89%" in result
        assert "Снижение риска 🟢" in result
        assert "- limit_bal: 100000" in result

    @patch("credit_risk_agent.agent.tools.simulate_custom_scenario.load_scaler_from_mlflow")
    @patch("credit_risk_agent.agent.tools.simulate_custom_scenario.load_and_preprocess_from_db")
    @patch("credit_risk_agent.agent.tools.simulate_custom_scenario.load_model_from_mlflow")
    def test_simulate_custom_scenario_client_not_found(
        self,
        mock_load_model: MagicMock,
        mock_load_data: MagicMock,
        mock_load_scaler: MagicMock,
    ) -> None:
        """Verify error message when specified client_id is not in database."""
        client_id = 999
        mock_df = pd.DataFrame({"client_id": [1, 2]})
        mock_load_data.return_value = mock_df

        mock_model_instance = MagicMock()
        mock_load_model.return_value = mock_model_instance

        result = simulate_custom_scenario(client_id, {"limit_bal": 100000})
        assert result == f"Клиент с client_id = {client_id} не был найден в базе данных."

    @patch("credit_risk_agent.model.predictor.prepare_dataset")
    @patch("credit_risk_agent.agent.tools.simulate_custom_scenario.load_scaler_from_mlflow")
    @patch("credit_risk_agent.agent.tools.simulate_custom_scenario.load_and_preprocess_from_db")
    @patch("credit_risk_agent.agent.tools.simulate_custom_scenario.load_model_from_mlflow")
    def test_simulate_custom_scenario_unknown_keys(
        self,
        mock_load_model: MagicMock,
        mock_load_data: MagicMock,
        mock_load_scaler: MagicMock,
        mock_prepare_dataset: MagicMock,
    ) -> None:
        """Verify error message when all parameters in params are unknown columns."""
        client_id = 15
        mock_df = pd.DataFrame({"client_id": [15], "limit_bal": [50000]})
        mock_load_data.return_value = mock_df

        mock_scaler_instance = MagicMock()
        mock_scaler_instance.transform.side_effect = lambda df, cols: df
        mock_load_scaler.return_value = mock_scaler_instance

        dummy_seq = torch.zeros((6, 3))
        dummy_static = torch.zeros((14,))
        mock_prepare_dataset.return_value = [(dummy_seq, dummy_static)]

        mock_model_instance = MagicMock()
        mock_model_instance.return_value = torch.tensor([[0.0]])
        mock_load_model.return_value = mock_model_instance

        result = simulate_custom_scenario(client_id, {"non_existent_key": 123})
        assert "Ошибка: Ни один из переданных параметров" in result
