from pathlib import Path
from unittest.mock import MagicMock, patch

import mlflow.exceptions
import pytest

from credit_risk_agent.config import BEST_MODEL_ALIAS, BEST_MODEL_NAME
from credit_risk_agent.model.loader import load_model_from_mlflow, load_scaler_from_mlflow


class TestLoadScalerFromMLflow:
    @patch("credit_risk_agent.model.loader.StandardScaler.load")
    @patch("credit_risk_agent.model.loader.mlflow.artifacts.download_artifacts")
    def test_load_scaler_with_run_id_success(self, mock_download: MagicMock, mock_scaler_load: MagicMock) -> None:
        """Verify downloading and loading scaler when run_id is explicitly provided."""
        # Arrange
        mock_download.return_value = "/tmp/dummy_scaler.json"
        mock_scaler_instance = MagicMock()
        mock_scaler_load.return_value = mock_scaler_instance

        # Act
        result = load_scaler_from_mlflow(run_id="run_123")

        # Assert
        mock_download.assert_called_once_with(run_id="run_123", artifact_path="preprocessing/scaler.json")
        mock_scaler_load.assert_called_once_with(Path("/tmp/dummy_scaler.json"))
        assert result == mock_scaler_instance

    @patch("credit_risk_agent.model.loader.StandardScaler.load")
    @patch("credit_risk_agent.model.loader.mlflow.artifacts.download_artifacts")
    @patch("credit_risk_agent.model.loader.mlflow.MlflowClient")
    def test_load_scaler_without_run_id_resolves_champion(
        self, mock_client_cls: MagicMock, mock_download: MagicMock, mock_scaler_load: MagicMock
    ) -> None:
        """Verify resolving champion run_id when run_id is None."""
        # Arrange
        mock_client = MagicMock()
        mock_champion = MagicMock()
        mock_champion.run_id = "champion_run_456"
        mock_client.get_model_version_by_alias.return_value = mock_champion
        mock_client_cls.return_value = mock_client

        mock_download.return_value = "/tmp/champion_scaler.json"
        mock_scaler_instance = MagicMock()
        mock_scaler_load.return_value = mock_scaler_instance

        # Act
        result = load_scaler_from_mlflow(run_id=None)

        # Assert
        mock_client.get_model_version_by_alias.assert_called_once_with(BEST_MODEL_NAME, BEST_MODEL_ALIAS)
        mock_download.assert_called_once_with(run_id="champion_run_456", artifact_path="preprocessing/scaler.json")
        assert result == mock_scaler_instance

    @patch("credit_risk_agent.model.loader.mlflow.MlflowClient")
    def test_load_scaler_no_champion_raises_runtime_error(self, mock_client_cls: MagicMock) -> None:
        """Verify RuntimeError is raised when no champion exists and run_id is None."""
        # Arrange
        mock_client = MagicMock()
        mock_client.get_model_version_by_alias.side_effect = mlflow.exceptions.MlflowException("Not found")
        mock_client_cls.return_value = mock_client

        # Act & Assert
        with pytest.raises(RuntimeError, match="Чемпионская модель еще не назначена"):
            load_scaler_from_mlflow(run_id=None)


class TestLoadModelFromMLflow:
    @patch("credit_risk_agent.model.loader.mlflow.pytorch.load_model")
    def test_load_model_with_run_id_success(self, mock_load_model: MagicMock) -> None:
        """Verify loading PyTorch model when run_id is explicitly provided."""
        # Arrange
        mock_model_instance = MagicMock()
        mock_load_model.return_value = mock_model_instance

        # Act
        result = load_model_from_mlflow(run_id="run_123")

        # Assert
        mock_load_model.assert_called_once_with("runs:/run_123/model")
        assert result == mock_model_instance

    @patch("credit_risk_agent.model.loader.mlflow.pytorch.load_model")
    def test_load_model_without_run_id_loads_champion(self, mock_load_model: MagicMock) -> None:
        """Verify loading champion model from Model Registry when run_id is None."""
        # Arrange
        mock_model_instance = MagicMock()
        mock_load_model.return_value = mock_model_instance

        # Act
        result = load_model_from_mlflow(run_id=None)

        # Assert
        expected_uri = f"models:/{BEST_MODEL_NAME}@{BEST_MODEL_ALIAS}"
        mock_load_model.assert_called_once_with(expected_uri)
        assert result == mock_model_instance

    @patch("credit_risk_agent.model.loader.mlflow.pytorch.load_model")
    def test_load_model_no_champion_raises_runtime_error(self, mock_load_model: MagicMock) -> None:
        """Verify RuntimeError is raised when loading champion model fails in MLflow."""
        # Arrange
        mock_load_model.side_effect = mlflow.exceptions.MlflowException("Model not found")

        # Act & Assert
        with pytest.raises(RuntimeError, match="Не удалось загрузить чемпионскую модель"):
            load_model_from_mlflow(run_id=None)
