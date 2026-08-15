from pathlib import Path
from unittest.mock import MagicMock, patch

import mlflow.exceptions
import pytest

from credit_risk_agent.model.loader import (
    load_model_from_registry,
    load_model_from_run,
    load_scaler_from_registry,
    load_scaler_from_run,
)


class TestLoadScalerFromRun:
    @patch("credit_risk_agent.model.loader.StandardScaler.load")
    @patch("credit_risk_agent.model.loader.mlflow.artifacts.download_artifacts")
    def test_load_scaler_with_run_id_success(self, mock_download: MagicMock, mock_scaler_load: MagicMock) -> None:
        """Verify downloading and loading scaler artifact from the specified run."""
        # Arrange
        mock_download.return_value = "/tmp/dummy_scaler.json"
        mock_scaler_instance = MagicMock()
        mock_scaler_load.return_value = mock_scaler_instance

        # Act
        result = load_scaler_from_run(run_id="run_123")

        # Assert
        mock_download.assert_called_once_with(run_id="run_123", artifact_path="preprocessing/scaler.json")
        mock_scaler_load.assert_called_once_with(Path("/tmp/dummy_scaler.json"))
        assert result == mock_scaler_instance


class TestLoadScalerFromRegistry:
    @patch("credit_risk_agent.model.loader.StandardScaler.load")
    @patch("credit_risk_agent.model.loader.mlflow.artifacts.download_artifacts")
    @patch("credit_risk_agent.model.loader.mlflow.MlflowClient")
    def test_load_scaler_resolves_champion_by_given_name_and_alias(
        self, mock_client_cls: MagicMock, mock_download: MagicMock, mock_scaler_load: MagicMock
    ) -> None:
        """Verify resolving champion run_id using the provided model name and alias."""
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
        result = load_scaler_from_registry(model_name="MyModel", model_alias="my-alias")

        # Assert
        mock_client.get_model_version_by_alias.assert_called_once_with("MyModel", "my-alias")
        mock_download.assert_called_once_with(run_id="champion_run_456", artifact_path="preprocessing/scaler.json")
        assert result == mock_scaler_instance

    @patch("credit_risk_agent.model.loader.mlflow.MlflowClient")
    def test_load_scaler_no_champion_raises_runtime_error(self, mock_client_cls: MagicMock) -> None:
        """Verify RuntimeError is raised when the model has no champion alias."""
        # Arrange
        mock_client = MagicMock()
        mock_client.get_model_version_by_alias.side_effect = mlflow.exceptions.MlflowException("Not found")
        mock_client_cls.return_value = mock_client

        # Act & Assert
        with pytest.raises(RuntimeError, match="Чемпионская модель еще не назначена"):
            load_scaler_from_registry(model_name="MyModel", model_alias="my-alias")


class TestLoadModelFromRun:
    @patch("credit_risk_agent.model.loader.mlflow.pytorch.load_model")
    def test_load_model_with_run_id_success(self, mock_load_model: MagicMock) -> None:
        """Verify loading PyTorch model from the specified run."""
        # Arrange
        mock_model_instance = MagicMock()
        mock_load_model.return_value = mock_model_instance

        # Act
        result = load_model_from_run(run_id="run_123")

        # Assert
        mock_load_model.assert_called_once_with("runs:/run_123/model")
        assert result == mock_model_instance


class TestLoadModelFromRegistry:
    @patch("credit_risk_agent.model.loader.mlflow.pytorch.load_model")
    def test_load_model_uses_given_name_and_alias(self, mock_load_model: MagicMock) -> None:
        """Verify loading model from registry using the provided model name and alias."""
        # Arrange
        mock_model_instance = MagicMock()
        mock_load_model.return_value = mock_model_instance

        # Act
        result = load_model_from_registry(model_name="MyModel", model_alias="my-alias")

        # Assert
        mock_load_model.assert_called_once_with("models:/MyModel@my-alias")
        assert result == mock_model_instance

    @patch("credit_risk_agent.model.loader.mlflow.pytorch.load_model")
    def test_load_model_no_champion_raises_runtime_error(self, mock_load_model: MagicMock) -> None:
        """Verify RuntimeError is raised when loading the champion model fails in MLflow."""
        # Arrange
        mock_load_model.side_effect = mlflow.exceptions.MlflowException("Model not found")

        # Act & Assert
        with pytest.raises(RuntimeError, match="Чемпионская модель еще не назначена"):
            load_model_from_registry(model_name="MyModel", model_alias="my-alias")
