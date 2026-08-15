"""
MLflow Model Registry and Artifact loading utilities.
"""

from pathlib import Path

import mlflow
import mlflow.artifacts
import mlflow.pytorch

from credit_risk_agent.data import StandardScaler
from credit_risk_agent.model.model import CreditDefaultModel


def load_scaler_from_registry(model_name: str, model_alias: str) -> StandardScaler:
    """
    Download and load the StandardScaler artifact from the champion run.

    Resolves the run of the champion model registered under the given
    name and alias in MLflow Model Registry, then downloads the scaler
    artifact from that run.

    Parameters
    ----------
    model_name : str
        Registered model name in MLflow Model Registry.
    model_alias : str
        MLflow Model Registry alias pointing to the champion version.

    Returns
    -------
    StandardScaler
        Loaded StandardScaler instance.

    Raises
    ------
    RuntimeError
        If no model version is registered under the given name and alias.
    """
    client = mlflow.MlflowClient()
    try:
        champion = client.get_model_version_by_alias(model_name, model_alias)
        run_id = champion.run_id
    except mlflow.exceptions.MlflowException:
        raise RuntimeError(
            "Чемпионская модель еще не назначена! "
            "Запустите обучение хотя бы один раз, чтобы создать чемпиона, "
            "либо явно укажите --run-id."
        )
    local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="preprocessing/scaler.json")
    return StandardScaler.load(Path(local_path))


def load_model_from_registry(model_name: str, model_alias: str) -> CreditDefaultModel:
    """
    Load the champion CreditDefaultModel from MLflow Model Registry.

    Parameters
    ----------
    model_name : str
        Registered model name in MLflow Model Registry.
    model_alias : str
        MLflow Model Registry alias pointing to the champion version.

    Returns
    -------
    CreditDefaultModel
        Loaded PyTorch CreditDefaultModel instance.

    Raises
    ------
    RuntimeError
        If no model version is registered under the given name and alias.
    """
    try:
        return mlflow.pytorch.load_model(f"models:/{model_name}@{model_alias}")
    except mlflow.exceptions.MlflowException:
        raise RuntimeError(
            "Чемпионская модель еще не назначена! "
            "Запустите обучение хотя бы один раз, чтобы создать чемпиона, "
            "либо явно укажите --run-id."
        )


def load_scaler_from_run(run_id: str) -> StandardScaler:
    """
    Download and load a StandardScaler artifact from a specific MLflow run.

    Parameters
    ----------
    run_id : str
        MLflow run ID to download the scaler artifact from.

    Returns
    -------
    StandardScaler
        Loaded StandardScaler instance.
    """
    local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="preprocessing/scaler.json")
    return StandardScaler.load(Path(local_path))


def load_model_from_run(run_id: str) -> CreditDefaultModel:
    """
    Load a trained PyTorch CreditDefaultModel from a specific MLflow run.

    Parameters
    ----------
    run_id : str
        MLflow run ID to load the model from.

    Returns
    -------
    CreditDefaultModel
        Loaded PyTorch CreditDefaultModel instance.
    """
    return mlflow.pytorch.load_model(f"runs:/{run_id}/model")
