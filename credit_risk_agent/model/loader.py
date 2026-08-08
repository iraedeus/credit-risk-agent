"""
MLflow Model Registry and Artifact loading utilities.
"""

from pathlib import Path

import mlflow
import mlflow.artifacts
import mlflow.pytorch

from credit_risk_agent.config import BEST_MODEL_ALIAS, BEST_MODEL_NAME
from credit_risk_agent.data import StandardScaler
from credit_risk_agent.model.model import CreditDefaultModel


def load_scaler_from_mlflow(run_id: str | None = None) -> StandardScaler:
    """
    Download and load a StandardScaler artifact from MLflow.

    If `run_id` is provided, downloads the scaler artifact from that specific run.
    Otherwise, resolves the `run_id` of the current champion model registered in MLflow Model Registry.

    Parameters
    ----------
    run_id : str or None, default=None
        MLflow run ID to download the scaler from. If None, resolves the champion model run ID.

    Returns
    -------
    StandardScaler
        Loaded StandardScaler instance.

    Raises
    ------
    RuntimeError
        If `run_id` is None and no champion model is currently registered in MLflow.
    """
    if run_id is None:
        client = mlflow.MlflowClient()
        try:
            champion = client.get_model_version_by_alias(BEST_MODEL_NAME, BEST_MODEL_ALIAS)
            run_id = champion.run_id
        except mlflow.exceptions.MlflowException:
            raise RuntimeError(
                "Чемпионская модель еще не назначена! "
                "Запустите обучение хотя бы один раз, чтобы создать чемпиона, "
                "либо явно укажите --run-id."
            )

    local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="preprocessing/scaler.json")
    return StandardScaler.load(Path(local_path))


def load_model_from_mlflow(run_id: str | None = None) -> CreditDefaultModel:
    """
    Load a trained PyTorch CreditDefaultModel from MLflow.

    If `run_id` is provided, loads the model logged under `runs:/{run_id}/model`.
    Otherwise, loads the champion model directly from MLflow Model Registry
    using alias `models:/{BEST_MODEL_NAME}@{BEST_MODEL_ALIAS}`.

    Parameters
    ----------
    run_id : str or None, default=None
        MLflow run ID to load the model from. If None, loads champion model from Model Registry.

    Returns
    -------
    CreditDefaultModel
        Loaded PyTorch CreditDefaultModel instance.

    Raises
    ------
    RuntimeError
        If `run_id` is None and no champion model is currently assigned in Model Registry.
    """
    if run_id:
        return mlflow.pytorch.load_model(f"runs:/{run_id}/model")

    try:
        return mlflow.pytorch.load_model(f"models:/{BEST_MODEL_NAME}@{BEST_MODEL_ALIAS}")
    except mlflow.exceptions.MlflowException:
        raise RuntimeError("Не удалось загрузить чемпионскую модель. Возможно, она еще не назначена в Model Registry.")
