"""
ML Service configuration settings.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Configuration settings for the ML Inference Microservice.

    Attributes
    ----------
    host : str, default="0.0.0.0"
        Host IP address on which the ML service runs.
    port : int, default=8002
        Port number on which the ML service runs.
    mlflow_url : str, default="sqlite:///mlflow.db"
        MLflow tracking server URI or SQLite database path.
    model_name : str, default="CreditRiskModel"
        Registered model name in MLflow Model Registry.
    model_alias : str, default="champion"
        MLflow Model Registry alias for champion model deployment.
    """

    model_config = SettingsConfigDict(
        env_prefix="ML_SERVICE_",
        env_file=".env",
        extra="ignore",
    )

    host: str = "0.0.0.0"
    port: int = 8002

    mlflow_url: str = "sqlite:///mlflow.db"
    model_name: str = "CreditRiskModel"
    model_alias: str = "champion"
