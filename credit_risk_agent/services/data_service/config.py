"""
Data Service configuration settings.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Configuration settings for Data Service microservice.
    """

    model_config = SettingsConfigDict(
        env_prefix="DATA_SERVICE_",
        env_file=".env",
        extra="ignore",
    )

    database_path: str = "data/test_database.db"
    host: str = "0.0.0.0"
    port: int = 8001
