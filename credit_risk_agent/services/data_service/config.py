"""
Data Service configuration settings.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Configuration settings for the Data Service microservice.

    Attributes
    ----------
    database_path : str, default="data/test_database.db"
        Path to the SQLite database file containing client and payment records.
    host : str, default="0.0.0.0"
        Host IP address on which the data service runs.
    port : int, default=8001
        Port number on which the data service runs.
    """

    model_config = SettingsConfigDict(
        env_prefix="DATA_SERVICE_",
        env_file=".env",
        extra="ignore",
    )

    database_path: str = "data/test_database.db"
    host: str = "0.0.0.0"
    port: int = 8001
