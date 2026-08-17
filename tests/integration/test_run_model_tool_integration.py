import pandas as pd
import pytest

from credit_risk_agent.agent.tools import run_model
from credit_risk_agent.config import DATA_PATH, DATA_SERVICE_URL, ML_SERVICE_URL
from credit_risk_agent.services.data_service.client import get_data_service_client
from credit_risk_agent.services.ml_service.client import get_ml_service_client


class TestRunModelToolIntegration:
    def test_run_model_integration_smoke(self) -> None:
        """Smoke test for run_model tool using real model weights and dataset artifacts."""

        if not get_data_service_client().get_healthcheck():
            pytest.skip(f"Data Service недоступен по {DATA_SERVICE_URL}, пропускаем тест.")

        if not get_ml_service_client().get_healthcheck():
            pytest.skip(f"ML Service недоступен по {ML_SERVICE_URL}, пропускаем тест.")

        test_clients_path = DATA_PATH / "test_clients.csv"
        if test_clients_path.exists():
            df = pd.read_csv(test_clients_path)
            if not df.empty and "client_id" in df.columns:
                valid_id = int(df["client_id"].iloc[0])
                result = run_model(valid_id)
                assert "выдала результат равный" in result
                return

        result = run_model(99999999)
        assert "не был найден в базе" in result
