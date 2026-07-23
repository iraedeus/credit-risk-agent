import pandas as pd
import pytest

from credit_risk_agent.agent.tools import run_model
from credit_risk_agent.config import DATA_PATH, MODEL_SAVE_PATH, RAW_DATABASE_PATH, SCALER_PATH


class TestRunModelToolIntegration:
    def test_run_model_integration_smoke(self) -> None:
        """Smoke test for run_model tool using real model weights and dataset artifacts."""
        if not (MODEL_SAVE_PATH.exists() and SCALER_PATH.exists() and RAW_DATABASE_PATH.exists()):
            pytest.skip("Model or dataset files are missing, skipping integration test.")

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
