import pandas as pd
import pytest

from agent.tools import run_model
from data.dataset_downloader import DATASET_PATH
from model.train import MODEL_SAVE_PATH, SCALER_PATH


def test_run_model_integration_smoke() -> None:
    """Интеграционный smoke-тест работы run_model на реальных весах и датасете."""
    if not (MODEL_SAVE_PATH.exists() and SCALER_PATH.exists() and DATASET_PATH.exists()):
        pytest.skip("Файлы модели или датасета отсутствуют, пропуск интеграционного теста.")

    test_clients_path = DATASET_PATH / "test_clients.csv"
    if test_clients_path.exists():
        df = pd.read_csv(test_clients_path)
        if not df.empty and "client_id" in df.columns:
            valid_id = int(df["client_id"].iloc[0])
            result = run_model(valid_id)
            assert "выдала результат равный" in result
            return

    result = run_model(99999999)
    assert "не был найден в базе" in result
