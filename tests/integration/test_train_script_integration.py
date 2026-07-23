import sqlite3
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from torch.utils.data import DataLoader

import scripts.train as train_module
from credit_risk_agent.data import StandardScaler
from credit_risk_agent.model import CreditDefaultPredictor, prepare_dataset


class TestTrainScriptIntegration:
    def test_train_end_to_end_pipeline(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify end-to-end training pipeline creates database artifacts, trains model, and serializes outputs."""
        # 1. Arrange: Setup synthetic database in tmp_path
        db_path = tmp_path / "database.db"
        train_db_path = tmp_path / "train_database.db"
        test_db_path = tmp_path / "test_database.db"
        artifacts_path = tmp_path / "artifacts"
        artifacts_path.mkdir()

        scaler_path = artifacts_path / "scaler.json"
        model_path = artifacts_path / "model.pth"

        monkeypatch.setattr(train_module, "RAW_DATABASE_PATH", db_path)
        monkeypatch.setattr(train_module, "TRAIN_DATABASE_PATH", train_db_path)
        monkeypatch.setattr(train_module, "TEST_DATABASE_PATH", test_db_path)
        monkeypatch.setattr(train_module, "MODEL_SAVE_PATH", model_path)
        monkeypatch.setattr(train_module, "SCALER_PATH", scaler_path)

        # Populate synthetic SQLite database with 10 clients and 60 payment records
        with sqlite3.connect(db_path) as conn:
            clients_data = {
                "client_id": list(range(1, 11)),
                "limit_bal": [50000.0] * 10,
                "sex": [1, 2] * 5,
                "education": [1, 2, 3, 4, 1, 2, 3, 4, 1, 2],
                "marriage": [1, 2] * 5,
                "age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
            }
            pd.DataFrame(clients_data).to_sql("clients", conn, index=False)

            gt_data = {
                "client_id": list(range(1, 11)),
                "default": [0, 0, 1, 0, 1, 0, 0, 1, 0, 1],
            }
            pd.DataFrame(gt_data).to_sql("ground_truth", conn, index=False)

            history_records = []
            for client_id in range(1, 11):
                for month in range(1, 7):
                    history_records.append(
                        {
                            "client_id": client_id,
                            "month": month,
                            "pay_status": 0.0,
                            "bill_amt": 1000.0 * month,
                            "pay_amt": 500.0 * month,
                        }
                    )
            pd.DataFrame(history_records).to_sql("payment_history", conn, index=False)

        # 2. Act: Execute ETL & Dataset preparation
        train_module.save_split_db()
        train_df = train_module.load_and_preprocess_from_db(train_db_path)
        test_df = train_module.load_and_preprocess_from_db(test_db_path)

        scaler = StandardScaler().fit(train_df, train_module.SCALER_COLS)
        scaler.save(scaler_path)
        train_df = scaler.transform(train_df, train_module.SCALER_COLS)
        test_df = scaler.transform(test_df, train_module.SCALER_COLS)

        train_dataset = prepare_dataset(train_df, id_col="client_id", target_col="default")
        test_dataset = prepare_dataset(test_df, id_col="client_id", target_col="default")

        train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=False)

        # Train for 1 epoch for fast integration test
        with patch("scripts.train.range", return_value=range(1)):
            model = train_module.train_model(train_loader, model_path)
            train_module.check_model_quality(model, test_loader)

        # 3. Assert: Verify generated artifacts
        assert scaler_path.exists()
        assert model_path.exists()
        assert train_db_path.exists()
        assert test_db_path.exists()

        loaded_scaler = StandardScaler.load(scaler_path)
        assert loaded_scaler.mean is not None

        loaded_model = CreditDefaultPredictor(hidden_size=64, num_layers=1, static_size=14, dropout_prob=0.28)
        loaded_model.load_state_dict(train_module.torch.load(model_path))
        assert loaded_model is not None
