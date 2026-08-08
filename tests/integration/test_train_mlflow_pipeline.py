import sqlite3
from pathlib import Path

import mlflow
import pandas as pd
import pytest
from torch.utils.data import DataLoader

import scripts.train as train_module
from credit_risk_agent.config import BEST_MODEL_ALIAS, BEST_MODEL_NAME
from credit_risk_agent.data import StandardScaler
from credit_risk_agent.model import CreditDefaultPredictor, prepare_dataset
from credit_risk_agent.model.loader import load_model_from_mlflow, load_scaler_from_mlflow


class TestMLflowPipeline:
    @pytest.fixture(autouse=True)
    def setup_environment(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Set up synthetic SQLite database and isolated MLflow SQLite tracking URI."""
        # 1. Paths setup
        db_path = tmp_path / "raw_database.db"
        train_db_path = tmp_path / "train_database.db"
        test_db_path = tmp_path / "test_database.db"
        artifacts_path = tmp_path / "artifacts"
        artifacts_path.mkdir(exist_ok=True)

        scaler_path = artifacts_path / "scaler.json"
        model_path = artifacts_path / "model.pt"

        monkeypatch.setattr(train_module, "RAW_DATABASE_PATH", db_path)
        monkeypatch.setattr(train_module, "TRAIN_DATABASE_PATH", train_db_path)
        monkeypatch.setattr(train_module, "TEST_DATABASE_PATH", test_db_path)
        monkeypatch.setattr(train_module, "MODEL_SAVE_PATH", model_path)
        monkeypatch.setattr(train_module, "SCALER_PATH", scaler_path)

        # 2. Populate synthetic database (10 clients, 60 payment records)
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

        # 3. Setup isolated MLflow tracking DB
        mlflow_db_path = tmp_path / "mlflow_test.db"
        tracking_uri = f"sqlite:///{mlflow_db_path}"
        mlflow.set_tracking_uri(tracking_uri)

    def test_full_mlflow_lifecycle(self, tmp_path: Path):
        """Verify complete MLflow lifecycle: error on clean state, 1st champion registration,
        ignoring worse runs, and promoting better runs to champion v2."""

        # --- Step 1: Verify clean state error handling ---
        with pytest.raises(RuntimeError, match="Чемпионская модель еще не назначена"):
            load_scaler_from_mlflow(run_id=None)

        with pytest.raises(RuntimeError, match="Не удалось загрузить чемпионскую модель"):
            load_model_from_mlflow(run_id=None)

        # Prepare datasets for training passes
        train_module.save_split_db()
        train_df = train_module.load_and_preprocess_from_db(train_module.TRAIN_DATABASE_PATH)
        test_df = train_module.load_and_preprocess_from_db(train_module.TEST_DATABASE_PATH)

        scaler = StandardScaler().fit(train_df, train_module.SCALER_COLS)
        scaler.save(train_module.SCALER_PATH)
        train_df = scaler.transform(train_df, train_module.SCALER_COLS)
        test_df = scaler.transform(test_df, train_module.SCALER_COLS)

        train_dataset = prepare_dataset(train_df, id_col="client_id", target_col="default")

        train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True)

        mlflow.set_experiment("credit_default_predictor")
        client = mlflow.MlflowClient()

        # --- Step 2: 1st Training Pass (Should become Champion v1) ---
        with mlflow.start_run(run_name="pass_1"):
            mlflow.log_artifact(str(train_module.SCALER_PATH), artifact_path="preprocessing")
            model1, loss1 = train_module.train_model(train_loader, train_module.MODEL_SAVE_PATH, epochs=2)
            mlflow.pytorch.log_model(model1, artifact_path="model", serialization_format="pickle")
            train_module.save_champion_model(loss1)

        champion_v1 = client.get_model_version_by_alias(BEST_MODEL_NAME, BEST_MODEL_ALIAS)
        assert str(champion_v1.version) == "1"

        # --- Step 3: Verify Inference loads Champion v1 ---
        loaded_scaler = load_scaler_from_mlflow(run_id=None)
        assert loaded_scaler is not None

        loaded_model = load_model_from_mlflow(run_id=None)
        assert isinstance(loaded_model, CreditDefaultPredictor)

        # --- Step 4: 2nd Training Pass with Worse Loss (Champion v1 should be kept) ---
        worse_loss = loss1 + 10.0  # Artificially higher loss
        with mlflow.start_run(run_name="pass_2_worse"):
            mlflow.log_metric("best_train_loss", worse_loss)
            mlflow.log_artifact(str(train_module.SCALER_PATH), artifact_path="preprocessing")
            model2 = CreditDefaultPredictor(hidden_size=64, num_layers=1, static_size=14, dropout_prob=0.28)
            mlflow.pytorch.log_model(model2, artifact_path="model", serialization_format="pickle")
            train_module.save_champion_model(worse_loss)

        champion_still_v1 = client.get_model_version_by_alias(BEST_MODEL_NAME, BEST_MODEL_ALIAS)
        assert str(champion_still_v1.version) == "1"

        # --- Step 5: 3rd Training Pass with Better Loss (Should promote to Champion v2) ---
        better_loss = loss1 - 0.1  # Artificially lower loss
        with mlflow.start_run(run_name="pass_3_better"):
            mlflow.log_metric("best_train_loss", better_loss)
            mlflow.log_artifact(str(train_module.SCALER_PATH), artifact_path="preprocessing")
            model3 = CreditDefaultPredictor(hidden_size=64, num_layers=1, static_size=14, dropout_prob=0.28)
            mlflow.pytorch.log_model(model3, artifact_path="model", serialization_format="pickle")
            train_module.save_champion_model(better_loss)

        champion_v2 = client.get_model_version_by_alias(BEST_MODEL_NAME, BEST_MODEL_ALIAS)
        assert str(champion_v2.version) == "2"

        # --- Step 6: Verify Inference loads Champion v2 ---
        loaded_model_v2 = load_model_from_mlflow(run_id=None)
        assert isinstance(loaded_model_v2, CreditDefaultPredictor)
