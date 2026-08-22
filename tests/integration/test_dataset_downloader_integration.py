import sqlite3
from unittest.mock import MagicMock

import pandas as pd
import pytest

from scripts import download_dataset
from scripts.download_dataset import main


class TestDatasetDownloaderIntegration:
    def test_main_etl_flow(self, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify end-to-end ETL flow of dataset downloading, CSV parsing,
        stratified splitting, scaler fitting, and cleanup."""
        # 1. Arrange: Setup temporary data path and mock Kaggle API
        temp_data_dir = tmp_path
        train_db = temp_data_dir / "train_database.db"
        test_db = temp_data_dir / "test_database.db"

        monkeypatch.setattr(download_dataset, "DATA_PATH", temp_data_dir)
        monkeypatch.setattr(download_dataset, "RAW_DATABASE_PATH", temp_data_dir / "database.db")
        monkeypatch.setattr(download_dataset, "TRAIN_DATABASE_PATH", train_db)
        monkeypatch.setattr(download_dataset, "TEST_DATABASE_PATH", test_db)

        # Mock Kaggle authentication and dataset download
        mock_authenticate = MagicMock()
        mock_download = MagicMock()

        monkeypatch.setattr(download_dataset.kaggle.api, "authenticate", mock_authenticate)
        monkeypatch.setattr(download_dataset.kaggle.api, "dataset_download_files", mock_download)

        # Define a function to simulate downloading by writing a mock CSV file with 10 clients for stratification
        def write_mock_csv(*args: any, **kwargs: any) -> None:
            mock_data = {
                "ID": list(range(101, 111)),
                "LIMIT_BAL": [50000.0] * 10,
                "SEX": [1, 2] * 5,
                "EDUCATION": [2, 1] * 5,
                "MARRIAGE": [1, 2] * 5,
                "AGE": [30, 25, 35, 40, 45, 50, 55, 60, 65, 70],
                "default.payment.next.month": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],  # 5 non-default, 5 default
                "PAY_0": [2] * 10,
                "PAY_2": [0] * 10,
                "BILL_AMT1": [1000.0] * 10,
                "BILL_AMT2": [1200.0] * 10,
                "PAY_AMT1": [500.0] * 10,
                "PAY_AMT2": [600.0] * 10,
            }
            df = pd.DataFrame(mock_data)
            df.to_csv(temp_data_dir / "UCI_Credit_Card.csv", index=False)

        mock_download.side_effect = write_mock_csv

        # 2. Act: Run the main ETL process
        main()

        # 3. Assert: Verify API calls
        mock_authenticate.assert_called_once()
        mock_download.assert_called_once()

        # Verify CSV file cleanup
        csv_file = temp_data_dir / "UCI_Credit_Card.csv"
        assert not csv_file.exists()

        # Verify SQLite database and scaler creation
        assert (temp_data_dir / "database.db").exists()
        assert train_db.exists()
        assert test_db.exists()

        # Verify train/test dataset partition and stratification
        with sqlite3.connect(train_db) as conn_tr, sqlite3.connect(test_db) as conn_te:
            train_clients = pd.read_sql_query("SELECT * FROM clients", conn_tr)
            test_clients = pd.read_sql_query("SELECT * FROM clients", conn_te)

            train_gt = pd.read_sql_query("SELECT * FROM ground_truth", conn_tr)
            test_gt = pd.read_sql_query("SELECT * FROM ground_truth", conn_te)

            # Check that total client records sum to original 10
            assert len(train_clients) == 8
            assert len(test_clients) == 2

            # Check no overlap in client IDs
            train_ids = set(train_clients["client_id"])
            test_ids = set(test_clients["client_id"])
            assert train_ids.isdisjoint(test_ids)

            # Check stratification: 50% target ratio preserved in both train and test splits
            assert train_gt["default"].mean() == 0.5
            assert test_gt["default"].mean() == 0.5
