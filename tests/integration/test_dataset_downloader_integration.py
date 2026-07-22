import sqlite3
from unittest.mock import MagicMock

import pandas as pd
import pytest

from credit_risk_agent.data import downloader
from credit_risk_agent.data.downloader import main


def test_main_etl_flow(tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    # 1. Arrange: Setup temporary data path and mock Kaggle API
    temp_data_dir = tmp_path
    monkeypatch.setattr(downloader, "DATA_PATH", temp_data_dir)
    monkeypatch.setattr(downloader, "DATABASE_PATH", temp_data_dir / "database.db")

    # Mock Kaggle authentication and dataset download
    mock_authenticate = MagicMock()
    mock_download = MagicMock()

    monkeypatch.setattr(downloader.kaggle.api, "authenticate", mock_authenticate)
    monkeypatch.setattr(downloader.kaggle.api, "dataset_download_files", mock_download)

    # Define a function to simulate downloading by writing a mock CSV file
    def write_mock_csv(*args: any, **kwargs: any) -> None:
        mock_data = {
            "ID": [101, 102],
            "LIMIT_BAL": [50000.0, 100000.0],
            "SEX": [1, 2],
            "EDUCATION": [2, 1],
            "MARRIAGE": [1, 2],
            "AGE": [30, 25],
            "default.payment.next.month": [0, 1],
            "PAY_0": [2, -1],  # PAY_0 will be renamed to PAY_1
            "PAY_2": [0, 2],
            "BILL_AMT1": [1000.0, 2000.0],
            "BILL_AMT2": [1200.0, 1800.0],
            "PAY_AMT1": [500.0, 1000.0],
            "PAY_AMT2": [600.0, 800.0],
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

    # Verify SQLite database creation
    db_file = temp_data_dir / "database.db"
    assert db_file.exists()

    # Verify SQLite data structures and contents
    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()

        # Check clients table
        cursor.execute("SELECT * FROM clients ORDER BY client_id;")
        clients = cursor.fetchall()
        assert len(clients) == 2
        # Columns: client_id, limit_bal, sex, education, marriage, age, default
        assert clients[0] == (101, 50000.0, 1, 2, 1, 30, 0)
        assert clients[1] == (102, 100000.0, 2, 1, 2, 25, 1)

        # Check payment_history table
        cursor.execute("SELECT * FROM payment_history ORDER BY client_id, month;")
        history = cursor.fetchall()
        assert len(history) == 4  # 2 clients * 2 months each
        # Columns: client_id, month, pay_status, bill_amt, pay_amt
        # client 101, month 1 (PAY_0 renamed to PAY_1 -> 2, BILL_AMT1 -> 1000, PAY_AMT1 -> 500)
        assert history[0] == (101, 1, 2.0, 1000.0, 500.0)
        # client 101, month 2 (PAY_2 -> 0, BILL_AMT2 -> 1200, PAY_AMT2 -> 600)
        assert history[1] == (101, 2, 0.0, 1200.0, 600.0)
        # client 102, month 1 (PAY_0 renamed to PAY_1 -> -1, BILL_AMT1 -> 2000, PAY_AMT1 -> 1000)
        assert history[2] == (102, 1, -1.0, 2000.0, 1000.0)
        # client 102, month 2 (PAY_2 -> 2, BILL_AMT2 -> 1800, PAY_AMT2 -> 800)
        assert history[3] == (102, 2, 2.0, 1800.0, 800.0)
