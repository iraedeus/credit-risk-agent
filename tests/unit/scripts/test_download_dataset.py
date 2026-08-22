from unittest.mock import MagicMock, patch

import pandas as pd

from scripts.download_dataset import (
    create_raw_sql_db,
    create_train_test_db,
    data_preprocessing,
    download_kaggle_dataset,
)


class TestDownloadDataset:
    @patch("scripts.download_dataset.kaggle")
    def test_download_kaggle_dataset_authenticates_and_downloads(self, mock_kaggle: MagicMock) -> None:
        """Verify downloading Kaggle dataset calls Kaggle API authentication and download."""
        download_kaggle_dataset()
        mock_kaggle.api.authenticate.assert_called_once()
        mock_kaggle.api.dataset_download_files.assert_called_once()


class TestDataPreprocessing:
    @patch("scripts.download_dataset.pd.read_csv")
    def test_data_preprocessing_transforms_wide_to_long(self, mock_read_csv: MagicMock) -> None:
        """Verify data preprocessing renames PAY_0 to PAY_1 and pivots payment history into long format."""
        mock_raw_df = pd.DataFrame(
            {
                "ID": [1],
                "LIMIT_BAL": [50000.0],
                "SEX": [1],
                "EDUCATION": [2],
                "MARRIAGE": [1],
                "AGE": [30],
                "default.payment.next.month": [0],
                "PAY_0": [2],
                "BILL_AMT1": [1000.0],
                "PAY_AMT1": [500.0],
            }
        )
        mock_read_csv.return_value = mock_raw_df

        client_df, history_df = data_preprocessing()

        assert "client_id" in client_df.columns
        assert list(client_df["client_id"]) == [1]

        assert "client_id" in history_df.columns
        assert "month" in history_df.columns
        assert "pay_status" in history_df.columns
        assert list(history_df["pay_status"]) == [2.0]


class TestCreateRawSqlDb:
    @patch("scripts.download_dataset.sqlite3.connect")
    def test_create_raw_sql_db_creates_tables_and_inserts_data(self, mock_connect: MagicMock) -> None:
        """Verify create_raw_sql_db creates clients, ground_truth, and payment_history tables."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        client_df = pd.DataFrame(
            {
                "client_id": [1],
                "limit_bal": [50000.0],
                "sex": [1],
                "education": [2],
                "marriage": [1],
                "age": [30],
                "default": [0],
            }
        )
        history_df = pd.DataFrame(
            {"client_id": [1], "month": [1], "pay_status": [0.0], "bill_amt": [1000.0], "pay_amt": [500.0]}
        )

        create_raw_sql_db(client_df, history_df)

        assert mock_cursor.execute.call_count >= 4
        assert mock_conn.commit.called


class TestCreateTrainTestDb:
    @patch("scripts.download_dataset.pd.DataFrame.to_sql")
    @patch("scripts.download_dataset.train_test_split")
    @patch("scripts.download_dataset.pd.read_sql_query")
    @patch("scripts.download_dataset.sqlite3.connect")
    def test_create_train_test_db_splits_and_saves(
        self,
        mock_connect: MagicMock,
        mock_read_sql: MagicMock,
        mock_split: MagicMock,
        mock_to_sql: MagicMock,
    ) -> None:
        """Verify create_train_test_db performs stratified train/test split and writes to train/test databases."""
        mock_conn = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_conn

        raw_clients = pd.DataFrame({"client_id": [1, 2], "age": [30, 40]})
        raw_history = pd.DataFrame({"client_id": [1, 2], "month": [1, 1]})
        raw_gt = pd.DataFrame({"client_id": [1, 2], "default": [0, 1]})
        mock_read_sql.side_effect = [raw_clients, raw_history, raw_gt]

        mock_split.return_value = (pd.Series([1]), pd.Series([2]))

        create_train_test_db()

        assert mock_read_sql.call_count == 3
        mock_split.assert_called_once()
        assert mock_to_sql.call_count == 6
