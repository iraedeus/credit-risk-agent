from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import torch

from scripts.train import (
    check_model_quality,
    load_and_preprocess_from_db,
    main,
    save_split_db,
    train_model,
)


class TestLoadData:
    @patch("scripts.train.preprocess")
    @patch("scripts.train.pd.read_sql_query")
    @patch("scripts.train.sqlite3.connect")
    def test_load_and_preprocess_from_db_success(
        self, mock_connect: MagicMock, mock_read_sql: MagicMock, mock_preprocess: MagicMock
    ) -> None:
        """Verify loading clients, payment history, and ground truth from SQLite database, merging and preprocessing."""
        # Arrange
        mock_conn = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_conn

        client_df = pd.DataFrame({"client_id": [1, 2], "age": [30, 40]})
        history_df = pd.DataFrame({"client_id": [1, 2], "month": [1, 1]})
        gt_df = pd.DataFrame({"client_id": [1, 2], "default": [0, 1]})
        mock_read_sql.side_effect = [client_df, history_df, gt_df]

        merged_df = pd.DataFrame({"client_id": [1, 2], "age": [30, 40], "month": [1, 1], "default": [0, 1]})
        mock_preprocess.return_value = merged_df

        # Act
        result = load_and_preprocess_from_db(Path("dummy.db"))

        # Assert
        assert mock_read_sql.call_count == 3
        mock_preprocess.assert_called_once()
        assert list(result["client_id"]) == [1, 2]


class TestSplitAndSave:
    @patch("scripts.train.pd.DataFrame.to_sql")
    @patch("scripts.train.train_test_split")
    @patch("scripts.train.pd.read_sql_query")
    @patch("scripts.train.sqlite3.connect")
    def test_save_split_db_success(
        self,
        mock_connect: MagicMock,
        mock_read_sql: MagicMock,
        mock_split: MagicMock,
        mock_to_sql: MagicMock,
    ) -> None:
        """Verify reading raw database tables, performing stratified train/test split,
        and saving to SQLite databases."""
        # Arrange
        mock_conn = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_conn

        raw_clients = pd.DataFrame({"client_id": [1, 2], "age": [30, 40]})
        raw_history = pd.DataFrame({"client_id": [1, 2], "month": [1, 1]})
        raw_gt = pd.DataFrame({"client_id": [1, 2], "default": [0, 1]})
        mock_read_sql.side_effect = [raw_clients, raw_history, raw_gt]

        mock_split.return_value = (pd.Series([1]), pd.Series([2]))

        # Act
        save_split_db()

        # Assert
        assert mock_read_sql.call_count == 3
        mock_split.assert_called_once()
        assert mock_to_sql.call_count == 6


class TestTrainModel:
    @patch("scripts.train.nn.BCEWithLogitsLoss")
    @patch("scripts.train.torch.save")
    @patch("scripts.train.CreditDefaultPredictor")
    def test_train_model_executes_epoch_loop_and_saves_weights(
        self,
        mock_predictor_cls: MagicMock,
        mock_torch_save: MagicMock,
        mock_loss_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Verify that train_model runs training iterations over loader batches and saves model weights to disk."""
        # Arrange
        mock_model = MagicMock()
        mock_param = torch.nn.Parameter(torch.zeros(1, requires_grad=True))
        mock_model.parameters.return_value = [mock_param]
        mock_model.return_value = torch.tensor([[0.5]])
        mock_predictor_cls.return_value = mock_model

        mock_loss_fn = MagicMock()
        mock_loss_val = MagicMock()
        mock_loss_val.item.return_value = 0.5
        mock_loss_fn.return_value = mock_loss_val
        mock_loss_cls.return_value = mock_loss_fn

        dummy_seq = torch.zeros((1, 6, 3))
        dummy_static = torch.zeros((1, 14))
        dummy_label = torch.tensor([[0.0]])

        mock_loader = [(dummy_seq, dummy_static, dummy_label)]
        save_path = tmp_path / "model.pth"

        # Act
        result_model = train_model(mock_loader, save_path)

        # Assert
        mock_torch_save.assert_called_once_with(mock_model.state_dict(), save_path)
        assert result_model == mock_model


class TestCheckModelQuality:
    @patch("scripts.train.classification_report")
    @patch("scripts.train.roc_auc_score")
    def test_check_model_quality_calculates_roc_auc_and_report(
        self,
        mock_roc_auc: MagicMock,
        mock_class_report: MagicMock,
    ) -> None:
        """Verify model quality evaluation calculates ROC-AUC and prints classification report."""
        # Arrange
        mock_model = MagicMock()
        mock_model.return_value = torch.tensor([[2.0], [-2.0]])

        dummy_seq = torch.zeros((2, 6, 3))
        dummy_static = torch.zeros((2, 14))
        dummy_labels = torch.tensor([[1.0], [0.0]])

        mock_loader = [(dummy_seq, dummy_static, dummy_labels)]
        mock_roc_auc.return_value = 0.95
        mock_class_report.return_value = "Mock Classification Report"

        # Act
        check_model_quality(mock_model, mock_loader)

        # Assert
        mock_model.eval.assert_called_once()
        mock_roc_auc.assert_called_once()
        mock_class_report.assert_called_once()


class TestMainCLI:
    @patch("scripts.train.check_model_quality")
    @patch("scripts.train.train_model")
    @patch("scripts.train.prepare_dataset")
    @patch("scripts.train.StandardScaler")
    @patch("scripts.train.load_and_preprocess_from_db")
    @patch("scripts.train.save_split_db")
    @patch("argparse.ArgumentParser.parse_args")
    def test_main_default_training_mode(
        self,
        mock_parse_args: MagicMock,
        mock_save_split: MagicMock,
        mock_load_db: MagicMock,
        mock_scaler_cls: MagicMock,
        mock_prep_ds: MagicMock,
        mock_train: MagicMock,
        mock_check_quality: MagicMock,
    ) -> None:
        """Verify default CLI execution runs data loading, splitting, training, and evaluation."""
        # Arrange
        mock_args = MagicMock()
        mock_args.view_quality = False
        mock_parse_args.return_value = mock_args

        mock_df = MagicMock()
        mock_load_db.return_value = mock_df

        mock_scaler_instance = MagicMock()
        mock_scaler_instance.transform.return_value = mock_df
        mock_scaler_cls.return_value.fit.return_value = mock_scaler_instance

        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 5
        mock_prep_ds.return_value = mock_dataset

        mock_train.return_value = MagicMock()

        # Act
        main()

        # Assert
        mock_save_split.assert_called_once()
        assert mock_load_db.call_count == 2
        mock_train.assert_called_once()
        mock_check_quality.assert_called_once()

    @patch("scripts.train.check_model_quality")
    @patch("scripts.train.CreditDefaultPredictor")
    @patch("scripts.train.prepare_dataset")
    @patch("scripts.train.StandardScaler.load")
    @patch("scripts.train.load_and_preprocess_from_db")
    @patch("scripts.train.save_split_db")
    @patch("scripts.train.torch.load")
    @patch("argparse.ArgumentParser.parse_args")
    def test_main_view_quality_mode(
        self,
        mock_parse_args: MagicMock,
        mock_torch_load: MagicMock,
        mock_save_split: MagicMock,
        mock_load_db: MagicMock,
        mock_scaler_load: MagicMock,
        mock_prep_ds: MagicMock,
        mock_predictor: MagicMock,
        mock_check_quality: MagicMock,
    ) -> None:
        """Verify --view-quality CLI execution skips training and evaluates saved model weights."""
        # Arrange
        mock_args = MagicMock()
        mock_args.view_quality = True
        mock_parse_args.return_value = mock_args

        mock_test_df = MagicMock()
        mock_load_db.return_value = mock_test_df
        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = mock_test_df
        mock_scaler_load.return_value = mock_scaler

        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 5
        mock_prep_ds.return_value = mock_dataset

        # Act
        main()

        # Assert
        mock_save_split.assert_called_once()
        mock_load_db.assert_called_once()
        mock_scaler_load.assert_called_once()
        mock_check_quality.assert_called_once()
