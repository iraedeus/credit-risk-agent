from unittest.mock import MagicMock, patch

import pandas as pd
import torch

from agent.tools import run_model


@patch("agent.tools.prepare_dataset")
@patch("agent.tools.normalize")
@patch("agent.tools.load_and_preprocess_test_data")
@patch("agent.tools.CreditDefaultPredictor")
@patch("torch.load")
def test_run_model_success(
    mock_torch_load: MagicMock,
    mock_predictor_cls: MagicMock,
    mock_load_data: MagicMock,
    mock_normalize: MagicMock,
    mock_prepare_dataset: MagicMock,
) -> None:
    """TC-RM-01: Проверка успешного выполнения скоринга для существующего клиента."""
    # 1. Arrange
    client_id = 15
    mock_df = pd.DataFrame({"client_id": [15, 20]})
    mock_load_data.return_value = mock_df
    mock_normalize.return_value = mock_df

    dummy_seq = torch.zeros((6, 3))
    dummy_static = torch.zeros((14,))
    mock_prepare_dataset.return_value = [(dummy_seq, dummy_static)]

    mock_model_instance = MagicMock()
    mock_model_instance.return_value = torch.tensor([[0.0]])
    mock_predictor_cls.return_value = mock_model_instance

    # 2. Act
    result = run_model(client_id)

    # 3. Assert
    assert result == "Модель на клиенте с id=15 выдала результат равный 0.5000."
    mock_prepare_dataset.assert_called_once()
    mock_model_instance.assert_called_once()


@patch("agent.tools.normalize")
@patch("agent.tools.load_and_preprocess_test_data")
@patch("agent.tools.CreditDefaultPredictor")
@patch("torch.load")
def test_run_model_client_not_found(
    mock_torch_load: MagicMock,
    mock_predictor_cls: MagicMock,
    mock_load_data: MagicMock,
    mock_normalize: MagicMock,
) -> None:
    """TC-RM-02: Проверка поведения при отсутствии client_id в выборке."""
    # 1. Arrange
    client_id = 999999
    mock_df = pd.DataFrame({"client_id": [1, 2, 3]})
    mock_load_data.return_value = mock_df
    mock_normalize.return_value = mock_df

    # 2. Act
    result = run_model(client_id)

    # 3. Assert
    assert result == f"Клиент с id={client_id} не был найден в базе."


@patch("agent.tools.prepare_dataset")
@patch("agent.tools.normalize")
@patch("agent.tools.load_and_preprocess_test_data")
@patch("agent.tools.CreditDefaultPredictor")
@patch("torch.load")
def test_run_model_formatting_precision(
    mock_torch_load: MagicMock,
    mock_predictor_cls: MagicMock,
    mock_load_data: MagicMock,
    mock_normalize: MagicMock,
    mock_prepare_dataset: MagicMock,
) -> None:
    """TC-RM-03: Проверка форматирования скоринга (точность до 4 знаков после запятой)."""
    # 1. Arrange
    client_id = 42
    mock_df = pd.DataFrame({"client_id": [42]})
    mock_load_data.return_value = mock_df
    mock_normalize.return_value = mock_df

    dummy_seq = torch.zeros((6, 3))
    dummy_static = torch.zeros((14,))
    mock_prepare_dataset.return_value = [(dummy_seq, dummy_static)]

    mock_model_instance = MagicMock()
    # Logit 2.1972246 -> sigmoid = 0.9000
    mock_model_instance.return_value = torch.tensor([[2.1972246]])
    mock_predictor_cls.return_value = mock_model_instance

    # 2. Act
    result = run_model(client_id)

    # 3. Assert
    assert result == "Модель на клиенте с id=42 выдала результат равный 0.9000."
