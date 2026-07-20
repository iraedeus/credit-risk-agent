import numpy as np
import pandas as pd
import pytest
import torch

from model.dataset import CreditDataset, prepare_dataset


def test_credit_dataset_length() -> None:
    # Arrange
    sequences = np.random.randn(10, 6, 3)
    static_features = np.random.randn(10, 14)
    labels = np.random.randint(0, 2, size=(10,))

    # Act
    dataset = CreditDataset(sequences, static_features, labels)

    # Assert
    assert len(dataset) == 10


def test_credit_dataset_getitem() -> None:
    # Arrange
    sequences = np.random.randn(2, 6, 3)
    static_features = np.random.randn(2, 14)
    labels = np.array([0, 1])

    # Act
    dataset = CreditDataset(sequences, static_features, labels)
    x_seq, x_static, y = dataset[0]

    # Assert
    assert isinstance(x_seq, torch.Tensor)
    assert isinstance(x_static, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x_seq.shape == (6, 3)
    assert x_static.shape == (14,)
    assert y.shape == (1,)
    assert y.item() == 0.0


def test_prepare_dataset_shape() -> None:
    # Arrange
    # Create mock dataframe for 2 clients, each having 6 months of data
    df = pd.DataFrame(
        {
            "client_id": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
            "month": [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6],
            "bill_amt": [100.0] * 12,
            "pay_amt": [50.0] * 12,
            "pay_status": [0.0] * 12,
            "default": [0] * 12,
            "limit_bal": [10000.0] * 6 + [20000.0] * 6,
            "sex": [1] * 6 + [2] * 6,
            "marriage": [1] * 6 + [2] * 6,
            "education": [1] * 6 + [2] * 6,
            "age": [25] * 6 + [40] * 6,
        }
    )

    # Act
    dataset = prepare_dataset(df)

    # Assert
    assert len(dataset) == 2
    x_seq, x_static, y = dataset[0]
    assert x_seq.shape == (6, 3)
    assert x_static.shape == (14,)
    assert y.shape == (1,)
    assert y.item() == 0.0


def test_prepare_dataset_invalid_sequence_lengths() -> None:
    # Arrange
    # Client 1 has 6 records (valid), Client 2 has only 5 records (invalid)
    df = pd.DataFrame(
        {
            "client_id": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "month": [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5],
            "bill_amt": [100.0] * 11,
            "pay_amt": [50.0] * 11,
            "pay_status": [0.0] * 11,
            "default": [0] * 11,
            "limit_bal": [10000.0] * 6 + [20000.0] * 5,
            "sex": [1] * 6 + [2] * 5,
            "marriage": [1] * 6 + [2] * 5,
            "education": [1] * 6 + [2] * 5,
            "age": [25] * 6 + [40] * 5,
        }
    )

    # Act & Assert
    with pytest.raises(ValueError, match="must have exactly 6 records"):
        prepare_dataset(df)


def test_credit_dataset_memory_sharing() -> None:
    # Arrange
    sequences = np.random.randn(2, 6, 3).astype(np.float32)
    static_features = np.random.randn(2, 14).astype(np.float32)
    labels = np.array([0.0, 1.0], dtype=np.float32)

    # Act
    dataset = CreditDataset(sequences, static_features, labels)

    # Modify original numpy array
    sequences[0, 0, 0] = 999.0
    static_features[0, 0] = 888.0

    # Assert
    assert dataset.sequences[0, 0, 0].item() == 999.0
    assert dataset.static_features[0, 0].item() == 888.0
