import numpy as np
import torch

from model.dataset import CreditDataset


def test_credit_dataset_length() -> None:
    # Arrange
    sequences = np.random.randn(10, 5, 3)
    labels = np.random.randint(0, 2, size=(10,))

    # Act
    dataset = CreditDataset(sequences, labels)

    # Assert
    assert len(dataset) == 10


def test_credit_dataset_getitem() -> None:
    # Arrange
    sequences = np.random.randn(2, 5, 3)
    labels = np.array([0, 1])

    # Act
    dataset = CreditDataset(sequences, labels)
    x, y = dataset[0]

    # Assert
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape == (5, 3)
    assert y.shape == (1,)
    assert y.item() == 0.0
