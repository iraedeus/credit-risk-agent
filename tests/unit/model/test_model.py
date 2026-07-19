import torch

from model.model import CreditDefaultPredictor


def test_model_initialization() -> None:
    # Arrange & Act
    model = CreditDefaultPredictor(hidden_size=32, num_layers=2)

    # Assert
    assert model.gru.hidden_size == 32
    assert model.gru.num_layers == 2
    assert model.gru.input_size == 3
    assert model.fc.in_features == 32
    assert model.fc.out_features == 1


def test_model_forward_shape() -> None:
    # Arrange
    batch_size = 8
    seq_len = 10
    input_size = 3
    hidden_size = 64

    model = CreditDefaultPredictor(hidden_size=hidden_size)
    x = torch.randn(batch_size, seq_len, input_size)

    # Act
    output = model(x)

    # Assert
    assert output.shape == (batch_size, 1)


def test_model_forward_different_batches() -> None:
    # Arrange
    model = CreditDefaultPredictor()

    # Test batch size of 1
    x_single = torch.randn(1, 5, 3)
    output_single = model(x_single)
    assert output_single.shape == (1, 1)

    # Test batch size of 16
    x_large = torch.randn(16, 20, 3)
    output_large = model(x_large)
    assert output_large.shape == (16, 1)
