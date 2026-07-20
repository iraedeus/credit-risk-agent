import torch

from model.model import CreditDefaultPredictor


def test_model_initialization() -> None:
    # Arrange & Act
    model = CreditDefaultPredictor(hidden_size=32, num_layers=2, static_size=5, dropout_prob=0.3)

    # Assert
    assert model.gru.hidden_size == 32
    assert model.gru.num_layers == 2
    assert model.gru.input_size == 3
    assert model.fc1.in_features == 37
    assert model.fc1.out_features == 32
    assert model.dropout.p == 0.3
    assert model.fc2.in_features == 32
    assert model.fc2.out_features == 1


def test_model_forward_shape() -> None:
    # Arrange
    batch_size = 8
    seq_len = 10
    input_size = 3
    hidden_size = 64
    static_size = 7

    model = CreditDefaultPredictor(hidden_size=hidden_size, static_size=static_size)
    x_seq = torch.randn(batch_size, seq_len, input_size)
    x_static = torch.randn(batch_size, static_size)

    # Act
    output = model(x_seq, x_static)

    # Assert
    assert output.shape == (batch_size, 1)


def test_model_forward_different_batches() -> None:
    # Arrange
    model = CreditDefaultPredictor()

    # Test batch size of 1
    x_single_seq = torch.randn(1, 5, 3)
    x_single_static = torch.randn(1, 5)
    output_single = model(x_single_seq, x_single_static)
    assert output_single.shape == (1, 1)

    # Test batch size of 16
    x_large_seq = torch.randn(16, 20, 3)
    x_large_static = torch.randn(16, 5)
    output_large = model(x_large_seq, x_large_static)
    assert output_large.shape == (16, 1)
