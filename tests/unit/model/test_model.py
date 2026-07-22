import pytest
import torch

from credit_risk_agent.model.model import CreditDefaultPredictor


class TestCreditDefaultPredictor:
    def test_model_initialization(self) -> None:
        """Verify correct initialization of GRU, linear layers, and dropout parameters."""
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

    def test_model_forward_shape(self) -> None:
        """Verify forward pass returns expected output tensor shape (batch_size, 1)."""
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

    @pytest.mark.parametrize(
        "batch_size,seq_len",
        [
            (1, 5),
            (16, 20),
        ],
    )
    def test_model_forward_batch_sizes(self, batch_size: int, seq_len: int) -> None:
        """Verify forward pass correctly processes various batch sizes and sequence lengths."""
        # Arrange
        model = CreditDefaultPredictor()
        x_seq = torch.randn(batch_size, seq_len, 3)
        x_static = torch.randn(batch_size, 5)

        # Act
        output = model(x_seq, x_static)

        # Assert
        assert output.shape == (batch_size, 1)
