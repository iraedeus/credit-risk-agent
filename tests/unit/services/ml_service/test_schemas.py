"""Unit tests for ML service Pydantic schemas."""

import re

import pydantic
import pytest

from credit_risk_agent.schemas.client_schemas import ClientPaymentHistory, ClientProfile
from credit_risk_agent.schemas.enums import Education, Marriage, Sex
from credit_risk_agent.services.ml_service.schemas import ClientProfileHistory, PredictionResponse


def _build_history(months: list[int]) -> list[ClientPaymentHistory]:
    """Build payment history records for the given month indexes."""
    return [
        ClientPaymentHistory(client_id=1, month=month, pay_status=-1, bill_amt=55000.0, pay_amt=18000.0)
        for month in months
    ]


class TestPredictionResponse:
    """Test suite for the PredictionResponse schema."""

    def test_nan_default_probability_rejected(self) -> None:
        """Test that a NaN default probability raises a ValidationError."""
        with pytest.raises(pydantic.ValidationError):
            PredictionResponse(default_probability=float("nan"))

    def test_inf_default_probability_rejected(self) -> None:
        """Test that an infinite default probability raises a ValidationError."""
        with pytest.raises(pydantic.ValidationError):
            PredictionResponse(default_probability=float("inf"))


class TestClientProfileHistory:
    """Test suite for the ClientProfileHistory schema."""

    def test_valid_history_passes_validation(self, profile_history) -> None:
        """Test that a 6-record history with unique months passes validation."""
        assert isinstance(profile_history, ClientProfileHistory)
        assert len(profile_history.history) == 6

    def test_history_with_five_records_rejected(self) -> None:
        """Test that a 5-record history raises an invalid-length error."""
        profile = ClientProfile(
            client_id=1,
            limit_bal=300000.0,
            age=35,
            sex=Sex.MALE,
            education=Education.UNIVERSITY,
            marriage=Marriage.MARRIED,
        )

        with pytest.raises(
            pydantic.ValidationError,
            match=re.escape("Invalid length of the client history. Should be equal to 6."),
        ):
            ClientProfileHistory(profile=profile, history=_build_history([1, 2, 3, 4, 5]))

    def test_history_with_duplicate_months_rejected(self) -> None:
        """Test that a history with duplicate months raises a uniqueness error."""
        profile = ClientProfile(
            client_id=1,
            limit_bal=300000.0,
            age=35,
            sex=Sex.MALE,
            education=Education.UNIVERSITY,
            marriage=Marriage.MARRIED,
        )

        with pytest.raises(
            pydantic.ValidationError,
            match=re.escape("Months in history must be unique."),
        ):
            ClientProfileHistory(profile=profile, history=_build_history([1, 1, 1, 1, 1, 1]))
