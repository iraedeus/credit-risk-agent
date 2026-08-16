"""Shared fixtures for ML service unit tests."""

import pytest

from credit_risk_agent.schemas.client_schemas import ClientPaymentHistory, ClientProfile
from credit_risk_agent.schemas.enums import Education, Marriage, Sex
from credit_risk_agent.services.ml_service.schemas import ClientProfileHistory


@pytest.fixture
def profile_history() -> ClientProfileHistory:
    """
    Build a valid ClientProfileHistory with 6 unique-month payment records.

    Returns
    -------
    ClientProfileHistory
        A fully populated profile with 6 payment history records (months 1..6).
    """
    return ClientProfileHistory(
        profile=ClientProfile(
            client_id=1,
            limit_bal=300000.0,
            age=35,
            sex=Sex.MALE,
            education=Education.UNIVERSITY,
            marriage=Marriage.MARRIED,
        ),
        history=[
            ClientPaymentHistory(client_id=1, month=month, pay_status=-1, bill_amt=55000.0, pay_amt=18000.0)
            for month in range(1, 7)
        ],
    )


@pytest.fixture
def profile_history_payload(profile_history: ClientProfileHistory) -> dict:
    """
    Build the JSON-serializable payload of a valid ClientProfileHistory.

    Parameters
    ----------
    profile_history : ClientProfileHistory
        The profile history fixture to serialize.

    Returns
    -------
    dict
        JSON-compatible payload produced via ``model_dump(mode="json")``.
    """
    return profile_history.model_dump(mode="json")
