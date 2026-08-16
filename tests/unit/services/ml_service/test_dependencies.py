from credit_risk_agent.schemas.client_schemas import ClientPaymentHistory, ClientProfile
from credit_risk_agent.schemas.enums import Education, Marriage, Sex
from credit_risk_agent.services.ml_service.dependencies import client_profile_history_to_df
from credit_risk_agent.services.ml_service.schemas import ClientProfileHistory


def test_client_profile_history_to_df():
    """
    Test that client_profile_history_to_df builds the expected feature DataFrame.

    Verifies that the resulting DataFrame replicates the client profile across
    six rows, one per payment history record, with all expected columns and values.
    """
    profile = ClientProfile(
        client_id=1,
        limit_bal=2000,
        sex=Sex(1),
        education=Education(2),
        marriage=Marriage(1),
        age=20,
    )

    history = [
        ClientPaymentHistory(client_id=1, month=1, pay_status=1, bill_amt=100, pay_amt=200),
        ClientPaymentHistory(client_id=1, month=2, pay_status=0, bill_amt=100, pay_amt=300),
        ClientPaymentHistory(client_id=1, month=3, pay_status=0, bill_amt=100, pay_amt=300),
        ClientPaymentHistory(client_id=1, month=4, pay_status=0, bill_amt=100, pay_amt=300),
        ClientPaymentHistory(client_id=1, month=5, pay_status=0, bill_amt=100, pay_amt=300),
        ClientPaymentHistory(client_id=1, month=6, pay_status=0, bill_amt=100, pay_amt=300),
    ]

    profile_history = ClientProfileHistory(profile=profile, history=history)

    df = client_profile_history_to_df(profile_history)

    assert len(df) == len(history)

    assert (df["client_id"] == 1).all()
    assert (df["sex"] == 1).all()
    assert (df["education"] == 2).all()
    assert (df["marriage"] == 1).all()
    assert (df["age"] == 20).all()
    assert (df["limit_bal"] == 2000).all()

    assert df["month"].tolist() == [1, 2, 3, 4, 5, 6]
    assert df["pay_status"].tolist() == [1, 0, 0, 0, 0, 0]
    assert df["bill_amt"].tolist() == [100, 100, 100, 100, 100, 100]
    assert df["pay_amt"].tolist() == [200, 300, 300, 300, 300, 300]
