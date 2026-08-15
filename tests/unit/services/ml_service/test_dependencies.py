from credit_risk_agent.schemas.client_schemas import ClientPaymentHistory, ClientProfile
from credit_risk_agent.schemas.enums import Education, Marriage, Sex
from credit_risk_agent.services.ml_service.dependencies import client_profile_history_to_df
from credit_risk_agent.services.ml_service.schemas import ClientProfileHistory


def test_client_profile_history_to_df():
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
    ]

    profile_history = ClientProfileHistory(profile=profile, history=history)

    df = client_profile_history_to_df(profile_history)

    assert len(df) == len(history)

    assert (df["sex"] == 1).all()
    assert (df["limit_bal"] == 2000).all()

    assert df["month"].tolist() == [1, 2]
    assert df["pay_status"].tolist() == [1, 0]
    assert df["bill_amt"].tolist() == [100, 100]
