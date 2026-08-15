import pandas as pd

from credit_risk_agent.services.ml_service.schemas import ClientProfileHistory


def client_profile_history_to_df(profile_history: ClientProfileHistory) -> pd.DataFrame:
    rows = []
    p = profile_history.profile
    for h in profile_history.history:
        rows.append(
            {
                "client_id": p.client_id,
                "limit_bal": p.limit_bal,
                "sex": int(p.sex),
                "education": int(p.education),
                "marriage": int(p.marriage),
                "age": p.age,
                "month": h.month,
                "pay_status": h.pay_status,
                "bill_amt": h.bill_amt,
                "pay_amt": h.pay_amt,
            }
        )
    return pd.DataFrame(rows)
