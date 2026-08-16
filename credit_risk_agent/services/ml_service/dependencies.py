"""
ML Service request-to-features conversion utilities.
"""

import pandas as pd

from credit_risk_agent.services.ml_service.schemas import ClientProfileHistory


def client_profile_history_to_df(profile_history: ClientProfileHistory) -> pd.DataFrame:
    """
    Convert a client profile with payment history into a raw feature DataFrame.

    One row is produced per payment history record, with the client's
    demographic profile replicated across all rows.

    Parameters
    ----------
    profile_history : ClientProfileHistory
        Client profile and 6-month payment history records.

    Returns
    -------
    pd.DataFrame
        Raw feature DataFrame with columns matching the model input schema.
    """
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
