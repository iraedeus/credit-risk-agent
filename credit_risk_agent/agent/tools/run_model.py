"""
Model evaluation tool for predicting client credit default probability.
"""

import pandas as pd

from credit_risk_agent.config import BEST_MODEL_ALIAS, BEST_MODEL_NAME
from credit_risk_agent.model.loader import load_model_from_registry, load_scaler_from_registry
from credit_risk_agent.model.predictor import CreditRiskPredictor
from credit_risk_agent.services.data_service.client import get_data_service_client
from credit_risk_agent.services.data_service.exceptions import DataServiceHTTPError
from credit_risk_agent.services.data_service.schemas import ClientFullInfo


def client_full_info_to_df(full_info: ClientFullInfo) -> pd.DataFrame:
    """
    Convert a ClientFullInfo schema object into a pandas DataFrame.

    Parameters
    ----------
    full_info : ClientFullInfo
        Aggregated client record containing profile and payment history.

    Returns
    -------
    pd.DataFrame
        DataFrame formatted for CreditRiskPredictor model evaluation.
    """
    rows = []
    p = full_info.profile
    for h in full_info.history:
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


def run_model(client_id: int) -> str:
    """
    Run the credit default prediction model for a specified client.

    Fetches full client details via DataServiceClient microservice, converts features,
    and evaluates the pre-trained PyTorch CreditDefaultModel neural network.

    Parameters
    ----------
    client_id : int
        The unique identifier of the client for whom to predict credit default risk.

    Returns
    -------
    str
        A string message containing the model's predicted default risk score
        formatted as a float between 0.0 and 1.0, or an error message.
    """

    try:
        data_service_client = get_data_service_client()
        client_full_info = data_service_client.get_client(client_id)

        if client_full_info is None:
            return f"Клиент с client_id = {client_id} не был найден в базе данных."

        client_test_df = client_full_info_to_df(client_full_info)

        model = load_model_from_registry(BEST_MODEL_NAME, BEST_MODEL_ALIAS)
        scaler = load_scaler_from_registry(BEST_MODEL_NAME, BEST_MODEL_ALIAS)

        predictor = CreditRiskPredictor(model, scaler)
        score = predictor.predict_pd(client_test_df)

        return f"Модель на клиенте с id={client_id} выдала результат равный {score:.4f}."
    except DataServiceHTTPError as err:
        return f"Ошибка Data Service: {err}"
