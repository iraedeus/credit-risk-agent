"""
Model evaluation tool for predicting client credit default probability.
"""

from credit_risk_agent.services.data_service.client import get_data_service_client
from credit_risk_agent.services.data_service.exceptions import DataServiceHTTPError
from credit_risk_agent.services.ml_service.client import get_ml_service_client
from credit_risk_agent.services.ml_service.exceptions import MLServiceHTTPError
from credit_risk_agent.services.ml_service.schemas import ClientProfileHistory


def run_model(client_id: int) -> str:
    """
    Run the credit default prediction model for a specified client.

    Fetches full client details via the data service microservice and sends them
    to the ML inference microservice to compute the default probability.

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
        ml_service_client = get_ml_service_client()
        client_full_info = data_service_client.get_client(client_id)

        if client_full_info is None:
            return f"Клиент с client_id = {client_id} не был найден в базе данных."

        profile_history = ClientProfileHistory(profile=client_full_info.profile, history=client_full_info.history)
        try:
            score = ml_service_client.predict(profile_history).default_probability
        except MLServiceHTTPError as err:
            return f"Ошибка ML Service: {err}"

        return f"Модель на клиенте с id={client_id} выдала результат равный {score:.4f}."
    except DataServiceHTTPError as err:
        return f"Ошибка Data Service: {err}"
