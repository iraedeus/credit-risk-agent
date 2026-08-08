from credit_risk_agent.config import TEST_DATABASE_PATH
from credit_risk_agent.data.loader import load_and_preprocess_from_db
from credit_risk_agent.model.loader import load_model_from_mlflow, load_scaler_from_mlflow
from credit_risk_agent.model.model import CreditRiskModel


def run_model(client_id: int) -> str:
    """
    Run the credit default prediction model for a specified client.

    Loads the pre-trained CreditDefaultPredictor PyTorch model, fetches and
    preprocesses the client's test features, and evaluates the neural network
    to obtain a credit default risk score (probability).

    Parameters
    ----------
    client_id : int
        The unique identifier of the client for whom to predict credit default risk.

    Returns
    -------
    str
        A string message containing the model's predicted default risk score
        formatted as a float between 0.0 and 1.0, or an error message if the client
        is not found in the test dataset.
    """

    test_df = load_and_preprocess_from_db(TEST_DATABASE_PATH)

    client_test_df = test_df[test_df["client_id"] == client_id]
    if len(client_test_df) == 0:
        return f"Клиент с id={client_id} не был найден в базе."

    model = load_model_from_mlflow()
    scaler = load_scaler_from_mlflow()

    risk_model = CreditRiskModel(model, scaler)
    score = risk_model.predict_pd(client_test_df)

    return f"Модель на клиенте с id={client_id} выдала результат равный {score:.4f}."
