import torch

from credit_risk_agent.config import MODEL_SAVE_PATH, SCALER_COLS, SCALER_PATH
from credit_risk_agent.data.normalization import normalize
from credit_risk_agent.model.dataset import prepare_dataset
from credit_risk_agent.model.model import CreditDefaultPredictor
from credit_risk_agent.model.train import load_and_preprocess_test_data


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
    model = CreditDefaultPredictor(hidden_size=64, num_layers=1, static_size=14, dropout_prob=0.28)
    state_dict = torch.load(MODEL_SAVE_PATH)
    model.load_state_dict(state_dict)
    model.eval()

    test_df = load_and_preprocess_test_data()
    test_df = normalize(test_df, SCALER_COLS, SCALER_PATH)
    client_test_df = test_df[test_df["client_id"] == client_id]
    if len(client_test_df) == 0:
        return f"Клиент с id={client_id} не был найден в базе."

    test_dataset = prepare_dataset(client_test_df)

    with torch.no_grad():
        score = torch.sigmoid(model(test_dataset[0][0].unsqueeze(0), test_dataset[0][1].unsqueeze(0))).item()
        return f"Модель на клиенте с id={client_id} выдала результат равный {score:.4f}."
