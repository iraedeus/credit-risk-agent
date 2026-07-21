import sqlite3

import torch

from data.dataset_downloader import DATASET_PATH
from data.normalization import normalize
from model.dataset import prepare_dataset
from model.model import CreditDefaultPredictor
from model.train import MODEL_SAVE_PATH, SCALER_COLS, SCALER_PATH, load_and_preprocess_test_data


def sql_query(query: str) -> str:
    """
    Execute a SQL query against the SQLite database and return the results.

    This tool allows the agent to retrieve client demographic data, credit limits,
    and historical payment information from the SQLite database.

    Available Tables
    ----------------
    clients
        Demographic and credit limit details for clients.
        Columns: client_id (int), limit_bal (float), sex (int), education (int),
        marriage (int), age (int), default (int).
    payment_history
        Monthly billing and payment history (up to 6 months per client).
        Columns: client_id (int), month (int, 1 to 6), pay_status (float),
        bill_amt (float), pay_amt (float).

    Parameters
    ----------
    query : str
        The SQL SELECT query string to execute.

    Returns
    -------
    str
        A comma-separated values (CSV-like) string representation of the rows
        with column headers, or an error message if the query fails.
    """
    try:
        with sqlite3.connect(DATASET_PATH / "database.db") as conn:
            cursor = conn.cursor()
            cursor.execute(query)

            rows = cursor.fetchmany(100)

            description = cursor.description

            if description:
                columns = [desc[0] for desc in cursor.description]
                header = ", ".join(columns)
                text_rows = []
                for row in rows:
                    text_rows.append(", ".join(str(val) for val in row))

                results = header + "\n" + "\n".join(text_rows)
                return results
            else:
                return "Запрос выполнен успешно"
    except Exception as err:
        return f"Ошибка SQL: {err}"


def get_client_financial_metrics(client_id: int) -> str:
    return ""


def run_model(client_id: int) -> str:
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
