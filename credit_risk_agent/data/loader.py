import sqlite3
from pathlib import Path

from credit_risk_agent.data.preprocessing import preprocess
import pandas as pd


def load_and_preprocess_from_db(db_path: Path) -> pd.DataFrame:
    """
    Load raw relational tables from a SQLite database, merge them, and apply preprocessing.

    Parameters
    ----------
    db_path : Path
        Path to the SQLite database file containing `clients`, `payment_history`,
        and `ground_truth` tables.

    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame containing combined client features and payment history.
    """

    with sqlite3.connect(db_path) as conn:
        client_df = pd.read_sql_query("SELECT * FROM clients", conn)
        history_df = pd.read_sql_query("SELECT * FROM payment_history", conn)
        gt_df = pd.read_sql_query("SELECT * FROM ground_truth", conn)

        df = pd.merge(client_df, gt_df, on="client_id")
        df = pd.merge(df, history_df, on="client_id")
        df = preprocess(df)
        return df
