"""
ETL script to download Kaggle UCI Credit Card dataset and build SQLite database.
"""

import sqlite3

import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from credit_risk_agent.config import (
    CLIENT_COLUMNS,
    DATA_PATH,
    ID_COL,
    RAW_DATABASE_PATH,
    TARGET_COL,
    TEST_DATABASE_PATH,
    TRAIN_DATABASE_PATH,
)

load_dotenv()

import kaggle  # noqa: E402


def download_kaggle_dataset() -> None:
    """
    Authenticate with Kaggle API and download the UCI Credit Card dataset.

    Downloads and unzips the dataset archive into the configured DATA_PATH directory.
    """
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files("uciml/default-of-credit-card-clients-dataset", path=DATA_PATH, unzip=True)


def data_preprocessing() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw CSV data and reshape it into relational client and payment history structures.

    Renames payment columns and converts payment history from wide to long format.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing:
        - client_df: Client demographics, features, and target defaults.
        - final_history_df: Monthly payment status, bill amounts, and payment amounts in long format.
    """
    df = pd.read_csv(DATA_PATH / "UCI_Credit_Card.csv")
    df = df.rename(columns={"PAY_0": "PAY_1"})

    # Separate client features and payment history
    client_features = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", "default.payment.next.month"]
    client_df = df[["ID", *client_features]].copy()
    history_df = df.drop(columns=client_features)

    # Rename columns for the Client table
    client_df.columns = CLIENT_COLUMNS

    # Transform payment history, balances, and payment amounts from wide to long format in a single pass
    # Rename PAY_1, PAY_2... to PAY1, PAY2... to standardize suffixes for a single-pass pd.wide_to_long
    history_df_renamed = history_df.rename(
        columns=lambda x: x.replace("PAY_", "PAY") if x.startswith("PAY_") and not x.startswith("PAY_AMT") else x
    )

    final_history_df = pd.wide_to_long(
        history_df_renamed, stubnames=["PAY", "BILL_AMT", "PAY_AMT"], i="ID", j="month", sep=""
    ).reset_index()

    final_history_df = final_history_df.rename(
        columns={"ID": "client_id", "PAY": "pay_status", "BILL_AMT": "bill_amt", "PAY_AMT": "pay_amt"}
    )

    return client_df, final_history_df


def create_raw_sql_db(client_df: pd.DataFrame, history_df: pd.DataFrame) -> None:
    """
    Create raw SQLite database and populate `clients`, `ground_truth`, and `payment_history` tables.

    Parameters
    ----------
    client_df : pd.DataFrame
        DataFrame containing client demographic data and target labels.
    history_df : pd.DataFrame
        DataFrame containing long-format monthly payment records.
    """
    with sqlite3.connect(RAW_DATABASE_PATH) as conn:
        cursor = conn.cursor()

        cursor.execute("PRAGMA foreign_keys = ON;")
        cursor.execute("DROP TABLE IF EXISTS payment_history;")
        cursor.execute("DROP TABLE IF EXISTS ground_truth;")
        cursor.execute("DROP TABLE IF EXISTS clients;")

        cursor.execute("""CREATE TABLE ground_truth (
            client_id INTEGER PRIMARY KEY,
            'default' INTEGER,

            FOREIGN KEY (client_id) REFERENCES clients ON DELETE CASCADE
            );""")

        cursor.execute("""CREATE TABLE clients (
            client_id INTEGER PRIMARY KEY,
            limit_bal REAL,
            sex INTEGER,
            education INTEGER,
            marriage INTEGER,
            age INTEGER
            );""")

        cursor.execute("""CREATE TABLE payment_history (
            client_id INTEGER,
            month INTEGER,
            pay_status REAL,
            bill_amt REAL,
            pay_amt REAL,
            PRIMARY KEY (client_id, month),

            FOREIGN KEY (client_id) REFERENCES clients ON DELETE CASCADE
            );""")

        conn.commit()

        client_df.drop(columns=["default"]).to_sql("clients", conn, if_exists="append", index=False)
        client_df[["client_id", "default"]].to_sql("ground_truth", conn, if_exists="append", index=False)
        history_df.to_sql("payment_history", conn, if_exists="append", index=False)


def create_train_test_db() -> None:
    """
    Perform stratified train-test split on raw database records and create separate train and test SQLite databases.

    Splits ground truth records with an 80/20 ratio stratified by target default label and partition
    clients, ground_truth, and payment_history tables into TRAIN_DATABASE_PATH and TEST_DATABASE_PATH.
    """
    with sqlite3.connect(RAW_DATABASE_PATH) as raw_conn:
        raw_clients = pd.read_sql_query("SELECT * FROM clients", raw_conn)
        raw_history = pd.read_sql_query("SELECT * FROM payment_history", raw_conn)
        raw_gt = pd.read_sql_query("SELECT * FROM ground_truth", raw_conn)

        train_ids, test_ids = train_test_split(
            raw_gt[ID_COL], test_size=0.2, stratify=raw_gt[TARGET_COL], random_state=42
        )

        train_clients = raw_clients[raw_clients["client_id"].isin(train_ids)]
        train_history = raw_history[raw_history["client_id"].isin(train_ids)]
        train_gt = raw_gt[raw_gt["client_id"].isin(train_ids)]

        test_clients = raw_clients[raw_clients["client_id"].isin(test_ids)]
        test_history = raw_history[raw_history["client_id"].isin(test_ids)]
        test_gt = raw_gt[raw_gt["client_id"].isin(test_ids)]

    with sqlite3.connect(TRAIN_DATABASE_PATH) as train_conn:
        train_clients.to_sql("clients", train_conn, if_exists="replace", index=False)
        train_history.to_sql("payment_history", train_conn, if_exists="replace", index=False)
        train_gt.to_sql("ground_truth", train_conn, if_exists="replace", index=False)

    with sqlite3.connect(TEST_DATABASE_PATH) as test_conn:
        test_clients.to_sql("clients", test_conn, if_exists="replace", index=False)
        test_history.to_sql("payment_history", test_conn, if_exists="replace", index=False)
        test_gt.to_sql("ground_truth", test_conn, if_exists="replace", index=False)


def main() -> None:
    """
    Execute the full ETL pipeline for the Credit Card dataset.

    Downloads the UCI Credit Card dataset from Kaggle, preprocesses it, splits
    client characteristics and payment history into raw SQLite tables, partitions them into
    stratified train and test SQLite databases, fits and saves feature scalers, and cleans up temporary files.
    """
    download_kaggle_dataset()
    client_df, history_df = data_preprocessing()
    create_raw_sql_db(client_df, history_df)
    create_train_test_db()
    (DATA_PATH / "UCI_Credit_Card.csv").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
