import sqlite3

import pandas as pd
from dotenv import load_dotenv

from credit_risk_agent.config import CLIENT_COLUMNS, DATA_PATH, DATABASE_PATH

load_dotenv()

import kaggle  # noqa: E402


def main() -> None:
    """
    Execute the ETL pipeline for the Credit Card dataset.

    Downloads the UCI Credit Card dataset from Kaggle, preprocesses it, splits
    client characteristics and payment history into separate long-format structures,
    saves them into a local SQLite database, and cleans up the temporary files.
    """

    # 1. Download dataset from Kaggle
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files("uciml/default-of-credit-card-clients-dataset", path=DATA_PATH, unzip=True)

    # 2. Data preprocessing
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

    # 3. Save data into the SQLite database
    with sqlite3.connect(DATABASE_PATH) as conn:
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
        final_history_df.to_sql("payment_history", conn, if_exists="append", index=False)

    # 4. Delete the temporary CSV file
    (DATA_PATH / "UCI_Credit_Card.csv").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
