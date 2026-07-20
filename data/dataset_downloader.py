import sqlite3
from pathlib import Path

import kaggle
import pandas as pd
from dotenv import load_dotenv

__ROOT__ = Path(__file__).parent.parent
DATASET_PATH = __ROOT__ / "data"
CLIENT_COLUMNS = ["client_id", "limit_bal", "sex", "education", "marriage", "age", "default"]
HISTORY_COLUMNS = ["client_id", "month", "pay_status", "bill_amt", "pay_amt"]


load_dotenv()


def wide_to_long(df: pd.DataFrame, column_prefix: str) -> pd.DataFrame:
    """
    Transform columns matching a prefix from wide format to long format.

    Identifies all columns that start with the given prefix followed by digits,
    melts them into a long format using the 'ID' column as identifier, and converts
    the column names into integer month numbers.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the wide format data.
    column_prefix : str
        The prefix to filter columns by (e.g., 'PAY_', 'BILL_AMT').

    Returns
    -------
    pd.DataFrame
        A DataFrame in long format containing the 'ID', 'month', and melted column.
    """

    # Find all columns matching the prefix (e.g., PAY_1, PAY_2...)
    value_vars = [
        col for col in df.columns if col.startswith(column_prefix) and col.replace(column_prefix, "").isdigit()
    ]
    if not value_vars:
        raise KeyError(f"No columns matching prefix {column_prefix} found.")

    # Select only the ID and matching columns to behave like melt
    df_sub = df[["ID", *value_vars]]

    # If the prefix has a trailing underscore (like PAY_), we temporarily rename the columns to remove it
    if column_prefix.endswith("_"):
        stub = column_prefix[:-1]
        sep = "_"
    else:
        stub = column_prefix
        sep = ""

    df_long = pd.wide_to_long(df_sub, stubnames=[stub], i="ID", j="month", sep=sep).reset_index()

    # If we stripped the trailing underscore, rename the column back
    if column_prefix.endswith("_"):
        df_long = df_long.rename(columns={stub: column_prefix})

    return df_long


def main() -> None:
    """
    Execute the ETL pipeline for the Credit Card dataset.

    Downloads the UCI Credit Card dataset from Kaggle, preprocesses it, splits
    client characteristics and payment history into separate long-format structures,
    saves them into a local SQLite database, and cleans up the temporary files.
    """

    # 1. Download dataset from Kaggle
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files("uciml/default-of-credit-card-clients-dataset", path=DATASET_PATH, unzip=True)

    # 2. Data preprocessing
    df = pd.read_csv(DATASET_PATH / "UCI_Credit_Card.csv")
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
    with sqlite3.connect(DATASET_PATH / "database.db") as conn:
        cursor = conn.cursor()

        cursor.execute("PRAGMA foreign_keys = ON;")
        cursor.execute("DROP TABLE IF EXISTS payment_history;")
        cursor.execute("DROP TABLE IF EXISTS clients;")

        cursor.execute("""CREATE TABLE clients (
            client_id INTEGER PRIMARY KEY,
            limit_bal REAL,
            sex INTEGER,
            education INTEGER,
            marriage INTEGER,
            age INTEGER,
            'default' INTEGER
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

        client_df.to_sql("clients", conn, if_exists="append", index=False)
        final_history_df.to_sql("payment_history", conn, if_exists="append", index=False)

    # 4. Delete the temporary CSV file
    (DATASET_PATH / "UCI_Credit_Card.csv").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
