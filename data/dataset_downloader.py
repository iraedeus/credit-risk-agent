import kaggle
import sqlite3
import pandas as pd
from pathlib import Path

__ROOT__ = Path(__file__).parent.parent
DATASET_PATH = __ROOT__ / "data"
CLIENT_COLUMNS = ["client_id", "limit_bal", "sex", "education", "marriage", "age", "default"]
HISTORY_COLUMNS = ["client_id", "month", "pay_status", "bill_amt", "pay_amt"]

def wide_to_long(df: pd.DataFrame, column_prefix: str) -> pd.DataFrame:
    # Находим все колонки, соответствующие префиксу (например, PAY_1, PAY_2...)
    value_vars = [
        col for col in df.columns
        if col.startswith(column_prefix) and col.replace(column_prefix, "").isdigit()
    ]

    # Преобразуем из широкого формата в длинный
    df_long = df.melt(id_vars=["ID"], value_vars=value_vars, var_name="month", value_name=column_prefix)
    # Превращаем название колонки (например, 'PAY_1') в номер месяца (1)
    df_long["month"] = df_long["month"].str.replace(column_prefix, "").astype(int)
    return df_long

def main():
# 1. Скачивание датасета с Kaggle
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('uciml/default-of-credit-card-clients-dataset', path=DATASET_PATH, unzip=True)

# 2. Обработка данных
    df = pd.read_csv(DATASET_PATH / "UCI_Credit_Card.csv")
    df = df.rename(columns={"PAY_0": "PAY_1"})

# Разделяем признаки клиента и историю платежей
    client_features = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", "default.payment.next.month"]
    client_df = df[["ID"] + client_features].copy()
    history_df = df.drop(columns=client_features)

# Переименовываем колонки для таблицы Client
    client_df.columns = CLIENT_COLUMNS

# Преобразуем историю платежей, балансов и оплат из широкого формата в длинный
    pay_df = wide_to_long(history_df, "PAY_")
    bill_df = wide_to_long(history_df, "BILL_AMT")
    pay_amt_df = wide_to_long(history_df, "PAY_AMT")

# Объединяем историю платежей в единую таблицу
    final_history_df = pay_df.merge(bill_df, on=["ID", "month"]).merge(pay_amt_df, on=["ID", "month"])
    final_history_df = final_history_df.rename(columns={
        "ID": "client_id",
        "PAY_": "pay_status",
        "BILL_AMT": "bill_amt",
        "PAY_AMT": "pay_amt"
    })

# 3. Сохранение в базу данных SQLite
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

# 4. Удаление временного CSV-файла
    (DATASET_PATH / "UCI_Credit_Card.csv").unlink(missing_ok=True)

if __name__ == "__main__":
    main()
