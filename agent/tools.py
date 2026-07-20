import sqlite3

from data.dataset_downloader import DATASET_PATH


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
