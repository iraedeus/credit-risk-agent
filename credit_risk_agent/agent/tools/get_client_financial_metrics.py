import sqlite3

from credit_risk_agent.config import DATABASE_PATH


def get_client_financial_metrics(client_id: int) -> str:
    """
    Calculate and return key financial metrics for a specific client.

    Retrieves demographic and payment history data for the given client ID from
    the SQLite database, then computes aggregated financial metrics including
    credit limit utilization rates, repayment ratios, and delinquency statistics
    over the 6-month historical period.

    Parameters
    ----------
    client_id : int
        The unique identifier of the client to analyze.

    Returns
    -------
    str
        A formatted multi-line string containing the client's financial metrics
        (credit limit, average bill, utilization rates, average payment,
        repayment rate, maximum delay status, and delinquency month count),
        or an error/not-found message if the client data cannot be retrieved.
    """
    try:
        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT limit_bal, age FROM clients WHERE client_id = ?", (client_id,))
            client_row = cursor.fetchone()
            if not client_row:
                return f"Клиент с client_id = {client_id} не был найден в базе данных."

            limit_bal = client_row[0]

            cursor.execute(
                """
                           SELECT
                           AVG(bill_amt) as avg_bill,
                           MAX(bill_amt) AS max_bill,
                           AVG(pay_amt) AS avg_pay,
                           SUM(pay_amt) AS sum_pay,
                           SUM(bill_amt) as sum_bill,
                           MAX(pay_status) as max_delay_status,
                           SUM(CASE WHEN pay_status > 0 THEN 1 ELSE 0 END) as delay_months_count
                           FROM payment_history WHERE client_id = ?
                           """,
                (client_id,),
            )

            metrics = cursor.fetchone()
            if not metrics or metrics[0] is None:
                return f"Для клиента с client_id={client_id} отсутствует история платежей."

            avg_bill, max_bill, avg_pay, sum_pay, sum_bill, max_delay_status, delay_months_count = metrics

            avg_utilization = (avg_bill / limit_bal * 100) if limit_bal else 0
            max_utilization = (max_bill / limit_bal * 100) if limit_bal else 0
            repayment_rate = (sum_pay / sum_bill * 100) if sum_bill and sum_bill > 0 else 0

            return (
                f"Финансовые метрики клиента id={client_id}:\n"
                f"- Кредитный лимит: {limit_bal:,.2f}\n"
                f"- Средний ежемесячный счет (bill_amt): {avg_bill:,.2f}\n"
                f"- Средняя утилизация лимита: {avg_utilization:.1f}%\n"
                f"- Максимальная утилизация лимита: {max_utilization:.1f}%\n"
                f"- Средний ежемесячный платеж (pay_amt): {avg_pay:,.2f}\n"
                f"- Коэффициент покрытия выставляемых счетов (Repayment Rate): {repayment_rate:.1f}%\n"
                f"- Максимальный статус просрочки за 6 мес.: {max_delay_status}\n"
                f"- Количество месяцев с просрочкой: {delay_months_count} из 6"
            )

    except Exception as err:
        return f"Ошибка SQL: {err}"
