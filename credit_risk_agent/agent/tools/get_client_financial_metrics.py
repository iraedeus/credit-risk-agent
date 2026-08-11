"""
Financial metrics calculation tool for client credit risk profiling.
"""

from credit_risk_agent.services.data_service.client import get_data_service_client
from credit_risk_agent.services.data_service.exceptions import DataServiceHTTPError


def get_client_financial_metrics(client_id: int) -> str:
    """
    Calculate and return key financial metrics for a specific test client.

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
        data_service_client = get_data_service_client()
        metrics = data_service_client.get_client_metrics(client_id)

        if metrics is None:
            return f"Клиент с client_id = {client_id} не был найден в базе данных."

        return (
            f"Финансовые метрики клиента id={metrics.client_id}:\n"
            f"- Кредитный лимит: {metrics.limit_bal:,.2f}\n"
            f"- Средний ежемесячный счет (bill_amt): {metrics.avg_bill:,.2f}\n"
            f"- Средняя утилизация лимита: {metrics.avg_utilization:.1f}%\n"
            f"- Максимальная утилизация лимита: {metrics.max_utilization:.1f}%\n"
            f"- Средний ежемесячный платеж (pay_amt): {metrics.avg_pay:,.2f}\n"
            f"- Коэффициент покрытия выставляемых счетов (Repayment Rate): {metrics.repayment_rate:.1f}%\n"
            f"- Максимальный статус просрочки за 6 мес.: {metrics.max_delay_status}\n"
            f"- Количество месяцев с просрочкой: {metrics.delay_months_count} из 6"
        )
    except DataServiceHTTPError as err:
        return f"Ошибка Data Service: {err}"
