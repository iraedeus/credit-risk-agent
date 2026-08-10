"""
Data repository providing database query abstractions for the Data Service.
"""

from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from credit_risk_agent.services.data_service.models import ClientDB, PaymentHistoryDB
from credit_risk_agent.services.data_service.schemas import (
    ClientFinancialMetrics,
    ClientFullInfo,
    ClientPaymentHistory,
    ClientProfile,
)


class DataRepository:
    """
    Repository layer for querying client demographics, payment histories, and financial metrics.

    Parameters
    ----------
    session : Session
        Active SQLAlchemy database session.
    """

    def __init__(self, session: Session) -> None:
        self.session = session

    def get_clients(self, limit: int = 20, offset: int = 0) -> list[int]:
        """
        Fetch a paginated list of client IDs sorted in ascending order.

        Parameters
        ----------
        limit : int, default=20
            Maximum number of client IDs to return.
        offset : int, default=0
            Number of client IDs to skip for pagination.

        Returns
        -------
        list of int
            List of client IDs retrieved from the database.
        """
        stmt = select(ClientDB.client_id).order_by(ClientDB.client_id).offset(offset).limit(limit)
        return list(self.session.scalars(stmt).all())

    def get_client_profile(self, client_id: int) -> ClientProfile | None:
        """
        Retrieve demographic and basic credit profile for a given client.

        Parameters
        ----------
        client_id : int
            Unique identifier of the target client.

        Returns
        -------
        ClientProfile or None
            Pydantic model containing client demographic details, or None if the client is not found.
        """
        client_row = self.session.get(ClientDB, client_id)

        if client_row is None:
            return None

        return ClientProfile.model_validate(client_row)

    def get_client_history(self, client_id: int) -> list[ClientPaymentHistory] | None:
        """
        Retrieve monthly payment history entries for a given client ordered by month.

        Parameters
        ----------
        client_id : int
            Unique identifier of the target client.

        Returns
        -------
        list of ClientPaymentHistory or None
            List of payment history records sorted by month, or None if no records exist.
        """
        stmt = select(PaymentHistoryDB).where(PaymentHistoryDB.client_id == client_id).order_by(PaymentHistoryDB.month)
        history_rows = self.session.scalars(stmt).all()

        if not history_rows:
            return None

        return [ClientPaymentHistory.model_validate(row) for row in history_rows]

    def get_client_financial(self, client_id: int) -> ClientFinancialMetrics | None:
        """
        Calculate aggregated financial metrics and delinquency statistics for a given client.

        Computes average bill, credit utilization ratios, average payment, total repayment rate,
        maximum payment delay status, and total count of delayed months across historical data.

        Parameters
        ----------
        client_id : int
            Unique identifier of the target client.

        Returns
        -------
        ClientFinancialMetrics or None
            Computed financial metrics for the client, or None if no record exists.
        """
        stmt = (
            select(
                ClientDB.limit_bal,
                func.avg(PaymentHistoryDB.bill_amt).label("avg_bill"),
                func.max(PaymentHistoryDB.bill_amt).label("max_bill"),
                func.avg(PaymentHistoryDB.pay_amt).label("avg_pay"),
                func.sum(PaymentHistoryDB.pay_amt).label("sum_pay"),
                func.sum(PaymentHistoryDB.bill_amt).label("sum_bill"),
                func.max(PaymentHistoryDB.pay_status).label("max_delay_status"),
                func.sum(case((PaymentHistoryDB.pay_status > 0, 1), else_=0)).label("delay_months_count"),
            )
            .join(PaymentHistoryDB, PaymentHistoryDB.client_id == ClientDB.client_id)
            .where(ClientDB.client_id == client_id)
            .group_by(ClientDB.client_id, ClientDB.limit_bal)
        )

        row = self.session.execute(stmt).first()

        if row is None:
            return None

        avg_utilization = (row.avg_bill / row.limit_bal * 100) if row.limit_bal else 0.0
        max_utilization = (row.max_bill / row.limit_bal * 100) if row.limit_bal else 0.0
        repayment_rate = (row.sum_pay / row.sum_bill * 100) if row.sum_bill and row.sum_bill > 0 else 0.0

        metrics_data = {
            "client_id": client_id,
            "limit_bal": row.limit_bal,
            "avg_bill": row.avg_bill,
            "avg_utilization": avg_utilization,
            "max_utilization": max_utilization,
            "avg_pay": row.avg_pay,
            "repayment_rate": repayment_rate,
            "max_delay_status": row.max_delay_status,
            "delay_months_count": row.delay_months_count,
        }

        return ClientFinancialMetrics.model_validate(metrics_data)

    def get_client_full(self, client_id: int) -> ClientFullInfo | None:
        """
        Retrieve combined client information including profile, payment history, and financial metrics.

        Parameters
        ----------
        client_id : int
            Unique identifier of the target client.

        Returns
        -------
        ClientFullInfo or None
            Aggregated client record containing profile, payment history, and metrics,
            or None if any component is missing.
        """
        profile = self.get_client_profile(client_id)
        history = self.get_client_history(client_id)
        metrics = self.get_client_financial(client_id)

        if profile is None or history is None or metrics is None:
            return None

        return ClientFullInfo(profile=profile, history=history, metrics=metrics)
