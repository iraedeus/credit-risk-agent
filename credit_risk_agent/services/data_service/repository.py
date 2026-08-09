from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from credit_risk_agent.schemas.data_schemas import (
    ClientFinancialMetrics,
    ClientFullInfo,
    ClientPaymentHistory,
    ClientProfile,
)
from credit_risk_agent.services.data_service.models import ClientDB, PaymentHistoryDB


class DataRepository:
    def __init__(self, session: Session):
        self.session = session

    def get_client_profile(self, client_id: int) -> ClientProfile | None:
        client_row = self.session.get(ClientDB, client_id)

        if client_row is None:
            return None

        return ClientProfile.model_validate(client_row)

    def get_client_history(self, client_id: int) -> list[ClientPaymentHistory] | None:
        stmt = select(PaymentHistoryDB).where(PaymentHistoryDB.client_id == client_id).order_by(PaymentHistoryDB.month)
        history_rows = list(self.session.scalars(stmt).all())

        if not history_rows:
            return None

        return [ClientPaymentHistory.model_validate(row) for row in history_rows]

    def get_client_financial(self, client_id: int) -> ClientFinancialMetrics | None:
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
        profile = self.get_client_profile(client_id)
        history = self.get_client_history(client_id)
        metrics = self.get_client_financial(client_id)

        if profile is None or history is None or metrics is None:
            return None

        return ClientFullInfo(profile=profile, history=history, metrics=metrics)
