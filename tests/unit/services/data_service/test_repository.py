from collections.abc import Generator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from credit_risk_agent.services.data_service.models import Base, ClientDB, PaymentHistoryDB
from credit_risk_agent.services.data_service.repository import DataRepository


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)

    testing_session_local = sessionmaker(bind=engine)
    session = testing_session_local()
    yield session
    session.close()


class TestDataRepository:
    def test_get_client_profile(self, db_session: Session):
        client = ClientDB(
            client_id=1,
            limit_bal=50000.0,
            sex=1,
            education=2,
            marriage=1,
            age=30,
        )

        db_session.add(client)
        db_session.commit()

        repo = DataRepository(db_session)
        profile = repo.get_client_profile(1)
        empty_profile = repo.get_client_profile(2)

        assert profile is not None
        assert profile.client_id == 1
        assert profile.limit_bal == 50000.0
        assert profile.age == 30
        assert profile.sex == 1
        assert profile.education == 2
        assert profile.marriage == 1

        assert empty_profile is None

    def test_get_client_history(self, db_session: Session):
        history_rows = [
            PaymentHistoryDB(client_id=1, month=i, pay_status=1, bill_amt=10000.0, pay_amt=100.0 * i)
            for i in range(1, 7)
        ]
        for row in history_rows:
            db_session.add(row)
            db_session.commit()

        repo = DataRepository(db_session)
        history = repo.get_client_history(1)
        empty_history = repo.get_client_history(2)

        assert history is not None
        assert len(history) == 6
        assert history[0].client_id == 1
        assert history[0].month == 1
        assert history[0].bill_amt == 10000.0
        assert history[0].pay_status == 1
        assert history[0].pay_amt == 100.0

        assert history[1].client_id == 1
        assert history[1].month == 2
        assert history[1].bill_amt == 10000.0
        assert history[1].pay_status == 1
        assert history[1].pay_amt == 200.0

        assert empty_history is None

    def test_get_financial_metrics(self, db_session: Session):
        client = ClientDB(
            client_id=1,
            limit_bal=100000.0,
            sex=1,
            education=2,
            marriage=1,
            age=30,
        )
        db_session.add(client)

        history_data = [
            (1, -1, 10000.0, 5000.0),
            (2, 1, 20000.0, 5000.0),
            (3, 2, 10000.0, 5000.0),
            (4, 0, 10000.0, 5000.0),
            (5, -1, 10000.0, 5000.0),
            (6, 0, 0.0, 5000.0),
        ]
        for month, status, bill, pay in history_data:
            db_session.add(
                PaymentHistoryDB(
                    client_id=1,
                    month=month,
                    pay_status=status,
                    bill_amt=bill,
                    pay_amt=pay,
                )
            )
        db_session.commit()

        repo = DataRepository(db_session)
        metrics = repo.get_client_financial(1)
        empty_metrics = repo.get_client_financial(2)

        assert metrics is not None
        assert metrics.client_id == 1
        assert metrics.limit_bal == 100000.0
        assert metrics.avg_bill == 10000.0
        assert metrics.avg_utilization == 10.0
        assert metrics.max_utilization == 20.0
        assert metrics.avg_pay == 5000.0
        assert round(metrics.repayment_rate, 1) == 50.0
        assert metrics.max_delay_status == 2
        assert metrics.delay_months_count == 2

        assert empty_metrics is None
