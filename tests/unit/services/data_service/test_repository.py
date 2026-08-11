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


@pytest.fixture
def seeded_db(db_session: Session) -> Generator[Session, None, None]:
    client = ClientDB(client_id=1, limit_bal=100000.0, sex=1, education=2, marriage=1, age=30)
    db_session.add(client)

    for month in range(1, 7):
        db_session.add(PaymentHistoryDB(client_id=1, month=month, pay_status=1, bill_amt=10000.0, pay_amt=5000.0))
    db_session.commit()
    db_session.add(ClientDB(client_id=3, limit_bal=50000.0, sex=1, education=1, marriage=2, age=25))
    db_session.commit()
    yield db_session


@pytest.fixture
def many_client_db(db_session: Session) -> Generator[Session, None, None]:
    for client_id in range(1, 20, 2):
        db_session.add(ClientDB(client_id=client_id, limit_bal=100000.0, sex=1, education=2, marriage=1, age=30))
    db_session.commit()
    yield db_session


class TestDataRepository:
    def test_get_clients(self, many_client_db: Session):
        repo = DataRepository(many_client_db)
        clients_off_0 = repo.get_clients(5, 0)
        clients_off_3 = repo.get_clients(5, 3)
        clients_lim_2 = repo.get_clients(2, 0)
        clients_lim_7 = repo.get_clients(7, 0)

        assert clients_off_0 == [1, 3, 5, 7, 9]
        assert clients_off_3 == [7, 9, 11, 13, 15]
        assert clients_lim_2 == [1, 3]
        assert clients_lim_7 == [1, 3, 5, 7, 9, 11, 13]

    def test_get_clients_with_big_offset(self, many_client_db: Session):
        repo = DataRepository(many_client_db)
        clients = repo.get_clients(0, 100)

        assert clients == []

    def test_get_client_profile(self, seeded_db: Session):
        repo = DataRepository(seeded_db)
        profile = repo.get_client_profile(1)

        assert profile is not None
        assert profile.client_id == 1
        assert profile.limit_bal == 100000.0
        assert profile.education == 2

        assert repo.get_client_profile(2) is None

    def test_get_client_history(self, seeded_db: Session):
        repo = DataRepository(seeded_db)
        history = repo.get_client_history(1)

        assert history is not None
        assert len(history) == 6
        assert history[0].client_id == 1
        assert history[0].month == 1
        assert history[0].bill_amt == 10000.0

        assert history[1].client_id == 1
        assert history[1].month == 2
        assert history[1].pay_status == 1

        assert repo.get_client_history(2) is None

    def test_get_financial_metrics(self, seeded_db: Session):
        repo = DataRepository(seeded_db)
        metrics = repo.get_client_financial(1)

        assert metrics is not None
        assert metrics.client_id == 1
        assert metrics.limit_bal == 100000.0
        assert metrics.avg_bill == 10000.0
        assert metrics.max_utilization == 10.0
        assert metrics.max_delay_status == 1

        assert repo.get_client_financial(2) is None

    def test_get_client_full(self, seeded_db: Session):
        repo = DataRepository(seeded_db)
        full_info = repo.get_client_full(1)

        assert full_info is not None

        assert full_info.metrics is not None
        assert full_info.metrics.client_id == 1
        assert full_info.metrics.limit_bal == 100000.0
        assert full_info.metrics.max_delay_status == 1

        assert full_info.profile is not None
        assert full_info.profile.client_id == 1
        assert full_info.profile.age == 30

        assert full_info.history is not None
        assert len(full_info.history) == 6
        assert full_info.history[0].client_id == 1
        assert full_info.history[0].month == 1
        assert full_info.history[0].pay_status == 1

        assert full_info.history[1].client_id == 1
        assert full_info.history[1].month == 2
        assert full_info.history[1].pay_amt == 5000.0

        assert repo.get_client_full(2) is None
        assert repo.get_client_full(3) is None
