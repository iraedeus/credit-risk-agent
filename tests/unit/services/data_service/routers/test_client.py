from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import StaticPool, create_engine
from sqlalchemy.orm import Session, sessionmaker

from credit_risk_agent.schemas.data_schemas import ClientFullInfo
from credit_risk_agent.services.data_service.dependencies import get_db
from credit_risk_agent.services.data_service.main import app
from credit_risk_agent.services.data_service.models import Base, ClientDB, PaymentHistoryDB


@pytest.fixture
def empty_db_session() -> Generator[Session, None, None]:
    """Provide an isolated in-memory SQLite database session with created tables."""

    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool, echo=False
    )
    Base.metadata.create_all(engine)

    testing_session_local = sessionmaker(bind=engine)
    session = testing_session_local()
    yield session
    session.close()


@pytest.fixture
def clients_empty_history_db(empty_db_session: Session) -> Generator[Session, None, None]:
    """Populate the test database session with multiple sample client records."""

    for client_id in range(1, 20, 2):
        empty_db_session.add(ClientDB(client_id=client_id, limit_bal=100000.0, sex=1, education=2, marriage=1, age=30))
    empty_db_session.commit()
    yield empty_db_session


@pytest.fixture
def full_db(empty_db_session: Session) -> Generator[Session, None, None]:
    """Populate the test database session with client demographic and payment history records."""

    client = ClientDB(client_id=1, limit_bal=100000.0, sex=1, education=2, marriage=1, age=30)
    empty_db_session.add(client)

    for month in range(1, 7):
        empty_db_session.add(PaymentHistoryDB(client_id=1, month=month, pay_status=1, bill_amt=10000.0, pay_amt=5000.0))
    empty_db_session.commit()
    empty_db_session.add(ClientDB(client_id=3, limit_bal=50000.0, sex=1, education=1, marriage=2, age=25))
    empty_db_session.commit()
    yield empty_db_session


@pytest.fixture
def empty_client(empty_db_session: Session):
    """Provide a TestClient instance configured with an empty database session."""

    app.dependency_overrides[get_db] = lambda: empty_db_session
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def clients_empty_history_client(clients_empty_history_db: Session):
    """Provide a TestClient instance configured with a populated database session."""

    app.dependency_overrides[get_db] = lambda: clients_empty_history_db
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def full_client(full_db: Session):
    """Provide a TestClient instance configured with a populated database session."""

    app.dependency_overrides[get_db] = lambda: full_db
    yield TestClient(app)
    app.dependency_overrides.clear()


class TestGetClients:
    """Integration test suite for the GET /api/v1/clients/ endpoint."""

    def test_get_clients_status_ok(self, clients_empty_history_client: TestClient):
        """Test that GET /api/v1/clients/ returns HTTP 200 status code."""

        response = clients_empty_history_client.get("/api/v1/clients/")
        assert response.status_code == 200

    def test_get_clients_empty_db(self, empty_client: TestClient):
        """Test that fetching clients from an empty database returns an empty list and HTTP 200."""

        response = empty_client.get("/api/v1/clients/")
        assert response.status_code == 200
        assert response.json() == []

    def test_get_clients_diff_limit(self, clients_empty_history_client: TestClient):
        """Test that limit query parameter controls the number of returned client records."""

        response_lim_3 = clients_empty_history_client.get("/api/v1/clients/", params={"limit": 3})
        response_lim_7 = clients_empty_history_client.get("/api/v1/clients/", params={"limit": 7})

        assert response_lim_3.json() == [1, 3, 5]
        assert response_lim_7.json() == [1, 3, 5, 7, 9, 11, 13]

    def test_get_clients_diff_offset(self, clients_empty_history_client: TestClient):
        """Test that offset query parameter skips the specified number of client records."""

        response_off_3 = clients_empty_history_client.get("/api/v1/clients/", params={"limit": 3, "offset": 3})
        response_off_4 = clients_empty_history_client.get("/api/v1/clients/", params={"limit": 3, "offset": 4})

        assert response_off_3.json() == [7, 9, 11]
        assert response_off_4.json() == [9, 11, 13]

    def test_get_clients_limit_gt_100(self, clients_empty_history_client: TestClient):
        """Test that limit greater than 100 triggers HTTP 422 validation error."""

        response = clients_empty_history_client.get("/api/v1/clients/", params={"limit": 99999})
        assert response.status_code == 422

    def test_get_clients_limit_lt_0(self, clients_empty_history_client: TestClient):
        """Test that negative limit triggers HTTP 422 validation error."""

        response = clients_empty_history_client.get("/api/v1/clients/", params={"limit": -10})
        assert response.status_code == 422

    def test_get_clients_offset_lt_0(self, clients_empty_history_client: TestClient):
        """Test that negative offset triggers HTTP 422 validation error."""

        response = clients_empty_history_client.get("/api/v1/clients/", params={"offset": -10})
        assert response.status_code == 422


class TestGetClient:
    """Integration test suite for the GET /api/v1/clients/{client_id}/ endpoint."""

    def test_get_client_success(self, full_client: TestClient):
        """Test that fetching an existing client returns HTTP 200 OK and full client details."""

        response = full_client.get("/api/v1/clients/1")
        assert response.status_code == 200

        data = response.json()
        client_info = ClientFullInfo.model_validate(data)
        assert client_info.profile.client_id == 1

    def test_get_client_not_found(self, empty_client: TestClient):
        """Test that fetching a non-existent client returns HTTP 404 Not Found."""

        response = empty_client.get("/api/v1/clients/1")
        assert response.status_code == 404

    def test_get_client_empty_history(self, clients_empty_history_client: TestClient):
        """Test that fetching a client without payment history returns HTTP 404 Not Found."""

        response = clients_empty_history_client.get("/api/v1/clients/1")
        assert response.status_code == 404

    def test_get_client_invalid_id(self, full_client: TestClient):
        """Test that non-integer client ID triggers HTTP 422 validation error."""

        response = full_client.get("/api/v1/clients/abc")
        assert response.status_code == 422
