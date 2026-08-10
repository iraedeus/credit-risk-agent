from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import StaticPool, create_engine
from sqlalchemy.orm import Session, sessionmaker

from credit_risk_agent.services.data_service.dependencies import get_db
from credit_risk_agent.services.data_service.main import app
from credit_risk_agent.services.data_service.models import Base, ClientDB


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
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
def many_client_db(db_session: Session) -> Generator[Session, None, None]:
    """Populate the test database session with multiple sample client records."""

    for client_id in range(1, 20, 2):
        db_session.add(ClientDB(client_id=client_id, limit_bal=100000.0, sex=1, education=2, marriage=1, age=30))
    db_session.commit()
    yield db_session


@pytest.fixture
def db_client(many_client_db: Session):
    """Provide a TestClient instance configured with a populated database session."""

    app.dependency_overrides[get_db] = lambda: many_client_db
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def empty_db_client(db_session: Session):
    """Provide a TestClient instance configured with an empty database session."""

    app.dependency_overrides[get_db] = lambda: db_session
    yield TestClient(app)
    app.dependency_overrides.clear()


class TestGetClients:
    """Integration test suite for the GET /api/v1/clients/ endpoint."""

    def test_get_clients_status_ok(self, db_client: TestClient):
        """Test that GET /api/v1/clients/ returns HTTP 200 status code."""

        response = db_client.get("/api/v1/clients/")
        assert response.status_code == 200

    def test_get_clients_empty_db(self, empty_db_client: TestClient):
        """Test that fetching clients from an empty database returns an empty list and HTTP 200."""

        response = empty_db_client.get("/api/v1/clients/")
        assert response.status_code == 200
        assert response.json() == []

    def test_get_clients_diff_limit(self, db_client: TestClient):
        """Test that limit query parameter controls the number of returned client records."""

        response_lim_3 = db_client.get("/api/v1/clients/", params={"limit": 3})
        response_lim_7 = db_client.get("/api/v1/clients/", params={"limit": 7})

        assert response_lim_3.json() == [1, 3, 5]
        assert response_lim_7.json() == [1, 3, 5, 7, 9, 11, 13]

    def test_get_clients_diff_offset(self, db_client: TestClient):
        """Test that offset query parameter skips the specified number of client records."""

        response_off_3 = db_client.get("/api/v1/clients/", params={"limit": 3, "offset": 3})
        response_off_4 = db_client.get("/api/v1/clients/", params={"limit": 3, "offset": 4})

        assert response_off_3.json() == [7, 9, 11]
        assert response_off_4.json() == [9, 11, 13]

    def test_get_clients_limit_gt_100(self, db_client: TestClient):
        """Test that limit greater than 100 triggers HTTP 422 validation error."""

        response = db_client.get("/api/v1/clients/", params={"limit": 99999})
        assert response.status_code == 422

    def test_get_clients_limit_lt_0(self, db_client: TestClient):
        """Test that negative limit triggers HTTP 422 validation error."""

        response = db_client.get("/api/v1/clients/", params={"limit": -10})
        assert response.status_code == 422

    def test_get_clients_offset_lt_0(self, db_client: TestClient):
        """Test that negative offset triggers HTTP 422 validation error."""

        response = db_client.get("/api/v1/clients/", params={"offset": -10})
        assert response.status_code == 422
