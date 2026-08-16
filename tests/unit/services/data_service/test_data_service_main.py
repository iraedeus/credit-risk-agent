import pytest
from fastapi.testclient import TestClient

from credit_risk_agent.services.data_service.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_healthcheck(client: TestClient):
    """Test that healthcheck endpoint returns HTTP 200 status code and status ok."""

    response = client.get("/api/v1/healthcheck")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
