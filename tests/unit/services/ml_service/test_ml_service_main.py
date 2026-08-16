from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from credit_risk_agent.services.ml_service.main import app


@pytest.fixture
def test_app():
    yield app
    if hasattr(app.state, "predictor"):
        delattr(app.state, "predictor")


def test_healthcheck_success(test_app):
    """Test that healthcheck endpoint returns HTTP 200 status code and status ok."""

    with (
        patch("credit_risk_agent.services.ml_service.main.load_scaler_from_registry", return_value=MagicMock()),
        patch("credit_risk_agent.services.ml_service.main.load_model_from_registry", return_value=MagicMock()),
    ):
        with TestClient(test_app) as client:
            response = client.get("/api/v1/healthcheck")
            assert hasattr(app.state, "predictor")
            assert response.status_code == 200
            assert response.json() == {"status": "ok"}


def test_healthcheck_predictor_not_available(test_app):
    client = TestClient(test_app)

    response = client.get("/api/v1/healthcheck")

    assert not hasattr(app.state, "predictor")
    assert response.status_code == 503
    assert response.json() == {"status": "error"}


def test_healthcheck_model_error(test_app):
    with patch("credit_risk_agent.services.ml_service.main.load_scaler_from_registry") as scaler_loader:
        scaler_loader.side_effect = RuntimeError()

        with TestClient(test_app) as client:
            response = client.get("/api/v1/healthcheck")
            assert response.status_code == 503
            assert response.json() == {"status": "error"}
