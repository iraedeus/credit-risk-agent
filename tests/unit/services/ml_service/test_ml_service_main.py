"""Unit tests for the ML service FastAPI application endpoints."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from credit_risk_agent.services.ml_service.main import app

EXPECTED_FEATURE_COLUMNS = [
    "client_id",
    "limit_bal",
    "sex",
    "education",
    "marriage",
    "age",
    "month",
    "pay_status",
    "bill_amt",
    "pay_amt",
]


@pytest.fixture
def test_app():
    """
    Provide the ML service application with a clean predictor state.

    Yields
    ------
    app
        The ML service FastAPI application instance.

    Notes
    -----
    Cleans up any predictor attached to ``app.state`` after the test runs.
    """
    yield app
    if hasattr(app.state, "predictor"):
        delattr(app.state, "predictor")


class TestHealthcheck:
    """Test suite for the /api/v1/healthcheck endpoint."""

    def test_healthcheck_success(self, test_app):
        """Test that healthcheck returns HTTP 200 when the predictor is loaded."""
        with (
            patch("credit_risk_agent.services.ml_service.main.load_scaler_from_registry", return_value=MagicMock()),
            patch("credit_risk_agent.services.ml_service.main.load_model_from_registry", return_value=MagicMock()),
        ):
            with TestClient(test_app) as client:
                response = client.get("/api/v1/healthcheck")
                assert hasattr(app.state, "predictor")
                assert response.status_code == 200
                assert response.json() == {"status": "ok"}

    def test_healthcheck_predictor_not_available(self, test_app):
        """Test that healthcheck returns HTTP 503 when no predictor is loaded."""
        client = TestClient(test_app)

        response = client.get("/api/v1/healthcheck")

        assert not hasattr(app.state, "predictor")
        assert response.status_code == 503
        assert response.json() == {"detail": "Service Unavailable"}

    def test_healthcheck_model_error(self, test_app):
        """Test that healthcheck returns HTTP 503 when model loading fails."""
        with patch("credit_risk_agent.services.ml_service.main.load_scaler_from_registry") as scaler_loader:
            scaler_loader.side_effect = RuntimeError()

            with TestClient(test_app) as client:
                response = client.get("/api/v1/healthcheck")
                assert response.status_code == 503
                assert response.json() == {"detail": "Service Unavailable"}


class TestPredict:
    """Test suite for the /api/v1/predict endpoint."""

    def test_predict_success(self, test_app, profile_history_payload):
        """Test that predict returns the mocked default probability for a valid request."""
        predictor = MagicMock()
        predictor.predict_pd.return_value = 0.42
        app.state.predictor = predictor

        client = TestClient(test_app)

        response = client.post("/api/v1/predict", json=profile_history_payload)

        assert response.status_code == 200
        assert response.json() == {"default_probability": 0.42}

        predictor.predict_pd.assert_called_once()
        df = predictor.predict_pd.call_args.args[0]
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 6
        assert list(df.columns) == EXPECTED_FEATURE_COLUMNS

    def test_predict_predictor_not_available(self, test_app, profile_history_payload):
        """Test that predict returns HTTP 503 when no predictor is available."""
        client = TestClient(test_app)

        response = client.post("/api/v1/predict", json=profile_history_payload)

        assert response.status_code == 503
        assert response.json() == {"detail": "Service Unavailable"}

    def test_predict_invalid_history_len(self, test_app):
        """Test that predict returns HTTP 422 for a history shorter than 6 records."""
        payload = {
            "profile": {
                "client_id": 1,
                "limit_bal": 300000.0,
                "age": 35,
                "sex": 1,
                "education": 2,
                "marriage": 1,
            },
            "history": [
                {"client_id": 1, "month": 1, "pay_status": -1, "bill_amt": 55000.0, "pay_amt": 18000.0},
            ],
        }

        client = TestClient(test_app)
        response = client.post("/api/v1/predict", json=payload)

        assert response.status_code == 422
        assert (
            response.json()["detail"][0]["msg"]
            == "Value error, Invalid length of the client history. Should be equal to 6."
        )

    def test_predict_not_unique_history_len(self, test_app):
        """Test that predict returns HTTP 422 when the history contains duplicate months."""
        payload = {
            "profile": {
                "client_id": 1,
                "limit_bal": 300000.0,
                "age": 35,
                "sex": 1,
                "education": 2,
                "marriage": 1,
            },
            "history": [
                {"client_id": 1, "month": 1, "pay_status": -1, "bill_amt": 55000.0, "pay_amt": 18000.0},
                {"client_id": 1, "month": 1, "pay_status": -1, "bill_amt": 55000.0, "pay_amt": 18000.0},
                {"client_id": 1, "month": 1, "pay_status": -1, "bill_amt": 55000.0, "pay_amt": 18000.0},
                {"client_id": 1, "month": 1, "pay_status": -1, "bill_amt": 55000.0, "pay_amt": 18000.0},
                {"client_id": 1, "month": 1, "pay_status": -1, "bill_amt": 55000.0, "pay_amt": 18000.0},
                {"client_id": 1, "month": 1, "pay_status": -1, "bill_amt": 55000.0, "pay_amt": 18000.0},
            ],
        }

        client = TestClient(test_app)
        response = client.post("/api/v1/predict", json=payload)

        assert response.status_code == 422
        assert response.json()["detail"][0]["msg"] == "Value error, Months in history must be unique."
