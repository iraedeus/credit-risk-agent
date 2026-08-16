from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from credit_risk_agent.schemas.client_schemas import ClientPaymentHistory, ClientProfile
from credit_risk_agent.schemas.enums import Education, Marriage, Sex
from credit_risk_agent.services.ml_service.main import app
from credit_risk_agent.services.ml_service.schemas import ClientProfileHistory, PredictionResponse


@pytest.fixture
def test_app():
    yield app
    if hasattr(app.state, "predictor"):
        delattr(app.state, "predictor")


class TestHealthcheck:
    def test_healthcheck_success(self, test_app):
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

    def test_healthcheck_predictor_not_available(self, test_app):
        client = TestClient(test_app)

        response = client.get("/api/v1/healthcheck")

        assert not hasattr(app.state, "predictor")
        assert response.status_code == 503
        assert response.json() == {"detail": "Service Unavailable"}

    def test_healthcheck_model_error(self, test_app):
        with patch("credit_risk_agent.services.ml_service.main.load_scaler_from_registry") as scaler_loader:
            scaler_loader.side_effect = RuntimeError()

            with TestClient(test_app) as client:
                response = client.get("/api/v1/healthcheck")
                assert response.status_code == 503
                assert response.json() == {"detail": "Service Unavailable"}


class TestPredict:
    def test_predict_success(self, test_app):
        profile_history = ClientProfileHistory(
            profile=ClientProfile(
                client_id=1,
                limit_bal=300000.0,
                age=35,
                sex=Sex.MALE,
                education=Education.UNIVERSITY,
                marriage=Marriage.MARRIED,
            ),
            history=[
                ClientPaymentHistory(client_id=1, month=1, pay_status=-1, bill_amt=55000.0, pay_amt=18000.0),
                ClientPaymentHistory(client_id=1, month=2, pay_status=-1, bill_amt=55000.0, pay_amt=18000.0),
                ClientPaymentHistory(client_id=1, month=3, pay_status=-1, bill_amt=55000.0, pay_amt=18000.0),
                ClientPaymentHistory(client_id=1, month=4, pay_status=-1, bill_amt=55000.0, pay_amt=18000.0),
                ClientPaymentHistory(client_id=1, month=5, pay_status=-1, bill_amt=55000.0, pay_amt=18000.0),
                ClientPaymentHistory(client_id=1, month=6, pay_status=-1, bill_amt=55000.0, pay_amt=18000.0),
            ],
        )

        payload = profile_history.model_dump(mode="json")

        with TestClient(test_app) as client:
            response = client.post("/api/v1/predict", json=payload)
            assert response.status_code == 200
            PredictionResponse.model_validate(response.json())

    def test_predict_predictor_not_available(self, test_app):
        profile_history = ClientProfileHistory(
            profile=ClientProfile(
                client_id=1,
                limit_bal=300000.0,
                age=35,
                sex=Sex.MALE,
                education=Education.UNIVERSITY,
                marriage=Marriage.MARRIED,
            ),
            history=[
                ClientPaymentHistory(client_id=1, month=1, pay_status=-1, bill_amt=55000.0, pay_amt=18000.0),
                ClientPaymentHistory(client_id=1, month=2, pay_status=-1, bill_amt=55000.0, pay_amt=18000.0),
                ClientPaymentHistory(client_id=1, month=3, pay_status=-1, bill_amt=55000.0, pay_amt=18000.0),
                ClientPaymentHistory(client_id=1, month=4, pay_status=-1, bill_amt=55000.0, pay_amt=18000.0),
                ClientPaymentHistory(client_id=1, month=5, pay_status=-1, bill_amt=55000.0, pay_amt=18000.0),
                ClientPaymentHistory(client_id=1, month=6, pay_status=-1, bill_amt=55000.0, pay_amt=18000.0),
            ],
        )

        payload = profile_history.model_dump(mode="json")

        client = TestClient(test_app)
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 503
        assert response.json() == {"detail": "Service Unavailable"}

    def test_predict_invalid_history_len(self, test_app):
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

        with TestClient(test_app) as client:
            response = client.post("/api/v1/predict", json=payload)
            assert response.status_code == 422
            print(response.json())
            assert (
                response.json()["detail"][0]["msg"]
                == "Value error, Invalid length of the client history. Should be equal to 6."
            )

    def test_predict_not_unique_history_len(self, test_app):
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

        with TestClient(test_app) as client:
            response = client.post("/api/v1/predict", json=payload)
            assert response.status_code == 422
            print(response.json())
            assert response.json()["detail"][0]["msg"] == "Value error, Months in history must be unique."
