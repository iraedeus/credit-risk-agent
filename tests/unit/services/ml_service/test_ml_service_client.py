import json

import httpx
import pytest

from credit_risk_agent.services.ml_service.client import MLServiceClient, get_ml_service_client
from credit_risk_agent.services.ml_service.exceptions import MLServiceHTTPError
from credit_risk_agent.services.ml_service.schemas import PredictionResponse


class TestClientProvider:
    """Test suite for get_ml_service_client provider function."""

    def test_get_client_returns_instance(self) -> None:
        """Test provider returns an instance of MLServiceClient."""
        client = get_ml_service_client()
        assert isinstance(client, MLServiceClient)

    def test_get_client_returns_singleton(self) -> None:
        """Test provider returns cached singleton instance on repeated calls."""
        client1 = get_ml_service_client()
        client2 = get_ml_service_client()

        assert client1 is client2


class TestGetHealthcheck:
    """Test suite for DataServiceClient.get_healthcheck method."""

    def test_get_healthcheck_success(self) -> None:
        """Test healthcheck returns True on HTTP status 200 response."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status_code=200, json={"status": "ok"})

        transport = httpx.MockTransport(handler)
        client = MLServiceClient(transport=transport)

        result = client.get_healthcheck()
        assert result is True

    def test_get_healthcheck_failure(self) -> None:
        """Test healthcheck returns False on HTTP status 404 response."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status_code=404)

        transport = httpx.MockTransport(handler)
        client = MLServiceClient(transport=transport)

        result = client.get_healthcheck()
        assert result is False

    def test_get_healthcheck_service_unavailable(self) -> None:

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status_code=503, json={"detail": "Service Unavailable"})

        transport = httpx.MockTransport(handler)
        client = MLServiceClient(transport=transport)

        result = client.get_healthcheck()
        assert result is False

    def test_get_healthcheck_connection_error(self) -> None:
        """Test healthcheck returns False on network connection error."""

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused")

        transport = httpx.MockTransport(handler)
        client = MLServiceClient(transport=transport)

        result = client.get_healthcheck()
        assert result is False


class TestPredict:
    def test_predict_success(self, profile_history) -> None:

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.method == "POST"
            assert request.url.path == "/api/v1/predict"
            assert json.loads(request.content) == profile_history.model_dump(mode="json")
            return httpx.Response(status_code=200, json={"default_probability": 0.42})

        transport = httpx.MockTransport(handler)
        client = MLServiceClient(transport=transport)

        result = client.predict(profile_history)
        assert isinstance(result, PredictionResponse)
        assert result.default_probability == 0.42

    def test_predict_service_unavailable(self, profile_history) -> None:

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status_code=503, json={"detail": "Service Unavailable"})

        transport = httpx.MockTransport(handler)
        client = MLServiceClient(transport=transport)

        with pytest.raises(MLServiceHTTPError) as exc:
            client.predict(profile_history)

        assert exc.value.status_code == 503
        assert exc.value.message == "Service Unavailable"

    def test_predict_invalid_request(self, profile_history) -> None:

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                status_code=422,
                json={
                    "detail": [
                        {
                            "type": "value_error",
                            "loc": ["body"],
                            "msg": "Value error, Invalid length of the client history. Should be equal to 6.",
                        }
                    ]
                },
            )

        transport = httpx.MockTransport(handler)
        client = MLServiceClient(transport=transport)

        with pytest.raises(MLServiceHTTPError) as exc:
            client.predict(profile_history)

        assert exc.value.status_code == 422
        assert exc.value.message == "Value error, Invalid length of the client history. Should be equal to 6."
