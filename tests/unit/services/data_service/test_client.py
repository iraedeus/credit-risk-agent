"""Unit tests for DataServiceClient."""

import httpx

from credit_risk_agent.services.data_service.client import DataServiceClient


class TestGetHealthcheck:
    """Test suite for DataServiceClient.get_healthcheck method."""

    def test_get_healthcheck_success(self) -> None:
        """Test healthcheck returns True on HTTP status 200 response."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status_code=200, json={"status": "ok"})

        transport = httpx.MockTransport(handler)
        client = DataServiceClient(transport=transport)

        result = client.get_healthcheck()
        assert result is True

    def test_get_healthcheck_failure(self) -> None:
        """Test healthcheck returns False on HTTP status 404 response."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status_code=404)

        transport = httpx.MockTransport(handler)
        client = DataServiceClient(transport=transport)

        result = client.get_healthcheck()
        assert result is False

    def test_get_healthcheck_connection_error(self) -> None:
        """Test healthcheck returns False on network connection error."""

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused")

        transport = httpx.MockTransport(handler)
        client = DataServiceClient(transport=transport)

        result = client.get_healthcheck()
        assert result is False
