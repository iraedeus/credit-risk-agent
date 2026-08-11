"""Unit tests for DataServiceClient."""

import httpx
import pytest

from credit_risk_agent.services.data_service.client import DataServiceClient
from credit_risk_agent.services.data_service.exceptions import DataServiceHTTPError


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


class TestGetClients:
    """Test suite for DataServiceClient.get_clients method."""

    def test_get_clients_success(self):
        """Test get_clients returns list of client IDs on HTTP status 200 response."""

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/api/v1/clients"
            assert request.url.params["limit"] == "3"
            assert request.url.params["offset"] == "0"
            return httpx.Response(status_code=200, json=[1, 2, 3])

        transport = httpx.MockTransport(handler)
        client = DataServiceClient(transport=transport)

        result = client.get_clients(3, 0)
        assert result == [1, 2, 3]

    def test_get_clients_invalid_limit(self):
        """Test get_clients raises DataServiceHTTPError on invalid limit parameter."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status_code=422, json={"detail": "must be between 0 and 100"})

        transport = httpx.MockTransport(handler)
        client = DataServiceClient(transport=transport)

        with pytest.raises(DataServiceHTTPError) as exc:
            client.get_clients(limit=-1)

        assert exc.value.status_code == 422
        assert exc.value.message != ""

        with pytest.raises(DataServiceHTTPError) as exc:
            client.get_clients(limit=999)

        assert exc.value.status_code == 422
        assert exc.value.message != ""

    def test_get_clients_invalid_offset(self):
        """Test get_clients raises DataServiceHTTPError on invalid offset parameter."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status_code=422, json={"detail": "must be greater or equal than 0"})

        transport = httpx.MockTransport(handler)
        client = DataServiceClient(transport=transport)

        with pytest.raises(DataServiceHTTPError) as exc:
            client.get_clients(offset=-1)

        assert exc.value.status_code == 422
        assert exc.value.message != ""

    def test_get_clients_connection_error(self):
        """Test get_clients raises ConnectError on network connection failure."""

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused")

        transport = httpx.MockTransport(handler)
        client = DataServiceClient(transport=transport)

        with pytest.raises(httpx.ConnectError):
            client.get_clients()
