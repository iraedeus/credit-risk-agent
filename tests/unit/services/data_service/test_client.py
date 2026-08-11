"""Unit tests for DataServiceClient."""

import httpx
import pytest

from credit_risk_agent.services.data_service.client import DataServiceClient
from credit_risk_agent.services.data_service.exceptions import DataServiceHTTPError
from credit_risk_agent.services.data_service.schemas import (
    ClientFinancialMetrics,
    ClientFullInfo,
    ClientPaymentHistory,
    ClientProfile,
)


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

    def test_get_clients_invalid_params(self):
        """Test get_clients raises DataServiceHTTPError on invalid limit parameter."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status_code=422, json={"detail": "Invalid limit"})

        transport = httpx.MockTransport(handler)
        client = DataServiceClient(transport=transport)

        with pytest.raises(DataServiceHTTPError) as exc:
            client.get_clients(limit=-1)

        assert exc.value.status_code == 422
        assert exc.value.message != ""

    def test_get_clients_server_error(self):
        """Test get_clients raises DataServiceHTTPError on HTTP status 500 server error."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status_code=500, json={"detail": "Internal Server Error"})

        transport = httpx.MockTransport(handler)
        client = DataServiceClient(transport=transport)

        with pytest.raises(DataServiceHTTPError) as exc:
            client.get_clients()

        assert exc.value.status_code == 500
        assert exc.value.message != ""

    def test_get_clients_connection_error(self):
        """Test get_clients raises ConnectError on network connection failure."""

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused")

        transport = httpx.MockTransport(handler)
        client = DataServiceClient(transport=transport)

        with pytest.raises(httpx.ConnectError):
            client.get_clients()


class TestGetClient:
    def test_get_client_success(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/api/v1/clients/1"
            sample_payload = {
                "profile": {"client_id": 1, "limit_bal": 100000.0, "sex": 1, "education": 2, "marriage": 1, "age": 30},
                "history": [{"client_id": 1, "month": 1, "pay_status": 1, "bill_amt": 10000.0, "pay_amt": 5000.0}],
                "metrics": {
                    "client_id": 1,
                    "limit_bal": 100000.0,
                    "avg_bill": 10000.0,
                    "avg_utilization": 10.0,
                    "max_utilization": 10.0,
                    "avg_pay": 5000.0,
                    "repayment_rate": 50.0,
                    "max_delay_status": 1,
                    "delay_months_count": 0,
                },
            }
            return httpx.Response(status_code=200, json=sample_payload)

        transport = httpx.MockTransport(handler)
        client = DataServiceClient(transport=transport)

        result = client.get_client(1)
        assert isinstance(result, ClientFullInfo)
        assert result.profile.client_id == 1

    def test_get_client_not_found(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status_code=404)

        transport = httpx.MockTransport(handler)
        client = DataServiceClient(transport=transport)

        result = client.get_client(1)
        assert result is None

    def test_get_client_invalid_params(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status_code=422, json={"detail": "Internal error"})

        transport = httpx.MockTransport(handler)
        client = DataServiceClient(transport=transport)

        with pytest.raises(DataServiceHTTPError) as exc:
            client.get_client(-1)

        assert exc.value.status_code == 422
        assert exc.value.message != ""

    def test_get_client_server_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status_code=500, json={"detail": "Internal error"})

        transport = httpx.MockTransport(handler)
        client = DataServiceClient(transport=transport)

        with pytest.raises(DataServiceHTTPError) as exc:
            client.get_client(1)

        assert exc.value.status_code == 500
        assert exc.value.message != ""

    def test_get_client_connection_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused")

        transport = httpx.MockTransport(handler)
        client = DataServiceClient(transport=transport)

        with pytest.raises(httpx.ConnectError):
            client.get_client(1)


class TestGetClientProfile:
    def test_get_client_profile_success(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/api/v1/clients/1/profile"
            sample_payload = {"client_id": 1, "limit_bal": 100000.0, "sex": 1, "education": 2, "marriage": 1, "age": 30}
            return httpx.Response(status_code=200, json=sample_payload)

        transport = httpx.MockTransport(handler)
        client = DataServiceClient(transport=transport)

        result = client.get_client_profile(1)
        assert isinstance(result, ClientProfile)
        assert result.client_id == 1

    def test_get_client_profile_not_found(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status_code=404)

        transport = httpx.MockTransport(handler)
        client = DataServiceClient(transport=transport)

        result = client.get_client_profile(1)
        assert result is None

    def test_get_client_profile_invalid_params(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status_code=422, json={"detail": "Internal error"})

        transport = httpx.MockTransport(handler)
        client = DataServiceClient(transport=transport)

        with pytest.raises(DataServiceHTTPError) as exc:
            client.get_client_profile(-1)

        assert exc.value.status_code == 422
        assert exc.value.message != ""

    def test_get_client_server_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status_code=500, json={"detail": "Internal error"})

        transport = httpx.MockTransport(handler)
        client = DataServiceClient(transport=transport)

        with pytest.raises(DataServiceHTTPError) as exc:
            client.get_client_profile(1)

        assert exc.value.status_code == 500
        assert exc.value.message != ""

    def test_get_client_connection_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused")

        transport = httpx.MockTransport(handler)
        client = DataServiceClient(transport=transport)

        with pytest.raises(httpx.ConnectError):
            client.get_client_profile(1)


class TestGetClientHistory:
    """Test suite for DataServiceClient.get_client_history method."""

    def test_get_client_history_success(self):
        """Test get_client_history returns list of ClientPaymentHistory on HTTP status 200 response."""

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/api/v1/clients/1/history"
            sample_payload = [{"client_id": 1, "month": 1, "pay_status": 1, "bill_amt": 10000.0, "pay_amt": 5000.0}]
            return httpx.Response(status_code=200, json=sample_payload)

        transport = httpx.MockTransport(handler)
        client = DataServiceClient(transport=transport)

        result = client.get_client_history(1)
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], ClientPaymentHistory)
        assert result[0].client_id == 1

    def test_get_client_history_not_found(self):
        """Test get_client_history returns None on HTTP status 404 response."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status_code=404)

        transport = httpx.MockTransport(handler)
        client = DataServiceClient(transport=transport)

        result = client.get_client_history(1)
        assert result is None

    def test_get_client_history_invalid_params(self):
        """Test get_client_history raises DataServiceHTTPError on invalid client_id parameter."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status_code=422, json={"detail": "Internal error"})

        transport = httpx.MockTransport(handler)
        client = DataServiceClient(transport=transport)

        with pytest.raises(DataServiceHTTPError) as exc:
            client.get_client_history(-1)

        assert exc.value.status_code == 422
        assert exc.value.message != ""

    def test_get_client_history_server_error(self):
        """Test get_client_history raises DataServiceHTTPError on HTTP status 500 server error."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status_code=500, json={"detail": "Internal error"})

        transport = httpx.MockTransport(handler)
        client = DataServiceClient(transport=transport)

        with pytest.raises(DataServiceHTTPError) as exc:
            client.get_client_history(1)

        assert exc.value.status_code == 500
        assert exc.value.message != ""

    def test_get_client_history_error(self):
        """Test get_client_history raises ConnectError on network connection failure."""

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused")

        transport = httpx.MockTransport(handler)
        client = DataServiceClient(transport=transport)

        with pytest.raises(httpx.ConnectError):
            client.get_client_history(1)


class TestGetClientMetrics:
    """Test suite for DataServiceClient.get_client_metrics method."""

    def test_get_client_metrics_success(self):
        """Test get_client_metrics returns ClientFinancialMetrics on HTTP status 200 response."""

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/api/v1/clients/1/metrics"
            sample_payload = {
                "client_id": 1,
                "limit_bal": 100000.0,
                "avg_bill": 10000.0,
                "avg_utilization": 10.0,
                "max_utilization": 10.0,
                "avg_pay": 5000.0,
                "repayment_rate": 50.0,
                "max_delay_status": 1,
                "delay_months_count": 0,
            }
            return httpx.Response(status_code=200, json=sample_payload)

        transport = httpx.MockTransport(handler)
        client = DataServiceClient(transport=transport)

        result = client.get_client_metrics(1)
        assert isinstance(result, ClientFinancialMetrics)
        assert result.client_id == 1

    def test_get_client_metrics_not_found(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status_code=404)

        transport = httpx.MockTransport(handler)
        client = DataServiceClient(transport=transport)

        result = client.get_client_metrics(1)
        assert result is None

    def test_get_client_metrics_invalid_params(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status_code=422, json={"detail": "Internal error"})

        transport = httpx.MockTransport(handler)
        client = DataServiceClient(transport=transport)

        with pytest.raises(DataServiceHTTPError) as exc:
            client.get_client_metrics(-1)

        assert exc.value.status_code == 422
        assert exc.value.message != ""

    def test_get_client_metrics_server_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status_code=500, json={"detail": "Internal error"})

        transport = httpx.MockTransport(handler)
        client = DataServiceClient(transport=transport)

        with pytest.raises(DataServiceHTTPError) as exc:
            client.get_client_metrics(1)

        assert exc.value.status_code == 500
        assert exc.value.message != ""

    def test_get_client_metrics_connection_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused")

        transport = httpx.MockTransport(handler)
        client = DataServiceClient(transport=transport)

        with pytest.raises(httpx.ConnectError):
            client.get_client_metrics(1)
