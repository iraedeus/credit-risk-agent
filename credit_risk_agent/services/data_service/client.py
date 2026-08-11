"""HTTP client for interacting with Credit Risk Data Microservice."""

import httpx

from credit_risk_agent.services.data_service.exceptions import DataServiceHTTPError
from credit_risk_agent.services.data_service.schemas import (
    ClientFinancialMetrics,
    ClientFullInfo,
    ClientPaymentHistory,
    ClientProfile,
)


class DataServiceClient:
    """
    Client for interacting with Credit Risk Data Microservice API.

    Parameters
    ----------
    base_url : str, default="http://localhost:8000"
        Base URL of the target microservice.
    timeout : float, default=5.0
        Request timeout duration in seconds.
    transport : httpx.BaseTransport or None, default=None
        Optional HTTPX transport instance for testing with mock handlers.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 5.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.client = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)

    def get_healthcheck(self) -> bool:
        """
        Check health status of the data service.

        Returns
        -------
        bool
            True if service returns HTTP status 200, False otherwise.
        """
        try:
            response = self.client.get("/api/v1/healthcheck")
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def get_clients(self, limit: int = 20, offset: int = 0) -> list[int]:
        """
        Fetch a paginated list of client IDs from the Data Service.

        Parameters
        ----------
        limit : int, default=20
            Maximum number of client IDs to return.
        offset : int, default=0
            Number of client IDs to skip for pagination.

        Returns
        -------
        list of int
            List of unique client IDs.

        Raises
        ------
        DataServiceHTTPError
            If the Data Service returns an HTTP status code 4xx or 5xx.
        """
        response = self.client.get("/api/v1/clients", params={"limit": limit, "offset": offset})
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text

            raise DataServiceHTTPError(status_code=response.status_code, message=detail) from exc

        return response.json()

    def get_client(self, client_id: int) -> ClientFullInfo | None:
        """
        Retrieve full aggregated information for a specific client by ID.

        Parameters
        ----------
        client_id : int
            Unique positive identifier of the target client.

        Returns
        -------
        ClientFullInfo or None
            Aggregated client record, or None if the client is not found.

        Raises
        ------
        DataServiceHTTPError
            If the Data Service returns an HTTP error status code other than 404.
        """
        response = self.client.get(f"/api/v1/clients/{client_id}")

        if response.status_code == 404:
            return None

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text

            raise DataServiceHTTPError(status_code=response.status_code, message=detail) from exc

        return ClientFullInfo.model_validate(response.json())

    def get_client_profile(self, client_id: int) -> ClientProfile | None:
        """
        Retrieve demographic and basic credit profile for a specific client by ID.

        Parameters
        ----------
        client_id : int
            Unique positive identifier of the target client.

        Returns
        -------
        ClientProfile or None
            Client demographic profile, or None if the client is not found.

        Raises
        ------
        DataServiceHTTPError
            If the Data Service returns an HTTP error status code other than 404.
        """

        response = self.client.get(f"/api/v1/clients/{client_id}/profile")

        if response.status_code == 404:
            return None

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text

            raise DataServiceHTTPError(status_code=response.status_code, message=detail) from exc

        return ClientProfile.model_validate(response.json())

    def get_client_history(self, client_id: int) -> list[ClientPaymentHistory] | None:
        """
        Retrieve monthly payment history records for a specific client by ID.

        Parameters
        ----------
        client_id : int
            Unique positive identifier of the target client.

        Returns
        -------
        list of ClientPaymentHistory or None
            List of monthly payment history entries, or None if the client is not found.

        Raises
        ------
        DataServiceHTTPError
            If the Data Service returns an HTTP error status code other than 404.
        """

        response = self.client.get(f"/api/v1/clients/{client_id}/history")

        if response.status_code == 404:
            return None

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text

            raise DataServiceHTTPError(status_code=response.status_code, message=detail) from exc

        return [ClientPaymentHistory.model_validate(item) for item in response.json()]

    def get_client_metrics(self, client_id: int) -> ClientFinancialMetrics | None:
        """
        Retrieve calculated financial metrics for a specific client by ID.

        Parameters
        ----------
        client_id : int
            Unique positive identifier of the target client.

        Returns
        -------
        ClientFinancialMetrics or None
            Calculated financial metrics, or None if the client is not found.

        Raises
        ------
        DataServiceHTTPError
            If the Data Service returns an HTTP error status code other than 404.
        """

        response = self.client.get(f"/api/v1/clients/{client_id}/metrics")

        if response.status_code == 404:
            return None

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text

            raise DataServiceHTTPError(status_code=response.status_code, message=detail) from exc

        return ClientFinancialMetrics.model_validate(response.json())
