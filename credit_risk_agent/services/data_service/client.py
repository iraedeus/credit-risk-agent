"""HTTP client for interacting with Credit Risk Data Microservice."""

import httpx


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
