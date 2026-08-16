"""HTTP client for interacting with the ML Inference Microservice."""

from functools import lru_cache

import httpx

from credit_risk_agent.config import ML_SERVICE_URL
from credit_risk_agent.services.ml_service.exceptions import MLServiceHTTPError
from credit_risk_agent.services.ml_service.schemas import ClientProfileHistory, PredictionResponse


class MLServiceClient:
    """Client for interacting with the ML Inference Microservice API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8002",
        timeout: float = 15.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """
        Initialize an ML service client backed by an httpx transport.

        Parameters
        ----------
        base_url : str, default="http://localhost:8002"
            Base URL of the target microservice.
        timeout : float, default=15.0
            Request timeout duration in seconds.
        transport : httpx.BaseTransport or None, default=None
            Optional HTTPX transport instance for testing with mock handlers.
        """
        self.client = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        """
        Raise an MLServiceHTTPError for non-success HTTP responses.

        Extracts the ``detail`` field from the error response. When ``detail``
        is a list (e.g. FastAPI validation errors), all messages are joined
        with "; ". Falls back to the raw response body when parsing fails.

        Parameters
        ----------
        response : httpx.Response
            HTTP response to inspect.

        Raises
        ------
        MLServiceHTTPError
            If the response carries an HTTP 4xx or 5xx status code.
        """
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            try:
                detail = response.json().get("detail", response.text)
                if isinstance(detail, list):
                    errors = [error["msg"] for error in detail]
                    detail = "; ".join(errors)
            except Exception:
                detail = response.text

            raise MLServiceHTTPError(status_code=response.status_code, message=detail) from exc

    def get_healthcheck(self) -> bool:
        """
        Check health status of the ML service.

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

    def predict(self, profile_history: ClientProfileHistory) -> PredictionResponse:
        """
        Send client profile history to the ML service and get default probability.

        Parameters
        ----------
        profile_history : ClientProfileHistory
            Client profile and payment history to score.

        Returns
        -------
        PredictionResponse
            Predicted probability of default.

        Raises
        ------
        MLServiceHTTPError
            If the ML service returns an HTTP 4xx or 5xx status code.
        """
        response = self.client.post("/api/v1/predict", json=profile_history.model_dump(mode="json"))

        self._raise_for_status(response)
        return PredictionResponse.model_validate(response.json())


@lru_cache
def get_ml_service_client() -> MLServiceClient:
    """
    Get a cached singleton instance of MLServiceClient.

    Returns
    -------
    MLServiceClient
        Configured client instance targeting the ML microservice.
    """
    return MLServiceClient(base_url=ML_SERVICE_URL)
