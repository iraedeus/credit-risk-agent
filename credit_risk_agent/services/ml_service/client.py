from functools import lru_cache

import httpx

from credit_risk_agent.config import ML_SERVICE_URL
from credit_risk_agent.services.ml_service.exceptions import MLServiceHTTPError
from credit_risk_agent.services.ml_service.schemas import ClientProfileHistory, PredictionResponse


class MLServiceClient:
    def __init__(
        self,
        base_url: str = "http://localhost:8002",
        timeout: float = 15.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.client = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
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
        try:
            response = self.client.get("/api/v1/healthcheck")
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def predict(self, profile_history: ClientProfileHistory) -> PredictionResponse:
        response = self.client.post("/api/v1/predict", json=profile_history.model_dump(mode="json"))

        self._raise_for_status(response)
        return PredictionResponse.model_validate(response.json())


@lru_cache
def get_ml_service_client() -> MLServiceClient:
    return MLServiceClient(base_url=ML_SERVICE_URL)
