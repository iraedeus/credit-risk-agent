"""FastAPI main application entry point and service configuration."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import mlflow
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from credit_risk_agent.model.loader import load_model_from_registry, load_scaler_from_registry
from credit_risk_agent.model.predictor import CreditRiskPredictor
from credit_risk_agent.services.ml_service.config import Settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    settings = Settings()
    mlflow.set_tracking_uri(settings.mlflow_url)
    try:
        scaler = load_scaler_from_registry(settings.model_name, settings.model_alias)
        model = load_model_from_registry(settings.model_name, settings.model_alias)
        app.state.predictor = CreditRiskPredictor(model, scaler)
    except Exception:
        pass

    yield


app = FastAPI(lifespan=lifespan)


@app.get("/api/v1/healthcheck")
def healthcheck(request: Request) -> dict[str, str] | JSONResponse:
    """
    Check service health status.

    Returns
    -------
    dict of str to str
        Status dictionary indicating service availability.
    """

    if not hasattr(request.app.state, "predictor"):
        return JSONResponse(status_code=503, content={"status": "error"})

    return {"status": "ok"}
