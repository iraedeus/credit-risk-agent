"""FastAPI main application entry point and service configuration."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import mlflow
from fastapi import FastAPI, HTTPException, Request

from credit_risk_agent.model.loader import load_model_from_registry, load_scaler_from_registry
from credit_risk_agent.model.predictor import CreditRiskPredictor
from credit_risk_agent.services.ml_service.config import Settings
from credit_risk_agent.services.ml_service.dependencies import client_profile_history_to_df
from credit_risk_agent.services.ml_service.schemas import ClientProfileHistory, PredictionResponse


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Load the champion model and scaler from MLflow on application startup.

    Sets the MLflow tracking URI, resolves the champion model version
    from the registry, and stores a CreditRiskPredictor on ``app.state``.
    If no champion is registered yet, the predictor is left unset and the
    service reports unhealthy via the healthcheck endpoint.

    Parameters
    ----------
    app : FastAPI
        FastAPI application instance.

    Yields
    ------
    None
        Control is yielded to the application while it is running.
    """
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
def healthcheck(request: Request) -> dict[str, str]:
    """
    Check service health status.

    Returns
    -------
    dict of str to str
        Status dictionary indicating service availability.
    """

    if not hasattr(request.app.state, "predictor"):
        raise HTTPException(status_code=503)

    return {"status": "ok"}


@app.post("/api/v1/predict")
def predict(request: Request, profile_history: ClientProfileHistory) -> PredictionResponse:
    if not hasattr(request.app.state, "predictor"):
        raise HTTPException(status_code=503)

    predictor = request.app.state.predictor
    profile_history_df = client_profile_history_to_df(profile_history)
    default_probability = predictor.predict_pd(profile_history_df)
    return PredictionResponse(default_probability=default_probability)
