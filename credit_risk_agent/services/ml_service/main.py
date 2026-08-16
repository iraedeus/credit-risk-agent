"""FastAPI main application entry point and service configuration."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import mlflow
from fastapi import FastAPI, HTTPException, Request

from credit_risk_agent.model.loader import load_model_from_registry, load_scaler_from_registry
from credit_risk_agent.model.predictor import CreditRiskPredictor
from credit_risk_agent.services.ml_service.config import Settings
from credit_risk_agent.services.ml_service.dependencies import client_profile_history_to_df
from credit_risk_agent.services.ml_service.schemas import ClientProfileHistory, PredictionResponse

logger = logging.getLogger(__name__)


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
        logger.info(
            "Model loaded successfully (model_name=%s, model_alias=%s)",
            settings.model_name,
            settings.model_alias,
        )
    except Exception:
        logger.exception(
            "Failed to load model (model_name=%s, model_alias=%s)",
            settings.model_name,
            settings.model_alias,
        )

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
def predict(profile_history: ClientProfileHistory, request: Request) -> PredictionResponse:
    """
    Compute the credit default probability for a client profile history.

    Parameters
    ----------
    profile_history : ClientProfileHistory
        Client demographic profile and monthly payment history.
    request : Request
        Incoming HTTP request, used to access the predictor on application state.

    Returns
    -------
    PredictionResponse
        Predicted probability of default in range [0.0, 1.0].

    Raises
    ------
    HTTPException
        With status code 503 if the predictor is not available.
    """
    if not hasattr(request.app.state, "predictor"):
        raise HTTPException(status_code=503)

    predictor = request.app.state.predictor
    profile_history_df = client_profile_history_to_df(profile_history)
    default_probability = predictor.predict_pd(profile_history_df)
    return PredictionResponse(default_probability=default_probability)
