"""FastAPI main application entry point and service configuration."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from credit_risk_agent.services.data_service.dependencies import engine
from credit_risk_agent.services.data_service.models import Base
from credit_risk_agent.services.data_service.routers.client import router as client_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    Base.metadata.create_all(engine)
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(client_router)


@app.get("/api/v1/healthcheck")
def healthcheck() -> dict[str, str]:
    """
    Check service health status.

    Returns
    -------
    dict of str to str
        Status dictionary indicating service availability.
    """

    return {"status": "ok"}
