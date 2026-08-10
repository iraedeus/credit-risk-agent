from fastapi import FastAPI

from credit_risk_agent.services.data_service.routers.client import router as client_router

app = FastAPI()
app.include_router(client_router)


@app.get("/api/v1/healthcheck")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}
