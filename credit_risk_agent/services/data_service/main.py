from fastapi import FastAPI

app = FastAPI()


@app.get("/api/v1/healthcheck")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}
