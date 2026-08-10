from typing import Annotated

from fastapi import APIRouter, Depends, Query

from credit_risk_agent.services.data_service.dependencies import get_repo
from credit_risk_agent.services.data_service.repository import DataRepository

router = APIRouter(prefix="/api/v1/clients", tags=["Clients"])


@router.get("/")
def get_clients(
    repo: Annotated[DataRepository, Depends(get_repo)],
    limit: Annotated[int, Query(ge=0, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[int]:
    return repo.get_clients(limit, offset)
