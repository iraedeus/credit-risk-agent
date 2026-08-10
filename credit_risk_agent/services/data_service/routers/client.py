"""API router providing endpoints for client data operations."""

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
    """
    Fetch a paginated list of client IDs sorted in ascending order.

    Parameters
    ----------
    repo : DataRepository
        Data repository instance injected via FastAPI dependency.
    limit : int, default=20
        Maximum number of client IDs to return (0 to 100).
    offset : int, default=0
        Number of client IDs to skip for pagination.

    Returns
    -------
    list of int
        List of client IDs matching the pagination parameters.
    """
    return repo.get_clients(limit, offset)
