"""API router providing endpoints for client data operations."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query

from credit_risk_agent.schemas.data_schemas import ClientFullInfo
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


@router.get("/{client_id}/")
def get_client(
    repo: Annotated[DataRepository, Depends(get_repo)],
    client_id: Annotated[int, Path(gt=0)],
) -> ClientFullInfo:
    """
    Retrieve full aggregated information for a specific client by ID.

    Parameters
    ----------
    repo : DataRepository
        Data repository instance injected via FastAPI dependency.
    client_id : int
        Unique positive identifier of the target client.

    Returns
    -------
    ClientFullInfo
        Aggregated client record containing profile, payment history, and financial metrics.

    Raises
    ------
    HTTPException
        HTTP 404 error if the client record or its payment history is not found.
    """
    client_info = repo.get_client_full(client_id)

    if client_info is None:
        raise HTTPException(status_code=404, detail="Client not found")

    return client_info
