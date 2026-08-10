from collections.abc import Generator
from typing import Annotated

from fastapi import Depends
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from credit_risk_agent.services.data_service.config import Settings
from credit_risk_agent.services.data_service.models import Base
from credit_risk_agent.services.data_service.repository import DataRepository

settings = Settings()
db_path = f"sqlite:///{settings.database_path}"

engine = create_engine(db_path)
Base.metadata.create_all(engine)

SessionLocal = sessionmaker(bind=engine)


def get_db() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def get_repo(session: Annotated[Session, Depends(get_db)]) -> DataRepository:
    return DataRepository(session)
