"""
ML Scoring Service Pydantic schemas for risk inference and What-If simulation.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from credit_risk_agent.schemas.enums import Education, Marriage, Sex


class PredictionRequest(BaseModel):
    """
    Standard credit risk scoring inference request.
    """

    client_id: int = Field(..., gt=0, description="Unique client identifier for scoring")


class ScenarioParams(BaseModel):
    """
    Set of hypothetical parameter overrides for What-If scenario simulation.
    """

    model_config = ConfigDict(extra="forbid")

    limit_bal: float | None = Field(default=None, gt=0, description="Hypothetical credit limit balance")
    age: int | None = Field(default=None, ge=18, le=120, description="Hypothetical client age")
    sex: Sex | None = Field(default=None, description="Hypothetical client gender")
    education: Education | None = Field(default=None, description="Hypothetical education level")
    marriage: Marriage | None = Field(default=None, description="Hypothetical marital status")
    pay_0: int | None = Field(
        default=None, description="Hypothetical repayment status for month 1 (-1: pay duly, 1..8: delay)"
    )
    bill_amt1: float | None = Field(default=None, description="Hypothetical bill statement amount for month 1")
    pay_amt1: float | None = Field(default=None, description="Hypothetical payment amount for month 1")


class ScenarioSimulationRequest(BaseModel):
    """
    What-If credit risk simulation inference request.
    """

    client_id: int = Field(..., gt=0, description="Unique client identifier for simulation")
    params: ScenarioParams = Field(
        default_factory=ScenarioParams,
        description="Set of hypothetical parameter overrides",
    )


class PredictionResponse(BaseModel):
    """
    Credit risk scoring prediction response from ML service.
    """

    client_id: int = Field(..., gt=0, description="Unique client identifier")
    default_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Predicted default probability between 0.0 and 1.0",
    )
    risk_tier: Literal["LOW", "MEDIUM", "HIGH"] = Field(
        ..., description="Categorical risk tier assessment (LOW, MEDIUM, HIGH)"
    )
    model_version: str = Field(..., description="ML model version string or MLflow Run ID")
