"""
Shared client domain schemas for credit risk services.
"""

from pydantic import BaseModel, ConfigDict, Field

from credit_risk_agent.schemas.enums import Education, Marriage, Sex


class ClientProfile(BaseModel):
    """
    Baseline demographic and financial profile of a credit client.
    """

    model_config = ConfigDict(from_attributes=True)

    client_id: int = Field(..., gt=0, description="Unique client identifier")
    limit_bal: float = Field(..., gt=0, description="Credit limit balance")
    age: int = Field(..., gt=0, ge=18, le=120, description="Client age in years")
    sex: Sex = Field(..., description="Client gender (1: Male, 2: Female)")
    education: Education = Field(
        ...,
        description="Education level (1: Graduate School, 2: University, 3: High School, 4: Others)",
    )
    marriage: Marriage = Field(
        ...,
        description="Marital status (1: Married, 2: Single, 3: Others)",
    )


class ClientPaymentHistory(BaseModel):
    """
    Monthly payment history record for a credit client.
    """

    model_config = ConfigDict(from_attributes=True)

    client_id: int = Field(..., gt=0, description="Unique client identifier")
    month: int = Field(..., ge=1, le=6, description="History month index (1 to 6, where 1 is most recent)")
    pay_status: int = Field(..., description="Repayment status (-1: pay duly, 1..8: payment delay in months)")
    bill_amt: float = Field(..., description="Bill statement amount for the month")
    pay_amt: float = Field(..., description="Previous payment amount for the month")


class ClientFinancialMetrics(BaseModel):
    """
    Aggregated financial metrics and delinquency statistics for a credit client.
    """

    model_config = ConfigDict(from_attributes=True)

    client_id: int = Field(..., gt=0, description="Unique client identifier")
    limit_bal: float = Field(..., gt=0, description="Credit limit balance")
    avg_bill: float = Field(..., description="Average monthly bill statement amount")
    avg_utilization: float = Field(..., description="Average credit limit utilization percentage")
    max_utilization: float = Field(..., description="Maximum credit limit utilization percentage")
    avg_pay: float = Field(..., ge=0, description="Average monthly payment amount")
    repayment_rate: float = Field(..., ge=0, description="Repayment coverage rate percentage")
    max_delay_status: int = Field(
        ...,
        description="Maximum payment delay status (-1: pay duly, 1..8: payment delay in months)",
    )
    delay_months_count: int = Field(
        ...,
        ge=0,
        le=6,
        description="Count of months with payment delay over 6-month historical period",
    )
