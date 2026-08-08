"""
Data Service Pydantic schemas for client profile and payment history.
"""

from pydantic import BaseModel, ConfigDict, Field, model_validator

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


class ClientFullInfo(BaseModel):
    """
    Aggregated full client record containing demographic profile and 6-month payment history.
    """

    model_config = ConfigDict(from_attributes=True)

    profile: ClientProfile = Field(..., description="Client demographic profile")
    history: list[ClientPaymentHistory] = Field(..., description="List of 6 monthly payment history records")

    @model_validator(mode="after")
    def validate_client_ids_match(self) -> "ClientFullInfo":
        """
        Validate that all client_ids in history match profile client_id.

        Returns
        -------
        ClientFullInfo
            Validated ClientFullInfo instance.

        Raises
        ------
        ValueError
            If any record in history has a mismatched client_id.
        """
        for record in self.history:
            if record.client_id != self.profile.client_id:
                raise ValueError(
                    f"Mismatched history client_id ({record.client_id}) "
                    f"and profile client_id ({self.profile.client_id})"
                )

        return self
