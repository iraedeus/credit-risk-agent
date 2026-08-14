"""
ML Service Pydantic schemas for risk inference requests.
"""

from pydantic import BaseModel, Field, model_validator

from credit_risk_agent.schemas.client_schemas import ClientPaymentHistory, ClientProfile


class ClientProfileHistory(BaseModel):
    """
    Full client feature set required for credit default inference.

    Contains the client demographic profile and 6-month payment history.
    Financial metrics are intentionally excluded — the model consumes only raw features.
    """

    profile: ClientProfile = Field(..., description="Client demographic profile")
    history: list[ClientPaymentHistory] = Field(..., description="List of 6 monthly payment history records")

    @model_validator(mode="after")
    def validate_client_ids_match(self) -> "ClientProfileHistory":
        """
        Validate that all client_ids in history match profile client_id.

        Returns
        -------
        ClientProfileHistory
            Validated ClientProfileHistory instance.

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


class PredictionResponse(BaseModel):
    """
    Credit default probability (PD) prediction response from ML service.
    """

    default_probability: float = Field(
        ...,
        le=1,
        ge=0,
        description="Probability of default in range [0.0, 1.0]",
    )
