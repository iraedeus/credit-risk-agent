"""
Data Service Pydantic schemas for aggregated client records.
"""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from credit_risk_agent.schemas.client_schemas import ClientFinancialMetrics, ClientPaymentHistory, ClientProfile


class ClientFullInfo(BaseModel):
    """
    Aggregated full client record containing demographic profile and 6-month payment history.
    """

    model_config = ConfigDict(from_attributes=True)

    profile: ClientProfile = Field(..., description="Client demographic profile")
    history: list[ClientPaymentHistory] = Field(..., description="List of 6 monthly payment history records")
    metrics: ClientFinancialMetrics = Field(..., description="Financial metrics of this client")

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

        if self.metrics.client_id != self.profile.client_id:
            raise ValueError(
                f"Mismatched metrics client_id ({self.metrics.client_id}) "
                f"and profile client_id ({self.profile.client_id})"
            )

        return self
