"""
Agent Service Pydantic schemas for ReAct agent interaction and reasoning steps.
"""

from typing import Literal

from pydantic import BaseModel, Field


class AssessmentRequest(BaseModel):
    """
    Underwriting assessment request submitted to the ReAct credit risk agent.
    """

    client_id: int = Field(..., gt=0, description="Unique client identifier for scoring")
    prompt: str | None = Field(None, description="Optional custom prompt or question from user")


class ReasoningStep(BaseModel):
    """
    Single reasoning event or tool call step within a ReAct agent execution loop.
    """

    step_type: Literal["thought", "tool_call", "observation", "final", "error"] = Field(
        ...,
        description="Type of reasoning step (thought, tool_call, observation, final, error)",
    )
    content: str = Field(..., description="Text content of reasoning step, tool arguments, or tool output")
    tool_name: str | None = Field(
        None,
        description="Name of executed tool (populated only when step_type='tool_call')",
    )
    step: int | None = Field(None, description="ReAct iteration step index")


class AssessmentResponse(BaseModel):
    """
    Final underwriting assessment report and reasoning trajectory returned by the ReAct agent.
    """

    client_id: int = Field(..., gt=0, description="Unique client identifier")
    agent_response: str = Field(..., description="Final detailed credit risk underwriting report")
    steps: list[ReasoningStep] = Field(
        default_factory=list,
        description="Chronological trajectory of ReAct agent reasoning steps",
    )
    risk_tier: Literal["LOW", "MEDIUM", "HIGH"] | None = Field(
        None,
        description="Assessed risk tier returned during ML scoring evaluation",
    )
