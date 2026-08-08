"""
Agent execution lifecycle stream events.
"""

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class ThoughtEvent:
    """
    Event emitted when the agent generates intermediate internal reasoning thoughts.

    Attributes
    ----------
    content : str
        Reasoning text emitted by LLM.
    step : int or None, default=None
        ReAct iteration step index.
    type : Literal['thought']
        Event discriminator identifier.
    """

    content: str
    step: int | None = None
    type: Literal["thought"] = "thought"


@dataclass(frozen=True)
class ToolCallEvent:
    """
    Event emitted when the agent requests execution of a domain tool function.

    Attributes
    ----------
    tool_name : str
        Name of requested tool function.
    tool_args : dict of str to Any
        Arguments dictionary passed to tool.
    step : int or None, default=None
        ReAct iteration step index.
    type : Literal['tool_call']
        Event discriminator identifier.
    """

    tool_name: str
    tool_args: dict[str, Any] = field(default_factory=dict)
    step: int | None = None
    type: Literal["tool_call"] = "tool_call"


@dataclass(frozen=True)
class ObservationEvent:
    """
    Event emitted when a tool returns execution output or observations.

    Attributes
    ----------
    tool_name : str
        Name of executed tool.
    content : str
        Observation output string.
    step : int or None, default=None
        ReAct iteration step index.
    type : Literal['observation']
        Event discriminator identifier.
    """

    tool_name: str
    content: str
    step: int | None = None
    type: Literal["observation"] = "observation"


@dataclass(frozen=True)
class ErrorEvent:
    """
    Event emitted when an error occurs during agent execution.

    Attributes
    ----------
    content : str
        Error message text.
    type : Literal['error']
        Event discriminator identifier.
    """

    content: str
    type: Literal["error"] = "error"


@dataclass(frozen=True)
class FinalEvent:
    """
    Event emitted when the agent produces its final answer to the user prompt.

    Attributes
    ----------
    content : str
        Final response text.
    type : Literal['final']
        Event discriminator identifier.
    """

    content: str
    type: Literal["final"] = "final"


AgentEvent = ThoughtEvent | ToolCallEvent | ObservationEvent | FinalEvent | ErrorEvent
