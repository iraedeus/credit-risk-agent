from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class ThoughtEvent:
    content: str
    step: int | None = None
    type: Literal["thought"] = "thought"


@dataclass(frozen=True)
class ToolCallEvent:
    tool_name: str
    tool_args: dict[str, Any] = field(default_factory=dict)
    step: int | None = None
    type: Literal["tool_call"] = "tool_call"


@dataclass(frozen=True)
class ObservationEvent:
    tool_name: str
    content: str
    step: int | None = None
    type: Literal["observation"] = "observation"


@dataclass(frozen=True)
class ErrorEvent:
    content: str
    type: Literal["error"] = "error"


@dataclass(frozen=True)
class FinalEvent:
    content: str
    type: Literal["final"] = "final"


AgentEvent = ThoughtEvent | ToolCallEvent | ObservationEvent | FinalEvent | ErrorEvent
