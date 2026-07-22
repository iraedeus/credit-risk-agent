from typing import Any

from gigachat.models import Function, FunctionParameters

from credit_risk_agent.agent.tools.tool import Tool


def sample_function(x: int, y: int = 10) -> int:
    """
    Calculate sum of x and y.

    Parameters
    ----------
    x : int
        First integer operand.
    y : int, default=10
        Second integer operand.

    Returns
    -------
    int
        The sum of x and y.
    """
    return x + y


def function_without_docstring(a: str) -> str:
    return f"Hello, {a}"


class TestTool:
    def test_tool_default_name_and_docstring(self) -> None:
        """Verify default tool name and docstring extraction when not explicitly provided."""
        # Arrange & Act
        tool = Tool(sample_function)

        # Assert
        assert tool.name == "sample_function"
        assert "Calculate sum of x and y." in tool.description

    def test_tool_custom_name_and_description(self) -> None:
        """Verify custom name and description override function attributes."""
        # Arrange & Act
        tool = Tool(sample_function, name="custom_sum", description="Custom tool description")

        # Assert
        assert tool.name == "custom_sum"
        assert tool.description == "Custom tool description"

    def test_tool_empty_docstring_fallback(self) -> None:
        """Verify fallback to empty string when function lacks docstring."""
        # Arrange & Act
        tool = Tool(function_without_docstring)

        # Assert
        assert tool.name == "function_without_docstring"
        assert tool.description == ""

    def test_tool_execution(self) -> None:
        """Verify calling Tool instance passes arguments to wrapped callable."""
        # Arrange
        tool = Tool(sample_function)

        # Act
        result = tool(5, y=15)

        # Assert
        assert result == 20

    def test_tool_to_gigachat_function(self) -> None:
        """Verify conversion of Tool instance to GigaChat Function object."""
        # Arrange
        tool = Tool(sample_function)
        parameter_schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "x": {"type": "integer"},
                "y": {"type": "integer"},
            },
            "required": ["x"],
        }

        # Act
        giga_func = tool.to_gigachat_function(parameter_schema)

        # Assert
        assert isinstance(giga_func, Function)
        assert giga_func.name == "sample_function"
        assert giga_func.description is not None
        assert "Calculate sum of x and y." in giga_func.description
        assert isinstance(giga_func.parameters, FunctionParameters)
