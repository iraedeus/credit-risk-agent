from unittest.mock import MagicMock

import pytest
from gigachat.models import FunctionCall, Messages, MessagesRole

from credit_risk_agent.agent.agent import DEFAULT_SYSTEM_PROMPT, CreditRiskAgent
from credit_risk_agent.agent.tools.tool import Tool


def mock_tool_func(client_id: int) -> str:
    return f"Metrics for client {client_id}"


class TestCreditRiskAgent:
    def test_agent_initialization(self) -> None:
        """Verify default and custom initialization attributes of CreditRiskAgent."""
        # Arrange
        mock_client = MagicMock()
        mock_tool = Tool(mock_tool_func, name="mock_tool", description="Mock tool")
        tools = {"mock_tool": mock_tool}

        # Act
        agent = CreditRiskAgent(
            client=mock_client,
            tools=tools,
            functions=[],
            system_prompt="Custom System Prompt",
            max_iterations=3,
        )

        # Assert
        assert agent.client == mock_client
        assert agent.tools == tools
        assert agent.system_prompt == "Custom System Prompt"
        assert agent.max_iterations == 3

    def test_agent_run_returns_final_answer_immediately(self) -> None:
        """Verify agent returns text content directly when no function calls are requested."""
        # Arrange
        mock_client = MagicMock()
        msg = Messages(role=MessagesRole.ASSISTANT, content="Credit approved.")
        mock_response = MagicMock(choices=[MagicMock(message=msg)])
        mock_client.chat.return_value = mock_response

        agent = CreditRiskAgent(client=mock_client, system_prompt=DEFAULT_SYSTEM_PROMPT)

        # Act
        result = agent.run("Evaluate client 101")

        # Assert
        assert result == "Credit approved."
        mock_client.chat.assert_called_once()

    def test_agent_run_executes_tool_call_and_returns_final_answer(self) -> None:
        """Verify agent executes requested tool, passes result to history, and returns final answer."""
        # Arrange
        mock_client = MagicMock()
        mock_tool = Tool(mock_tool_func, name="mock_tool")

        # Step 1: Model requests tool call
        msg_1 = Messages(
            role=MessagesRole.ASSISTANT,
            function_call=FunctionCall(name="mock_tool", arguments={"client_id": 101}),
        )

        # Step 2: Model returns final answer
        msg_2 = Messages(role=MessagesRole.ASSISTANT, content="Assessment complete. Low risk.")

        mock_resp_1 = MagicMock(choices=[MagicMock(message=msg_1)])
        mock_resp_2 = MagicMock(choices=[MagicMock(message=msg_2)])
        mock_client.chat.side_effect = [mock_resp_1, mock_resp_2]

        agent = CreditRiskAgent(client=mock_client, tools={"mock_tool": mock_tool})

        # Act
        result = agent.run("Analyze client 101")

        # Assert
        assert result == "Assessment complete. Low risk."
        assert mock_client.chat.call_count == 2

    def test_agent_run_handles_unknown_tool_name(self) -> None:
        """Verify agent passes error message to function role when requested tool name is unknown."""
        # Arrange
        mock_client = MagicMock()
        msg_1 = Messages(
            role=MessagesRole.ASSISTANT,
            function_call=FunctionCall(name="unknown_tool", arguments={}),
        )
        msg_2 = Messages(role=MessagesRole.ASSISTANT, content="Handled unknown tool")

        mock_resp_1 = MagicMock(choices=[MagicMock(message=msg_1)])
        mock_resp_2 = MagicMock(choices=[MagicMock(message=msg_2)])
        mock_client.chat.side_effect = [mock_resp_1, mock_resp_2]

        agent = CreditRiskAgent(client=mock_client, tools={})

        # Act
        result = agent.run("Execute query")

        # Assert
        assert result == "Handled unknown tool"
        assert mock_client.chat.call_count == 2

    def test_agent_run_handles_invalid_json_arguments(self) -> None:
        """Verify agent catches json.JSONDecodeError when function_call arguments are malformed."""
        # Arrange
        mock_client = MagicMock()
        msg_1 = Messages.model_construct(
            role=MessagesRole.ASSISTANT,
            function_call=FunctionCall.model_construct(name="mock_tool", arguments="{invalid_json}"),
        )
        msg_2 = Messages(role=MessagesRole.ASSISTANT, content="Handled invalid JSON")

        mock_resp_1 = MagicMock(choices=[MagicMock(message=msg_1)])
        mock_resp_2 = MagicMock(choices=[MagicMock(message=msg_2)])
        mock_client.chat.side_effect = [mock_resp_1, mock_resp_2]

        agent = CreditRiskAgent(client=mock_client, tools={})

        # Act
        result = agent.run("Evaluate client")

        # Assert
        assert result == "Handled invalid JSON"
        assert mock_client.chat.call_count == 2

    def test_agent_run_max_iterations_reached(self) -> None:
        """Verify agent returns fallback message when max_iterations limit is exceeded."""
        # Arrange
        mock_client = MagicMock()
        mock_tool = Tool(mock_tool_func, name="mock_tool")

        msg = Messages(
            role=MessagesRole.ASSISTANT,
            function_call=FunctionCall(name="mock_tool", arguments={"client_id": 101}),
        )
        mock_resp = MagicMock(choices=[MagicMock(message=msg)])
        mock_client.chat.return_value = mock_resp

        agent = CreditRiskAgent(client=mock_client, tools={"mock_tool": mock_tool}, max_iterations=2)

        # Act
        result = agent.run("Infinite loop test")

        # Assert
        assert result == "Достигнуто максимальное количество итераций без итогового вердикта."
        assert mock_client.chat.call_count == 2

    def test_agent_run_verbose_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify intermediate reasoning steps and tool execution logs are printed when verbose=True."""
        # Arrange
        mock_client = MagicMock()
        mock_tool = Tool(mock_tool_func, name="mock_tool")

        # Step 1: Model returns thought + tool call
        msg_1 = Messages(
            role=MessagesRole.ASSISTANT,
            content="Checking metrics first...",
            function_call=FunctionCall(name="mock_tool", arguments={"client_id": 101}),
        )

        # Step 2: Model returns final decision
        msg_2 = Messages(role=MessagesRole.ASSISTANT, content="Final decision: APPROVED")

        mock_resp_1 = MagicMock(choices=[MagicMock(message=msg_1)])
        mock_resp_2 = MagicMock(choices=[MagicMock(message=msg_2)])
        mock_client.chat.side_effect = [mock_resp_1, mock_resp_2]

        agent = CreditRiskAgent(client=mock_client, tools={"mock_tool": mock_tool})

        # Act
        result = agent.run("Evaluate client 101", verbose=True)

        # Assert
        assert result == "Final decision: APPROVED"
        captured = capsys.readouterr()
        assert "[Мысль 1]: Checking metrics first..." in captured.out
        assert "[Действие 1]: Вызов инструмента mock_tool" in captured.out
        assert "Аргументы: {'client_id': 101}" in captured.out
        assert "[Наблюдение 1]: Metrics for client 101" in captured.out
