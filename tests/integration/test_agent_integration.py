from unittest.mock import MagicMock

import pytest
from gigachat.models import FunctionCall, Messages, MessagesRole

from credit_risk_agent.agent.agent import CreditRiskAgent
from credit_risk_agent.agent.tools import GIGACHAT_FUNCTIONS, TOOLS
from credit_risk_agent.config import DATABASE_PATH, MODEL_SAVE_PATH, SCALER_PATH


class TestAgentIntegration:
    def test_agent_end_to_end_with_real_tools(self) -> None:
        """Verify end-to-end agent execution with mocked LLM client and real database/model artifacts."""
        if not (DATABASE_PATH.exists() and MODEL_SAVE_PATH.exists() and SCALER_PATH.exists()):
            pytest.skip("Database or model artifacts are missing, skipping agent integration test.")

        # Arrange: Setup mock GigaChat client with 3-step conversation
        mock_client = MagicMock()

        # Step 1: Model requests get_client_financial_metrics for client 1
        msg_1 = Messages(
            role=MessagesRole.ASSISTANT,
            function_call=FunctionCall(name="get_client_financial_metrics", arguments={"client_id": 1}),
        )

        # Step 2: Model requests run_model for client 1
        msg_2 = Messages(
            role=MessagesRole.ASSISTANT,
            function_call=FunctionCall(name="run_model", arguments={"client_id": 1}),
        )

        # Step 3: Model returns final verdict
        msg_3 = Messages(
            role=MessagesRole.ASSISTANT,
            content="[ОДОБРЕНО]: Вероятность дефолта низкая, финансовые метрики в норме.",
        )

        resp_1 = MagicMock(choices=[MagicMock(message=msg_1)])
        resp_2 = MagicMock(choices=[MagicMock(message=msg_2)])
        resp_3 = MagicMock(choices=[MagicMock(message=msg_3)])

        mock_client.chat.side_effect = [resp_1, resp_2, resp_3]

        agent = CreditRiskAgent(
            client=mock_client,
            tools=TOOLS,
            functions=GIGACHAT_FUNCTIONS,
        )

        # Act
        verdict = agent.run("Оцени кредитный риск для клиента 1")

        # Assert
        assert "[ОДОБРЕНО]" in verdict
        assert mock_client.chat.call_count == 3
