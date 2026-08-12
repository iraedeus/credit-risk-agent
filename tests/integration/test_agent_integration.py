from unittest.mock import MagicMock

import pytest
from gigachat.models import FunctionCall, Messages, MessagesRole

from credit_risk_agent.agent.agent import CreditRiskAgent
from credit_risk_agent.agent.tools import GIGACHAT_FUNCTIONS, TOOLS
from credit_risk_agent.model.loader import load_model_from_mlflow, load_scaler_from_mlflow
from credit_risk_agent.services.data_service.client import get_data_service_client


class TestAgentIntegration:
    def test_agent_end_to_end_with_real_tools(self) -> None:
        """Verify end-to-end agent execution with mocked LLM client and real database/model artifacts."""

        if not get_data_service_client().get_healthcheck():
            pytest.skip("Data Service недоступен по DATA_SERVICE_URL, пропускаем тест.")

        try:
            load_model_from_mlflow()
            load_scaler_from_mlflow()
        except Exception:
            pytest.skip("Champion model or scaler not found in MLflow, skipping agent integration test.")

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
