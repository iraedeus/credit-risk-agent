import os

import pytest
from dotenv import load_dotenv
from gigachat import GigaChat

from credit_risk_agent.agent.agent import CreditRiskAgent
from credit_risk_agent.config import DATABASE_PATH, MODEL_SAVE_PATH

load_dotenv()


class TestAgentLive:
    def test_agent_live_gigachat_api_call(self) -> None:
        """Verify real end-to-end execution against live GigaChat API when credentials are provided."""
        credentials = os.getenv("GIGACHAT_CREDENTIALS")
        if not credentials or credentials == "your_gigachat_authorization_data":
            pytest.skip("GIGACHAT_CREDENTIALS environment variable is not set, skipping live API test.")

        if not (DATABASE_PATH.exists() and MODEL_SAVE_PATH.exists()):
            pytest.skip("Database or model artifacts are missing, skipping live API test.")

        with GigaChat(credentials=credentials, verify_ssl_certs=False) as client:
            agent = CreditRiskAgent(client=client, max_iterations=10)
            response = agent.run("Оцени кредитоспособность клиента с client_id=1")

            assert isinstance(response, str)
            assert len(response) > 0
            assert response != "Достигнуто максимальное количество итераций без итогового вердикта."
