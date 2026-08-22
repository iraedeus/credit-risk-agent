import os

import pytest
from dotenv import load_dotenv
from gigachat import GigaChat

from credit_risk_agent.agent.agent import CreditRiskAgent
from credit_risk_agent.config import DATA_SERVICE_URL, GIGACHAT_MODEL, ML_SERVICE_URL
from credit_risk_agent.services.data_service.client import get_data_service_client
from credit_risk_agent.services.ml_service.client import get_ml_service_client

load_dotenv()


class TestAgentLive:
    def test_agent_live_gigachat_api_call(self) -> None:
        """Verify real end-to-end execution against live GigaChat API when credentials are provided."""
        credentials = os.getenv("GIGACHAT_CREDENTIALS")
        if not credentials or credentials == "your_gigachat_authorization_data":
            pytest.skip("GIGACHAT_CREDENTIALS environment variable is not set, skipping live API test.")

        if not get_data_service_client().get_healthcheck():
            pytest.skip(f"Data Service недоступен по {DATA_SERVICE_URL}, пропускаем тест.")

        if not get_ml_service_client().get_healthcheck():
            pytest.skip(f"ML Service недоступен по {ML_SERVICE_URL}, пропускаем тест.")

        with GigaChat(credentials=credentials, model=GIGACHAT_MODEL, verify_ssl_certs=False) as client:
            agent = CreditRiskAgent(client=client, max_iterations=10)
            response = agent.run("Оцени кредитоспособность клиента с client_id=1")

            assert isinstance(response, str)
            assert len(response) > 0
            assert response != "Достигнуто максимальное количество итераций без итогового вердикта."
