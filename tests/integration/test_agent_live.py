import os

import pytest
from dotenv import load_dotenv
from gigachat import GigaChat

from credit_risk_agent.agent.agent import CreditRiskAgent
from credit_risk_agent.config import BEST_MODEL_ALIAS, BEST_MODEL_NAME, GIGACHAT_MODEL
from credit_risk_agent.model.loader import load_model_from_registry, load_scaler_from_registry
from credit_risk_agent.services.data_service.client import get_data_service_client

load_dotenv()


class TestAgentLive:
    def test_agent_live_gigachat_api_call(self) -> None:
        """Verify real end-to-end execution against live GigaChat API when credentials are provided."""
        credentials = os.getenv("GIGACHAT_CREDENTIALS")
        if not credentials or credentials == "your_gigachat_authorization_data":
            pytest.skip("GIGACHAT_CREDENTIALS environment variable is not set, skipping live API test.")

        if not get_data_service_client().get_healthcheck():
            pytest.skip("Data Service недоступен по DATA_SERVICE_URL, пропускаем тест.")

        try:
            load_model_from_registry(BEST_MODEL_NAME, BEST_MODEL_ALIAS)
            load_scaler_from_registry(BEST_MODEL_NAME, BEST_MODEL_ALIAS)
        except Exception:
            pytest.skip("Champion model or scaler not found in MLflow, skipping live API test.")

        with GigaChat(credentials=credentials, model=GIGACHAT_MODEL, verify_ssl_certs=False) as client:
            agent = CreditRiskAgent(client=client, max_iterations=10)
            response = agent.run("Оцени кредитоспособность клиента с client_id=1")

            assert isinstance(response, str)
            assert len(response) > 0
            assert response != "Достигнуто максимальное количество итераций без итогового вердикта."
