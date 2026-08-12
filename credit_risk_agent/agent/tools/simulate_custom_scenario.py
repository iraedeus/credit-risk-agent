"""
What-If scenario simulation tool for evaluating modified client feature hypotheses.
"""

from typing import Any

from credit_risk_agent.agent.tools.run_model import client_full_info_to_df
from credit_risk_agent.model.loader import load_model_from_mlflow, load_scaler_from_mlflow
from credit_risk_agent.model.predictor import CreditRiskPredictor
from credit_risk_agent.services.data_service.client import get_data_service_client
from credit_risk_agent.services.data_service.exceptions import DataServiceHTTPError


def _get_risk_level(pd_val: float) -> str:
    """
    Map probability of default (PD) value to human-readable risk category.

    Parameters
    ----------
    pd_val : float
        Probability of default float between 0.0 and 1.0.

    Returns
    -------
    str
        Risk level description string.
    """
    if pd_val < 0.35:
        return "Низкий риск"
    elif pd_val < 0.55:
        return "Умеренный/Средний риск"
    else:
        return "Высокий риск"


def simulate_custom_scenario(client_id: int, params: dict[str, Any]) -> str:
    """
    Simulate custom 'What-If' scenarios for a client by modifying specific features.

    Fetches full client details via DataServiceClient microservice, applies custom
    feature modifications, and predicts the new credit default probability (PD).

    Parameters
    ----------
    client_id : int
        The unique identifier of the client to analyze.
    params : dict[str, Any]
        Dictionary mapping feature names to their new simulated values.
        Example: {"limit_bal": 100000, "pay_0": 0}

    Returns
    -------
    str
        Formatted string containing baseline PD, simulated PD, delta change,
        and interpretation of the risk impact.
    """

    try:
        data_service_client = get_data_service_client()
        client_full_info = data_service_client.get_client(client_id)

        if client_full_info is None:
            return f"Клиент с client_id = {client_id} не был найден в базе данных."

        client_df = client_full_info_to_df(client_full_info)
        simulated_raw_df = client_df.copy()
        applied_changes = []
        ignored_keys = []

        for k, v in params.items():
            if k in simulated_raw_df.columns:
                simulated_raw_df[k] = v
                applied_changes.append(f"  - {k}: {v}")
            else:
                ignored_keys.append(k)

        if not applied_changes:
            return (
                f"Ошибка: Ни один из переданных параметров {list(params.keys())} "
                f"не содержится в признаках клиентов. Проверьте правильность имён полей."
            )

        scaler = load_scaler_from_mlflow()
        model = load_model_from_mlflow()

        predictor = CreditRiskPredictor(model, scaler)

        old_pd = predictor.predict_pd(client_df)

        new_pd = predictor.predict_pd(simulated_raw_df)
        delta_pd = new_pd - old_pd
        delta_pct = delta_pd * 100

        old_risk = _get_risk_level(old_pd)
        new_risk = _get_risk_level(new_pd)

        changes_str = "\n".join(applied_changes)
        warning_str = f"\n⚠️ Игнорируемые неизвестные ключи: {ignored_keys}" if ignored_keys else ""

        sign = "+" if delta_pd > 0 else ""
        if delta_pd < 0:
            dynamics_str = "Снижение риска"
        elif delta_pd > 0:
            dynamics_str = "Увеличение риска"
        else:
            dynamics_str = "Без изменений"

        return (
            f"Результаты What-If симуляции для клиента id={client_id}:\n\n"
            f"1. Базовый вариант (Baseline):\n"
            f"   - Вероятность дефолта (PD): {old_pd * 100:.2f}%\n"
            f"   - Категория риска: {old_risk}\n\n"
            f"2. Симулируемый сценарий:\n"
            f"   Примененные изменения:\n{changes_str}{warning_str}\n"
            f"   - Симулированный PD: {new_pd * 100:.2f}%\n"
            f"   - Категория риска: {new_risk}\n\n"
            f"3. Итог симуляции:\n"
            f"   - Изменение PD: {sign}{delta_pct:.2f}% п.п.\n"
            f"   - Динамика риска: {dynamics_str}"
        )
    except DataServiceHTTPError as err:
        return f"Ошибка Data Service: {err}"
