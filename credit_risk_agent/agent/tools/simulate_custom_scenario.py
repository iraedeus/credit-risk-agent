"""
What-If scenario simulation tool for evaluating modified client feature hypotheses.
"""

from typing import Any

from credit_risk_agent.schemas.client_schemas import ClientPaymentHistory, ClientProfile
from credit_risk_agent.services.data_service.client import get_data_service_client
from credit_risk_agent.services.data_service.exceptions import DataServiceHTTPError
from credit_risk_agent.services.ml_service.client import get_ml_service_client
from credit_risk_agent.services.ml_service.exceptions import MLServiceHTTPError
from credit_risk_agent.services.ml_service.schemas import ClientProfileHistory

VALID_PROFILE_KEYS = {"limit_bal", "age", "sex", "education", "marriage"}
VALID_HISTORY_KEYS = {"pay_status", "bill_amt", "pay_amt"}
ALL_VALID_KEYS = VALID_PROFILE_KEYS | VALID_HISTORY_KEYS


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


def _update_profile_history_params(
    profile_history: ClientProfileHistory, params: dict[str, Any]
) -> ClientProfileHistory:
    """
    Apply scenario modifications to a client's profile and payment history.

    Parameters
    ----------
    profile_history : ClientProfileHistory
        Original client profile and 6-month payment history.
    params : dict[str, Any]
        Dictionary mapping modified feature keys to new values.

    Returns
    -------
    ClientProfileHistory
        New ClientProfileHistory instance with updated values.
    """
    profile_updates = {k: v for k, v in params.items() if k in VALID_PROFILE_KEYS}
    profile_dict = profile_history.profile.model_dump()
    updated_profile = ClientProfile.model_validate({**profile_dict, **profile_updates})

    updated_history = []
    for h in profile_history.history:
        if h.month == 1:
            history_updates = {k: v for k, v in params.items() if k in VALID_HISTORY_KEYS}
            history_dict = h.model_dump()
            updated_history.append(ClientPaymentHistory.model_validate({**history_dict, **history_updates}))
        else:
            updated_history.append(h.model_copy())

    return ClientProfileHistory(profile=updated_profile, history=updated_history)


def simulate_custom_scenario(client_id: int, params: dict[str, Any]) -> str:
    """
    Simulate custom 'What-If' scenarios for a client by modifying specific features.

    Fetches full client details via Data Service microservice, applies custom
    feature modifications, and evaluates baseline and simulated default
    probabilities (PD) via the ML inference microservice.

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

    applied_changes = []
    ignored_keys = []

    for k, v in params.items():
        if k in ALL_VALID_KEYS:
            applied_changes.append(f"  - {k}: {v}")
        else:
            ignored_keys.append(k)

    if not applied_changes:
        return f"Были переданы некорректные параметры. Доступные для передачи параметры: {ALL_VALID_KEYS}"

    try:
        data_service_client = get_data_service_client()
        ml_service_client = get_ml_service_client()
        client_full_info = data_service_client.get_client(client_id)

        if client_full_info is None:
            return f"Клиент с client_id = {client_id} не был найден в базе данных."

        profile_history = ClientProfileHistory(profile=client_full_info.profile, history=client_full_info.history)
        simulated_profile_history = _update_profile_history_params(profile_history, params)

        try:
            old_pd = ml_service_client.predict(profile_history).default_probability
            new_pd = ml_service_client.predict(simulated_profile_history).default_probability
        except MLServiceHTTPError as err:
            return f"Ошибка ML Service: {err}"

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
