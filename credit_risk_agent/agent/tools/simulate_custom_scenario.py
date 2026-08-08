from typing import Any

from credit_risk_agent.config import TEST_DATABASE_PATH
from credit_risk_agent.data.loader import load_and_preprocess_from_db
from credit_risk_agent.model.loader import load_model_from_mlflow, load_scaler_from_mlflow
from credit_risk_agent.model.predictor import CreditRiskPredictor


def _get_risk_level(pd_val: float) -> str:
    if pd_val < 0.35:
        return "Низкий риск"
    elif pd_val < 0.55:
        return "Умеренный/Средний риск"
    else:
        return "Высокий риск"


def simulate_custom_scenario(client_id: int, params: dict[str, Any]) -> str:
    """
    Simulate custom 'What-If' scenarios for a client by modifying specific features
    and predicting the new credit default probability (PD).

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
    raw_df = load_and_preprocess_from_db(TEST_DATABASE_PATH)
    client_raw_df = raw_df[raw_df["client_id"] == client_id]
    if len(client_raw_df) == 0:
        return f"Клиент с client_id = {client_id} не был найден в базе данных."

    scaler = load_scaler_from_mlflow()
    model = load_model_from_mlflow()

    predictor = CreditRiskPredictor(model, scaler)

    old_pd = predictor.predict_pd(client_raw_df)

    simulated_raw_df = client_raw_df.copy()
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

    new_pd = predictor.predict_pd(simulated_raw_df)
    delta_pd = new_pd - old_pd
    delta_pct = delta_pd * 100

    old_risk = _get_risk_level(old_pd)
    new_risk = _get_risk_level(new_pd)

    changes_str = "\n".join(applied_changes)
    warning_str = f"\n⚠️ Игнорируемые неизвестные ключи: {ignored_keys}" if ignored_keys else ""

    sign = "+" if delta_pd > 0 else ""
    if delta_pd < 0:
        dynamics_str = "Снижение риска 🟢"
    elif delta_pd > 0:
        dynamics_str = "Увеличение риска 🔴"
    else:
        dynamics_str = "Без изменений ⚪"

    return (
        f"📊 Результаты What-If симуляции для клиента id={client_id}:\n\n"
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
