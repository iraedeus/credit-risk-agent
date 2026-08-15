"""
Streamlit dashboard page for client profile analysis and risk evaluation.
"""

import pandas as pd
import streamlit as st

from credit_risk_agent.agent.tools.run_model import client_full_info_to_df
from credit_risk_agent.config import BEST_MODEL_ALIAS, BEST_MODEL_NAME
from credit_risk_agent.model.loader import load_model_from_registry, load_scaler_from_registry
from credit_risk_agent.model.predictor import CreditRiskPredictor
from credit_risk_agent.services.data_service.client import get_data_service_client
from credit_risk_agent.services.data_service.schemas import ClientFullInfo

SEX_MAP = {1: "Мужской", 2: "Женский"}
EDUCATION_MAP = {1: "Аспирантура/Магистратура", 2: "Университет", 3: "Старшая школа", 4: "Другое"}
MARRIAGE_MAP = {1: "Женат / Замужем", 2: "Холост / Не замужем", 3: "Другое"}


@st.cache_data(ttl="30m")
def get_available_clients_id(offset: int = 0) -> list[int]:
    """
    Fetch array of unique client IDs present in the test database.

    Returns
    -------
    ndarray
        Array of integer client IDs.
    """
    data_service_client = get_data_service_client()
    return data_service_client.get_clients(offset=offset)


@st.cache_data(ttl="30m")
def get_client_full_data(client_id: int) -> ClientFullInfo | None:
    """
    Fetch demographic and 6-month payment history records for a specified client ID.

    Parameters
    ----------
    client_id : int
        Unique client identifier.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        Tuple containing client demographics DataFrame and payment history DataFrame.
    """
    data_service_client = get_data_service_client()
    return data_service_client.get_client(client_id)


@st.cache_resource
def get_credit_risk_predictor() -> CreditRiskPredictor:
    """
    Load MLflow model and scaler once and cache CreditRiskPredictor instance.

    Returns
    -------
    CreditRiskPredictor
        Cached predictor instance.
    """
    model = load_model_from_registry(BEST_MODEL_NAME, BEST_MODEL_ALIAS)
    scaler = load_scaler_from_registry(BEST_MODEL_NAME, BEST_MODEL_ALIAS)
    return CreditRiskPredictor(model, scaler)


st.title("Профиль клиента", anchor=False)

with st.container(border=True):
    col1, col2 = st.columns(2)
    with col1:
        page = st.number_input("Страница", min_value=1, step=1)
        offset = (page - 1) * 20
    with col2:
        available_ids = get_available_clients_id(offset)
        selected_client_id = st.selectbox("Выберите ID клиента", options=available_ids, index=0)

client_info = get_client_full_data(selected_client_id)

if client_info is None:
    st.warning(f"Клиент с id = {selected_client_id} не найден.")
else:
    with st.expander("Демографический профиль клиента", icon=":material/badge:"):
        demo_col1, demo_col2, demo_col3, demo_col4 = st.columns(4)

        with demo_col1:
            st.write(f"**Возраст**: {int(client_info.profile.age)} лет")
        with demo_col2:
            st.write(f"**Пол**: {SEX_MAP.get(client_info.profile.sex, 'Не указан')}")
        with demo_col3:
            st.write(f"**Образование**: {EDUCATION_MAP.get(client_info.profile.education, 'Не указано')}")
        with demo_col4:
            st.write(f"**Семейный статус**: {MARRIAGE_MAP.get(client_info.profile.marriage, 'Не указано')}")

    limit_bal = client_info.profile.limit_bal
    avg_bill = client_info.metrics.avg_bill
    avg_pay = client_info.metrics.avg_pay
    utilization = client_info.metrics.avg_utilization if limit_bal else 0
    repayment_rate = client_info.metrics.repayment_rate
    delay_count = client_info.metrics.delay_months_count

    with st.container(border=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Кредитный лимит", f"{limit_bal:,.0f} ₽", delta=f"{utilization:.1f}% утилизация", delta_color="off"
            )
        with col2:
            st.metric("Средний счёт", f"{avg_bill:,.0f} ₽", delta=f"Платёж: {avg_pay:,.0f} ₽", delta_color="off")
        with col3:
            st.metric("Покрытие счетов", f"{repayment_rate:.1f}%")
        with col4:
            st.write("**Статус дисциплины**")
            if delay_count > 0:
                st.badge(f"Просрочек: {delay_count} из 6 мес", color="red", icon=":material/warning:")
            else:
                st.badge("Без просрочек", color="green", icon=":material/check_circle:")

            client_info_df = client_full_info_to_df(client_info)
            predictor = get_credit_risk_predictor()
            score = predictor.predict_pd(client_info_df)
            is_high_risk = score >= 0.5

            st.metric(
                "Риск дефолта (ML)",
                f"{score * 100:.1f}%",
                delta="Высокий риск" if is_high_risk else "Низкий риск",
                delta_color="inverse" if is_high_risk else "normal",
            )

    history_df = pd.DataFrame([h.model_dump() for h in client_info.history])
    tab1, tab2 = st.tabs(["Динамика счетов и выплат", "История просрочек"])
    with tab1:
        st.bar_chart(history_df, x="month", y=["bill_amt", "pay_amt"])
    with tab2:
        st.bar_chart(history_df, x="month", y="pay_status")
