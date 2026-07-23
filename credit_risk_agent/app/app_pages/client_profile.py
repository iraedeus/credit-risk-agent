import sqlite3

import pandas as pd
import streamlit as st
import torch
from numpy import ndarray

from credit_risk_agent.config import ID_COL, MODEL_SAVE_PATH, SCALER_COLS, SCALER_PATH, TEST_DATABASE_PATH
from credit_risk_agent.data.standard_scaler import StandardScaler
from credit_risk_agent.model.dataset import prepare_dataset
from credit_risk_agent.model.model import CreditDefaultPredictor
from scripts.train import load_and_preprocess_from_db

SEX_MAP = {1: "Мужской", 2: "Женский"}
EDUCATION_MAP = {1: "Аспирантура/Магистратура", 2: "Университет", 3: "Старшая школа", 4: "Другое"}
MARRIAGE_MAP = {1: "Женат / Замужем", 2: "Холост / Не замужем", 3: "Другое"}


@st.cache_data(ttl="30m")
def get_available_clients_id() -> ndarray:
    with sqlite3.connect(TEST_DATABASE_PATH) as conn:
        client_ids = pd.read_sql_query("SELECT client_id FROM clients", conn)
        return client_ids[ID_COL].values.astype(int)


@st.cache_data(ttl="30m")
def get_client_full_data(client_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    with sqlite3.connect(TEST_DATABASE_PATH) as conn:
        client_info = pd.read_sql_query("SELECT * FROM clients WHERE client_id = ?", conn, params=[client_id])
        history = pd.read_sql_query(
            "SELECT month, pay_status, bill_amt, pay_amt FROM payment_history WHERE client_id = ? ORDER BY month ASC",
            conn,
            params=[client_id],
        )

        return client_info, history


@st.cache_resource
def load_ml_model() -> CreditDefaultPredictor:
    model = CreditDefaultPredictor(hidden_size=64, num_layers=1, static_size=14, dropout_prob=0.28)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    return model


@st.cache_data(ttl="30m")
def load_processed_test_dataset() -> pd.DataFrame:
    test_df = load_and_preprocess_from_db(TEST_DATABASE_PATH)
    scaler = StandardScaler.load(SCALER_PATH)
    return scaler.transform(test_df, SCALER_COLS)


@st.cache_data(ttl="30m")
def run_model(client_id: int) -> float:
    model = load_ml_model()
    test_df = load_processed_test_dataset()

    client_test_df = test_df[test_df["client_id"] == client_id]

    test_dataset = prepare_dataset(client_test_df)

    with torch.no_grad():
        score = torch.sigmoid(model(test_dataset[0][0].unsqueeze(0), test_dataset[0][1].unsqueeze(0))).item()
        return float(score)


st.title("Профиль клиента", anchor=False)
available_ids = get_available_clients_id()
selected_client_id = st.selectbox("Выберите ID клиента", options=available_ids, index=0)

client_info, history = get_client_full_data(selected_client_id)

row = client_info.iloc[0]

with st.expander("Демографический профиль клиента", icon=":material/badge:"):
    demo_col1, demo_col2, demo_col3, demo_col4 = st.columns(4)

    with demo_col1:
        st.write(f"**Возраст**: {int(row['age'])} лет")
    with demo_col2:
        st.write(f"**Пол**: {SEX_MAP.get(row['sex'], 'Не указан')}")
    with demo_col3:
        st.write(f"**Образование**: {EDUCATION_MAP.get(row['education'], 'Не указано')}")
    with demo_col4:
        st.write(f"**Семейный статус**: {MARRIAGE_MAP.get(row['marriage'], 'Не указано')}")

limit_bal = client_info["limit_bal"].iloc[0]
avg_bill = history["bill_amt"].mean()
avg_pay = history["pay_amt"].mean()
utilization = (avg_bill / limit_bal * 100) if limit_bal else 0
sum_bill = history["bill_amt"].sum()
sum_pay = history["pay_amt"].sum()
repayment_rate = (sum_pay / sum_bill * 100) if sum_bill > 0 else 0
delay_count = (history["pay_status"] > 0).sum()

with st.container(border=True):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Кредитный лимит", f"{limit_bal:,.0f} ₽", delta=f"{utilization:.1f}% утилизация", delta_color="off")
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

        score = run_model(selected_client_id)
        is_high_risk = score >= 0.5

        st.metric(
            "Риск дефолта (ML)",
            f"{score * 100:.1f}%",
            delta="Высокий риск" if is_high_risk else "Низкий риск",
            delta_color="inverse" if is_high_risk else "normal",
        )


tab1, tab2 = st.tabs(["Динамика счетов и выплат", "История просрочек"])
with tab1:
    st.bar_chart(history, x="month", y=["bill_amt", "pay_amt"])
with tab2:
    st.bar_chart(history, x="month", y="pay_status")
