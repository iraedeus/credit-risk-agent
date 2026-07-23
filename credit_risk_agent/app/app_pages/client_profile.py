import sqlite3

import pandas as pd
import streamlit as st
from numpy import ndarray

from credit_risk_agent.config import ID_COL, TEST_DATABASE_PATH


@st.cache_data(ttl="30m")
def get_available_clients_id() -> ndarray:
    with sqlite3.connect(TEST_DATABASE_PATH) as conn:
        client_ids = pd.read_sql_query("SELECT client_id FROM clients", conn)
        return client_ids[ID_COL].values.astype(int)


@st.cache_data(ttl="30m")
def get_client_full_data(client_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    with sqlite3.connect(TEST_DATABASE_PATH) as conn:
        client_info = pd.read_sql_query("SELECT limit_bal FROM clients WHERE client_id = ?", conn, params=[client_id])
        history = pd.read_sql_query(
            "SELECT month, pay_status, bill_amt, pay_amt FROM payment_history WHERE client_id = ? ORDER BY month ASC",
            conn,
            params=[client_id],
        )

        return client_info, history


st.title("Профиль клиента", anchor=False)
available_ids = get_available_clients_id()
selected_client_id = st.selectbox("Выберите ID клиента", options=available_ids, index=0)

client_info, history = get_client_full_data(selected_client_id)

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


tab1, tab2 = st.tabs(["Динамика счетов и выплат", "История просрочек"])
with tab1:
    st.bar_chart(history, x="month", y=["bill_amt", "pay_amt"])
with tab2:
    st.bar_chart(history, x="month", y="pay_status")
