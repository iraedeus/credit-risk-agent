import os

import streamlit as st
from dotenv import load_dotenv
from gigachat import GigaChat

from credit_risk_agent.agent.agent import CreditRiskAgent
from credit_risk_agent.agent.events import (
    ErrorEvent,
    FinalEvent,
    ObservationEvent,
    ThoughtEvent,
    ToolCallEvent,
)

st.title("AI-агент")

load_dotenv()


# Инициализация GigaChat клиента и CreditRiskAgent
@st.cache_resource
def get_agent() -> CreditRiskAgent | None:
    credentials = os.getenv("GIGACHAT_CREDENTIALS") or st.secrets.get("GIGACHAT_CREDENTIALS", None)
    if not credentials or credentials == "your_gigachat_authorization_data":
        return None

    client = GigaChat(credentials=credentials, verify_ssl_certs=False)
    return CreditRiskAgent(client=client, max_iterations=15)


agent = get_agent()

if agent is None:
    st.warning(
        "⚠️ Не найден валидный `GIGACHAT_CREDENTIALS`. Укажите ключ в файле `.env` или `.streamlit/secrets.toml`."
    )
else:
    # Инициализация истории сообщений в session_state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Кнопка очистки диалога в сайдбаре
    with st.sidebar:
        if st.button("🗑️ Очистить историю", key="clear_chat"):
            st.session_state.messages = []
            agent.clear_history()
            st.rerun()

    # Готовые подсказки для быстрого старта (показываются при пустой истории)
    if not st.session_state.messages:
        suggestions = {
            "📊 Оцени риск клиента 10": "Оцени кредитный риск заёмщика с client_id = 10",
            "🔍 Проверь клиента 42": "Проведи полный анализ заёмщика client_id = 42",
        }
        selected = st.pills("Быстрый старт:", list(suggestions.keys()), label_visibility="collapsed")
        if selected:
            st.session_state.messages.append({"role": "user", "content": suggestions[selected]})
            st.rerun()

    # Отображение сохраненной истории сообщений
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Поле ввода сообщения
    if prompt := st.chat_input("Введите ID клиента или вопрос риск-офицеру..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Выполнение ReAct цикла агента
        with st.chat_message("assistant"):
            with st.status("🧠 Агент анализирует данные заёмщика...", expanded=True) as status:
                final_response = ""

                for event in agent.run_stream(prompt):
                    if isinstance(event, ThoughtEvent):
                        st.write(f"💭 **Размышление:** {event.content}")

                    elif isinstance(event, ToolCallEvent):
                        st.write(f"🛠️ **Вызов инструмента `{event.tool_name}`**")
                        st.json(event.tool_args)

                    elif isinstance(event, ObservationEvent):
                        st.write(f"👁️ **Результат инструмента `{event.tool_name}`:**")
                        st.code(event.content, language="json")

                    elif isinstance(event, FinalEvent):
                        final_response = event.content
                        status.update(
                            label="✅ Анализ завершен!",
                            state="complete",
                            expanded=False,
                        )

                    elif isinstance(event, ErrorEvent):
                        st.error(f"❌ Ошибка: {event.content}")
                        final_response = f"Произошла ошибка при анализе: {event.content}"
                        status.update(
                            label="❌ Анализ завершен с ошибкой",
                            state="error",
                            expanded=True,
                        )

            if final_response:
                st.markdown(final_response)
                st.session_state.messages.append({"role": "assistant", "content": final_response})
