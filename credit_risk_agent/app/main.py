import streamlit as st

# Глобальная настройка страницы (должна быть первой вызвана)
st.set_page_config(
    page_title="Credit Risk Intelligence System",
    page_icon=":material/analytics:",
    layout="wide",
)

# Определение страниц приложения через st.Page
# Используем директорию app_pages вместо устаревшего авто-обнаружения pages/
page_profile = st.Page(
    "app_pages/client_profile.py",
    title="Профиль клиента",
    icon=":material/account_circle:",
    default=True,
)

page_chat = st.Page(
    "app_pages/chat.py",
    title="Чат с AI-агентом",
    icon=":material/chat:",
)

# Группировка страниц в навигацию
pg = st.navigation(
    {
        "Аналитика": [page_profile],
        "Ассистент": [page_chat],
    },
    position="sidebar",
)

# Общий заголовок приложения в сайдбаре или сверху
st.logo("https://raw.githubusercontent.com/streamlit/streamlit/main/docs/logo.png")  # Иконка/логотип

# Запуск активной страницы
pg.run()
