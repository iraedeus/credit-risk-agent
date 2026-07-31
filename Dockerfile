FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

ENV POETRY_VIRTUALENVS_CREATE=false
RUN pip install poetry

COPY pyproject.toml poetry.lock ./
RUN poetry install --no-interaction --no-ansi --no-root

COPY . .
RUN poetry install --no-interaction --no-ansi

EXPOSE 8501

CMD ["streamlit", "run", "credit_risk_agent/app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
