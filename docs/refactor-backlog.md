# ML Service Refactor Backlog

Отложенные улучшения. Рефакторить после того, как сервис будет готов.
Собрано в TDD-сессии: healthcheck + predict endpoint.

## Tests

1. **Happy-path тест качает настоящую модель из MLflow**
   `tests/unit/services/ml_service/test_ml_service_main.py::TestPredict::test_predict_success`
   - 10+ секунд на файл, зависимость от состояния `mlflow.db` (артефакты, алиасы).
   - Это unit-тест с поведением интеграционного.
   - Что делать: подсунуть `app.state.predictor` как `MagicMock` (или реальный `CreditRiskPredictor` с мокнутыми model/scaler) до запроса, без `with TestClient(...)` (lifespan не должен грузить модель).
   - Что проверять в unit-тесте: эндпоинт зовёт `predict_pd` с правильным DataFrame и заворачивает результат в `PredictionResponse`. Внутренности `predict_pd` — забота тестов модели.

2. **Копипаста payload**
   4 раза один и тот же профиль + история в `TestPredict`. Вынести в fixture/helper (`conftest.py` или модуль-билдер), тесты на 422 оставить на сырых dict.

3. **`print(response.json())` в тестах** — убрать.

4. **Нет docstring'ов в новых тестах** — в репо docstring'и есть у всего (NumPy-стиль). Добавить.

5. **422-тесты ходят через `with TestClient(...)`** — lifespan запускается зря (валидация происходит до эндпоинта). Убрать `with` там, где lifespan не нужен (или смириться — но знать, что MLflow дёргается).

## Main / API

6. **`request:Request` без пробела** в сигнатуре `predict` (main.py). Форматирование + порядок параметров (тело перед request — на вкус, но единообразно с healthcheck).

7. **Логирование в `lifespan`** — изначальный гэп:
   - `except Exception: pass` молча глотает причину недоступности модели.
   - Нужен logger: почему не подгрузилась модель (трейсбек в debug), какой алиас/имя модели пытались загрузить.
   - Обсудить: `logger.warning` при старте + healthcheck 503 — readiness-паттерн, причина должна быть видна в логах.

## Schemas

8. **Дублирование валидаторов** в `ClientProfileHistory`: `validate_history_len` и `validate_unique_months` можно объединить в один `model_validator(mode="after")` — оба про структуру history. Подумать про единое сообщение об ошибке (сейчас два разных текста).

9. **Docstring схемы** `ClientProfileHistory` обещает «6 monthly records» — теперь это валидатор, можно сослаться.

## Открытые вопросы (проверить)

10. **NaN в `default_probability`**: пропустит ли `PredictionResponse` (ge=0, le=1) NaN от `torch.sigmoid`? Поведение pydantic v2 с `allow_inf_nan` не проверяли. Если пропустит — решить, нужен ли guard.
11. **`HTTPException(503)` в predict** без `detail` — осознанное решение: канонический текст Starlette. Если решите от него отказаться — править оба эндпоинта и тесты.
12. **`ml_service/client.py` пустой** — контракт 422 (`{"detail": [{loc, msg, type}]}`) и 503 (`{"detail": "Service Unavailable"}`) пригодится, когда будем писать клиент.
