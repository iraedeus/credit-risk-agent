import json

from gigachat import GigaChat
from gigachat.models import Chat, Function, Messages, MessagesRole

from credit_risk_agent.agent.tools import GIGACHAT_FUNCTIONS, TOOLS
from credit_risk_agent.agent.tools.tool import Tool

DEFAULT_SYSTEM_PROMPT = """
Ты — Старший Риск-Офицер (Senior Credit Risk Officer) в департаменте рисков коммерческого банка.
Твоя задача — проводить глубокий, объективный и мотивированный анализ кредитоспособности заёмщиков
и выносить итоговое вердикт-решение по заявке.

### 1. ДОСТУПНЫЕ ИНСТРУМЕНТЫ
- `run_model(client_id)` — вызывает ML-модель (Predictor) и возвращает вероятность дефолта
  (PD, Probability of Default) в диапазоне [0.0, 1.0].
- `get_client_financial_metrics(client_id)` — рассчитывает ключевые агрегированные показатели:
  кредитный лимит, средний счет, среднюю/максимальную утилизацию лимита, средний платеж,
  Repayment Rate, максимальный статус просрочки и количество месяцев просрочек.
- `sql_query(query)` — позволяет выполнять произвольные SELECT-запросы к БД (таблицы `clients`
  и `payment_history`). ВАЖНО: колонка `default` в таблице `clients` — зарезервированное слово SQLite,
  её нужно оборачивать в кавычки: `"default"`.

### 2. АЛГОРИТМ ПРОВЕДЕНИЯ АНАЛИЗА
1. **Сбор базовых данных:** Всегда вызывай `run_model` и `get_client_financial_metrics` для целевого `client_id`.
2. **Дополнительная детализация (при необходимости):** Используй `sql_query`, если нужно проверить
   социально-демографические параметры (возраст, образование, семейное положение) или динамику счетов по месяцах.
3. **Сопоставление с Риск-Политикой:**
   - **Вероятность дефолта (PD из ML-модели):**
     - PD < 0.35 — Низкий риск
     - 0.35 <= PD < 0.55 — Умеренный/Средний риск
     - PD >= 0.55 — Высокий риск
   - **Дисциплина платежей (Max Delay Status):**
     - 0 или отрицательные значения (-1, -2) — Положительная история
     - 1 — Просрочка до 30 дней (Повышенный контроль)
     - >= 2 — Просрочка от 60 дней и более (Критический риск)
   - **Утилизация лимита (Utilization Rate):**
     - > 80% — Высокая долговая нагрузка (заёмщик «живёт на кредитку»)
   - **Коэффициент покрытия счетов (Repayment Rate):**
     - < 50% — Недостаточное погашение выставляемых счетов
4. **Формирование вердикта:**
   - **ОДОБРЕНО:** Низкий/умеренный PD (< 0.55), отсутствие длительных просрочек (статус <= 1),
     адекватный Repayment Rate.
   - **ОТКАЗАНО:** Высокий PD (>= 0.55) ИЛИ наличие тяжелых просрочек (статус >= 2) ИЛИ
     систематическое непогашение долга (Repayment Rate < 30%).
   - **РУЧНОЙ АНДЕРРАЙТИНГ:** Противоречивые метрики (например, низкая модель, но высокая утилизация > 90%
     и свежая просрочка), либо пограничный PD (0.50–0.55).

### 3. СТРУКТУРА ФИНАЛЬНОГО ОТВЕТА
Твой финальный ответ должен быть структурирован строго по следующему шаблону:

### 📊 Отчет по оценке кредитного риска заёмщика [ID клиента]

1. **Метрики риск-модели:**
   - Вероятность дефолта (PD): `X.XX%`
   - Оценка уровня риска: `[Низкий / Умеренный / Высокий]`

2. **Финансовый профиль и дисциплина:**
   - Кредитный лимит: `...`
   - Средняя утилизация лимита: `...%`
   - Коэффициент покрытия (Repayment Rate): `...%`
   - Максимальная просрочка за 6 мес.: `[Значение]` (Месяцев с просрочкой: `X из 6`)

3. **Факторы риска и сильные стороны:**
   - ➕ *Сильные стороны:* [Перечисли 1-2 фактора, например: высокий Repayment Rate, стабильные платежи]
   - ⚠️ *Факторы риска:* [Перечисли 1-2 фактора, например: высокая утилизация, наличие просрочек]

4. **Итоговое решение:**
   **[ОДОБРЕНО / ОТКАЗАНО / РУЧНОЙ АНДЕРРАЙТИНГ]** — *Краткое мотивированное резюме (1-2 предложения).*
""".strip()


class CreditRiskAgent:
    """
    ReAct agent for automated credit risk assessment using GigaChat function calling.

    Orchestrates the Reasoning-Acting loop by querying GigaChat API, invoking
    available domain tools (financial metrics calculation, ML model prediction,
    SQL querying), and aggregating step-by-step observations to formulate a final
    underwriting decision.

    Parameters
    ----------
    client : GigaChat
        An authenticated instance of GigaChat SDK client.
    tools : dict[str, Tool], default=TOOLS
        Dictionary mapping tool names to executable Tool wrapper instances.
    functions : list[Function], default=GIGACHAT_FUNCTIONS
        List of GigaChat Function schema definitions available to the agent.
    system_prompt : str, default=DEFAULT_SYSTEM_PROMPT
        System prompt guiding the agent's role, behavioral policy, and response structure.
    max_iterations : int, default=5
        Maximum number of Reasoning-Acting iterations before forcing termination.
    """

    def __init__(
        self,
        client: GigaChat,
        tools: dict[str, Tool] = TOOLS,
        functions: list[Function] = GIGACHAT_FUNCTIONS,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_iterations: int = 5,
    ) -> None:
        self.client = client
        self.tools = tools
        self.functions = functions
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations

        self.history: list[Messages] = [Messages(role=MessagesRole.SYSTEM, content=system_prompt)]

    def run(self, user_prompt: str, verbose: bool = False) -> str:
        """
        Execute the ReAct agent loop for a given user prompt.

        Parameters
        ----------
        user_prompt : str
            The input query or assessment instruction from the user.

        Returns
        -------
        str
            The agent's final text answer or termination message if max iterations are reached.
        """

        self.history.append(Messages(role=MessagesRole.USER, content=user_prompt))

        for i in range(self.max_iterations):
            response = self.client.chat(Chat(messages=self.history, functions=self.functions))
            message = response.choices[0].message
            self.history.append(message)

            if message.function_call:
                if message.content and verbose:
                    print(f"[Мысль {i + 1}]: {message.content}")

                func_name = message.function_call.name
                func_args = message.function_call.arguments or {}

                if verbose:
                    print(f"[Действие {i + 1}]: Вызов инструмента {func_name}")
                    print(f"Аргументы: {func_args}")

                if isinstance(func_args, str):
                    try:
                        func_args = json.loads(func_args)
                    except json.JSONDecodeError:
                        tool_res = "Ошибка: невалидный формат JSON в аргументах инструмента."
                        content_json = json.dumps({"result": tool_res}, ensure_ascii=False)
                        self.history.append(Messages(role=MessagesRole.FUNCTION, name=func_name, content=content_json))
                        continue

                if func_name in self.tools:
                    tool_res = self.tools[func_name](**func_args)
                else:
                    tool_res = f"Ошибка: Инструмент '{func_name}' не найден."

                if verbose:
                    print(f"[Наблюдение {i + 1}]: {tool_res}\n\n" + "=" * 50 + "\n")

                content_json = json.dumps({"result": tool_res}, ensure_ascii=False)
                self.history.append(Messages(role=MessagesRole.FUNCTION, name=func_name, content=content_json))

            else:
                return message.content or ""

        return "Достигнуто максимальное количество итераций без итогового вердикта."

    def clear_history(self) -> None:
        self.history = [Messages(role=MessagesRole.SYSTEM, content=self.system_prompt)]
