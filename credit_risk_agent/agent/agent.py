import json

from gigachat import GigaChat
from gigachat.models import Chat, Function, Messages, MessagesRole

from credit_risk_agent.agent.tools import GIGACHAT_FUNCTIONS, TOOLS
from credit_risk_agent.agent.tools.tool import Tool

DEFAULT_SYSTEM_PROMPT = """
Ты — старший кредитный рисковик в банке (Risk Officer).
Твоя задача — проводить комплексный анализ кредитоспособности клиентов и выносить мотивированное решение по заявке.

Инструкции:
1. Обязательно используй get_client_financial_metrics и run_model для полной оценки риска клиента.
2. Аргументируй вывод конкретными значениями: вероятностью дефолта (PD), уровнем утилизации лимита и статусом просрочек.
3. В конце давай четкое решение: [ОДОБРЕНО / ОТКАЗАНО / РУЧНОЙ АНДЕРРАЙТИНГ].
"""


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

    def run(self, user_prompt: str) -> str:
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
        messages = [
            Messages(role=MessagesRole.SYSTEM, content=self.system_prompt),
            Messages(role=MessagesRole.USER, content=user_prompt),
        ]

        for _ in range(self.max_iterations):
            response = self.client.chat(Chat(messages=messages, functions=self.functions))

            message = response.choices[0].message
            messages.append(message)

            if message.function_call:
                func_name = message.function_call.name
                func_args = message.function_call.arguments or {}

                if isinstance(func_args, str):
                    try:
                        func_args = json.loads(func_args)
                    except json.JSONDecodeError:
                        tool_res = "Ошибка: невалидный формат JSON в аргументах инструмента."
                        content_json = json.dumps({"result": tool_res}, ensure_ascii=False)
                        messages.append(Messages(role=MessagesRole.FUNCTION, name=func_name, content=content_json))
                        continue

                if func_name in self.tools:
                    tool_res = self.tools[func_name](**func_args)
                else:
                    tool_res = f"Ошибка: Инструмент '{func_name}' не найден."

                content_json = json.dumps({"result": tool_res}, ensure_ascii=False)
                messages.append(Messages(role=MessagesRole.FUNCTION, name=func_name, content=content_json))

            else:
                return message.content or ""

        return "Достигнуто максимальное количество итераций без итогового вердикта."
