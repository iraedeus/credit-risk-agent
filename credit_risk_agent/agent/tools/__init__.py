from credit_risk_agent.agent.tools.get_client_financial_metrics import get_client_financial_metrics
from credit_risk_agent.agent.tools.run_model import run_model
from credit_risk_agent.agent.tools.simulate_custom_scenario import simulate_custom_scenario
from credit_risk_agent.agent.tools.sql_query import sql_query
from credit_risk_agent.agent.tools.tool import Tool

__all__ = ["Tool", "get_client_financial_metrics", "run_model", "simulate_custom_scenario", "sql_query"]


TOOLS = {
    "get_client_financial_metrics": Tool(get_client_financial_metrics),
    "run_model": Tool(run_model),
    "simulate_custom_scenario": Tool(simulate_custom_scenario),
    "sql_query": Tool(sql_query),
}

PARAM_SCHEMAS = {
    "get_client_financial_metrics": {
        "type": "object",
        "properties": {"client_id": {"type": "integer", "description": "Уникальный идентификатор клиента"}},
        "required": ["client_id"],
    },
    "run_model": {
        "type": "object",
        "properties": {"client_id": {"type": "integer", "description": "Уникальный идентификатор клиента"}},
        "required": ["client_id"],
    },
    "simulate_custom_scenario": {
        "type": "object",
        "properties": {
            "client_id": {"type": "integer", "description": "Уникальный идентификатор клиента"},
            "params": {
                "type": "object",
                "description": "Словарь измененных параметров клиента для оценки гипотетических сценариев 'Что-Если'",
                "properties": {
                    "limit_bal": {"type": "number", "description": "Кредитный лимит (кредитный баланс)"},
                    "pay_0": {
                        "type": "integer",
                        "description": "Статус просрочки (-1: в срок, 1..8: просрочка в мес.)",
                    },
                    "pay_amt1": {"type": "number", "description": "Сумма последнего платежа"},
                    "bill_amt1": {"type": "number", "description": "Сумма последнего выставленного счета"},
                    "age": {"type": "integer", "description": "Возраст клиента"},
                },
            },
        },
        "required": ["client_id", "params"],
    },
    "sql_query": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Строка SQL SELECT запроса к базе данных."
                "Примечание: колонка default в таблице clients является зарезервированным словом SQLite,"
                'оборачивайте её в кавычки "default".',
            }
        },
        "required": ["query"],
    },
}


GIGACHAT_FUNCTIONS = [tool.to_gigachat_function(PARAM_SCHEMAS[name]) for name, tool in TOOLS.items()]
