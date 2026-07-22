from credit_risk_agent.agent.tools.get_client_financial_metrics import get_client_financial_metrics
from credit_risk_agent.agent.tools.run_model import run_model
from credit_risk_agent.agent.tools.sql_query import sql_query
from credit_risk_agent.agent.tools.tool import Tool

__all__ = ["Tool", "get_client_financial_metrics", "run_model", "sql_query"]


TOOLS = {
    "get_client_financial_metrics": Tool(get_client_financial_metrics),
    "run_model": Tool(run_model),
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
