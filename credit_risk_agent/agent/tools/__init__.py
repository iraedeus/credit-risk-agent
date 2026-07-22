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
