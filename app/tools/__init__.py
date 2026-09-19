"""Tool registry shared by the agents."""

from app.tools.calculator import calculator
from app.tools.sql_tool import sql_query, sql_schema
from app.tools.web_search import web_search

# Every tool the LLM may call. Adding a tool here makes it immediately
# available to the answering agent - no other change required.
TOOLS = [calculator, web_search, sql_query, sql_schema]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

__all__ = ["TOOLS", "TOOLS_BY_NAME", "calculator", "web_search", "sql_query", "sql_schema"]
