"""Tool tests: calculator, SQL, web search fallback."""

import pytest

from app.tools.calculator import calculator, evaluate
from app.tools.sql_tool import sql_query, sql_schema
from app.tools.web_search import web_search


def test_calculator_basic_arithmetic():
    assert evaluate("0.25 * 840") == 210
    assert evaluate("(120 - 80) / 80 * 100") == 50.0
    assert evaluate("sqrt(144)") == 12.0


def test_calculator_tool_returns_text():
    assert "210" in calculator.invoke({"expression": "0.25 * 840"})


def test_calculator_blocks_unsafe_input():
    with pytest.raises(ValueError):
        evaluate("__import__('os').system('ls')")
    assert "error" in calculator.invoke({"expression": "open('x')"}).lower()


def test_calculator_handles_division_by_zero():
    assert "division by zero" in calculator.invoke({"expression": "1/0"}).lower()


def test_sql_tool_reads_sample_sales():
    result = sql_query.invoke({"query": "SELECT SUM(amount) AS total FROM sales"})
    assert "total" in result.lower()
    assert "119700" in result.replace(".0", "")   # sum of the seeded rows


def test_sql_tool_blocks_writes():
    assert "only SELECT" in sql_query.invoke({"query": "DELETE FROM sales"})


def test_sql_tool_reports_errors():
    assert "SQL error" in sql_query.invoke({"query": "SELECT * FROM nope"})


def test_sql_schema_lists_tables():
    schema = sql_schema.invoke({})
    assert "sales(" in schema and "documents(" in schema


def test_web_search_without_api_key_explains_itself():
    message = web_search.invoke({"query": "langgraph"})
    assert "TAVILY_API_KEY" in message
