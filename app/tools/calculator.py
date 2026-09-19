"""Safe calculator tool.

Uses Python's `ast` module instead of `eval()` so arbitrary code cannot be
executed.  The LLM decides when to call this tool - nothing is keyword
triggered.
"""

from __future__ import annotations

import ast
import math
import operator

from langchain_core.tools import tool

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
    "pow": pow,
}
_CONSTANTS = {"pi": math.pi, "e": math.e}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name = node.func.id
        if name not in _FUNCTIONS:
            raise ValueError(f"unsupported function: {name}")
        return _FUNCTIONS[name](*[_eval(a) for a in node.args])
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval(e) for e in node.elts]  # type: ignore[return-value]
    raise ValueError("unsupported expression")


def evaluate(expression: str) -> float:
    """Evaluate a maths expression, raising ValueError on anything unsafe."""
    expression = expression.strip().replace("^", "**").replace(",", "")
    if not expression:
        raise ValueError("empty expression")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid expression: {exc.msg}") from exc
    return _eval(tree)


@tool("calculator")
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the numeric result.

    Use this for any arithmetic, percentages, ratios or growth rates instead
    of doing the maths yourself.  Write the expression in Python syntax, e.g.
    "0.25 * 840" for 25% of 840, or "(120 - 80) / 80 * 100" for a percentage
    increase.  Supported helpers: sqrt, log, log10, exp, floor, ceil, round,
    abs, min, max, sum, pi, e.
    """
    try:
        result = evaluate(expression)
    except ZeroDivisionError:
        return "Calculator error: division by zero."
    except ValueError as exc:
        return f"Calculator error: {exc}"
    except Exception as exc:  # pragma: no cover - defensive
        return f"Calculator error: {exc}"
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return f"{expression} = {result}"
