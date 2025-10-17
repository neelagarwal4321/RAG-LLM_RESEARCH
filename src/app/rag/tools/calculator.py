from __future__ import annotations

import ast
import operator
from typing import Any, Dict

ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


class UnsafeExpressionError(Exception):
    pass


def evaluate(expression: str) -> Dict[str, Any]:
    node = ast.parse(expression, mode="eval")
    value = _evaluate_node(node.body)
    return {"expression": expression, "value": value}


def _evaluate_node(node: ast.AST) -> float:
    if isinstance(node, ast.Num):  # type: ignore[attr-defined]
        return float(node.n)  # type: ignore[attr-defined]
    if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED_OPERATORS:
        return ALLOWED_OPERATORS[type(node.op)](_evaluate_node(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_OPERATORS:
        left = _evaluate_node(node.left)
        right = _evaluate_node(node.right)
        return ALLOWED_OPERATORS[type(node.op)](left, right)
    raise UnsafeExpressionError(f"Unsupported expression: {ast.dump(node)}")
