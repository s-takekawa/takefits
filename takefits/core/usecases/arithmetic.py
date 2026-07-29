"""Arithmetic usecases."""
from __future__ import annotations

from typing import Literal, Optional

import ast
import numpy as np

from takefits.core.app_state import AppState
from takefits.logic.data_tools import materialize_elementwise_inputs


ArithmeticOp = Literal["add", "subtract", "multiply", "divide", "expression"]


def compute_arithmetic(
    data_a: np.ndarray,
    operation: str,
    data_b: Optional[np.ndarray] = None,
    expression: Optional[str] = None,
    scalar: Optional[float] = None
) -> np.ndarray:
    """
    Perform arithmetic operation on data arrays.

    Args:
        data_a: Primary data array (referred to as 'A' in expressions)
        operation: "add", "subtract", "multiply", "divide", or "expression"
        data_b: Secondary data array (referred to as 'B' in expressions)
        expression: NumPy expression string (e.g., "np.log10(A)", "A + B * 2")
        scalar: Scalar value for simple operations

    Returns:
        Result array

    Examples:
        # Simple addition
        result = compute_arithmetic(data, "add", scalar=10)

        # Array operation
        result = compute_arithmetic(data_a, "divide", data_b=data_b)

        # Custom expression
        result = compute_arithmetic(data, "expression",
                                    expression="np.where(A > 0, np.log10(A), np.nan)")
    """
    import warnings

    if operation == "expression":
        if expression is None:
            raise ValueError("Expression required for 'expression' operation")
        data_a, data_b = materialize_elementwise_inputs(
            data_a,
            data_b,
            operation_name="Arithmetic expression",
            # The evaluated expression and its mandatory float64 cast can
            # coexist with an additional NumPy intermediate.
            output_array_count=3,
        )

        # Build evaluation context
        context = {"np": np, "A": data_a}
        if data_b is not None:
            context["B"] = data_b

        # Add convenience functions
        context.update({
            'log10': np.log10,
            'exp': np.exp,
            'sqrt': np.sqrt,
            'where': np.where,
            'abs': np.abs,
            'sin': np.sin,
            'cos': np.cos,
            'tan': np.tan,
            'nan': np.nan,
            'inf': np.inf,
        })

        try:
            # Parse and evaluate expression
            tree = ast.parse(expression, mode='eval')

            # Transform chained comparisons for numpy compatibility
            tree = _ChainedComparisonTransformer().visit(tree)
            ast.fix_missing_locations(tree)

            code = compile(tree, filename="<expression>", mode="eval")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = eval(code, {"__builtins__": {}}, context)

            # Handle scalar results
            if isinstance(result, (int, float, np.number)):
                result = np.full_like(data_a, result, dtype=np.float64)

            # Handle boolean masks
            if result.dtype == bool:
                result = np.where(result, data_a, np.nan)

            return result.astype(np.float64)

        except Exception as e:
            raise ValueError(f"Expression evaluation failed: {e}")

    data_a, data_b = materialize_elementwise_inputs(
        data_a,
        data_b,
        operation_name="Arithmetic",
        # Array division uses both the quotient and np.where result.
        output_array_count=2 if operation == "divide" and data_b is not None else 1,
    )

    # Simple operations
    if operation == "add":
        if scalar is not None:
            return data_a + scalar
        elif data_b is not None:
            return data_a + data_b
        else:
            raise ValueError("Either scalar or data_b required for 'add'")

    elif operation == "subtract":
        if scalar is not None:
            return data_a - scalar
        elif data_b is not None:
            return data_a - data_b
        else:
            raise ValueError("Either scalar or data_b required for 'subtract'")

    elif operation == "multiply":
        if scalar is not None:
            return data_a * scalar
        elif data_b is not None:
            return data_a * data_b
        else:
            raise ValueError("Either scalar or data_b required for 'multiply'")

    elif operation == "divide":
        if scalar is not None:
            return data_a / scalar
        elif data_b is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                return np.where(data_b != 0, data_a / data_b, np.nan)
        else:
            raise ValueError("Either scalar or data_b required for 'divide'")

    else:
        raise ValueError(f"Unknown operation: {operation}")


def apply_arithmetic(
    state: AppState,
    operation: str,
    data_b_path: Optional[str] = None,
    expression: Optional[str] = None,
    scalar: Optional[float] = None,
) -> AppState:
    """
    Apply arithmetic to state.data in-place.

    If `data_b_path` is provided, data is loaded from the FITS primary HDU.
    """
    if state.data is None:
        raise ValueError("No data loaded")

    data_b = None
    if data_b_path:
        from astropy.io import fits

        with fits.open(data_b_path) as hdul:
            data_b = hdul[0].data

    state.data = compute_arithmetic(
        data_a=state.data,
        operation=operation,
        data_b=data_b,
        expression=expression,
        scalar=scalar,
    )
    return state


class _ChainedComparisonTransformer(ast.NodeTransformer):
    """Transform chained comparisons to bitwise AND for numpy compatibility."""

    def visit_Compare(self, node):
        if len(node.ops) > 1:
            comparisons = []
            left = node.left
            for op, right in zip(node.ops, node.comparators):
                comparisons.append(
                    ast.Compare(left=left, ops=[op], comparators=[right])
                )
                left = right

            result = comparisons[0]
            for next_comp in comparisons[1:]:
                result = ast.BinOp(left=result, op=ast.BitAnd(), right=next_comp)

            return self.visit(result)
        return node
