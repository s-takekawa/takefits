"""Scaling helpers for usecases."""
from __future__ import annotations

import numpy as np

from takefits.core.app_state import AppState
from takefits.logic.data_tools import materialize_elementwise_inputs


def compute_scaled(data: np.ndarray, scale_factor: float) -> np.ndarray:
    """Return a scaled copy of the input data."""
    (data,) = materialize_elementwise_inputs(
        data,
        operation_name="Scaling",
        output_array_count=1,
    )
    return data * float(scale_factor)


def apply_scaling(state: AppState, scale_factor: float) -> AppState:
    """Apply a scalar multiplication to state.data in-place."""
    if state.data is None:
        raise ValueError("No data loaded")
    state.data = compute_scaled(state.data, scale_factor)
    return state
