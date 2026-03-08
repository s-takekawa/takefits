"""Scaling helpers for usecases."""
from __future__ import annotations

import numpy as np

from takefits.core.app_state import AppState


def compute_scaled(data: np.ndarray, scale_factor: float) -> np.ndarray:
    """Return a scaled copy of the input data."""
    return data * float(scale_factor)


def apply_scaling(state: AppState, scale_factor: float) -> AppState:
    """Apply a scalar multiplication to state.data in-place."""
    if state.data is None:
        raise ValueError("No data loaded")
    state.data = compute_scaled(state.data, scale_factor)
    return state
