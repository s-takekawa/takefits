"""Regrid usecase helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np

from takefits.core.app_state import AppState


@dataclass
class RegridResult:
    """Result from regrid operation."""
    data: np.ndarray
    header: Any
    wcs: Any


def compute_regrid(
    state: AppState,
    params: Dict[str, Any],
    progress_callback: Optional[Callable[[int], None]] = None
) -> RegridResult:
    """
    Regrid the current data cube according to the provided parameters.

    Args:
        state: AppState with data, header, and WCS
        params: Parameter dict compatible with RegridPanel/logic.regrid_core
        progress_callback: Optional callback for progress updates (0-100)

    Returns:
        RegridResult with data, header, and WCS
    """
    if state.data is None:
        raise ValueError("No data loaded")

    from takefits.logic.regrid_core import RegridEngine
    from astropy.wcs import WCS

    engine = RegridEngine(
        state.data,
        state.wcs,
        state.header,
        filename=state.filepath,
        progress_callback=progress_callback,
    )
    data, header = engine.perform_regrid(params)

    new_wcs = None
    try:
        if header is not None:
            new_wcs = WCS(header)
    except Exception:
        new_wcs = None

    return RegridResult(data=data, header=header, wcs=new_wcs)
