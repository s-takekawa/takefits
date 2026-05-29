"""Export helpers for usecases."""
from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np

from takefits.core.app_state import AppState
from takefits.core.io.save_fits import write_fits


def export_data_fits(
    state: AppState,
    output_path: str,
    history_entries: Optional[list] = None
) -> str:
    """
    Export the current state data to a FITS file.

    Args:
        state: AppState containing data and header
        output_path: Path for output FITS file
        history_entries: Optional list of HISTORY entries

    Returns:
        The output file path
    """
    from astropy.io import fits

    if state.data is None:
        raise ValueError("No data loaded")

    data_to_save = state.data
    if isinstance(data_to_save, np.ma.MaskedArray):
        data_to_save = data_to_save.filled(np.nan)

    header = state.header.copy() if state.header is not None else fits.Header()
    header['NAXIS'] = data_to_save.ndim
    for i, dim in enumerate(data_to_save.shape[::-1], 1):
        header[f'NAXIS{i}'] = dim

    if history_entries:
        for entry in history_entries:
            header.add_history(entry)

    return write_fits(output_path, data_to_save, header)


def export_figure(
    figure: Any,
    output_path: str,
    dpi: int = 300,
    transparent: bool = True
) -> str:
    """
    Export a matplotlib figure to a file.

    Args:
        figure: matplotlib.figure.Figure object.
        output_path: Path to save the file.
        dpi: Resolution in dots per inch.
        transparent: Whether to save with a transparent background.

    Returns:
        The output file path.
    """
    ext = os.path.splitext(str(output_path))[1].lower()
    is_vector = ext in {".pdf", ".eps", ".svg"}
    try:
        if is_vector:
            import matplotlib as mpl

            # Preserve vector editability while avoiding backend path simplification.
            with mpl.rc_context({
                "path.simplify": False,
                "path.simplify_threshold": 0.0,
                "agg.path.chunksize": 0,
            }):
                figure.savefig(output_path, dpi=dpi, transparent=transparent)
        else:
            figure.savefig(output_path, dpi=dpi, transparent=transparent)
    except Exception as e:
        raise RuntimeError(f"Failed to save figure to {output_path}: {e}")

    return output_path
