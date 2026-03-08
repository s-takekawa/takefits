"""Headless FITS save helpers (no PyQt dependencies)."""
from __future__ import annotations

from typing import Any

import numpy as np


def update_datamin_datamax_if_present(header: Any, data: Any) -> None:
    """Update DATAMIN/DATAMAX in header if either keyword exists."""
    if header is None:
        return
    if 'DATAMIN' not in header and 'DATAMAX' not in header:
        return

    data_array = data
    if np.ma.isMaskedArray(data_array):
        finite = data_array.compressed()
    else:
        with np.errstate(all='ignore'):
            finite = data_array[np.isfinite(data_array)]

    if finite.size > 0:
        header['DATAMIN'] = float(np.min(finite))
        header['DATAMAX'] = float(np.max(finite))
    else:
        header['DATAMIN'] = np.nan
        header['DATAMAX'] = np.nan


def write_fits(output_path: str, data: Any, header: Any, *, overwrite: bool = True) -> str:
    """Write a FITS file with optional DATAMIN/DATAMAX refresh."""
    from astropy.io import fits

    update_datamin_datamax_if_present(header, data)
    fits.writeto(output_path, data, header, overwrite=overwrite)
    return output_path
