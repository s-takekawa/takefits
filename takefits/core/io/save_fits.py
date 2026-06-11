"""Headless FITS save helpers (no PyQt dependencies)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

_DATAMIN_DATAMAX_CHUNK_BYTES = 1 * 1024 ** 3


def _data_nbytes(data: Any) -> int | None:
    try:
        nbytes = getattr(data, "nbytes", None)
        if nbytes is not None:
            return int(nbytes)
    except (TypeError, ValueError):
        pass

    try:
        return int(data.size) * np.dtype(data.dtype).itemsize
    except (AttributeError, TypeError, ValueError):
        return None


def _should_chunk_extrema(data: Any) -> bool:
    if isinstance(data, np.memmap):
        return True
    nbytes = _data_nbytes(data)
    return bool(nbytes is not None and nbytes > _DATAMIN_DATAMAX_CHUNK_BYTES)


def _update_extrema_from_block(
    block: Any,
    current_min: float,
    current_max: float,
    any_finite: bool,
) -> tuple[float, float, bool]:
    if np.ma.isMaskedArray(block):
        values = block.compressed()
    else:
        values = np.asarray(block)

    if values.size == 0:
        return current_min, current_max, any_finite

    with np.errstate(all="ignore"):
        finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        return current_min, current_max, any_finite

    finite_values = values[finite_mask]
    block_min = float(np.min(finite_values))
    block_max = float(np.max(finite_values))
    if not any_finite:
        return block_min, block_max, True
    return min(current_min, block_min), max(current_max, block_max), True


def _finite_extrema_chunked(data: Any) -> tuple[float, float, bool]:
    shape = tuple(int(dim) for dim in getattr(data, "shape", ()) or ())
    if not shape:
        return _update_extrema_from_block(data, np.inf, -np.inf, False)

    try:
        dtype_itemsize = np.dtype(getattr(data, "dtype", np.float64)).itemsize
    except TypeError:
        dtype_itemsize = np.dtype(np.float64).itemsize

    trailing = 1
    for dim in shape[1:]:
        trailing *= max(1, int(dim))
    bytes_per_outer = max(1, dtype_itemsize * trailing)
    rows_per_chunk = max(1, _DATAMIN_DATAMAX_CHUNK_BYTES // bytes_per_outer)

    dmin, dmax, any_finite = np.inf, -np.inf, False
    for start in range(0, shape[0], rows_per_chunk):
        end = min(shape[0], start + rows_per_chunk)
        block = data[start:end]
        dmin, dmax, any_finite = _update_extrema_from_block(
            block, dmin, dmax, any_finite
        )
    return dmin, dmax, any_finite


def update_datamin_datamax_if_present(header: Any, data: Any) -> None:
    """Update DATAMIN/DATAMAX in header if either keyword exists."""
    if header is None:
        return
    if 'DATAMIN' not in header and 'DATAMAX' not in header:
        return

    data_array = data
    if _should_chunk_extrema(data_array):
        dmin, dmax, any_finite = _finite_extrema_chunked(data_array)
        if any_finite:
            header['DATAMIN'] = float(dmin)
            header['DATAMAX'] = float(dmax)
        else:
            header['DATAMIN'] = np.nan
            header['DATAMAX'] = np.nan
        return

    if np.ma.isMaskedArray(data_array):
        values = data_array.compressed()
    else:
        values = np.asarray(data_array)

    with np.errstate(all='ignore'):
        finite = values[np.isfinite(values)]

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
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fits.writeto(target, data, header, overwrite=overwrite)
    return str(target)
