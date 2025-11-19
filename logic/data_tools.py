"""Utility helpers for working with large FITS data arrays."""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

# Default thresholds for data handling. Adjust here if you need different limits.
DEFAULT_MEMMAP_THRESHOLD_MB = 1024
MEMMAP_THRESHOLD_BYTES = DEFAULT_MEMMAP_THRESHOLD_MB * 1024 * 1024

# Default sampling target when deriving quick statistics from large arrays.
# Lowering the cap keeps lazy-loading responsive on very large cubes.
_DEFAULT_MAX_SAMPLES = 1_000_000


def estimate_array_nbytes(array: np.ndarray | np.memmap | None) -> int | None:
    """Return the approximate number of bytes for the given array.

    The helper works for both ``numpy.ndarray`` and ``numpy.memmap`` instances.
    ``None`` is returned if the shape or dtype is not available.
    """
    if array is None:
        return None

    try:
        return int(array.size) * array.dtype.itemsize
    except (AttributeError, TypeError, ValueError):
        return None


def _select_sample(flat: np.ndarray, max_samples: int) -> np.ndarray:
    """Select a representative 1-D sample from ``flat`` with at most ``max_samples`` values."""
    total = flat.size
    if total == 0:
        return flat

    if total <= max_samples:
        return flat

    step = max(1, total // max_samples)
    sample = flat[::step]
    if sample.size > max_samples:
        sample = sample[:max_samples]

    # Ensure the last element is included so we do not miss a possible extremum.
    if sample.size == 0 or sample[-1] != flat[-1]:
        sample = np.concatenate((sample, flat[-1:]))

    return sample


def fast_nanminmax(
    array: np.ndarray | np.memmap,
    max_samples: int = _DEFAULT_MAX_SAMPLES,
) -> Tuple[float, float]:
    """Return an approximate ``(nanmin, nanmax)`` pair for ``array``.

    The function is designed for very large arrays backed by memory maps. Instead
    of materialising the full dataset, a sub-sample is inspected. The result is
    sufficient for display initialisation while avoiding multi-gigabyte scans.
    """
    arr = np.asanyarray(array)
    if arr.size == 0:
        return (math.nan, math.nan)

    flat = arr.reshape(-1)
    sample = _select_sample(flat, max_samples)

    with np.errstate(all="ignore"):
        finite = sample[np.isfinite(sample)]
        if finite.size == 0:
            return (math.nan, math.nan)

        return (float(np.min(finite)), float(np.max(finite)))
