"""Utility helpers for working with large FITS data arrays."""
from __future__ import annotations

import math
from typing import Any, Tuple

import numpy as np

# Default thresholds for data handling. Adjust here if you need different limits.
DEFAULT_MEMMAP_THRESHOLD_MB = 1024
MEMMAP_THRESHOLD_BYTES = DEFAULT_MEMMAP_THRESHOLD_MB * 1024 * 1024

# Large Data Mode is intentionally a higher bar than the memmap threshold.
# The 1 GiB threshold enables lighter-weight code paths, while the 8 GiB
# threshold advertises a dedicated browse-first experience.
DEFAULT_LARGE_DATA_MODE_THRESHOLD_MB = 8192
LARGE_DATA_MODE_THRESHOLD_BYTES = DEFAULT_LARGE_DATA_MODE_THRESHOLD_MB * 1024 * 1024

# FITS scaling keywords disable the fast memmap path in astropy, so use a lower
# threshold for large-data handling when those keywords are present.
DEFAULT_LARGE_DATA_NO_MEMMAP_THRESHOLD_MB = 2048
LARGE_DATA_MODE_NO_MEMMAP_THRESHOLD_BYTES = (
    DEFAULT_LARGE_DATA_NO_MEMMAP_THRESHOLD_MB * 1024 * 1024
)

# Keep browse-mode display work near screen resolution.
DEFAULT_LARGE_DATA_DISPLAY_MAX_DIM = 2048

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


def format_nbytes(num_bytes: int | None) -> str:
    """Return a compact human-readable byte string."""
    if num_bytes is None:
        return "unknown size"
    if num_bytes < 0:
        num_bytes = 0

    value = float(num_bytes)
    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
    unit_index = 0
    while value >= 1024.0 and unit_index < len(units) - 1:
        value /= 1024.0
        unit_index += 1
    if unit_index == 0:
        return f"{int(value)} {units[unit_index]}"
    return f"{value:.1f} {units[unit_index]}"


def header_has_scaling_keywords(header: Any | None) -> bool:
    """Return True when FITS scaling keywords are present."""
    if header is None:
        return False
    for key in ("BZERO", "BSCALE", "BLANK"):
        try:
            if key in header:
                return True
        except Exception:
            continue
    return False


def _threshold_bytes_from_config(
    config: dict[str, Any] | None,
    key: str,
    default_mb: int,
) -> int:
    """Return a threshold in bytes, with optional config override in MiB."""
    if not isinstance(config, dict):
        return int(default_mb) * 1024 * 1024

    value = config.get(key, default_mb)
    try:
        value_mb = int(value)
    except (TypeError, ValueError):
        value_mb = int(default_mb)

    if value_mb <= 0:
        value_mb = int(default_mb)
    return value_mb * 1024 * 1024


def build_large_data_profile(
    array: np.ndarray | np.memmap | None,
    *,
    header: Any | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe whether the array should surface as Large Data Mode."""
    approx_bytes = estimate_array_nbytes(array)
    has_scaling_keywords = header_has_scaling_keywords(header)
    regular_threshold_bytes = _threshold_bytes_from_config(
        config,
        "large_data_mode_threshold_mb",
        DEFAULT_LARGE_DATA_MODE_THRESHOLD_MB,
    )
    no_memmap_threshold_bytes = _threshold_bytes_from_config(
        config,
        "large_data_no_memmap_threshold_mb",
        DEFAULT_LARGE_DATA_NO_MEMMAP_THRESHOLD_MB,
    )
    threshold_bytes = (
        no_memmap_threshold_bytes
        if has_scaling_keywords
        else regular_threshold_bytes
    )
    enabled = approx_bytes is not None and approx_bytes >= threshold_bytes

    if approx_bytes is None:
        reason = "Large Data Mode unavailable because data size could not be estimated."
    elif has_scaling_keywords and enabled:
        reason = (
            f"estimated size {format_nbytes(approx_bytes)} exceeds "
            f"{format_nbytes(threshold_bytes)} and FITS scaling keywords limit "
            "the fast memmap path"
        )
    elif enabled:
        reason = (
            f"estimated size {format_nbytes(approx_bytes)} exceeds "
            f"{format_nbytes(threshold_bytes)}"
        )
    else:
        reason = (
            f"estimated size {format_nbytes(approx_bytes)} is within the "
            f"{format_nbytes(threshold_bytes)} Large Data Mode threshold"
        )

    return {
        "enabled": bool(enabled),
        "estimated_size_bytes": approx_bytes,
        "estimated_size_text": format_nbytes(approx_bytes),
        "threshold_bytes": threshold_bytes,
        "threshold_text": format_nbytes(threshold_bytes),
        "has_scaling_keywords": bool(has_scaling_keywords),
        "reason": reason,
    }


def downsample_2d_for_display(
    array: np.ndarray | np.memmap,
    *,
    max_dimension: int = DEFAULT_LARGE_DATA_DISPLAY_MAX_DIM,
) -> np.ndarray:
    """Return a 2-D display array capped to roughly ``max_dimension`` pixels per axis."""
    arr = np.asanyarray(array)
    if arr.ndim != 2:
        return arr

    max_dimension = max(1, int(max_dimension))
    height, width = arr.shape
    step_y = max(1, int(math.ceil(height / max_dimension)))
    step_x = max(1, int(math.ceil(width / max_dimension)))
    if step_y == 1 and step_x == 1:
        if getattr(arr, "flags", None) is not None and arr.flags.c_contiguous:
            return arr
        return np.array(arr, copy=True)

    return np.array(arr[::step_y, ::step_x], copy=True)


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
