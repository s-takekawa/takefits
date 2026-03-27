"""Utility helpers for working with large FITS data arrays."""
from __future__ import annotations

import math
import os
import sys
from typing import Any, Tuple

import numpy as np


def _get_total_ram_bytes() -> int | None:
    """Return total physical RAM in bytes, or None if unavailable.

    Works on macOS, Linux, and Windows without external dependencies.
    The call reads kernel metadata and completes in microseconds.
    """
    try:
        if sys.platform == "win32":
            import ctypes

            class _MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return int(stat.ullTotalPhys)
            return None
        else:
            # POSIX (macOS, Linux)
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            if pages > 0 and page_size > 0:
                return pages * page_size
            return None
    except (OSError, ValueError, AttributeError):
        return None


def _auto_large_data_threshold_mb() -> int:
    """Return a Large Data Mode threshold in MiB based on system RAM.

    Heuristic: 25% of total RAM, clamped to [2048, 8192] MiB.
    Falls back to 4096 MiB if RAM detection fails.
    """
    total = _get_total_ram_bytes()
    if total is None:
        return 4096
    quarter_mb = total // 4 // (1024 * 1024)
    return max(2048, min(8192, int(quarter_mb)))


# Default thresholds for data handling. Adjust here if you need different limits.
DEFAULT_MEMMAP_THRESHOLD_MB = 1024
MEMMAP_THRESHOLD_BYTES = DEFAULT_MEMMAP_THRESHOLD_MB * 1024 * 1024

# Large Data Mode activates downsampled display and sampled statistics.
# Threshold auto-scales to 25% of system RAM (clamped 2–8 GiB).
DEFAULT_LARGE_DATA_MODE_THRESHOLD_MB = _auto_large_data_threshold_mb()
LARGE_DATA_MODE_THRESHOLD_BYTES = DEFAULT_LARGE_DATA_MODE_THRESHOLD_MB * 1024 * 1024

# FITS scaling keywords disable the fast memmap path in astropy.  When lazy
# scaling is not available (legacy path), use a lower threshold.
DEFAULT_LARGE_DATA_NO_MEMMAP_THRESHOLD_MB = 2048
LARGE_DATA_MODE_NO_MEMMAP_THRESHOLD_BYTES = (
    DEFAULT_LARGE_DATA_NO_MEMMAP_THRESHOLD_MB * 1024 * 1024
)

# Size threshold (bytes) above which load_fits keeps a raw memmap and applies
# BZERO/BSCALE lazily per-slice instead of letting astropy materialise the
# entire scaled array.  Files below this use the traditional eager path
# (memmap=False with astropy handling BZERO/BSCALE/BLANK), which is fast
# enough for files that fit comfortably in memory.
LAZY_SCALING_THRESHOLD_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB

# Keep browse-mode display work near screen resolution.
DEFAULT_LARGE_DATA_DISPLAY_MAX_DIM = 2048

# Default sampling target when deriving quick statistics from large arrays.
# Lowering the cap keeps lazy-loading responsive on very large cubes.
_DEFAULT_MAX_SAMPLES = 1_000_000


class LazyScaledArray:
    """Wraps a raw FITS memmap and applies BZERO/BSCALE lazily per-slice.

    This avoids the astropy behaviour of materialising the entire scaled float
    array when BZERO/BSCALE/BLANK keywords are present, keeping the file
    memory-mapped.  Scaling is applied only when a 2-D or smaller slice is
    extracted (the common display/analysis access pattern).

    Higher-dimensional slices (ndim >= 3) are returned as another
    ``LazyScaledArray`` so that cube → sub-cube access chains remain lazy.
    """

    def __init__(self, raw, bzero: float = 0.0, bscale: float = 1.0, blank=None):
        self._raw = raw
        self._bzero = float(bzero)
        self._bscale = float(bscale)
        self._blank = blank  # integer BLANK value or None

    # ------------------------------------------------------------------
    # numpy-compatible attribute interface
    # ------------------------------------------------------------------
    @property
    def shape(self):
        return self._raw.shape

    @property
    def ndim(self):
        return self._raw.ndim

    @property
    def dtype(self):
        return np.dtype(np.float64)

    @property
    def size(self):
        return self._raw.size

    @property
    def nbytes(self):
        return self._raw.size * 8  # float64

    @property
    def flags(self):
        return self._raw.flags

    def __len__(self):
        return len(self._raw)

    def __repr__(self):
        return (
            f"LazyScaledArray(shape={self.shape}, bzero={self._bzero}, "
            f"bscale={self._bscale}, blank={self._blank})"
        )

    def __copy__(self):
        """Shallow copy — share the underlying memmap."""
        return LazyScaledArray(self._raw, self._bzero, self._bscale, self._blank)

    def __deepcopy__(self, memo):
        """Deep copy without duplicating the memory-mapped buffer.

        The raw memmap is a read-only view into the file and can be safely
        shared across copies.  Duplicating it would defeat the purpose of
        lazy scaling and cause multi-gigabyte allocations.
        """
        return LazyScaledArray(self._raw, self._bzero, self._bscale, self._blank)

    # ------------------------------------------------------------------
    # Scaling helpers
    # ------------------------------------------------------------------
    def _apply_scaling(self, raw_data):
        """Apply BZERO/BSCALE to raw data, returning a float64 array."""
        result = np.array(raw_data, dtype=np.float64)
        if self._bscale != 1.0:
            result *= self._bscale
        if self._bzero != 0.0:
            result += self._bzero
        if self._blank is not None:
            with np.errstate(invalid="ignore"):
                blank_mask = raw_data == self._blank
                if np.any(blank_mask):
                    result[blank_mask] = np.nan
        return result

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------
    def __getitem__(self, key):
        raw = self._raw[key]
        if isinstance(raw, np.ndarray):
            if raw.ndim >= 3:
                return LazyScaledArray(raw, self._bzero, self._bscale, self._blank)
            return self._apply_scaling(raw)
        # Scalar access
        if self._blank is not None and raw == self._blank:
            return np.nan
        return float(raw) * self._bscale + self._bzero

    def __array__(self, dtype=None):
        """Support ``np.asarray(obj)`` — materialises the entire array."""
        result = self._apply_scaling(self._raw)
        if dtype is not None:
            return result.astype(dtype)
        return result

    def reshape(self, *args, **kwargs):
        """Reshape the underlying raw array (view when possible)."""
        return LazyScaledArray(
            self._raw.reshape(*args, **kwargs),
            self._bzero,
            self._bscale,
            self._blank,
        )

    # ------------------------------------------------------------------
    # View-like operations on the raw memmap
    # ------------------------------------------------------------------
    def _raw_view_op(self, func, *args, **kwargs):
        """Apply a numpy view operation (flip, expand_dims, …) to the raw array."""
        return LazyScaledArray(
            func(self._raw, *args, **kwargs),
            self._bzero,
            self._bscale,
            self._blank,
        )

    def astype(self, dtype, **kwargs):
        """Return a materialised array cast to *dtype*."""
        return self.__array__().astype(dtype, **kwargs)


def is_lazy_scaled(obj) -> bool:
    """Return True if *obj* is a :class:`LazyScaledArray`."""
    return isinstance(obj, LazyScaledArray)


def estimate_array_nbytes(array: np.ndarray | np.memmap | LazyScaledArray | None) -> int | None:
    """Return the approximate number of bytes for the given array.

    The helper works for ``numpy.ndarray``, ``numpy.memmap``, and
    ``LazyScaledArray`` instances.  ``None`` is returned if the shape or
    dtype is not available.
    """
    if array is None:
        return None

    if isinstance(array, LazyScaledArray):
        return int(array._raw.size) * array._raw.dtype.itemsize

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


def _parse_header_float(header: Any, key: str, default: float) -> float | None:
    """Return a numeric header value, or ``None`` when parsing fails."""
    try:
        if key not in header:
            return float(default)
        value = header[key]
    except Exception:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def header_has_scaling_keywords(header: Any | None) -> bool:
    """Return True when FITS data requires scaling or BLANK handling.

    BLANK is only meaningful for integer BITPIX (>0).  For floating-point
    data (BITPIX < 0) the FITS standard says BLANK must be ignored, and
    astropy does exactly that, so it never prevents memory-mapping.
    """
    if header is None:
        return False

    try:
        bitpix = int(header.get("BITPIX", 0))
    except (TypeError, ValueError):
        bitpix = 0

    try:
        if "BLANK" in header and bitpix > 0:
            return True
    except Exception:
        return True

    bzero = _parse_header_float(header, "BZERO", 0.0)
    bscale = _parse_header_float(header, "BSCALE", 1.0)
    if bzero is None or bscale is None:
        return True

    return not (bzero == 0.0 and bscale == 1.0)


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


def sanitize_slice(array: np.ndarray) -> np.ndarray:
    """Replace invalid values (< -100000) and infinities with NaN in-place.

    Intended for per-slice use so that the full cube is never scanned at load
    time.  Returns *array* (possibly converted to float) for convenience.
    """
    if array is None or array.size == 0:
        return array
    with np.errstate(invalid="ignore"):
        bad = (array < -100000) | ~np.isfinite(array)
    if np.any(bad):
        if not np.issubdtype(array.dtype, np.floating):
            array = array.astype(np.float32)
        array[bad] = np.nan
    return array


def downsample_2d_for_display(
    array: np.ndarray | np.memmap | LazyScaledArray,
    *,
    max_dimension: int = DEFAULT_LARGE_DATA_DISPLAY_MAX_DIM,
) -> np.ndarray:
    """Return a 2-D display array capped to roughly ``max_dimension`` pixels per axis."""
    max_dimension = max(1, int(max_dimension))

    # For LazyScaledArray, stride the raw memmap first, then scale only the
    # small result so the full array is never materialised.
    if isinstance(array, LazyScaledArray) and array.ndim == 2:
        height, width = array.shape
        step_y = max(1, int(math.ceil(height / max_dimension)))
        step_x = max(1, int(math.ceil(width / max_dimension)))
        if step_y == 1 and step_x == 1:
            return array._apply_scaling(array._raw)
        raw_ds = array._raw[::step_y, ::step_x]
        return array._apply_scaling(raw_ds)

    arr = np.asanyarray(array)
    if arr.ndim != 2:
        return arr

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
    array: np.ndarray | np.memmap | LazyScaledArray,
    max_samples: int = _DEFAULT_MAX_SAMPLES,
) -> Tuple[float, float]:
    """Return an approximate ``(nanmin, nanmax)`` pair for ``array``.

    The function is designed for very large arrays backed by memory maps. Instead
    of materialising the full dataset, a sub-sample is inspected. The result is
    sufficient for display initialisation while avoiding multi-gigabyte scans.

    For :class:`LazyScaledArray` instances the sample is drawn from the raw
    memmap and then scaled, so the full array is never materialised.
    """
    if isinstance(array, LazyScaledArray):
        raw_flat = array._raw.reshape(-1)
        raw_sample = _select_sample(raw_flat, max_samples)
        sample = array._apply_scaling(raw_sample)
    else:
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
