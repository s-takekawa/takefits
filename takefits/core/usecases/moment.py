"""Moment map usecases."""
from __future__ import annotations

import os
import warnings
from typing import Any, List, Literal, Optional, Tuple, Union

import numpy as np

from takefits.core.app_state import AppState
from takefits.logic.data_tools import (
    _get_available_memory_bytes,
    format_nbytes,
    is_lazy_scaled,
)
from .utils import axis_pixel_to_world, axis_world_to_pixel, update_datamin_datamax_if_present


MomentType = Literal["moment0", "moment1", "moment2", "average", "peak"]


_MOMENT_HISTORY_PREFIX = "Integration executed by takefits on "
_MOMENT_HISTORY_FIELD_PREFIXES = (
    ("Source file:", "source_file"),
    ("Mode:", "mode"),
    ("Axis:", "axis"),
    ("Clipping:", "clipping"),
)


def _normalize_unit_text(unit) -> str:
    """Normalize unit text while preserving product-factor spacing."""
    text = " ".join(str(unit or "").strip().split())
    text = text.replace(" / ", "/").replace("/ ", "/").replace(" /", "/")
    text = text.replace(" * ", "*").replace("* ", "*").replace(" *", "*")
    return text


def _normalize_unit_factor(unit) -> str:
    """Normalize one multiplicative unit factor."""
    return _normalize_unit_text(unit).replace(" ", "")


def _canonical_moment_type(moment_type: str) -> str:
    key = str(moment_type or "").strip().lower()
    aliases = {
        "int": "moment0",
        "moment0": "moment0",
        "mom1": "moment1",
        "moment1": "moment1",
        "mom2": "moment2",
        "moment2": "moment2",
        "average": "average",
        "peak": "peak",
        "peak_int": "peak",
        "peak_coord": "peak_coord",
        "peak_corrd": "peak_coord",
        "median": "median",
        "median_int": "median",
        "rms": "rms",
        "sigma": "sigma",
    }
    return aliases.get(key, key)


def _format_history_scalar(value: Union[float, int, str, None]) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, int):
        return str(value)
    return str(value)


def _moment_axis_name(integration_axis: int) -> str:
    axis_name = "Unknown"
    if integration_axis == 0:
        axis_name = "Z (Depth/Spectral)"
    elif integration_axis == 1:
        axis_name = "Y (Lat/Dec)"
    elif integration_axis == 2:
        axis_name = "X (Lon/RA)"
    return axis_name


def _moment_range_label(
    source: Any,
    integration_axis: int,
    history_metadata: Optional[dict] = None,
) -> str:
    metadata = history_metadata or {}
    metadata_label = str(metadata.get("range_label") or "").strip()
    if metadata_label and metadata_label.lower() != "range":
        return metadata_label

    fits_axis = {0: 3, 1: 2, 2: 1}.get(int(integration_axis), 3)
    header = getattr(source, "header", None)
    ctype = ""
    if header is not None:
        try:
            ctype = str(header.get(f"CTYPE{fits_axis}", "") or "").strip()
        except Exception:
            ctype = ""

    if not ctype:
        wcs = getattr(source, "wcs", None)
        if wcs is not None:
            try:
                ctype = str(wcs.wcs.ctype[fits_axis - 1] or "").strip()
            except Exception:
                ctype = ""

    base = ctype.split("-")[0].strip() if ctype else ""
    return base or f"Axis {fits_axis}"


def _moment_range_history_line(
    source: Any,
    integration_axis: int,
    range_text: str,
    history_metadata: Optional[dict] = None,
) -> str:
    return f"{_moment_range_label(source, integration_axis, history_metadata)}: {range_text}"


def _parse_moment_history_field(line: str, block_meta: dict) -> bool:
    for prefix, key in _MOMENT_HISTORY_FIELD_PREFIXES:
        if line.startswith(prefix):
            block_meta[key] = line[len(prefix):].strip()
            return True

    if line.startswith("Range:"):
        block_meta["range"] = line[len("Range:"):].strip()
        block_meta["range_label"] = "Range"
        return True

    if ":" not in line or "range" in block_meta:
        return False

    label, value = line.split(":", 1)
    label = label.strip()
    value = value.strip()
    if not label or not value or " " in label:
        return False

    block_meta["range_label"] = label
    block_meta["range"] = value
    return True


def _sanitize_moment_history_entries(history_entries: Optional[list]) -> Tuple[dict, List[str]]:
    entries = [str(entry) for entry in (history_entries or []) if entry is not None]
    metadata: dict = {}
    sanitized: List[str] = []
    idx = 0

    while idx < len(entries):
        line = entries[idx]
        if not line.startswith(_MOMENT_HISTORY_PREFIX):
            sanitized.append(line)
            idx += 1
            continue

        block_meta: dict = {}
        idx += 1
        while idx < len(entries):
            field_line = entries[idx]
            if _parse_moment_history_field(field_line, block_meta):
                idx += 1
                continue
            else:
                break

        if not metadata and block_meta:
            metadata = block_meta

    return metadata, sanitized


def _format_history_range_from_world_range(
    world_range: Tuple[Union[float, str], Union[float, str]],
) -> str:
    return f"{_format_history_scalar(world_range[0])} to {_format_history_scalar(world_range[1])}"


def _format_history_range_from_pixel_range(
    state: AppState,
    pixel_range: Tuple[float, float],
    integration_axis: int,
) -> str:
    lo = float(pixel_range[0])
    hi = float(pixel_range[1])
    if lo > hi:
        lo, hi = hi, lo

    if state.wcs is not None:
        wcs_axis = {0: 2, 1: 1, 2: 0}.get(int(integration_axis), 2)
        try:
            reference_pixel = None
            if state.wcs.naxis >= 4 and getattr(state.data, "ndim", 0) == 4:
                reference_pixel = [
                    crpix - 1 for crpix in state.wcs.wcs.crpix
                ]
                reference_pixel[3] = max(
                    0,
                    min(int(state.current_s), int(state.data.shape[0]) - 1),
                )
            world_lo = axis_pixel_to_world(
                state,
                lo,
                wcs_axis,
                reference_pixel=reference_pixel,
            )
            world_hi = axis_pixel_to_world(
                state,
                hi,
                wcs_axis,
                reference_pixel=reference_pixel,
            )
            return f"{_format_history_scalar(world_lo)} to {_format_history_scalar(world_hi)}"
        except Exception:
            pass

    return f"ch {_format_history_scalar(lo)} to {_format_history_scalar(hi)}"


def _derive_history_range_text(
    state: AppState,
    integration_axis: int,
    history_metadata: dict,
    pixel_range: Optional[Tuple[float, float]] = None,
    world_range: Optional[Tuple[Union[float, str], Union[float, str]]] = None,
) -> str:
    if world_range is not None:
        return _format_history_range_from_world_range(world_range)

    if pixel_range is not None:
        return _format_history_range_from_pixel_range(state, pixel_range, integration_axis)

    history_range = str(history_metadata.get("range", "") or "").strip()
    if history_range and history_range.lower() != "none to none":
        return history_range

    if state.integ_min is not None and state.integ_max is not None:
        return (
            f"{_format_history_scalar(state.integ_min)} to "
            f"{_format_history_scalar(state.integ_max)}"
        )

    if state.integ_min_pix is not None and state.integ_max_pix is not None:
        return _format_history_range_from_pixel_range(
            state,
            (float(state.integ_min_pix), float(state.integ_max_pix)),
            integration_axis,
        )

    return "full range"


def _axis_unit_for_integration_axis(state: AppState, integration_axis: int) -> str:
    header = getattr(state, "header", None)
    spectral_meta = getattr(state, "spectral_metadata", {}) or {}

    try:
        axis = int(integration_axis)
    except Exception:
        axis = 0
    axis = max(0, min(axis, 2))

    spectral_unit = _normalize_unit_factor(spectral_meta.get("current_axis_unit", ""))
    fits_axis = {0: 3, 1: 2, 2: 1}.get(axis, 3)

    if axis == 0 and spectral_unit:
        return spectral_unit

    if header is not None:
        try:
            header_unit = _normalize_unit_factor(header.get(f"CUNIT{fits_axis}", ""))
            if header_unit:
                return header_unit
        except Exception:
            pass

    try:
        spectral_axis = int(spectral_meta.get("axis_index", 0) or 0)
    except Exception:
        spectral_axis = 0
    if spectral_axis == fits_axis and spectral_unit:
        return spectral_unit

    if header is not None:
        try:
            ctype = str(header.get(f"CTYPE{fits_axis}", "") or "").strip().upper()
        except Exception:
            ctype = ""
        # Fallback for headers without explicit CUNIT on celestial axes.
        if ctype.startswith("RA") or ctype.startswith("DEC") or ("LON" in ctype) or ("LAT" in ctype):
            return "deg"

    return ""


def _compose_product_unit(base_unit: str, axis_unit: str) -> str:
    base = _normalize_unit_text(base_unit)
    axis = _normalize_unit_factor(axis_unit)
    if not base:
        return axis
    if not axis:
        return base
    if any(token.lower() == axis.lower() for token in base.split()):
        return base
    return f"{base} {axis}"


def _moment_bunit(state: AppState, moment_type: str, integration_axis: int) -> str:
    canonical = _canonical_moment_type(moment_type)
    header = getattr(state, "header", None)
    base_unit = _normalize_unit_text(header.get("BUNIT", "")) if header is not None else ""
    axis_unit = _axis_unit_for_integration_axis(state, integration_axis)

    if canonical == "moment0":
        return _compose_product_unit(base_unit, axis_unit)
    if canonical in {"moment1", "moment2", "peak_coord"}:
        return axis_unit or "pix"
    if canonical in {"average", "peak", "median", "rms", "sigma"}:
        return base_unit
    return base_unit


def _get_integration_pixel_range(
    state: AppState,
    axis: int = 0,
    *,
    axis_len: Optional[int] = None,
) -> Tuple[int, int, float, float]:
    """
    Get pixel range for integration, handling fractional boundaries.

    Returns:
        (min_pixel, max_pixel, min_fraction, max_fraction)
        where fractions indicate partial pixel coverage at boundaries
    """
    if state.data is None:
        raise ValueError("No data loaded")

    if axis_len is None:
        axis_len = state.data.shape[axis]
    axis_len = int(axis_len)

    # Get pixel limits from state (now supports float)
    min_pixel_float = 0.0
    max_pixel_float = float(axis_len - 1)

    if state.integ_min_pix is not None:
        min_pixel_float = float(state.integ_min_pix)
    if state.integ_max_pix is not None:
        max_pixel_float = float(state.integ_max_pix)

    # Ensure min < max
    if min_pixel_float > max_pixel_float:
        min_pixel_float, max_pixel_float = max_pixel_float, min_pixel_float

    # Edge case adjustment (match integration.py logic: if max is exactly at edge - 0.5)
    if max_pixel_float == axis_len - 0.5:
        max_pixel_float -= 0.00001

    # Round to nearest integer indices
    min_pixel = int(round(min_pixel_float))
    max_pixel = int(round(max_pixel_float))

    if min_pixel > max_pixel:
        max_pixel, min_pixel = min_pixel, max_pixel

    # Calculate fractions for edge pixels
    # Logic from integration.py:
    # min_fraction = min_pixel - min_pixel_float - 0.5
    # max_fraction = max_pixel_float - max_pixel + 0.5
    min_fraction = min_pixel - min_pixel_float - 0.5
    max_fraction = max_pixel_float - max_pixel + 0.5

    # Clamp min_pixel to valid range
    if min_pixel < 0:
        min_pixel = 0
        # If min_pixel was clamped, the fractional logic might need adjustment?
        # integration.py clamps but doesn't seem to recalculate fraction based on clamped index,
        # it just uses the fraction derived from the *original* float.
        # But wait, later in integration.py it says:
        # total_sum = nan_sum(..., squeeze(first_pixel_value) * min_fraction)
        # where first_pixel_value is take(..., [min_pixel]).
        # So if min_pixel=0, we act on index 0.
        # If original float was -10.0, min_pixel=-10 -> clamped to 0.
        # min_frac = -10 - (-10.0) - 0.5 = -0.5.
        # So we add -0.5 * pixel[0].
        # Is that correct?
        # integration.py has specific logic to update input TEXT fields when clamped,
        # but the calculation variables (min_pixel) are clamped.
        # It seems the `min_fraction` remains as calculated from the potentially out-of-bounds float.
        pass

    # Clamp max_pixel
    if max_pixel >= axis_len:
        max_pixel = axis_len - 1

    if max_pixel < 0:
        max_pixel = 0

    # Ensure min doesn't exceed max after clamping
    if min_pixel > max_pixel:
        min_pixel = max_pixel

    return min_pixel, max_pixel, min_fraction, max_fraction


def _integration_indices_and_weights(
    state: AppState,
    *,
    axis: int,
    axis_len: int,
    pixel_range: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resolve a continuous pixel range into exact pixel-overlap weights.

    Pixel centres are integers and their cells span ``i - 0.5`` to
    ``i + 0.5``.  An omitted bound uses the outer edge of the data, so the
    default operation includes every pixel at full weight.  A range wholly
    outside the cube yields one zero-weight placeholder; callers consequently
    return an all-NaN map without touching an unbounded slice.
    """
    axis_len = int(axis_len)
    if axis_len <= 0:
        raise ValueError("Cannot integrate an empty axis")

    if pixel_range is None:
        lower = (
            -0.5
            if state.integ_min_pix is None
            else float(state.integ_min_pix)
        )
        upper = (
            float(axis_len) - 0.5
            if state.integ_max_pix is None
            else float(state.integ_max_pix)
        )
    else:
        if len(pixel_range) != 2:
            raise ValueError("pixel_range must contain exactly two values")
        lower = float(pixel_range[0])
        upper = float(pixel_range[1])

    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("Integration pixel range must be finite")
    if lower > upper:
        lower, upper = upper, lower

    clipped_lower = max(lower, -0.5)
    clipped_upper = min(upper, float(axis_len) - 0.5)
    if clipped_upper <= clipped_lower:
        nearest = int(np.clip(round((lower + upper) * 0.5), 0, axis_len - 1))
        return (
            np.asarray([nearest], dtype=np.int64),
            np.asarray([0.0], dtype=np.float64),
        )

    first = max(0, int(np.floor(clipped_lower + 0.5)))
    stop = min(axis_len, int(np.ceil(clipped_upper + 0.5)))
    indices = np.arange(first, stop, dtype=np.int64)
    left_edges = indices.astype(np.float64) - 0.5
    right_edges = left_edges + 1.0
    weights = np.minimum(clipped_upper, right_edges) - np.maximum(
        clipped_lower,
        left_edges,
    )
    np.maximum(weights, 0.0, out=weights)

    positive = weights > 0.0
    if np.any(positive):
        return indices[positive], weights[positive]

    nearest = int(np.clip(round((lower + upper) * 0.5), 0, axis_len - 1))
    return (
        np.asarray([nearest], dtype=np.int64),
        np.asarray([0.0], dtype=np.float64),
    )


def _compute_moment_legacy(
    state: AppState,
    moment_type: str = "moment0",
    axis: int = 0,
    clip_threshold: Optional[float] = None,
    pixel_range: Optional[Tuple[float, float]] = None,
    world_range: Optional[Tuple[Union[float, str], Union[float, str]]] = None
) -> np.ndarray:
    """
    Compute moment map or other integration from the data cube.

    Args:
        state: AppState containing data and parameters
        moment_type: Type of moment ("moment0", "moment1", "moment2", "average", "peak", "peak_coord", "median", "rms")
        axis: Axis along which to integrate (0=z/velocity, 1=y, 2=x for numpy array order)
        clip_threshold: If provided, values less than this are set to NaN before processing.
        pixel_range: Optional (min_pix, max_pix) range along the axis. Takes precedence over state values.
        world_range: Optional (min_world, max_world) range in world coordinates for the axis.
                    Units depend on WCS CTYPE (e.g., km/s for velocity, deg for RA/Dec).
                    Requires WCS to be available. pixel_range takes precedence if both specified.

    Returns:
        2D numpy array containing the moment map

    Note:
        The axis parameter uses numpy array indexing convention:
        - For a 3D cube with shape (nz, ny, nx):
          - axis=0 integrates along Z (velocity/frequency axis)
          - axis=1 integrates along Y (Dec/GLAT axis)
          - axis=2 integrates along X (RA/GLON axis)

        The WCS axis order is typically reversed from numpy (X, Y, Z).
        world_range uses WCS axis numbering: WCS axis 0=X, 1=Y, 2=Z.
        Internally, this is mapped to the correct numpy axis.
    """
    if state.data is None:
        raise ValueError("No data loaded")

    data = state.data
    wcs = state.wcs

    # Handle 4D data by selecting current S slice
    if data.ndim == 4:
        data = data[state.current_s]

    # Apply clipping if requested
    if clip_threshold is not None:
        # We need a copy to avoid modifying original data in state
        data = data.copy()
        data[data < clip_threshold] = np.nan

    if data.ndim != 3:
        raise ValueError(f"Expected 3D data cube, got {data.ndim}D")

    # Determine the pixel range to use
    # Priority: pixel_range > world_range > state values
    use_pixel_range = None

    if pixel_range is not None:
        use_pixel_range = pixel_range
    elif world_range is not None:
        if wcs is None:
            raise ValueError("WCS is required for world_range conversion")
        # Convert world range to pixel range
        # WCS axis index: for 3D cube, numpy axis 0 (z) = WCS axis 2
        # numpy axis order is (z, y, x), WCS axis order is (x, y, z)
        wcs_axis = data.ndim - 1 - axis  # Convert numpy axis to WCS axis
        min_world, max_world = world_range
        min_pix_w = axis_world_to_pixel(state, min_world, wcs_axis)
        max_pix_w = axis_world_to_pixel(state, max_world, wcs_axis)
        # Handle case where world coordinates are inverted
        if min_pix_w > max_pix_w:
            min_pix_w, max_pix_w = max_pix_w, min_pix_w
        use_pixel_range = (min_pix_w, max_pix_w)

    # Temporarily update state if pixel_range provided directly
    original_min_pix = state.integ_min_pix
    original_max_pix = state.integ_max_pix

    if use_pixel_range is not None:
        state.integ_min_pix = use_pixel_range[0]
        state.integ_max_pix = use_pixel_range[1]

    try:
        # Calculate fractions for accurate integration
        min_pix, max_pix, min_frac, max_frac = _get_integration_pixel_range(state, axis)
    finally:
        # Restore original state values
        state.integ_min_pix = original_min_pix
        state.integ_max_pix = original_max_pix

    if moment_type == "moment0":
        # Integrated intensity (M0) with partial pixel handling

        # 1. Base sum: pixels in range [min_pix, max_pix) (exclusive of max_pix)
        # Note: range(min, max) behaves as slice(min, max)
        slices_base = [slice(None)] * data.ndim
        # Ensure min_pix <= max_pix for valid slice
        slice_end = max(min_pix, max_pix)
        slices_base[axis] = slice(min_pix, slice_end)

        base_data = data[tuple(slices_base)]
        
        # Track positions where ALL values along axis are NaN
        all_nan_mask = np.all(np.isnan(base_data), axis=axis)
        
        result = np.nansum(base_data, axis=axis)

        # 2. Add contribution from min_pix (Weighted)
        # Effective weight: 1.0 + min_frac (since it's included in base sum)
        # Wait, if logic is "included in base sum", we just add `min_frac * val`.
        # Because base sum has `1.0 * val`. Total = `(1+min_frac) * val`.
        if min_pix < data.shape[axis]:
            slice_min = [slice(None)] * data.ndim
            slice_min[axis] = slice(min_pix, min_pix + 1)
            val_min = data[tuple(slice_min)]
            # Squeeze carefully (remove the dimension we sliced)
            # But result has that dimension removed.
            val_min = np.squeeze(val_min, axis=axis)
            result += np.nan_to_num(val_min) * min_frac

        # 3. Add contribution from max_pix (Weighted)
        # Effective weight: max_frac (since it's NOT in base sum)
        if max_pix < data.shape[axis] and max_pix >= 0:
            slice_max = [slice(None)] * data.ndim
            slice_max[axis] = slice(max_pix, max_pix + 1)
            val_max = data[tuple(slice_max)]
            val_max = np.squeeze(val_max, axis=axis)
            result += np.nan_to_num(val_max) * max_frac

        # Apply CDELT scaling if WCS available
        if wcs is not None:
            try:
                cdelt = abs(wcs.wcs.cdelt[2 - axis])
                result = result * cdelt
            except (AttributeError, IndexError):
                pass
        
        # Restore NaN for positions where all values were NaN
        result[all_nan_mask] = np.nan
        
        return result


    elif moment_type == "average":
        # Mean intensity (Weighted Sum / Width)

        # Calculate weighted sum (same as Moment 0 but without CDELT)
        slices_base = [slice(None)] * data.ndim
        slice_end = max(min_pix, max_pix)
        slices_base[axis] = slice(min_pix, slice_end)

        base_data = data[tuple(slices_base)]
        
        # Track positions where ALL values along axis are NaN
        all_nan_mask = np.all(np.isnan(base_data), axis=axis)
        
        total_sum = np.nansum(base_data, axis=axis)

        if min_pix < data.shape[axis]:
            slice_min = [slice(None)] * data.ndim
            slice_min[axis] = slice(min_pix, min_pix + 1)
            val_min = data[tuple(slice_min)]
            val_min = np.squeeze(val_min, axis=axis)
            total_sum += np.nan_to_num(val_min) * min_frac

        if max_pix < data.shape[axis] and max_pix >= 0:
            slice_max = [slice(None)] * data.ndim
            slice_max[axis] = slice(max_pix, max_pix + 1)
            val_max = data[tuple(slice_max)]
            val_max = np.squeeze(val_max, axis=axis)
            total_sum += np.nan_to_num(val_max) * max_frac

        # Calculate width in pixels
        # width = (max_pix - min_pix) + max_frac - min_frac?
        # Check sign of min_frac.
        # min_frac = min - min_f - 0.5. (e.g. -0.5 to 0.5).
        # max_frac = max_f - max + 0.5. (e.g. -0.5 to 0.5).
        # Correct logic as derived: arithmetic sum adds up to float difference.
        # But wait, min_frac is usually negative (we add a negative amount of the pixel value).
        # So effective width of min_pixel is (1+min_frac).
        # Effective width of max_pixel is max_frac.
        # Width of base (min to max-1) is (max - min).
        # Total width = (max - min) + min_frac + max_frac.
        width = (max_pix - min_pix) + min_frac + max_frac

        if width <= 0:
            width = 1.0  # Prevent division by zero

        result = total_sum / width
        
        # Restore NaN for positions where all values were NaN
        result[all_nan_mask] = np.nan
        
        return result


    elif moment_type == "peak":
        # Peak intensity
        slices = [slice(None)] * data.ndim
        slices[axis] = slice(min_pix, max_pix + 1)
        sliced_data = data[tuple(slices)]
        return np.nanmax(sliced_data, axis=axis)

    elif moment_type in ("moment1", "moment2", "rms", "sigma"):
        # Get world coordinates for velocity axis
        indices = np.arange(min_pix, max_pix + 1)
        n_pix = len(indices)

        # Define sliced_data (inclusive range)
        slices = [slice(None)] * data.ndim
        slices[axis] = slice(min_pix, max_pix + 1)
        sliced_data = data[tuple(slices)]

        # --- create fractional weights ---
        # Default weights = 1.0
        weights_1d = np.ones(n_pix, dtype=float)

        # Apply boundary fractions
        if n_pix > 0:
            weights_1d[0] += min_frac
            weights_1d[-1] += (max_frac - 1.0)

        # Handle case where weights become non-positive due to floating precision or empty range
        weights_1d[weights_1d < 0] = 0.0

        # Reshape weights for broadcasting against data
        w_shape = [1] * data.ndim
        w_shape[axis] = n_pix
        weights = weights_1d.reshape(w_shape)

        if moment_type == "rms":
            # RMS (Root Mean Square)
            # Weighted RMS: sqrt( sum(w * x^2) / sum(w) )
            squared = np.square(sliced_data)

            # Weighted sum of squares
            valid_mask = ~np.isnan(sliced_data)
            w_expanded = weights * valid_mask  # Broadcast

            weighted_sq_sum = np.nansum(squared * weights, axis=axis)
            sum_weights = np.nansum(w_expanded, axis=axis)

            with np.errstate(divide='ignore', invalid='ignore'):
                mean_squared = weighted_sq_sum / sum_weights

            return np.sqrt(mean_squared)

        elif moment_type == "sigma":
            # Sigma (Weighted Standard Deviation)
            # sqrt( sum(w * (x - mean)^2) / sum(w) )
            # Or: sqrt( sum(w*x^2)/sum(w) - (sum(w*x)/sum(w))^2 )
            # Using the second form is faster (one pass if we reuse sums)

            valid_mask = ~np.isnan(sliced_data)
            w_expanded = weights * valid_mask
            sum_weights = np.nansum(w_expanded, axis=axis)

            # Weighted sum (mean calculation)
            weighted_sum = np.nansum(sliced_data * weights, axis=axis)

            # Weighted sum of squares
            squared = np.square(sliced_data)
            weighted_sq_sum = np.nansum(squared * weights, axis=axis)

            with np.errstate(divide='ignore', invalid='ignore'):
                weighted_mean = weighted_sum / sum_weights
                mean_sq = weighted_sq_sum / sum_weights
                variance = mean_sq - weighted_mean**2

            # Variance can be slightly negative due to precision, clip to 0
            variance[variance < 0] = 0

            return np.sqrt(variance)

        # For Moment 1 & 2, we also need World Coordinates
        if wcs is None:
            # Use pixel coordinates if no WCS
            world_coords_axis = indices.astype(float)
        else:
            # Map data axis to WCS axis (FITS convention)
            data_to_wcs_axis = {0: 2, 1: 1, 2: 0}
            wcs_axis = data_to_wcs_axis.get(axis, axis)

            # Build pixel coordinate array for WCS transformation
            num_wcs_axes = wcs.naxis
            pixel_coords = np.zeros((n_pix, num_wcs_axes))
            pixel_coords[:, wcs_axis] = indices

            # Use CRPIX for other axes
            for i in range(num_wcs_axes):
                if i != wcs_axis:
                    pixel_coords[:, i] = wcs.wcs.crpix[i] - 1

            # Convert to world coordinates
            world_coords = wcs.wcs_pix2world(pixel_coords, 0)
            world_coords_axis = world_coords[:, wcs_axis]

        # Reshape for broadcasting
        shape = [1] * data.ndim
        shape[axis] = -1
        world_coords_axis = world_coords_axis.reshape(shape)

        # Calculate weighted intensity sum (Moment 0 equivalent but with WCS-axis-weights)
        # Note: Moment 1 definition is sum(I * v) / sum(I).
        # We apply fractional weights to I.
        # I_eff = I * weight

        weighted_data = sliced_data * weights
        intensity_sum = np.nansum(weighted_data, axis=axis)

        if moment_type == "moment1":
            # Intensity-weighted mean velocity
            # sum(I_eff * v) / sum(I_eff)
            val_weighted_sum = np.nansum(weighted_data * world_coords_axis, axis=axis)

            with np.errstate(divide='ignore', invalid='ignore'):
                result = val_weighted_sum / intensity_sum

            # Clip to valid range
            world_min = np.nanmin(world_coords_axis)
            world_max = np.nanmax(world_coords_axis)
            if world_min > world_max:
                world_min, world_max = world_max, world_min
            result[(result < world_min) | (result > world_max)] = np.nan
            return result

        else:  # moment2
            # First calculate M1
            val_weighted_sum = np.nansum(weighted_data * world_coords_axis, axis=axis)
            with np.errstate(divide='ignore', invalid='ignore'):
                m1 = val_weighted_sum / intensity_sum

            # Velocity dispersion (sqrt of variance)
            # sum(I_eff * v^2) / sum(I_eff) - M1^2
            val_sq_weighted_sum = np.nansum(weighted_data * world_coords_axis**2, axis=axis)

            with np.errstate(divide='ignore', invalid='ignore'):
                result = np.sqrt((val_sq_weighted_sum / intensity_sum) - m1**2)

            # Clip invalid values
            result[result < 0] = np.nan
            return result

    elif moment_type == "median":
        # Median intensity
        slices = [slice(None)] * data.ndim
        slices[axis] = slice(min_pix, max_pix + 1)
        sliced_data = data[tuple(slices)]
        return np.nanmedian(sliced_data, axis=axis)

    elif moment_type == "rms":
        # RMS (Root Mean Square)
        slices = [slice(None)] * data.ndim
        slices[axis] = slice(min_pix, max_pix + 1)
        sliced_data = data[tuple(slices)]

        squared = np.square(sliced_data)
        mean_squared = np.nanmean(squared, axis=axis)
        return np.sqrt(mean_squared)

    elif moment_type == "peak_coord":
        # Peak Coordinate (World Coordinate of the peak pixel)
        slices = [slice(None)] * data.ndim
        slices[axis] = slice(min_pix, max_pix + 1)
        sliced_data = data[tuple(slices)]

        # Determine shape of sliced_data: e.g. (Z, Y, X) for axis=0
        # Determine axis index within sliced_data (it's the same numerical index)

        # To avoid "All-NaN slice encountered" warning/error, we mask
        all_nan_mask = np.all(np.isnan(sliced_data), axis=axis)

        # Fill NaNs with -inf temporarily to find argmax safely
        temp_data = sliced_data.copy()
        temp_data[np.isnan(temp_data)] = -np.inf

        # Get peak indices relative to the sliced block
        local_peak_indices = np.nanargmax(temp_data, axis=axis)

        # Convert local indices to global indices (add min_pix)
        global_peak_indices = local_peak_indices + min_pix

        # Reshape for broadcasting grid logic
        # We need to construct full coordinates for each resulting pixel to map to WCS

        # Output shape
        res_shape = list(data.shape)
        del res_shape[axis]

        n_points = np.prod(res_shape)

        # Dimensions excluding the integration axis
        dims = list(range(data.ndim))
        dims.remove(axis)

        # Grid indices for the map
        grid_indices = np.indices(res_shape)

        # Prepare pixel_coords array: (N_points, N_dim)
        pixel_coords = np.zeros((n_points, data.ndim))

        # Fill in the map coordinates
        for i, dim_idx in enumerate(dims):
            pixel_coords[:, dim_idx] = grid_indices[i].flatten()

        # Fill in the peak index coordinate
        pixel_coords[:, axis] = global_peak_indices.flatten()

        if wcs is not None:
            # pixel_coords is in numpy axis order (Z, Y, X), but wcs expects FITS order (X, Y, Z)
            pixel_coords_fits = pixel_coords[:, ::-1]
            world_coords = wcs.wcs_pix2world(pixel_coords_fits, 0)
            # Extract the relevant axis column - in FITS order, axis 0→2, 1→1, 2→0
            data_to_wcs_axis = {0: 2, 1: 1, 2: 0}  # numpy axis to FITS axis index
            target_wcs_axis = data_to_wcs_axis.get(axis, axis)

            result_flat = world_coords[:, target_wcs_axis]

        else:
            # Just return pixel index
            result_flat = global_peak_indices.flatten().astype(float)

        result_map = result_flat.reshape(res_shape)

        # Restore NaNs where data was all NaN
        result_map[all_nan_mask] = np.nan

        return result_map

    raise ValueError(f"Unknown moment type: {moment_type}")


# Keep each materialized 3-D tile comfortably below the size at which NumPy's
# reduction temporaries become disruptive.  Tests may monkeypatch this value to
# exercise the tiled path with small arrays.
_MOMENT_TILE_TARGET_BYTES = 32 * 1024 * 1024
# Peak-coordinate WCS conversion creates several coordinate arrays per output
# pixel. Reserve this overhead even for short integration ranges so a
# one-channel moment cannot turn into a hundreds-of-MiB coordinate tile.
_MOMENT_TILE_OUTPUT_OVERHEAD_BYTES = 96
_MOMENT_OUTPUT_FALLBACK_LIMIT_BYTES = 512 * 1024 * 1024
_MOMENT_OUTPUT_RAM_FRACTION = 0.80


def _moment_tile_target_bytes() -> int:
    """Return the target byte size for one materialized moment tile."""
    value = os.environ.get("TAKEFITS_MOMENT_TILE_MB")
    if value is not None:
        try:
            size_mb = int(value)
        except (TypeError, ValueError):
            size_mb = 0
        if size_mb > 0:
            return size_mb * 1024 * 1024
    return _MOMENT_TILE_TARGET_BYTES


def _ensure_moment_output_memory_budget(output_shape: Tuple[int, int]) -> None:
    """Preflight the result and GUI image copies before allocating them."""
    output_pixels = int(output_shape[0]) * int(output_shape[1])
    output_bytes = output_pixels * np.dtype(np.float64).itemsize
    # The returned map, Matplotlib image copy, and previous displayed image can
    # coexist during a GUI update. Include one tile for compute scratch.
    working_bytes = 3 * output_bytes + _moment_tile_target_bytes()
    available = _get_available_memory_bytes()
    if available is None:
        limit = _MOMENT_OUTPUT_FALLBACK_LIMIT_BYTES
    else:
        limit = max(
            1 * 1024 * 1024,
            int(available * _MOMENT_OUTPUT_RAM_FRACTION),
        )
    if working_bytes <= limit:
        return
    raise MemoryError(
        "Moment map cannot safely allocate the requested output "
        f"({int(output_shape[0]):,} x {int(output_shape[1]):,}, "
        f"{format_nbytes(output_bytes)}; estimated working set "
        f"{format_nbytes(working_bytes)}). The current available-memory safety "
        f"limit is {format_nbytes(limit)}. Cut out a smaller spatial region first."
    )


def _iter_moment_tiles(
    data_shape: Tuple[int, int, int],
    axis: int,
    min_pix: int,
    max_pix: int,
):
    """Yield bounded input/output slices for a 3-D moment reduction.

    Tiling both non-integrated dimensions is important for cubes whose
    integration axis is X or Y: a row-only strategy can still leave a single
    tile hundreds of MiB wide.
    """
    output_axes = [dim for dim in range(3) if dim != axis]
    output_shape = (int(data_shape[output_axes[0]]), int(data_shape[output_axes[1]]))
    integration_len = max(1, int(max_pix) - int(min_pix) + 1)
    bytes_per_output_pixel = (
        integration_len * np.dtype(np.float64).itemsize
        + _MOMENT_TILE_OUTPUT_OVERHEAD_BYTES
    )
    max_pixels = max(1, _moment_tile_target_bytes() // max(1, bytes_per_output_pixel))

    if output_shape[1] <= max_pixels:
        col_step = output_shape[1]
        row_step = max(1, min(output_shape[0], max_pixels // max(1, col_step)))
    else:
        row_step = 1
        col_step = max_pixels

    for row_start in range(0, output_shape[0], row_step):
        row_stop = min(output_shape[0], row_start + row_step)
        for col_start in range(0, output_shape[1], col_step):
            col_stop = min(output_shape[1], col_start + col_step)
            input_slices = [slice(None)] * 3
            input_slices[axis] = slice(min_pix, max_pix + 1)
            input_slices[output_axes[0]] = slice(row_start, row_stop)
            input_slices[output_axes[1]] = slice(col_start, col_stop)
            output_slices = (
                slice(row_start, row_stop),
                slice(col_start, col_stop),
            )
            yield tuple(input_slices), output_slices, output_axes


def _materialize_moment_tile(
    data,
    input_slices,
    *,
    clip_threshold: Optional[float],
) -> np.ndarray:
    """Read, scale, sanitize, and optionally clip one bounded 3-D tile."""
    source = data[input_slices]
    if is_lazy_scaled(source):
        # LazyScaledArray already owns the newly scaled float64 buffer.
        values = np.asarray(source, dtype=np.float64)
    else:
        # FITS memmaps are read-only/copy-on-write and must never be modified by
        # clipping or NaN normalization.
        values = np.array(source, dtype=np.float64, copy=True)

    with np.errstate(invalid="ignore"):
        bad = ~np.isfinite(values)
        bad |= values < -100000
        if clip_threshold is not None:
            bad |= values < float(clip_threshold)
    if np.any(bad):
        values[bad] = np.nan
    return values


def _weighted_tile_sum(values: np.ndarray, coefficients: np.ndarray, axis: int) -> np.ndarray:
    """Reduce a NaN-free tile with 1-D coefficients without a 3-D product."""
    moved = np.moveaxis(values, axis, 0)
    return np.einsum(
        "i,i...->...",
        np.asarray(coefficients, dtype=np.float64),
        moved,
        dtype=np.float64,
        casting="unsafe",
        optimize=False,
    )


def _weighted_valid_sum(valid: np.ndarray, weights: np.ndarray, axis: int) -> np.ndarray:
    """Return the fractional count of valid samples for every output pixel."""
    moved = np.moveaxis(valid, axis, 0)
    return np.einsum(
        "i,i...->...",
        np.asarray(weights, dtype=np.float64),
        moved,
        dtype=np.float64,
        casting="unsafe",
        optimize=False,
    )


def _any_positive_weight_valid(valid: np.ndarray, weights: np.ndarray, axis: int) -> np.ndarray:
    """Return whether any positive-weight sample is valid per output pixel."""
    moved = np.moveaxis(valid, axis, 0)
    result = np.zeros(moved.shape[1:], dtype=bool)
    for index, weight in enumerate(weights):
        if weight > 0.0:
            result |= moved[index]
    return result


def _moment_world_coordinates(
    state: AppState,
    axis: int,
    indices: np.ndarray,
) -> np.ndarray:
    """Return integration-axis world coordinates using the legacy WCS anchor."""
    wcs = state.wcs
    if wcs is None:
        return indices.astype(np.float64)

    data_to_wcs_axis = {0: 2, 1: 1, 2: 0}
    wcs_axis = data_to_wcs_axis.get(axis, axis)
    num_wcs_axes = int(wcs.naxis)
    pixel_coords = np.zeros((indices.size, num_wcs_axes), dtype=np.float64)
    pixel_coords[:, wcs_axis] = indices
    for idx in range(num_wcs_axes):
        if idx != wcs_axis:
            pixel_coords[:, idx] = wcs.wcs.crpix[idx] - 1
    if (
        num_wcs_axes >= 4
        and getattr(state.data, "ndim", 0) == 4
    ):
        pixel_coords[:, 3] = max(
            0,
            min(int(state.current_s), int(state.data.shape[0]) - 1),
        )
    world_coords = wcs.wcs_pix2world(pixel_coords, 0)
    return np.asarray(world_coords[:, wcs_axis], dtype=np.float64)


def _peak_coordinate_tile(
    state: AppState,
    *,
    axis: int,
    min_pix: int,
    local_peak_indices: np.ndarray,
    output_slices,
    output_axes,
    all_nan_mask: np.ndarray,
) -> np.ndarray:
    """Convert a bounded tile of peak indices to pixel/world coordinates."""
    global_peak_indices = np.asarray(local_peak_indices, dtype=np.int64) + int(min_pix)
    tile_shape = global_peak_indices.shape
    grid_indices = np.indices(tile_shape, dtype=np.float64)
    pixel_coords_numpy = np.zeros((global_peak_indices.size, 3), dtype=np.float64)

    for grid_axis, data_axis in enumerate(output_axes):
        start = int(output_slices[grid_axis].start or 0)
        pixel_coords_numpy[:, data_axis] = grid_indices[grid_axis].reshape(-1) + start
    pixel_coords_numpy[:, axis] = global_peak_indices.reshape(-1)

    wcs = state.wcs
    if wcs is None:
        result = global_peak_indices.astype(np.float64)
    else:
        num_wcs_axes = int(wcs.naxis)
        pixel_coords_fits = np.zeros(
            (pixel_coords_numpy.shape[0], num_wcs_axes),
            dtype=np.float64,
        )
        for wcs_axis in range(num_wcs_axes):
            pixel_coords_fits[:, wcs_axis] = wcs.wcs.crpix[wcs_axis] - 1
        if (
            num_wcs_axes >= 4
            and getattr(state.data, "ndim", 0) == 4
        ):
            pixel_coords_fits[:, 3] = max(
                0,
                min(int(state.current_s), int(state.data.shape[0]) - 1),
            )
        for data_axis in range(3):
            wcs_axis = 2 - data_axis
            if wcs_axis < num_wcs_axes:
                pixel_coords_fits[:, wcs_axis] = pixel_coords_numpy[:, data_axis]
        world_coords = wcs.wcs_pix2world(pixel_coords_fits, 0)
        target_wcs_axis = {0: 2, 1: 1, 2: 0}.get(axis, axis)
        result = np.asarray(
            world_coords[:, target_wcs_axis],
            dtype=np.float64,
        ).reshape(tile_shape)

    result = np.asarray(result, dtype=np.float64)
    result[all_nan_mask] = np.nan
    return result


def compute_moment(
    state: AppState,
    moment_type: str = "moment0",
    axis: int = 0,
    clip_threshold: Optional[float] = None,
    pixel_range: Optional[Tuple[float, float]] = None,
    world_range: Optional[Tuple[Union[float, str], Union[float, str]]] = None,
) -> np.ndarray:
    """Compute a moment map with a bounded-memory tiled reduction.

    Only a small 3-D tile is materialized at once, so FITS memmaps and
    ``LazyScaledArray`` remain useful even when the source cube is many GiB.
    Accumulation is float64 for stable results; the unavoidable resident output
    is only the final 2-D map.
    """
    if state.data is None:
        raise ValueError("No data loaded")

    data = state.data
    if data.ndim == 4:
        current_s = max(0, min(int(state.current_s), data.shape[0] - 1))
        data = data[current_s]
    if data.ndim != 3:
        raise ValueError(f"Expected 3D data cube, got {data.ndim}D")

    try:
        axis = int(axis)
    except (TypeError, ValueError) as exc:
        raise ValueError("axis must be 0, 1, or 2") from exc
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")

    canonical = _canonical_moment_type(moment_type)
    supported = {
        "moment0",
        "moment1",
        "moment2",
        "average",
        "peak",
        "peak_coord",
        "median",
        "rms",
        "sigma",
    }
    if canonical not in supported:
        raise ValueError(f"Unknown moment type: {moment_type}")
    if clip_threshold is not None and not np.isfinite(float(clip_threshold)):
        raise ValueError("clip_threshold must be finite")

    use_pixel_range = pixel_range
    if use_pixel_range is None and world_range is not None:
        if state.wcs is None:
            raise ValueError("WCS is required for world_range conversion")
        wcs_axis = data.ndim - 1 - axis
        min_world, max_world = world_range
        reference_pixel = None
        if state.wcs.naxis >= 4 and getattr(state.data, "ndim", 0) == 4:
            reference_pixel = [
                crpix - 1 for crpix in state.wcs.wcs.crpix
            ]
            reference_pixel[3] = max(
                0,
                min(int(state.current_s), int(state.data.shape[0]) - 1),
            )
        min_pix_w = axis_world_to_pixel(
            state,
            min_world,
            wcs_axis,
            reference_pixel=reference_pixel,
        )
        max_pix_w = axis_world_to_pixel(
            state,
            max_world,
            wcs_axis,
            reference_pixel=reference_pixel,
        )
        if min_pix_w > max_pix_w:
            min_pix_w, max_pix_w = max_pix_w, min_pix_w
        use_pixel_range = (min_pix_w, max_pix_w)

    indices, weights = _integration_indices_and_weights(
        state,
        axis=axis,
        axis_len=data.shape[axis],
        pixel_range=use_pixel_range,
    )
    min_pix = int(indices[0])
    max_pix = int(indices[-1])

    output_shape = tuple(data.shape[dim] for dim in range(3) if dim != axis)
    _ensure_moment_output_memory_budget(output_shape)
    result = np.empty(output_shape, dtype=np.float64)
    world_coords = None
    if canonical in {"moment1", "moment2"}:
        world_coords = _moment_world_coordinates(state, axis, indices)

    for input_slices, output_slices, output_axes in _iter_moment_tiles(
        tuple(int(size) for size in data.shape),
        axis,
        min_pix,
        max_pix,
    ):
        values = _materialize_moment_tile(
            data,
            input_slices,
            clip_threshold=clip_threshold,
        )
        valid = ~np.isnan(values)
        positive_weight_shape = [1, 1, 1]
        positive_weight_shape[axis] = weights.size
        valid &= (weights > 0.0).reshape(positive_weight_shape)

        if canonical in {"peak", "peak_coord", "median"}:
            any_valid = np.any(valid, axis=axis)
        else:
            any_valid = _any_positive_weight_valid(valid, weights, axis)

        if canonical == "peak":
            values[~valid] = -np.inf
            tile_result = np.max(values, axis=axis)
            tile_result[~any_valid] = np.nan

        elif canonical == "median":
            values[~valid] = np.nan
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                tile_result = np.nanmedian(values, axis=axis)
            tile_result[~any_valid] = np.nan

        elif canonical == "peak_coord":
            values[~valid] = -np.inf
            local_peak_indices = np.argmax(values, axis=axis)
            tile_result = _peak_coordinate_tile(
                state,
                axis=axis,
                min_pix=min_pix,
                local_peak_indices=local_peak_indices,
                output_slices=output_slices,
                output_axes=output_axes,
                all_nan_mask=~any_valid,
            )

        else:
            values[~valid] = 0.0
            if canonical == "moment0":
                tile_result = _weighted_tile_sum(values, weights, axis)
                tile_result[~any_valid] = np.nan

            elif canonical == "average":
                tile_result = _weighted_tile_sum(values, weights, axis)
                valid_weight = _weighted_valid_sum(valid, weights, axis)
                with np.errstate(divide="ignore", invalid="ignore"):
                    tile_result /= valid_weight
                tile_result[~any_valid] = np.nan

            elif canonical == "moment1":
                intensity_sum = _weighted_tile_sum(values, weights, axis)
                reference_coord = float(world_coords[0])
                coordinate_offsets = world_coords - reference_coord
                numerator = _weighted_tile_sum(
                    values,
                    weights * coordinate_offsets,
                    axis,
                )
                with np.errstate(divide="ignore", invalid="ignore"):
                    tile_result = reference_coord + numerator / intensity_sum
                world_min = float(np.nanmin(world_coords))
                world_max = float(np.nanmax(world_coords))
                tile_result[
                    (tile_result < world_min) | (tile_result > world_max)
                ] = np.nan

            elif canonical == "moment2":
                intensity_sum = _weighted_tile_sum(values, weights, axis)
                reference_coord = float(world_coords[0])
                coordinate_offsets = world_coords - reference_coord
                numerator = _weighted_tile_sum(
                    values,
                    weights * coordinate_offsets,
                    axis,
                )
                with np.errstate(divide="ignore", invalid="ignore"):
                    mean_offset = numerator / intensity_sum

                moved_values = np.moveaxis(values, axis, 0)
                variance_numerator = np.zeros_like(mean_offset, dtype=np.float64)
                scratch = np.empty_like(mean_offset, dtype=np.float64)
                for index, weight in enumerate(weights):
                    if weight <= 0.0:
                        continue
                    np.subtract(
                        coordinate_offsets[index],
                        mean_offset,
                        out=scratch,
                    )
                    np.square(scratch, out=scratch)
                    scratch *= moved_values[index]
                    variance_numerator += weight * scratch
                with np.errstate(divide="ignore", invalid="ignore"):
                    variance = variance_numerator / intensity_sum
                tiny_negative = (variance < 0.0) & (
                    variance
                    >= -np.finfo(np.float64).eps
                    * np.maximum(1.0, np.square(mean_offset))
                    * 8.0
                )
                variance[tiny_negative] = 0.0
                variance[variance < 0.0] = np.nan
                tile_result = np.sqrt(variance)

            elif canonical == "rms":
                sum_weights = _weighted_valid_sum(valid, weights, axis)
                np.square(values, out=values)
                squared_sum = _weighted_tile_sum(values, weights, axis)
                with np.errstate(divide="ignore", invalid="ignore"):
                    tile_result = np.sqrt(squared_sum / sum_weights)

            else:  # sigma
                sum_weights = _weighted_valid_sum(valid, weights, axis)
                weighted_sum = _weighted_tile_sum(values, weights, axis)
                with np.errstate(divide="ignore", invalid="ignore"):
                    weighted_mean = weighted_sum / sum_weights

                moved_values = np.moveaxis(values, axis, 0)
                moved_valid = np.moveaxis(valid, axis, 0)
                variance_sum = np.zeros_like(weighted_mean, dtype=np.float64)
                for index, weight in enumerate(weights):
                    if weight <= 0.0:
                        continue
                    moved_values[index] -= weighted_mean
                    moved_values[index][~moved_valid[index]] = 0.0
                    np.square(moved_values[index], out=moved_values[index])
                    variance_sum += weight * moved_values[index]
                with np.errstate(divide="ignore", invalid="ignore"):
                    variance = variance_sum / sum_weights
                tile_result = np.sqrt(variance)

        result[output_slices] = tile_result

    if canonical == "moment0" and state.wcs is not None:
        try:
            result *= abs(state.wcs.wcs.cdelt[2 - axis])
        except (AttributeError, IndexError):
            pass
    return result

def export_moment_map_fits(
    state: AppState,
    output_path: str,
    moment_type: str = "moment0",
    axis: int = 0,
    pixel_range: Optional[Tuple[float, float]] = None,
    world_range: Optional[Tuple[Union[float, str], Union[float, str]]] = None,
    history_entries: Optional[list] = None,
    display_fits_axes: Optional[Tuple[int, int]] = None,
) -> str:
    """
    Compute a moment map and export it as a FITS file (for CLI actions).

    Args:
        state: AppState containing data and parameters
        output_path: Path for output FITS file
        moment_type: Type of moment map (e.g., "moment0")
        axis: Axis to integrate along (0=z, 1=y, 2=x)
        pixel_range: Integration range in pixels
        world_range: Integration range in world coords
        history_entries: Optional list of HISTORY entries
        display_fits_axes: Original FITS axes to keep as (axis1, axis2)
    """
    moment_data = compute_moment(
        state=state,
        moment_type=moment_type,
        axis=axis,
        pixel_range=pixel_range,
        world_range=world_range,
    )
    return export_moment_fits(
        state=state,
        moment_data=moment_data,
        output_path=output_path,
        moment_type=moment_type,
        history_entries=history_entries,
        display_fits_axes=display_fits_axes,
        integration_axis=axis,
        pixel_range=pixel_range,
        world_range=world_range,
    )


def export_moment_fits(
    state: AppState,
    moment_data: np.ndarray,
    output_path: str,
    moment_type: str = "moment0",
    history_entries: Optional[list] = None,
    display_fits_axes: Optional[Tuple[int, int]] = None,
    integration_axis: int = 0,
    pixel_range: Optional[Tuple[float, float]] = None,
    world_range: Optional[Tuple[Union[float, str], Union[float, str]]] = None,
) -> str:
    """
    Export a moment map to a FITS file.

    Args:
        state: AppState with original header/WCS info
        moment_data: 2D moment map array
        output_path: Path for output FITS file
        moment_type: Type of moment map (for BUNIT)
        history_entries: Optional list of HISTORY entries
        display_fits_axes: Original FITS axes to keep as (axis1, axis2) in output.
            Examples: (1,2)=XY, (1,3)=XZ, (3,2)=ZY.
        integration_axis: Numpy axis integrated in the source cube
            (0=z/spectral, 1=y, 2=x). Used for BUNIT inference.
        pixel_range: Optional integration range in pixels for HISTORY generation.
        world_range: Optional integration range in world coordinates for HISTORY generation.

    Returns:
        The output file path
    """
    from astropy.io import fits

    source_header = state.header.copy() if state.header is not None else fits.Header()
    kept_axes = display_fits_axes if display_fits_axes is not None else (1, 2)
    try:
        kept_axes = (int(kept_axes[0]), int(kept_axes[1]))
    except Exception:
        kept_axes = (1, 2)
    if kept_axes[0] == kept_axes[1]:
        kept_axes = (1, 2)
    kept_axes = tuple(max(1, axis) for axis in kept_axes)

    # Rebuild 2D WCS keywords from selected source axes.
    header = source_header.copy()
    axis_prefixes = ('NAXIS', 'CTYPE', 'CRPIX', 'CRVAL', 'CDELT', 'CUNIT', 'CROTA')
    for key in list(header.keys()):
        key_upper = key.upper()
        removed = False
        for prefix in axis_prefixes:
            suffix = key_upper[len(prefix):] if key_upper.startswith(prefix) else ""
            if suffix.isdigit():
                del header[key]
                removed = True
                break
        if removed:
            continue
        if key_upper.startswith(('PC', 'CD')) and '_' in key_upper:
            row_col = key_upper[2:].split('_', 1)
            if len(row_col) == 2 and row_col[0].isdigit() and row_col[1].isdigit():
                del header[key]
                continue
        if key_upper.startswith(('PV', 'PS')) and '_' in key_upper:
            axis_token = key_upper[2:].split('_', 1)[0]
            if axis_token.isdigit():
                del header[key]
                continue
        if key_upper == 'WCSAXES':
            del header[key]
            continue

    header['NAXIS'] = 2
    header['NAXIS1'] = int(moment_data.shape[1])
    header['NAXIS2'] = int(moment_data.shape[0])

    for new_axis, src_axis in enumerate(kept_axes, start=1):
        for prefix in ('CTYPE', 'CRPIX', 'CRVAL', 'CDELT', 'CUNIT', 'CROTA'):
            src_key = f"{prefix}{src_axis}"
            if src_key in source_header:
                header[f"{prefix}{new_axis}"] = source_header[src_key]

    for prefix in ('PC', 'CD'):
        for new_row, src_row in enumerate(kept_axes, start=1):
            for new_col, src_col in enumerate(kept_axes, start=1):
                src_key = f"{prefix}{src_row}_{src_col}"
                if src_key in source_header:
                    header[f"{prefix}{new_row}_{new_col}"] = source_header[src_key]

    for key in source_header.keys():
        key_upper = key.upper()
        if not key_upper.startswith(('PV', 'PS')) or '_' not in key_upper:
            continue
        axis_part, remainder = key_upper[2:].split('_', 1)
        if not axis_part.isdigit():
            continue
        src_axis = int(axis_part)
        if src_axis not in kept_axes:
            continue
        new_axis = kept_axes.index(src_axis) + 1
        new_key = f"{key_upper[:2]}{new_axis}_{remainder}"
        header[new_key] = source_header[key]

    header['WCSAXES'] = 2

    # Set BUNIT based on operation and integrated axis.
    inferred_bunit = _moment_bunit(state, moment_type, integration_axis)
    if inferred_bunit:
        header["BUNIT"] = inferred_bunit

    # Add history
    from datetime import datetime

    full_history = []
    history_metadata, sanitized_history_entries = _sanitize_moment_history_entries(history_entries)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_history.append(f"Integration executed by takefits on {timestamp}")

    metadata_source_file = str(history_metadata.get("source_file") or "").strip()
    filepath = getattr(state, "filepath", None)
    safe_filepath = (
        metadata_source_file
        or (os.path.basename(filepath) if filepath else "unknown_source.fits")
    )
    full_history.append(f"Source file: {safe_filepath}")

    mode_map = {
        'int': 'Integration', 'moment0': 'Integration', 'moment1': 'Moment 1', 'moment2': 'Moment 2',
        'average': 'Average', 'peak_int': 'Peak Intensity', 'peak': 'Peak Intensity', 
        'peak_corrd': 'Peak Coordinate', 'median_int': 'Median', 'rms': 'RMS',
        'sigma': 'Sigma (Std Dev)'
    }
    mode_str = str(history_metadata.get("mode") or "").strip() or mode_map.get(moment_type, moment_type)
    full_history.append(f"Mode: {mode_str}")
    
    axis_name = _moment_axis_name(integration_axis)
    full_history.append(f"Axis: {axis_name}")

    range_str = _derive_history_range_text(
        state,
        integration_axis,
        history_metadata,
        pixel_range=pixel_range,
        world_range=world_range,
    )
    full_history.append(
        _moment_range_history_line(
            state,
            integration_axis,
            range_str,
            history_metadata=history_metadata,
        )
    )

    clip_thresh = history_metadata.get(
        "clipping",
        getattr(state, 'clip_threshold', getattr(state, 'moment_clip', 'None')),
    )
    full_history.append(f"Clipping: {clip_thresh}")

    if sanitized_history_entries:
        for entry in sanitized_history_entries:
            full_history.append(entry)

    # Add back to header retaining order
    if 'HISTORY' in header:
        del header['HISTORY']
    for entry in full_history:
        header.add_history(entry)

    update_datamin_datamax_if_present(header, moment_data)

    # Write file
    from takefits.core.io.save_fits import atomic_write_fits

    hdu = fits.PrimaryHDU(
        data=moment_data.astype(np.float32, copy=False),
        header=header,
    )
    atomic_write_fits(
        output_path,
        lambda temporary: hdu.writeto(temporary, overwrite=True),
    )

    return output_path
