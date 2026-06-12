"""Build contour overlay states from an external FITS file.

Contour polylines are computed on the source file's native pixel grid and
carried as world coordinates, so importing the resulting state onto a layer
reprojects the vertices through the target WCS without regridding the data.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from matplotlib.figure import Figure

try:
    from scipy.ndimage import gaussian_filter
except Exception:
    gaussian_filter = None

from takefits.core.contour_manager import (
    ContourItemState,
    ContourParameters,
    ContourSegment,
    ContourState,
    _frame_name_from_wcs,
    _pixel_coords_to_world,
)


class ExternalContourError(ValueError):
    """Raised when contours cannot be built from an external dataset."""


def estimate_rms(data2d) -> Optional[float]:
    """Robust rms estimate via the median absolute deviation (MAD)."""
    arr = np.asarray(data2d, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    mad = float(np.median(np.abs(finite - np.median(finite))))
    rms = 1.4826 * mad
    if not np.isfinite(rms) or rms <= 0:
        std = float(np.std(finite))
        return std if np.isfinite(std) and std > 0 else None
    return rms


def sigma_levels(
    rms: float,
    factors: Sequence[float],
    include_negative: bool = False,
) -> List[float]:
    """Expand sigma factors (e.g. 3, 6, 12) into data-unit contour levels."""
    base: List[float] = []
    for factor in factors or []:
        try:
            factor = float(factor)
        except (TypeError, ValueError):
            continue
        if np.isfinite(factor) and factor > 0:
            base.append(factor)
    base = sorted(set(base))
    levels = [factor * rms for factor in base]
    if include_negative:
        levels = [-factor * rms for factor in reversed(base)] + levels
    return levels


def describe_source(state) -> Dict[str, str]:
    """Short metadata strings (shape/bunit/frame) for display purposes."""
    info: Dict[str, str] = {}
    shape = getattr(state, "shape", None)
    if shape:
        info["shape"] = "×".join(str(n) for n in reversed(tuple(shape)))
    header = getattr(state, "header", None)
    try:
        bunit = str(header.get("BUNIT", "")).strip() if header is not None else ""
    except Exception:
        bunit = ""
    if bunit:
        info["bunit"] = bunit
    try:
        frame = _frame_name_from_wcs(getattr(state, "wcs", None).celestial)
    except Exception:
        frame = None
    if frame:
        info["frame"] = frame if frame in ("ICRS", "FK4", "FK5") else frame.capitalize()
    return info


def channel_world_value(state, channel: int) -> Optional[Tuple[float, str]]:
    """World value (e.g. velocity) and unit for a channel index, if derivable."""
    wcs = getattr(state, "wcs", None)
    if wcs is None:
        return None
    meta = getattr(state, "spectral_metadata", None) or {}
    axis_index = meta.get("axis_index")
    if axis_index is None:
        if getattr(wcs, "naxis", 0) >= 3:
            axis_index = 3
        else:
            return None
    try:
        sub = wcs.sub([int(axis_index)])
        value = float(np.atleast_1d(sub.pixel_to_world_values(float(channel)))[0])
        unit = str(meta.get("current_axis_unit") or sub.wcs.cunit[0] or "").strip()
        if unit.replace(" ", "").lower() == "m/s":
            value /= 1000.0
            unit = "km/s"
        return value, unit
    except Exception:
        return None


def default_levels(data2d, n_levels: int = 5) -> List[float]:
    """Suggest evenly spaced levels strictly between the data min and max."""
    arr = np.asarray(data2d, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return []
    vmin = float(finite.min())
    vmax = float(finite.max())
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        return []
    n_levels = max(1, int(n_levels))
    return [float(v) for v in np.linspace(vmin, vmax, n_levels + 2)[1:-1]]


def _normalize_levels(levels: Sequence[float]) -> List[float]:
    values = []
    for value in levels or []:
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return sorted(set(values))


def _celestial_wcs(wcs) -> object:
    if wcs is None:
        raise ExternalContourError(
            "The FITS file has no celestial WCS; contours cannot be aligned."
        )
    try:
        celestial = wcs.celestial
        if celestial is None or celestial.naxis != 2:
            raise ValueError
    except Exception:
        raise ExternalContourError(
            "The FITS file has no celestial WCS; contours cannot be aligned."
        )
    return celestial


def _slice_xy_plane(state, channel: Optional[int]) -> np.ndarray:
    if state.data is None:
        raise ExternalContourError("The FITS file contains no image data.")
    if channel is not None and getattr(state.data, "ndim", 0) >= 3:
        n_channels = state.n_channels
        if not 0 <= int(channel) < n_channels:
            raise ExternalContourError(
                f"Channel {channel} is out of range (0..{n_channels - 1})."
            )
        state.current_z = int(channel)
    data2d = state.get_slice_2d('xy')
    if data2d is None or getattr(data2d, "ndim", 0) != 2 or data2d.size == 0:
        raise ExternalContourError("Could not extract a 2D image plane for contours.")
    return np.asarray(data2d, dtype=float)


def smooth_plane(data2d, sigma: float):
    """Gaussian-smooth a 2D plane (same behaviour as the main contour panel)."""
    if not sigma or sigma <= 0 or gaussian_filter is None or data2d is None:
        return data2d
    arr = np.asarray(data2d, dtype=float)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    try:
        return gaussian_filter(arr, sigma=float(sigma))
    except Exception:
        return data2d


def _contour_polylines(data2d: np.ndarray, levels: List[float]):
    """Return matplotlib allsegs: per level, a list of (N, 2) x/y pixel arrays."""
    fig = Figure()
    ax = fig.add_subplot()
    contour_set = ax.contour(data2d, levels=levels)
    return contour_set.allsegs


def build_contour_state_from_app_state(
    state,
    levels: Sequence[float],
    *,
    channel: Optional[int] = None,
    color: str = "white",
    linewidth: float = 1.0,
    smoothing: float = 0.0,
    label: Optional[str] = None,
    source_meta: Optional[Dict[str, object]] = None,
) -> ContourState:
    """
    Compute contours of an AppState's XY plane as a world-coordinate ContourState.

    The returned state can be imported onto any contour layer with
    ContourManager.import_layer_state(); vertices are reprojected through the
    target WCS on import.
    """
    level_values = _normalize_levels(levels)
    if not level_values:
        raise ExternalContourError("No valid contour levels were provided.")

    wcs_cel = _celestial_wcs(getattr(state, "wcs", None))
    data2d = smooth_plane(_slice_xy_plane(state, channel), smoothing)

    if label is None:
        filepath = getattr(state, "filepath", None) or ""
        label = os.path.basename(str(filepath)) or "external contours"

    segments: List[ContourSegment] = []
    for level, segs in zip(level_values, _contour_polylines(data2d, level_values)):
        for vertices in segs:
            vertices = np.asarray(vertices, dtype=float)
            if vertices.ndim != 2 or vertices.shape[0] < 2:
                continue
            finite_mask = np.isfinite(vertices).all(axis=1)
            if np.count_nonzero(finite_mask) < 2:
                continue
            world = _pixel_coords_to_world(vertices[finite_mask], wcs_cel)
            if world is None or world.shape[0] < 2:
                continue
            segments.append(
                ContourSegment(
                    level=level,
                    world=world,
                    pixels=None,
                    linestyle="dashed" if level < 0 else "solid",
                )
            )

    if not segments:
        raise ExternalContourError("No contours were produced at the requested levels.")

    parameters = ContourParameters(
        levels=list(level_values),
        color=color,
        linewidth=float(linewidth),
        smoothing=float(smoothing or 0.0),
    )
    return ContourState(
        layer_id="",
        plane=None,
        label=label,
        parameters=parameters,
        levels=list(level_values),
        items=[ContourItemState(item_label=label, segments=segments)],
        world_frame=_frame_name_from_wcs(wcs_cel),
        source_meta=dict(source_meta) if source_meta else None,
    )


def build_contour_state_from_fits(
    filepath: str,
    levels: Sequence[float],
    *,
    channel: Optional[int] = None,
    color: str = "white",
    linewidth: float = 1.0,
    smoothing: float = 0.0,
    label: Optional[str] = None,
) -> ContourState:
    """Load a FITS file and build a world-coordinate ContourState from it."""
    from takefits.core.usecases import load_fits_data

    state = load_fits_data(str(filepath))
    return build_contour_state_from_app_state(
        state,
        levels,
        channel=channel,
        color=color,
        linewidth=linewidth,
        smoothing=smoothing,
        label=label or os.path.basename(str(filepath)),
    )
