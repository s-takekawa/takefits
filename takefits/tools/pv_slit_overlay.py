"""Static PV-slit overlays for copying a slit onto another xy window.

A PV slit is normally drawn (and edited) on the main viewer's overlay axis by
``PVdiagram``; those artists vanish when the PV panel closes. For publication
figures it is useful to drop a *static, read-only* copy of the slit onto another
window that shares the same xy pixel grid (e.g. an integrated/moment-map result
window).

This module is intentionally Qt-free so the geometry can be unit-tested without
a running application. ``build_slit_overlay`` packages a sampled slit centreline
(plus per-sample normals, which ``compute_pv`` already provides) into a plain
dict; ``serialize_slit_overlay`` converts it to workspace-safe JSON data, and
``draw_slit_overlay`` renders it onto any matplotlib axes whose data coordinates
are main-image pixels.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

# Drawn above the image but below interactive cursor/region overlays.
_SLIT_OVERLAY_ZORDER = 5
_SLIT_OVERLAY_TAG = "_takefits_pv_slit_overlay"


def _finite_float(value, default: float) -> float:
    try:
        number = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(number):
        return float(default)
    return number


def _point_array(value, *, min_rows: int) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(value, dtype=float)
    except Exception:
        return None
    if arr.ndim != 2 or arr.shape[0] < min_rows or arr.shape[1] < 2:
        return None
    arr = arr[:, :2]
    if not np.isfinite(arr).all():
        return None
    return arr


def build_slit_overlay(
    xs,
    ys,
    normal_x,
    normal_y,
    *,
    width_px: float = 0.0,
    color: str = "yellow",
    linewidth: float = 1.5,
    closed: bool = False,
    label: str = "PV slit",
) -> Optional[dict]:
    """Package a sampled slit centreline into a serialisable overlay dict.

    ``xs``/``ys`` are the centreline in main-image pixel coordinates and
    ``normal_x``/``normal_y`` are the per-sample unit normals (as returned by
    ``takefits.core.usecases.pv.sample_path_points``). ``width_px`` is the slit
    integration width; when positive, the overlay carries the two parallel edge
    lines. Returns ``None`` when the centreline has fewer than two finite points.
    """
    xs = np.asarray(xs, dtype=float).ravel()
    ys = np.asarray(ys, dtype=float).ravel()
    if xs.size < 2 or xs.size != ys.size:
        return None
    centerline = np.column_stack([xs, ys])
    finite = np.isfinite(centerline).all(axis=1)
    if np.count_nonzero(finite) < 2:
        return None

    normals = None
    nx = np.asarray(normal_x, dtype=float).ravel()
    ny = np.asarray(normal_y, dtype=float).ravel()
    if nx.size == xs.size and ny.size == xs.size:
        normals = np.column_stack([nx, ny])

    return {
        "centerline": centerline,
        "normals": normals,
        "width_px": float(width_px or 0.0),
        "color": str(color),
        "linewidth": float(linewidth),
        "closed": bool(closed),
        "label": str(label),
    }


def deserialize_slit_overlay(payload: dict) -> Optional[dict]:
    """Normalize a workspace payload back into a drawable slit overlay."""
    if not isinstance(payload, dict):
        return None

    centerline = _point_array(payload.get("centerline"), min_rows=2)
    if centerline is None:
        return None

    normals = None
    raw_normals = payload.get("normals")
    if raw_normals is not None:
        parsed_normals = _point_array(raw_normals, min_rows=centerline.shape[0])
        if parsed_normals is not None and parsed_normals.shape == centerline.shape:
            normals = parsed_normals

    linewidth = max(0.1, _finite_float(payload.get("linewidth", 1.5), 1.5))
    width_px = max(0.0, _finite_float(payload.get("width_px", 0.0), 0.0))

    return {
        "centerline": centerline,
        "normals": normals,
        "width_px": width_px,
        "color": str(payload.get("color") or "yellow"),
        "linewidth": linewidth,
        "closed": bool(payload.get("closed", False)),
        "label": str(payload.get("label") or "PV slit"),
    }


def serialize_slit_overlay(overlay: dict) -> Optional[dict]:
    """Return a JSON-serializable representation of a slit overlay."""
    normalized = deserialize_slit_overlay(overlay)
    if normalized is None:
        return None

    normals = normalized.get("normals")
    return {
        "format": "takefits.pv_slit_overlay",
        "schema": 1,
        "centerline": normalized["centerline"].tolist(),
        "normals": normals.tolist() if normals is not None else None,
        "width_px": float(normalized.get("width_px", 0.0)),
        "color": str(normalized.get("color") or "yellow"),
        "linewidth": float(normalized.get("linewidth", 1.5)),
        "closed": bool(normalized.get("closed", False)),
        "label": str(normalized.get("label") or "PV slit"),
    }


def slit_edge_lines(
    centerline: np.ndarray,
    normals: Optional[np.ndarray],
    width_px: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return the two parallel slit edges offset by ``±width_px/2`` along normals."""
    half = float(width_px) / 2.0
    if half <= 0.0 or normals is None:
        return None, None
    centerline = np.asarray(centerline, dtype=float)
    normals = np.asarray(normals, dtype=float)
    if centerline.shape != normals.shape or centerline.ndim != 2:
        return None, None
    left = centerline + normals * half
    right = centerline - normals * half
    return left, right


def draw_slit_overlay(ax, overlay: dict) -> List:
    """Draw a slit overlay onto ``ax`` (data coords = main-image pixels).

    Returns the list of created artists, each tagged so it can be identified and
    removed later. Width edges (if any) are drawn dashed and thinner than the
    spine.
    """
    from matplotlib.lines import Line2D

    artists: List = []
    if not isinstance(overlay, dict):
        return artists
    center = np.asarray(overlay.get("centerline"), dtype=float)
    if center.ndim != 2 or center.shape[0] < 2:
        return artists

    color = overlay.get("color", "yellow")
    lw = float(overlay.get("linewidth", 1.5))
    closed = bool(overlay.get("closed", False))

    cx = center[:, 0].copy()
    cy = center[:, 1].copy()
    if closed:
        cx = np.append(cx, cx[0])
        cy = np.append(cy, cy[0])

    spine = Line2D(
        cx, cy, color=color, linewidth=lw, linestyle="-",
        zorder=_SLIT_OVERLAY_ZORDER, solid_capstyle="round",
    )
    setattr(spine, _SLIT_OVERLAY_TAG, True)
    ax.add_line(spine)
    artists.append(spine)

    left, right = slit_edge_lines(center, overlay.get("normals"), overlay.get("width_px", 0.0))
    for edge in (left, right):
        if edge is None:
            continue
        ex = edge[:, 0].copy()
        ey = edge[:, 1].copy()
        if closed:
            ex = np.append(ex, ex[0])
            ey = np.append(ey, ey[0])
        edge_line = Line2D(
            ex, ey, color=color, linewidth=max(0.6, lw * 0.6), linestyle="--",
            alpha=0.7, zorder=_SLIT_OVERLAY_ZORDER,
        )
        setattr(edge_line, _SLIT_OVERLAY_TAG, True)
        ax.add_line(edge_line)
        artists.append(edge_line)

    return artists


def iter_slit_overlay_artists(ax):
    """Yield the slit-overlay artists currently living on ``ax``."""
    for line in list(getattr(ax, "lines", [])):
        if getattr(line, _SLIT_OVERLAY_TAG, False):
            yield line
