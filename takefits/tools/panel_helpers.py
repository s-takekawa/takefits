"""Shared helper functions used by multiple tool panels."""
from __future__ import annotations


def _resolve_xz_subwindow(subwindows):
    """Return the first (XZ) subwindow, or None if unavailable."""
    if not subwindows:
        return None
    try:
        window = subwindows[0]
    except Exception:
        return None
    return window if window is not None else None


def _resolve_z_view_limits(fits_viewer, subwindows):
    """Return the Z-axis pixel limits ``(zmin, zmax)`` from the best available source."""
    xz_window = _resolve_xz_subwindow(subwindows)
    if xz_window is not None:
        try:
            return tuple(xz_window.ax.get_ylim())
        except Exception:
            pass

    limits = getattr(fits_viewer, "original_zlim", None)
    if isinstance(limits, (tuple, list)) and len(limits) == 2:
        return tuple(limits)

    data = getattr(fits_viewer, "data", None)
    if getattr(data, "ndim", 0) > 2:
        depth_len = data.shape[data.ndim - 3]
        return (-0.5, depth_len - 0.5)

    return (0.0, 0.0)
