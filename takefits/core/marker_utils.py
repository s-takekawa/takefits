"""Utilities for marker state handling that do not depend on PyQt."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple


#: Angular units accepted for a marker length, expressed in degrees.
ANGULAR_UNIT_DEG = {
    "deg": 1.0,
    "degree": 1.0,
    "degrees": 1.0,
    "arcmin": 1.0 / 60.0,
    "arcminute": 1.0 / 60.0,
    "'": 1.0 / 60.0,
    "arcsec": 1.0 / 3600.0,
    "arcsecond": 1.0 / 3600.0,
    '"': 1.0 / 3600.0,
}


def pixel_scale_deg(wcs: Any) -> Optional[float]:
    """Degrees per pixel along the first celestial axis, or None.

    Headless counterpart of the viewer-bound resolver in
    ``core/marker_manager.py``; it needs only a WCS object.
    """
    if wcs is None:
        return None
    try:
        from astropy.wcs.utils import proj_plane_pixel_scales

        celestial = wcs.celestial if getattr(wcs, "has_celestial", False) else wcs
        scales = proj_plane_pixel_scales(celestial)
    except Exception:
        return None
    for scale in scales:
        try:
            value = abs(float(scale))
        except (TypeError, ValueError):
            continue
        if value > 0.0:
            return value
    return None


def angular_length_to_pixels(
    value: float,
    unit: str,
    wcs: Any,
) -> Optional[float]:
    """Convert an angular marker length to pixels using *wcs*.

    Returns None when the unit is not angular or the WCS has no usable
    celestial pixel scale, so callers can fall back to a pixel length.
    """
    factor = ANGULAR_UNIT_DEG.get(str(unit or "").strip().lower())
    if factor is None:
        return None
    scale = pixel_scale_deg(wcs)
    if not scale:
        return None
    try:
        return abs(float(value)) * factor / scale
    except (TypeError, ValueError):
        return None


def resolve_anchor_fraction(
    anchor_frac: Sequence[float],
    shape: Sequence[int],
) -> Optional[Tuple[float, float]]:
    """Map an axes fraction to pixel coordinates for an image of *shape*.

    ``shape`` is the ``(ny, nx)`` of the rendered image. ``(0, 0)`` is the
    bottom-left corner of the image and ``(1, 1)`` the top-right, matching how
    the map is drawn with ``origin='lower'``.
    """
    try:
        fx, fy = (float(v) for v in tuple(anchor_frac)[:2])
        ny, nx = (int(v) for v in tuple(shape)[-2:])
    except (TypeError, ValueError):
        return None
    if nx <= 0 or ny <= 0:
        return None
    # Pixel centres run from -0.5 to n-0.5 in Matplotlib image coordinates.
    return (fx * nx - 0.5, fy * ny - 0.5)


def _coerce_world_defaults(values: Optional[Sequence[object]]) -> Optional[List[float]]:
    if values is None:
        return None
    defaults = [0.0, 0.0, 0.0, 0.0]
    try:
        sequence = list(values)
    except Exception:
        return None
    for idx in range(min(len(sequence), len(defaults))):
        try:
            defaults[idx] = float(sequence[idx]) if sequence[idx] is not None else 0.0
        except Exception:
            defaults[idx] = 0.0
    return defaults


def shared_world_defaults(viewer: Any, plane: Optional[str] = None) -> List[float]:
    """
    Best-effort world defaults for non-plane axes when importing markers.

    Works with viewers that provide:
    - marker_world_defaults(plane) method, or
    - _get_shared_world_x/y/z/s() methods, or
    - world_x/world_y/world_z/world_s attributes.
    Missing values are returned as 0.0.
    """
    defaults = [0.0, 0.0, 0.0, 0.0]
    if viewer is None:
        return defaults

    custom_defaults = getattr(viewer, "marker_world_defaults", None)
    if callable(custom_defaults):
        try:
            resolved = _coerce_world_defaults(custom_defaults(plane))
        except Exception:
            resolved = None
        if resolved is not None:
            return resolved

    def _read_world(method_name: str, attr_name: str) -> float:
        getter = getattr(viewer, method_name, None)
        if callable(getter):
            try:
                value = getter()
                return float(value) if value is not None else 0.0
            except Exception:
                pass
        raw = getattr(viewer, attr_name, None)
        try:
            return float(raw) if raw is not None else 0.0
        except Exception:
            return 0.0

    defaults[0] = _read_world("_get_shared_world_x", "world_x")
    defaults[1] = _read_world("_get_shared_world_y", "world_y")
    defaults[2] = _read_world("_get_shared_world_z", "world_z")
    defaults[3] = _read_world("_get_shared_world_s", "world_s")
    return defaults
