"""Utilities for marker state handling that do not depend on PyQt."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence


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
