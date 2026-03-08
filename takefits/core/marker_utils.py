"""Utilities for marker state handling that do not depend on PyQt."""

from __future__ import annotations

from typing import Any, List


def shared_world_defaults(viewer: Any) -> List[float]:
    """
    Best-effort world defaults for non-plane axes when importing markers.

    Works with viewers that provide:
    - _get_shared_world_x/y/z/s() methods, or
    - world_x/world_y/world_z/world_s attributes.
    Missing values are returned as 0.0.
    """
    defaults = [0.0, 0.0, 0.0, 0.0]
    if viewer is None:
        return defaults

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

