"""Render styling usecases (TF-302).

Headless image exports read their tick/label/font/colorbar styling from the
stored ``config.yaml``. That makes the styling reachable from the GUI only.
These usecases put a sparse override dict on :class:`AppState` so the same
styling is reachable from an action, a CLI manifest, and therefore from the
AI tool surface.

The override dict is validated against the known config keys, so a typo fails
loudly instead of silently rendering the default.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from takefits.core.app_state import AppState
from takefits.core.config import DEFAULT_CONFIG_KEYS


#: Config keys handled by the semantic colorbar auto-layout
#: (``core/colorbar_layout.py``) rather than the manual ``cbar_pos_*``
#: rectangle. Headless export keeps its manual rectangle unless one of these
#: appears in the overrides, at which point it runs the same layout the GUI
#: uses. Set ``colorbar_auto_layout: false`` alongside them to opt back out.
COLORBAR_AUTO_LAYOUT_KEYS = frozenset({
    "colorbar_auto_layout",
    "colorbar_placement",
    "colorbar_align",
    "colorbar_gap_px",
    "colorbar_gap_x_px",
    "colorbar_gap_y_px",
    "colorbar_thickness_px",
    "colorbar_length_mode",
    "colorbar_length_value",
})


def _validate_overrides(overrides: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(overrides, dict):
        raise TypeError("render config overrides must be a mapping")

    unknown = sorted(key for key in overrides if key not in DEFAULT_CONFIG_KEYS)
    if unknown:
        raise ValueError(
            "Unknown render config key(s): "
            + ", ".join(unknown)
            + ". Valid keys come from the takefits config defaults "
            "(see docs/dev/internal_commands.md)."
        )

    return {str(key): value for key, value in overrides.items()}


def colorbar_auto_layout_requested(state: Optional[AppState]) -> bool:
    """True when the overrides ask for semantic colorbar placement.

    Headless export keeps its historical manual ``cbar_pos_*`` rectangle unless
    the caller explicitly names one of the auto-layout keys, so existing
    pipelines render unchanged. An explicit ``colorbar_auto_layout: false``
    always wins.
    """
    overrides = getattr(state, "render_config", None) or {}
    if overrides.get("colorbar_auto_layout") is False:
        return False
    return bool(set(overrides) & COLORBAR_AUTO_LAYOUT_KEYS)


def set_render_config(
    state: AppState,
    overrides: Optional[Dict[str, Any]] = None,
    replace: bool = False,
) -> AppState:
    """Merge *overrides* into ``state.render_config``.

    Args:
        state: AppState to update.
        overrides: Sparse mapping of config keys to values. Every key must be a
            known takefits config key.
        replace: When True the previous overrides are discarded first.

    Returns:
        The updated state.

    Raises:
        ValueError: if any key is not a known config key.
    """
    validated = _validate_overrides(overrides or {})
    if replace:
        state.render_config = validated
    else:
        merged = dict(state.render_config or {})
        merged.update(validated)
        state.render_config = merged
    return state


def clear_render_config(state: AppState) -> AppState:
    """Drop all render styling overrides, restoring stored config behavior."""
    state.render_config = {}
    return state


def resolve_render_config(
    state: Optional[AppState],
    base: Dict[str, Any],
) -> Dict[str, Any]:
    """Return *base* with the state's render overrides applied.

    The caller keeps ownership of *base*; a copy is returned so a shared
    ``ConfigManager.config`` is never mutated in place.
    """
    resolved = dict(base)
    overrides = getattr(state, "render_config", None) or {}
    resolved.update(overrides)
    return resolved
