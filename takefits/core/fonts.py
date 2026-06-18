"""Font helpers for Matplotlib rendering."""
from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

from matplotlib import font_manager

DEFAULT_MPL_FONT_FAMILY = "DejaVu Sans"

_GENERIC_FONT_FAMILIES = {
    "cursive",
    "fantasy",
    "monospace",
    "sans",
    "sans serif",
    "sans-serif",
    "serif",
}


@lru_cache(maxsize=1)
def _available_font_names() -> set[str]:
    return {font.name.casefold() for font in font_manager.fontManager.ttflist}


def resolve_mpl_font_family(font_family, *, default: str = DEFAULT_MPL_FONT_FAMILY):
    """Return a Matplotlib font family that exists on the current system.

    Missing concrete font names such as ``Arial`` make Matplotlib emit repeated
    ``findfont`` warnings on Linux/WSL.  Keep installed fonts unchanged and use
    Matplotlib's bundled DejaVu Sans as the cross-platform fallback.
    """
    if isinstance(font_family, Sequence) and not isinstance(font_family, str):
        resolved = [
            name
            for name in (resolve_mpl_font_family(item, default="") for item in font_family)
            if name
        ]
        return resolved or default

    name = str(font_family or "").strip()
    if not name:
        return default

    normalized = name.casefold()
    if normalized in _GENERIC_FONT_FAMILIES:
        return name
    if normalized in _available_font_names():
        return name
    return default
