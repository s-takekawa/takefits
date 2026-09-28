"""Line edits that hold spectral world numbers (TF-415 slice A).

A workspace restores the text of every line edit.  The ones marked here are
converted when the workspace was written in another spectral unit (km/s and
m/s, Hz and GHz).
"""
from __future__ import annotations

from takefits.core.spectral_records import SPECTRAL_FIELD_PROPERTY


def mark_spectral_field(*widgets, on: bool = True) -> None:
    """Mark (or unmark) line edits whose text is a spectral world number."""
    for widget in widgets:
        if widget is not None:
            widget.setProperty(SPECTRAL_FIELD_PROPERTY, bool(on))


def is_spectral_field(widget) -> bool:
    try:
        return bool(widget.property(SPECTRAL_FIELD_PROPERTY))
    except Exception:
        return False
