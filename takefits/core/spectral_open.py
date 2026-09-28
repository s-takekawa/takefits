"""How a radio cube's spectral axis opens: the Preferences, the launch option, saved records.

TF-415 slice A (roadmap 10.7).  ``load_fits(frequency_axis=...)`` does the
work; this module decides the value.  The Preferences set the default, the
launch option ``--spectral-axis`` overrides it for one launch, and a
workspace opens in the mode it was saved with.  The mode that applied is
kept in ``spectral_metadata['spectral_axis_mode']`` and written into the
source record of workspaces, recipes and range files.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

from takefits.core.io.fits import FREQUENCY_AXIS_MODES

FREQUENCY_AXIS_PREFERENCE = "spectral_axis_frequency"
# --spectral-axis values -> load_fits(frequency_axis=...)
LAUNCH_SPECTRAL_AXIS = {"vel": "velocity", "freq": "frequency"}


def preferred_frequency_axis(config: Optional[Mapping[str, Any]]) -> str:
    """The Preferences' mode for frequency axes ('velocity' unless set otherwise)."""
    value = str((config or {}).get(FREQUENCY_AXIS_PREFERENCE) or "velocity").strip().lower()
    return value if value in FREQUENCY_AXIS_MODES else "velocity"


def recorded_frequency_axis(document: Any) -> Optional[str]:
    """The mode a saved document's source was opened with, or None (older files)."""
    source = document.get("source") if isinstance(document, dict) else None
    mode = source.get("spectral_axis") if isinstance(source, dict) else None
    value = mode.get("frequency_axis") if isinstance(mode, dict) else None
    return value if value in FREQUENCY_AXIS_MODES else None


def workspace_frequency_axis(workspace_path) -> str:
    """The mode to reopen a workspace's cube with: its record, else 'velocity' (older workspaces)."""
    try:
        document = json.loads(Path(workspace_path).read_text(encoding="utf-8"))
    except Exception:
        return "velocity"
    return recorded_frequency_axis(document) or "velocity"
