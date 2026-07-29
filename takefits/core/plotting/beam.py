"""HPBW beam geometry and headless rendering (TF-303).

The beam ellipse is effectively mandatory in radio-astronomy figures. The
existing `logic/add_hpbw.py: AddHPBW` already draws one with plain Matplotlib,
but it is built for the interactive viewer: it caches the first config in a
class attribute, hooks draw/motion events, wraps `figure.savefig`, and keeps the
artist invisible between blits. None of that suits a one-shot headless render.

This module holds the part worth sharing — the pixel geometry — plus a plain
always-visible renderer for headless exports. `AddHPBW` delegates its geometry
here so the GUI and the CLI cannot drift apart.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np


def beam_pixel_geometry(header: Any) -> Optional[Tuple[float, float, float]]:
    """Return ``(major_px, minor_px, bpa_deg)`` for the header's beam.

    Returns None when the header carries no usable beam: missing or
    non-positive ``BMAJ`` / ``BMIN``, or celestial units that are not degrees
    (the pixel scale would not be comparable).
    """
    if header is None:
        return None

    def _get(key, default=None):
        try:
            return header.get(key, default)
        except Exception:
            return default

    cunit1 = str(_get('CUNIT1', '') or '').strip().lower()
    cunit2 = str(_get('CUNIT2', '') or '').strip().lower()
    if (cunit1 or cunit2) and (cunit1 != 'deg' or cunit2 != 'deg'):
        return None

    try:
        bmaj = float(_get('BMAJ', 0) or 0.0)
        bmin = float(_get('BMIN', 0) or 0.0)
        bpa = float(_get('BPA', 0) or 0.0)
    except (TypeError, ValueError):
        return None
    if bmaj <= 0.0 or bmin <= 0.0:
        return None

    try:
        cdelt1 = abs(float(header['CDELT1']))
        cdelt2 = abs(float(header['CDELT2']))
    except Exception:
        # Coordinates defined by a CD matrix; derive the axis scales from it.
        try:
            cdelt1 = float(np.hypot(header['CD1_1'], header['CD2_1']))
            cdelt2 = float(np.hypot(header['CD1_2'], header['CD2_2']))
        except Exception:
            return None
    if not cdelt1 or not cdelt2:
        return None

    return bmaj / cdelt1, bmin / cdelt2, bpa


def draw_beam_on_axes(ax: Any, header: Any, config: Any) -> Optional[Any]:
    """Draw the HPBW beam ellipse on *ax*, or return None if there is no beam.

    The ellipse is placed at the configured relative position inside the
    current axes limits, so it follows whatever view range the export used.
    """
    geometry = beam_pixel_geometry(header)
    if geometry is None or ax is None:
        return None

    from matplotlib.patches import Ellipse

    major_px, minor_px, bpa = geometry
    config = config or {}

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    rel_x = float(config.get('beam_pos_x', 0.1))
    rel_y = float(config.get('beam_pos_y', 0.1))
    center = (
        xlim[0] + rel_x * (xlim[1] - xlim[0]),
        ylim[0] + rel_y * (ylim[1] - ylim[0]),
    )

    ellipse = Ellipse(
        center,
        width=major_px,
        height=minor_px,
        # `AddHPBW` uses BPA + 90 so the major axis runs along the sky
        # position angle; keep the same convention.
        angle=bpa + 90.0,
        facecolor=config.get('beam_facecolor', 'white'),
        edgecolor=config.get('beam_edgecolor', 'None'),
        linewidth=config.get('beam_linewidth', 0),
        zorder=8,
    )
    ax.add_patch(ellipse)
    return ellipse
