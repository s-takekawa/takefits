"""Spectral-axis usecases (TF-415): the line's rest frequency and the systemic redshift."""
from __future__ import annotations

from typing import Optional

from takefits.core.app_state import AppState
from takefits.core.spectral_units import apply_spectral_axis


def set_spectral_axis(
    state: AppState,
    intent: str,
    restfreq_hz: Optional[float] = None,
    z: Optional[float] = None,
) -> AppState:
    """Set the line's rest frequency and/or the systemic redshift z; ``intent`` says what stays.

    Velocities are measured from the effective rest frequency
    ``restfreq_hz / (1 + z)``.  A value left out keeps the current one, and
    z = 0 removes the systemic redshift.

    - ``'rereference'``: the observed frequencies stay, and the velocity axis
      is re-derived (CRVAL and the channel width together, so the width in
      Hz is unchanged).
    - ``'metadata'``: the velocity axis stays; only the rest frequency
      changes (for example RESTFRQ for K <-> Jy/beam on a velocity-native
      cube).

    The data are never resampled.  The header (RESTFRQ = f0 / (1 + z),
    ZSOURCE), the WCS and ``spectral_metadata`` change together; see
    ``takefits.core.spectral_units.apply_spectral_axis``.
    """
    if state.header is None or state.wcs is None:
        raise ValueError("No FITS header/WCS loaded")
    if not isinstance(state.spectral_metadata, dict):
        state.spectral_metadata = {}
    apply_spectral_axis(
        state.wcs, state.header, state.spectral_metadata, restfreq_hz=restfreq_hz, z=z, intent=intent
    )
    return state
