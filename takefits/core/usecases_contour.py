"""Usecases for contour operations (UI-facing)."""
from __future__ import annotations

from typing import Dict, Iterable, Optional

from takefits.core.contour_manager import ContourManager, ContourParameters, ContourState


def compute_contours(
    layer_ids: Iterable[str],
    params: ContourParameters,
    overlay_ids: Optional[Iterable[str]] = None,
) -> Dict[str, Optional[ContourState]]:
    """
    Apply contour parameters to selected layers/overlays.

    Returns a mapping of layer_id -> ContourState (or None if skipped).
    """
    manager = ContourManager.instance()
    results: Dict[str, Optional[ContourState]] = {}

    layer_ids = list(layer_ids)
    overlay_ids = list(overlay_ids) if overlay_ids is not None else []

    if layer_ids:
        manager.update_parameters(params)
        results = manager.apply_to_layers(layer_ids)

    if overlay_ids:
        manager.apply_overlays(overlay_ids, params.color, params.linewidth)

    return results


def clear_contours(
    layer_ids: Iterable[str],
    overlay_ids: Iterable[str],
) -> None:
    manager = ContourManager.instance()
    layer_ids = list(layer_ids)
    overlay_ids = list(overlay_ids)
    if layer_ids:
        manager.clear_layers(layer_ids)
    if overlay_ids:
        manager.clear_overlays(overlay_ids)
