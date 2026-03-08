"""Annotation (regions/markers) usecases.

These are small, PyQt-free state mutations so annotations can be:
- driven from CLI (ActionSession)
- recorded/replayed for future Undo/Redo
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from takefits.core.app_state import AppState, MarkerSpec, RegionSpec


def set_regions(state: AppState, regions: Sequence[Dict[str, Any] | RegionSpec]) -> AppState:
    """Replace state.regions with the provided list (small snapshot action)."""
    new_regions: List[RegionSpec] = []
    for entry in regions:
        if isinstance(entry, RegionSpec):
            new_regions.append(entry)
        else:
            new_regions.append(RegionSpec.from_dict(entry))
    state.regions = new_regions
    return state


def clear_regions(state: AppState) -> AppState:
    state.regions = []
    return state


def add_region(state: AppState, region: Dict[str, Any] | RegionSpec) -> AppState:
    spec = region if isinstance(region, RegionSpec) else RegionSpec.from_dict(region)
    if spec.region_id is None:
        spec.region_id = _next_region_id(state)
    state.regions.append(spec)
    return state


def update_region(state: AppState, region_id: int | str, updates: Dict[str, Any]) -> AppState:
    """Update an existing region in-place."""
    idx = _find_region_index(state, region_id)
    if idx is None:
        raise ValueError(f"Region id '{region_id}' not found")
    current = state.regions[idx]
    patch = dict(current.to_dict())
    patch.update(dict(updates or {}))
    # Ensure the id we targeted remains stable unless explicitly overwritten.
    patch.setdefault("id", current.region_id)
    state.regions[idx] = RegionSpec.from_dict(patch)
    return state


def delete_region(state: AppState, region_id: int | str) -> AppState:
    idx = _find_region_index(state, region_id)
    if idx is None:
        return state
    state.regions.pop(idx)
    return state


def set_markers(state: AppState, markers: Sequence[Dict[str, Any] | MarkerSpec]) -> AppState:
    """Replace state.markers with the provided list (small snapshot action)."""
    new_markers: List[MarkerSpec] = []
    for entry in markers:
        if isinstance(entry, MarkerSpec):
            new_markers.append(entry)
        else:
            new_markers.append(MarkerSpec.from_dict(entry))
    state.markers = new_markers
    return state


def clear_markers(state: AppState) -> AppState:
    state.markers = []
    return state


def add_marker(state: AppState, marker: Dict[str, Any] | MarkerSpec) -> AppState:
    spec = marker if isinstance(marker, MarkerSpec) else MarkerSpec.from_dict(marker)
    if not spec.marker_id:
        spec.marker_id = MarkerSpec.from_dict(spec.to_dict()).marker_id
    # Replace by id if already present.
    idx = _find_marker_index(state, spec.marker_id)
    if idx is None:
        state.markers.append(spec)
    else:
        state.markers[idx] = spec
    return state


def update_marker(state: AppState, marker_id: str, updates: Dict[str, Any]) -> AppState:
    idx = _find_marker_index(state, marker_id)
    if idx is None:
        raise ValueError(f"Marker id '{marker_id}' not found")
    current = state.markers[idx]
    patch = dict(current.to_dict())
    patch.update(dict(updates or {}))
    patch.setdefault("id", current.marker_id)
    state.markers[idx] = MarkerSpec.from_dict(patch)
    return state


def delete_marker(state: AppState, marker_id: str) -> AppState:
    idx = _find_marker_index(state, marker_id)
    if idx is None:
        return state
    state.markers.pop(idx)
    return state


def _find_region_index(state: AppState, region_id: int | str) -> Optional[int]:
    for idx, region in enumerate(state.regions):
        if str(region.region_id) == str(region_id):
            return idx
    return None


def _next_region_id(state: AppState) -> int:
    """Auto-assign integer ids for regions when one isn't provided."""
    used: set[int] = set()
    for region in state.regions:
        try:
            if region.region_id is None:
                continue
            used.add(int(region.region_id))
        except Exception:
            continue
    candidate = 1
    while candidate in used:
        candidate += 1
    return candidate


def _find_marker_index(state: AppState, marker_id: str) -> Optional[int]:
    for idx, marker in enumerate(state.markers):
        if str(marker.marker_id) == str(marker_id):
            return idx
    return None

