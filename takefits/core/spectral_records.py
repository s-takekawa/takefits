"""Spectral world numbers in saved files: recorded actions, annotations, GUI state.

TF-415 slice A (D2).  A file keeps spectral world numbers in the display unit
of the cube that wrote it and records that unit
(``spectral_units.SPECTRAL_UNIT_KEY``).  A reader takes the factor to the
display unit of the cube it applies them to from
``spectral_units.stored_spectral_factor`` and hands it to the helpers here,
which know where those numbers sit.  A factor of 1 changes nothing; None
(units of different kinds) is for the caller to handle.

The spectral axis is WCS axis 2 of a cube (0-based); on the XZ plane it is
the second member of a world pair, on the ZY plane the first.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

from takefits.core.spectral_units import SPECTRAL_UNIT_KEY, scale_spectral_value


def document_spectral_unit(document: Any) -> Optional[str]:
    """The spectral unit a saved document records, or None (a file from before the record).

    Looked up on the document, on its ``source``, and on the source's
    ``wcs_signature`` (range files, workspaces and recipes carry it there).
    """
    if not isinstance(document, dict):
        return None
    holders: List[Any] = [document]
    source = document.get("source")
    if isinstance(source, dict):
        holders.append(source)
        holders.append(source.get("wcs_signature"))
    for holder in holders:
        if isinstance(holder, dict):
            value = holder.get(SPECTRAL_UNIT_KEY)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def spectral_index_of_plane(plane: Any) -> Optional[int]:
    """Which member of a plane's world pair is spectral: 1 on XZ, 0 on ZY, None on XY."""
    text = str(plane or "").lower()
    if "xz" in text:
        return 1
    if "zy" in text:
        return 0
    return None


def _scale_member(pair: Any, index: Optional[int], factor: float) -> Any:
    if index is None or not isinstance(pair, (list, tuple)) or index >= len(pair):
        return pair
    items = list(pair)
    items[index] = scale_spectral_value(items[index], factor)
    return tuple(items) if isinstance(pair, tuple) else items


def rescale_marker_entry(entry: Any, factor: float) -> Any:
    """A marker entry (``MarkerState`` / ``MarkerSpec`` dict) with its spectral world numbers scaled."""
    if factor == 1.0 or not isinstance(entry, dict):
        return entry
    index = spectral_index_of_plane(entry.get("plane"))
    if index is None:
        return entry
    result = copy.deepcopy(entry)
    if result.get("world") is not None:
        result["world"] = _scale_member(result["world"], index, factor)
    metadata = result.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("world_endpoints"), (list, tuple)):
        metadata["world_endpoints"] = [
            _scale_member(point, index, factor) for point in metadata["world_endpoints"]
        ]
    return result


def rescale_region_entry(entry: Any, factor: float, spectral_axis: int = 2) -> Any:
    """A region entry (``RegionSpec`` dict or region-file entry) with its spectral world numbers scaled.

    ``world.center`` and ``world.axes`` hold a world pair of the region's
    plane; ``world.context_world`` holds the full world vector by WCS axis.
    Sizes in degrees (``width_deg`` ...) are left alone.
    """
    if factor == 1.0 or not isinstance(entry, dict):
        return entry
    world = entry.get("world")
    if not isinstance(world, dict):
        return entry
    result = copy.deepcopy(entry)
    world = result["world"]
    index = spectral_index_of_plane(result.get("plane"))
    if "center" in world:
        world["center"] = _scale_member(world["center"], index, factor)
    axes = world.get("axes")
    if isinstance(axes, dict):
        for key in list(axes):
            axes[key] = _scale_member(axes[key], index, factor)
    context = world.get("context_world")
    if isinstance(context, dict):
        key = str(int(spectral_axis))
        if key in context:
            context[key] = scale_spectral_value(context[key], factor)
    return result


def _scale_list_member(values: Any, index: int, factor: float) -> Any:
    if not isinstance(values, (list, tuple)) or not 0 <= index < len(values):
        return values
    items = list(values)
    items[index] = scale_spectral_value(items[index], factor)
    return items


def _scale_keys(mapping: Any, keys, factor: float) -> None:
    if isinstance(mapping, dict):
        for key in keys:
            if key in mapping:
                mapping[key] = scale_spectral_value(mapping[key], factor)


def rescale_action_params(action: str, params: Any, factor: float, spectral_axis: int = 2) -> Any:
    """The parameters of a recorded action with its spectral world numbers scaled by ``factor``."""
    if factor == 1.0 or not isinstance(params, dict):
        return params
    name = str(action or "")
    result = copy.deepcopy(params)

    def _along_spectrum() -> bool:
        try:
            return int(result.get("axis", 0) or 0) == 0  # numpy axis 0 of (z, y, x)
        except (TypeError, ValueError):
            return False

    if name in ("compute_moment", "export_moment_fits", "export_moment_image"):
        if _along_spectrum():
            _scale_keys(result, ("world_range",), factor)
    elif name in ("compute_channel_map", "export_channel_map_image"):
        if _along_spectrum():
            _scale_keys(result, ("start_world", "end_world", "interval_world"), factor)
    elif name == "apply_baseline_subtraction":
        ranges = result.get("world_ranges")
        if isinstance(ranges, list):
            result["world_ranges"] = [scale_spectral_value(pair, factor) for pair in ranges]
    elif name == "compute_cutout":
        result["world_bounds"] = _scale_list_member(result.get("world_bounds"), spectral_axis, factor)
    elif name == "compute_regrid":
        inner = result.get("params")
        if isinstance(inner, dict):
            for key in ("anchor_world", "grid_cdelt"):
                if key in inner:
                    inner[key] = _scale_list_member(inner[key], spectral_axis, factor)
    elif name == "fit_spectrum_gaussian":
        _scale_keys(result, ("min_sigma", "max_sigma"), factor)
    elif name == "export_spectrum_image":
        _scale_keys(result, ("xlim",), factor)
        _scale_keys(result.get("fit_kwargs"), ("min_sigma", "max_sigma"), factor)
    elif name == "set_render_config":
        _scale_keys(result.get("overrides"), ("z_tick_spacing",), factor)

    if name == "set_markers" and isinstance(result.get("markers"), list):
        result["markers"] = [rescale_marker_entry(item, factor) for item in result["markers"]]
    elif name == "add_marker":
        result["marker"] = rescale_marker_entry(result.get("marker"), factor)
    elif name == "set_regions" and isinstance(result.get("regions"), list):
        result["regions"] = [rescale_region_entry(item, factor, spectral_axis) for item in result["regions"]]
    elif "region" in result and isinstance(result.get("region"), dict):
        result["region"] = rescale_region_entry(result["region"], factor, spectral_axis)
    return result


def rescale_history_entries(entries: Any, factor: float, spectral_axis: int = 2) -> Any:
    """Recorded actions (dicts with ``action`` / ``name`` and ``params``) with spectral numbers scaled."""
    if factor == 1.0 or not isinstance(entries, list):
        return entries
    rescaled = []
    for entry in entries:
        if isinstance(entry, dict):
            entry = dict(entry)
            name = entry.get("action") or entry.get("name")
            entry["params"] = rescale_action_params(name, entry.get("params") or {}, factor, spectral_axis)
        rescaled.append(entry)
    return rescaled


def rescale_range_payload(payload: Dict[str, Any], factor: float) -> Dict[str, Any]:
    """A range-file payload with its z range scaled (texts and native numbers)."""
    if factor == 1.0 or not isinstance(payload, dict):
        return payload
    result = copy.deepcopy(payload)
    z_entry = (result.get("ranges") or {}).get("z")
    _scale_keys(z_entry, ("min_text", "max_text", "native_min", "native_max"), factor)
    return result


def rescale_named_values(document: Any, names, factor: float) -> Any:
    """A copy of ``document`` with every value under a key in ``names`` scaled, at any depth.

    For formats whose spectral numbers sit under distinctive keys (the
    ``spectral_world`` of PV paths).
    """
    if factor == 1.0:
        return document
    wanted = set(names)

    def _walk(value):
        if isinstance(value, dict):
            return {
                key: (scale_spectral_value(item, factor) if key in wanted else _walk(item))
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [_walk(item) for item in value]
        return value

    return _walk(document)


# Qt dynamic property on line edits that hold spectral world numbers.  A
# workspace restores every line edit's text; these are converted when the
# workspace was written in another spectral unit (see takefits.ui.spectral_fields).
SPECTRAL_FIELD_PROPERTY = "takefits_spectral_world"


def _rescale_annotation_lists(value: Any, factor: float, spectral_axis: int) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key == "marker_specs" and isinstance(item, list):
                result[key] = [rescale_marker_entry(entry, factor) for entry in item]
            elif key == "region_specs" and isinstance(item, list):
                result[key] = [rescale_region_entry(entry, factor, spectral_axis) for entry in item]
            else:
                result[key] = _rescale_annotation_lists(item, factor, spectral_axis)
        return result
    if isinstance(value, list):
        return [_rescale_annotation_lists(item, factor, spectral_axis) for item in value]
    return value


def rescale_workspace_state(workspace_state: Any, factor: float, spectral_axis: int = 2) -> Any:
    """A workspace's ``workspace_state`` with its spectral world numbers scaled.

    Covers the shared cursor, the saved z range, the PV, spectrum and baseline
    windows, the z inputs of integration and channel-map windows and the
    annotations.  Line edits restored from ``ui_state`` are converted when
    they are restored (``SPECTRAL_FIELD_PROPERTY``).
    """
    if factor == 1.0 or not isinstance(workspace_state, dict):
        return workspace_state
    state = copy.deepcopy(workspace_state)
    _scale_keys(state.get("shared_cursor"), ("world_z", "world_native_z"), factor)
    world_ranges = state.get("world_ranges")
    if isinstance(world_ranges, dict):
        _scale_keys(world_ranges.get("z"), ("min", "max"), factor)
    if isinstance(state.get("pv_state"), dict):
        state["pv_state"] = rescale_named_values(state["pv_state"], ("spectral_world", "vel_min", "vel_max"), factor)
    spectrum = state.get("spectrum_state")
    if isinstance(spectrum, dict):
        _scale_keys(spectrum.get("axis_ranges"), ("x_min", "x_max"), factor)
        _scale_keys(spectrum.get("world"), ("z",), factor)
        if isinstance(spectrum.get("active_region"), dict):
            spectrum["active_region"] = rescale_region_entry(spectrum["active_region"], factor, spectral_axis)
    baseline = state.get("baseline_state")
    if isinstance(baseline, dict) and isinstance(baseline.get("ranges"), list):
        for entry in baseline["ranges"]:
            _scale_keys(entry, ("min", "max"), factor)
    geometry = state.get("geometry_state")
    if isinstance(geometry, dict):
        for key in ("integration_windows", "channel_windows"):
            for window in geometry.get(key) or []:
                range_state = window.get("range_state") if isinstance(window, dict) else None
                inputs = range_state.get("inputs") if isinstance(range_state, dict) else None
                if isinstance(inputs, dict):
                    _scale_keys(inputs.get("z"), ("min", "max"), factor)
    if isinstance(state.get("annotation_state"), dict):
        state["annotation_state"] = _rescale_annotation_lists(state["annotation_state"], factor, spectral_axis)
    return state
