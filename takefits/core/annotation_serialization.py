from __future__ import annotations

from typing import Any, Dict, Iterable, List


def _to_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            converted = to_dict()
            if isinstance(converted, dict):
                return dict(converted)
        except Exception:
            return {}
    try:
        return dict(value or {})
    except Exception:
        return {}


def snapshot_marker_specs(marker_manager: Any) -> List[Dict[str, Any]]:
    if marker_manager is None:
        return []
    try:
        marker_manager.refresh_world_coordinates()
    except Exception:
        pass

    specs: List[Dict[str, Any]] = []
    layers = getattr(marker_manager, "_layers", None) or {}
    for plane, layer in layers.items():
        frame_name = str(getattr(layer, "world_frame", "") or "")
        if not frame_name:
            frame_lookup = getattr(marker_manager, "world_frame_for_plane", None)
            if callable(frame_lookup):
                try:
                    frame_name = str(frame_lookup(plane) or "")
                except Exception:
                    frame_name = ""
        markers = getattr(layer, "markers", {}) or {}
        for marker in markers.values():
            try:
                marker_payload = marker.to_state().to_dict()
            except Exception:
                continue
            if frame_name:
                marker_payload["world_frame"] = frame_name
            specs.append(marker_payload)
    specs.sort(key=lambda s: (str(s.get("plane") or ""), str(s.get("id") or "")))
    return specs


def snapshot_region_specs(region_manager: Any, *, default_plane: str = "xy") -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    if region_manager is None:
        return specs

    world_frame = ""
    exported_world_entries: Dict[tuple[str, str, str], Dict[str, Any]] = {}
    exported_world_order: List[Dict[str, Any]] = []
    export_regions = getattr(region_manager, "export_regions_to_dict", None)
    if callable(export_regions):
        try:
            exported_payload = export_regions()
        except Exception:
            exported_payload = None
        if isinstance(exported_payload, dict):
            world_frame = str(exported_payload.get("world_frame") or "")
            exported_regions = list(exported_payload.get("regions") or [])
            for entry in exported_regions:
                if not isinstance(entry, dict):
                    continue
                exported_world_order.append(entry)
                key = (
                    str(entry.get("id") or ""),
                    str(entry.get("plane") or "").lower(),
                    str(entry.get("kind") or "").lower(),
                )
                exported_world_entries.setdefault(key, entry)

    for region_index, region in enumerate(list(getattr(region_manager, "regions", []) or [])):
        try:
            state = region.get_state() if hasattr(region, "get_state") else {}
        except Exception:
            state = {}

        kind_resolver = getattr(region_manager, "_region_kind", None)
        region_type = None
        if callable(kind_resolver):
            try:
                region_type = kind_resolver(region)
            except Exception:
                region_type = None
        region_type = region_type or getattr(region, "__class__", type(region)).__name__.lower()

        cx = cy = 0.0
        if isinstance(state, dict) and "center" in state:
            try:
                cx, cy = state["center"]
            except Exception:
                pass
        elif isinstance(state, dict) and "xy" in state and "width" in state and "height" in state:
            try:
                x0, y0 = state["xy"]
                cx = float(x0) + float(state.get("width", 0.0)) / 2.0
                cy = float(y0) + float(state.get("height", 0.0)) / 2.0
            except Exception:
                pass

        params: Dict[str, float] = {}
        if region_type == "circle":
            if "radius" in state:
                try:
                    params["radius"] = float(state.get("radius", 0.0))
                except Exception:
                    pass
        elif region_type in ("rectangle", "ellipse", "cube"):
            for key in ("width", "height", "angle", "z_min", "z_max"):
                if key in state:
                    try:
                        params[key] = float(state.get(key))
                    except Exception:
                        pass

        plane_name = str(default_plane or "xy").lower()
        plane_for_axes = getattr(region_manager, "_plane_for_axes", None)
        if callable(plane_for_axes):
            try:
                resolved = plane_for_axes(getattr(region, "axes", None))
                if resolved:
                    plane_name = str(resolved).lower()
            except Exception:
                pass

        region_entry: Dict[str, Any] = {
            "id": getattr(region, "region_id", None),
            "type": region_type,
            "plane": plane_name,
            "center_x": float(cx),
            "center_y": float(cy),
            "params": params,
            "label": str(state.get("label") or ""),
            "style": dict(state.get("style") or {}),
        }

        lookup_key = (
            str(region_entry.get("id") or ""),
            str(plane_name or "").lower(),
            str(region_type or "").lower(),
        )
        exported_entry = exported_world_entries.get(lookup_key)
        if exported_entry is None and region_index < len(exported_world_order):
            exported_entry = exported_world_order[region_index]
        if isinstance(exported_entry, dict):
            raw_world = exported_entry.get("world")
            if isinstance(raw_world, dict) and raw_world:
                region_entry["world"] = dict(raw_world)
            if world_frame:
                region_entry["world_frame"] = world_frame

        specs.append(region_entry)

    specs.sort(key=lambda s: (str(s.get("plane") or ""), str(s.get("id") or ""), str(s.get("type") or "")))
    return specs


def build_region_payload_from_specs(
    region_specs: Iterable[Any], *, default_plane: str = "xy"
) -> Dict[str, Any]:
    entries: List[Dict[str, Any]] = []
    world_frame = ""

    for region in region_specs:
        region_dict = _to_dict(region)
        kind = str(region_dict.get("type") or "circle").lower()
        center_x = float(region_dict.get("center_x", 0.0))
        center_y = float(region_dict.get("center_y", 0.0))
        params = dict(region_dict.get("params") or {})
        style = dict(region_dict.get("style") or {})
        label = str(region_dict.get("label") or "")
        plane_name = str(region_dict.get("plane") or default_plane or "xy").lower()
        candidate_frame = str(region_dict.get("world_frame") or "").strip()
        if candidate_frame and not world_frame:
            world_frame = candidate_frame
        world_payload = region_dict.get("world")

        if kind == "circle":
            state_payload: Dict[str, Any] = {
                "center": (center_x, center_y),
                "radius": float(params.get("radius", 0.0)),
                "label": label,
                "style": style,
            }
        else:
            width = float(params.get("width", 0.0))
            height = float(params.get("height", 0.0))
            state_payload = {
                "xy": (center_x - width / 2.0, center_y - height / 2.0),
                "width": width,
                "height": height,
                "angle": float(params.get("angle", 0.0)),
                "label": label,
                "style": style,
            }
            if kind == "cube":
                state_payload["z_min"] = float(params.get("z_min", 0.0))
                state_payload["z_max"] = float(params.get("z_max", 1.0))

        entry: Dict[str, Any] = {
            "id": region_dict.get("id"),
            "kind": kind,
            "plane": plane_name,
            "state": state_payload,
        }
        if isinstance(world_payload, dict) and world_payload:
            entry["world"] = dict(world_payload)
        entries.append(entry)

    payload: Dict[str, Any] = {
        "format": "takefits.region",
        "version": 1,
        "plane": "all",
        "regions": entries,
    }
    if world_frame:
        payload["world_frame"] = world_frame
    return payload


def build_marker_payload_from_specs(marker_specs: Iterable[Any]) -> Dict[str, Any]:
    markers: List[Dict[str, Any]] = []
    world_frame = ""
    for marker in marker_specs:
        marker_dict = _to_dict(marker)
        candidate_frame = str(marker_dict.get("world_frame") or "").strip()
        if candidate_frame and not world_frame:
            world_frame = candidate_frame
        markers.append(marker_dict)

    payload: Dict[str, Any] = {
        "format": "takefits.marker",
        "version": 1,
        "plane": "all",
        "markers": markers,
    }
    if world_frame:
        payload["world_frame"] = world_frame
    return payload
