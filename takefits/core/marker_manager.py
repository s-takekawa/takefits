from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import math
import numpy as np

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import proj_plane_pixel_scales, skycoord_to_pixel, wcs_to_celestial_frame
from PySide6.QtCore import QObject, Signal as pyqtSignal, Qt



from .marker import (
    Marker,
    MarkerId,
    MarkerState,
    PlaneId,
    MarkerStyle,
    marker_from_state,
    serialize_marker_states,
    deserialize_marker_states,
    SymbolMarker,
    LineMarker,
    TextMarker,
)
from .marker_utils import shared_world_defaults

FRAME_ALIASES = {
    "ICRS": "ICRS",
    "FK5": "FK5",
    "J2000": "FK5",
    "FK4": "FK4",
    "B1950": "FK4",
    "GAL": "GALACTIC",
    "GALACTIC": "GALACTIC",
}


def _normalize_frame_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    upper = str(name).strip().upper()
    return FRAME_ALIASES.get(upper, upper if upper in {"ICRS", "FK5", "FK4", "GALACTIC"} else None)


def _frame_name_from_frame(frame: Optional[object]) -> Optional[str]:
    if frame is None:
        return None
    try:
        candidate = getattr(frame, "name", None)
        if candidate:
            normalized = _normalize_frame_name(candidate)
            if normalized:
                return normalized
    except Exception:
        pass
    return _normalize_frame_name(frame.__class__.__name__)


def _frame_name_from_wcs(wcs: Optional[object]) -> Optional[str]:
    if wcs is None:
        return None
    try:
        frame = wcs_to_celestial_frame(wcs)
    except Exception:
        frame = None
    return _frame_name_from_frame(frame)


def _make_skycoord(coords: np.ndarray, frame_name: Optional[str]) -> Optional[SkyCoord]:
    name = _normalize_frame_name(frame_name)
    if name is None:
        return None
    lon = coords[:, 0] * u.deg
    lat = coords[:, 1] * u.deg
    try:
        if name == "ICRS":
            return SkyCoord(lon, lat, frame="icrs")
        if name == "FK5":
            return SkyCoord(lon, lat, frame="fk5")
        if name == "FK4":
            return SkyCoord(lon, lat, frame="fk4")
        if name == "GALACTIC":
            return SkyCoord(lon, lat, frame="galactic")
    except Exception:
        return None
    return None


def _world_to_pixel(
    world: Optional[Tuple[float, ...]],
    wcs: Optional[object],
    frame_hint: Optional[str],
) -> Optional[Tuple[float, float]]:
    if wcs is None or world is None:
        return None
    arr = np.asarray(world, dtype=float)
    arr = np.atleast_2d(arr)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return None
    coord = _make_skycoord(arr, frame_hint) or _make_skycoord(arr, _frame_name_from_wcs(wcs))
    if coord is None:
        try:
            coord = SkyCoord(arr[:, 0] * u.deg, arr[:, 1] * u.deg)
        except Exception:
            return None

    # Prefer celestial WCS to avoid axes beyond RA/Dec when projecting endpoints.
    candidate_wcs = getattr(wcs, "celestial", None) or wcs
    try:
        px, py = skycoord_to_pixel(coord, candidate_wcs, origin=0)
    except Exception:
        return None
    px_arr = np.asarray(px, dtype=float).ravel()
    py_arr = np.asarray(py, dtype=float).ravel()
    if px_arr.size == 0 or py_arr.size == 0:
        return None
    x_val, y_val = float(px_arr[0]), float(py_arr[0])
    if not (math.isfinite(x_val) and math.isfinite(y_val)):
        return None
    return x_val, y_val


def _world_to_pixel_with_converter(
    world: Optional[Tuple[float, float]],
    plane: PlaneId,
    wcs: Optional[object],
    converter,
    axis_indices: Optional[Tuple[int, int]],
    frame_hint: Optional[str],
    default_world: Optional[List[float]] = None,
) -> Optional[Tuple[float, float]]:
    base_plane = (plane or "xy").lower()

    # Prefer converter for non-celestial plane mixes (e.g., xz/zy) to avoid treating spectral axes as Dec.
    prefer_converter_first = base_plane not in {"xy"}

    def _via_converter() -> Optional[Tuple[float, float]]:
        if world is None or wcs is None:
            return None
        indices = axis_indices
        if indices is None:
            indices = _plane_axis_indices(plane, wcs)
        if indices is None:
            return None
        try:
            naxis = getattr(wcs, "naxis", 0) or 0
            length = max(naxis, max(indices) + 1)
            vector = [0.0] * length
            # Seed with current slice/world values to keep non-plane axes stable.
            vector[0] = float(default_world[0]) if default_world and len(default_world) > 0 else 0.0
            if length > 1:
                vector[1] = float(default_world[1]) if default_world and len(default_world) > 1 else 0.0
            if length > 2:
                vector[2] = float(default_world[2]) if default_world and len(default_world) > 2 else 0.0
            if length > 3:
                vector[3] = float(default_world[3]) if default_world and len(default_world) > 3 else 0.0
            try:
                vector[indices[0]] = float(world[0])
                vector[indices[1]] = float(world[1])
            except Exception:
                return None
            # Prefer raw WCS transform to avoid converter axis assumptions.
            try:
                pix = wcs.wcs_world2pix([vector], 0)[0]
                x_val = float(pix[indices[0]])
                y_val = float(pix[indices[1]])
                if math.isfinite(x_val) and math.isfinite(y_val):
                    return x_val, y_val
            except Exception:
                pass
            if converter is not None:
                result = converter.world_to_pix(*vector)
                if isinstance(result, (list, tuple)) and len(result) >= max(indices) + 1:
                    x_val = float(result[indices[0]])
                    y_val = float(result[indices[1]])
                    if math.isfinite(x_val) and math.isfinite(y_val):
                        return x_val, y_val
        except Exception:
            return None
        return None

    def _via_skycoord() -> Optional[Tuple[float, float]]:
        return _world_to_pixel(world, wcs, frame_hint)

    if prefer_converter_first:
        return _via_converter() or _via_skycoord()
    return _via_skycoord() or _via_converter()


@dataclass
class MarkerLayer:
    """Container for markers within a single plane."""

    plane: PlaneId
    markers: Dict[MarkerId, Marker]  # runtime marker instances
    world_frame: Optional[str] = None

    def states(self) -> Dict[MarkerId, MarkerState]:
        return {marker_id: marker.to_state() for marker_id, marker in self.markers.items()}


class MarkerManager(QObject):
    """Coordinates markers across viewer planes and handles user interactions."""

    markers_changed = pyqtSignal(str)
    selection_changed = pyqtSignal(object)
    geometry_changed = pyqtSignal(object)

    def __init__(self, viewer: Optional[object] = None) -> None:
        super().__init__(viewer)
        self.viewer = viewer
        self._layers: Dict[PlaneId, MarkerLayer] = {}
        self._selected_marker: Optional[Marker] = None
        self._active_plane: PlaneId = "xy"
        self._hit_tolerance: float = 6.0
        self._drag_markers: List[Marker] = []
        self._primary_drag_marker: Optional[Marker] = None
        self._drag_plane: Optional[PlaneId] = None
        self._drag_start: Optional[Tuple[float, float]] = None
        self._drag_start_pixels: List[Tuple[float, ...]] = []
        self._is_dragging: bool = False
        self._drag_handle: Optional[str] = None
        self._drag_handle_anchor: Optional[Tuple[float, float]] = None
        self._drag_canvas = None
        self._pending_placement: Optional[Dict[str, object]] = None
        self._placement_continuous: bool = False
        self._handle_cursor = Qt.CursorShape.CrossCursor
        self._shift_active = False
        self._rotation_mode = False
        self._rotation_center: Optional[Tuple[float, float]] = None
        self._rotation_reference_angle: Optional[float] = None
        self._rotation_targets: List[Marker] = []
        self._rotation_initial_angles: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Plane/layer bookkeeping
    def ensure_plane(self, plane: PlaneId) -> MarkerLayer:
        layer = self._layers.get(plane)
        if layer is None:
            layer = MarkerLayer(plane=plane, markers={})
            self._layers[plane] = layer
        return layer

    def set_active_plane(self, plane: PlaneId) -> None:
        if plane == self._active_plane:
            return
        previous_marker = self._selected_marker
        self._active_plane = plane
        if self._pending_placement and self._pending_placement.get("plane") is not None:
            self._pending_placement["plane"] = plane
        if previous_marker is None:
            self.selection_changed.emit(None)
            return
        if previous_marker.plane != plane:
            self._selected_marker = None
            self.selection_changed.emit(None)
            return
        self.selection_changed.emit(previous_marker)

    def active_plane(self) -> PlaneId:
        return self._active_plane

    # ------------------------------------------------------------------
    # Marker CRUD
    def add_marker(self, marker: Marker) -> Marker:
        layer = self.ensure_plane(marker.plane)
        layer.markers[marker.marker_id] = marker
        self._attach_marker(marker)
        if marker.world is None:
            self._update_marker_world(marker)
        self.markers_changed.emit(marker.plane)
        return marker

    def create_symbol_marker(
        self,
        plane: PlaneId,
        pixel: Tuple[float, float],
        *,
        symbol: str = "o",
        **kwargs,
    ) -> SymbolMarker:
        marker = SymbolMarker(plane, pixel, symbol=symbol, **kwargs)
        self.add_marker(marker)
        return marker

    def create_line_marker(
        self,
        plane: PlaneId,
        pixel: Tuple[float, float],
        *,
        length: float = 10.0,
        angle_deg: float = 0.0,
        unit: str = "pixel",
        style_mode: str = "solid",
        **kwargs,
    ) -> LineMarker:
        marker = LineMarker(
            plane,
            pixel,
            length=length,
            angle_deg=angle_deg,
            unit=unit,
            style_mode=style_mode,
            **kwargs,
        )
        self.add_marker(marker)
        return marker

    def create_text_marker(
        self,
        plane: PlaneId,
        pixel: Tuple[float, float],
        *,
        text: str = "",
        **kwargs,
    ) -> TextMarker:
        marker = TextMarker(plane, pixel, text=text, **kwargs)
        self.add_marker(marker)
        return marker

    def remove_marker(self, marker_id: MarkerId, plane: Optional[PlaneId] = None) -> None:
        target_planes: Iterable[PlaneId]
        if plane is not None:
            target_planes = (plane,)
        else:
            target_planes = self._layers.keys()

        for plane_id in target_planes:
            layer = self._layers.get(plane_id)
            if layer is None:
                continue
            marker = layer.markers.pop(marker_id, None)
            if marker is None:
                continue
            marker.remove_from_axes()
            if marker is self._selected_marker:
                self._selected_marker = None
                self.selection_changed.emit(None)
            self.markers_changed.emit(plane_id)

    def clear_plane(self, plane: PlaneId) -> None:
        layer = self._layers.pop(plane, None)
        if layer is None:
            return
        for marker in layer.markers.values():
            marker.remove_from_axes()
        if self._selected_marker and self._selected_marker.plane == plane:
            self._selected_marker = None
            self.selection_changed.emit(None)
        self.markers_changed.emit(plane)

    def markers_for_plane(self, plane: PlaneId) -> List[Marker]:
        layer = self._layers.get(plane)
        if layer is None:
            return []
        return list(layer.markers.values())

    def marker_for_id(self, marker_id: MarkerId, plane: Optional[PlaneId] = None) -> Optional[Marker]:
        if plane is not None:
            layer = self._layers.get(plane)
            if layer is None:
                return None
            return layer.markers.get(marker_id)
        for layer in self._layers.values():
            marker = layer.markers.get(marker_id)
            if marker is not None:
                return marker
        return None

    def viewer_for_plane(self, plane: PlaneId):
        """Expose viewer lookup so external panels can resolve converters safely."""
        return self._viewer_for_plane(plane)

    def update_marker_pixel(self, marker: Marker, pixel: Tuple[float, float]) -> None:
        """Update marker pixel coordinates and keep world coordinates in sync."""
        if marker is None:
            return
        layer = self._layers.get(marker.plane)
        if layer is not None and marker.marker_id not in layer.markers:
            layer.markers[marker.marker_id] = marker
        marker.apply_new_pixel(pixel)
        self._update_marker_world(marker)
        self.geometry_changed.emit(marker)
        self.markers_changed.emit(marker.plane)
        self.redraw_plane(marker.plane)

    def refresh_world_coordinates(self, plane: Optional[PlaneId] = None) -> None:
        """Recompute world coordinates for markers on the given plane(s)."""
        target_planes: Iterable[PlaneId]
        if plane is None:
            target_planes = tuple(self._layers.keys())
        else:
            target_planes = (plane,)
        for plane_id in target_planes:
            layer = self._layers.get(plane_id)
            if layer is None:
                continue
            viewer = self._viewer_for_plane(plane_id)
            wcs = getattr(viewer, "wcs", None)
            frame_name = _frame_name_from_wcs(wcs)
            if frame_name:
                layer.world_frame = frame_name
            for marker in layer.markers.values():
                self._update_marker_world(marker)

    # Placement helpers ------------------------------------------------
    def begin_placement(
        self,
        kind: str,
        plane: Optional[PlaneId],
        *,
        continuous: bool = False,
        **kwargs,
    ) -> None:
        """Prepare to create markers on subsequent clicks."""
        self._pending_placement = {"kind": kind, "plane": plane, "kwargs": kwargs}
        self._placement_continuous = bool(continuous)
        if plane is not None:
            self.set_active_plane(plane)

    def cancel_placement(self) -> None:
        self._pending_placement = None
        self._placement_continuous = False
        canvases: List[object] = []
        planes_to_check = []
        active = getattr(self, "_active_plane", None)
        if active:
            planes_to_check.append(active)
        planes_to_check.extend(p for p in ("xy", "xz", "zy") if p not in planes_to_check)
        for plane in planes_to_check:
            canvas = self._canvas_for_plane(plane)
            if canvas is not None and canvas not in canvases:
                canvases.append(canvas)
        extra_canvas = getattr(self.viewer, "canvas", None) if self.viewer is not None else None
        if extra_canvas is not None and extra_canvas not in canvases:
            canvases.append(extra_canvas)
        for canvas in canvases:
            try:
                canvas.setCursor(Qt.CursorShape.ArrowCursor)
            except Exception:
                continue

    def pending_placement(self) -> bool:
        return self._pending_placement is not None

    def selected_marker(self) -> Optional[Marker]:
        return self._selected_marker

    def select_marker(self, marker: Optional[Marker], quiet: bool = False) -> None:
        if marker is self._selected_marker:
            return
        self._selected_marker = marker
        if not quiet:
            self.selection_changed.emit(marker)

    # ------------------------------------------------------------------
    # Serialization helpers
    def export_plane_to_dict(self, plane: PlaneId) -> Dict[str, object]:
        layer = self._layers.get(plane)
        if layer is None:
            layer = MarkerLayer(plane=plane, markers={})
        self.refresh_world_coordinates(plane)
        # Always prefer the current viewer's frame; override stale cached frame.
        current_frame = self.world_frame_for_plane(plane)
        if current_frame:
            layer.world_frame = current_frame
        world_frame = layer.world_frame or current_frame
        states = layer.states()
        viewer = self._viewer_for_plane(plane)
        format_pix = getattr(viewer, "format_pix", None) if viewer else None
        wcs = getattr(viewer, "wcs", None) if viewer else None
        base_plane = self._base_plane_for(plane)

        # Attach world endpoints for lines so orientation survives frame changes on load.
        if format_pix is not None and wcs is not None:
            for marker_id, state in list(states.items()):
                marker = layer.markers.get(marker_id)
                if not isinstance(marker, LineMarker):
                    continue
                try:
                    start, end = marker._endpoints()
                    wx0, wy0 = format_pix.pix_to_wcs(wcs, start[0], start[1], base_plane)
                    wx1, wy1 = format_pix.pix_to_wcs(wcs, end[0], end[1], base_plane)
                    meta = dict(state.metadata)
                    meta["pixel_endpoints"] = [(float(start[0]), float(start[1])), (float(end[0]), float(end[1]))]
                    meta["world_endpoints"] = [(float(wx0), float(wy0)), (float(wx1), float(wy1))]
                    state.metadata = meta
                except Exception:
                    continue

        return serialize_marker_states(states, plane=layer.plane, world_frame=world_frame)

    def import_from_dict(self, payload: Dict[str, object]) -> PlaneId:
        plane, world_frame, states = deserialize_marker_states(payload)
        remapper = getattr(self.viewer, "remap_loaded_marker_state", None)
        host_viewer = self.viewer

        # First pass: determine target planes per state.
        targets_by_state: List[Tuple[MarkerState, List[str]]] = []
        planes_to_clear: set[str] = set()
        first_target_plane: Optional[str] = None
        for state in states.values():
            mapped: Optional[Iterable[str]] = None
            if callable(remapper):
                try:
                    mapped = remapper(state, source_plane=plane, world_frame=world_frame)
                except Exception:
                    mapped = None
            target_planes = list(dict.fromkeys(mapped)) if mapped else [state.plane or plane]
            # Keep only planes the host viewer can display.
            supported_targets: List[str] = []
            for tp in target_planes:
                if self._viewer_handles_plane(host_viewer, tp):
                    supported_targets.append(tp)
            targets_by_state.append((state, supported_targets))
            if supported_targets:
                planes_to_clear.update(supported_targets)
                if first_target_plane is None:
                    first_target_plane = supported_targets[0]

        if not planes_to_clear:
            raise ValueError(f"No compatible marker planes for file plane '{plane}'.")

        # Clear existing markers on affected planes.
        for target_plane in planes_to_clear:
            self.clear_plane(target_plane)

        planes_changed: set[str] = set()
        primary_plane = plane
        imported_any = False

        for state, target_planes in targets_by_state:
            if not target_planes:
                continue
            for idx, target_plane in enumerate(target_planes):
                state_copy = state
                if len(target_planes) > 1 or idx > 0:
                    # Clone to keep marker ids unique across planes
                    state_copy = MarkerState.from_dict(state.to_dict())
                state_copy.plane = target_plane
                layer = self.ensure_plane(target_plane)
                if world_frame and not layer.world_frame:
                    layer.world_frame = world_frame
                
                # Prepare defaults
                defaults = self._shared_world_defaults()

                viewer = self._viewer_for_plane(target_plane)
                target_wcs = getattr(viewer, "wcs", None)
                converter = getattr(viewer, "converter", None)
                axis_indices = self._plane_axis_indices(target_plane, target_wcs)
                # If line markers carry world endpoints, re-derive geometry in target frame.
                resolved_from_endpoints = False
                if state_copy.kind == LineMarker.kind and target_wcs is not None:
                    base_meta = state_copy.metadata if isinstance(state_copy.metadata, dict) else {}
                    raw_length_val = base_meta.get("length_source", base_meta.get("length"))
                    unit_val = base_meta.get("unit", "pixel")
                    raw_length_px = None
                    if raw_length_val is not None:
                        try:
                            raw_length_px = self._length_pixels_from_unit(raw_length_val, unit_val, target_plane)
                        except Exception:
                            raw_length_px = None
                    endpoints = state_copy.metadata.get("world_endpoints") if isinstance(state_copy.metadata, dict) else None
                    if endpoints and len(endpoints) == 2:
                        try:
                            px0 = _world_to_pixel_with_converter(endpoints[0], target_plane, target_wcs, converter, axis_indices, world_frame, defaults)
                            px1 = _world_to_pixel_with_converter(endpoints[1], target_plane, target_wcs, converter, axis_indices, world_frame, defaults)
                        except Exception:
                            px0 = px1 = None
                        if px0 is not None and px1 is not None:
                            sx, sy = px0
                            ex, ey = px1
                            cx, cy = (sx + ex) / 2.0, (sy + ey) / 2.0
                            dx, dy = ex - sx, ey - sy
                            length_from_endpoints = float(math.hypot(dx, dy))
                            angle_deg = float(math.degrees(math.atan2(dy, dx)))
                            if length_from_endpoints < 1e-6:
                                angle_deg = float(base_meta.get("angle_deg", angle_deg))
                            # Prefer the original length converted to target pixels if available;
                            # fall back to the projected endpoint length.
                            length_px_final = raw_length_px if raw_length_px is not None else length_from_endpoints
                            meta = dict(base_meta)
                            if length_px_final is None:
                                length_px_final = length_from_endpoints
                            # If projection jumps far from the stored center, treat as bad and fall back.
                            stored_center = state_copy.pixel if state_copy.pixel else (cx, cy)
                            delta_center = math.hypot(cx - stored_center[0], cy - stored_center[1])
                            length_guard = max(length_px_final or 0.0, length_from_endpoints)
                            if delta_center > max(length_guard * 3.0, 1000.0):
                                resolved_from_endpoints = False
                                state_copy.pixel = stored_center
                            else:
                                meta["length"] = length_px_final
                                if raw_length_val is not None:
                                    meta["length_source"] = raw_length_val
                                else:
                                    meta["length_source"] = length_px_final
                                meta["angle_deg"] = angle_deg
                                state_copy.metadata = meta
                                state_copy.pixel = (cx, cy)
                                state_copy.world = None
                                resolved_from_endpoints = True
                    if not resolved_from_endpoints:
                        pix_endpoints = base_meta.get("pixel_endpoints")
                        if pix_endpoints and len(pix_endpoints) == 2:
                            try:
                                sx, sy = float(pix_endpoints[0][0]), float(pix_endpoints[0][1])
                                ex, ey = float(pix_endpoints[1][0]), float(pix_endpoints[1][1])
                                cx, cy = (sx + ex) / 2.0, (sy + ey) / 2.0
                                dx, dy = ex - sx, ey - sy
                                length_from_pix = float(math.hypot(dx, dy))
                                angle_deg = float(math.degrees(math.atan2(dy, dx)))
                                meta = dict(base_meta)
                                length_px_final = raw_length_px if raw_length_px is not None else length_from_pix
                                if length_px_final is None or length_px_final <= 0.0:
                                    length_px_final = length_from_pix
                                meta["length"] = length_px_final
                                if raw_length_val is not None:
                                    meta["length_source"] = raw_length_val
                                else:
                                    meta["length_source"] = meta.get("length", length_from_pix)
                                meta["angle_deg"] = angle_deg
                                state_copy.metadata = meta
                                state_copy.pixel = (cx, cy)
                                state_copy.world = None
                                resolved_from_endpoints = True
                            except Exception:
                                resolved_from_endpoints = False

                if not resolved_from_endpoints:
                    pixel_override = _world_to_pixel_with_converter(state_copy.world, target_plane, target_wcs, converter, axis_indices, world_frame, defaults)
                    if pixel_override is not None:
                        state_copy.pixel = tuple(pixel_override)
                        state_copy.world = None  # Recompute in the target frame for consistency
                    if state_copy.kind == LineMarker.kind:
                        meta = state_copy.metadata if isinstance(state_copy.metadata, dict) else {}
                        raw_length = meta.get("length_source", meta.get("length"))
                        unit = meta.get("unit", "pixel")
                        angle = meta.get("angle_deg", 0.0)
                        if raw_length is not None:
                            length_px = self._length_pixels_from_unit(raw_length, unit, target_plane)
                            meta = dict(meta)
                            meta["length"] = length_px
                            meta["length_source"] = raw_length
                            meta["angle_deg"] = angle
                            state_copy.metadata = meta

                marker = marker_from_state(state_copy)
                if isinstance(marker, LineMarker):
                    meta = state_copy.metadata if isinstance(state_copy.metadata, dict) else {}
                    raw_length = meta.get("length_source", meta.get("length"))
                    unit = meta.get("unit", "pixel")
                    angle_val = meta.get("angle_deg", marker.angle_deg)
                    style_mode = meta.get("style_mode", getattr(marker, "style_mode", "solid"))
                    length_px = meta.get("length")
                    if length_px is None or not math.isfinite(length_px) or length_px <= 0.0:
                        if raw_length is not None:
                            try:
                                length_px = self._length_pixels_from_unit(raw_length, unit, target_plane)
                            except Exception:
                                length_px = None
                    if length_px is None or not math.isfinite(length_px) or length_px <= 0.0:
                        length_px = getattr(marker, "length", 10.0) or 10.0
                    marker.set_length(length_px, source_value=raw_length if raw_length is not None else length_px)
                    marker.set_angle(angle_val)
                    marker.set_style_mode(style_mode)
                    try:
                        marker._on_geometry_changed()
                    except Exception:
                        pass

                self._attach_marker(marker)
                layer.markers[marker.marker_id] = marker
                if marker.world is None:
                    self._update_marker_world(marker)
                planes_changed.add(target_plane)
                if primary_plane == plane and first_target_plane:
                    primary_plane = first_target_plane
                imported_any = True

        if planes_changed:
            for plane_id in planes_changed:
                self.markers_changed.emit(plane_id)
        else:
            self.markers_changed.emit(plane)
        if not imported_any:
            raise ValueError(f"No markers imported; plane '{plane}' is incompatible with this viewer.")
        return primary_plane

    def _shared_world_defaults(self) -> List[float]:
        return shared_world_defaults(getattr(self, "viewer", None))

    def world_frame_for_plane(self, plane: PlaneId) -> Optional[str]:
        layer = self._layers.get(plane)
        if layer is not None and layer.world_frame:
            return layer.world_frame
        viewer = self._viewer_for_plane(plane)
        wcs = getattr(viewer, "wcs", None)
        frame = _frame_name_from_wcs(wcs)
        if layer is not None and frame:
            layer.world_frame = frame
        return frame

    # ------------------------------------------------------------------
    # Matplotlib interaction hooks (to be fleshed out later)
    def _attach_marker(self, marker: Marker) -> None:
        ax = self._axes_for_plane(marker.plane)
        if ax is None:
            return
        marker.add_to_axes(ax)

    def redraw_plane(self, plane: PlaneId) -> None:
        viewer = self._viewer_for_plane(plane)
        if viewer is not None:
            redraw_overlay = getattr(viewer, "redraw_overlay_for_plane", None)
            if callable(redraw_overlay):
                try:
                    redraw_overlay(plane)
                    return
                except Exception:
                    pass
            redraw_main = getattr(viewer, "redraw_main_overlay_and_blit", None)
            if callable(redraw_main):
                try:
                    redraw_main()
                    return
                except Exception:
                    pass
            canvas = getattr(viewer, "canvas", None)
            if canvas is not None:
                canvas.draw_idle()
                return
        ax = self._axes_for_plane(plane)
        if ax is not None:
            fig = getattr(ax, "figure", None)
            if fig is not None:
                canvas = getattr(fig, "canvas", None)
                if canvas is not None:
                    canvas.draw_idle()

    def redraw_planes(self, planes: Iterable[PlaneId]) -> None:
        seen: set[str] = set()
        for plane in planes:
            if plane in seen:
                continue
            seen.add(plane)
            self.redraw_plane(plane)

    def _axes_for_plane(self, plane: PlaneId):
        viewer = self._viewer_for_plane(plane)
        base_plane = self._base_plane_for(plane)
        if viewer is not None:
            custom_lookup = getattr(viewer, "marker_axes_for_plane", None)
            if callable(custom_lookup):
                try:
                    axes = custom_lookup(plane)
                    if axes is not None:
                        return axes
                except Exception:
                    pass
            attr_candidates = []
            if base_plane == "xy":
                attr_candidates = ["overlay_ax"]
            elif base_plane == "xz":
                attr_candidates = ["overlay_ax_xz", "overlay_ax"]
            elif base_plane == "zy":
                attr_candidates = ["overlay_ax_zy", "overlay_ax"]
            else:
                attr_candidates = ["overlay_ax"]
            for attr in attr_candidates:
                viewer_axes = getattr(viewer, attr, None)
                if viewer_axes is not None:
                    return viewer_axes
        return None

    def _plane_for_axes(self, axes) -> Optional[PlaneId]:
        if axes is None:
            return None
        candidates = self._viewer_candidates()
        for candidate in candidates:
            resolver = getattr(candidate, "marker_plane_for_axes", None)
            if callable(resolver):
                try:
                    plane = resolver(axes)
                except Exception:
                    plane = None
                if plane:
                    return plane

        mapping = {}
        def _bind(candidate_axes, plane_name):
            if candidate_axes is not None:
                mapping[candidate_axes] = plane_name

        for candidate in candidates:
            plane_name = str(getattr(candidate, "plane", "") or "").lower()
            if plane_name:
                _bind(getattr(candidate, "overlay_ax", None), plane_name)
                _bind(getattr(candidate, "ax", None), plane_name)
            _bind(getattr(candidate, "overlay_ax_xz", None), "xz")
            _bind(getattr(candidate, "overlay_ax_zy", None), "zy")
            _bind(getattr(candidate, "ax_xz", None), "xz")
            _bind(getattr(candidate, "ax_zy", None), "zy")

        for candidate_axes, plane_name in mapping.items():
            if axes is candidate_axes:
                return plane_name
        return None

    def _plane_axis_indices(self, plane: Optional[PlaneId], wcs) -> Optional[Tuple[int, int]]:
        """Resolve which WCS axes correspond to the marker plane."""
        if wcs is None:
            return None
        viewer = self._viewer_for_plane(plane) if plane is not None else self.viewer
        if viewer is not None:
            resolver = getattr(viewer, "marker_axis_indices", None)
            if callable(resolver):
                try:
                    indices = resolver(plane)
                except Exception:
                    indices = None
                if indices:
                    try:
                        first, second = indices[:2]
                        return (int(first), int(second))
                    except Exception:
                        pass
        base_plane = self._base_plane_for(plane or "xy")
        naxis = getattr(wcs, "naxis", 0) or 0
        if base_plane == "xy" and naxis >= 2:
            return (0, 1)
        if base_plane == "xz" and naxis >= 3:
            return (0, 2)
        if base_plane == "zy" and naxis >= 3:
            return (2, 1)
        return None

    def _extract_data_coords(self, event, axes_override=None) -> Tuple[Optional[float], Optional[float]]:
        xdata = getattr(event, "xdata", None)
        ydata = getattr(event, "ydata", None)
        axes = axes_override or getattr(event, "inaxes", None)
        if xdata is not None and ydata is not None:
            return float(xdata), float(ydata)
        if axes is not None and getattr(event, "x", None) is not None:
            try:
                xdata, ydata = axes.transData.inverted().transform((event.x, event.y))
                return float(xdata), float(ydata)
            except Exception:
                pass
        return None, None

    def _hit_tolerance_for_axes(self, axes) -> float:
        if axes is None:
            return self._hit_tolerance
        try:
            p0 = axes.transData.inverted().transform((0, 0))
            p1 = axes.transData.inverted().transform((self._hit_tolerance, self._hit_tolerance))
            return max(abs(p1[0] - p0[0]), abs(p1[1] - p0[1]))
        except Exception:
            return self._hit_tolerance

    def _find_marker_at(self, plane: PlaneId, x: float, y: float, tolerance: float) -> Optional[Marker]:
        layer = self._layers.get(plane)
        if layer is None:
            return None
        for marker in reversed(list(layer.markers.values())):
            try:
                if marker.contains_pixel(x, y, tolerance):
                    return marker
            except Exception:
                continue
        return None

    # ------------------------------------------------------------------
    # Event entry points (stubs for later detailed implementation)
    def handle_press(self, event) -> None:  # pragma: no cover - GUI
        if getattr(event, "button", None) not in (1,):  # Left button only
            return
        axes = getattr(event, "inaxes", None)
        if axes is None:
            return
        plane = self._plane_for_axes(axes)
        if plane is None:
            return
        xdata, ydata = self._extract_data_coords(event)
        if xdata is None or ydata is None:
            return
        tolerance = self._hit_tolerance_for_axes(axes)
        marker = self._find_marker_at(plane, xdata, ydata, tolerance)
        handle_name: Optional[str] = None
        if marker is not None:
            handle_name = self._find_handle_hit(marker, xdata, ydata, tolerance)

        if marker is None and self._pending_placement:
            config_plane = self._pending_placement.get("plane")
            if config_plane is None or config_plane == plane:
                created_markers = self._create_marker_from_config(plane, (xdata, ydata))
                if not created_markers:
                    return

                primary_marker = created_markers[0]
                self._attach_marker_to_plane(primary_marker, plane, ensure_existing=True)
                self.set_active_plane(plane)

                panel = getattr(self.viewer, "marker_panel", None)
                if panel and hasattr(panel, "set_selection_by_id"):
                    panel.set_selection_by_id([m.marker_id for m in created_markers])
                    if isinstance(primary_marker, TextMarker):
                        focus_text_entry = getattr(panel, "focus_text_entry_for_marker", None)
                        if callable(focus_text_entry):
                            try:
                                focus_text_entry(primary_marker)
                            except Exception:
                                pass
                else:
                    self.select_marker(primary_marker)

                if not self._placement_continuous:
                    self.cancel_placement()

                planes_to_redraw = {m.plane for m in created_markers}
                self.redraw_planes(planes_to_redraw)
                return
        if marker is None:
            self.select_marker(None)
            return

        self.set_active_plane(plane)

        panel = getattr(self.viewer, "marker_panel", None)
        current_selection = []
        if panel and hasattr(panel, "_selected_markers"):
            current_selection = panel._selected_markers()

        is_part_of_selection = any(m.marker_id == marker.marker_id for m in current_selection)

        if not is_part_of_selection:
            self.select_marker(marker)
            self._drag_markers = [marker]
        else:
            if self._selected_marker is not marker:
                self.select_marker(marker, quiet=True)
            self._drag_markers = current_selection

        self._primary_drag_marker = marker
        self._drag_plane = plane
        self._drag_start = (xdata, ydata)
        self._drag_start_pixels = [m.pixel for m in self._drag_markers]
        self._is_dragging = True

        # Ensure drag starts from a clean background snapshot for this plane.
        viewer = self._viewer_for_plane(plane)
        if viewer is not None:
            refresh_bg = getattr(viewer, "_refresh_overlay_background", None)
            if callable(refresh_bg):
                try:
                    refresh_bg(plane)
                except Exception:
                    pass

        if handle_name is not None:
            anchor = self._handle_anchor_for(marker, handle_name)
            if anchor is not None:
                self._drag_handle = handle_name
                self._drag_handle_anchor = anchor
            else:
                self._drag_handle = None
                self._drag_handle_anchor = None
        else:
            self._drag_handle = None
            self._drag_handle_anchor = None
        canvas = self._canvas_for_plane(plane, event)
        if canvas is not None:
            try:
                if handle_name is None:
                    shape = Qt.CursorShape.ClosedHandCursor
                else:
                    shape = self._handle_cursor
                canvas.setCursor(shape)
            except Exception:
                pass
        self._drag_canvas = canvas
        self._rotation_mode = False
        self._rotation_targets = []
        self._rotation_initial_angles = {}
        self._rotation_center = None
        self._rotation_reference_angle = None
        if self._event_has_shift(event) or self._shift_active:
            self._begin_rotation_mode(xdata, ydata)

    def handle_release(self, event) -> None:  # pragma: no cover - GUI
        if not self._is_dragging:
            return

        plane = self._drag_plane
        if self._drag_markers and plane is not None:
            for marker in self._drag_markers:
                self._update_marker_world(marker)
                self.geometry_changed.emit(marker)

            self.markers_changed.emit(plane)
            self.redraw_plane(plane)

        if self._primary_drag_marker:
            self.select_marker(self._primary_drag_marker)

        self._finish_drag()

    def handle_motion(self, event) -> None:  # pragma: no cover - GUI
        if not self._is_dragging or not self._drag_markers or self._drag_plane is None:
            return
        axes = getattr(event, "inaxes", None)
        plane = self._plane_for_axes(axes) if axes is not None else None
        if plane != self._drag_plane:
            axes = self._axes_for_plane(self._drag_plane)
            plane = self._drag_plane
        if axes is None or plane != self._drag_plane:
            return
        xdata, ydata = self._extract_data_coords(event, axes_override=axes)
        if xdata is None or ydata is None or self._drag_start is None or not self._drag_start_pixels:
            return

        if self._drag_handle and isinstance(self._primary_drag_marker, LineMarker):
            primary_marker = self._primary_drag_marker
            old_length = primary_marker.length
            old_angle = primary_marker.angle_deg

            anchor = self._drag_handle_anchor
            if anchor is None:
                return

            if self._drag_handle == "start":
                start = (xdata, ydata)
                end = anchor
            else:
                start = anchor
                end = (xdata, ydata)

            primary_marker.update_from_endpoints(start, end)

            delta_length = primary_marker.length - old_length
            delta_angle = primary_marker.angle_deg - old_angle

            if delta_angle > 180:
                delta_angle -= 360
            if delta_angle < -180:
                delta_angle += 360

            for marker in self._drag_markers:
                if not isinstance(marker, LineMarker) or marker is primary_marker:
                    continue

                # Calculate shift in center required to keep one end stationary
                old_angle_rad = np.deg2rad(marker.angle_deg)
                old_half_len = marker.length / 2.0
                old_dx = old_half_len * np.cos(old_angle_rad)
                old_dy = old_half_len * np.sin(old_angle_rad)

                new_angle = marker.angle_deg + delta_angle
                new_length = marker.length + delta_length
                new_angle_rad = np.deg2rad(new_angle)
                new_half_len = new_length / 2.0
                new_dx = new_half_len * np.cos(new_angle_rad)
                new_dy = new_half_len * np.sin(new_angle_rad)

                if self._drag_handle == 'end':  # 'start' handle is stationary
                    center_dx = -old_dx + new_dx
                    center_dy = -old_dy + new_dy
                else:  # 'start' handle is being dragged, 'end' is stationary
                    center_dx = old_dx - new_dx
                    center_dy = old_dy - new_dy

                # Apply new properties
                marker.set_length(new_length)
                marker.set_angle(new_angle)
                marker.apply_new_pixel((marker.pixel[0] + center_dx, marker.pixel[1] + center_dy))

            for marker in self._drag_markers:
                if isinstance(marker, LineMarker):
                    marker.length_source_value = self._length_in_unit(marker, marker.length)
                    self._update_marker_world(marker)
                    self.geometry_changed.emit(marker)

            if self._drag_plane is not None:
                self.markers_changed.emit(self._drag_plane)
            self.redraw_plane(self._drag_plane)
            return

        if self._rotation_mode and self._rotation_center is not None and self._rotation_targets:
            self._apply_rotation_drag(xdata, ydata)
            return

        dx = xdata - self._drag_start[0]
        dy = ydata - self._drag_start[1]

        for i, marker in enumerate(self._drag_markers):
            start_pixel = self._drag_start_pixels[i]
            new_pixel = list(start_pixel)
            if new_pixel:
                new_pixel[0] += dx
            if len(new_pixel) > 1:
                new_pixel[1] += dy
            marker.apply_new_pixel(tuple(new_pixel))
            self._update_marker_world(marker)

        self.redraw_plane(self._drag_plane)

    def handle_key_press(self, event) -> None:  # pragma: no cover - GUI
        if self._event_has_shift(event):
            self._shift_active = True

    def handle_key_release(self, event) -> None:  # pragma: no cover - GUI
        if self._event_has_shift(event):
            self._shift_active = False

    def _finish_drag(self) -> None:
        if self._drag_canvas is not None:
            try:
                self._drag_canvas.setCursor(Qt.CursorShape.ArrowCursor)
            except Exception:
                pass
        self._drag_markers = []
        self._primary_drag_marker = None
        self._drag_plane = None
        self._drag_start = None
        self._drag_start_pixels = []
        self._is_dragging = False
        self._drag_handle = None
        self._drag_handle_anchor = None
        self._drag_canvas = None
        self._rotation_mode = False
        self._rotation_center = None
        self._rotation_reference_angle = None
        self._rotation_targets = []
        self._rotation_initial_angles = {}

    def is_dragging(self) -> bool:
        return self._is_dragging

    def handle_hover(self, event) -> None:  # pragma: no cover - GUI
        if self._is_dragging:
            return
        axes = getattr(event, "inaxes", None)
        plane = self._plane_for_axes(axes) if axes is not None else None
        canvas = self._canvas_for_plane(plane, event)
        if canvas is None:
            return
        if not self.pending_placement():
            try:
                canvas.setCursor(Qt.CursorShape.ArrowCursor)
            except Exception:
                pass
            return
        if plane is None or axes is None:
            canvas.setCursor(Qt.CursorShape.ArrowCursor)
            return
        xdata, ydata = self._extract_data_coords(event)
        if xdata is None or ydata is None:
            canvas.setCursor(Qt.CursorShape.ArrowCursor)
            return
        tolerance = self._hit_tolerance_for_axes(axes)
        marker = self._find_marker_at(plane, xdata, ydata, tolerance)
        if marker is not None:
            handle_hit = self._find_handle_hit(marker, xdata, ydata, tolerance)
            try:
                if handle_hit is not None and isinstance(marker, LineMarker):
                    canvas.setCursor(self._handle_cursor)
                else:
                    canvas.setCursor(Qt.CursorShape.OpenHandCursor)
            except Exception:
                pass
        else:
            try:
                canvas.setCursor(Qt.CursorShape.ArrowCursor)
            except Exception:
                pass

    def _event_has_shift(self, event) -> bool:
        key = getattr(event, "key", None)
        if key is None:
            return False
        try:
            key_str = str(key).lower()
        except Exception:
            return False
        return "shift" in key_str

    def _begin_rotation_mode(self, xdata: float, ydata: float) -> bool:
        if not self._drag_markers:
            return False
        targets = [m for m in self._drag_markers if isinstance(m, (LineMarker, TextMarker))]
        if not targets:
            return False
        primary = self._primary_drag_marker if isinstance(self._primary_drag_marker, (LineMarker, TextMarker)) else targets[0]
        cx, cy = primary.pixel[:2]
        self._rotation_center = (cx, cy)
        try:
            reference = math.degrees(math.atan2(ydata - cy, xdata - cx))
        except Exception:
            reference = 0.0
        self._rotation_reference_angle = reference
        self._rotation_targets = targets
        self._rotation_initial_angles = {marker.marker_id: self._marker_rotation_value(marker) for marker in targets}
        self._rotation_mode = True
        self._drag_handle = None
        self._drag_handle_anchor = None
        return True

    def _apply_rotation_drag(self, xdata: float, ydata: float) -> None:
        center = self._rotation_center
        if center is None or not self._rotation_targets:
            return
        dx = xdata - center[0]
        dy = ydata - center[1]
        try:
            angle = math.degrees(math.atan2(dy, dx))
        except Exception:
            angle = 0.0
        reference = self._rotation_reference_angle or 0.0
        delta = angle - reference
        while delta > 180.0:
            delta -= 360.0
        while delta < -180.0:
            delta += 360.0
        for marker in self._rotation_targets:
            base = self._rotation_initial_angles.get(marker.marker_id)
            if base is None:
                continue
            self._set_marker_rotation(marker, base + delta)
            self._update_marker_world(marker)
            self.geometry_changed.emit(marker)
        if self._drag_plane is not None:
            self.markers_changed.emit(self._drag_plane)
            self.redraw_plane(self._drag_plane)

    def _marker_rotation_value(self, marker: Marker) -> float:
        if isinstance(marker, LineMarker):
            return float(marker.angle_deg)
        if isinstance(marker, TextMarker):
            return float(getattr(marker.style, "rotation", 0.0) or 0.0)
        return 0.0

    def _set_marker_rotation(self, marker: Marker, angle: float) -> None:
        normalized = angle
        if isinstance(marker, LineMarker):
            marker.set_angle(normalized)
        elif isinstance(marker, TextMarker):
            style_dict = marker.style.to_dict()
            style_dict["rotation"] = normalized
            marker.update_style(MarkerStyle.from_dict(style_dict))

    def _viewer_handles_plane(self, viewer: Optional[object], plane: PlaneId) -> bool:
        if viewer is None:
            return False
        viewer_plane = str(getattr(viewer, "plane", "") or "").lower()
        target_plane = str(plane or "").lower()
        if viewer_plane == target_plane:
            return True
        handler = getattr(viewer, "has_marker_plane", None)
        if callable(handler):
            try:
                return bool(handler(plane))
            except Exception:
                return False
        return False

    def _viewer_candidates(self) -> List[object]:
        candidates: List[object] = []
        seen: set[int] = set()
        if self.viewer is not None:
            candidates.append(self.viewer)
            seen.add(id(self.viewer))
            subwindows = getattr(self.viewer, "subwindows", None)
            if subwindows:
                for subwindow in subwindows:
                    if subwindow is None:
                        continue
                    ident = id(subwindow)
                    if ident not in seen:
                        candidates.append(subwindow)
                        seen.add(ident)

        return candidates

    def _viewer_for_plane(self, plane: PlaneId):
        viewer = self.viewer if self._viewer_handles_plane(self.viewer, plane) else None
        if viewer is not None:
            return viewer
        return None

    def _canvas_for_plane(self, plane: Optional[PlaneId], event=None):
        viewer = self._viewer_for_plane(plane) if plane is not None else None
        if viewer is not None:
            canvas_hook = getattr(viewer, "marker_canvas_for_plane", None)
            if callable(canvas_hook):
                try:
                    canvas = canvas_hook(plane)
                    if canvas is not None:
                        return canvas
                except Exception:
                    pass
            canvas = getattr(viewer, "canvas", None)
            if canvas is not None:
                return canvas
        if event is not None:
            canvas = getattr(event, "canvas", None)
            if canvas is not None:
                return canvas
        base_plane = self._base_plane_for(plane) if plane is not None else None

        if self.viewer is not None:
            return getattr(self.viewer, "canvas", None)
        return None

    def _base_plane_for(self, plane: PlaneId) -> PlaneId:
        viewer = self._viewer_for_plane(plane)
        if viewer is not None:
            resolver = getattr(viewer, "marker_plane_base", None)
            if callable(resolver):
                try:
                    base = resolver(plane)
                    if base:
                        return str(base)
                except Exception:
                    pass
        return plane

    def _update_marker_world(self, marker: Marker) -> None:
        plane = marker.plane
        viewer = self._viewer_for_plane(plane)
        if viewer is None:
            return
        format_pix = getattr(viewer, "format_pix", None)
        wcs = getattr(viewer, "wcs", None)
        if format_pix is None or wcs is None:
            return
        try:
            base_plane = self._base_plane_for(plane)
            wx, wy = format_pix.pix_to_wcs(wcs, marker.pixel[0], marker.pixel[1], base_plane)
        except Exception:
            return
        marker.apply_new_world((wx, wy))
        layer = self._layers.get(plane)
        if layer is None:
            return
        frame_name = _frame_name_from_wcs(wcs)
        if frame_name and not layer.world_frame:
            layer.world_frame = frame_name
        if marker.marker_id not in layer.markers:
            layer.markers[marker.marker_id] = marker

    def _create_marker_from_config(self, plane: PlaneId, pixel: Tuple[float, float]) -> List[Marker]:
        config = self._pending_placement or {}
        kind = config.get("kind", "symbol")
        kwargs = dict(config.get("kwargs") or {})
        style = kwargs.pop("style", None)
        label = kwargs.pop("label", "")
        marker: Marker
        if kind == "line":
            marker = self.create_line_marker(plane, pixel, **kwargs)
        elif kind == "text":
            marker = self.create_text_marker(plane, pixel, **kwargs)
        else:
            marker = self.create_symbol_marker(plane, pixel, **kwargs)
        if style is not None:
            marker.update_style(style)
        if label:
            marker.set_label(str(label))
        self._update_marker_world(marker)

        created_markers = [marker]
        viewer = self._viewer_for_plane(plane)
        if viewer is not None:
            mirror_fn = getattr(viewer, "mirror_marker_creation", None)
            if callable(mirror_fn):
                try:
                    mirrored = mirror_fn(marker, plane, source="primary")
                    if mirrored:
                        created_markers.extend(mirrored)
                except Exception:
                    pass
        return created_markers

    # ------------------------------------------------------------------
    # Blitting support
    def _attach_marker_to_plane(self, marker: Marker, plane: PlaneId, *, ensure_existing: bool = False) -> None:
        target_ax = self._axes_for_plane(plane)
        if target_ax is None:
            return
        if ensure_existing:
            layer = self._layers.get(plane)
            if layer is not None:
                for existing in layer.markers.values():
                    if existing is marker:
                        continue
                    ax = getattr(existing, "artist", None)
                    axes = getattr(ax, "axes", None) if ax is not None else None
                    if axes is not target_ax or ax is None:
                        existing.remove_from_axes()
                        existing.add_to_axes(target_ax)
                        try:
                            existing._on_geometry_changed()
                        except Exception:
                            pass
                        try:
                            existing._on_style_changed()
                        except Exception:
                            pass
        marker.remove_from_axes()
        marker.add_to_axes(target_ax)
        try:
            marker._on_geometry_changed()
        except Exception:
            pass
        try:
            marker._on_style_changed()
        except Exception:
            pass

    def _find_handle_hit(self, marker: Marker, x: float, y: float, tolerance: float) -> Optional[str]:
        handles_fn = getattr(marker, "handles", None)
        if not callable(handles_fn):
            return None
        handles = handles_fn()
        if not handles:
            return None
        best_name = None
        best_dist = float("inf")
        handle_radius = max(tolerance * 0.6, 4.0)
        for name, point in handles.items():
            if point is None or len(point) < 2:
                continue
            px, py = point[:2]
            dist = math.hypot(px - x, py - y)
            if dist <= handle_radius and dist < best_dist:
                best_name = name
                best_dist = dist
        return best_name

    def _handle_anchor_for(self, marker: Marker, handle: str) -> Optional[Tuple[float, float]]:
        handles_fn = getattr(marker, "handles", None)
        if not callable(handles_fn):
            return None
        handles = handles_fn()
        if not handles:
            return None
        if handle == "start":
            return handles.get("end")
        if handle == "end":
            return handles.get("start")
        return None

    def _pixel_scale_deg(self, plane: PlaneId) -> Optional[float]:
        viewer = self._viewer_for_plane(plane)
        if viewer is None:
            return None
        wcs = getattr(viewer, "wcs", None)
        if wcs is None:
            return None
        base_plane = self._base_plane_for(plane)
        axis_indices = None
        resolver = getattr(viewer, "marker_axis_indices", None)
        if callable(resolver):
            try:
                axis_indices = resolver(plane)
            except Exception:
                axis_indices = None
        if not axis_indices:
            default_map = {
                "xy": (0, 1),
                "xz": (0, 2),
                "zy": (2, 1),
            }
            axis_indices = default_map.get(base_plane, (0, 1))
        try:
            scales = proj_plane_pixel_scales(wcs)
            if scales is not None and len(scales) >= 2:
                values: List[float] = []
                for idx in axis_indices:
                    if idx is None or idx < 0 or idx >= len(scales):
                        continue
                    val = scales[idx]
                    try:
                        values.append(abs(val.to_value(u.deg)))
                    except AttributeError:
                        try:
                            values.append(abs(float(val)))
                        except Exception:
                            continue
                if values:
                    avg = sum(values) / len(values)
                    if avg > 0:
                        return float(avg)
        except Exception:
            pass
        try:
            cdelt = getattr(getattr(wcs, "wcs", None), "cdelt", None)
            if cdelt is not None and len(cdelt) > 0:
                values: List[float] = []
                for idx in axis_indices:
                    if idx is None or idx < 0 or idx >= len(cdelt):
                        continue
                    scale = abs(cdelt[idx])
                    if scale > 0:
                        values.append(float(scale))
                if values:
                    avg = sum(values) / len(values)
                    if avg > 0:
                        return avg
        except Exception:
            pass
        return None

    def _length_in_unit(self, marker: LineMarker, length_pixels: float) -> float:
        unit = getattr(marker, "unit", "pixel") or "pixel"
        if unit == "pixel":
            return float(length_pixels)
        scale_deg = self._pixel_scale_deg(marker.plane)
        if not scale_deg:
            return float(length_pixels)
        if unit == "deg":
            return float(length_pixels) * scale_deg
        if unit == "arcmin":
            return float(length_pixels) * scale_deg * 60.0
        if unit == "arcsec":
            return float(length_pixels) * scale_deg * 3600.0
        return float(length_pixels)

    def _length_pixels_from_unit(self, value: float, unit: str, plane: PlaneId) -> float:
        unit = unit or "pixel"
        if unit == "pixel":
            return float(value)
        scale_deg = self._pixel_scale_deg(plane)
        if not scale_deg:
            return float(value)
        if unit == "deg":
            return float(value) / scale_deg
        if unit == "arcmin":
            return float(value) / (scale_deg * 60.0)
        if unit == "arcsec":
            return float(value) / (scale_deg * 3600.0)
        return float(value)
    def prepare_for_background_capture(self, plane: Optional[PlaneId] = None) -> List[object]:
        hidden: List[object] = []
        layers: Iterable[MarkerLayer]
        if plane is not None:
            layer = self._layers.get(plane)
            layers = (layer,) if layer is not None else ()
        else:
            layers = self._layers.values()
        for layer in layers:
            if layer is None:
                continue
            for marker in layer.markers.values():
                artist = getattr(marker, "artist", None)
                if artist is not None:
                    try:
                        if not artist.get_animated():
                            artist.set_animated(True)
                    except Exception:
                        pass
                    if artist.get_visible():
                        artist.set_visible(False)
                        hidden.append(artist)
                label = getattr(marker, "label_artist", None)
                if label is not None and label.get_visible():
                    label.set_visible(False)
                    hidden.append(label)
        return hidden

    def restore_after_background_capture(self, hidden: Iterable[object]) -> None:
        for artist in hidden:
            try:
                artist.set_visible(True)
            except Exception:
                continue

    def draw_markers_for_blit(self, plane: Optional[PlaneId] = None) -> None:
        if plane is not None:
            layers = (self._layers.get(plane),)
        else:
            layers = self._layers.values()
        for layer in layers:
            if layer is None:
                continue
            target_ax = self._axes_for_plane(layer.plane)
            if target_ax is None:
                continue
            for marker in layer.markers.values():
                artist = getattr(marker, "artist", None)
                artist_axes = getattr(artist, "axes", None) if artist is not None else None
                if artist is None or artist_axes is None or artist_axes is not target_ax:
                    marker.remove_from_axes()
                    marker.add_to_axes(target_ax)
                    try:
                        marker._on_geometry_changed()
                    except Exception:
                        pass
                    try:
                        marker._on_style_changed()
                    except Exception:
                        pass
                    artist = getattr(marker, "artist", None)
                    artist_axes = getattr(artist, "axes", None) if artist is not None else None
                if artist is None or artist_axes is None:
                    continue
                try:
                    artist_axes.draw_artist(artist)
                except Exception:
                    pass
                label = getattr(marker, "label_artist", None)
                if label is not None and label.get_visible():
                    label_axes = getattr(label, "axes", None)
                    if label_axes is not target_ax:
                        try:
                            label.remove()
                        except Exception:
                            pass
                        marker.label_artist = None
                        ensure_fn = getattr(marker, "_ensure_label", None)
                        if callable(ensure_fn):
                            ensure_fn(target_ax)
                        label = marker.label_artist
                        label_axes = getattr(label, "axes", None) if label is not None else None
                    if label_axes is not None:
                        try:
                            label_axes.draw_artist(label)
                        except Exception:
                            pass
