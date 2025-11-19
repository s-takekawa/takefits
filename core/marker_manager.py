from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union
import math
import numpy as np

from astropy import units as u
from astropy.wcs.utils import proj_plane_pixel_scales
from PyQt6.QtCore import QObject, pyqtSignal, Qt

from .common import Common

from .marker import (
    Marker,
    MarkerId,
    MarkerState,
    PlaneId,
    marker_from_state,
    serialize_marker_states,
    deserialize_marker_states,
    SymbolMarker,
    LineMarker,
    TextMarker,
)


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
        return serialize_marker_states(layer.states(), plane=layer.plane, world_frame=layer.world_frame)

    def import_from_dict(self, payload: Dict[str, object]) -> PlaneId:
        plane, world_frame, states = deserialize_marker_states(payload)
        layer = self.ensure_plane(plane)
        layer.world_frame = world_frame
        # Remove existing markers before importing
        for marker in layer.markers.values():
            marker.remove_from_axes()
        layer.markers.clear()
        for state in states.values():
            marker = marker_from_state(state)
            self._attach_marker(marker)
            layer.markers[marker.marker_id] = marker
            if marker.world is None:
                self._update_marker_world(marker)
        self.markers_changed.emit(plane)
        return plane

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
            if plane not in self._layers:
                continue
            viewer = self._viewer_for_plane(plane)
            if viewer is None:
                continue
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
        common_attr = {
            "xy": "overlay_ax_xy",
            "xz": "overlay_ax_xz",
            "zy": "overlay_ax_zy",
        }.get(base_plane)
        if common_attr:
            return getattr(Common, common_attr, None)
        return None

    def _plane_for_axes(self, axes) -> Optional[PlaneId]:
        if axes is None:
            return None
        for candidate in self._viewer_candidates():
            resolver = getattr(candidate, "marker_plane_for_axes", None)
            if callable(resolver):
                try:
                    plane = resolver(axes)
                except Exception:
                    plane = None
                if plane:
                    return plane
        mapping = {}
        if self.viewer is not None:
            mapping.update({
                getattr(self.viewer, "overlay_ax", None): "xy",
                getattr(self.viewer, "overlay_ax_xz", None): "xz",
                getattr(self.viewer, "overlay_ax_zy", None): "zy",
                getattr(self.viewer, "ax", None): "xy",
                getattr(self.viewer, "ax_xz", None): "xz",
                getattr(self.viewer, "ax_zy", None): "zy",
            })
        mapping.update({
            getattr(Common, "overlay_ax_xy", None): "xy",
            getattr(Common, "overlay_ax_xz", None): "xz",
            getattr(Common, "overlay_ax_zy", None): "zy",
            getattr(Common, "ax_xy", None): "xy",
            getattr(Common, "ax_xz", None): "xz",
            getattr(Common, "ax_zy", None): "zy",
        })
        for ax, plane in mapping.items():
            if ax is not None and axes is ax:
                return plane
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
        pass

    def handle_key_release(self, event) -> None:  # pragma: no cover - GUI
        pass

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

    def _viewer_handles_plane(self, viewer: Optional[object], plane: PlaneId) -> bool:
        if viewer is None:
            return False
        if getattr(viewer, "plane", None) == plane:
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
        main = getattr(Common, "main_window", None)
        if main is not None and id(main) not in seen:
            candidates.append(main)
            seen.add(id(main))
            subwindows = getattr(main, "subwindows", None)
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
        main = getattr(Common, "main_window", None)
        if self._viewer_handles_plane(main, plane):
            return main
        if main is not None:
            for subwindow in getattr(main, "subwindows", []):
                if self._viewer_handles_plane(subwindow, plane):
                    return subwindow
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
        if base_plane in ("xy", "xz", "zy"):
            canvas = getattr(Common, f"canvas_{base_plane}", None)
            if canvas is not None:
                return canvas
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
