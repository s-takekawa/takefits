from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar
import uuid

import numpy as np
import matplotlib as mpl
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.text import Text

from takefits.core.fonts import resolve_mpl_font_family

mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams.setdefault("mathtext.rm", "cm")


MarkerId = str
PlaneId = str


def detach_artist(artist) -> None:
    """Drop ``artist`` from its axes, tolerating an already-detached artist.

    ``Axes.clear()`` -- and therefore ``Figure.clear()`` -- orphans every child
    it drops by setting ``_remove_method`` to ``None``, and Matplotlib then
    answers a later ``artist.remove()`` with ``NotImplementedError('cannot
    remove artist')``.  A second removal instead raises ``ValueError`` from the
    child list.  Marker teardown can legitimately run after a figure has been
    cleared (window close, channel-map relayout), so treat an artist that is
    already off its axes as nothing left to do.
    """
    if artist is None:
        return
    try:
        artist.remove()
    except (NotImplementedError, ValueError):
        pass


@dataclass
class MarkerStyle:
    """Rendering attributes shared across marker types."""

    color: str = "white"
    size: float = 12.0           # marker size or font size depending on type
    edgecolor: Optional[str] = None
    linewidth: float = 1.5
    opacity: float = 1.0
    rotation: float = 0.0        # degrees, used by shapes that support rotation
    marker_symbol: str = "o"     # Matplotlib marker symbol for symbol markers
    font_family: str = "DejaVu Sans"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "color": self.color,
            "size": float(self.size),
            "edgecolor": self.edgecolor,
            "linewidth": float(self.linewidth),
            "opacity": float(self.opacity),
            "rotation": float(self.rotation),
            "marker_symbol": self.marker_symbol,
            "font_family": self.font_family,
        }

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> "MarkerStyle":
        if not payload:
            return cls()
        return cls(
            color=str(payload.get("color", "white")),
            size=float(payload.get("size", 12.0)),
            edgecolor=payload.get("edgecolor"),
            linewidth=float(payload.get("linewidth", 1.5)),
            opacity=float(payload.get("opacity", 1.0)),
            rotation=float(payload.get("rotation", 0.0)),
            marker_symbol=str(payload.get("marker_symbol", "o")),
            font_family=str(payload.get("font_family", "DejaVu Sans")),
        )


@dataclass
class MarkerState:
    """Serialized representation of a marker."""

    marker_id: MarkerId
    plane: PlaneId
    kind: str
    pixel: Tuple[float, ...]
    world: Optional[Tuple[float, ...]] = None
    label: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    style: MarkerStyle = field(default_factory=MarkerStyle)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "id": self.marker_id,
            "plane": self.plane,
            "kind": self.kind,
            "pixel": list(map(float, self.pixel)),
            "label": self.label,
            "metadata": dict(self.metadata),
            "style": self.style.to_dict(),
        }
        if self.world is not None:
            payload["world"] = list(map(float, self.world))
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MarkerState":
        if not isinstance(payload, dict):
            raise TypeError("MarkerState.from_dict expects a mapping payload")
        marker_id = str(payload.get("id") or uuid.uuid4().hex)
        plane = str(payload.get("plane") or "xy")
        kind = str(payload.get("kind") or "symbol")
        pixel = payload.get("pixel") or ()
        if not pixel:
            raise ValueError("Marker payload is missing 'pixel' coordinates")
        pixel_tuple = tuple(float(v) for v in pixel)
        world_payload = payload.get("world")
        world_tuple: Optional[Tuple[float, ...]] = None
        if world_payload is not None:
            world_tuple = tuple(float(v) for v in world_payload)
        style = MarkerStyle.from_dict(payload.get("style"))
        label = str(payload.get("label", ""))
        metadata = dict(payload.get("metadata") or {})
        return cls(
            marker_id=marker_id,
            plane=plane,
            kind=kind,
            pixel=pixel_tuple,
            world=world_tuple,
            label=label,
            metadata=metadata,
            style=style,
        )


class Marker:
    """Runtime marker representation with Matplotlib bindings."""

    kind: str = "marker"

    def __init__(
        self,
        plane: PlaneId,
        pixel: Tuple[float, ...],
        *,
        world: Optional[Tuple[float, ...]] = None,
        label: str = "",
        style: Optional[MarkerStyle] = None,
        marker_id: Optional[MarkerId] = None,
    ) -> None:
        self.plane = plane
        self.pixel = tuple(float(v) for v in pixel)
        self.world = tuple(float(v) for v in world) if world is not None else None
        self.label = label
        self.style = style or MarkerStyle()
        self.marker_id = marker_id or uuid.uuid4().hex
        self.artist = None  # Matplotlib Artist
        self.label_artist = None  # Optional text artist

    # ------------------------------------------------------------------
    # Serialization helpers
    def to_state(self) -> MarkerState:
        return MarkerState(
            marker_id=self.marker_id,
            plane=self.plane,
            kind=self.kind,
            pixel=self.pixel,
            world=self.world,
            label=self.label,
            metadata=self._state_metadata(),
            style=self.style,
        )

    @classmethod
    def from_state(cls: Type["Marker"], state: MarkerState) -> "Marker":
        return cls(
            plane=state.plane,
            pixel=state.pixel,
            world=state.world,
            label=state.label,
            style=state.style,
            marker_id=state.marker_id,
            **cls._kwargs_from_metadata(state.metadata),
        )

    def _state_metadata(self) -> Dict[str, Any]:
        """Override in subclasses to dump custom fields."""
        return {}

    @classmethod
    def _kwargs_from_metadata(cls, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Override in subclasses to restore custom fields."""
        return {}

    # ------------------------------------------------------------------
    # Matplotlib integration hooks (to be implemented by subclasses)
    def add_to_axes(self, ax) -> None:  # pragma: no cover - requires mpl context
        raise NotImplementedError

    def remove_from_axes(self) -> None:  # pragma: no cover - requires mpl context
        detach_artist(self.artist)
        self.artist = None
        detach_artist(self.label_artist)
        self.label_artist = None

    def apply_new_pixel(self, pixel: Tuple[float, ...]) -> None:
        self.pixel = tuple(float(v) for v in pixel)
        self._on_geometry_changed()

    def apply_new_world(self, world: Optional[Tuple[float, ...]]) -> None:
        self.world = tuple(float(v) for v in world) if world is not None else None

    def update_style(self, style: MarkerStyle) -> None:
        self.style = style
        self._on_style_changed()

    def set_label(self, label: str) -> None:
        self.label = label.strip()
        self._on_label_changed()

    def _on_geometry_changed(self) -> None:
        """Subclasses update artist geometry."""
        if self.artist is not None:
            self.artist.set_zorder(10)

    def _on_style_changed(self) -> None:
        """Subclasses update artist style."""
        if self.artist is not None:
            self.artist.set_alpha(self.style.opacity)

    def _on_label_changed(self) -> None:
        """Subclasses update label artist if needed."""
        pass
    # ------------------------------------------------------------------
    # Interaction helpers (override per subclass)
    def contains_pixel(self, x: float, y: float, tolerance: float = 5.0) -> bool:
        return bool(np.hypot(self.pixel[0] - x, self.pixel[1] - y) <= tolerance)

    def drag(self, dx: float, dy: float) -> None:
        x, y = self.pixel[:2]
        self.apply_new_pixel((x + dx, y + dy))

    def handles(self) -> Dict[str, Tuple[float, float]]:
        """Return named handle positions for resize/rotate operations."""
        return {}


class SymbolMarker(Marker):
    """Simple point marker rendered with Matplotlib markers."""

    kind = "symbol"

    def __init__(self, plane: PlaneId, pixel: Tuple[float, ...], *, symbol: Optional[str] = None, **kwargs: Any) -> None:
        self.symbol = symbol or kwargs.pop("marker_symbol", "o")
        super().__init__(plane, pixel, **kwargs)

    def add_to_axes(self, ax) -> None:  # pragma: no cover - requires mpl context
        x, y = self.pixel[:2]
        style = self.style
        marker_symbol = self.symbol or style.marker_symbol or "o"
        line = Line2D(
            [x],
            [y],
            marker=marker_symbol,
            linestyle="None",
            markersize=style.size,
            markerfacecolor=style.color,
            markeredgecolor=style.edgecolor or style.color,
            markeredgewidth=style.linewidth,
            alpha=style.opacity,
            zorder=10,
        )
        ax.add_line(line)
        self.artist = line
        self._ensure_label(ax)
        self._on_geometry_changed()

    def _ensure_label(self, ax) -> None:
        if not self.label:
            return
        if self.label_artist is None:
            self.label_artist = ax.text(
                self.pixel[0],
                self.pixel[1],
                self.label,
                color=self._label_color(),
                fontsize=max(self.style.size * 0.8, 8),
                ha="left",
                va="bottom",
                zorder=11,
            )
        self._update_label_position()

    def _label_color(self) -> str:
        color = self.style.color
        if not color or str(color).lower() == "none":
            edge = self.style.edgecolor
            if edge and str(edge).lower() != "none":
                return edge
            return "white"
        return color

    def _update_label_position(self) -> None:
        if self.label_artist is None or not self.label:
            return
        offset = self.style.size * 0.1
        self.label_artist.set_position((self.pixel[0] + offset, self.pixel[1] + offset))
        self.label_artist.set_text(self.label)
        self.label_artist.set_color(self._label_color())
        self.label_artist.set_zorder(11)

    def _state_metadata(self) -> Dict[str, Any]:
        return {"symbol": self.symbol}

    @classmethod
    def _kwargs_from_metadata(cls, metadata: Dict[str, Any]) -> Dict[str, Any]:
        return {"symbol": metadata.get("symbol")}

    def _on_geometry_changed(self) -> None:
        super()._on_geometry_changed()
        if self.artist is None:
            return
        x, y = self.pixel[:2]
        self.artist.set_data([x], [y])
        if self.label_artist is not None:
            self._update_label_position()

    def _on_style_changed(self) -> None:
        super()._on_style_changed()
        if self.artist is None:
            return
        style = self.style
        marker_symbol = self.symbol or style.marker_symbol
        if marker_symbol:
            self.artist.set_marker(marker_symbol)
        self.artist.set_markersize(style.size)
        self.artist.set_markerfacecolor(style.color)
        self.artist.set_markeredgecolor(style.edgecolor or style.color)
        self.artist.set_markeredgewidth(style.linewidth)
        self.artist.set_alpha(style.opacity)
        if self.label_artist is not None:
            self.label_artist.set_color(self._label_color())
            self._update_label_position()
        self._on_label_changed()

    def _on_label_changed(self) -> None:
        if not self.label:
            if self.label_artist is not None:
                self.label_artist.set_visible(False)
            return
        ax = None
        if self.artist is not None:
            ax = getattr(self.artist, "axes", None)
        if ax is None:
            return
        self._ensure_label(ax)
        if self.label_artist is not None:
            self.label_artist.set_visible(True)
            self._update_label_position()


class LineMarker(Marker):
    """Line segment or arrow marker driven by length and angle."""

    kind = "line"

    _LINE_STYLES = {"solid", "dashed", "dotted", "arrow", "double_arrow", "scale"}

    def __init__(
        self,
        plane: PlaneId,
        pixel: Tuple[float, ...],
        *,
        length: float = 10.0,
        angle_deg: float = 0.0,
        unit: str = "pixel",
        length_source_value: Optional[float] = None,
        style_mode: str = "solid",
        anchor: str = "center",
        **kwargs: Any,
    ) -> None:
        length_val = float(length)
        angle_val = float(angle_deg)
        style_mode = str(style_mode or "solid").lower()
        if style_mode not in self._LINE_STYLES:
            style_mode = "solid"
        anchor = (anchor or "center").lower()
        # Convert legacy "start" anchoring to centered coordinates
        pixel_tuple = tuple(float(v) for v in pixel)
        if anchor == "start":
            pixel_tuple = self._center_from_start(pixel_tuple, length_val, angle_val)
            anchor = "center"
        super().__init__(plane, pixel_tuple, **kwargs)
        self.length = length_val
        self.angle_deg = angle_val
        self.unit = unit
        self.length_source_value = float(length_source_value) if length_source_value is not None else float(length_val)
        self.style_mode = style_mode

    def add_to_axes(self, ax) -> None:  # pragma: no cover - requires mpl context
        self.remove_from_axes()
        artist = self._build_artist(ax)
        self.artist = artist
        self._ensure_label(ax)
        self._on_geometry_changed()

    def _ensure_label(self, ax) -> None:
        if not self.label:
            return
        if self.label_artist is None:
            self.label_artist = ax.text(
                self.pixel[0],
                self.pixel[1],
                self.label,
                color=self.style.color,
                fontsize=self._label_fontsize(),
                rotation=self.angle_deg,
                ha="center",
                va="bottom",
                zorder=10,
            )
        self._update_label_position()

    def _update_label_position(self) -> None:
        if self.label_artist is None or not self.label:
            return
        cx, cy = self.pixel[:2]
        dx, dy = self._label_offset()
        self.label_artist.set_position((cx + dx, cy + dy))
        self.label_artist.set_rotation(self.angle_deg)
        self.label_artist.set_color(self.style.color)
        self.label_artist.set_fontsize(self._label_fontsize())
        self.label_artist.set_text(self.label)
        self.label_artist.set_zorder(10)

    def _label_offset(self) -> Tuple[float, float]:
        """Return a small offset perpendicular to the line so label clears the stroke."""
        style = self.style
        size = float(getattr(style, "size", 12.0) or 12.0)
        linewidth = float(getattr(style, "linewidth", 1.0) or 1.0)
        magnitude = max(size * 0.08, linewidth * 0.4, 1.0)
        angle_rad = np.deg2rad(self.angle_deg)
        # Rotate 90 degrees counter-clockwise to get a normal vector.
        normal_x = -np.sin(angle_rad)
        normal_y = np.cos(angle_rad)
        return normal_x * magnitude, normal_y * magnitude

    def _label_fontsize(self) -> float:
        size = float(getattr(self.style, "size", 12.0) or 12.0)
        return max(size * 1.1, 12.0)

    def _endpoints(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        angle_rad = np.deg2rad(self.angle_deg)
        half = self.length / 2.0
        dx = half * np.cos(angle_rad)
        dy = half * np.sin(angle_rad)
        cx, cy = self.pixel[:2]
        start = (cx - dx, cy - dy)
        end = (cx + dx, cy + dy)
        return start, end

    def _center_from_start(self, start: Tuple[float, float], length: float, angle_deg: float) -> Tuple[float, float]:
        angle_rad = np.deg2rad(angle_deg)
        half = length / 2.0
        dx = half * np.cos(angle_rad)
        dy = half * np.sin(angle_rad)
        x0, y0 = start[:2]
        return x0 + dx, y0 + dy

    def _build_artist(self, ax):
        style = self.style
        color = style.color
        linewidth = style.linewidth if style.linewidth is not None else 1.0
        linewidth = float(linewidth if linewidth > 0 else (1.0 if self.style_mode != "solid" else 0.0))
        start, end = self._endpoints()
        if self.style_mode in {"arrow", "double_arrow"}:
            arrow = FancyArrowPatch(
                start,
                end,
                arrowstyle=self._arrowstyle(),
                mutation_scale=self._arrow_mutation_scale(linewidth),
                linewidth=max(linewidth, 0.1),
                fill=True,
                facecolor=color,
                edgecolor=color,
                alpha=style.opacity,
                zorder=9,
            )
            try:
                arrow.set_joinstyle("miter")
                arrow.set_capstyle("butt")
            except Exception:
                pass
            ax.add_patch(arrow)
            return arrow
        if self.style_mode == "scale":
            segments = self._scale_segments()
            collection = LineCollection(
                segments,
                colors=[color],
                linewidths=[max(linewidth, 0.1)] * len(segments),
                alpha=style.opacity,
                zorder=9,
            )
            ax.add_collection(collection)
            return collection
        linestyle = self._mpl_linestyle()
        line = Line2D(
            (start[0], end[0]),
            (start[1], end[1]),
            linewidth=max(linewidth, 0.1),
            linestyle=linestyle,
            color=color,
            alpha=style.opacity,
            zorder=9,
        )
        ax.add_line(line)
        return line

    def _mpl_linestyle(self) -> str:
        mapping = {
            "solid": "solid",
            "dashed": "dashed",
            "dotted": "dotted",
        }
        return mapping.get(self.style_mode, "solid")

    def _arrowstyle(self) -> str:
        if self.style_mode == "double_arrow":
            return "<|-|>"
        return "-|>"

    def _arrow_mutation_scale(self, linewidth: float) -> float:
        size_component = getattr(self.style, "size", 0.0) or 12.0
        base = max(size_component, 6.0)
        return base

    def _scale_segments(self) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        start, end = self._endpoints()
        segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = [(start, end)]
        start_vec = np.array(start, dtype=float)
        end_vec = np.array(end, dtype=float)
        direction = end_vec - start_vec
        length = float(np.linalg.norm(direction))
        if length == 0.0:
            return segments
        direction /= length
        normal = np.array([-direction[1], direction[0]])
        base = max(self.style.linewidth or 1.0, 1.5)
        tick_length = min(length * 0.1, base * 3.0)
        tick_vec = normal * tick_length
        segments.append((tuple(start_vec - tick_vec * 0.5), tuple(start_vec + tick_vec * 0.5)))
        segments.append((tuple(end_vec - tick_vec * 0.5), tuple(end_vec + tick_vec * 0.5)))
        return segments

    def _state_metadata(self) -> Dict[str, Any]:
        return {
            "length": float(self.length),
            "angle_deg": float(self.angle_deg),
            "unit": self.unit,
            "length_source": float(self.length_source_value),
            "style_mode": self.style_mode,
            "anchor": "center",
        }

    @classmethod
    def _kwargs_from_metadata(cls, metadata: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "length": metadata.get("length", 10.0),
            "angle_deg": metadata.get("angle_deg", 0.0),
            "unit": metadata.get("unit", "pixel"),
            "length_source_value": metadata.get("length_source", metadata.get("length", 10.0)),
            "style_mode": metadata.get("style_mode", "solid"),
            "anchor": metadata.get("anchor", "start"),
        }

    def _on_geometry_changed(self) -> None:
        super()._on_geometry_changed()
        if self.artist is None:
            return
        start, end = self._endpoints()
        if isinstance(self.artist, FancyArrowPatch):
            self.artist.set_positions(start, end)
        elif isinstance(self.artist, LineCollection):
            segments = self._scale_segments()
            self.artist.set_segments(segments)
        elif isinstance(self.artist, Line2D):
            self.artist.set_data((start[0], end[0]), (start[1], end[1]))
        if self.label_artist is not None:
            self._update_label_position()

    def _on_style_changed(self) -> None:
        super()._on_style_changed()
        if self.artist is None:
            return
        style = self.style
        color = style.color
        linewidth = style.linewidth if style.linewidth is not None else 1.0
        linewidth = float(linewidth if linewidth > 0 else (1.0 if self.style_mode != "solid" else 0.1))
        if isinstance(self.artist, FancyArrowPatch):
            self.artist.set_color(color)
            self.artist.set_linewidth(max(linewidth, 0.1))
            self.artist.set_alpha(style.opacity)
            try:
                self.artist.set_mutation_scale(self._arrow_mutation_scale(linewidth))
            except Exception:
                pass
        elif isinstance(self.artist, LineCollection):
            segments = self._scale_segments()
            self.artist.set_segments(segments)
            self.artist.set_color(color)
            self.artist.set_alpha(style.opacity)
            self.artist.set_linewidths([max(linewidth, 0.1)] * len(segments))
        elif isinstance(self.artist, Line2D):
            self.artist.set_color(color)
            self.artist.set_linewidth(max(linewidth, 0.1))
            self.artist.set_alpha(style.opacity)
            self.artist.set_linestyle(self._mpl_linestyle())
        if self.label_artist is not None:
            self.label_artist.set_color(color)
            self.label_artist.set_rotation(self.angle_deg)
            self.label_artist.set_fontsize(self._label_fontsize())
            self._update_label_position()
        self._on_label_changed()

    def _on_label_changed(self) -> None:
        if not self.label:
            if self.label_artist is not None:
                self.label_artist.set_visible(False)
            return
        ax = None
        if self.artist is not None:
            ax = getattr(self.artist, "axes", None)
        if ax is None:
            return
        self._ensure_label(ax)
        if self.label_artist is not None:
            self.label_artist.set_visible(True)
            self._update_label_position()

    def handles(self) -> Dict[str, Tuple[float, float]]:
        start, end = self._endpoints()
        return {"start": start, "end": end}

    def update_from_endpoints(self, start: Tuple[float, float], end: Tuple[float, float]) -> None:
        start_vec = np.array(start[:2], dtype=float)
        end_vec = np.array(end[:2], dtype=float)
        center = (start_vec + end_vec) / 2.0
        delta = end_vec - start_vec
        length = float(np.linalg.norm(delta))
        angle = float(np.degrees(np.arctan2(delta[1], delta[0]))) if length > 0 else self.angle_deg
        self.pixel = (float(center[0]), float(center[1]))
        self.length = length
        self.angle_deg = angle
        if (self.unit or "pixel") == "pixel":
            self.length_source_value = length
        else:
            try:
                self.length_source_value = float(self.length_source_value)
            except (TypeError, ValueError):
                self.length_source_value = length
        self._on_geometry_changed()

    def contains_pixel(self, x: float, y: float, tolerance: float = 5.0) -> bool:
        start, end = self._endpoints()
        start = np.array(start, dtype=float)
        end = np.array(end, dtype=float)
        point = np.array([x, y], dtype=float)
        seg = end - start
        seg_norm = np.dot(seg, seg)
        if seg_norm <= 1e-12:
            extra = max(tolerance, getattr(self.style, "linewidth", 1.0) * 2.5)
            return bool(np.hypot(*(point - start)) <= extra)
        t = np.clip(np.dot(point - start, seg) / seg_norm, 0.0, 1.0)
        proj = start + t * seg
        extra = max(tolerance, getattr(self.style, "linewidth", 1.0) * 1.8)
        return bool(np.hypot(*(point - proj)) <= extra)

    def set_length(self, length: float, *, source_value: Optional[float] = None) -> None:
        self.length = float(length)
        if source_value is not None:
            self.length_source_value = float(source_value)
        self._on_geometry_changed()

    def set_angle(self, angle_deg: float) -> None:
        self.angle_deg = float(angle_deg)
        self._on_geometry_changed()

    def set_unit(self, unit: str) -> None:
        self.unit = unit

    def set_style_mode(self, style_mode: str) -> None:
        normalized = str(style_mode or "solid").lower()
        if normalized not in self._LINE_STYLES:
            normalized = "solid"
        if normalized == self.style_mode:
            return
        self.style_mode = normalized
        ax = getattr(self.artist, "axes", None) if self.artist is not None else None
        self.remove_from_axes()
        if ax is not None:
            self.add_to_axes(ax)


class TextMarker(Marker):
    """Free-form text annotation."""

    kind = "text"

    def __init__(self, plane: PlaneId, pixel: Tuple[float, ...], *, text: str = "", **kwargs: Any) -> None:
        # Avoid passing label twice when restoring from state.
        label_value = kwargs.pop("label", text)
        super().__init__(plane, pixel, label=label_value, **kwargs)
        actual_text = text or label_value
        self.text = actual_text or ""
        self.label = self.text

    def add_to_axes(self, ax) -> None:  # pragma: no cover - requires mpl context
        x, y = self.pixel[:2]
        style = self.style
        font_family = self._font_family_for_text()
        text_artist = ax.text(
            x,
            y,
            self.text,
            color=style.color,
            fontsize=style.size,
            rotation=style.rotation,
            fontfamily=font_family,
            alpha=style.opacity,
            ha="left",
            va="bottom",
            zorder=10,
        )
        self.artist = text_artist
        self.label_artist = text_artist
        self._on_geometry_changed()

    def _state_metadata(self) -> Dict[str, Any]:
        return {"text": self.text}

    @classmethod
    def _kwargs_from_metadata(cls, metadata: Dict[str, Any]) -> Dict[str, Any]:
        return {"text": metadata.get("text", "")}

    def update_text(self, text: str) -> None:
        cleaned = text.strip()
        self.text = cleaned
        self.label = cleaned
        self._on_style_changed()

    def set_label(self, label: str) -> None:
        self.update_text(label)

    def _on_geometry_changed(self) -> None:
        super()._on_geometry_changed()
        if isinstance(self.artist, Text):
            self.artist.set_position(self.pixel[:2])

    def _on_style_changed(self) -> None:
        super()._on_style_changed()
        if isinstance(self.artist, Text):
            style = self.style
            self.artist.set_color(style.color)
            self.artist.set_fontsize(style.size)
            self.artist.set_rotation(style.rotation)
            self.artist.set_alpha(style.opacity)
            self.artist.set_text(self.text)
            font_family = self._font_family_for_text()
            self.artist.set_fontfamily(font_family)
            self.style.font_family = font_family

    def contains_pixel(self, x: float, y: float, tolerance: float = 5.0) -> bool:
        if isinstance(self.artist, Text):
            renderer = None
            if self.artist.figure and self.artist.figure.canvas:
                try:
                    renderer = self.artist.figure.canvas.get_renderer()
                except Exception:
                    renderer = None
            try:
                bbox = self.artist.get_window_extent(renderer=renderer)
            except Exception:
                bbox = None
            if bbox is not None:
                inv = self.artist.axes.transData.inverted()
                corners = inv.transform(
                    np.array(
                        [
                            [bbox.xmin, bbox.ymin],
                            [bbox.xmax, bbox.ymin],
                            [bbox.xmax, bbox.ymax],
                            [bbox.xmin, bbox.ymax],
                        ]
                    )
                )
                xs, ys = corners[:, 0], corners[:, 1]
                pad = max(tolerance, getattr(self.style, "size", 12.0) * 0.3)
                return bool(xs.min() - pad <= x <= xs.max() + pad and ys.min() - pad <= y <= ys.max() + pad)
        return super().contains_pixel(x, y, tolerance)

    def _is_math_text(self, text: str) -> bool:
        if not text:
            return False
        text = text.strip()
        if text.count("$") >= 2:
            return True
        return False

    def _font_family_for_text(self) -> str:
        if self._is_math_text(self.text):
            return "STIXGeneral"
        return resolve_mpl_font_family(self.style.font_family)


MarkerT = TypeVar("MarkerT", bound=Marker)


def marker_from_state(state: MarkerState) -> Marker:
    """Factory creating the appropriate Marker subclass based on state."""
    registry: Dict[str, Type[Marker]] = {
        SymbolMarker.kind: SymbolMarker,
        LineMarker.kind: LineMarker,
        TextMarker.kind: TextMarker,
    }
    cls = registry.get(state.kind, SymbolMarker)
    return cls.from_state(state)


def serialize_marker_states(markers: Dict[MarkerId, MarkerState], *, plane: PlaneId, world_frame: Optional[str] = None) -> Dict[str, Any]:
    """Serialize marker states into a dict ready for JSON export."""
    return {
        "format": "takefits.marker",
        "version": 1,
        "plane": plane,
        "world_frame": world_frame,
        "markers": [state.to_dict() for state in markers.values()],
    }


def deserialize_marker_states(payload: Dict[str, Any]) -> Tuple[PlaneId, Optional[str], Dict[MarkerId, MarkerState]]:
    """Deserialize JSON payload into marker states keyed by id."""
    if payload.get("format") != "takefits.marker":
        raise ValueError("Unsupported marker file format")
    version = int(payload.get("version", 0))
    if version != 1:
        raise ValueError(f"Unsupported marker format version: {version}")
    plane = str(payload.get("plane") or "xy")
    world_frame = payload.get("world_frame")
    markers_payload = payload.get("markers") or []
    result: Dict[MarkerId, MarkerState] = {}
    for entry in markers_payload:
        state = MarkerState.from_dict(entry)
        # Older channel-map replication could serialize one source id into
        # several tile entries. Preserve every entry instead of silently
        # overwriting the earlier marker in this id-keyed mapping.
        while state.marker_id in result:
            state.marker_id = uuid.uuid4().hex
        result[state.marker_id] = state
    return plane, world_frame, result


def marker_states_to_json(states: Dict[MarkerId, MarkerState], *, plane: PlaneId, world_frame: Optional[str] = None) -> str:
    payload = serialize_marker_states(states, plane=plane, world_frame=world_frame)
    return json.dumps(payload, indent=2)


def marker_states_from_json(data: str) -> Tuple[PlaneId, Optional[str], Dict[MarkerId, MarkerState]]:
    payload = json.loads(data)
    return deserialize_marker_states(payload)


def marker_states_to_ds9(
    states: Dict[MarkerId, MarkerState],
    plane: PlaneId,
    *,
    coordinate_system: str = "image",
) -> str:
    """Convert marker states to a DS9 region file string."""

    def _safe_color(color: Optional[str]) -> str:
        if not color or str(color).lower() == "none":
            return "white"
        return str(color)

    def _symbol_name(symbol: Optional[str]) -> str:
        mapping = {
            "o": "circle",
            "s": "box",
            "d": "diamond",
            "D": "diamond",
            "^": "triangle",
            "v": "triangle",
            "*": "x",
            "+": "plus",
            "x": "x",
        }
        return mapping.get(symbol, "circle")

    lines: List[str] = ["# Region file format: DS9 version 4.1", coordinate_system]
    for state in states.values():
        if not state.pixel or len(state.pixel) < 2:
            continue
        x = state.pixel[0] + 1.0
        y = state.pixel[1] + 1.0
        color = _safe_color(state.style.color)
        label = state.label.replace('\n', ' ') if state.label else ""

        if state.kind == SymbolMarker.kind:
            symbol = state.metadata.get("symbol") or state.style.marker_symbol
            point_type = _symbol_name(symbol)
            size = max(float(state.style.size) * 0.6, 1.0)
            parts = [
                f"point({x:.6f},{y:.6f})",
                "#",
                f"point={point_type}",
                f"color={color}",
                f"pointsize={size:.2f}",
            ]
            if label:
                parts.append(f"text={{{label}}}")
            lines.append(" ".join(parts))

        elif state.kind == LineMarker.kind:
            length = float(state.metadata.get("length", 0.0))
            angle_deg = float(state.metadata.get("angle_deg", 0.0))
            angle_rad = math.radians(angle_deg)
            x2 = x + length * math.cos(angle_rad)
            y2 = y + length * math.sin(angle_rad)
            linewidth = max(int(round(state.style.linewidth or 1.0)), 1)
            parts = [
                f"line({x:.6f},{y:.6f},{x2:.6f},{y2:.6f})",
                "#",
                f"color={color}",
                f"linewidth={linewidth}",
            ]
            if label:
                parts.append(f"text={{{label}}}")
            lines.append(" ".join(parts))

        elif state.kind == TextMarker.kind:
            rotation = float(state.style.rotation)
            size = max(float(state.style.size), 6.0)
            text_value = state.metadata.get("text") or state.label or "text"
            parts = [
                f"text({x:.6f},{y:.6f})",
                "#",
                f"color={color}",
                f"text={{ {text_value} }}",
                f"font=\"helvetica {size:.1f}\"",
                f"rotate={rotation:.1f}",
            ]
            lines.append(" ".join(parts))

    return "\n".join(lines) + "\n"
    def contains_pixel(self, x: float, y: float, tolerance: float = 5.0) -> bool:
        size = getattr(self.style, "size", 12.0)
        effective_tol = max(tolerance, size * 0.45)
        return super().contains_pixel(x, y, effective_tol)
