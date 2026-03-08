import copy
import json
import math
import os
import weakref
from datetime import datetime
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING, Set, Any

import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from PySide6.QtCore import QObject, Signal as pyqtSignal

from astropy.coordinates import SkyCoord
from astropy.wcs.utils import wcs_to_celestial_frame, skycoord_to_pixel, pixel_to_skycoord
import astropy.units as u
from matplotlib.path import Path

try:
    from scipy.ndimage import gaussian_filter
except Exception:  # pragma: no cover - scipy is expected but guard just in case
    gaussian_filter = None

from matplotlib.collections import LineCollection



ContourItemProvider = Callable[[], Sequence["ContourItem"]]

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.contour import QuadContourSet


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
            return _normalize_frame_name(candidate)
    except Exception:
        candidate = None
    return _normalize_frame_name(frame.__class__.__name__ if frame is not None else None)


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


def _transform_coords_between_frames(
    coords: np.ndarray,
    source_frame_name: Optional[str],
    target_frame: Optional[object],
) -> Optional[np.ndarray]:
    if coords is None:
        return None
    arr = np.asarray(coords, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return None
    if target_frame is None:
        return arr
    target_name = _frame_name_from_frame(target_frame)
    source_name = _normalize_frame_name(source_frame_name) or target_name
    if target_name is None:
        return arr
    if source_name == target_name:
        return arr
    src_coord = _make_skycoord(arr, source_name)
    if src_coord is None:
        return arr
    try:
        dst = src_coord.transform_to(target_frame)
        return np.column_stack([dst.spherical.lon.deg, dst.spherical.lat.deg])
    except Exception:
        return arr


def _ds9_keyword_from_frame(frame_name: Optional[str]) -> Optional[str]:
    name = _normalize_frame_name(frame_name)
    if name is None:
        return None
    return {
        "ICRS": "ICRS",
        "FK5": "FK5",
        "FK4": "FK4",
        "GALACTIC": "GALACTIC",
    }.get(name)


def _pixel_coords_to_world(pixels: np.ndarray, wcs: Optional[object]) -> Optional[np.ndarray]:
    if wcs is None:
        return None
    arr = np.asarray(pixels, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return None
    try:
        sky = pixel_to_skycoord(arr[:, 0], arr[:, 1], wcs, origin=0)
    except Exception:
        return None
    try:
        lon = sky.spherical.lon.deg
        lat = sky.spherical.lat.deg
    except Exception:
        return None
    return np.column_stack([lon, lat])


def _world_coords_to_pixel(
    world: np.ndarray,
    wcs: Optional[object],
    frame_hint: Optional[str] = None,
) -> Optional[np.ndarray]:
    if wcs is None or world is None:
        return None
    arr = np.asarray(world, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return None
    coord = _make_skycoord(arr, frame_hint) or _make_skycoord(arr, _frame_name_from_wcs(wcs))
    if coord is None:
        try:
            coord = SkyCoord(arr[:, 0] * u.deg, arr[:, 1] * u.deg)
        except Exception:
            return None
    try:
        px, py = skycoord_to_pixel(coord, wcs, origin=0)
    except Exception:
        return None
    return np.column_stack([np.asarray(px, dtype=float), np.asarray(py, dtype=float)])


@dataclass
class ContourParameters:
    """User-selected contour styling and generation parameters."""

    level_min: Optional[float] = None
    level_max: Optional[float] = None
    level_step: Optional[float] = None
    smoothing: float = 0.0
    linewidth: float = 1.0
    color: str = "white"

    def clone(self) -> "ContourParameters":
        return ContourParameters(
            level_min=self.level_min,
            level_max=self.level_max,
            level_step=self.level_step,
            smoothing=self.smoothing,
            linewidth=self.linewidth,
            color=self.color,
        )


@dataclass
class ContourSegment:
    """A single contour polyline."""

    level: float
    world: Optional[np.ndarray] = None  # (N, 2) array of world coordinates
    pixels: Optional[np.ndarray] = None  # (N, 2) array of pixel coordinates
    color: Optional[np.ndarray] = None  # RGBA color for this segment
    original_color: Optional[np.ndarray] = None  # Backup of original color for restoration
    linestyle: str = "solid"


@dataclass
class ContourItemState:
    """Stored contour information for a single axes item."""

    item_label: str
    segments: List[ContourSegment] = field(default_factory=list)


@dataclass
class ContourState:
    """Full contour state for a layer."""

    layer_id: str
    plane: Optional[str]
    label: str
    parameters: ContourParameters
    levels: Sequence[float]
    items: List[ContourItemState] = field(default_factory=list)
    # Optional source world coordinate frame for saved world coords (e.g., 'ICRS', 'FK5', 'GALACTIC').
    world_frame: Optional[str] = None
    overlay_id: Optional[str] = None


@dataclass
class ContourItem:
    """
    Represents a drawable data item on which contours can be computed.

    Attributes
    ----------
    ax:
        Matplotlib Axes (ideally WCSAxes) the contour should be drawn onto.
    data:
        2D numpy array for contour calculation.
    label:
        Human-readable label for the item (e.g. subplot title).
    extent:
        Optional extent tuple (xmin, xmax, ymin, ymax). Defaults to imshow default.
    origin:
        Origin keyword passed to contour. Defaults to 'lower' for FITS images.
    metadata:
        Extra information stored alongside saved contours.
    """

    ax: "Axes"
    data: np.ndarray
    label: str
    extent: Optional[Tuple[float, float, float, float]] = None
    origin: str = "lower"
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class ContourListEntry:
    id: str
    label: str
    plane: Optional[str]
    kind: str  # 'layer' or 'overlay'
    parent_id: Optional[str] = None
    active: bool = False


class ContourLayer:
    """
    Wraps one logical drawing surface (main window, subwindow, PV plot, etc.).
    Provides helpers to generate and clear contours on the associated axes.
    """

    def __init__(
        self,
        manager: "ContourManager",
        layer_id: str,
        label: str,
        plane: Optional[str],
        provider: ContourItemProvider,
        owner: object,
        ):
        self._manager = manager
        self.id = layer_id
        self.label = label
        self.plane = plane
        self._provider = provider
        self._owner = weakref.ref(owner) if owner is not None else None
        self._active = False
        self._last_parameters: Optional[ContourParameters] = None
        self._gen_artists: List[object] = []
        
        self._last_contour_sets: Dict[str, "QuadContourSet"] = {}
        
        self._overlay_artists: Dict[str, List[object]] = {}
        self._overlay_states: List[ContourState] = []
        self._overlay_counter: int = 0
        
        self._state: Optional[ContourState] = None 
        
        self._source_path: Optional[str] = self._infer_source_path(owner)

    def owner_alive(self) -> bool:
        if self._owner is None:
            return True
        return self._owner() is not None

    def _infer_source_path(self, owner: object) -> Optional[str]:
        if owner is None:
            return None
        for attr in ("filename_path", "original_filename", "filename"):
            value = getattr(owner, attr, None)
            if isinstance(value, str):
                stripped = value.strip()
                if stripped:
                    return stripped

        parent = getattr(owner, "parent", None)
        if parent is not None and parent is not owner:
            for attr in ("filename_path", "original_filename", "filename"):
                value = getattr(parent, attr, None)
                if isinstance(value, str):
                    stripped = value.strip()
                    if stripped:
                        return stripped

        nested = getattr(owner, "fits_viewer", None)
        if nested is not None and nested is not owner:
            return self._infer_source_path(nested)
        return None


    def default_source_filename(self) -> Optional[str]:
        return self._source_path

    def _make_overlay_id(self) -> str:
        self._overlay_counter += 1
        return f"{self.id}::overlay::{self._overlay_counter}"

    def is_active(self) -> bool:
        return self._active

    def data_bounds(self) -> Optional[Tuple[float, float]]:
        items = self._provider()
        clim_mins: List[float] = []
        clim_maxs: List[float] = []
        mins: List[float] = []
        maxs: List[float] = []
        for item in items:
            clim = None
            try:
                clim = item.metadata.get("clim")
            except Exception:
                clim = None
            if clim is not None:
                try:
                    cmin, cmax = float(clim[0]), float(clim[1])
                except Exception:
                    pass
                else:
                    if np.isfinite(cmin) and np.isfinite(cmax):
                        clim_mins.append(cmin)
                        clim_maxs.append(cmax)
            arr = np.asarray(item.data, dtype=float)
            if arr.size == 0:
                continue
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
            mins.append(float(arr.min()))
            maxs.append(float(arr.max()))
        if clim_mins and clim_maxs:
            return float(min(clim_mins)), float(max(clim_maxs))
        if not mins or not maxs:
            return None
        return float(min(mins)), float(max(maxs))

    def clear(self) -> None:
        canvases = set()
        for artist in list(self._gen_artists):
            axes = getattr(artist, "axes", None)
            canvas = getattr(getattr(axes, "figure", None), "canvas", None)
            if canvas is not None:
                canvases.add(canvas)
        for artists in list(self._overlay_artists.values()):
            for artist in artists:
                axes = getattr(artist, "axes", None)
                canvas = getattr(getattr(axes, "figure", None), "canvas", None)
                if canvas is not None:
                    canvases.add(canvas)
        for artist in list(self._gen_artists):
            try:
                artist.remove()
            except Exception:
                pass
        self._gen_artists.clear()
        for artists in list(self._overlay_artists.values()):
            for artist in artists:
                try:
                    artist.remove()
                except Exception:
                    pass
        self._overlay_artists.clear()
        self._overlay_states.clear()
        self._overlay_counter = 0
        self._active = False
        self._state = None
        for canvas in canvases:
            try:
                canvas.draw_idle()
            except Exception:
                pass

    def _clear_generated(self) -> None:
        canvases = set()
        for artist in self._gen_artists:
            axes = getattr(artist, "axes", None)
            canvas = getattr(getattr(axes, "figure", None), "canvas", None)
            if canvas is not None:
                canvases.add(canvas)
        for artist in self._gen_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._gen_artists.clear()
        
        self._last_contour_sets.clear()
        self._state = None
        
        self._last_parameters = None 
        
        # Active remains true if overlays exist
        self._active = bool(self._overlay_artists)
        for canvas in canvases:
            try:
                canvas.draw_idle()
            except Exception:
                pass

    def clear_generated(self) -> bool:
        """Remove only generated contours, keeping imported overlays intact."""
        if not self._gen_artists and self._state is None:
            return False
        self._clear_generated()
        return True

    def _remove_overlay_artists(self, overlay_ids: Optional[Iterable[str]] = None) -> None:
        canvases = set()
        if overlay_ids is None:
            target_items = list(self._overlay_artists.items())
        else:
            ids = {oid for oid in overlay_ids if oid in self._overlay_artists}
            target_items = [(oid, self._overlay_artists[oid]) for oid in ids]
        for overlay_id, artists in target_items:
            for artist in artists:
                axes = getattr(artist, "axes", None)
                canvas = getattr(getattr(axes, "figure", None), "canvas", None)
                if canvas is not None:
                    canvases.add(canvas)
                try:
                    artist.remove()
                except Exception:
                    pass
            self._overlay_artists.pop(overlay_id, None)
        # Active remains true if generated exist or overlays remain
        self._active = bool(self._gen_artists or self._overlay_artists)
        for canvas in canvases:
            try:
                canvas.draw_idle()
            except Exception:
                pass

    @staticmethod
    def _extract_path_polylines(path: Path) -> List[np.ndarray]:
        if path is None:
            return []
        verts = getattr(path, "vertices", None)
        if verts is None or len(verts) == 0:
            return []
        codes = getattr(path, "codes", None)
        if codes is None:
            try:
                polygons = path.to_polygons(closed_only=False)
            except Exception:
                polygons = None
            if not polygons:
                polygons = [verts]
            return [np.asarray(poly, dtype=float) for poly in polygons if np.asarray(poly, dtype=float).size]

        polylines: List[np.ndarray] = []
        current: List[np.ndarray] = []
        for vertex, code in zip(verts, codes):
            if code == Path.MOVETO:
                if len(current) >= 2:
                    polylines.append(np.asarray(current, dtype=float))
                current = [np.asarray(vertex, dtype=float)]
            elif code == Path.LINETO:
                current.append(np.asarray(vertex, dtype=float))
            elif code == Path.CLOSEPOLY:
                if current:
                    current.append(np.asarray(current[0], dtype=float))
                    if len(current) >= 2:
                        polylines.append(np.asarray(current, dtype=float))
                current = []
            else:
                if len(current) >= 2:
                    polylines.append(np.asarray(current, dtype=float))
                current = []
        if len(current) >= 2:
            polylines.append(np.asarray(current, dtype=float))
        return polylines

    def _split_polyline(
        self,
        pixels: np.ndarray,
        world: Optional[np.ndarray] = None,
    ) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
        arr = np.asarray(pixels, dtype=float)
        if arr.ndim != 2 or arr.shape[0] < 2:
            return []

        finite_rows = np.isfinite(arr).all(axis=1)
        diffs = np.diff(arr, axis=0)
        dists = np.hypot(diffs[:, 0], diffs[:, 1])
        finite_pairs = finite_rows[:-1] & finite_rows[1:]
        valid_dists = dists[finite_pairs]
        typical = 1.0
        if valid_dists.size and np.isfinite(valid_dists).any():
            try:
                typical = float(np.nanmedian(valid_dists))
            except Exception:
                typical = 1.0
            if not np.isfinite(typical) or typical <= 0:
                finite_vals = valid_dists[np.isfinite(valid_dists)]
                if finite_vals.size:
                    typical = float(np.nanmedian(finite_vals))
                if not np.isfinite(typical) or typical <= 0:
                    typical = 1.0
        thresh = max(6.0 * typical, 3.0)

        ranges: List[Tuple[int, int]] = []
        start = 0
        for idx in range(1, arr.shape[0]):
            must_split = (not (finite_rows[idx - 1] and finite_rows[idx])) or (dists[idx - 1] > thresh)
            if must_split:
                if idx - start >= 2:
                    ranges.append((start, idx))
                start = idx
        if arr.shape[0] - start >= 2:
            ranges.append((start, arr.shape[0]))
        if not ranges:
            ranges.append((0, arr.shape[0]))

        results: List[Tuple[np.ndarray, Optional[np.ndarray]]] = []
        world_arr = None
        if world is not None:
            world_arr = np.asarray(world, dtype=float)
            if world_arr.ndim != 2 or world_arr.shape[0] != arr.shape[0]:
                world_arr = None

        for start_idx, end_idx in ranges:
            chunk = arr[start_idx:end_idx]
            if chunk.shape[0] < 2:
                continue
            length = float(np.sum(np.hypot(np.diff(chunk[:, 0]), np.diff(chunk[:, 1]))))
            if chunk.shape[0] < 3 and length > max(3.0 * typical, 3.0):
                continue
            world_chunk = None
            if world_arr is not None:
                world_chunk = world_arr[start_idx:end_idx].copy()
            results.append((chunk.copy(), world_chunk))
        return results

    def _canonicalize_overlay_state(self, state: ContourState) -> Optional[ContourState]:
        all_provider_items = list(self._provider())
        if not all_provider_items or state is None or not state.items:
            return None

        # Collect all segments from the source state into one list.
        all_segments_to_load = [
            seg
            for item_state in state.items
            for seg in item_state.segments
            if seg.world is not None or seg.pixels is not None
        ]
        if not all_segments_to_load:
            return None

        canonical_items: List[ContourItemState] = []
        derived_world_frame: Optional[str] = None

        # Iterate over every panel in the destination layer (e.g., all channel map tiles).
        for target_item in all_provider_items:
            ax = target_item.ax
            wcs = getattr(ax, "wcs", None)
            target_frame = wcs_to_celestial_frame(wcs) if wcs else None
            frame_name = _frame_name_from_frame(target_frame)
            if frame_name is not None:
                derived_world_frame = frame_name
            elif derived_world_frame is None:
                derived_world_frame = _normalize_frame_name(state.world_frame)

            new_segments_for_this_item: List[ContourSegment] = []
            # Apply every loaded segment to the current panel.
            for segment in all_segments_to_load:
                pixel_coords = None
                world_coords = None

                if segment.world is not None and np.asarray(segment.world).size:
                    transformed = _transform_coords_between_frames(
                        np.asarray(segment.world, dtype=float),
                        state.world_frame,
                        target_frame,
                    )
                    if transformed is not None:
                        world_coords = transformed
                if wcs is not None and world_coords is not None:
                    pixel_coords = _world_coords_to_pixel(world_coords, wcs, _frame_name_from_frame(target_frame))
                if pixel_coords is None and segment.pixels is not None and np.asarray(segment.pixels).size:
                    pixel_coords = np.asarray(segment.pixels, dtype=float)
                if wcs is not None and world_coords is None and pixel_coords is not None:
                    world_coords = _pixel_coords_to_world(pixel_coords, wcs)

                if pixel_coords is None or pixel_coords.ndim != 2 or pixel_coords.shape[0] < 2:
                    continue

                finite_mask = np.isfinite(pixel_coords).all(axis=1)
                if np.count_nonzero(finite_mask) < 2:
                    continue
                pixel_coords = pixel_coords[finite_mask]
                if world_coords is not None and world_coords.shape[0] == len(finite_mask):
                    world_coords = np.asarray(world_coords, dtype=float)[finite_mask]
                else:
                    world_coords = world_coords if world_coords is None else np.asarray(world_coords, dtype=float)

                for pix_chunk, world_chunk in self._split_polyline(pixel_coords, world_coords):
                    if world_chunk is None and wcs is not None:
                        world_chunk = _pixel_coords_to_world(pix_chunk, wcs)
                    color_copy = None
                    if segment.color is not None:
                        try:
                            color_copy = np.asarray(segment.color, dtype=float).copy()
                        except Exception:
                            color_copy = segment.color
                    new_segments_for_this_item.append(
                        ContourSegment(
                            level=segment.level,
                            world=world_chunk,
                            pixels=pix_chunk,
                            color=color_copy,
                            linestyle=segment.linestyle,
                        )
                    )

            if new_segments_for_this_item:
                canonical_items.append(
                    ContourItemState(item_label=target_item.label, segments=new_segments_for_this_item)
                )

        if not canonical_items:
            return None

        canonical_label = state.label if state.label else self.label
        canonical_state = ContourState(
            layer_id=self.id,
            plane=self.plane,
            label=canonical_label,
            parameters=state.parameters.clone(),
            levels=list(state.levels),
            items=canonical_items,
            world_frame=derived_world_frame,
        )
        if state.overlay_id:
            canonical_state.overlay_id = state.overlay_id
        else:
            canonical_state.overlay_id = self._make_overlay_id()
        return canonical_state

    def _replot_overlay_states(self) -> None:
        if not self._overlay_states:
            return
        
        self._remove_overlay_artists()
        
        for overlay_state in self._overlay_states:
            self._draw_overlay_state(overlay_state)
            
        canvases = set()
        for artists in self._overlay_artists.values():
            for artist in artists:
                if hasattr(artist, "axes") and artist.axes is not None and artist.axes.figure is not None:
                    canvases.add(artist.axes.figure.canvas)
        for canvas in canvases:
            try:
                canvas.draw()
            except Exception:
                pass
        self._active = True

    def _draw_overlay_state(self, state: ContourState) -> None:
        if state is None or not state.items:
            return

        if state.overlay_id is None:
            state.overlay_id = self._make_overlay_id()
        overlay_id = state.overlay_id

        overlay_params = state.parameters.clone()
        color_mode = (overlay_params.color or "").lower()
        cmap = None
        level_norm = None
        if color_mode == "rainbow" and state.levels:
            try:
                cmap = cm.get_cmap("rainbow")
                level_norm = mcolors.Normalize(vmin=min(state.levels), vmax=max(state.levels))
            except Exception:
                cmap = None
                level_norm = None

        provider_items = list(self._provider())
        items_by_label = {item.label: item for item in provider_items if item.label}
        new_artists: List[object] = []

        for idx, item_state in enumerate(state.items):
            target_item = items_by_label.get(item_state.item_label)
            if target_item is None and idx < len(provider_items):
                target_item = provider_items[idx]
            if target_item is None:
                continue
            ax = target_item.ax
            wcs = getattr(ax, "wcs", None)

            # Optimization: Group segments by style (color, linewidth, linestyle) to use LineCollection
            # Key: (color_tuple_or_string, linewidth, linestyle)
            # Value: list of segments (pixels) along with color if it varies per segment
            # Actually LineCollection handles varying colors well if we pass a list.
            # But line style/width usually constant per collection for best performance.
            
            # For "rainbow" mode or per-segment color, we will have varying colors.
            # We can use one LineCollection per (linewidth, linestyle) and pass a list of colors.

            grouped_segments: Dict[Tuple[float, str], List[Tuple[np.ndarray, Any]]] = {}

            for segment in item_state.segments:
                coords = None
                if segment.world is not None and wcs is not None:
                    coords = _world_coords_to_pixel(segment.world, wcs, state.world_frame)
                if coords is None and segment.pixels is not None and np.asarray(segment.pixels).size:
                    coords = np.asarray(segment.pixels, dtype=float)
                if coords is None or coords.ndim != 2 or coords.shape[0] < 2:
                    continue
                finite_mask = np.isfinite(coords).all(axis=1)
                if np.count_nonzero(finite_mask) < 2:
                    continue
                coords = coords[finite_mask]
                
                # Check if we need to update transformed pixels back to segment for cache? 
                # The original code did: segment.pixels = coords
                # And also back-calculated world. We should probably preserve that behavior 
                # if it's needed for canonicalization, but here we are just drawing.
                # Strictly speaking _draw_overlay_state shouldn't mutate state permanently ideally, 
                # but let's keep it consistent if needed. 
                # Actually, skipping mutation for drawing optimization is safer.

                color = None
                if segment.color is not None:
                    color = segment.color
                elif color_mode == "rainbow" and cmap is not None and level_norm is not None:
                    try:
                        color = np.asarray(cmap(level_norm(segment.level)), dtype=float)
                    except Exception:
                        color = None
                if color is None:
                    color = overlay_params.color

                if isinstance(color, np.ndarray):
                    color_to_apply = tuple(color.tolist())
                else:
                    color_to_apply = color

                linewidth = overlay_params.linewidth or 1.0
                linestyle = getattr(segment, "linestyle", None) or ("dashed" if segment.level < 0 else "solid")

                key = (linewidth, linestyle)
                if key not in grouped_segments:
                    grouped_segments[key] = []
                grouped_segments[key].append((coords, color_to_apply))

            # Create LineCollections
            for (lw, ls), seg_data in grouped_segments.items():
                if not seg_data:
                    continue
                pix_segments = [s[0] for s in seg_data]
                colors = [s[1] for s in seg_data]
                
                lc = LineCollection(
                    pix_segments,
                    colors=colors,
                    linewidths=lw,
                    linestyles=ls
                )
                ax.add_collection(lc)
                new_artists.append(lc)

        if new_artists:
            self._overlay_artists[overlay_id] = new_artists

    def has_generated_state(self) -> bool:
        return self._state is not None

    def has_overlays(self) -> bool:
        return bool(self._overlay_states)

    def refresh_overlays(self) -> None:
        self._replot_overlay_states()

    def overlay_states(self) -> List[ContourState]:
        for state in self._overlay_states:
            if state.overlay_id is None:
                state.overlay_id = self._make_overlay_id()
        return list(self._overlay_states)

    def clear_overlay(self, overlay_id: str) -> bool:
        removed = False
        remaining: List[ContourState] = []
        for state in self._overlay_states:
            if state.overlay_id == overlay_id:
                removed = True
            else:
                remaining.append(state)
        if removed:
            self._overlay_states = remaining
            self._remove_overlay_artists([overlay_id])
        return removed

    def clear_overlays(self, overlay_ids: Iterable[str]) -> bool:
        overlay_set = set(overlay_ids)
        if not overlay_set:
            return False
        kept_states: List[ContourState] = []
        removed_ids: List[str] = []
        for state in self._overlay_states:
            if state.overlay_id in overlay_set:
                removed_ids.append(state.overlay_id)
            else:
                kept_states.append(state)
        if not removed_ids:
            return False
        self._overlay_states = kept_states
        self._remove_overlay_artists(removed_ids)
        return True

    def update_overlay_style(self, color: str, linewidth: float, overlay_ids: Optional[Iterable[str]] = None) -> bool:
        if not self._overlay_states:
            return False
        target_ids = None if overlay_ids is None else set(overlay_ids)
        changed = False
        preserve_colors = (color.lower() == "original")
        for state in self._overlay_states:
            if target_ids is not None and state.overlay_id not in target_ids:
                continue
            
            # Always allow linewidth changes
            if state.parameters.linewidth != linewidth:
                state.parameters.linewidth = linewidth
                changed = True
            
            if preserve_colors:
                # Restore original colors if available
                for item_state in state.items:
                    for segment in item_state.segments:
                        if segment.original_color is not None and segment.color is None:
                            # Restore from backup
                            segment.color = segment.original_color.copy()
                            changed = True
                # Set parameters color to a valid color for rendering (fallback)
                if state.parameters.color == "original":
                    state.parameters.color = "white"  # Fallback for rendering
            else:
                # Backup original colors before clearing (only if not already backed up)
                for item_state in state.items:
                    for segment in item_state.segments:
                        if segment.color is not None and segment.original_color is None:
                            segment.original_color = segment.color.copy()
                        if segment.color is not None:
                            segment.color = None
                            changed = True
                # Update color parameter
                if state.parameters.color != color:
                    state.parameters.color = color
                    changed = True
                    
        if changed:
            self._replot_overlay_states()
        return changed

    def _ensure_levels(
        self,
        params: ContourParameters,
        data_min: float,
        data_max: float,
    ) -> Optional[List[float]]:
        vmin = params.level_min if params.level_min is not None else data_min
        vmax = params.level_max if params.level_max is not None else data_max
        if vmin > vmax:
            vmin, vmax = vmax, vmin
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            return None
        if math.isclose(vmin, vmax):
            return [vmin]
        if params.level_step is None or params.level_step == 0:
            return [vmin, vmax]
        step = params.level_step
        if step < 0:
            step = abs(step)
        if step == 0:
            step = (vmax - vmin) / 7.0 if vmax != vmin else 1.0
        count = int(math.floor((vmax - vmin) / step)) + 1
        if count < 1:
            count = 1
        levels = [vmin + i * step for i in range(count)]
        ordered: List[float] = []
        for val in levels:
            if not ordered or not math.isclose(ordered[-1], val):
                ordered.append(val)
        return ordered


    def update(self, params: ContourParameters) -> Optional[ContourState]:        
        items = self._provider()
        if not items:
            self._clear_generated()
            self._replot_overlay_states()
            return None

        # Get viewer and hide all overlay artists to prevent burn-in during ax.contour
        viewer_instance = None
        hidden_artists: List[object] = []
        vis_states: Dict[object, bool] = {}
        hidden_regions: List[object] = []

        if self._owner is not None:
            owner = self._owner()
            # FITSViewer or SubWindow
            if hasattr(owner, 'plane'):
                viewer_instance = owner
                plane = owner.plane
                state = getattr(owner, 'state', None)
                
                # 1. Collect standard artists (lines, labels)
                # Prefer viewer state, fall back to Common
                if state is not None:
                    if state.vline: hidden_artists.append(state.vline)
                    if state.hline: hidden_artists.append(state.hline)
                    if state.chlabel: hidden_artists.append(state.chlabel)
                    if state.hpbw and hasattr(state.hpbw, 'ellipse') and state.hpbw.ellipse:
                        hidden_artists.append(state.hpbw.ellipse)

                if plane == 'xy':
                    # Also hide PVD lines (must access via main_window or owner safely)
                    main_window = getattr(owner, 'parent', owner)  # Handle SubWindow or MainWindow
                    if hasattr(main_window, 'control_panel') and main_window.control_panel and main_window.control_panel.pvd_panel:
                        pvd = main_window.control_panel.pvd_panel
                        if pvd.arrow_artist: hidden_artists.append(pvd.arrow_artist)
                        hidden_artists.extend(pvd.width_indicators)
                        if pvd.pos_indicator_on_arrow: hidden_artists.append(pvd.pos_indicator_on_arrow)

                # 2. Hide regions
                if hasattr(owner, 'region_manager'):
                    hidden_regions = owner.region_manager.prepare_for_background_capture()

                # 3. Store visibility and hide the artists
                for artist in hidden_artists:
                    if artist:
                        try:
                            vis_states[artist] = artist.get_visible()
                            artist.set_visible(False)
                        except Exception: pass # Ignore errors
        
        # Pre-compute global min/max for level generation.
        mins = []
        maxs = []
        for item in items:
            arr = np.asarray(item.data, dtype=float)
            if arr.size == 0:
                continue
            finite_mask = np.isfinite(arr)
            if not finite_mask.any():
                continue
            finite_vals = arr[finite_mask]
            mins.append(float(finite_vals.min()))
            maxs.append(float(finite_vals.max()))

        try:
            if not mins or not maxs:
                self._clear_generated()
                self._replot_overlay_states()
                return None

            data_min = float(np.nanmin(mins))
            data_max = float(np.nanmax(maxs))
            levels = self._ensure_levels(params, data_min, data_max)
            if levels is None or len(levels) == 0:
                self._clear_generated()
                self._replot_overlay_states()
                return None

            use_cmap = (params.color or "").lower() == "rainbow"
            cmap = cm.get_cmap("rainbow") if use_cmap else None
            
            new_gen_artists: List[object] = []
            new_last_contour_sets: Dict[str, "QuadContourSet"] = {}

            for item in items:
                arr = np.asarray(item.data, dtype=float)
                if arr.ndim != 2 or arr.size == 0:
                    continue
                arr = np.where(np.isfinite(arr), arr, np.nan)

                if params.smoothing and params.smoothing > 0 and gaussian_filter is not None:
                    sigma = max(params.smoothing, 0.0)
                    arr = gaussian_filter(arr, sigma=sigma)

                finite_mask = np.isfinite(arr)
                if not finite_mask.any():
                    continue
                arr_min = float(np.nanmin(arr))
                arr_max = float(np.nanmax(arr))
                if math.isclose(arr_min, arr_max):
                    continue

                drawn_artists: List[object] = []
                try:
                    ny, nx = arr.shape
                    x_coords = np.arange(nx, dtype=float)
                    y_coords = np.arange(ny, dtype=float)
                    if item.extent is not None:
                        try:
                            xmin, xmax, ymin, ymax = item.extent
                        except Exception:
                            xmin = xmax = ymin = ymax = None
                        else:
                            if nx > 0 and xmin is not None and xmax is not None:
                                dx = (xmax - xmin) / float(nx)
                                x_coords = xmin + (np.arange(nx, dtype=float) + 0.5) * dx
                            if ny > 0 and ymin is not None and ymax is not None:
                                dy = (ymax - ymin) / float(ny)
                                y_coords = ymin + (np.arange(ny, dtype=float) + 0.5) * dy
                    contour_kwargs = dict(
                        levels=levels,
                        linewidths=params.linewidth,
                        origin=item.origin,
                    )
                    if use_cmap:
                        contour_kwargs["cmap"] = cmap
                    else:
                        # Fallback 'original' to 'white' for generated contours
                        effective_color = params.color if params.color.lower() != "original" else "white"
                        contour_kwargs["colors"] = effective_color
                    
                    # This call now happens while all overlays are hidden
                    contour = item.ax.contour(
                        x_coords,
                        y_coords,
                        arr,
                        **contour_kwargs,
                    )

                    for coll in contour.collections:
                        drawn_artists.append(coll)
                        
                        try:
                            level = coll.get_array()[0]
                            if level < 0:
                                coll.set_linestyle("dashed")
                            else:
                                coll.set_linestyle("solid")
                        except Exception:
                            pass

                    new_gen_artists.extend(drawn_artists)
                    new_last_contour_sets[item.label] = contour

                except Exception:
                    continue

            old_artists = list(self._gen_artists)
            for artist in old_artists:
                try:
                    artist.remove()
                except Exception:
                    pass
            
            self._gen_artists = new_gen_artists
            self._last_contour_sets = new_last_contour_sets
            self._state = None
            
            self._active = True
            self._last_parameters = params.clone()

        finally:
            # Restore cursor/overlay visibility *before* emitting the update signal
            if viewer_instance is not None:
                for artist, visible in vis_states.items():
                    if artist:
                        try: artist.set_visible(visible)
                        except Exception: pass
                if hasattr(viewer_instance, 'region_manager') and hidden_regions:
                    viewer_instance.region_manager.restore_after_background_capture(hidden_regions)

        return None

    def redraw_from_state(self, state: ContourState) -> None:
        """Store and plot overlay contours from a saved state."""
        if state is None or not state.items:
            return

        canonical_state = self._canonicalize_overlay_state(state)
        if canonical_state is None:
            return
        self._overlay_states.append(canonical_state)
        self._replot_overlay_states()

    def export_state(self) -> Optional[ContourState]:
        if self._state is not None:
            return self._state
        
        if not self._last_contour_sets or self._last_parameters is None:
            return None
        
        params = self._last_parameters
        first_key = list(self._last_contour_sets.keys())[0]
        levels = self._last_contour_sets[first_key].levels
        
        segments_per_item: List[ContourItemState] = []
        state_world_frame: Optional[str] = None

        use_cmap = (params.color or "").lower() == "rainbow"
        cmap = cm.get_cmap("rainbow") if use_cmap else None
        level_norm = (
            mcolors.Normalize(vmin=min(levels), vmax=max(levels)) if use_cmap else None
        )

        provider_items = list(self._provider())
        items_by_label = {item.label: item for item in provider_items if item.label}

        for item_label, contour in self._last_contour_sets.items():
            
            target_item = items_by_label.get(item_label)
            if target_item is None:
                continue 

            segments = ContourItemState(item_label=item_label)
            wcs = getattr(target_item.ax, "wcs", None)

            for level, collection in zip(contour.levels, contour.collections):
                linestyle = "solid"
                if level < 0:
                    linestyle = "dashed"
                
                collection_color = None
                try:
                    colors_arr = collection.get_colors()
                    if colors_arr is not None and len(colors_arr):
                        collection_color = np.asarray(colors_arr[0], dtype=float)
                except Exception:
                    collection_color = None

                if collection_color is None and use_cmap and level_norm is not None:
                    try:
                        collection_color = np.asarray(
                            cmap(level_norm(level)), dtype=float
                        )
                    except Exception:
                        collection_color = None

                frame_name = _frame_name_from_wcs(wcs) if wcs is not None else None
                if state_world_frame is None and frame_name is not None:
                    state_world_frame = frame_name

                for path in collection.get_paths():
                    polylines = self._extract_path_polylines(path)
                    for poly_arr in polylines:
                        if poly_arr.size == 0 or poly_arr.ndim != 2:
                            continue
                        pixels = np.asarray(poly_arr, dtype=float)
                        finite_mask = np.isfinite(pixels).all(axis=1)
                        if np.count_nonzero(finite_mask) < 2:
                            continue
                        pixels = pixels[finite_mask]
                        world = _pixel_coords_to_world(pixels, wcs)
                        for pix_chunk, world_chunk in self._split_polyline(pixels, world):
                            if world_chunk is None and wcs is not None:
                                world_chunk = _pixel_coords_to_world(pix_chunk, wcs)
                            if frame_name is not None and state_world_frame is None:
                                state_world_frame = frame_name
                            color_copy = None
                            if collection_color is not None:
                                try:
                                    color_copy = np.asarray(collection_color, dtype=float).copy()
                                except Exception:
                                    color_copy = collection_color
                            segments.segments.append(
                                ContourSegment(
                                    level=float(level),
                                    world=world_chunk,
                                    pixels=pix_chunk,
                                    color=color_copy,
                                    original_color=color_copy.copy() if color_copy is not None else None,
                                    linestyle=linestyle,
                                )
                            )
            
            if segments.segments:
                segments_per_item.append(segments)
        

        if not segments_per_item:
            return None

        state = ContourState(
            layer_id=self.id,
            plane=self.plane,
            label=self.label,
            parameters=params.clone(),
            levels=list(levels),
            items=segments_per_item,
            world_frame=state_world_frame,
        )
        
        self._state = state
        return self._state

    def get_generated_artists(self) -> List[object]:
        return self._gen_artists

    def get_overlay_artists(self) -> List[object]:
        all_artists = []
        for artists_list in self._overlay_artists.values():
            all_artists.extend(artists_list)
        return all_artists


class ContourManager(QObject):
    """
    Singleton-style manager coordinating contour layers and parameters across the UI.
    """

    targets_changed = pyqtSignal()
    contour_updated = pyqtSignal(str)

    _instance: Optional["ContourManager"] = None

    @classmethod
    def instance(cls) -> "ContourManager":
        if cls._instance is None:
            cls._instance = ContourManager()
        return cls._instance

    def __init__(self):
        super().__init__()
        self._layers: Dict[str, ContourLayer] = {}
        self._params = ContourParameters()

    def registered_layers(self) -> Dict[str, ContourLayer]:
        return dict(self._layers)

    def entries(self) -> List[ContourListEntry]:
        result: List[ContourListEntry] = []
        for layer_id, layer in self._layers.items():
            base_label = layer.label if layer.label else layer_id
            result.append(
                ContourListEntry(
                    id=layer_id,
                    label=base_label,
                    plane=layer.plane,
                    kind="layer",
                    parent_id=None,
                    active=layer.is_active(),
                )
            )
            for state in layer.overlay_states():
                entry_label = state.label or f"{base_label} overlay"
                overlay_id = state.overlay_id
                if overlay_id is None:
                    overlay_id = layer._make_overlay_id()
                    state.overlay_id = overlay_id
                result.append(
                    ContourListEntry(
                        id=overlay_id,
                        label=entry_label,
                        plane=layer.plane,
                        kind="overlay",
                        parent_id=layer_id,
                        active=True,
                    )
                )
        return result

    def layer_ids_for_owner(self, owner: object) -> List[str]:
        if owner is None:
            return []
        result: List[str] = []
        for layer_id, layer in self._layers.items():
            if layer._owner is None:
                continue
            ref = layer._owner()
            if ref is owner:
                result.append(layer_id)
        return result

    def layer_data_bounds(self, layer_id: str) -> Optional[Tuple[float, float]]:
        layer = self._layers.get(layer_id)
        if layer is None:
            return None
        return layer.data_bounds()

    def register_layer(
        self,
        layer_id: str,
        label: str,
        plane: Optional[str],
        provider: ContourItemProvider,
        owner: object,
    ) -> str:
        if layer_id in self._layers:
            raise ValueError(f"Contour layer '{layer_id}' already registered")
        layer = ContourLayer(self, layer_id, label, plane, provider, owner)
        self._layers[layer_id] = layer
        self.targets_changed.emit()
        return layer_id

    def default_save_basename(
        self,
        *,
        layer_id: Optional[str] = None,
        overlay_id: Optional[str] = None,
    ) -> Optional[str]:
        layer: Optional[ContourLayer] = None
        if layer_id is not None:
            layer = self._layers.get(layer_id)
        elif overlay_id is not None:
            layer = self._layer_for_overlay(overlay_id)
        if layer is None:
            return None
        filename = layer.default_source_filename()
        if filename:
            basename = os.path.basename(filename)
            root, _ = os.path.splitext(basename)
            return root or basename
        label = layer.label or layer.id
        if not label:
            return None
        base_label = label.split("[")[0].strip()
        root, _ = os.path.splitext(base_label)
        return root or base_label

    def unregister_layer(self, layer_id: str) -> None:
        layer = self._layers.pop(layer_id, None)
        if layer is None:
            return
        layer.clear()
        self.targets_changed.emit()

    def _layer_for_overlay(self, overlay_id: Optional[str]) -> Optional[ContourLayer]:
        if overlay_id is None:
            return None
        base_id = None
        if "::overlay::" in overlay_id:
            base_id = overlay_id.split("::overlay::", 1)[0]
        if base_id:
            layer = self._layers.get(base_id)
            if layer is not None:
                for state in layer.overlay_states():
                    if state.overlay_id == overlay_id:
                        return layer
        for layer in self._layers.values():
            for state in layer.overlay_states():
                if state.overlay_id == overlay_id:
                    return layer
        return None

    def update_parameters(self, params: ContourParameters) -> None:
        self._params = params.clone()

    def get_parameters(self) -> ContourParameters:
        return self._params.clone()

    def rename_layer(self, layer_id: str, label: str) -> None:
        layer = self._layers.get(layer_id)
        if layer is None:
            return
        layer.label = label
        self.targets_changed.emit()

    def apply_to_layers(self, layer_ids: Iterable[str]) -> Dict[str, Optional[ContourState]]:
        results: Dict[str, Optional[ContourState]] = {}
        for layer_id in layer_ids:
            layer = self._layers.get(layer_id)
            if layer is None or not layer.owner_alive():
                continue
            
            state = layer.update(self._params) 
            results[layer_id] = state

            self.contour_updated.emit(layer_id)
            
        return results

    def clear_layers(self, layer_ids: Iterable[str]) -> None:
        changed = False
        for layer_id in layer_ids:
            layer = self._layers.get(layer_id)
            if layer is None:
                continue
            if layer.clear_generated():
                self.contour_updated.emit(layer_id)
                changed = True
        if changed:
            self.targets_changed.emit()

    def clear_overlays(self, overlay_ids: Iterable[str]) -> None:
        overlays_by_layer: Dict[str, Set[str]] = {}
        for overlay_id in overlay_ids:
            layer = self._layer_for_overlay(overlay_id)
            if layer is None:
                continue
            overlays_by_layer.setdefault(layer.id, set()).add(overlay_id)
        if not overlays_by_layer:
            return
        changed_any = False
        for layer_id, ids in overlays_by_layer.items():
            layer = self._layers.get(layer_id)
            if layer is None:
                continue
            if layer.clear_overlays(ids):
                self.contour_updated.emit(layer_id)
                changed_any = True
        if changed_any:
            self.targets_changed.emit()

    def refresh_layer(self, layer_id: str) -> None:
        layer = self._layers.get(layer_id)
        if layer is None:
            return

        # Get the parameters that were last used to generate contours on this layer
        last_params = layer._last_parameters

        if last_params is not None:
            # If parameters exist, re-apply them.
            # This updates data-dependent (generated) contours only.
            layer.update(last_params) 
            self.contour_updated.emit(layer_id)

    def update_overlay_style(self, layer_ids: Iterable[str], color: str, linewidth: float) -> None:
        changed_any = False
        for layer_id in layer_ids:
            layer = self._layers.get(layer_id)
            if layer is None:
                continue
            if layer.update_overlay_style(color, linewidth):
                self.contour_updated.emit(layer_id)
                changed_any = True
        if changed_any:
            self.targets_changed.emit()

    def apply_overlays(self, overlay_ids: Iterable[str], color: str, linewidth: float) -> None:
        overlays_by_layer: Dict[str, Set[str]] = {}
        for overlay_id in overlay_ids:
            layer = self._layer_for_overlay(overlay_id)
            if layer is None:
                continue
            overlays_by_layer.setdefault(layer.id, set()).add(overlay_id)
        if not overlays_by_layer:
            return
        changed_any = False
        for layer_id, ids in overlays_by_layer.items():
            layer = self._layers.get(layer_id)
            if layer is None:
                continue
            if layer.update_overlay_style(color, linewidth, ids):
                self.contour_updated.emit(layer_id)
                changed_any = True
        if changed_any:
            self.targets_changed.emit()

    def export_layer_state(self, layer_id: str) -> Optional[ContourState]:
        layer = self._layers.get(layer_id)
        if layer is None:
            return None
        return layer.export_state()

    def export_overlay_state(self, overlay_id: str) -> Optional[ContourState]:
        layer = self._layer_for_overlay(overlay_id)
        if layer is None:
            return None
        for state in layer.overlay_states():
            if state.overlay_id == overlay_id:
                return copy.deepcopy(state)
        return None

    def import_layer_state(self, layer_id: str, state: ContourState) -> None:
        layer = self._layers.get(layer_id)
        if layer is None:
            return
        # If state lacks plane info, default it to the target layer's plane
        if state.plane is None:
            try:
                state.plane = layer.plane
            except Exception:
                pass
        layer.redraw_from_state(state)
        self.targets_changed.emit()

    def import_overlay_state(self, layer_id: str, state: ContourState) -> Optional[str]:
        """
        Import an overlay contour state into the specified layer.

        Returns the overlay_id assigned to the imported state (or None on failure).
        """
        layer = self._layers.get(layer_id)
        if layer is None or state is None:
            return None

        try:
            state.layer_id = layer.id
            if state.plane is None:
                state.plane = layer.plane
        except Exception:
            pass

        layer.redraw_from_state(state)
        self.contour_updated.emit(layer_id)
        self.targets_changed.emit()

        try:
            overlay_id = state.overlay_id
        except Exception:
            overlay_id = None
        return overlay_id


def serialize_state_to_json(state: ContourState) -> str:
    """
    Serialize a contour state to a JSON string.
    """

    def _segment_to_dict(segment: ContourSegment) -> Dict[str, object]:
        data: Dict[str, object] = {"level": segment.level}
        if segment.world is not None:
            data["world"] = segment.world.tolist()
        if segment.pixels is not None:
            data["pixels"] = segment.pixels.tolist()
        if segment.color is not None:
            data["color"] = segment.color.tolist()
        if segment.linestyle:
            data["linestyle"] = segment.linestyle
        return data

    payload = {
        "format": "takefits.contour",
        "version": 1,
        "layer_id": state.layer_id,
        "plane": state.plane,
        "label": state.label,
        "parameters": {
            "level_min": state.parameters.level_min,
            "level_max": state.parameters.level_max,
            "level_step": state.parameters.level_step,
            "smoothing": state.parameters.smoothing,
            "linewidth": state.parameters.linewidth,
            "color": state.parameters.color,
        },
        "levels": list(state.levels),
        "items": [
            {
                "label": item.item_label,
                "segments": [_segment_to_dict(segment) for segment in item.segments],
            }
            for item in state.items
        ],
        "world_frame": state.world_frame,
    }
    return json.dumps(payload, indent=2)


def deserialize_state_from_json(data: str) -> ContourState:
    payload = json.loads(data)
    if payload.get("format") != "takefits.contour":
        raise ValueError("Unsupported contour file format")
    params_payload = payload.get("parameters", {})
    params = ContourParameters(
        level_min=params_payload.get("level_min"),
        level_max=params_payload.get("level_max"),
        level_step=params_payload.get("level_step"),
        smoothing=float(params_payload.get("smoothing", 0.0)),
        linewidth=float(params_payload.get("linewidth", 1.0)),
        color=params_payload.get("color", "white"),
    )
    items_payload = payload.get("items", [])
    items: List[ContourItemState] = []
    for item_payload in items_payload:
        item_state = ContourItemState(item_label=item_payload.get("label", ""))
        for segment_payload in item_payload.get("segments", []):
            world = segment_payload.get("world")
            pixels = segment_payload.get("pixels")
            color = segment_payload.get("color")
            linestyle = segment_payload.get("linestyle", "solid")
            segment = ContourSegment(
                level=float(segment_payload.get("level", 0.0)),
                world=np.asarray(world, dtype=float) if world is not None else None,
                pixels=np.asarray(pixels, dtype=float) if pixels is not None else None,
                color=np.asarray(color, dtype=float) if color is not None else None,
                linestyle=linestyle,
            )
            item_state.segments.append(segment)
        items.append(item_state)
    return ContourState(
        layer_id=payload.get("layer_id", ""),
        plane=payload.get("plane"),
        label=payload.get("label", ""),
        parameters=params,
        levels=payload.get("levels", []),
        items=items,
        world_frame=payload.get("world_frame"),
    )


def write_state_to_ds9(path: str, state: ContourState) -> None:
    level_segments: Dict[float, List[ContourSegment]] = {}
    for item in state.items:
        for segment in item.segments:
            level_segments.setdefault(float(segment.level), []).append(segment)

    if not level_segments:
        raise ValueError("No contour segments available to save.")

    levels_sorted = sorted(level_segments.keys())

    use_world = True
    for segments in level_segments.values():
        for segment in segments:
            if segment.world is None or np.asarray(segment.world).size == 0:
                use_world = False
                break
        if not use_world:
            break

    frame_keyword = _ds9_keyword_from_frame(state.world_frame) if use_world else None
    coord_line = (frame_keyword or "FK5").lower() if use_world else "image"

    color = (state.parameters.color or "white").lower()
    if color == "rainbow":
        color = "white"
    linewidth = float(state.parameters.linewidth or 1.0)
    linewidth_ds9 = max(int(round(linewidth)), 1)
    dash = "no"
    for segments in level_segments.values():
        for segment in segments:
            linestyle = (segment.linestyle or "").lower()
            if "dash" in linestyle:
                dash = "yes"
                break
        if dash == "yes":
            break

    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    levels_comment = " ".join(f"{level:.10g}" for level in levels_sorted)

    lines: List[str] = []
    lines.append("# Contour file format: DS9 version 7.5")
    lines.append("# generated_by=takefits")
    lines.append(f"# timestamp={timestamp}")
    if levels_comment:
        lines.append(f"# levels=( {levels_comment} )")
    lines.append(f"global color={color} width={linewidth_ds9} dash={dash} dashlist=8 3")
    lines.append(coord_line)

    for level in levels_sorted:
        segments = level_segments[level]
        if not segments:
            continue
        lines.append(f"level={level:.10f}")
        for segment in segments:
            coords_arr = None
            if use_world and segment.world is not None and np.asarray(segment.world).size:
                coords_arr = np.asarray(segment.world, dtype=float)
            elif segment.pixels is not None and np.asarray(segment.pixels).size:
                coords_arr = np.asarray(segment.pixels, dtype=float)
            if coords_arr is None or coords_arr.ndim != 2 or coords_arr.shape[0] == 0:
                continue

            output_coords = np.asarray(coords_arr, dtype=float)
            if not use_world:
                output_coords = output_coords + 1.0  # DS9 pixel convention is 1-based

            lines.append("(")
            for x, y in output_coords:
                lines.append(f" {x:.10f} {y:.10f}")
            lines.append(")")

    lines.append("# END")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def read_state_from_ds9(path: str) -> ContourState:
    with open(path, "r", encoding="utf-8") as handle:
        raw_lines = [line.rstrip("\n") for line in handle]

    coordinate_mode = "PIXEL"
    coordinate_keyword: Optional[str] = None
    color = "white"
    width = 1.0

    segments: List[Tuple[Optional[float], List[Tuple[float, float]]]] = []
    current_level: Optional[float] = None
    current_coords: List[Tuple[float, float]] = []
    collecting_parentheses = False

    def _store_segment() -> None:
        nonlocal current_coords
        if current_level is not None and current_coords:
            segments.append((current_level, list(current_coords)))
        current_coords = []

    for raw in raw_lines:
        line = raw.strip()
        if not line:
            if not collecting_parentheses:
                _store_segment()
            continue

        lower = line.lower()

        if lower.startswith("#"):
            if "frame" in lower:
                tokens = line.replace("#", " ").replace("=", " ").split()
                for token in tokens:
                    upper_token = token.upper()
                    if upper_token in {"FK5", "ICRS", "FK4", "B1950", "GALACTIC"}:
                        coordinate_mode = "WORLD"
                        coordinate_keyword = upper_token
            continue

        if lower == "contour":
            continue
        if lower.startswith("global"):
            tokens = line.split()
            for token in tokens:
                if "=" not in token:
                    continue
                key, value = token.split("=", 1)
                key = key.lower()
                value = value.strip()
                if key == "color" and value:
                    color = value.lower()
                elif key == "width":
                    try:
                        width = float(value)
                    except ValueError:
                        pass
            continue
        if lower in {"wcs", "world"}:
            coordinate_mode = "WORLD"
            continue
        if lower in {"fk5", "icrs", "galactic", "fk4", "b1950", "j2000"}:
            coordinate_mode = "WORLD"
            coordinate_keyword = lower.upper()
            continue
        if lower in {"pixel", "image", "physical"}:
            coordinate_mode = "PIXEL"
            coordinate_keyword = "PIXEL"
            continue
        if lower.startswith("level"):
            if not collecting_parentheses:
                _store_segment()
            value_str = ""
            if "=" in line:
                value_str = line.split("=", 1)[1].strip()
            else:
                parts = line.split()
                if len(parts) > 1:
                    value_str = parts[1]
            try:
                current_level = float(value_str)
            except ValueError:
                current_level = None
            continue
        if line.startswith("("):
            collecting_parentheses = True
            current_coords = []
            continue
        if line.startswith(")"):
            _store_segment()
            collecting_parentheses = False
            continue
        if lower == "end":
            _store_segment()
            break

        parts = line.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            x = float(parts[0])
            y = float(parts[1])
        except ValueError:
            continue
        current_coords.append((x, y))

    _store_segment()

    if not segments:
        raise ValueError("No contour data found in DS9 file.")

    item_state = ContourItemState(item_label="DS9")
    unique_levels: List[float] = []

    for level, coords in segments:
        if level is None or not coords:
            continue
        if not any(math.isclose(existing, level) for existing in unique_levels):
            unique_levels.append(level)
        array = np.asarray(coords, dtype=float)
        if coordinate_mode == "PIXEL":
            array = array - 1.0
            segment = ContourSegment(level=level, pixels=array)
        else:
            segment = ContourSegment(level=level, world=array)
        item_state.segments.append(segment)

    params = ContourParameters(
        level_min=None,
        level_max=None,
        level_step=None,
        smoothing=0.0,
        linewidth=float(width),
        color=color,
    )

    world_frame = coordinate_keyword or ("WORLD" if coordinate_mode == "WORLD" else "PIXEL")

    return ContourState(
        layer_id="",
        plane=None,
        label="Imported Contour",
        parameters=params,
        levels=unique_levels,
        items=[item_state],
        world_frame=world_frame,
    )
