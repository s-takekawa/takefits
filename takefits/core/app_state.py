"""
AppState: PyQt-free application state container for headless operations.

This module provides a state container that can be used by usecase functions
without requiring PyQt. It holds loaded FITS data, WCS, and cursor state.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
import uuid
import numpy as np


@dataclass
class CursorState:
    """Cursor position state (pixel and world coordinates)."""
    xpix: int = 0
    ypix: int = 0
    zpix: int = 0
    spix: int = 0
    world_x: Optional[float] = None
    world_y: Optional[float] = None
    world_z: Optional[float] = None
    world_s: Optional[float] = None


@dataclass
class ViewState:
    """View state for a single plane."""
    xlim: Optional[Tuple[float, float]] = None
    ylim: Optional[Tuple[float, float]] = None
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    cmap: str = "viridis"
    # Color settings for display
    log_scale: bool = False
    gamma: float = 1.0
    invert_cmap: bool = False


@dataclass
class RegionSpec:
    """Specification for a region (circle, rectangle, ellipse)."""
    type: str  # "circle", "rectangle", "ellipse"
    center_x: float
    center_y: float
    # For circle: radius
    # For rectangle: width, height
    # For ellipse: semi_major, semi_minor, angle
    params: Dict[str, float] = field(default_factory=dict)
    region_id: Optional[int | str] = None
    plane: str = "xy"
    label: str = ""
    style: Dict[str, Any] = field(default_factory=dict)
    world: Dict[str, Any] = field(default_factory=dict)
    world_frame: str = ""

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "type": self.type,
            "center_x": float(self.center_x),
            "center_y": float(self.center_y),
            "params": dict(self.params),
        }
        if self.region_id is not None:
            payload["id"] = self.region_id
        if self.plane:
            payload["plane"] = str(self.plane)
        if self.label:
            payload["label"] = str(self.label)
        if self.style:
            payload["style"] = dict(self.style)
        if self.world:
            payload["world"] = dict(self.world)
        if self.world_frame:
            payload["world_frame"] = str(self.world_frame)
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RegionSpec":
        if not isinstance(payload, dict):
            raise TypeError("RegionSpec.from_dict expects a mapping payload")
        # Accept either 'type' or legacy 'kind' key.
        region_type = str(payload.get("type") or payload.get("kind") or "circle")
        center_x = float(payload.get("center_x", 0.0))
        center_y = float(payload.get("center_y", 0.0))
        params = dict(payload.get("params") or {})
        region_id = payload.get("id", payload.get("region_id"))
        plane = str(payload.get("plane") or "xy")
        label = str(payload.get("label") or "")
        style = dict(payload.get("style") or {})
        world = dict(payload.get("world") or {})
        world_frame = str(payload.get("world_frame") or "")
        return cls(
            type=region_type,
            center_x=center_x,
            center_y=center_y,
            params=params,
            region_id=region_id,
            plane=plane,
            label=label,
            style=style,
            world=world,
            world_frame=world_frame,
        )


def _new_marker_id() -> str:
    return uuid.uuid4().hex


@dataclass
class MarkerSpec:
    """PyQt-free marker specification for headless state/actions."""

    pixel: Tuple[float, ...]
    marker_id: str = field(default_factory=_new_marker_id)
    plane: str = "xy"
    kind: str = "symbol"
    world: Optional[Tuple[float, ...]] = None
    world_frame: str = ""
    label: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    style: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "id": self.marker_id,
            "plane": str(self.plane),
            "kind": str(self.kind),
            "pixel": list(map(float, self.pixel)),
            "label": str(self.label),
            "metadata": dict(self.metadata),
            "style": dict(self.style),
        }
        if self.world is not None:
            payload["world"] = list(map(float, self.world))
        if self.world_frame:
            payload["world_frame"] = str(self.world_frame)
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MarkerSpec":
        if not isinstance(payload, dict):
            raise TypeError("MarkerSpec.from_dict expects a mapping payload")
        marker_id = str(payload.get("id") or payload.get("marker_id") or _new_marker_id())
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
        world_frame = str(payload.get("world_frame") or "")
        label = str(payload.get("label") or "")
        metadata = dict(payload.get("metadata") or {})
        style = dict(payload.get("style") or {})
        return cls(
            pixel=pixel_tuple,
            marker_id=marker_id,
            plane=plane,
            kind=kind,
            world=world_tuple,
            world_frame=world_frame,
            label=label,
            metadata=metadata,
            style=style,
        )


@dataclass
class AppState:
    """
    PyQt-free application state container.

    This holds all state needed for headless operations:
    - Loaded FITS data, header, WCS
    - Spectral metadata
    - Cursor position
    - View state for each plane
    - Regions and markers

    This class is designed to be serializable and can be used
    by usecase functions without any PyQt dependencies.
    """
    # FITS data
    data: Optional[np.ndarray] = None
    header: Optional[Any] = None  # FITS header (dict-like)
    wcs: Optional[Any] = None  # astropy WCS object

    # Source file path
    filepath: Optional[str] = None

    # Spectral metadata from fits_loader
    spectral_metadata: Dict[str, Any] = field(default_factory=dict)

    # Current slice index for each axis (z is the channel/velocity axis)
    current_z: int = 0
    current_s: int = 0  # 4th axis if present

    # Cursor state
    cursor: CursorState = field(default_factory=CursorState)

    # View state for each plane
    view_xy: ViewState = field(default_factory=ViewState)
    view_xz: ViewState = field(default_factory=ViewState)
    view_zy: ViewState = field(default_factory=ViewState)

    # Regions (for integration, cutout, etc.)
    regions: List[RegionSpec] = field(default_factory=list)
    markers: List[MarkerSpec] = field(default_factory=list)

    # Integration range (velocity/channel range for moment maps)
    integ_min: Optional[float] = None  # World coordinate (e.g., km/s)
    integ_max: Optional[float] = None
    integ_min_pix: Optional[float] = None  # Pixel coordinate (can be float)
    integ_max_pix: Optional[float] = None

    # PV slice endpoints (pixel coordinates)
    pv_x0: Optional[float] = None
    pv_y0: Optional[float] = None
    pv_x1: Optional[float] = None
    pv_y1: Optional[float] = None
    pv_width: float = 0.0

    def get_view_state(self, plane: str) -> ViewState:
        """Get view state for a specific plane."""
        if plane == 'xy':
            return self.view_xy
        elif plane == 'xz':
            return self.view_xz
        elif plane == 'zy':
            return self.view_zy
        raise ValueError(f"Unknown plane: {plane}")

    def get_slice_2d(self, plane: str = 'xy') -> Optional[np.ndarray]:
        """
        Get a 2D slice of the data cube for the specified plane.

        Args:
            plane: 'xy' (channel slice), 'xz' (y-slice), or 'zy' (x-slice)

        Returns:
            2D numpy array or None if data is not loaded
        """
        if self.data is None:
            return None

        ndim = self.data.ndim
        if ndim < 2:
            return None
        elif ndim == 2:
            return self.data
        elif ndim == 3:
            z = max(0, min(self.current_z, self.data.shape[0] - 1))
            y = max(0, min(self.cursor.ypix, self.data.shape[1] - 1))
            x = max(0, min(self.cursor.xpix, self.data.shape[2] - 1))

            if plane == 'xy':
                return self.data[z, :, :]
            elif plane == 'xz':
                return self.data[:, y, :]
            elif plane == 'zy':
                return self.data[:, :, x]
        elif ndim == 4:
            s = max(0, min(self.current_s, self.data.shape[0] - 1))
            z = max(0, min(self.current_z, self.data.shape[1] - 1))
            y = max(0, min(self.cursor.ypix, self.data.shape[2] - 1))
            x = max(0, min(self.cursor.xpix, self.data.shape[3] - 1))

            if plane == 'xy':
                return self.data[s, z, :, :]
            elif plane == 'xz':
                return self.data[s, :, y, :]
            elif plane == 'zy':
                return self.data[s, :, :, x]

        return None

    @property
    def shape(self) -> Optional[Tuple[int, ...]]:
        """Return the shape of the data cube."""
        if self.data is None:
            return None
        return self.data.shape

    @property
    def n_channels(self) -> int:
        """Return the number of channels (z-axis length)."""
        if self.data is None:
            return 0
        if self.data.ndim == 2:
            return 1
        elif self.data.ndim == 3:
            return self.data.shape[0]
        elif self.data.ndim >= 4:
            return self.data.shape[1]
        return 0

    @property
    def has_4th_axis(self) -> bool:
        """Check if data has a 4th axis (e.g., Stokes)."""
        return self.data is not None and self.data.ndim >= 4


def create_app_state(
    data: np.ndarray,
    header: Any,
    wcs: Any,
    filepath: Optional[str] = None,
    spectral_metadata: Optional[Dict[str, Any]] = None
) -> AppState:
    """
    Factory function to create an AppState from loaded FITS data.

    Args:
        data: FITS data array
        header: FITS header
        wcs: WCS object
        filepath: Source file path
        spectral_metadata: Spectral metadata dict from fits_loader

    Returns:
        Initialized AppState instance
    """
    return AppState(
        data=data,
        header=header,
        wcs=wcs,
        filepath=filepath,
        spectral_metadata=spectral_metadata or {},
        current_z=0,
        current_s=0,
    )
