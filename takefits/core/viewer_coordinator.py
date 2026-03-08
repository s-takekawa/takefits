"""
ViewerCoordinator: Cross-plane coordination for FITSViewer instances.

Manages viewer registration, provides cross-plane access, and owns
the shared CoordinateState. Owned by MainWindow.
"""
from typing import Optional, Dict, Any, TYPE_CHECKING
from weakref import ref as weakref

from PySide6.QtCore import QObject

from takefits.core.coordinate_state import CoordinateState
from takefits.core.viewer_state import ViewerState

if TYPE_CHECKING:
    from takefits.ui.viewer import FITSViewer


class ViewerCoordinator(QObject):
    """
    Coordinates multiple FITSViewer instances across different planes.

    This class:
    - Manages viewer registration for each plane (xy, xz, zy)
    - Provides get_viewer(plane) for cross-plane access
    - Owns the shared CoordinateState
    - Is owned by MainWindow
    """

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)

        # Shared coordinate state
        self.coord_state = CoordinateState(self)

        # Weak references to viewers by plane
        self._viewers: Dict[str, Optional[weakref]] = {
            'xy': None,
            'xz': None,
            'zy': None,
        }

        # ViewerState instances by plane
        self._states: Dict[str, Optional[ViewerState]] = {
            'xy': None,
            'xz': None,
            'zy': None,
        }

        # Integration and channel map colorbar lists (moved from Common)
        self.integ_cax = []
        self.integ_colorbar = []
        self.ch_cax = None
        self.ch_colorbar = None

    def register_viewer(self, plane: str, viewer: 'FITSViewer'):
        """
        Register a viewer for a specific plane.

        Args:
            plane: The plane identifier ('xy', 'xz', or 'zy')
            viewer: The FITSViewer instance to register
        """
        if plane not in self._viewers:
            raise ValueError(f"Invalid plane: {plane}. Must be 'xy', 'xz', or 'zy'.")

        self._viewers[plane] = weakref(viewer) if viewer is not None else None

        # Create and associate a ViewerState if the viewer has one
        state = getattr(viewer, 'state', None)
        if state is not None:
            self._states[plane] = state
            state.set_viewer(viewer)

    def unregister_viewer(self, plane: str):
        """
        Unregister the viewer for a specific plane.

        Args:
            plane: The plane identifier ('xy', 'xz', or 'zy')
        """
        if plane in self._viewers:
            self._viewers[plane] = None
            self._states[plane] = None

    def get_viewer(self, plane: str) -> Optional['FITSViewer']:
        """
        Get the viewer for a specific plane.

        Args:
            plane: The plane identifier ('xy', 'xz', or 'zy')

        Returns:
            The FITSViewer instance or None if not registered or dead
        """
        ref = self._viewers.get(plane)
        if ref is None:
            return None
        return ref()

    def get_state(self, plane: str) -> Optional[ViewerState]:
        """
        Get the ViewerState for a specific plane.

        Args:
            plane: The plane identifier ('xy', 'xz', or 'zy')

        Returns:
            The ViewerState instance or None if not registered
        """
        return self._states.get(plane)

    def get_all_viewers(self) -> Dict[str, Optional['FITSViewer']]:
        """
        Get all registered viewers.

        Returns:
            Dictionary mapping plane names to viewer instances (or None)
        """
        return {
            plane: (ref() if ref is not None else None)
            for plane, ref in self._viewers.items()
        }

    def get_all_states(self) -> Dict[str, Optional[ViewerState]]:
        """
        Get all ViewerState instances.

        Returns:
            Dictionary mapping plane names to ViewerState instances (or None)
        """
        return dict(self._states)

    # ----- Coordinate state shortcuts -----

    @property
    def xpix(self) -> int:
        return self.coord_state.xpix

    @xpix.setter
    def xpix(self, value: int):
        self.coord_state.xpix = value

    @property
    def ypix(self) -> int:
        return self.coord_state.ypix

    @ypix.setter
    def ypix(self, value: int):
        self.coord_state.ypix = value

    @property
    def zpix(self) -> int:
        return self.coord_state.zpix

    @zpix.setter
    def zpix(self, value: int):
        self.coord_state.zpix = value

    @property
    def spix(self) -> int:
        return self.coord_state.spix

    @spix.setter
    def spix(self, value: int):
        self.coord_state.spix = value

    @property
    def world_x(self) -> Optional[float]:
        return self.coord_state.world_x

    @world_x.setter
    def world_x(self, value: Optional[float]):
        self.coord_state.world_x = value

    @property
    def world_y(self) -> Optional[float]:
        return self.coord_state.world_y

    @world_y.setter
    def world_y(self, value: Optional[float]):
        self.coord_state.world_y = value

    @property
    def world_z(self) -> Optional[float]:
        return self.coord_state.world_z

    @world_z.setter
    def world_z(self, value: Optional[float]):
        self.coord_state.world_z = value

    @property
    def world_s(self) -> Optional[float]:
        return self.coord_state.world_s

    @world_s.setter
    def world_s(self, value: Optional[float]):
        self.coord_state.world_s = value

    @property
    def world_x_str(self) -> str:
        return self.coord_state.world_x_str

    @world_x_str.setter
    def world_x_str(self, value: str):
        self.coord_state.world_x_str = value

    @property
    def world_y_str(self) -> str:
        return self.coord_state.world_y_str

    @world_y_str.setter
    def world_y_str(self, value: str):
        self.coord_state.world_y_str = value

    @property
    def world_z_str(self) -> str:
        return self.coord_state.world_z_str

    @world_z_str.setter
    def world_z_str(self, value: str):
        self.coord_state.world_z_str = value

    @property
    def world_s_str(self) -> str:
        return self.coord_state.world_s_str

    @world_s_str.setter
    def world_s_str(self, value: str):
        self.coord_state.world_s_str = value

    @property
    def display_frame(self) -> str:
        return self.coord_state.display_frame

    @display_frame.setter
    def display_frame(self, value: str):
        self.coord_state.display_frame = value

    # ----- Click state shortcuts -----

    @property
    def clicked_xy(self) -> bool:
        return self.coord_state.clicked_xy

    @clicked_xy.setter
    def clicked_xy(self, value: bool):
        self.coord_state.clicked_xy = value

    @property
    def clicked_xz(self) -> bool:
        return self.coord_state.clicked_xz

    @clicked_xz.setter
    def clicked_xz(self, value: bool):
        self.coord_state.clicked_xz = value

    @property
    def clicked_zy(self) -> bool:
        return self.coord_state.clicked_zy

    @clicked_zy.setter
    def clicked_zy(self, value: bool):
        self.coord_state.clicked_zy = value

    # ----- Batch update methods -----

    def update_pix(self, x: int, y: int, z: int, s: int = 0, emit: bool = True):
        """Update all pixel coordinates."""
        self.coord_state.update_pix(x, y, z, s, emit)

    def update_world_xyz(self, x: Optional[float], y: Optional[float], z: Optional[float], s: Optional[float] = None):
        """Update all world coordinates."""
        self.coord_state.update_world_xyz(x, y, z, s)

    def update_world_xyz_str(self, x_str: str, y_str: str, z_str: str, s_str: str = ""):
        """Update all world coordinate strings."""
        self.coord_state.update_world_xyz_str(x_str, y_str, z_str, s_str)

    # ----- Cross-plane operations -----

    def get_canvas(self, plane: str) -> Any:
        """Get canvas for a specific plane."""
        state = self.get_state(plane)
        return state.canvas if state else None

    def get_overlay_ax(self, plane: str) -> Any:
        """Get overlay_ax for a specific plane."""
        state = self.get_state(plane)
        return state.overlay_ax if state else None

    def get_background(self, plane: str) -> Any:
        """Get cached background for a specific plane."""
        state = self.get_state(plane)
        return state._background if state else None

    def get_hline(self, plane: str) -> Any:
        """Get horizontal crosshair line for a specific plane."""
        state = self.get_state(plane)
        return state.hline if state else None

    def get_vline(self, plane: str) -> Any:
        """Get vertical crosshair line for a specific plane."""
        state = self.get_state(plane)
        return state.vline if state else None

    def get_im(self, plane: str) -> Any:
        """Get image object for a specific plane."""
        state = self.get_state(plane)
        return state.im if state else None

    def get_ax(self, plane: str) -> Any:
        """Get main axes for a specific plane."""
        state = self.get_state(plane)
        return state.ax if state else None

    def get_slider(self, plane: str) -> Any:
        """Get slider widget for a specific plane."""
        state = self.get_state(plane)
        return state.slider if state else None

    def get_chlabel(self, plane: str) -> Any:
        """Get channel label for a specific plane."""
        state = self.get_state(plane)
        return state.chlabel if state else None

    def get_plabel(self, plane: str) -> Any:
        """Get position label for a specific plane."""
        state = self.get_state(plane)
        return state.plabel if state else None

    def copy_overlay_background(self, plane: str) -> Any:
        """
        Copy the overlay background for a specific plane.

        This is a convenience method that delegates to the ViewerState.
        """
        state = self.get_state(plane)
        if state is None:
            return None
        return state.copy_overlay_background()

    # ----- Integration/Channel colorbar management -----

    def update_integ_cax(self, cax):
        """Add an integration colorbar axes."""
        self.integ_cax.append(cax)

    def remove_integ_cax(self, cax):
        """Remove an integration colorbar axes."""
        if cax in self.integ_cax:
            self.integ_cax.remove(cax)

    def update_integ_colorbar(self, colorbar):
        """Add an integration colorbar."""
        self.integ_colorbar.append(colorbar)

    def remove_integ_colorbar(self, colorbar):
        """Remove an integration colorbar."""
        if colorbar in self.integ_colorbar:
            self.integ_colorbar.remove(colorbar)

    def update_ch_colorbar(self, colorbar):
        """Set the channel map colorbar."""
        self.ch_colorbar = colorbar

    def update_ch_cax(self, cax):
        """Set the channel map colorbar axes."""
        self.ch_cax = cax
