"""
Mixin class providing coordinator and cross-plane state access methods for FITSViewer.

This mixin contains pure delegation methods that access the ViewerCoordinator
and ViewerState without directly creating or manipulating PyQt widgets.
"""
from typing import Optional


class ViewerCoordinatorMixin:
    """
    Mixin providing coordinator access and shared coordinate state methods.

    Assumes the inheriting class has:
    - self.coordinator (optional, may be None for subwindows)
    - self.plane (str: 'xy', 'xz', or 'zy')
    - self.state (ViewerState instance)
    - self.canvas, self.overlay_ax, self.ax, self.im, self.hline, self.vline
    - self.ch_label, self.label, self.slider, self.chval_box (for current plane)
    - Class attribute main_window (set by MainWindow)
    """

    def get_coordinator(self):
        """Get the ViewerCoordinator, either from self or from main_window."""
        if hasattr(self, 'coordinator') and self.coordinator is not None:
            return self.coordinator
        # Access main_window from the class that inherits this mixin
        main_window = getattr(self.__class__, 'main_window', None)
        if main_window is not None and hasattr(main_window, 'coordinator'):
            return main_window.coordinator
        return None

    def get_coord_state(self):
        """Get the shared CoordinateState from the coordinator."""
        coord = self.get_coordinator()
        if coord is not None:
            return coord.coord_state
        return None

    def get_viewer_state(self, plane: str = None):
        """Get ViewerState for a specific plane (defaults to self.plane)."""
        if plane is None or plane == getattr(self, 'plane', None):
            return getattr(self, 'state', None)
        coord = self.get_coordinator()
        if coord is not None:
            return coord.get_state(plane)
        return None

    # ----- Coordinate property shortcuts (read from coordinator) -----

    def _get_shared_xpix(self) -> int:
        coord = self.get_coordinator()
        if coord is not None:
            return coord.xpix
        return 0

    def _get_shared_ypix(self) -> int:
        coord = self.get_coordinator()
        if coord is not None:
            return coord.ypix
        return 0

    def _get_shared_zpix(self) -> int:
        coord = self.get_coordinator()
        if coord is not None:
            return coord.zpix
        return 0

    def _set_shared_xpix(self, value: int):
        coord = self.get_coordinator()
        if coord is not None:
            coord.xpix = int(value)

    def _set_shared_ypix(self, value: int):
        coord = self.get_coordinator()
        if coord is not None:
            coord.ypix = int(value)

    def _set_shared_zpix(self, value: int):
        coord = self.get_coordinator()
        if coord is not None:
            coord.zpix = int(value)

    def _set_shared_spix(self, value: int):
        coord = self.get_coordinator()
        if coord is not None:
            coord.spix = int(value)

    def _get_shared_world_x(self):
        coord = self.get_coordinator()
        if coord is not None:
            return coord.world_x
        return None

    def _get_shared_world_y(self):
        coord = self.get_coordinator()
        if coord is not None:
            return coord.world_y
        return None

    def _get_shared_world_z(self):
        coord = self.get_coordinator()
        if coord is not None:
            return coord.world_z
        return None

    def _get_shared_world_s(self):
        coord = self.get_coordinator()
        if coord is not None:
            return coord.world_s
        return None

    def _get_shared_world_x_str(self) -> str:
        coord = self.get_coordinator()
        if coord is not None:
            return coord.world_x_str
        return ""

    def _get_shared_world_y_str(self) -> str:
        coord = self.get_coordinator()
        if coord is not None:
            return coord.world_y_str
        return ""

    def _get_shared_world_z_str(self) -> str:
        coord = self.get_coordinator()
        if coord is not None:
            return coord.world_z_str
        return ""

    def _get_shared_world_s_str(self) -> str:
        coord = self.get_coordinator()
        if coord is not None:
            return coord.world_s_str
        return ""

    def _get_shared_display_frame(self) -> str:
        coord = self.get_coordinator()
        if coord is not None:
            return str(getattr(coord, "display_frame", "native") or "native")
        return "native"

    def _set_shared_display_frame(self, frame: str):
        coord = self.get_coordinator()
        if coord is not None:
            coord.display_frame = str(frame or "native")

    def _get_clicked(self, plane: str) -> bool:
        coord = self.get_coordinator()
        if coord is not None:
            return coord.coord_state.get_clicked(plane)
        return False

    def _set_clicked(self, plane: str, value: bool):
        coord = self.get_coordinator()
        if coord is not None:
            coord.coord_state.set_clicked(plane, value)

    def _update_shared_pix(self, x: int, y: int, z: int, s: int = 0):
        coord = self.get_coordinator()
        if coord is not None:
            coord.update_pix(x, y, z, s)

    def _update_shared_world_xyz(self, x, y, z, s=None):
        coord = self.get_coordinator()
        if coord is not None:
            coord.update_world_xyz(x, y, z, s)

    def _update_shared_world_xyz_str(self, x_str: str, y_str: str, z_str: str, s_str: str = ""):
        coord = self.get_coordinator()
        if coord is not None:
            coord.update_world_xyz_str(x_str, y_str, z_str, s_str)

    # ----- Cross-plane state access helpers -----

    def _get_plane_canvas(self, plane: str):
        """Get canvas for a specific plane (current or other)."""
        if plane == self.plane:
            return self.canvas
        state = self.get_viewer_state(plane)
        if state is not None and state.canvas is not None:
            return state.canvas
        return None

    def _get_plane_overlay_ax(self, plane: str):
        """Get overlay_ax for a specific plane."""
        if plane == self.plane:
            return self.overlay_ax
        state = self.get_viewer_state(plane)
        if state is not None and state.overlay_ax is not None:
            return state.overlay_ax
        return None

    def _get_plane_ax(self, plane: str):
        """Get main ax for a specific plane."""
        if plane == self.plane:
            return self.ax
        state = self.get_viewer_state(plane)
        if state is not None and state.ax is not None:
            return state.ax
        return None

    def _get_plane_im(self, plane: str):
        """Get image object for a specific plane."""
        if plane == self.plane:
            return self.im
        state = self.get_viewer_state(plane)
        if state is not None and state.im is not None:
            return state.im
        return None

    def _get_plane_background(self, plane: str):
        """Get cached background for a specific plane."""
        if plane == self.plane:
            state_bg = getattr(self.state, '_background', None) if hasattr(self, 'state') else None
            if state_bg is not None:
                return state_bg
            return getattr(self, '_background', None)
        state = self.get_viewer_state(plane)
        if state is not None:
            return state._background
        return None

    def _invalidate_plane_background(self, plane: str):
        """Invalidate cached background for a specific plane.

        This must be called after set_data() on an image to ensure
        the stale background is not used for blitting.
        """
        if plane == self.plane:
            self._background = None
            try:
                self._background_initialized = False
            except Exception:
                pass
            if hasattr(self, 'state') and self.state is not None:
                self.state._background = None
                self.state.image_background = None
        else:
            state = self.get_viewer_state(plane)
            if state is not None:
                state._background = None
                state.image_background = None
                viewer = getattr(state, 'viewer', None)
                if viewer is not None:
                    try:
                        viewer._background = None
                        viewer._background_initialized = False
                    except Exception:
                        pass

    def _get_plane_hline(self, plane: str):
        """Get horizontal crosshair line for a specific plane."""
        if plane == self.plane:
            return self.hline
        state = self.get_viewer_state(plane)
        if state is not None:
            return state.hline
        return None

    def _get_plane_vline(self, plane: str):
        """Get vertical crosshair line for a specific plane."""
        if plane == self.plane:
            return self.vline
        state = self.get_viewer_state(plane)
        if state is not None:
            return state.vline
        return None

    def _get_plane_cpoint(self, plane: str):
        """Get center marker artist for a specific plane."""
        if plane == self.plane:
            return getattr(self, "cpoint", None)
        state = self.get_viewer_state(plane)
        if state is not None:
            return getattr(state, "cpoint", None)
        return None

    def _get_plane_chlabel(self, plane: str):
        """Get channel label for a specific plane."""
        if plane == self.plane:
            return self.ch_label
        state = self.get_viewer_state(plane)
        if state is not None:
            return state.chlabel
        return None

    def _get_plane_plabel(self, plane: str):
        """Get position label for a specific plane."""
        if plane == self.plane:
            return self.label
        state = self.get_viewer_state(plane)
        if state is not None:
            return state.plabel
        return None

    def _get_plane_slider(self, plane: str):
        """Get slider for a specific plane."""
        if plane == self.plane:
            return getattr(self, 'slider', None)
        state = self.get_viewer_state(plane)
        if state is not None:
            return state.slider
        return None

    def _get_plane_chval_box(self, plane: str):
        """Get channel value box for a specific plane."""
        if plane == self.plane:
            return getattr(self, 'chval_box', None)
        state = self.get_viewer_state(plane)
        if state is not None:
            return state.chval_box
        return None

    def _update_plane_cursor(self, plane: str, *, x: Optional[float] = None, y: Optional[float] = None):
        state = self.get_viewer_state(plane)
        if state is None:
            return
        if x is not None:
            state.cursor_x = float(x)
        if y is not None:
            state.cursor_y = float(y)
