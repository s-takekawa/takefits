"""
CoordinateState: Shared coordinate state for all viewer planes.

Holds pixel and world coordinates that are shared across all planes,
and emits signals when positions are updated.
"""
from typing import Optional
from PySide6.QtCore import QObject, Signal as pyqtSignal


class CoordinateState(QObject):
    """
    Manages shared coordinate state across all viewer planes.

    This replaces the coordinate-related class attributes in Common:
    - xpix, ypix, zpix (pixel coordinates)
    - world_x, world_y, world_z (world coordinates as floats)
    - world_x_str, world_y_str, world_z_str (formatted strings)
    """

    # Emitted when pixel coordinates change: (xpix, ypix, zpix)
    position_updated = pyqtSignal(int, int, int)

    # Emitted when world coordinates change: (world_x, world_y, world_z)
    world_position_updated = pyqtSignal(float, float, float)
    display_frame_changed = pyqtSignal(str)

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)

        # Pixel coordinates
        self._xpix: int = 0
        self._ypix: int = 0
        self._zpix: int = 0

        # World coordinates (floats)
        self._world_x: Optional[float] = None
        self._world_y: Optional[float] = None
        self._world_z: Optional[float] = None

        # World coordinates (formatted strings)
        self._world_x_str: str = ""
        self._world_y_str: str = ""
        self._world_z_str: str = ""

        # Click state flags for each plane
        self._clicked_xy: bool = False
        self._clicked_xz: bool = False
        self._clicked_zy: bool = False
        self._display_frame: str = "native"

    # ----- Pixel coordinate properties -----
    @property
    def xpix(self) -> int:
        return self._xpix

    @xpix.setter
    def xpix(self, value: int):
        self._xpix = int(value)

    @property
    def ypix(self) -> int:
        return self._ypix

    @ypix.setter
    def ypix(self, value: int):
        self._ypix = int(value)

    @property
    def zpix(self) -> int:
        return self._zpix

    @zpix.setter
    def zpix(self, value: int):
        self._zpix = int(value)

    @property
    def spix(self) -> int:
        return getattr(self, "_spix", 0)

    @spix.setter
    def spix(self, value: int):
        self._spix = int(value)

    # ----- World coordinate properties -----

    @property
    def world_x(self) -> Optional[float]:
        return self._world_x

    @world_x.setter
    def world_x(self, value: Optional[float]):
        self._world_x = value

    @property
    def world_y(self) -> Optional[float]:
        return self._world_y

    @world_y.setter
    def world_y(self, value: Optional[float]):
        self._world_y = value

    @property
    def world_z(self) -> Optional[float]:
        return self._world_z

    @world_z.setter
    def world_z(self, value: Optional[float]):
        self._world_z = value

    @property
    def world_s(self) -> Optional[float]:
        return getattr(self, "_world_s", None)

    @world_s.setter
    def world_s(self, value: Optional[float]):
        self._world_s = value

    # ----- World coordinate string properties -----

    @property
    def world_x_str(self) -> str:
        return self._world_x_str

    @world_x_str.setter
    def world_x_str(self, value: str):
        self._world_x_str = str(value)

    @property
    def world_y_str(self) -> str:
        return self._world_y_str

    @world_y_str.setter
    def world_y_str(self, value: str):
        self._world_y_str = str(value)

    @property
    def world_z_str(self) -> str:
        return self._world_z_str

    @world_z_str.setter
    def world_z_str(self, value: str):
        self._world_z_str = str(value)

    @property
    def world_s_str(self) -> str:
        return getattr(self, "_world_s_str", "")

    @world_s_str.setter
    def world_s_str(self, value: str):
        self._world_s_str = str(value)

    @property
    def display_frame(self) -> str:
        return self._display_frame

    @display_frame.setter
    def display_frame(self, value: str):
        frame = str(value or "native").strip().lower() or "native"
        if frame == self._display_frame:
            return
        self._display_frame = frame
        self.display_frame_changed.emit(frame)

    # ----- Click state properties -----

    @property
    def clicked_xy(self) -> bool:
        return self._clicked_xy

    @clicked_xy.setter
    def clicked_xy(self, value: bool):
        self._clicked_xy = bool(value)

    @property
    def clicked_xz(self) -> bool:
        return self._clicked_xz

    @clicked_xz.setter
    def clicked_xz(self, value: bool):
        self._clicked_xz = bool(value)

    @property
    def clicked_zy(self) -> bool:
        return self._clicked_zy

    @clicked_zy.setter
    def clicked_zy(self, value: bool):
        self._clicked_zy = bool(value)

    # ----- Batch update methods (matching Common API) -----

    def update_pix(self, x: int, y: int, z: int, s: int = 0, emit: bool = True):
        """
        Update all pixel coordinates at once.

        Args:
            x: X pixel coordinate
            y: Y pixel coordinate
            z: Z pixel coordinate
            s: S pixel coordinate (optional)
            emit: Whether to emit the position_updated signal
        """
        self._xpix = int(x)
        self._ypix = int(y)
        self._zpix = int(z)
        self._spix = int(s)
        if emit:
            # We unfortunately can't change the signal signature easily without breaking things,
            # so we'll just emit the 3D position for now. Consumers needing 's' should read the property.
            self.position_updated.emit(self._xpix, self._ypix, self._zpix)

    def update_world_xyz(self, x: Optional[float], y: Optional[float], z: Optional[float], s: Optional[float] = None):
        """Update all world coordinates at once."""
        self._world_x = x
        self._world_y = y
        self._world_z = z
        self._world_s = s

    def update_world_xyz_str(self, x_str: str, y_str: str, z_str: str, s_str: str = ""):
        """Update all world coordinate strings at once."""
        self._world_x_str = str(x_str)
        self._world_y_str = str(y_str)
        self._world_z_str = str(z_str)
        self._world_s_str = str(s_str)

    def set_clicked(self, plane: str, value: bool = True):
        """
        Set the clicked state for a specific plane, clearing others.

        Args:
            plane: The plane that was clicked ('xy', 'xz', or 'zy')
            value: The clicked state to set
        """
        if value:
            # When setting a plane as clicked, clear the others
            self._clicked_xy = (plane == 'xy')
            self._clicked_xz = (plane == 'xz')
            self._clicked_zy = (plane == 'zy')
        else:
            # When clearing, only clear the specified plane
            if plane == 'xy':
                self._clicked_xy = False
            elif plane == 'xz':
                self._clicked_xz = False
            elif plane == 'zy':
                self._clicked_zy = False

    def clear_all_clicked(self):
        """Clear all clicked states."""
        self._clicked_xy = False
        self._clicked_xz = False
        self._clicked_zy = False

    def get_clicked(self, plane: str) -> bool:
        """Get the clicked state for a specific plane."""
        if plane == 'xy':
            return self._clicked_xy
        elif plane == 'xz':
            return self._clicked_xz
        elif plane == 'zy':
            return self._clicked_zy
        return False
