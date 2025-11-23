
from typing import Dict, List, Optional, Tuple, Iterable

import math
import os
import re

import astropy.units as u
import numpy as np
from astropy.coordinates import Angle
import matplotlib as mpl
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import (QButtonGroup, QComboBox, QDialog, QGridLayout,
                             QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                             QMessageBox, QPushButton, QRadioButton, QWidget,
                             QVBoxLayout, QMainWindow)

from core.common import Common
from core.coordinate import CoordinateConverter, Format_pix_to_wcs
from core.marker import Marker, MarkerState, marker_from_state
from core.marker_manager import MarkerManager
from logic.add_hpbw import AddHPBW
from tools.color_scale import ColorSettingsPanel, ColorMode
from ui.navigation_toolbar import MyNavigationToolbar
from core.contour_manager import ContourManager, ContourItem


class _ChannelMarkerFormatWrapper:
    """Adapter that delegates plane-aware conversions for channel map marker usage."""

    def __init__(self, converters: Dict[str, Format_pix_to_wcs], plane_bases: Dict[str, str], default_base: str):
        self._converters = converters
        self._plane_bases = plane_bases
        self._default_base = default_base
        sample = next(iter(converters.values()), None)
        self.slices = getattr(sample, "slices", None) if sample is not None else None

    def _resolve_converter(self, plane: Optional[str]) -> Optional[Format_pix_to_wcs]:
        if plane in self._converters:
            return self._converters[plane]
        if self._converters:
            return next(iter(self._converters.values()))
        return None

    def _base_plane(self, plane: Optional[str]) -> str:
        if plane in self._plane_bases:
            return self._plane_bases[plane]
        return self._default_base

    def convert(self, plane: str, xdata: float, ydata: float):
        converter = self._resolve_converter(plane)
        if converter is None:
            raise RuntimeError("No coordinate converter available for channel map markers.")
        base_plane = self._base_plane(plane)
        return converter.convert(base_plane, xdata, ydata)

    def pix_to_wcs(self, wcs, xpix: float, ypix: float, plane: str):
        converter = self._resolve_converter(plane)
        if converter is None:
            raise RuntimeError("No coordinate converter available for channel map markers.")
        return converter.pix_to_wcs(wcs, xpix, ypix, self._base_plane(plane))

    def convert_chpix_to_world(self, plane: str, x: float, y: float, z: float):
        converter = self._resolve_converter(plane)
        if converter is None:
            raise RuntimeError("No coordinate converter available for channel map markers.")
        return converter.convert_chpix_to_world(self._base_plane(plane), x, y, z)

    def convert_chval_to_world_str(self, plane: str, value: float):
        converter = self._resolve_converter(plane)
        if converter is None:
            raise RuntimeError("No coordinate converter available for channel map markers.")
        return converter.convert_chval_to_world_str(self._base_plane(plane), value)

class ChannelMapWindow(QMainWindow):
    """
    A window to display the channel map as a separate window.
    The channel map is displayed as a grid of subplots using the provided images and labels.
    Paging is implemented if the total number of channel maps exceeds the number of grid tiles.
    The UI (including range–setting controls, paging buttons, and a Colorscale button)
    mimics the IntegResultWindow style.
    """
    def __init__(self, fits_viewer, subwindows, ch_imdata, range_label, tiles_x, tiles_y, wcs, dir_num, chlabel_num, plane_num=0, title="Channel Map", parent=None):
        """
        Initialize the ChannelMapWindow.

        Parameters:
            ch_imdata (list): List of 2D numpy arrays (channel map images).
            range_label (list): List of labels for each channel (e.g. [from, center, to]).
            tiles_x (int): Number of rows in the grid.
            tiles_y (int): Number of columns in the grid.
            wcs (astropy.wcs.WCS): WCS object for subplot projection.
            plane_num (int): 0 for "xy", 1 for "xz", 2 for "zy".
            title (str): Window title.
            parent (QWidget): Parent widget.
        """
        super().__init__(parent)
        self.color_settings_panel = None
        self._contour_layer_id = None
        self._contour_title_connected = False
        self.fits_viewer = fits_viewer
        self.subwindows = subwindows
        self.ch_imdata = ch_imdata
        self.flattened_chdata = np.array(self.ch_imdata).flatten()
        self.range_label = range_label
        self.tiles_x = tiles_x
        self.tiles_y = tiles_y
        self.dir_num = dir_num
        self.chlabel_num = chlabel_num
        self.wcs = wcs
        self.plane_num = plane_num
        self.setWindowTitle(title)

        self.marker_manager = MarkerManager(self)
        self.marker_mode_enabled = False
        self.marker_panel = None
        self._marker_axes: Dict[str, object] = {}
        self._axes_marker_plane: Dict[object, str] = {}
        self._marker_formats: Dict[str, Format_pix_to_wcs] = {}
        self._marker_planes: List[str] = []
        self._marker_plane_base: Dict[str, str] = {}
        self.marker_link_all = False
        
        self.original_xlim = self.fits_viewer.ax.get_xlim()
        self.original_ylim = self.fits_viewer.ax.get_ylim()
        self.original_zlim = self.subwindows[0].ax.get_ylim()
        self.converter = CoordinateConverter(self.wcs, self.fits_viewer.config_manager.config)

        if self.wcs.naxis == 3:
            self.znpix = self.fits_viewer.data.shape[0]-1
            self.ynpix = self.fits_viewer.data.shape[1]-1
            self.xnpix = self.fits_viewer.data.shape[2]-1
        elif self.wcs.naxis == 4:
            self.znpix = self.fits_viewer.data[0].shape[0]-1
            self.ynpix = self.fits_viewer.data[0].shape[1]-1
            self.xnpix = self.fits_viewer.data[0].shape[2]-1
        
        self.xlabel = Common.ax_coord_xy[0].get_axislabel()
        self.ylabel = Common.ax_coord_xy[1].get_axislabel()
        self.zlabel = Common.ax_coord_xz[1].get_axislabel()
        # Paging variables.
        self.current_page = 0
        self.num_per_page = self.tiles_x * self.tiles_y

        # Default configuration (customize as needed).
        """
        self.config = {
            'fig_background_color': 'white',
            'figure_width': 800,
            'figure_height': 800,
            'ax_background_color': 'white',
            'axislabel_fontsize': 12,
            'axislabel_fontfamily': 'Arial',
            'axislabel_color': 'black',
            'xticklabel_position': 'b',
            'yticklabel_position': 'l',
            'tick_xlabelrotation': 0,
            'tick_pad_x': 5,
            'tick_ylabelrotation': 0,
            'tick_pad_y': 5,
            'default_ticks_position': 'btlr',
            'x_mtick_freq': 5,
            'y_mtick_freq': 5,
            'z_mtick_freq': 5,
            'tick_direction': 'out',
            'tick_length': 4,
            'mtick_length': 2,
            'tick_color': 'black',
            'tick_width': 1,
            'tick_labelsize': 10,
            'tick_labelcolor': 'black',
            'decimal': True,  # if True, show coordinates in degrees (not DMS)
            'coord_wrap': 180
        }
        """
        self.config = self.fits_viewer.config_manager.config
        self.figure_width = 900
        self.figure_height = 900
        #self.config['figure_width'] = 900
        #self.config['figure_height'] = 900

        self.decimal = self.config['decimal'] 
        self.number_decimals = self.config['number_decimals']
        self.coord_wrap = self.config['coord_wrap']
        
        
        self.color_pattern = (
            ColorSettingsPanel.settings[ColorMode.MAIN]['color_pattern'] or 
            self.fits_viewer.displaymap.config.get('colorscale')
        )
        if ColorSettingsPanel.settings[ColorMode.CHANNEL]['color_pattern']:
            self.color_pattern = ColorSettingsPanel.settings[ColorMode.CHANNEL]['color_pattern']

        self.initialize_ranges()
        self.initUI()
        self._register_contour_layer()

    def get_projection_slices(self):
        """
        Return the appropriate projection slices based on the WCS dimensionality and plane.
        For a 3D WCS:
            plane_num==0 ("xy"):  horizontal = x, vertical = y.
            plane_num==1 ("xz"):  horizontal = x, vertical = z.
            plane_num==2 ("zy"):  horizontal = z, vertical = y.
        For 4D WCS similar logic applies.
        """
        if self.wcs.naxis == 3:
            if self.plane_num == 0:
                return ('x', 'y', 0)
            elif self.plane_num == 1:
                return ('x', 0, 'y')
            elif self.plane_num == 2:
                return (0, 'y', 'x')
        elif self.wcs.naxis == 4:
            if self.plane_num == 0:
                return ('x', 'y', 0, 0)
            elif self.plane_num == 1:
                return ('x', 0, 'y', 0)
            elif self.plane_num == 2:
                return (0, 'y', 'x', 0)

    def formatter(self, x, y):
        xstr, ystr = self.format_pix.convert(self.plane, x, y)
        xstr = ("{:>.%ds}" % (self.number_decimals+6)).format(xstr)
        ystr = ("{:>.%ds}" % (self.number_decimals+6)).format(ystr)
        if self.plane == 'xy': return f'x={xstr}, y={ystr}'
        elif  self.plane == 'xz': return f'x={xstr}, z={ystr}'
        elif  self.plane == 'zy': return f'z={xstr}, y={ystr}'

    def initUI(self):
        self.fig = Figure(figsize=(self.tiles_y * 3, self.tiles_x * 3))
        self.fig.set_constrained_layout_pads(
            wspace=0.1,
            hspace=0.1,
            w_pad=0.1,
            h_pad=0.1,
            #rect=(0.0, 0.0, 0.9, 0.9)
        )
        self.canvas = FigureCanvas(self.fig)



        layout = QGridLayout()
        layout.setHorizontalSpacing(5)
        layout.setVerticalSpacing(5)
        layout.setContentsMargins(12, 0, 12, 12)
    
        if self.plane_num in [0, 1]:
            self.xr_ch_label = QLabel("X:")
            self.xr_ch_label.setFixedWidth(20)
            self.x_min_ch_input = QLineEdit()
            self.x_min_ch_input.setPlaceholderText("X min value")
            self.x_min_ch_input.setFixedWidth(80)
            self.x_max_ch_input = QLineEdit()
            self.x_max_ch_input.setPlaceholderText("X max value")
            self.x_max_ch_input.setFixedWidth(80)
            self.x_ch_button = QPushButton("Set X")

            self.x_min_ch_input.returnPressed.connect(self.set_x_range)
            self.x_max_ch_input.returnPressed.connect(self.set_x_range)
            self.x_ch_button.clicked.connect(self.set_x_range)
            #self.x_min_ch_input.setText(str(self.xmin_val))
            #self.x_max_ch_input.setText(str(self.xmax_val))

            layout.addWidget(self.xr_ch_label,    0, 0, 1, 1)
            layout.addWidget(self.x_min_ch_input, 0, 1, 1, 1)
            layout.addWidget(self.x_max_ch_input, 0, 2, 1, 1)
            layout.addWidget(self.x_ch_button,    0, 3, 1, 1)
            

    
        if self.plane_num in [0, 2]:
            self.yr_ch_label = QLabel("Y:")
            self.yr_ch_label.setFixedWidth(20)
            self.y_min_ch_input = QLineEdit()
            self.y_min_ch_input.setPlaceholderText("Y min value")
            self.y_min_ch_input.setFixedWidth(80)
            self.y_max_ch_input = QLineEdit()
            self.y_max_ch_input.setPlaceholderText("Y max value")
            self.y_max_ch_input.setFixedWidth(80)
            self.y_ch_button = QPushButton("Set Y")

            self.y_min_ch_input.returnPressed.connect(self.set_y_range)
            self.y_max_ch_input.returnPressed.connect(self.set_y_range)
            self.y_ch_button.clicked.connect(self.set_y_range)
            #self.y_min_ch_input.setText(str(self.ymin_val))
            #self.y_max_ch_input.setText(str(self.ymax_val))

            layout.addWidget(self.yr_ch_label,    1, 0, 1, 1)
            layout.addWidget(self.y_min_ch_input, 1, 1, 1, 1)
            layout.addWidget(self.y_max_ch_input, 1, 2, 1, 1)
            layout.addWidget(self.y_ch_button,    1, 3, 1, 1)
            
    
        if self.plane_num in [1, 2]:
            if self.plane_num == 1: z_pos = 1
            elif self.plane_num == 2: z_pos = 0
            self.zr_ch_label = QLabel("Z:")
            self.zr_ch_label.setFixedWidth(20)
            self.z_min_ch_input = QLineEdit()
            self.z_min_ch_input.setPlaceholderText("Z min value")
            self.z_min_ch_input.setFixedWidth(80)
            self.z_max_ch_input = QLineEdit()
            self.z_max_ch_input.setPlaceholderText("Z max value")
            self.z_max_ch_input.setFixedWidth(80)
            self.z_ch_button = QPushButton("Set Z")

            self.z_min_ch_input.returnPressed.connect(self.set_z_range)
            self.z_max_ch_input.returnPressed.connect(self.set_z_range)
            self.z_ch_button.clicked.connect(self.set_z_range)
            #self.z_min_ch_input.setText(str(self.zmin_val))
            #self.z_max_ch_input.setText(str(self.zmax_val))

            layout.addWidget(self.zr_ch_label,    z_pos, 0, 1, 1)
            layout.addWidget(self.z_min_ch_input, z_pos, 1, 1, 1)
            layout.addWidget(self.z_max_ch_input, z_pos, 2, 1, 1)
            layout.addWidget(self.z_ch_button,    z_pos, 3, 1, 1)
            
        self.full_ch_button = QPushButton("Full")
        self.sync_ch_button = QPushButton("Sync")
        layout.addWidget(self.full_ch_button, 0, 4, 1, 1)
        layout.addWidget(self.sync_ch_button, 1, 4, 1, 1)

        self.full_ch_button.clicked.connect(self.set_full_range)
        self.sync_ch_button.clicked.connect(self.sync_range)

        self.prev_button = QPushButton("Prev")
        self.next_button = QPushButton("Next")
        self.colorscale_button = QPushButton("Colorscale")
        self.marker_button = QPushButton("Markers")
        self.prev_button.clicked.connect(self.prev_page)
        self.next_button.clicked.connect(self.next_page)
        self.colorscale_button.clicked.connect(self.open_color_settings)
        self.marker_button.clicked.connect(self.open_marker_panel)
    
        layout.addWidget(self.prev_button, 1, 12, 1, 1)
        layout.addWidget(self.next_button, 1, 13, 1, 1)
        layout.addWidget(self.marker_button, 1, 14, 1, 1)
        layout.addWidget(self.colorscale_button, 1, 15, 1, 1)
    

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
        
        if self.plane_num == 0:
            self.plane = 'xy'
        elif self.plane_num == 1:
            self.plane = 'xz'
        elif self.plane_num == 2:
            self.plane = 'zy'
        self.marker_plane_prefix = f"channel_{self.plane}"
        self._marker_axes.clear()
        self._axes_marker_plane.clear()
        self._marker_formats.clear()
        self._marker_planes.clear()
        self._marker_plane_base.clear()

        self.projection_slices = self.get_projection_slices()
        self.axes = []
        gs = self.fig.add_gridspec(self.tiles_x, self.tiles_y)
    
        for i in range(self.tiles_x):
            for j in range(self.tiles_y):
                ax = self.fig.add_subplot(gs[i, j], projection=self.wcs, slices=self.projection_slices)
                plane_id = f"{self.marker_plane_prefix}_{len(self._marker_planes)}"
                converter = Format_pix_to_wcs(self.wcs, self.projection_slices, ax, self.plane, self.decimal, self.number_decimals, self.coord_wrap)
                self._marker_formats[plane_id] = converter
                self._marker_axes[plane_id] = ax
                self._axes_marker_plane[ax] = plane_id
                self._marker_plane_base[plane_id] = self.plane
                self._marker_planes.append(plane_id)
                ax.format_coord = self.formatter
                self.axes.append(ax)
        if self.dir_num == 1:
            self.axes = np.array(self.axes).reshape(self.tiles_x, self.tiles_y).T.flatten()
        self.ax = self.axes[0] if self.axes else None

        if self._marker_formats:
            self.format_pix = _ChannelMarkerFormatWrapper(self._marker_formats, self._marker_plane_base, self.plane)
        else:
            self.format_pix = None
        
        self.sync_range()
        self.update_display_initial()
        
        self.toolbar = MyNavigationToolbar(self.canvas, self, self.plane, self.axes[0], color_mode = ColorMode.CHANNEL, default_image_name = self.fits_viewer.filename)
        self.canvas.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.canvas.setFocus()
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.canvas.mpl_connect('key_release_event', self.on_key_release)
        
        layout.addWidget(self.canvas,  2,   0,  1,  16)
        layout.addWidget(self.toolbar, 3, 0,  1,  16)

        if os.path.exists(os.path.join("config", "subplot_params.yaml")):
            dialog = self.toolbar.configure_subplots()
            dialog._import_values()
            self.toolbar._subplot_dialog.hide()

    def _base_plane_from_name(self, plane: Optional[str]) -> str:
        value = (plane or "").lower()
        if "xz" in value:
            return "xz"
        if "zy" in value:
            return "zy"
        if "xy" in value:
            return "xy"
        return self.plane

    # ------------------------------------------------------------------
    # Marker manager integration
    def default_marker_plane(self) -> Optional[str]:
        return self._marker_planes[0] if self._marker_planes else None

    def has_marker_plane(self, plane: str) -> bool:
        return plane in self._marker_axes

    def marker_axes_for_plane(self, plane: str):
        return self._marker_axes.get(plane)

    def marker_plane_for_axes(self, axes) -> Optional[str]:
        return self._axes_marker_plane.get(axes)

    def marker_canvas_for_plane(self, plane: str):
        return self.canvas

    def marker_plane_base(self, plane: str) -> str:
        return self._marker_plane_base.get(plane, self.plane)

    def marker_axis_indices(self, plane: str) -> Optional[Tuple[int, int]]:
        base = self.marker_plane_base(plane)
        if base == 'xy':
            return (0, 1)
        if base == 'xz':
            return (0, 2)
        if base == 'zy':
            return (2, 1)
        return (0, 1)

    def redraw_overlay_for_plane(self, plane: str) -> None:
        self.canvas.draw_idle()

    def supports_marker_link_all(self) -> bool:
        return len(self._marker_planes) > 1

    def marker_link_all_enabled(self) -> bool:
        return bool(self.marker_link_all)

    def set_marker_link_all(self, enabled: bool) -> None:
        self.marker_link_all = bool(enabled)
        panel = getattr(self, "marker_panel", None)
        checkbox = getattr(panel, "link_all_checkbox", None) if panel is not None else None
        if checkbox is not None:
            try:
                checkbox.blockSignals(True)
                checkbox.setChecked(self.marker_link_all)
            finally:
                checkbox.blockSignals(False)

    def can_update_marker_positions(self, markers: Iterable[Marker]) -> bool:
        markers = list(markers)
        if not markers:
            return False
        return all(getattr(marker, "plane", None) in self._marker_planes for marker in markers)

    def mirror_marker_creation(self, marker, plane: str, *, source: str = "primary") -> List[Marker]:
        mirrored_markers = []
        if not self.marker_link_all or source != "primary":
            return mirrored_markers
        manager = getattr(self, "marker_manager", None)
        if manager is None:
            return mirrored_markers
        state_dict_base = marker.to_state().to_dict()
        state_dict_base.pop("id", None)
        state_dict_base.pop("world", None)
        for target_plane in self._marker_planes:
            if target_plane == plane:
                continue
            existing_markers = manager.markers_for_plane(target_plane)
            duplicate = any(
                cand.kind == marker.kind
                and len(cand.pixel) >= 2
                and len(marker.pixel) >= 2
                and math.isclose(cand.pixel[0], marker.pixel[0], rel_tol=0.0, abs_tol=1e-6)
                and math.isclose(cand.pixel[1], marker.pixel[1], rel_tol=0.0, abs_tol=1e-6)
                for cand in existing_markers
            )
            if duplicate:
                continue
            state_dict = dict(state_dict_base)
            state_dict["plane"] = target_plane
            try:
                new_state = MarkerState.from_dict(state_dict)
                new_marker = marker_from_state(new_state)
                manager.add_marker(new_marker)
                mirrored_markers.append(new_marker)
            except Exception:
                continue
        return mirrored_markers

    def open_marker_panel(self):
        needs_new_panel = False
        if self.marker_panel is None:
            needs_new_panel = True
        else:
            try:
                needs_new_panel = not self.marker_panel.isVisible()
            except Exception:
                # Underlying widget was deleted; recreate.
                self.marker_panel = None
                needs_new_panel = True

        if needs_new_panel:
            from tools.marker_panel import MarkerPanel
            self.marker_panel = MarkerPanel(self, self.marker_manager)
            try:
                self.marker_panel.destroyed.connect(lambda: setattr(self, "marker_panel", None))
            except Exception:
                pass
            self.marker_panel.setProperty("_marker_panel_positioned", False)
        self.marker_panel.show()
        if not bool(self.marker_panel.property("_marker_panel_positioned")):
            self._position_marker_panel(self.marker_panel)
            self.marker_panel.setProperty("_marker_panel_positioned", True)
        self.marker_panel.raise_()
        self.marker_panel.activateWindow()

    def _position_marker_panel(self, panel):
        if panel is None:
            return
        try:
            panel.adjustSize()
        except Exception:
            pass
        anchor = self
        anchor_frame = anchor.frameGeometry()
        top_right = anchor_frame.topRight()
        size_hint = panel.sizeHint()
        width = size_hint.width() if size_hint.width() > 0 else panel.width()
        height = size_hint.height() if size_hint.height() > 0 else panel.height()
        margin = 20
        target_x = top_right.x() + margin
        target_y = top_right.y()

        screen = QGuiApplication.screenAt(top_right)
        if screen is None:
            screen = QGuiApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            if width > 0:
                target_x = min(target_x, available.right() - width)
            if height > 0:
                target_y = max(available.top(), min(target_y, available.bottom() - height))
            target_x = max(target_x, available.left())

        panel.move(target_x, target_y)

    def set_marker_mode(self, enabled: bool = True):
        enabled = bool(enabled)
        self.marker_mode_enabled = enabled
        marker_manager = getattr(self, "marker_manager", None)
        if not enabled:
            if marker_manager is not None:
                marker_manager.cancel_placement()
            panel = getattr(self, "marker_panel", None)
            if panel is not None and getattr(panel, "placement_toggle", None) is not None:
                if panel.placement_toggle.isChecked():
                    panel.placement_toggle.blockSignals(True)
                    panel.placement_toggle.setChecked(False)
                    panel.placement_toggle.blockSignals(False)
                    try:
                        panel._on_placement_toggled(False)
                    except Exception:
                        pass
        else:
            default_plane = self.default_marker_plane()
            if marker_manager is not None and default_plane:
                marker_manager.set_active_plane(default_plane)
            if getattr(self, "toolbar", None) is not None:
                try:
                    if self.toolbar.mode in ("zoom rect", "pan/zoom"):
                        self.toolbar.mode = ""
                        self.toolbar._update_buttons_checked()
                except Exception:
                    pass
        if enabled:
            self.canvas.setFocus()

    def on_release(self, event):
        marker_manager = getattr(self, "marker_manager", None)
        if (
            marker_manager is not None
            and getattr(self, "marker_mode_enabled", False)
            and getattr(event, "button", None) == 1
        ):
            marker_manager.handle_release(event)
            self.canvas.draw_idle()

    def on_motion(self, event):
        marker_manager = getattr(self, "marker_manager", None)
        if marker_manager is not None and getattr(self, "marker_mode_enabled", False):
            if marker_manager.is_dragging():
                marker_manager.handle_motion(event)
            else:
                marker_manager.handle_hover(event)
            return

    def on_key_press(self, event):
        marker_manager = getattr(self, "marker_manager", None)
        if marker_manager is not None:
            marker_manager.handle_key_press(event)
            if getattr(self, "marker_mode_enabled", False) and event.key in ("delete", "backspace"):
                panel = getattr(self, "marker_panel", None)
                markers_to_delete = []
                if panel and hasattr(panel, "_selected_markers"):
                    markers_to_delete = panel._selected_markers()

                if not markers_to_delete:
                    marker = marker_manager.selected_marker()
                    if marker:
                        markers_to_delete = [marker]

                if markers_to_delete:
                    planes_to_redraw = set()
                    for marker in markers_to_delete:
                        planes_to_redraw.add(marker.plane)
                        marker_manager.remove_marker(marker.marker_id, marker.plane)

                    for plane in planes_to_redraw:
                        marker_manager.redraw_plane(plane)

    def on_key_release(self, event):
        marker_manager = getattr(self, "marker_manager", None)
        if marker_manager is not None:
            marker_manager.handle_key_release(event)

    def remap_loaded_marker_state(self, state, *, source_plane: Optional[str] = None, world_frame: Optional[str] = None):
        """
        Map incoming marker states to the appropriate channel-map planes.
        - If the state plane matches this viewer's base plane (e.g., saved from main),
          replicate to all tiles.
        - If the state plane matches a channel-map plane (channel_<base>_N), restore to that tile index.
        - Ignore states whose base plane differs.
        """
        base = self._base_plane_from_name(state.plane)
        if base != self.plane:
            return []

        prefix = f"{self.marker_plane_prefix}_"
        plane_name = state.plane or ""
        if plane_name.startswith(prefix):
            try:
                index = int(plane_name[len(prefix):])
            except Exception:
                return []
            if 0 <= index < len(self._marker_planes):
                return [self._marker_planes[index]]
            return []

        return list(self._marker_planes)




    def update_display_initial(self):
        start_index = self.current_page * self.num_per_page
        end_index = start_index + self.num_per_page
        
        total_images = len(self.ch_imdata)
        self.im_list = []
        self.ch_labels = []
        vmin, vmax = np.nanmin(self.flattened_chdata), np.nanmax(self.flattened_chdata)
        for idx, ax in enumerate(self.axes):
            global_index = start_index + idx
            if global_index < total_images:
                tile = self.ch_imdata[global_index]
                if self.plane == 'xy': aspect = 'equal'
                else: aspect = 'auto'
                im = ax.imshow(tile, aspect = aspect, origin="lower", cmap=self.color_pattern)
                im.set_clim(vmin, vmax)
                self.im_list.append(im)
                # Optionally, set a title using the corresponding range label.
                if self.range_label and global_index < len(self.range_label):
                    label = self.range_label[global_index]
                    #ax.set_title(f"Tile {global_index+1}\n{label[0]} ({label[1]}) {label[2]}", fontsize=9)
                if isinstance(label, list):
                    if self.chlabel_num == 0:
                        label = label[1]
                    elif self.chlabel_num == 1:
                        label = f"{label[0]} to {label[2]}"
                self.ch_labels.append(ax.text(self.config['pos_chlabel_x'], self.config['pos_chlabel_y'], label,
                                        transform = ax.transAxes, verticalalignment = 'bottom', horizontalalignment = 'right',
                                        fontfamily=self.config['ch_label_font'], color = self.config['ch_label_color']))
                
            else:
                ax.axis("off")

            axis_unit = []
            axis_type = []
            for i in range(self.wcs.naxis): 
                axis_unit.append(ax.coords[i].get_format_unit())
                if self.wcs.world_axis_physical_types[i] is None: axis_type.append(None)
                else: axis_type.append(self.wcs.world_axis_physical_types[i].split('.')[-1])
            if 'glon' in ax.coords:
                ax.coords['glon'].set_coord_type(coord_wrap = self.config.get('coord_wrap'), coord_type = 'longitude')
            axis_format_decimal = np.isin(axis_type, ['lon', 'lat'])
            if self.config.get('decimal') == False: axis_format_decimal = [False for _ in axis_format_decimal]
            
            for i in np.where(axis_format_decimal)[0]:
                ax.coords[i].set_format_unit(axis_unit[i] , decimal=axis_format_decimal[i])


            # Set tick parameters for all subplots.
            ax.tick_params(which='major', direction=self.config['tick_direction'],
                           length=self.config['tick_length'], color=self.config['tick_color'],
                           width=self.config['tick_width'], labelsize=self.config['tick_labelsize'],
                           labelcolor=self.config['tick_labelcolor'])
            ax.tick_params(which='minor', length=self.config['mtick_length'])
            
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_zorder(5)
                spine.set_linewidth(self.config['tick_width'])
                spine.set_color(self.config['tick_color'])
            # Enable minor ticks on all coordinates.
            #coords = list(ax.coords)
            disp_idx = tuple(True if i else False for i in self.projection_slices)
            for idx in np.where(disp_idx)[0]:
                ax.coords[idx].set_ticks_position(self.config['default_ticks_position'])
                ax.coords[idx].set_ticklabel(exclude_overlapping=True)
                ax.coords[idx].set_ticklabel_visible(True)
                ax.coords[idx].display_minor_ticks(True)
            if self.plane_num == 2:                
                ax.coords[1].set_minor_frequency(self.config['y_mtick_freq'])
                ax.coords[2].set_minor_frequency(self.config['z_mtick_freq'])
                ax.coords[0].set_ticks_visible(False)
                ax.coords[0].set_ticklabel_visible(False)
            elif self.plane_num == 1:
                ax.coords[0].set_minor_frequency(self.config['x_mtick_freq'])
                ax.coords[2].set_minor_frequency(self.config['z_mtick_freq'])
                ax.coords[1].set_ticks_visible(False)
                ax.coords[1].set_ticklabel_visible(False)
            elif self.plane_num == 0:
                ax.coords[0].set_minor_frequency(self.config['x_mtick_freq'])
                ax.coords[1].set_minor_frequency(self.config['y_mtick_freq'])
                ax.coords[2].set_ticks_visible(False)
                ax.coords[2].set_ticklabel_visible(False)


 
        # Create the colorbar axes using parameters from the configuration file.
        self.cax = self.fig.add_axes([
            self.config['cbar_pos_x'],
            self.config['cbar_pos_y'],
            self.config['cbar_width'],
            self.config['cbar_height']
        ])
        self.cax.set_gid('colorbar')
        # Create the colorbar on the first image using the configured orientation.
        self.colorbar = self.fig.colorbar(self.im_list[0],
                                        cax=self.cax,
                                        orientation=self.config['colorbar_orientation'])
        # Set tick parameters for the colorbar's y-axis.
        self.cax.tick_params(
            axis='y', which='both',
            left=self.config['colorbar_tick_left'],
            right=self.config['colorbar_tick_right'],
            labelleft=self.config['colorbar_tick_labelleft'],
            labelright=(not self.config['colorbar_tick_labelleft']),
            width=self.config['colorbar_tick_width'],
            length=self.config['colorbar_tick_length'],
            color=self.config['colorbar_tick_color'],
            direction=self.config['colorbar_tick_direction'],
            labelcolor=self.config['colorbar_tick_labelcolor']
        )
        # Set tick parameters for the colorbar's x-axis.
        self.cax.tick_params(
            axis='x', which='both',
            top=self.config['colorbar_tick_top'],
            bottom=self.config['colorbar_tick_bottom'],
            labeltop=self.config['colorbar_tick_labeltop'],
            labelbottom=(not self.config['colorbar_tick_labeltop']),
            width=self.config['colorbar_tick_width'],
            length=self.config['colorbar_tick_length'],
            color=self.config['colorbar_tick_color'],
            direction=self.config['colorbar_tick_direction'],
            labelcolor=self.config['colorbar_tick_labelcolor']
        )
        # Configure the colorbar outline and label.
        self.colorbar.outline.set_color(self.config['colorbar_tick_color'])
        self.colorbar.outline.set_linewidth(self.config['colorbar_tick_width'])
        self.colorbar.set_label(self.config['colorbar_label'],
                                fontsize=self.config['colorbar_label_fontsize'],
                                color=self.config['colorbar_label_color'])
        # Turn on minor ticks on the colorbar axes.
        self.colorbar.ax.minorticks_on()
        self.colorbar.ax.tick_params(
            which='minor',
            length=self.config['colorbar_mtick_length'],
            color=self.config['colorbar_tick_color']
        )
        # Set minor tick locators for both axes.
        self.colorbar.ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(self.config['colorbar_mtick_freq']))
        self.colorbar.ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(self.config['colorbar_mtick_freq']))


        Common.update_ch_colorbar(self.colorbar)
        Common.update_ch_cax(self.cax)

        ColorSettingsPanel.apply_colorbar_settings(cax = self.cax, colorbar = self.colorbar, config=self.config)   
        
        
        # Determine the bottom–left panel index.
        if self.dir_num == 1:
            bottom_left_index = self.tiles_x - 1
        else:
            bottom_left_index = (self.tiles_x - 1) * self.tiles_y

        # For the bottom–left panel, show tick labels and set axis labels.
        try:
            bl_ax = self.axes[bottom_left_index]
        except IndexError:
            bl_ax = self.axes[0]
        # Get list of coordinate objects.
        coords = list(bl_ax.coords)
        if self.plane_num == 0:  # xy plane: horizontal = X, vertical = Y
            self.hpbw = AddHPBW(bl_ax, self.fits_viewer.header, self.config)
            if len(coords) >= 2:
                coords[0].set_ticklabel_visible(True)
                coords[1].set_ticklabel_visible(True)
                coords[0].set_axislabel(self.xlabel, fontsize=self.config['axislabel_fontsize'],
                                          fontfamily=self.config['axislabel_fontfamily'],
                                          color=self.config['axislabel_color'])
                coords[1].set_axislabel(self.ylabel, fontsize=self.config['axislabel_fontsize'],
                                          fontfamily=self.config['axislabel_fontfamily'],
                                          color=self.config['axislabel_color'])
        elif self.plane_num == 1:  # xz plane: horizontal = X, vertical = Z
            if len(coords) >= 2:
                coords[0].set_ticklabel_visible(True)
                coords[2].set_ticklabel_visible(True)
                coords[0].set_axislabel(self.xlabel, fontsize=self.config['axislabel_fontsize'],
                                          fontfamily=self.config['axislabel_fontfamily'],
                                          color=self.config['axislabel_color'])
                coords[2].set_axislabel(self.zlabel, fontsize=self.config['axislabel_fontsize'],
                                          fontfamily=self.config['axislabel_fontfamily'],
                                          color=self.config['axislabel_color'])
        elif self.plane_num == 2:  # zy plane: horizontal = Z, vertical = Y
            if len(coords) >= 2:  
                coords[2].set_ticklabel_visible(True)
                coords[1].set_ticklabel_visible(True)
                coords[2].set_axislabel(self.zlabel, fontsize=self.config['axislabel_fontsize'],
                                          fontfamily=self.config['axislabel_fontfamily'],
                                          color=self.config['axislabel_color'])
                coords[1].set_axislabel(self.ylabel, fontsize=self.config['axislabel_fontsize'],
                                          fontfamily=self.config['axislabel_fontfamily'],
                                          color=self.config['axislabel_color'])

                coords[1].set_ticklabel_position(self.config['yticklabel_position'])
                coords[1].set_axislabel_position(self.config['yticklabel_position'])

        # For all other panels, hide tick labels and axis labels.
        for idx, ax in enumerate(self.axes):
            if idx != bottom_left_index:
                for coord in ax.coords:
                    coord.set_ticklabel_visible(False)
                    coord.set_axislabel("")

        # Set figure background.
        self.fig.set_facecolor(self.config['fig_background_color'])

        
        # Enable/disable paging buttons.
        self.prev_button.setEnabled(self.current_page > 0)
        self.next_button.setEnabled(end_index < total_images)
        
        self.resize(self.figure_width, self.figure_height)
        #self.fig.tight_layout(rect=(0.1, 0.1, 0.95, 0.9))
        self.canvas.draw()

    def update_images(self):
        total_images = len(self.ch_imdata)
        start_index = self.current_page * self.num_per_page
        end_index = min(start_index + self.num_per_page, total_images)
    
        for idx, ax in enumerate(self.axes):
            global_index = start_index + idx
            if global_index < end_index:
                #if hasattr(ax, 'im_obj'):
                self.im_list[idx].set_data(self.ch_imdata[global_index])
                ax.set_visible(True)
                if self.range_label and global_index < len(self.range_label):
                    label = self.range_label[global_index]
                    #ax.set_title(f"Tile {global_index+1}\n{label[0]} ({label[1]}) {label[2]}", fontsize=9)
                if isinstance(label, list):
                    if self.chlabel_num == 0:
                        label = label[1]
                    elif self.chlabel_num == 1:
                        label = f"{label[0]} to {label[2]}"
                self.ch_labels[idx].set_text(label)
                
                #else:
                    #im = ax.imshow(self.ch_imdata[global_index], origin="lower", cmap=self.color_pattern)
                    #ax.im_obj = im
                    #ax.set_visible(True)
            else:
                ax.set_visible(False)

        self._refresh_contours()
        self.update_axis_labels()
        self.canvas.draw_idle()
        self.prev_button.setEnabled(self.current_page > 0)
        self.next_button.setEnabled(end_index < total_images)
        


    def update_axis_labels(self):
        for ax in self.axes:
            if ax.get_visible():
                for coord in list(ax.coords):
                    coord.set_ticklabel_visible(False)
                    coord.set_axislabel("")
    
        visible_axes = [ax for ax in self.axes if ax.get_visible()]
        if visible_axes:
            bottom_left_ax = max(
                visible_axes,
                key=lambda ax: (ax.get_subplotspec().rowspan.start, -ax.get_subplotspec().colspan.start)
            )
        else:
            bottom_left_ax = self.axes[0]
        coords = list(bottom_left_ax.coords)
        if self.plane_num == 0:
            if len(coords) >= 2:
                coords[0].set_ticklabel_visible(True)
                coords[1].set_ticklabel_visible(True)
                coords[0].set_axislabel(self.xlabel, fontsize=self.config['axislabel_fontsize'],
                                        fontfamily=self.config['axislabel_fontfamily'],
                                        color=self.config['axislabel_color'])
                coords[1].set_axislabel(self.ylabel, fontsize=self.config['axislabel_fontsize'],
                                        fontfamily=self.config['axislabel_fontfamily'],
                                        color=self.config['axislabel_color'])
                if hasattr(self, 'hpbw') and self.hpbw is not None:
                    self.hpbw.update_ax(bottom_left_ax)
                    bottom_left_ax.figure.canvas.draw_idle()
                    
        elif self.plane_num == 1:
            if len(coords) >= 3:
                coords[0].set_ticklabel_visible(True)
                coords[2].set_ticklabel_visible(True)
                coords[0].set_axislabel(self.xlabel, fontsize=self.config['axislabel_fontsize'],
                                        fontfamily=self.config['axislabel_fontfamily'],
                                        color=self.config['axislabel_color'])
                coords[2].set_axislabel(self.zlabel, fontsize=self.config['axislabel_fontsize'],
                                        fontfamily=self.config['axislabel_fontfamily'],
                                        color=self.config['axislabel_color'])
        elif self.plane_num == 2:
            if len(coords) >= 2:
                coords[2].set_ticklabel_visible(True)
                coords[1].set_ticklabel_visible(True)
                coords[2].set_axislabel(self.zlabel, fontsize=self.config['axislabel_fontsize'],
                                        fontfamily=self.config['axislabel_fontfamily'],
                                        color=self.config['axislabel_color'])
                coords[1].set_axislabel(self.ylabel, fontsize=self.config['axislabel_fontsize'],
                                        fontfamily=self.config['axislabel_fontfamily'],
                                        color=self.config['axislabel_color'])
                                        
                coords[1].set_ticklabel_position(self.config['yticklabel_position'])
                coords[1].set_axislabel_position(self.config['yticklabel_position'])

    def prev_page(self):
        """Go to the previous page and update the display."""
        if self.current_page > 0:
            self.current_page -= 1
            self.update_images()

    def next_page(self):
        """Go to the next page and update the display."""
        if (self.current_page + 1) * self.num_per_page < len(self.ch_imdata):
            self.current_page += 1
            self.update_images()

    def open_color_settings(self):
        """
        Open the Colorscale settings panel.
        (For now, simply display an information message.)
        """

        if self.color_settings_panel is None:
            self.color_settings_panel = ColorSettingsPanel(
                mode=ColorMode.CHANNEL,
                fits_viewer=self,
                data=self.flattened_chdata,
                config=self.config,
                color_pattern= self.color_pattern,
                filename = self.fits_viewer.filename,
                bad_color = self.fits_viewer.displaymap.bad_color
            )
            self.color_settings_panel.show()
            self.color_settings_panel.destroyed.connect(self.on_color_settings_closed)
        else:
            self.color_settings_panel.raise_()
            self.color_settings_panel.activateWindow()

    def on_color_settings_closed(self):
        self.color_settings_panel = None

    def _default_contour_label(self) -> str:
        title = self.windowTitle() or "Channel Map"
        plane = getattr(self, "plane", None)
        if plane:
            return f"{title} [{plane.upper()}]"
        return title

    def _contour_items_provider(self):
        items = []
        label_base = self._default_contour_label()
        for idx, (ax, image) in enumerate(zip(self.axes, self.im_list)):
            if ax is None or image is None:
                continue
            arr = image.get_array()
            if arr is None:
                continue
            if np.ma.isMaskedArray(arr):
                arr = arr.filled(np.nan)
            data = np.asarray(arr)
            label = f"{label_base} Tile {idx + 1}"
            metadata = {}
            try:
                clim = image.get_clim()
            except Exception:
                clim = None
            if clim is not None:
                metadata["clim"] = tuple(clim)
            items.append(ContourItem(ax=ax, data=data, label=label, metadata=metadata))
        return items

    def _register_contour_layer(self):
        if self._contour_layer_id is not None:
            return
        manager = ContourManager.instance()
        layer_id = f"channel-{id(self)}"
        try:
            manager.register_layer(
                layer_id=layer_id,
                label=self._default_contour_label(),
                plane=getattr(self, "plane", None),
                provider=self._contour_items_provider,
                owner=self,
            )
        except ValueError:
            return
        self._contour_layer_id = layer_id
        manager.contour_updated.connect(self._on_contour_updated)
        if not self._contour_title_connected:
            try:
                self.windowTitleChanged.connect(self._handle_title_change_for_contours)
                self._contour_title_connected = True
            except Exception:
                pass

    def _handle_title_change_for_contours(self, _title: str) -> None:
        if not self._contour_layer_id:
            return
        ContourManager.instance().rename_layer(self._contour_layer_id, self._default_contour_label())

    def _on_contour_updated(self, layer_id: str):
        """
        Slot to receive signal from ContourManager after contours are updated.
        """
        if layer_id == self._contour_layer_id:
            self._background = None 
            self.canvas.draw_idle()

    def _refresh_contours(self):
        if not self._contour_layer_id:
            return
        ContourManager.instance().refresh_layer(self._contour_layer_id)

    def _unregister_contour_layer(self):
        if not self._contour_layer_id:
            return
        ContourManager.instance().unregister_layer(self._contour_layer_id)
        self._contour_layer_id = None

    def closeEvent(self, event):
        #self.remove_colorbar()
        #self.toolbar._subplot_dialog = None
        self._unregister_contour_layer()
        if self.color_settings_panel is not None:
            self.color_settings_panel.close()
            self.color_settings_panel = None
        self.set_marker_mode(False)
        self.set_marker_link_all(False)
        if self.marker_panel is not None:
            try:
                self.marker_panel.close()
            except Exception:
                pass
            self.marker_panel = None
    
        if self.canvas is not None:
            self.canvas.close()
            self.canvas = None
        if self.fig is not None:
            self.fig.clear()
            self.fig = None
        #print("Result window closed")
        if self.toolbar._subplot_dialog is not None:
            self.toolbar._subplot_dialog.close()
            self.toolbar._subplot_dialog = None
        super().closeEvent(event)
        #event.accept()
        try:
            self.destroyed.emit()
        except Exception:
            pass


    def initialize_ranges(self):
        if self.fits_viewer.data.ndim == 3:
            self.xmin_val = self.converter.pix_to_world(self.original_xlim[0], self.original_ylim[0], 0)[0]
            self.xmax_val = self.converter.pix_to_world(self.original_xlim[1], self.original_ylim[1], 0)[0]
            self.ymin_val = self.converter.pix_to_world(self.original_xlim[0], self.original_ylim[0], 0)[1]
            self.ymax_val = self.converter.pix_to_world(self.original_xlim[1], self.original_ylim[1], 0)[1]
            self.zmin_val = self.converter.pix_to_world(0, 0, self.original_zlim[0])[2]
            self.zmax_val = self.converter.pix_to_world(0, 0, self.original_zlim[1])[2]
        elif self.fits_viewer.data.ndim == 4:
            self.xmin_val = self.converter.pix_to_world(self.original_xlim[0], self.original_ylim[0], 0, 0)[0]
            self.xmax_val = self.converter.pix_to_world(self.original_xlim[1], self.original_ylim[1], 0, 0)[0]
            self.ymin_val = self.converter.pix_to_world(self.original_xlim[0], self.original_ylim[0], 0, 0)[1]
            self.ymax_val = self.converter.pix_to_world(self.original_xlim[1], self.original_ylim[1], 0, 0)[1]
            self.zmin_val = self.converter.pix_to_world(0, 0, self.original_zlim[0], 0)[2]
            self.zmax_val = self.converter.pix_to_world(0, 0, self.original_zlim[1], 0)[2]


    def sync_range(self):
        if self.plane == 'xy' or self.plane == 'xz':
            x_min = Common.xmin_input_xy.text()
            x_max = Common.xmax_input_xy.text()
            self.x_min_ch_input.setText(x_min)
            self.x_max_ch_input.setText(x_max)
            self.set_x_range()
            
        if self.plane == 'xy' or self.plane == 'zy':
            y_min = Common.ymin_input_xy.text()
            y_max = Common.ymax_input_xy.text()
            self.y_min_ch_input.setText(y_min)
            self.y_max_ch_input.setText(y_max)
            self.set_y_range()
            
        if self.plane == 'xz' or self.plane == 'zy':
            z_min = Common.zmin_input_xz.text()
            z_max = Common.zmax_input_xz.text()
            self.z_min_ch_input.setText(z_min)
            self.z_max_ch_input.setText(z_max)
            self.set_z_range()


    def set_full_range(self):
        if self.fits_viewer.data.ndim == 3:
            self.xmin_val_full = self.converter.pix_to_world(-0.5, 0, 0)[0]
            self.xmax_val_full = self.converter.pix_to_world(self.xnpix+0.5, 0, 0)[0]
            self.ymin_val_full = self.converter.pix_to_world(0, -0.5, 0)[1]
            self.ymax_val_full = self.converter.pix_to_world(0, self.ynpix+0.5, 0)[1]
            self.zmin_val_full = self.converter.pix_to_world(0, 0, -0.5)[2]
            self.zmax_val_full = self.converter.pix_to_world(0, 0, self.znpix+0.5)[2]
        elif self.fits_viewer.data.ndim == 4:
            self.xmin_val_full = self.converter.pix_to_world(-0.5, 0, 0, 0)[0]
            self.xmax_val_full = self.converter.pix_to_world(self.xnpix+0.5, 0, 0, 0)[0]
            self.ymin_val_full = self.converter.pix_to_world(0, -0.5, 0, 0)[1]
            self.ymax_val_full = self.converter.pix_to_world(0, self.ynpix+0.5, 0, 0)[1]
            self.zmin_val_full = self.converter.pix_to_world(0, 0, -0.5, 0)[2]
            self.zmax_val_full = self.converter.pix_to_world(0, 0, self.znpix+0.5, 0)[2]
        
        if self.plane == 'xy' or self.plane == 'xz':
            self.x_min_ch_input.setText(str(self.xmin_val_full))
            self.x_max_ch_input.setText(str(self.xmax_val_full))
            self.set_x_range()
            
        if self.plane == 'xy' or self.plane == 'zy':
            self.y_min_ch_input.setText(str(self.ymin_val_full))
            self.y_max_ch_input.setText(str(self.ymax_val_full))
            self.set_y_range()
            
        if self.plane == 'xz' or self.plane == 'zy':
            self.z_min_ch_input.setText(str(self.zmin_val_full))
            self.z_max_ch_input.setText(str(self.zmax_val_full))
            self.set_z_range()
            
    def set_x_range(self):
        try:
            x_min = self.x_min_ch_input.text()
            x_max = self.x_max_ch_input.text()
            if self.fits_viewer.data.ndim == 3:
                xp_min = float(self.converter.world_to_pix(x_min, self.ymin_val, self.zmin_val)[0])
                xp_max = float(self.converter.world_to_pix(x_max, self.ymax_val, self.zmax_val)[0])
            elif self.fits_viewer.data.ndim == 4:
                xp_min = float(self.converter.world_to_pix(x_min, self.ymin_val, self.zmin_val, 0)[0])
                xp_max = float(self.converter.world_to_pix(x_max, self.ymax_val, self.zmax_val, 0)[0])
                
            if xp_min > xp_max: xp_min, xp_max = xp_max, xp_min

            for ax in self.axes: ax.set_xlim(xp_min, xp_max)
            self.canvas.draw()
        except ValueError:
            QMessageBox.warning(self, 'Invalid Input', 'Please enter valid numeric values for the X range.')
            
    def set_y_range(self):
        try:
            y_min = self.y_min_ch_input.text()
            y_max = self.y_max_ch_input.text()
            if self.fits_viewer.data.ndim == 3:
                yp_min = float(self.converter.world_to_pix(self.xmin_val, y_min, self.zmin_val)[1])
                yp_max = float(self.converter.world_to_pix(self.xmax_val, y_max, self.zmax_val)[1])
            elif self.fits_viewer.data.ndim == 4:
                yp_min = float(self.converter.world_to_pix(self.xmin_val, y_min, self.zmin_val, 0)[1])
                yp_max = float(self.converter.world_to_pix(self.xmax_val, y_max, self.zmax_val, 0)[1])
                
            if yp_min > yp_max: yp_min, yp_max = yp_max, yp_min

            for ax in self.axes: ax.set_ylim(yp_min, yp_max)
            self.canvas.draw()
        except ValueError:
            QMessageBox.warning(self, 'Invalid Input', 'Please enter valid numeric values for the Y range.')


    def set_z_range(self):
        try:
            z_min = self.z_min_ch_input.text()
            z_max = self.z_max_ch_input.text()
            if self.fits_viewer.data.ndim == 3:
                zp_min = float(self.converter.world_to_pix(self.xmin_val, self.ymin_val, z_min)[2])
                zp_max = float(self.converter.world_to_pix(self.xmax_val, self.ymax_val, z_max)[2])
            elif self.fits_viewer.data.ndim == 4:
                zp_min = float(self.converter.world_to_pix(self.xmin_val, self.ymin_val, z_min, 0)[2])
                zp_max = float(self.converter.world_to_pix(self.xmax_val, self.ymax_val, z_max, 0)[2])
                
            if zp_min > zp_max: zp_min, zp_max = zp_max, zp_min
            
            if self.plane == 'xz':
                for ax in self.axes: ax.set_ylim(zp_min, zp_max)
            elif self.plane == 'zy':
                for ax in self.axes: ax.set_xlim(zp_min, zp_max)
            self.canvas.draw()
        except ValueError:
            QMessageBox.warning(self, 'Invalid Input', 'Please enter valid numeric values for the Z range.')

    def update_ranges(self, plane, xlim, ylim):
        if plane == "xy":
            new_xlim = xlim
            new_ylim = ylim
            if self.fits_viewer.data.ndim == 3:
                xmin_val = self.converter.pix_to_world(new_xlim[0], new_ylim[0], 0)[0]
                xmax_val = self.converter.pix_to_world(new_xlim[1], new_ylim[1], 0)[0]
                ymin_val = self.converter.pix_to_world(new_xlim[0], new_ylim[0], 0)[1]
                ymax_val = self.converter.pix_to_world(new_xlim[1], new_ylim[1], 0)[1]
            elif self.fits_viewer.data.ndim == 4:
                xmin_val = self.converter.pix_to_world(new_xlim[0], new_ylim[0], 0, 0)[0]
                xmax_val = self.converter.pix_to_world(new_xlim[1], new_ylim[1], 0, 0)[0]
                ymin_val = self.converter.pix_to_world(new_xlim[0], new_ylim[0], 0, 0)[1]
                ymax_val = self.converter.pix_to_world(new_xlim[1], new_ylim[1], 0, 0)[1]
            
            
            self.x_min_ch_input.setText(str(xmin_val))
            self.x_max_ch_input.setText(str(xmax_val))
            self.y_min_ch_input.setText(str(ymin_val))
            self.y_max_ch_input.setText(str(ymax_val))
            
            self.set_x_range()
            self.set_y_range()

        elif plane == "xz":
            new_xlim = xlim
            new_zlim = ylim
            if self.fits_viewer.data.ndim == 3:
                xmin_val = self.converter.pix_to_world(new_xlim[0], 0, 0)[0]
                xmax_val = self.converter.pix_to_world(new_xlim[1], 0, 0)[0]
                zmin_val = self.converter.pix_to_world(0, 0, new_zlim[0])[2]
                zmax_val = self.converter.pix_to_world(0, 0, new_zlim[1])[2]
            elif self.fits_viewer.data.ndim == 4:
                xmin_val = self.converter.pix_to_world(new_xlim[0], 0, 0, 0)[0]
                xmax_val = self.converter.pix_to_world(new_xlim[1], 0, 0, 0)[0]
                zmin_val = self.converter.pix_to_world(0, 0, new_zlim[0], 0)[2]
                zmax_val = self.converter.pix_to_world(0, 0, new_zlim[1], 0)[2]
                
            self.x_min_ch_input.setText(str(xmin_val))
            self.x_max_ch_input.setText(str(xmax_val))
            self.z_min_ch_input.setText(str(zmin_val))
            self.z_max_ch_input.setText(str(zmax_val))

            self.set_x_range()
            self.set_z_range()
            
        elif plane == "zy":
            new_zlim = xlim
            new_ylim = ylim
            if self.fits_viewer.data.ndim == 3:
                zmin_val = self.converter.pix_to_world(0, 0, new_zlim[0])[2]
                zmax_val = self.converter.pix_to_world(0, 0, new_zlim[1])[2]
                ymin_val = self.converter.pix_to_world(0, new_ylim[0], 0)[1]
                ymax_val = self.converter.pix_to_world(0, new_ylim[1], 0)[1]
            elif self.fits_viewer.data.ndim == 4:
                zmin_val = self.converter.pix_to_world(0, 0, new_zlim[0], 0)[2]
                zmax_val = self.converter.pix_to_world(0, 0, new_zlim[1], 0)[2]
                ymin_val = self.converter.pix_to_world(0, new_ylim[0], 0, 0)[1]
                ymax_val = self.converter.pix_to_world(0, new_ylim[1], 0, 0)[1]
            
            
            self.z_min_ch_input.setText(str(zmin_val))
            self.z_max_ch_input.setText(str(zmax_val))
            self.y_min_ch_input.setText(str(ymin_val))
            self.y_max_ch_input.setText(str(ymax_val))
            
            self.set_z_range()
            self.set_y_range()
            
    def on_click(self, event):
        if event.inaxes and event.inaxes.get_gid() == 'colorbar': return
        self.toolbar.set_axes(event.inaxes)
        if event.dblclick:
            current_mode = self.toolbar.mode
            if current_mode == 'pan/zoom':
                self.toolbar.pan(False)
                self.toolbar._active = None 
                release_event = mpl.backend_bases.MouseEvent(
                    name='button_release_event',
                    canvas=self.canvas,
                    x=event.x,
                    y=event.y,
                    button=event.button,
                    key=event.key,
                    step=event.step,
                    dblclick=event.dblclick,
                    guiEvent=event.guiEvent
                )
                self.toolbar.release_pan(release_event)
    
            elif current_mode == 'zoom rect':
                self.toolbar.zoom(False)
                self.toolbar._active = None
                release_event = mpl.backend_bases.MouseEvent(
                    name='button_release_event',
                    canvas=self.canvas,
                    x=event.x,
                    y=event.y,
                    button=event.button,
                    key=event.key,
                    step=event.step,
                    dblclick=event.dblclick,
                    guiEvent=event.guiEvent
                )
                self.toolbar.release_zoom(release_event)
            else:
                return
            self.toolbar._update_buttons_checked()
            self.toolbar.set_message('')
            return

        if (
            getattr(self, "marker_mode_enabled", False)
            and self.toolbar.mode == ''
            and getattr(event, "button", None) == 1
            and event.inaxes in self.axes
        ):
            marker_manager = getattr(self, "marker_manager", None)
            if marker_manager is not None:
                plane = self.marker_plane_for_axes(event.inaxes)
                if plane:
                    self.canvas.setFocus()
                    marker_manager.set_active_plane(plane)
                    marker_manager.handle_press(event)
                    self.canvas.draw_idle()
            return

    def resizeEvent(self, event):
        if self.toolbar._subplot_dialog is not None:
            self.toolbar._subplot_dialog.width_spin.setValue(int(self.window().width()))
            self.toolbar._subplot_dialog.height_spin.setValue(int(self.window().height()))
        super().resizeEvent(event)

###########
class ChannelMapSettingPanel(QDialog):
    def __init__(self, fits_viewer, subwindows):
        super().__init__()
        self.fits_viewer = fits_viewer
        self.wcs = self.fits_viewer.wcs
        self.subwindows = subwindows

        # Default values for settings
        self.tiles_x = 4
        self.tiles_y = 4
        self.interval_val = 1
        
        
        self.plane_num = 0   # 0 => X-Y, 1 => X-V, 2 => V-Y
        self.mode_num = 0    # 0 => Integ, 1 => Average, 2 => Slice
        self.worldch_num = 0   # 0 => ch, 1 => world
        self.dir_num = 0     # 0 => L to R, 1 => T to B

        self.channel_result_windows = []
        self.coord_wrap = self.fits_viewer.config_manager.config.get('coord_wrap')
        
        self.original_xlim = self.fits_viewer.ax.get_xlim()
        self.original_ylim = self.fits_viewer.ax.get_ylim()
        self.original_zlim = self.subwindows[0].ax.get_ylim()
        
        self.from_val =  0.5
        self.to_val =  self.fits_viewer.header['NAXIS3']+0.5
        self.from_pix = 0.5
        self.to_pix = self.fits_viewer.header['NAXIS3']+0.5
        self.interval_pix = 1

        self.initUI()

    def initUI(self):
        self.converter = CoordinateConverter(self.wcs, self.fits_viewer.config_manager.config)
        self.setWindowTitle(f"Channel Map Setting: {self.fits_viewer.filename}")

        # Create main layout for the dialog
        main_layout = QGridLayout()
        #main_layout.setHorizontalSpacing(5)
        #main_layout.setVerticalSpacing(5)
        main_layout.setContentsMargins(15, 5, 15, 5)

        # --- Plane GroupBox ---
        plane_group = QGroupBox("Plane")
        plane_layout = QVBoxLayout()
        # Radio buttons for plane selection: 0 => X-Y, 1 => X-V, 2 => V-Y
        self.plane_xy_radio = QRadioButton("X-Y")
        self.plane_xz_radio = QRadioButton("X-V")
        self.plane_zy_radio = QRadioButton("V-Y")
        self.plane_xy_radio.setChecked(True)
        # Connect signals
        self.plane_xy_radio.toggled.connect(lambda: self.set_plane_num(0))
        self.plane_xz_radio.toggled.connect(lambda: self.set_plane_num(1))
        self.plane_zy_radio.toggled.connect(lambda: self.set_plane_num(2))
        plane_layout.addWidget(self.plane_xy_radio)
        plane_layout.addWidget(self.plane_xz_radio)
        plane_layout.addWidget(self.plane_zy_radio)
        plane_group.setLayout(plane_layout)
        plane_group.setFixedWidth(70)
        main_layout.addWidget(plane_group, 0, 0, 2, 1)

        # --- Mode GroupBox ---
        mode_group = QGroupBox("Mode")
        mode_layout = QVBoxLayout()
        # Radio buttons for mode selection: 0 => Integ, 1 => Average, 2 => Slice
        self.mode_integ_radio = QRadioButton("Integ.")
        self.mode_avg_radio = QRadioButton("Average")
        self.mode_slice_radio = QRadioButton("Slice")
        self.mode_integ_radio.setChecked(True)
        self.mode_integ_radio.toggled.connect(lambda: self.set_mode_num(0))
        self.mode_avg_radio.toggled.connect(lambda: self.set_mode_num(1))
        self.mode_slice_radio.toggled.connect(lambda: self.set_mode_num(2))
        mode_layout.addWidget(self.mode_integ_radio)
        mode_layout.addWidget(self.mode_avg_radio)
        mode_layout.addWidget(self.mode_slice_radio)
        mode_group.setLayout(mode_layout)
        mode_group.setFixedWidth(90)
        main_layout.addWidget(mode_group, 0, 1, 2, 1)

        # --- Tiles GroupBox (Number of Panels + Direction) ---
        tiles_group = QGroupBox("Tiles")
        tiles_layout = QHBoxLayout()
        tiles_layout.setSpacing(2)  # Reduce spacing between widgets
        tiles_label = QLabel("row")
        self.tiles_x_combo = QComboBox()
        self.tiles_y_combo = QComboBox()
        # Add values 1..10
        for i in range(1, 11):
            self.tiles_x_combo.addItem(str(i))
            self.tiles_y_combo.addItem(str(i))
        self.tiles_x_combo.setCurrentText(str(self.tiles_x))
        self.tiles_y_combo.setCurrentText(str(self.tiles_y))
        self.tiles_x_combo.setFixedWidth(55)
        self.tiles_y_combo.setFixedWidth(55)
        self.tiles_x_combo.currentIndexChanged.connect(self.update_tiles_x)
        self.tiles_y_combo.currentIndexChanged.connect(self.update_tiles_y)
        tiles_layout.addWidget(tiles_label, alignment=Qt.AlignmentFlag.AlignLeft)
        tiles_layout.addWidget(self.tiles_x_combo, alignment=Qt.AlignmentFlag.AlignLeft)
        # Remove extra spacing between "Column:" label and combobox by adding them consecutively
        col_label = QLabel("column")
        tiles_layout.addWidget(col_label, alignment=Qt.AlignmentFlag.AlignLeft)
        tiles_layout.addWidget(self.tiles_y_combo, alignment=Qt.AlignmentFlag.AlignLeft)
        # Add Direction radio buttons within the Tiles group
        direction_label = QLabel("  direction")
        self.dir_l2r_radio = QRadioButton("L to R")
        self.dir_t2b_radio = QRadioButton("T to B")
        self.dir_l2r_radio.setChecked(True)
        # Create a button group to enforce exclusivity
        self.direction_group = QButtonGroup(self)
        self.direction_group.addButton(self.dir_l2r_radio, 0)
        self.direction_group.addButton(self.dir_t2b_radio, 1)
        self.dir_l2r_radio.toggled.connect(lambda: self.set_dir_num(0))
        self.dir_t2b_radio.toggled.connect(lambda: self.set_dir_num(1))
        tiles_layout.addWidget(direction_label, alignment=Qt.AlignmentFlag.AlignRight)
        tiles_layout.addWidget(self.dir_l2r_radio, alignment=Qt.AlignmentFlag.AlignLeft)
        tiles_layout.addWidget(self.dir_t2b_radio, alignment=Qt.AlignmentFlag.AlignLeft)
        tiles_group.setLayout(tiles_layout)
        main_layout.addWidget(tiles_group, 1, 2, 2, 1)

        # --- Range GroupBox (Range, Interval, and Unit) ---
        range_group = QGroupBox("Range and Interval")
        range_layout = QHBoxLayout()
        range_layout.setSpacing(2)  # Reduce spacing between widgets in the Range group
        # Add From, To, and Interval fields
        self.from_edit = QLineEdit(str(self.from_val))
        self.to_edit = QLineEdit(str(self.to_val))
        interval_label = QLabel("  Interval:")
        self.interval_edit = QLineEdit(str(self.interval_val))
        self.interval_edit.setFixedWidth(65)
        range_layout.addWidget(self.from_edit)
        range_layout.addWidget(self.to_edit)
        range_layout.addWidget(interval_label)
        range_layout.addWidget(self.interval_edit)
        # Add Unit radio buttons into the Range group
        #unit_label = QLabel("Unit:")
        self.ch_radio = QRadioButton("ch")
        self.val_radio = QRadioButton("world")
        # Create a button group for exclusivity
        self.ch_val_group = QButtonGroup(self)
        self.ch_val_group.addButton(self.ch_radio, 0)
        self.ch_val_group.addButton(self.val_radio, 1)
        self.ch_radio.toggled.connect(lambda: self.set_worldch_num(0))
        self.val_radio.toggled.connect(lambda: self.set_worldch_num(1))
        self.val_radio.setChecked(True)
        #range_layout.addWidget(unit_label, alignment=Qt.AlignmentFlag.AlignRight)
        range_layout.addWidget(self.ch_radio)
        range_layout.addWidget(self.val_radio)
        range_group.setLayout(range_layout)
        main_layout.addWidget(range_group, 0, 2, 1, 2)
        
        chlabel_group = QGroupBox("Ch. label type")
        chlabel_radio_layout = QHBoxLayout()
        chlabel_radio_layout.setSpacing(0)
        self.chlabel_radio_middle = QRadioButton("middle")
        self.chlabel_radio_range = QRadioButton("range")
        self.chlabel_radio_group = QButtonGroup(self)
        self.chlabel_radio_group.addButton(self.chlabel_radio_middle, 0)
        self.chlabel_radio_group.addButton(self.chlabel_radio_range, 1)
        self.chlabel_radio_middle.toggled.connect(lambda: self.set_chlabel_num(0))
        self.chlabel_radio_range.toggled.connect(lambda: self.set_chlabel_num(1))
        chlabel_radio_layout.addWidget(self.chlabel_radio_middle)
        chlabel_radio_layout.addWidget(self.chlabel_radio_range)
        self.chlabel_radio_middle.setChecked(True)
        chlabel_group.setLayout(chlabel_radio_layout)
        main_layout.addWidget(chlabel_group, 2, 0, 1, 2)
        
        # --- Auto Set Interval Button ---
        self.auto_interval_button = QPushButton("Auto Interval")
        self.auto_interval_button.clicked.connect(self.auto_set_interval)
        main_layout.addWidget(self.auto_interval_button, 1, 3, 1, 1)

        # --- Execute Button ---
        self.execute_button = QPushButton("Execute")
        self.execute_button.clicked.connect(self.mkchmap)
        self.execute_button.setAutoDefault(True) 
        self.execute_button.setDefault(True)
        main_layout.addWidget(self.execute_button, 2, 3, 1, 1)



        # Set the main layout for the dialog
        self.setLayout(main_layout)
        self.move_to_default_position()

    # ----------------------------
    # Placeholder functions for handling events:
    # ----------------------------

    def set_plane_num(self, num):
        """Set plane number based on selected plane radio button."""
        if not self.sender().isChecked(): return
        self.plane_num = num
        if self.worldch_num == 0: #ch
            self.from_val =  0.5
            if self.plane_num == 0: #X-Y
                self.to_val = self.fits_viewer.header['NAXIS3']+0.5
            elif self.plane_num == 1:
                self.to_val = self.fits_viewer.header['NAXIS2']+0.5
            elif self.plane_num == 2:
                self.to_val = self.fits_viewer.header['NAXIS1']+0.5
            self.interval = 1
            
        elif self.worldch_num == 1: #world
            if self.plane_num == 0: #X-Y
                if self.fits_viewer.data.ndim == 3:
                    self.from_val = self.converter.pix_to_world(0, 0, -0.5)[2]
                    self.to_val = self.converter.pix_to_world(0, 0, int(self.fits_viewer.header['NAXIS3'])-0.5)[2]
                elif self.fits_viewer.data.ndim == 4:
                    self.from_val = self.converter.pix_to_world(0, 0, -0.5, 0)[2]
                    self.to_val = self.converter.pix_to_world(0, 0, int(self.fits_viewer.header['NAXIS3'])-0.5, 0)[2]
            elif self.plane_num == 1: #X-Z
                if self.fits_viewer.data.ndim == 3:
                    self.from_val = self.converter.pix_to_world(0, -0.5, 0)[1]
                    self.to_val = self.converter.pix_to_world(0, int(self.fits_viewer.header['NAXIS2'])-0.5, 0)[1]
                elif self.fits_viewer.data.ndim == 4:
                    self.from_val = self.converter.pix_to_world(0, -0.5, 0, 0)[1]
                    self.to_val = self.converter.pix_to_world(0, int(self.fits_viewer.header['NAXIS2'])-0.5, 0, 0)[1]
            elif self.plane_num == 2: #Z-Y
                if self.fits_viewer.data.ndim == 3:
                    self.from_val = self.converter.pix_to_world(-0.5, 0, 0)[0]
                    self.to_val = self.converter.pix_to_world(int(self.fits_viewer.header['NAXIS1'])-0.5, 0, 0)[0]
                elif self.fits_viewer.data.ndim == 4:
                    self.from_val = self.converter.pix_to_world(-0.5, 0, 0, 0)[0]
                    self.to_val = self.converter.pix_to_world(int(self.fits_viewer.header['NAXIS1'])-0.5, 0, 0, 0)[0]
                    
            interval_val = abs(self.wcs.wcs.cdelt[2-self.plane_num])
            axis_type = self.converter.get_axis_types()[2-self.plane_num]
            self.interval = self.converter.format_world_coordinate(interval_val, axis_type)
            
        self.from_edit.setText(str(self.from_val))
        self.to_edit.setText(str(self.to_val))
        self.interval_edit.setText(str(self.interval))
        if self.mode_num == 2:
            self.set_mode_num(2)

    def set_worldch_num(self, num):
        """Set unit selection (0 for 'ch', 1 for 'world')."""
        sender = self.sender()
        # Ignore if sender is being unchecked.
        if not sender.isChecked():
            return
    
        # Save current state in case we need to revert.
        previous_state = self.worldch_num  # 0 for 'ch', 1 for 'world'
        self.worldch_num = num
    
        if self.val_radio.isChecked():
            try:
                # Convert the text fields to float (for pixel indices)
                from_pix = float(self.from_edit.text())
                to_pix =float(self.to_edit.text())
                self.interval_pix = float(self.interval_edit.text())
                
            except ValueError:
                QMessageBox.warning(self, 'Error', 'Invalid range values')
                # Revert radio button selection to previous state:
                self._revert_worldch_radio(previous_state)
                return
            try:
                # Convert pixel indices to world values according to selected plane
                if self.plane_num == 0:
                    if self.fits_viewer.data.ndim == 3:
                        from_val = self.converter.pix_to_world(0, 0, from_pix-1)[2]
                        to_val = self.converter.pix_to_world(0, 0, to_pix-1)[2]
                    elif self.fits_viewer.data.ndim == 4:
                        from_val = self.converter.pix_to_world(0, 0, from_pix-1, 0)[2]
                        to_val = self.converter.pix_to_world(0, 0, to_pix-1, 0)[2]
                    interval_val = abs(self.interval_pix * self.wcs.wcs.cdelt[2])
                    axis_type = self.converter.get_axis_types()[2]
                elif self.plane_num == 1:
                    if self.fits_viewer.data.ndim == 3:
                        from_val = self.converter.pix_to_world(0, from_pix-1, 0)[1]
                        to_val = self.converter.pix_to_world(0, to_pix-1, 0)[1]
                    elif self.fits_viewer.data.ndim == 4:
                        from_val = self.converter.pix_to_world(0, from_pix-1, 0, 0)[1]
                        to_val = self.converter.pix_to_world(0, to_pix-1, 0, 0)[1]
                    interval_val = abs(self.interval_pix * self.wcs.wcs.cdelt[1])
                    axis_type = self.converter.get_axis_types()[1]
                elif self.plane_num == 2:
                    if self.fits_viewer.data.ndim == 3:
                        from_val = self.converter.pix_to_world(from_pix-1, 0, 0)[0]
                        to_val = self.converter.pix_to_world(to_pix-1, 0, 0)[0]
                    elif self.fits_viewer.data.ndim == 4:
                        from_val = self.converter.pix_to_world(from_pix-1, 0, 0, 0)[0]
                        to_val = self.converter.pix_to_world(to_pix-1, 0, 0, 0)[0]
                    interval_val = abs(self.interval_pix * self.wcs.wcs.cdelt[0])
                    axis_type = self.converter.get_axis_types()[0]
                # If conversion is successful, update text fields with world values.
                self.from_edit.setText(str(from_val))
                self.to_edit.setText(str(to_val))
                self.interval_edit.setText(str(self.converter.format_world_coordinate(interval_val, axis_type)))
                self.interval_edit.setCursorPosition(0)
                
            except Exception as e:
                QMessageBox.warning(self, 'Error', f'Conversion error: {e}')
                self._revert_worldch_radio(previous_state)
                return
    
        elif self.ch_radio.isChecked():
            try:
                if self.plane_num == 0:
                    from_world = self.from_edit.text()
                    to_world = self.to_edit.text()
                    interval_val = self.interval_edit.text()
                else:
                    from_world = float( Angle(self.from_edit.text(), unit=u.deg).degree)
                    to_world =float( Angle(self.to_edit.text(), unit=u.deg).degree)
                    interval_val = float( Angle(self.interval_edit.text(), unit=u.deg).degree)
                
                if self.plane_num == 0:
                    if self.fits_viewer.data.ndim == 3:
                        self.origin_xval, self.origin_yval, self.origin_zval = self.converter.pix_to_world(0, 0, 0)
                        from_val = self.converter.world_to_pix(self.origin_xval, self.origin_yval, from_world)[2] + 1
                        to_val = self.converter.world_to_pix(self.origin_xval, self.origin_yval, to_world)[2] + 1
                    elif self.fits_viewer.data.ndim == 4:
                        self.origin_xval, self.origin_yval, self.origin_zval, _ = self.converter.pix_to_world(0, 0, 0, 0)
                        from_val = self.converter.world_to_pix(self.origin_xval, self.origin_yval, from_world, 0)[2] + 1
                        to_val = self.converter.world_to_pix(self.origin_xval, self.origin_yval, to_world, 0)[2] + 1
                    axis_type = self.converter.get_axis_types()[2]
                    self.interval_pix = float(interval_val)/self.wcs.wcs.cdelt[2]
    
                elif self.plane_num == 1:
                    if self.fits_viewer.data.ndim == 3:
                        self.origin_xval, self.origin_yval, self.origin_zval = self.converter.pix_to_world(0, 0, 0)
                        from_val = self.converter.world_to_pix(self.origin_xval, from_world, self.origin_zval)[1] + 1
                        to_val = self.converter.world_to_pix(self.origin_xval, to_world, self.origin_zval)[1] + 1
                    elif self.fits_viewer.data.ndim == 4:
                        self.origin_xval, self.origin_yval, self.origin_zval, _ = self.converter.pix_to_world(0, 0, 0, 0)
                        from_val = self.converter.world_to_pix(self.origin_xval, from_world, self.origin_zval, 0)[1] + 1
                        to_val = self.converter.world_to_pix(self.origin_xval, to_world, self.origin_zval, 0)[1] + 1
                    axis_type = self.converter.get_axis_types()[1]
                    if self.fits_viewer.config_manager.config.get('decimal'):
                        self.interval_pix = float(interval_val)/self.wcs.wcs.cdelt[1]
                    else:
                        interval_val = Angle(interval_val, unit=u.deg).degree
                        self.interval_pix = float(interval_val)/self.wcs.wcs.cdelt[1]
    
                elif self.plane_num == 2:
                    if self.fits_viewer.data.ndim == 3:
                        self.origin_xval, self.origin_yval, self.origin_zval = self.converter.pix_to_world(0, 0, 0)
                        from_val = self.converter.world_to_pix(from_world, self.origin_yval, self.origin_zval)[0] + 1
                        to_val = self.converter.world_to_pix(to_world, self.origin_yval, self.origin_zval)[0] + 1
                    elif self.fits_viewer.data.ndim == 4:
                        self.origin_xval, self.origin_yval, self.origin_zval, _ = self.converter.pix_to_world(0, 0, 0, 0)
                        from_val = self.converter.world_to_pix(from_world, self.origin_yval, self.origin_zval, 0)[0] + 1
                        to_val = self.converter.world_to_pix(to_world, self.origin_yval, self.origin_zval, 0)[0] + 1
                    
                    
                    axis_type = self.converter.get_axis_types()[0]
                    if self.fits_viewer.config_manager.config.get('decimal'):
                        self.interval_pix = float(interval_val)/self.wcs.wcs.cdelt[0]
                    else:
                        # --- Non-Decimal (HMS/DMS) Handling ---
                        interval_val_deg = None # Initialize degree value
                        interval_text = self.interval_edit.text() # *** Use original text for check ***

                        if axis_type[:2] == 'RA':
                            # Check the *original text* for hms indicators
                            if 'h' in interval_text or 'm' in interval_text or 's' in interval_text:
                                # If hms found, parse the *original text* as hourangle
                                try:
                                     interval_val_deg = Angle(interval_text, unit=u.hourangle).degree
                                except ValueError as e:
                                     raise ValueError(f"Invalid HMS format for interval '{interval_text}': {e}") from e
                            else:
                                # If no hms chars, parse the *original text* as degree
                                try:
                                     interval_val_deg = Angle(interval_text, unit=u.deg).degree
                                except ValueError as e:
                                     raise ValueError(f"Invalid degree format for interval '{interval_text}': {e}") from e
                        else: # Assume DEC or other degree axis
                            # Parse the *original text* as degree
                            try:
                                 interval_val_deg = Angle(interval_text, unit=u.deg).degree
                            except ValueError as e:
                                 raise ValueError(f"Invalid degree format for interval '{interval_text}': {e}") from e

                        # Calculate pixel interval using the correctly parsed degree value
                        if self.wcs.wcs.cdelt[0] == 0:
                             raise ValueError("CDELT for X-axis is zero, cannot calculate pixel interval.")
                        self.interval_pix = float(interval_val_deg) / self.wcs.wcs.cdelt[0]
                        
                self.from_edit.setText(str(round(float(from_val), 2)))
                self.to_edit.setText(str(round(float(to_val), 2)))
                self.interval_edit.setText(str(round(abs(float(self.interval_pix)), 2)))
                self.interval_edit.setCursorPosition(0)
            except ValueError:
                QMessageBox.warning(self, 'Error', 'Invalid range values')
                self._revert_worldch_radio(previous_state)
                return
    
    
    def _revert_worldch_radio(self, previous_state):
        """Revert the radio button selection based on the previous state."""
        # Block signals to avoid recursion.
        self.ch_radio.blockSignals(True)
        self.val_radio.blockSignals(True)
        if previous_state == 0:
            self.ch_radio.setChecked(True)
            self.val_radio.setChecked(False)
            self.worldch_num = 0
        else:
            self.val_radio.setChecked(True)
            self.ch_radio.setChecked(False)
            self.worldch_num = 1
        self.ch_radio.blockSignals(False)
        self.val_radio.blockSignals(False)

    def set_mode_num(self, num):
        """Set mode number based on selected mode radio button."""
        sender = self.sender()
        if not sender.isChecked():
            return
        self.mode_num = num
        if self.mode_num == 2:
            if self.worldch_num == 0:
                from_pix = np.ceil(float(self.from_edit.text()))
                to_pix = np.floor(float(self.to_edit.text()))
                self.from_edit.setText(str(from_pix))
                self.to_edit.setText(str(to_pix))
            elif self.worldch_num == 1:
                self.ch_radio.setChecked(True)
                from_pix = np.ceil(float(self.from_edit.text()))
                to_pix = np.floor(float(self.to_edit.text()))
                self.from_edit.setText(str(from_pix))
                self.to_edit.setText(str(to_pix))
                self.val_radio.setChecked(True)
            self.chlabel_radio_middle.setEnabled(False)
            self.chlabel_radio_range.setEnabled(False)
        else:
            self.chlabel_radio_middle.setEnabled(True)
            self.chlabel_radio_range.setEnabled(True)

    def update_tiles_x(self):
        """Update the number of rows (tiles_x) when user changes the combobox."""
        self.tiles_x = int(self.tiles_x_combo.currentText())

    def update_tiles_y(self):
        """Update the number of columns (tiles_y) when user changes the combobox."""
        self.tiles_y = int(self.tiles_y_combo.currentText())

    def set_dir_num(self, num):
        """Set direction (0 for L to R, 1 for T to B)."""
        self.dir_num = num

    def set_chlabel_num(self, num):
        self.chlabel_num = num

    def auto_set_interval(self):
        self.tiles_x = int(self.tiles_x_combo.currentText())
        self.tiles_y = int(self.tiles_y_combo.currentText())
        self.total_tiles = self.tiles_x * self.tiles_y
        """
        Automatically set the 'From', 'To', and 'Interval' fields.
        (Implement logic as needed.)
        """
        if self.ch_radio.isChecked():
            from_pix = self.from_edit.text()
            to_pix = self.to_edit.text()
            try:
                from_pix = float(from_pix)
                to_pix = float(to_pix)
            except ValueError:
                QMessageBox.warning(self, 'Error', 'Invalid range values')
                return
            interval_pix = round(abs(to_pix - from_pix)/self.total_tiles, 2)
            self.interval_edit.setText(str(interval_pix))
            self.interval_edit.setCursorPosition(0)
        
        if self.val_radio.isChecked():
            self.ch_radio.setChecked(True)
            from_pix = self.from_edit.text()
            to_pix = self.to_edit.text()
            #self.ch_radio.setChecked(False)
            try:
                from_pix = float(from_pix)
                to_pix = float(to_pix)
                interval_pix = round(abs(to_pix - from_pix)/self.total_tiles, 2)
                self.interval_edit.setText(str(interval_pix))
            except: pass
            self.val_radio.setChecked(True)
            self.interval_edit.setCursorPosition(0)
        
    def custom_nansum(self, arr, axis):
        return np.where(
            np.all(np.isnan(arr), axis=axis),
            np.nan,
            np.nansum(arr, axis=axis)
        )

    def nan_sum(self, arr1, arr2):
        both_nan_mask = np.isnan(arr1) & np.isnan(arr2)
        result = np.where(np.isnan(arr1), arr2, arr1)
        result = np.where(~np.isnan(arr1) & ~np.isnan(arr2), arr1 + arr2, result)
        result[both_nan_mask] = np.nan
        return result


    def mkchmap(self):
        if self.fits_viewer.data.ndim == 4:
            self.integ_data = self.fits_viewer.data[0]
        elif self.fits_viewer.data.ndim == 3:
            self.integ_data = self.fits_viewer.data
        self.tiles_x = int(self.tiles_x_combo.currentText())
        self.tiles_y = int(self.tiles_y_combo.currentText())
        self.total_tiles = self.tiles_x * self.tiles_y
        min_from_pix = 0.5
        max_to_pix = self.fits_viewer.header[f'NAXIS{3-self.plane_num}'] + 0.5
        if self.ch_radio.isChecked():
            from_pix = self.from_edit.text()
            to_pix = self.to_edit.text()
            interval_pix =  self.interval_edit.text()
            try:
                from_pix = float(from_pix)
                to_pix = float(to_pix)
                self.interval_pix = round(float(interval_pix), 3)
                if from_pix < min_from_pix or to_pix < min_from_pix or from_pix > max_to_pix or to_pix > max_to_pix:
                    raise ValueError("Invalid range values")
            except ValueError:
                QMessageBox.warning(self, 'Error', 'Invalid range values')
                if self.plane_num == 0:
                    self.plane_xz_radio.setChecked(True)
                    self.plane_xy_radio.setChecked(True)
                elif self.plane_num == 1:
                    self.plane_zy_radio.setChecked(True)
                    self.plane_xz_radio.setChecked(True)
                elif self.plane_num == 2:
                    self.plane_xy_radio.setChecked(True)
                    self.plane_zy_radio.setChecked(True)
                return
                
        elif self.val_radio.isChecked():
            self.ch_radio.setChecked(True)
            from_pix = self.from_edit.text()
            to_pix = self.to_edit.text()
            interval_pix =  self.interval_edit.text()
            try:
                from_pix = float(from_pix)
                to_pix = float(to_pix)
                self.interval_pix = round(float(interval_pix), 3)
                if from_pix < min_from_pix or to_pix < min_from_pix or from_pix > max_to_pix or to_pix > max_to_pix:
                    self.val_radio.setChecked(True)
                    QMessageBox.warning(self, 'Error', 'Invalid range values')
                    if self.plane_num == 0:
                        self.plane_xz_radio.setChecked(True)
                        self.plane_xy_radio.setChecked(True)
                    elif self.plane_num == 1:
                        self.plane_zy_radio.setChecked(True)
                        self.plane_xz_radio.setChecked(True)
                    elif self.plane_num == 2:
                        self.plane_xy_radio.setChecked(True)
                        self.plane_zy_radio.setChecked(True)
                    return
            except: 
                self.val_radio.setChecked(True)
                return
            self.val_radio.setChecked(True)
        
        print("Creating ChannelMaps...")
        #print(f"Plane num = {self.plane_num}, Mode = {self.mode_num}")
        #print(f"Tiles: {self.tiles_x} x {self.tiles_y}")
        #print(f"From: {from_pix}, To: {to_pix}, Interval: {self.interval_pix}")
        #print(f"Unit = {('ch' if self.worldch_num == 0 else 'world')}, Direction = {('L to R' if self.dir_num == 0 else 'T to B')}")
        
        
        ##integ mode ###
        def format_value(value):
            """
            Format a value by removing trailing zeros and an unnecessary decimal point
            from any numeric parts. If the input is a number, it is formatted directly.
            If the input is a string containing numeric parts (e.g. "1d2m3.40000s"),
            each numeric substring is formatted (e.g. "1d2m3.4s"). Other parts of the
            string are left unchanged.
            """
            def format_num(n):
                # Format the number using fixed-point notation and remove trailing zeros and dot.
                s = f"{n:f}"
                if '.' in s:
                    s = s.rstrip('0').rstrip('.')
                return s
        
            if isinstance(value, (int, float)):
                return format_num(value)
            elif isinstance(value, str):
                # Pattern matches numbers with or without a decimal point.
                pattern = r"(\d+\.\d+|\d+)"
                def repl(match):
                    num_str = match.group(0)
                    try:
                        num_val = float(num_str)
                        return format_num(num_val)
                    except ValueError:
                        return num_str
                # Substitute each numeric substring with its formatted version.
                return re.sub(pattern, repl, value)
            else:
                return str(value)
        
        def format_ranges(range_min, range_max):
            if self.plane_num == 0:
                if self.fits_viewer.data.ndim == 3:
                    from_val = self.converter.pix_to_world(0, 0, range_min-1)[2]
                    to_val = self.converter.pix_to_world(0, 0, range_max-1)[2]
                    center_val = self.converter.pix_to_world(0, 0, (range_max-1 + range_min - 1)/2.)[2]
                elif self.fits_viewer.data.ndim == 4:
                    from_val = self.converter.pix_to_world(0, 0, range_min-1, 0)[2]
                    to_val = self.converter.pix_to_world(0, 0, range_max-1, 0)[2]
                    center_val = self.converter.pix_to_world(0, 0, (range_max-1 + range_min - 1)/2., 0)[2]
                #interval_val = abs(self.interval_pix * self.wcs.wcs.cdelt[2])
                #axis_type = self.converter.get_axis_types()[2]
            elif self.plane_num == 1:
                if self.fits_viewer.data.ndim == 3:
                    from_val = self.converter.pix_to_world(0, range_min-1, 0)[1]
                    to_val = self.converter.pix_to_world(0, range_max-1, 0)[1]
                    center_val = self.converter.pix_to_world(0, (range_max-1 + range_min - 1)/2., 0)[1]
                elif self.fits_viewer.data.ndim == 4:
                    from_val = self.converter.pix_to_world(0, range_min-1, 0, 0)[1]
                    to_val = self.converter.pix_to_world(0, range_max-1, 0, 0)[1]
                    center_val = self.converter.pix_to_world(0, (range_max-1 + range_min - 1)/2., 0, 0)[1]
                #interval_val = abs(self.interval_pix * self.wcs.wcs.cdelt[1])
                #axis_type = self.converter.get_axis_types()[1]
            elif self.plane_num == 2:
                if self.fits_viewer.data.ndim == 3:
                    from_val = self.converter.pix_to_world(range_min-1, 0, 0)[0]
                    to_val = self.converter.pix_to_world(range_max-1, 0, 0)[0]
                    center_val = self.converter.pix_to_world((range_max-1 + range_min - 1)/2., 0, 0)[0]
                elif self.fits_viewer.data.ndim == 4:
                    from_val = self.converter.pix_to_world(range_min-1, 0, 0, 0)[0]
                    to_val = self.converter.pix_to_world(range_max-1, 0, 0, 0)[0]
                    center_val = self.converter.pix_to_world((range_max-1 + range_min - 1)/2., 0, 0, 0)[0]
                #interval_val = abs(self.interval_pix * self.wcs.wcs.cdelt[0])
                #axis_type = self.converter.get_axis_types()[0]
            
            return from_val, center_val, to_val
        
        def integration(min_pixel_float, max_pixel_float):
            axis = self.plane_num
            if min_pixel_float > max_pixel_float:
                min_pixel_float, max_pixel_float = max_pixel_float, min_pixel_float
            if max_pixel_float == self.integ_data.shape[axis] - 0.5: max_pixel_float -= 0.00001
            min_pixel = int(np.floor(min_pixel_float + 0.5))
            max_pixel = int(np.floor(max_pixel_float + 0.5))
            if min_pixel > max_pixel: max_pixel, min_pixel = min_pixel, max_pixel
            min_fraction = min_pixel - min_pixel_float - 0.5
            max_fraction = max_pixel_float - max_pixel + 0.5
            if min_pixel < 0 or max_pixel > self.integ_data.shape[axis] - 1: return
            
            """
            try:
                if min_pixel < 0 or max_pixel > self.integ_data.shape[axis] - 1:
                    raise IndexError(
                        f"Pixel range out of bounds: min_pixel={min_pixel}, max_pixel={max_pixel}, "
                        f"valid range is [0, {self.integ_data.shape[axis] - 1}]"
                    )
            except IndexError as e:
                QMessageBox.critical(
                    None, "Range Error", f"Invalid pixel range!\n{e}"
                )
                return
            """
            sliced_data = np.take(self.integ_data, indices=range(min_pixel, max_pixel), axis=axis)
            total_sum = self.custom_nansum(sliced_data, axis=axis)
    
            first_pixel_value = np.take(self.integ_data, indices=[min_pixel], axis=axis)
            total_sum = self.nan_sum(total_sum, np.squeeze(first_pixel_value, axis=axis) * min_fraction)
            last_pixel_value = np.take(self.integ_data, indices=[max_pixel], axis=axis)
            total_sum = self.nan_sum(total_sum, np.squeeze(last_pixel_value, axis=axis) * max_fraction)
            if self.mode_num == 0:
                integrated_data = total_sum * abs(self.wcs.wcs.cdelt[2 - axis])
            elif self.mode_num == 1:
                integrated_data = total_sum/self.interval_pix
            return integrated_data

        
        i = 0
        self.range_label = []
        self.ch_label = []
        self.ch_imdata = []
        if self.mode_num != 2: # Integ or Average
            if from_pix > to_pix:
                while from_pix - self.interval_pix * i > to_pix: 
                    range_min = from_pix - self.interval_pix * i
                    range_max = from_pix - self.interval_pix * (i+1)
                    i += 1
                    if range_max < 0: continue
                    
                    from_val, center_val, to_val =  format_ranges(range_min, range_max)
                    integ_result = integration(range_min-1, range_max-1)
                    if  integ_result is not None: 
                        self.range_label.append([format_value(from_val), format_value(center_val), format_value(to_val)]) #from (center) to
                        if self.plane_num == 2: self.ch_imdata.append(integ_result.T)
                        else: self.ch_imdata.append(integ_result)
                    
            elif from_pix <= to_pix:
                while from_pix + self.interval_pix * i < to_pix:
                    range_min = from_pix +  self.interval_pix * i
                    range_max = from_pix +  self.interval_pix * (i+1)
                    i += 1
                    if range_max - 1 > self.fits_viewer.header[f'NAXIS{3-self.plane_num}']: continue

                    from_val, center_val, to_val =  format_ranges(range_min, range_max)
                    integ_result = integration(range_min-1, range_max-1)
                    if  integ_result is not None: 
                        self.range_label.append([format_value(from_val), format_value(center_val), format_value(to_val)]) #from (center) to
                        if self.plane_num == 2: self.ch_imdata.append(integ_result.T)
                        else: self.ch_imdata.append(integ_result)
                    
        elif self.mode_num == 2:  # Slice
            total_ch_num = int(abs(to_pix - from_pix)/self.interval_pix + 1)
            if total_ch_num > self.fits_viewer.header[f'NAXIS{3-self.plane_num}']:
                total_ch_num = self.fits_viewer.header[f'NAXIS{3-self.plane_num}']
            if from_pix > to_pix:
                #while from_pix - self.interval_pix * i > to_pix:
                for i in range(total_ch_num):
                    slice_pix = from_pix - self.interval_pix * i - 1
                    idx = int(np.floor(slice_pix-0.5))
                    i += 1
                    slice_val =  format_ranges(slice_pix + 1, slice_pix + 1)
                    self.ch_label.append(format_value(slice_val[0]))
                    if self.plane_num == 0:
                        self.ch_imdata.append(self.integ_data[idx])
                    elif self.plane_num == 1:
                        self.ch_imdata.append(self.integ_data[:, idx, :])
                    elif self.plane_num == 2:
                        self.ch_imdata.append(self.integ_data[:, :, idx].T)
                    
            elif from_pix <= to_pix:
                for i in range(total_ch_num):
                    slice_pix = from_pix + self.interval_pix * i 
                    idx = int(np.floor(slice_pix-0.5))                    
                    if idx == self.fits_viewer.header[f'NAXIS{3-self.plane_num}']: idx -= 1
                    i += 1
                    slice_val =  format_ranges(slice_pix, slice_pix)
                    self.ch_label.append(format_value(slice_val[0]))
                    if self.plane_num == 0:
                        self.ch_imdata.append(self.integ_data[idx])
                    elif self.plane_num == 1:
                        self.ch_imdata.append(self.integ_data[:, idx, :])
                    elif self.plane_num == 2:
                        self.ch_imdata.append(self.integ_data[:, :, idx].T)
            self.range_label = self.ch_label
        
        
        self.channel_map_window = ChannelMapWindow(
            fits_viewer = self.fits_viewer,
            subwindows = self.subwindows,
            ch_imdata=self.ch_imdata,
            range_label=self.range_label,
            tiles_x=self.tiles_x,
            tiles_y=self.tiles_y,
            dir_num = self.dir_num,
            chlabel_num = self.chlabel_num,
            wcs=self.wcs,
            plane_num=self.plane_num,
            title="Channel Map"
        )
        self.channel_result_windows.append(self.channel_map_window)
        self.channel_map_window.show()


    def move_to_default_position(self):
        # Get MainWindow geometry
        mainwindow_geometry = self.fits_viewer.geometry()
        mainwindow_x = mainwindow_geometry.x()
        mainwindow_y = mainwindow_geometry.y()
        mainwindow_width = mainwindow_geometry.width()

        # Move ControlPanel to the right of MainWindow
        self.move(mainwindow_x + mainwindow_width, mainwindow_y - 28)
