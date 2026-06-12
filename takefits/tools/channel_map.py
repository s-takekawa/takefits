
from typing import Dict, List, Optional, Tuple, Iterable

from takefits.app_paths import app_config_path

import math
import os
import json
import uuid
import re

import astropy.units as u
import numpy as np
from astropy.coordinates import Angle
import matplotlib as mpl
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTimer, QSignalBlocker
from PySide6.QtGui import QGuiApplication, QKeySequence, QShortcut
from PySide6.QtWidgets import (QButtonGroup, QComboBox, QDialog, QGridLayout,
                             QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                             QMessageBox, QPushButton, QRadioButton, QWidget,
                             QVBoxLayout, QMainWindow)

from takefits.core.coordinate import CoordinateConverter, Format_pix_to_wcs
from takefits.core.marker import Marker, MarkerState, marker_from_state
from takefits.core.marker_manager import MarkerManager
from takefits.logic.add_hpbw import AddHPBW
from takefits.tools.color_scale import ColorSettingsPanel, ColorMode
from takefits.ui.navigation_toolbar import MyNavigationToolbar
from takefits.core.contour_manager import ContourManager, ContourItem
from takefits.core.colorbar_layout import compute_colorbar_geometry, orientation_for_placement
from takefits.core.usecases import compute_channel_map
from takefits.core.app_state import AppState, MarkerSpec, create_app_state
from takefits.core.action_session import ActionSession
from takefits.core.actions import ActionRegistry, register_default_actions
from takefits.tools.base_panel import record_action_preview, clear_action_preview_record
from takefits.tools.panel_helpers import _resolve_xz_subwindow, _resolve_z_view_limits


def _axis_length_for_channel_map(fits_viewer, axis_number):
    data = getattr(fits_viewer, "data", None)
    ndim = getattr(data, "ndim", 0)

    if ndim == 3:
        shape_map = {1: data.shape[2], 2: data.shape[1], 3: data.shape[0]}
        if axis_number in shape_map:
            return int(shape_map[axis_number])
    elif ndim >= 4:
        cube = data[0]
        shape_map = {1: cube.shape[2], 2: cube.shape[1], 3: cube.shape[0]}
        if axis_number in shape_map:
            return int(shape_map[axis_number])

    header = getattr(fits_viewer, "header", None)
    if header is not None:
        value = header.get(f"NAXIS{axis_number}")
        if value is not None:
            return int(value)

    return 0


def _axis_upper_pixel_edge(fits_viewer, axis_number):
    axis_len = _axis_length_for_channel_map(fits_viewer, axis_number)
    return axis_len + 0.5 if axis_len > 0 else 0.5


def _axis_last_pixel_center(fits_viewer, axis_number):
    axis_len = _axis_length_for_channel_map(fits_viewer, axis_number)
    return axis_len - 0.5 if axis_len > 0 else -0.5


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
    _VIEW_BACK_KEY_TOKENS = {"ctrl+z", "cmd+z", "meta+z", "super+z"}
    _VIEW_FORWARD_KEY_TOKENS = {
        "ctrl+y",
        "cmd+y",
        "meta+y",
        "super+y",
        "ctrl+shift+z",
        "cmd+shift+z",
        "meta+shift+z",
        "super+shift+z",
    }
    _VIEW_BACK_SEQUENCES = (QKeySequence.StandardKey.Undo, "Meta+Z", "Ctrl+Z")
    _VIEW_FORWARD_SEQUENCES = (
        QKeySequence.StandardKey.Redo,
        "Meta+Shift+Z",
        "Ctrl+Shift+Z",
        "Meta+Y",
        "Ctrl+Y",
    )
    def __init__(
        self,
        fits_viewer,
        subwindows,
        ch_imdata,
        range_label,
        tiles_x,
        tiles_y,
        wcs,
        dir_num,
        chlabel_num,
        plane_num=0,
        title="Channel Map",
        parent=None,
        intensity_unit: Optional[str] = None,
    ):
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
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
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
        self._view_history = []
        self._view_history_index = -1
        self._suspend_view_history_recording = True
        self.bunit = str(intensity_unit or getattr(self.fits_viewer, "bunit", "") or "").strip()

        self.marker_manager = MarkerManager(self)
        self.marker_mode_enabled = False
        self.marker_panel = None
        self._setup_marker_action_bridge()
        self._marker_axes: Dict[str, object] = {}
        self._axes_marker_plane: Dict[object, str] = {}
        self._marker_formats: Dict[str, Format_pix_to_wcs] = {}
        self._marker_planes: List[str] = []
        self._marker_plane_base: Dict[str, str] = {}
        self.marker_link_all = False
        self._colorbar_auto_layout_override = None
        self._colorbar_layout_from_draw_event = False
        self._colorbar_sync_redraw_in_progress = False
        self._applying_colorbar_auto_layout = False
        self._colorbar_auto_anchor_sig = None
        
        self.original_xlim = self.fits_viewer.ax.get_xlim()
        self.original_ylim = self.fits_viewer.ax.get_ylim()
        self.original_zlim = _resolve_z_view_limits(self.fits_viewer, self.subwindows)
        self.converter = CoordinateConverter(self.wcs, self.fits_viewer.config_manager.config)

        if self.wcs.naxis == 3:
            self.znpix = self.fits_viewer.data.shape[0]-1
            self.ynpix = self.fits_viewer.data.shape[1]-1
            self.xnpix = self.fits_viewer.data.shape[2]-1
        elif self.wcs.naxis == 4:
            self.znpix = self.fits_viewer.data[0].shape[0]-1
            self.ynpix = self.fits_viewer.data[0].shape[1]-1
            self.xnpix = self.fits_viewer.data[0].shape[2]-1
        
        # Get axis labels from ViewerState
        fv = self.fits_viewer
        if hasattr(fv, 'state') and fv.state.ax_coord is not None:
            self.xlabel = fv.state.ax_coord[0].get_axislabel()
            self.ylabel = fv.state.ax_coord[1].get_axislabel()
        else:
            self.xlabel = ""
            self.ylabel = ""
        # Get zlabel from xz plane
        coord = fv.get_coordinator() if hasattr(fv, 'get_coordinator') else None
        xz_state = coord.get_state('xz') if coord else None
        if xz_state is not None and xz_state.ax_coord is not None:
            self.zlabel = xz_state.ax_coord[1].get_axislabel()
        else:
            self.zlabel = ""
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
        self.auto_precision_digits = bool(self.config.get('auto_precision_digits', True))
        self.number_decimals = self.config['number_decimals']
        self.coord_wrap = self.config['coord_wrap']
        
        
        self.color_pattern = (
            ColorSettingsPanel.settings[ColorMode.MAIN]['color_pattern'] or 
            self.fits_viewer.displaymap.config.get('colorscale')
        )
        if ColorSettingsPanel.settings[ColorMode.CHANNEL]['color_pattern']:
            self.color_pattern = ColorSettingsPanel.settings[ColorMode.CHANNEL]['color_pattern']
        self._color_panel_hint = dict(ColorSettingsPanel.settings.get(ColorMode.CHANNEL, {}) or {})

        self.initialize_ranges()
        self.initUI()
        self._initialize_marker_history_seed()
        self._setup_undo_redo_shortcuts()
        self._view_history = []
        self._view_history_index = -1
        self._suspend_view_history_recording = False
        self._record_local_view_history(reason="init", force=True)
        self._refresh_view_navigation_actions()
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

    @staticmethod
    def _normalize_label_text(text) -> str:
        return " ".join(str(text).split()).strip()

    @staticmethod
    def _plane_coord_keys(plane: Optional[str]) -> tuple[str, str]:
        key = str(plane or "").lower()
        if key == "xz":
            return ("x", "z")
        if key == "zy":
            return ("z", "y")
        return ("x", "y")

    @staticmethod
    def _extract_unit_from_axis_label(axis_label: str) -> str:
        text = str(axis_label or "")
        left = text.rfind("[")
        right = text.rfind("]")
        if left == -1 or right == -1 or right <= left + 1:
            return ""
        return text[left + 1:right].strip().replace(" ", "")

    def _axis_unit_from_axis_label(self, axis_key: str) -> str:
        axis = str(axis_key or "").lower()
        xkey, ykey = self._plane_coord_keys(self.plane)
        if axis == xkey:
            label_text = str(getattr(self, "xlabel", "") or "")
        elif axis == ykey:
            if ykey == "z":
                label_text = str(getattr(self, "zlabel", "") or "")
            else:
                label_text = str(getattr(self, "ylabel", "") or "")
        else:
            label_text = ""
        return self._extract_unit_from_axis_label(label_text)

    def _axis_unit_label_for_value(self, axis_key: str, value_text: str) -> str:
        unit_getter = getattr(self.fits_viewer, "_axis_unit_label_for_value", None)
        if callable(unit_getter):
            try:
                return self._normalize_label_text(unit_getter(self.plane, axis_key, value_text))
            except Exception:
                pass
        return self._axis_unit_from_axis_label(axis_key)

    def _format_value_with_axis_unit(self, axis_key: str, value_text: str) -> str:
        text = self._normalize_label_text(value_text)
        unit = self._axis_unit_label_for_value(axis_key, text)
        if unit:
            return f"{text} {unit}"
        return text

    def _cursor_coordinate_frame_label(self) -> str:
        getter = getattr(self.fits_viewer, "_cursor_coordinate_frame_label", None)
        if callable(getter):
            try:
                return self._normalize_label_text(getter(self.plane))
            except Exception:
                pass
        return "WCS"

    def _format_intensity_with_unit(self, value) -> str:
        try:
            text = self.fits_viewer._format_significant_digits(float(value), 4)
        except Exception:
            text = self._normalize_label_text(value)
        unit = self._normalize_label_text(getattr(self, "bunit", "") or "")
        if unit:
            return f"{text} {unit}"
        return text

    def _tile_data_for_axes(self, axes):
        try:
            idx = int(self.axes.index(axes))
        except Exception:
            return None
        global_idx = (int(self.current_page) * int(self.num_per_page)) + idx
        if global_idx < 0 or global_idx >= len(self.ch_imdata):
            return None
        try:
            return np.asarray(self.ch_imdata[global_idx])
        except Exception:
            return None

    def _sample_intensity(self, axes, x: float, y: float):
        tile = self._tile_data_for_axes(axes)
        if tile is None:
            return None
        try:
            xp = int(round(float(x)))
            yp = int(round(float(y)))
        except Exception:
            return None
        if yp < 0 or xp < 0 or yp >= tile.shape[0] or xp >= tile.shape[1]:
            return None
        try:
            return tile[yp, xp]
        except Exception:
            return None

    def _formatter_for_axes(self, axes, x: float, y: float, plane_id: Optional[str] = None) -> str:
        converter = None
        if plane_id and plane_id in self._marker_formats:
            converter = self._marker_formats.get(plane_id)
        if converter is None:
            converter = next(iter(self._marker_formats.values()), None)
        if converter is None:
            return ""
        xstr, ystr = converter.convert(self.plane, x, y)
        xkey, ykey = self._plane_coord_keys(self.plane)
        xstr = self._format_value_with_axis_unit(xkey, xstr)
        ystr = self._format_value_with_axis_unit(ykey, ystr)
        if self.plane == 'xy':
            line1 = f"x={xstr}, y={ystr}"
        elif self.plane == 'xz':
            line1 = f"x={xstr}, z={ystr}"
        else:
            line1 = f"z={xstr}, y={ystr}"
        frame = self._cursor_coordinate_frame_label()
        value = self._sample_intensity(axes, x, y)
        if value is not None:
            line2 = f"[{self._normalize_label_text(self._format_intensity_with_unit(value))}] {frame}"
        else:
            line2 = frame
        return f"{line1}\n{line2}"

    def formatter(self, x, y):
        active_ax = getattr(self.toolbar, "ax", None) if getattr(self, "toolbar", None) is not None else None
        if active_ax is None and getattr(self, "axes", None):
            active_ax = self.axes[0]
        plane_id = self.marker_plane_for_axes(active_ax) if active_ax is not None else None
        return self._formatter_for_axes(active_ax, x, y, plane_id=plane_id)

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
                converter = Format_pix_to_wcs(
                    self.wcs,
                    self.projection_slices,
                    ax,
                    self.plane,
                    self.decimal,
                    self.number_decimals,
                    self.coord_wrap,
                    fits_viewer=self.fits_viewer,
                    auto_precision_digits=self.auto_precision_digits,
                )
                self._marker_formats[plane_id] = converter
                self._marker_axes[plane_id] = ax
                self._axes_marker_plane[ax] = plane_id
                self._marker_plane_base[plane_id] = self.plane
                self._marker_planes.append(plane_id)
                ax.format_coord = (lambda x, y, _ax=ax, _plane_id=plane_id: self._formatter_for_axes(_ax, x, y, plane_id=_plane_id))
                self.axes.append(ax)
        if self.dir_num == 1:
            self.axes = list(np.array(self.axes, dtype=object).reshape(self.tiles_x, self.tiles_y).T.flat)
        self.ax = self.axes[0] if len(self.axes) > 0 else None

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
        self.canvas.mpl_connect('draw_event', self._on_canvas_draw_for_layout)
        
        layout.addWidget(self.canvas,  2,   0,  1,  16)
        layout.addWidget(self.toolbar, 3, 0,  1,  16)

        subplot_params_path = app_config_path('subplot_params.yaml')
        if os.path.exists(subplot_params_path):
            dialog = self.toolbar.configure_subplots()
            dialog._import_values(show_message=False)
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

    def refresh_coordinate_display(self):
        toolbar = getattr(self, "toolbar", None)
        if toolbar is not None:
            try:
                axis = getattr(toolbar, "ax", None)
                if axis is None:
                    visible_axes = [ax for ax in list(getattr(self, "axes", []) or []) if ax is not None and ax.get_visible()]
                    axis = visible_axes[0] if visible_axes else (self.axes[0] if getattr(self, "axes", None) else None)
                if axis is not None:
                    xmid = sum(axis.get_xlim()) / 2.0
                    ymid = sum(axis.get_ylim()) / 2.0
                    plane_id = self.marker_plane_for_axes(axis)
                    toolbar.set_message(self._formatter_for_axes(axis, xmid, ymid, plane_id=plane_id))
            except Exception:
                pass

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

    def _marker_plane_tile_index(self, plane: Optional[str]) -> Optional[int]:
        plane_name = str(plane or "").strip()
        prefix = f"{self.marker_plane_prefix}_"
        if not plane_name.startswith(prefix):
            return None
        try:
            return int(plane_name[len(prefix):])
        except Exception:
            return None

    def _marker_plane_global_index(self, plane: Optional[str]) -> Optional[int]:
        tile_index = self._marker_plane_tile_index(plane)
        if tile_index is None or tile_index < 0:
            return None
        global_index = (int(self.current_page) * int(self.num_per_page)) + tile_index
        if global_index < 0 or global_index >= len(list(self.range_label or [])):
            return None
        return global_index

    def _marker_plane_center_pixel(self, plane: Optional[str]) -> Optional[float]:
        global_index = self._marker_plane_global_index(plane)
        if global_index is None:
            return None
        try:
            label = list(self.range_label or [])[global_index]
            return float(label[1])
        except Exception:
            return None

    def _shared_marker_pixel_context(self) -> Tuple[float, float, float, float]:
        source = getattr(self, "fits_viewer", None)

        def _read(method_name: str, default: float = 0.0) -> float:
            getter = getattr(source, method_name, None)
            if callable(getter):
                try:
                    value = getter()
                    return float(value) if value is not None else float(default)
                except Exception:
                    return float(default)
            return float(default)

        return (
            _read("_get_shared_xpix"),
            _read("_get_shared_ypix"),
            _read("_get_shared_zpix"),
            _read("_get_shared_spix"),
        )

    def _marker_pixel_vector(
        self,
        plane: Optional[str],
        *,
        x_pix: Optional[float] = None,
        y_pix: Optional[float] = None,
    ) -> Optional[List[float]]:
        wcs = getattr(self, "wcs", None)
        if wcs is None:
            return None
        try:
            naxis = int(getattr(wcs, "naxis", 0) or 0)
        except Exception:
            naxis = 0
        if naxis <= 0:
            return None

        shared_x, shared_y, shared_z, shared_s = self._shared_marker_pixel_context()
        base = self.marker_plane_base(plane or self.default_marker_plane() or self.plane)
        fixed_center = self._marker_plane_center_pixel(plane)
        if fixed_center is None:
            if base == "xy":
                fixed_center = shared_z
            elif base == "xz":
                fixed_center = shared_y
            elif base == "zy":
                fixed_center = shared_x
            else:
                fixed_center = 0.0

        vector: List[float] = []
        for axis in range(naxis):
            if axis == 0:
                if base == "zy":
                    vector.append(float(fixed_center))
                else:
                    vector.append(float(shared_x if x_pix is None else x_pix))
            elif axis == 1:
                if base == "xz":
                    vector.append(float(fixed_center))
                else:
                    vector.append(float(shared_y if y_pix is None else y_pix))
            elif axis == 2:
                if base == "xy":
                    vector.append(float(fixed_center))
                elif base == "xz":
                    vector.append(float(shared_z if y_pix is None else y_pix))
                elif base == "zy":
                    vector.append(float(shared_z if x_pix is None else x_pix))
                else:
                    vector.append(float(shared_z))
            elif axis == 3:
                vector.append(float(shared_s))
            else:
                vector.append(0.0)
        return vector

    def marker_world_defaults(self, plane: Optional[str] = None) -> Optional[List[float]]:
        vector = self._marker_pixel_vector(plane)
        if vector is None:
            return None
        try:
            world = list(np.asarray(self.wcs.wcs_pix2world([vector], 0)[0], dtype=float).tolist())
        except Exception:
            return None
        while len(world) < 4:
            world.append(0.0)
        return world[:4]

    def marker_pix_to_world(
        self,
        plane: Optional[str],
        pixel: Tuple[float, ...],
    ) -> Optional[Tuple[float, float]]:
        if pixel is None or len(pixel) < 2:
            return None
        vector = self._marker_pixel_vector(
            plane,
            x_pix=float(pixel[0]),
            y_pix=float(pixel[1]),
        )
        if vector is None:
            return None
        try:
            world = np.asarray(self.wcs.wcs_pix2world([vector], 0)[0], dtype=float)
        except Exception:
            return None
        base = self.marker_plane_base(plane or self.default_marker_plane() or self.plane)
        if base == "xy" and len(world) >= 2:
            return (float(world[0]), float(world[1]))
        if base == "xz" and len(world) >= 3:
            return (float(world[0]), float(world[2]))
        if base == "zy" and len(world) >= 3:
            return (float(world[2]), float(world[1]))
        if len(world) >= 2:
            return (float(world[0]), float(world[1]))
        return None

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
            from takefits.tools.marker_panel import MarkerPanel
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
        if self._end_colorbar_drag(event):
            return
        marker_manager = getattr(self, "marker_manager", None)
        if (
            marker_manager is not None
            and getattr(self, "marker_mode_enabled", False)
            and getattr(event, "button", None) == 1
        ):
            marker_manager.handle_release(event)
            self.canvas.draw_idle()

    def on_motion(self, event):
        if self._drag_colorbar(event):
            return
        self._update_magnifier_from_event(event)
        marker_manager = getattr(self, "marker_manager", None)
        if marker_manager is not None and getattr(self, "marker_mode_enabled", False):
            if marker_manager.is_dragging():
                marker_manager.handle_motion(event)
            else:
                marker_manager.handle_hover(event)
            return

    def _update_magnifier_from_event(self, event):
        panel = getattr(getattr(self, "fits_viewer", None), "magnifier_panel", None)
        if panel is None:
            return False
        try:
            if not panel.isVisible():
                return False
        except Exception:
            return False
        axes = getattr(event, "inaxes", None)
        if axes not in list(getattr(self, "axes", []) or []):
            return False
        if getattr(event, "xdata", None) is None or getattr(event, "ydata", None) is None:
            return False
        updater = getattr(panel, "update_from_cursor", None)
        if not callable(updater):
            return False
        try:
            return bool(
                updater(
                    self,
                    getattr(self, "plane", "xy"),
                    event.xdata,
                    event.ydata,
                    source_axes=axes,
                )
            )
        except Exception:
            return False

    def on_key_press(self, event):
        key = str(getattr(event, "key", "") or "").lower()
        if key in self._VIEW_BACK_KEY_TOKENS:
            self.view_back()
            return
        if key in self._VIEW_FORWARD_KEY_TOKENS:
            self.view_forward()
            return
        if key == "f":
            panel = getattr(getattr(self, "fits_viewer", None), "magnifier_panel", None)
            toggler = getattr(panel, "toggle_lock", None)
            if callable(toggler):
                try:
                    if panel.isVisible():
                        toggler()
                        return
                except Exception:
                    pass
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

    # Marker ActionSession bridge ---------------------------------------
    def _setup_marker_action_bridge(self):
        source_data = getattr(self.fits_viewer, "data", None)
        if source_data is None:
            source_data = np.asarray(self.ch_imdata) if self.ch_imdata else np.zeros((1, 1), dtype=float)
        self.app_state = create_app_state(
            data=np.asarray(source_data),
            header=getattr(self.fits_viewer, "header", None),
            wcs=self.wcs,
            filepath=getattr(self.fits_viewer, "filename", None),
        )
        registry = ActionRegistry()
        register_default_actions(registry)
        self.action_session = ActionSession(registry=registry, state=self.app_state)
        self._suspend_action_recording = False
        self._last_markers_fingerprint = None
        self._markers_commit_timer = QTimer(self)
        self._markers_commit_timer.setSingleShot(True)
        self._markers_commit_timer.timeout.connect(self._commit_markers_to_session)
        try:
            self.marker_manager.markers_changed.connect(lambda *_: self._schedule_markers_commit())
        except Exception:
            pass

    def _initialize_marker_history_seed(self):
        try:
            self.app_state.markers = [MarkerSpec.from_dict(entry) for entry in self._marker_specs_snapshot()]
        except Exception:
            pass
        try:
            self.action_session.set_initial_state_seed()
        except Exception:
            pass
        self._last_markers_fingerprint = json.dumps(
            self._marker_specs_snapshot(), sort_keys=True, separators=(",", ":")
        )
        self._notify_main_window_undo_redo_state()

    def _setup_undo_redo_shortcuts(self):
        self._undo_shortcut = QShortcut(QKeySequence("Alt+Left"), self)
        self._redo_shortcut = QShortcut(QKeySequence("Alt+Right"), self)
        self._undo_shortcut.activated.connect(self.undo_last_action)
        self._redo_shortcut.activated.connect(self.redo_last_action)
        self._view_back_shortcuts = self._create_view_shortcuts(
            self._VIEW_BACK_SEQUENCES,
            self.view_back,
        )
        self._view_forward_shortcuts = self._create_view_shortcuts(
            self._VIEW_FORWARD_SEQUENCES,
            self.view_forward,
        )

    def _create_view_shortcuts(self, sequences, callback):
        shortcuts = []
        seen = set()
        for sequence in sequences:
            try:
                key_sequence = QKeySequence(sequence)
            except Exception:
                continue
            portable = key_sequence.toString(QKeySequence.SequenceFormat.PortableText)
            if not portable or portable in seen:
                continue
            seen.add(portable)
            shortcut = QShortcut(key_sequence, self)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(callback)
            shortcuts.append(shortcut)
        return shortcuts

    def _notify_main_window_undo_redo_state(self):
        main_window = getattr(self, "fits_viewer", None)
        refresh = getattr(main_window, "_refresh_undo_redo_actions", None)
        if callable(refresh):
            try:
                refresh()
            except Exception:
                pass

    def _flush_pending_annotation_commits(self):
        if getattr(self, "_markers_commit_timer", None) is not None and self._markers_commit_timer.isActive():
            self._markers_commit_timer.stop()
            self._commit_markers_to_session()

    def _schedule_markers_commit(self, delay_ms: int = 200) -> None:
        if getattr(self, "_suspend_action_recording", False):
            return
        if hasattr(self, "_markers_commit_timer") and self._markers_commit_timer is not None:
            self._markers_commit_timer.start(int(delay_ms))

    def _marker_specs_snapshot(self):
        marker_manager = getattr(self, "marker_manager", None)
        if marker_manager is None:
            return []
        try:
            marker_manager.refresh_world_coordinates()
        except Exception:
            pass
        specs = []
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
                    if frame_name:
                        marker_payload["world_frame"] = frame_name
                    specs.append(marker_payload)
                except Exception:
                    continue
        specs.sort(key=lambda s: (str(s.get("plane") or ""), str(s.get("id") or "")))
        return specs

    def _commit_markers_to_session(self) -> None:
        if getattr(self, "_suspend_action_recording", False):
            return
        specs = self._marker_specs_snapshot()
        fingerprint = json.dumps(specs, sort_keys=True, separators=(",", ":"))
        if fingerprint == self._last_markers_fingerprint:
            return
        self._last_markers_fingerprint = fingerprint
        try:
            self.action_session.execute("set_markers", markers=specs)
            self._notify_main_window_undo_redo_state()
        except Exception:
            try:
                self.app_state.markers = [MarkerSpec.from_dict(entry) for entry in specs]
            except Exception:
                pass

    def _build_marker_payload_from_state(self):
        state = getattr(self.action_session, "state", None)
        marker_specs = list(getattr(state, "markers", []) or []) if state is not None else []
        markers = []
        world_frame = ""
        for marker in marker_specs:
            marker_dict = marker.to_dict() if isinstance(marker, MarkerSpec) else dict(marker or {})
            candidate_frame = str(marker_dict.get("world_frame") or "").strip()
            if candidate_frame and not world_frame:
                world_frame = candidate_frame
            markers.append(marker_dict)
        payload = {
            "format": "takefits.marker",
            "version": 1,
            "plane": "all",
            "markers": markers,
        }
        if world_frame:
            payload["world_frame"] = world_frame
        return payload

    def _apply_action_session_state_to_viewer(self):
        state = getattr(self.action_session, "state", None)
        if state is None:
            return
        marker_manager = getattr(self, "marker_manager", None)
        if marker_manager is None:
            return
        self._suspend_action_recording = True
        try:
            self.app_state = state
            planes_to_redraw = set()
            marker_layers = list((getattr(marker_manager, "_layers", None) or {}).keys())
            for plane in marker_layers:
                plane_name = str(plane or "").lower()
                if plane_name:
                    planes_to_redraw.add(plane_name)
                try:
                    marker_manager.clear_plane(plane)
                except Exception:
                    continue
            marker_payload = self._build_marker_payload_from_state()
            marker_entries = list(marker_payload.get("markers") or [])
            for marker_entry in marker_entries:
                if not isinstance(marker_entry, dict):
                    continue
                plane_name = str(marker_entry.get("plane") or "").lower()
                if plane_name:
                    planes_to_redraw.add(plane_name)
            if marker_entries:
                try:
                    imported_plane = marker_manager.import_from_dict(marker_payload)
                    imported_name = str(imported_plane or "").lower()
                    if imported_name:
                        planes_to_redraw.add(imported_name)
                except Exception:
                    pass
            if planes_to_redraw:
                try:
                    marker_manager.redraw_planes(planes_to_redraw)
                except Exception:
                    for plane_name in planes_to_redraw:
                        try:
                            marker_manager.redraw_plane(plane_name)
                        except Exception:
                            continue
            if getattr(self, "canvas", None) is not None:
                self.canvas.draw_idle()
            self._last_markers_fingerprint = json.dumps(
                self._marker_specs_snapshot(), sort_keys=True, separators=(",", ":")
            )
        finally:
            self._suspend_action_recording = False
            self._notify_main_window_undo_redo_state()

    def undo_last_action(self):
        session = getattr(self, "action_session", None)
        if session is None:
            return
        self._flush_pending_annotation_commits()
        if not session.can_undo():
            self._notify_main_window_undo_redo_state()
            return
        try:
            session.undo()
            self._apply_action_session_state_to_viewer()
        except Exception as exc:
            QMessageBox.warning(self, "Undo", f"Failed to undo action: {exc}")
        finally:
            self._notify_main_window_undo_redo_state()

    def redo_last_action(self):
        session = getattr(self, "action_session", None)
        if session is None:
            return
        self._flush_pending_annotation_commits()
        if not session.can_redo():
            self._notify_main_window_undo_redo_state()
            return
        try:
            session.redo()
            self._apply_action_session_state_to_viewer()
        except Exception as exc:
            QMessageBox.warning(self, "Redo", f"Failed to redo action: {exc}")
        finally:
            self._notify_main_window_undo_redo_state()



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
                                        fontsize=self.config.get('ch_label_size', 10),
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
                try:
                    coord_wrap = float(self.config.get('coord_wrap', 180)) * u.deg
                except Exception:
                    coord_wrap = 180 * u.deg
                ax.coords['glon'].set_coord_type(coord_wrap=coord_wrap, coord_type='longitude')
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
        old_colorbar = getattr(self, "colorbar", None)
        old_cax = getattr(self, "cax", None)
        if old_colorbar is not None:
            try:
                old_colorbar.remove()
            except Exception:
                pass
        elif old_cax is not None:
            try:
                old_cax.remove()
            except Exception:
                pass
        self.cax = self.fig.add_axes([
            self.config.get('cbar_pos_x', 0.9),
            self.config.get('cbar_pos_y', 0.11),
            self.config.get('cbar_width', 0.04),
            self.config.get('cbar_height', 0.77)
        ])
        self.cax.set_gid('colorbar')
        self.cax.set_zorder(300)
        # Create the colorbar on the first image using the configured orientation.
        self.colorbar = self.fig.colorbar(self.im_list[0],
                                        cax=self.cax,
                                        orientation=self.config.get('colorbar_orientation', 'vertical'))
        self.colorbar.ax.set_zorder(300)
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



        ColorSettingsPanel.apply_colorbar_settings(cax = self.cax, colorbar = self.colorbar, config=self.config)
        self._set_colorbar_zorder()
        
        
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
        self.apply_preferences(redraw=False)
        self.canvas.draw()
        self._schedule_colorbar_auto_layout_if_anchor_changed(force=False)

    def _set_colorbar_zorder(self, zorder: float = 300.0):
        cax = getattr(self, "cax", None)
        if cax is not None:
            try:
                cax.set_zorder(float(zorder))
            except Exception:
                pass
        colorbar = getattr(self, "colorbar", None)
        cbar_ax = getattr(colorbar, "ax", None) if colorbar is not None else None
        if cbar_ax is not None:
            try:
                cbar_ax.set_zorder(float(zorder))
            except Exception:
                pass

    def _visible_axes_bounds(self):
        axes = [ax for ax in list(getattr(self, "axes", []) or []) if getattr(ax, "get_visible", None) and ax.get_visible()]
        if not axes:
            return None
        bounds = []
        for ax in axes:
            try:
                bounds.append(tuple(float(v) for v in ax.get_position().bounds))
            except Exception:
                continue
        if not bounds:
            return None
        left = min(b[0] for b in bounds)
        bottom = min(b[1] for b in bounds)
        right = max(b[0] + b[2] for b in bounds)
        top = max(b[1] + b[3] for b in bounds)
        if right <= left or top <= bottom:
            return None
        return (left, bottom, right - left, top - bottom)

    def _is_colorbar_auto_layout_enabled(self) -> bool:
        root = getattr(self, "fits_viewer", None)
        resolver = getattr(root, "_get_main_viewer", None)
        if callable(resolver):
            try:
                root = resolver()
            except Exception:
                pass
        if bool(getattr(root, "_workspace_colorbar_restore_in_progress", False)):
            return False
        override = getattr(self, "_colorbar_auto_layout_override", None)
        if override is not None:
            return bool(override)
        config = getattr(self, "config", None)
        if not isinstance(config, dict):
            return False
        return bool(config.get("colorbar_auto_layout", False))

    def _request_canvas_redraw(self, *, immediate: bool = False) -> bool:
        canvas = getattr(self, "canvas", None)
        if canvas is None:
            return False

        root = getattr(self, "fits_viewer", None)
        resolver = getattr(root, "_get_main_viewer", None)
        if callable(resolver):
            try:
                root = resolver()
            except Exception:
                pass

        dispatcher = getattr(root, "_request_canvas_redraw", None)
        if callable(dispatcher):
            try:
                return bool(dispatcher(canvas, immediate=immediate))
            except Exception:
                pass

        draw_name = "draw" if immediate else "draw_idle"
        draw = getattr(canvas, draw_name, None)
        if callable(draw):
            try:
                draw()
                return True
            except Exception:
                pass
        if not immediate:
            draw = getattr(canvas, "draw", None)
            if callable(draw):
                try:
                    draw()
                    return True
                except Exception:
                    pass
        return False

    def _set_colorbar_orientation_config(self, orientation: str):
        orientation = str(orientation or "").strip().lower()
        if orientation not in ("vertical", "horizontal"):
            return
        config = getattr(self, "config", None)
        if isinstance(config, dict):
            config["colorbar_orientation"] = orientation
        config_mgr = getattr(getattr(self, "fits_viewer", None), "config_manager", None)
        shared = getattr(config_mgr, "config", None) if config_mgr is not None else None
        if isinstance(shared, dict):
            shared["colorbar_orientation"] = orientation

    def _current_colorbar_orientation(self) -> str:
        orientation = str(getattr(getattr(self, "colorbar", None), "orientation", "") or "").lower()
        if orientation in ("vertical", "horizontal"):
            return orientation
        config = getattr(self, "config", None)
        if isinstance(config, dict):
            cfg_orientation = str(config.get("colorbar_orientation", "") or "").lower()
            if cfg_orientation in ("vertical", "horizontal"):
                return cfg_orientation
        return "vertical"

    def _rebuild_colorbar(self, orientation: str):
        config = getattr(self, "config", None)
        if not isinstance(config, dict):
            return False
        image = next((im for im in list(getattr(self, "im_list", []) or []) if im is not None), None)
        if image is None:
            return False

        old_cax = getattr(self, "cax", None)
        bounds = None
        if old_cax is not None:
            try:
                bounds = [float(v) for v in old_cax.get_position().bounds]
            except Exception:
                bounds = None
        if bounds is None:
            bounds = [
                float(config.get("cbar_pos_x", 0.9)),
                float(config.get("cbar_pos_y", 0.11)),
                float(config.get("cbar_width", 0.04)),
                float(config.get("cbar_height", 0.77)),
            ]

        colorbar = getattr(self, "colorbar", None)
        if colorbar is not None:
            try:
                colorbar.remove()
            except Exception:
                pass
        elif old_cax is not None:
            try:
                old_cax.remove()
            except Exception:
                pass

        self.cax = self.fig.add_axes(bounds)
        self.cax.set_gid("colorbar")
        self.cax.set_zorder(300)
        self.colorbar = self.fig.colorbar(image, cax=self.cax, orientation=orientation)
        self.colorbar.ax.set_zorder(300)
        ColorSettingsPanel.apply_colorbar_settings(cax=self.cax, colorbar=self.colorbar, config=config)
        self._set_colorbar_zorder()
        return True

    def _apply_colorbar_auto_layout(self, force: bool = False, *, redraw: bool = True) -> bool:
        if not force and not self._is_colorbar_auto_layout_enabled():
            return False
        if bool(getattr(self, "_applying_colorbar_auto_layout", False)):
            return False

        config = getattr(self, "config", None)
        cax = getattr(self, "cax", None)
        fig = getattr(self, "fig", None)
        if not isinstance(config, dict) or cax is None or fig is None:
            return False

        anchor_bounds = self._visible_axes_bounds()
        if anchor_bounds is None:
            return False

        try:
            cbar_bounds = tuple(float(v) for v in cax.get_position().bounds)
            fig_w_px = float(getattr(getattr(fig, "bbox", None), "width", 0.0) or 0.0)
            fig_h_px = float(getattr(getattr(fig, "bbox", None), "height", 0.0) or 0.0)
        except Exception:
            return False
        if fig_w_px <= 0.0:
            fig_w_px = 1.0
        if fig_h_px <= 0.0:
            fig_h_px = 1.0

        placement = str(config.get("colorbar_placement", "right") or "right")
        align = str(config.get("colorbar_align", "center") or "center")
        gap_px = config.get("colorbar_gap_px", 24.0)
        gap_x_px = config.get("colorbar_gap_x_px", gap_px)
        gap_y_px = config.get("colorbar_gap_y_px", gap_px)
        thickness_px = config.get("colorbar_thickness_px", 24.0)
        length_mode = str(config.get("colorbar_length_mode", "ratio") or "ratio")
        length_value = config.get("colorbar_length_value", 1.0)
        current_orientation = self._current_colorbar_orientation()
        target_orientation = orientation_for_placement(
            placement,
            fallback=config.get("colorbar_orientation", current_orientation),
        )
        orientation_changed = target_orientation != current_orientation

        if orientation_changed:
            self._set_colorbar_orientation_config(target_orientation)
            if not self._rebuild_colorbar(target_orientation):
                return False
            try:
                cbar_bounds = tuple(float(v) for v in self.cax.get_position().bounds)
            except Exception:
                return False

        target = compute_colorbar_geometry(
            anchor_bounds,
            fig_w_px,
            fig_h_px,
            placement=placement,
            align=align,
            gap_px=gap_px,
            gap_x_px=gap_x_px,
            gap_y_px=gap_y_px,
            thickness_px=thickness_px,
            length_mode=length_mode,
            length_value=length_value,
        )[:4]
        eps = 1e-6
        changed = force or orientation_changed or any(abs(cur - tgt) > eps for cur, tgt in zip(cbar_bounds, target))
        if not changed:
            return False

        self._applying_colorbar_auto_layout = True
        try:
            self.cax.set_position(list(target))
            self.cax.set_gid("colorbar")
            self._set_colorbar_zorder()
            config["cbar_pos_x"] = float(target[0])
            config["cbar_pos_y"] = float(target[1])
            config["cbar_width"] = float(target[2])
            config["cbar_height"] = float(target[3])
            if getattr(self, "canvas", None) is not None:
                if redraw:
                    self._request_canvas_redraw()
        finally:
            self._applying_colorbar_auto_layout = False
        return True

    def fit_colorbar_now(self) -> bool:
        return bool(self._apply_colorbar_auto_layout(force=True))

    def _schedule_colorbar_auto_layout(self, force: bool = False):
        if not force and bool(getattr(self, "_suspend_colorbar_auto_layout", False)):
            return
        if not self._is_colorbar_auto_layout_enabled():
            return
        if bool(getattr(self, "_colorbar_sync_redraw_in_progress", False)):
            return
        from_draw_event = bool(getattr(self, "_colorbar_layout_from_draw_event", False))
        changed = bool(self._apply_colorbar_auto_layout(force=force, redraw=not from_draw_event))
        if changed and from_draw_event:
            self._colorbar_sync_redraw_in_progress = True
            try:
                self._request_canvas_redraw(immediate=True)
            finally:
                self._colorbar_sync_redraw_in_progress = False

    def _colorbar_layout_anchor_signature(self):
        anchor = self._visible_axes_bounds()
        fig = getattr(self, "fig", None)
        if anchor is None or fig is None:
            return None
        try:
            fig_w = round(float(getattr(getattr(fig, "bbox", None), "width", 0.0) or 0.0), 3)
            fig_h = round(float(getattr(getattr(fig, "bbox", None), "height", 0.0) or 0.0), 3)
        except Exception:
            return None
        return tuple(round(float(v), 8) for v in anchor) + (fig_w, fig_h)

    def _schedule_colorbar_auto_layout_if_anchor_changed(self, force: bool = False):
        if not force and bool(getattr(self, "_suspend_colorbar_auto_layout", False)):
            return
        if force:
            if not self._is_colorbar_auto_layout_enabled():
                return
            self._colorbar_auto_anchor_sig = None
            self._schedule_colorbar_auto_layout(force=True)
            return
        if not self._is_colorbar_auto_layout_enabled():
            return
        sig = self._colorbar_layout_anchor_signature()
        if sig is None:
            return
        if sig == getattr(self, "_colorbar_auto_anchor_sig", None):
            return
        self._colorbar_auto_anchor_sig = sig
        self._schedule_colorbar_auto_layout(force=False)

    def _on_canvas_draw_for_layout(self, event):
        if getattr(event, "canvas", None) is not self.canvas:
            return
        self._colorbar_layout_from_draw_event = True
        try:
            self._schedule_colorbar_auto_layout_if_anchor_changed(force=False)
        finally:
            self._colorbar_layout_from_draw_event = False

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
        self.apply_preferences(redraw=False)
        self.canvas.draw_idle()
        self.prev_button.setEnabled(self.current_page > 0)
        self.next_button.setEnabled(end_index < total_images)
        


    def apply_channel_label_settings(self, redraw: bool = True):
        """Apply shared channel-label preference settings to this window."""
        config = self.config if isinstance(self.config, dict) else {}
        position = (
            config.get('pos_chlabel_x', 0.98),
            config.get('pos_chlabel_y', 0.02),
        )
        fontsize = config.get('ch_label_size', 10)
        fontfamily = config.get('ch_label_font', 'Arial')
        color = config.get('ch_label_color', 'grey')

        for label in list(getattr(self, 'ch_labels', []) or []):
            if label is None:
                continue
            label.set_position(position)
            label.set_fontsize(fontsize)
            label.set_fontfamily(fontfamily)
            label.set_color(color)

        if redraw and hasattr(self, 'canvas'):
            self.canvas.draw_idle()

    def _axis_role_for_coord_index(self, coord_index: int) -> str:
        if self.plane_num == 0:
            return "x" if coord_index == 0 else "y"
        if self.plane_num == 1:
            return "x" if coord_index == 0 else "y"
        if self.plane_num == 2:
            return "x" if coord_index == 2 else "y"
        return "x" if coord_index == 0 else "y"

    def _apply_ticklabel_style(self, coord, axis_role: str, config: dict):
        if coord is None:
            return
        if axis_role == "x":
            rotation = config.get('tick_xlabelrotation')
            pad = config.get('tick_pad_x')
            position = config.get('xticklabel_position')
        else:
            rotation = config.get('tick_ylabelrotation')
            pad = config.get('tick_pad_y')
            position = config.get('yticklabel_position')
        coord.set_ticklabel(
            rotation=rotation,
            pad=pad,
            size=config.get('tick_labelsize'),
            color=config.get('tick_labelcolor'),
            fontfamily=config.get('tick_font'),
            exclude_overlapping=True,
        )
        coord.set_ticklabel_position(position)
        coord.set_axislabel_position(position)
        coord.set_ticks_position(config.get('default_ticks_position'))

    def _apply_coordinate_format_preferences(self, ax, config: dict):
        try:
            if 'glon' in ax.coords:
                try:
                    coord_wrap = float(config.get('coord_wrap', 180)) * u.deg
                except Exception:
                    coord_wrap = 180 * u.deg
                ax.coords['glon'].set_coord_type(coord_wrap=coord_wrap, coord_type='longitude')

            axis_units = []
            axis_types = []
            for i in range(self.wcs.naxis):
                axis_units.append(ax.coords[i].get_format_unit())
                physical_type = self.wcs.world_axis_physical_types[i]
                axis_types.append(None if physical_type is None else physical_type.split('.')[-1])
            axis_format_decimal = np.isin(axis_types, ['lon', 'lat'])
            if config.get('decimal') is False:
                axis_format_decimal = [False for _ in axis_format_decimal]
            for i in np.where(axis_format_decimal)[0]:
                ax.coords[i].set_format_unit(axis_units[i], decimal=axis_format_decimal[i])
        except Exception:
            pass

    def _apply_axis_preferences(self, config: dict):
        if getattr(self, "fig", None) is not None:
            self.fig.set_facecolor(config.get('fig_background_color'))

        displayed = [bool(value) for value in getattr(self, "projection_slices", [])]
        for ax in list(getattr(self, "axes", []) or []):
            if ax is None:
                continue
            ax.set_facecolor(config.get('ax_background_color'))
            self._apply_coordinate_format_preferences(ax, config)
            ax.tick_params(
                which='major',
                direction=config.get('tick_direction'),
                length=config.get('tick_length'),
                color=config.get('tick_color'),
                width=config.get('tick_width'),
                labelsize=config.get('tick_labelsize'),
                labelcolor=config.get('tick_labelcolor'),
            )
            ax.tick_params(which='minor', length=config.get('mtick_length'))
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_zorder(5)
                spine.set_linewidth(config.get('tick_width'))
                spine.set_color(config.get('tick_color'))

            try:
                coords = list(ax.coords)
            except Exception:
                continue
            for coord_index, coord in enumerate(coords):
                if coord_index < len(displayed) and not displayed[coord_index]:
                    try:
                        coord.set_ticks_visible(False)
                        coord.set_ticklabel_visible(False)
                    except Exception:
                        pass
                    continue
                role = self._axis_role_for_coord_index(coord_index)
                self._apply_ticklabel_style(coord, role, config)
                try:
                    if coord_index == 0:
                        coord.set_minor_frequency(config.get('x_mtick_freq', 5))
                    elif coord_index == 1:
                        coord.set_minor_frequency(config.get('y_mtick_freq', 5))
                    elif coord_index == 2:
                        coord.set_minor_frequency(config.get('z_mtick_freq', 5))
                    coord.display_minor_ticks(True)
                except Exception:
                    pass

    def _apply_colorbar_preferences(self, config: dict):
        cax = getattr(self, "cax", None)
        colorbar = getattr(self, "colorbar", None)
        if cax is None or colorbar is None:
            return
        bounds = [
            config.get('cbar_pos_x', 0.9),
            config.get('cbar_pos_y', 0.11),
            config.get('cbar_width', 0.04),
            config.get('cbar_height', 0.77),
        ]
        try:
            cax.set_position(bounds)
        except Exception:
            pass
        orientation = str(config.get('colorbar_orientation', 'vertical') or '').lower()
        if orientation not in ("vertical", "horizontal"):
            orientation = "vertical"
        if orientation != self._current_colorbar_orientation():
            self._rebuild_colorbar(orientation)
        else:
            ColorSettingsPanel.apply_colorbar_settings(cax=cax, colorbar=colorbar, config=config)
            self._set_colorbar_zorder()
        if self._is_colorbar_auto_layout_enabled():
            self._schedule_colorbar_auto_layout_if_anchor_changed(force=True)

    def apply_preferences(self, redraw: bool = True):
        """Apply the shared Preferences config to an open channel-map window."""
        config = getattr(getattr(self, "fits_viewer", None), "config_manager", None)
        config = getattr(config, "config", None) if config is not None else self.config
        if not isinstance(config, dict):
            return
        self.config = config
        self.decimal = config.get('decimal', True)
        self.auto_precision_digits = bool(config.get('auto_precision_digits', True))
        self.number_decimals = config.get('number_decimals', 6)
        self.coord_wrap = config.get('coord_wrap', 180)

        for formatter in list((getattr(self, "_marker_formats", {}) or {}).values()):
            formatter.decimal = self.decimal
            formatter.auto_precision_digits = self.auto_precision_digits
            formatter.number_decimals = self.number_decimals
            formatter.coord_wrap = self.coord_wrap

        for image in list(getattr(self, "im_list", []) or []):
            try:
                image.cmap.set_bad(config.get('bad_color'))
            except Exception:
                pass

        self._apply_axis_preferences(config)
        self.update_axis_labels()
        self.apply_channel_label_settings(redraw=False)
        self._apply_colorbar_preferences(config)

        if redraw and getattr(self, "canvas", None) is not None:
            self.canvas.draw_idle()


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

    def _seed_color_panel_settings_from_current_image(self):
        settings = {
            "min_val": None,
            "max_val": None,
            "log_scale": False,
            "gamma_value": 1.0,
            "invert": False,
            "color_pattern": None,
        }
        raw = dict(ColorSettingsPanel.settings.get(ColorMode.CHANNEL, {}) or {})
        if isinstance(raw, dict):
            settings.update(raw)
        hint = getattr(self, "_color_panel_hint", None)
        if isinstance(hint, dict):
            settings.update(hint)

        image = None
        images = list(getattr(self, "im_list", []) or [])
        if images:
            image = images[0]
        if image is not None:
            try:
                clim = image.get_clim()
                if clim is not None:
                    settings["min_val"] = float(clim[0])
                    settings["max_val"] = float(clim[1])
            except Exception:
                pass
            try:
                settings["log_scale"] = isinstance(getattr(image, "norm", None), mpl.colors.LogNorm)
            except Exception:
                pass
            try:
                cmap_name = str(getattr(image.get_cmap(), "name", "") or "")
                if cmap_name.endswith("_r"):
                    base = cmap_name[:-2]
                    if base and base != "from_list":
                        settings["color_pattern"] = base
                        settings["invert"] = True
                elif cmap_name and cmap_name != "from_list":
                    settings["color_pattern"] = cmap_name
                    settings["invert"] = False
            except Exception:
                pass
        try:
            settings["gamma_value"] = float(settings.get("gamma_value", 1.0) or 1.0)
        except Exception:
            settings["gamma_value"] = 1.0

        self._color_panel_hint = dict(settings)
        ColorSettingsPanel.settings[ColorMode.CHANNEL] = dict(settings)
        return settings

    def open_color_settings(self):
        """
        Open the Colorscale settings panel.
        (For now, simply display an information message.)
        """
        self._seed_color_panel_settings_from_current_image()

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
        try:
            self._color_panel_hint = dict(ColorSettingsPanel.settings.get(ColorMode.CHANNEL, {}) or {})
            pattern = str(self._color_panel_hint.get("color_pattern") or "")
            if pattern:
                if bool(self._color_panel_hint.get("invert")) and not pattern.endswith("_r"):
                    pattern = f"{pattern}_r"
                self.color_pattern = pattern
        except Exception:
            pass
        self.color_settings_panel = None

    def _view_history_state(self):
        history = list(getattr(self, "_view_history", []) or [])
        index = int(getattr(self, "_view_history_index", -1))
        total = len(history)
        can_back = total > 0 and index > 0
        can_forward = total > 0 and 0 <= index < (total - 1)
        return can_back, can_forward

    def _refresh_view_navigation_actions(self):
        toolbar = getattr(self, "toolbar", None)
        setter = getattr(toolbar, "set_external_history_state", None)
        can_back, can_forward = self._view_history_state()
        if callable(setter):
            try:
                setter(can_back, can_forward)
            except Exception:
                pass

        root_viewer = getattr(self, "fits_viewer", None)
        refresh = getattr(root_viewer, "_refresh_view_navigation_actions", None)
        if callable(refresh):
            try:
                refresh()
            except Exception:
                pass

    def _extract_live_color_settings(self):
        panel = getattr(self, "color_settings_panel", None)
        if panel is None:
            return None
        settings = {}
        current = getattr(panel, "current_settings", None)
        if isinstance(current, dict):
            settings.update(current)
        try:
            combo = getattr(panel, "colorscale_combo", None)
            if combo is not None:
                text = str(combo.currentText() or "").strip()
                if text:
                    settings["color_pattern"] = text
        except Exception:
            pass
        try:
            invert = getattr(panel, "invert_checkbox", None)
            if invert is not None:
                settings["invert"] = bool(invert.isChecked())
        except Exception:
            pass
        try:
            logbox = getattr(panel, "log_checkbox", None)
            if logbox is not None:
                settings["log_scale"] = bool(logbox.isChecked())
        except Exception:
            pass
        try:
            gamma_spin = getattr(panel, "gamma_spinbox", None)
            if gamma_spin is not None:
                settings["gamma_value"] = float(gamma_spin.value())
        except Exception:
            pass
        return settings if settings else None

    def _normalize_view_history_color_settings(self, settings=None, fallback=None):
        normalized = {
            "min_val": None,
            "max_val": None,
            "log_scale": False,
            "gamma_value": 1.0,
            "invert": False,
            "color_pattern": None,
        }
        for source in (fallback, settings):
            if not isinstance(source, dict):
                continue
            if "min_val" in source:
                normalized["min_val"] = source.get("min_val")
            if "max_val" in source:
                normalized["max_val"] = source.get("max_val")
            if "log_scale" in source:
                normalized["log_scale"] = bool(source.get("log_scale", False))
            if "gamma_value" in source:
                try:
                    normalized["gamma_value"] = float(source.get("gamma_value", 1.0) or 1.0)
                except Exception:
                    pass
            if "invert" in source:
                normalized["invert"] = bool(source.get("invert", False))
            if "color_pattern" in source:
                value = source.get("color_pattern")
                normalized["color_pattern"] = str(value) if value is not None else None

        for key in ("min_val", "max_val"):
            value = normalized.get(key)
            if value is None:
                continue
            try:
                number = float(value)
                normalized[key] = number if np.isfinite(number) else None
            except Exception:
                normalized[key] = None

        try:
            gamma_value = float(normalized.get("gamma_value", 1.0) or 1.0)
        except Exception:
            gamma_value = 1.0
        if (not np.isfinite(gamma_value)) or gamma_value <= 0:
            gamma_value = 1.0
        normalized["gamma_value"] = gamma_value

        pattern = str(normalized.get("color_pattern") or "").strip()
        if pattern.endswith("_r"):
            pattern = pattern[:-2]
            normalized["invert"] = True
        if not pattern:
            display_pattern = str(getattr(self, "color_pattern", "") or "").strip()
            if display_pattern.endswith("_r"):
                display_pattern = display_pattern[:-2]
                normalized["invert"] = True
            if display_pattern:
                pattern = display_pattern
        normalized["color_pattern"] = pattern or None
        return normalized

    def _capture_view_history_limits(self):
        axis = None
        current = getattr(self, "ax", None)
        if current is not None:
            axis = current
        if axis is None:
            visible_axes = [ax for ax in list(getattr(self, "axes", []) or []) if ax is not None and ax.get_visible()]
            if visible_axes:
                axis = visible_axes[0]
            else:
                axes = list(getattr(self, "axes", []) or [])
                axis = axes[0] if axes else None
        if axis is None:
            return None
        try:
            xlim = axis.get_xlim()
            ylim = axis.get_ylim()
            return {
                "xlim": [float(xlim[0]), float(xlim[1])],
                "ylim": [float(ylim[0]), float(ylim[1])],
            }
        except Exception:
            return None

    def _capture_view_history_color(self):
        fallback = dict(ColorSettingsPanel.settings.get(ColorMode.CHANNEL, {}) or {})
        hint = getattr(self, "_color_panel_hint", None)
        if isinstance(hint, dict):
            fallback.update(hint)

        settings = self._normalize_view_history_color_settings(
            self._extract_live_color_settings(),
            fallback=fallback,
        )
        images = [im for im in list(getattr(self, "im_list", []) or []) if im is not None]
        image = images[0] if images else None
        if image is not None:
            try:
                clim = image.get_clim()
                if clim is not None:
                    settings["min_val"] = float(clim[0])
                    settings["max_val"] = float(clim[1])
            except Exception:
                pass
            try:
                settings["log_scale"] = isinstance(getattr(image, "norm", None), mpl.colors.LogNorm)
            except Exception:
                pass
            try:
                cmap_name = str(getattr(image.get_cmap(), "name", "") or "")
                if cmap_name.endswith("_r"):
                    base = cmap_name[:-2]
                    if base and base != "from_list":
                        settings["color_pattern"] = base
                        settings["invert"] = True
                elif cmap_name and cmap_name != "from_list":
                    settings["color_pattern"] = cmap_name
                    settings["invert"] = False
            except Exception:
                pass
        return self._normalize_view_history_color_settings(settings, fallback=fallback)

    def _view_history_signature(self, limits, color):
        if not isinstance(limits, dict):
            return tuple()
        xlim = limits.get("xlim")
        ylim = limits.get("ylim")
        if not (
            isinstance(xlim, (list, tuple))
            and len(xlim) == 2
            and isinstance(ylim, (list, tuple))
            and len(ylim) == 2
        ):
            return tuple()
        try:
            x0, x1 = float(xlim[0]), float(xlim[1])
            y0, y1 = float(ylim[0]), float(ylim[1])
        except Exception:
            return tuple()

        normalized = self._normalize_view_history_color_settings(color)

        def _round_or_none(value, digits):
            try:
                number = float(value)
            except Exception:
                return None
            if not np.isfinite(number):
                return None
            return round(number, digits)

        return (
            round(x0, 9),
            round(x1, 9),
            round(y0, 9),
            round(y1, 9),
            str(normalized.get("color_pattern") or ""),
            bool(normalized.get("invert", False)),
            bool(normalized.get("log_scale", False)),
            _round_or_none(normalized.get("gamma_value"), 6),
            _round_or_none(normalized.get("min_val"), 9),
            _round_or_none(normalized.get("max_val"), 9),
        )

    def _record_local_view_history(self, reason: str = "", *, force: bool = False):
        if bool(getattr(self, "_suspend_view_history_recording", False)):
            return False
        limits = self._capture_view_history_limits()
        if not isinstance(limits, dict):
            return False
        color = self._capture_view_history_color()
        signature = self._view_history_signature(limits, color)
        if not signature:
            return False

        history = list(getattr(self, "_view_history", []) or [])
        index = int(getattr(self, "_view_history_index", -1))
        if not force and history and 0 <= index < len(history):
            if history[index].get("signature") == signature:
                self._refresh_view_navigation_actions()
                return False

        if 0 <= index < (len(history) - 1):
            history = history[: index + 1]

        history.append(
            {
                "limits": limits,
                "color": color,
                "signature": signature,
                "reason": str(reason or ""),
            }
        )

        max_entries = 200
        if len(history) > max_entries:
            history = history[-max_entries:]

        self._view_history = history
        self._view_history_index = len(history) - 1
        self._refresh_view_navigation_actions()
        return True

    def _build_view_history_gamma_cmap(self, pattern: str, gamma: float):
        cmap = mpl.colormaps.get_cmap(pattern)
        gamma_value = max(1e-6, float(gamma))
        rgba = cmap(np.linspace(0.0, 1.0, cmap.N) ** gamma_value)
        built = mpl.colors.ListedColormap(rgba, name=str(getattr(cmap, "name", pattern) or pattern))
        bad_color = self.config.get("bad_color") if isinstance(self.config, dict) else None
        if not bad_color:
            try:
                bad_color = self.fits_viewer.displaymap.bad_color
            except Exception:
                bad_color = "black"
        try:
            built.set_bad(color=bad_color)
        except Exception:
            pass
        return built

    def _sync_open_color_panel_settings(self, settings):
        panel = getattr(self, "color_settings_panel", None)
        if panel is None or not isinstance(settings, dict):
            return

        merged = dict(getattr(panel, "current_settings", {}) or {})
        merged.update(settings)
        merged = self._normalize_view_history_color_settings(merged)
        ColorSettingsPanel.settings[ColorMode.CHANNEL] = dict(merged)
        panel.current_settings = ColorSettingsPanel.settings[ColorMode.CHANNEL]

        blockers = []
        for widget in (
            getattr(panel, "colorscale_combo", None),
            getattr(panel, "invert_checkbox", None),
            getattr(panel, "log_checkbox", None),
            getattr(panel, "gamma_spinbox", None),
            getattr(panel, "intensity_min", None),
            getattr(panel, "intensity_max", None),
        ):
            if widget is None:
                continue
            try:
                blockers.append(QSignalBlocker(widget))
            except Exception:
                continue

        min_val = merged.get("min_val")
        max_val = merged.get("max_val")
        pattern = str(merged.get("color_pattern") or "").strip()
        invert = bool(merged.get("invert", False))
        log_scale = bool(merged.get("log_scale", False))
        gamma = float(merged.get("gamma_value", 1.0) or 1.0)

        try:
            panel.intensity_min.setText("" if min_val is None else f"{float(min_val):.3g}")
            panel.intensity_max.setText("" if max_val is None else f"{float(max_val):.3g}")
            panel.gamma_spinbox.setValue(gamma)
            panel.invert_checkbox.setChecked(invert)
            panel.log_checkbox.setChecked(log_scale)
            if pattern:
                panel.colorscale_combo.setCurrentText(pattern)
            panel.color_pattern = pattern or getattr(panel, "color_pattern", None)
            panel.auto_button.setEnabled(not log_scale)
            panel.min_max_button.setEnabled(not log_scale)
            if (
                min_val is not None
                and max_val is not None
                and getattr(panel, "min_line", None) is not None
                and getattr(panel, "max_line", None) is not None
            ):
                panel.update_histogram_lines(float(min_val), float(max_val))
            panel.canvas.draw_idle()
        except Exception:
            pass

    def _apply_view_history_color(self, color_state):
        if not isinstance(color_state, dict):
            return False
        settings = self._normalize_view_history_color_settings(color_state)
        pattern = str(settings.get("color_pattern") or "").strip()
        if not pattern:
            return False

        display_pattern = pattern
        if bool(settings.get("invert")) and not display_pattern.endswith("_r"):
            display_pattern = f"{display_pattern}_r"
        self.color_pattern = display_pattern
        self._color_panel_hint = dict(settings)
        ColorSettingsPanel.settings[ColorMode.CHANNEL] = dict(settings)

        try:
            cmap = self._build_view_history_gamma_cmap(display_pattern, settings.get("gamma_value", 1.0))
        except Exception:
            return False

        images = [im for im in list(getattr(self, "im_list", []) or []) if im is not None]
        if not images:
            return False

        min_val = settings.get("min_val")
        max_val = settings.get("max_val")
        use_limits = False
        try:
            min_float = float(min_val)
            max_float = float(max_val)
            use_limits = np.isfinite(min_float) and np.isfinite(max_float)
        except Exception:
            min_float = max_float = None
            use_limits = False

        try:
            for image in images:
                image.set_cmap(cmap)
                if use_limits:
                    if bool(settings.get("log_scale", False)) and min_float > 0 and max_float > 0:
                        image.set_norm(mpl.colors.LogNorm(vmin=min_float, vmax=max_float))
                    else:
                        image.set_norm(mpl.colors.Normalize(vmin=min_float, vmax=max_float))
                    image.set_clim(min_float, max_float)
        except Exception:
            return False

        colorbar = getattr(self, "colorbar", None)
        if colorbar is not None:
            try:
                colorbar.update_normal(images[0])
            except Exception:
                pass
            try:
                ColorSettingsPanel.apply_colorbar_settings(
                    cax=self.cax,
                    colorbar=colorbar,
                    config=self.config,
                )
            except Exception:
                pass

        self._sync_open_color_panel_settings(settings)
        self._background = None
        if getattr(self, "canvas", None) is not None:
            try:
                self.canvas.draw_idle()
            except Exception:
                pass
        return True

    def _apply_local_view_history_entry(self, entry):
        if not isinstance(entry, dict):
            return False
        limits = entry.get("limits")
        color = entry.get("color")
        has_limits = isinstance(limits, dict)
        has_color = isinstance(color, dict)
        if not has_limits and not has_color:
            return False

        applied_limits = False
        applied_color = False
        self._suspend_view_history_recording = True
        try:
            if has_limits:
                xlim = limits.get("xlim")
                ylim = limits.get("ylim")
                if (
                    isinstance(xlim, (list, tuple))
                    and len(xlim) == 2
                    and isinstance(ylim, (list, tuple))
                    and len(ylim) == 2
                ):
                    try:
                        x0, x1 = float(xlim[0]), float(xlim[1])
                        y0, y1 = float(ylim[0]), float(ylim[1])
                        self.update_ranges(self.plane, (x0, x1), (y0, y1))
                        applied_limits = True
                    except Exception:
                        applied_limits = False

            if has_color:
                applied_color = bool(self._apply_view_history_color(color))

            self._background = None
            if getattr(self, "canvas", None) is not None:
                try:
                    self.canvas.draw_idle()
                except Exception:
                    pass
        finally:
            self._suspend_view_history_recording = False
        return applied_limits or applied_color

    def view_back(self):
        can_back, _ = self._view_history_state()
        if not can_back:
            self._refresh_view_navigation_actions()
            return False
        history = list(getattr(self, "_view_history", []) or [])
        index = int(getattr(self, "_view_history_index", -1))
        if not history or index <= 0:
            self._refresh_view_navigation_actions()
            return False
        previous_index = index
        index -= 1
        self._view_history_index = index
        applied = False
        try:
            applied = self._apply_local_view_history_entry(history[index])
        except Exception:
            applied = False
        if not applied:
            self._view_history_index = previous_index
        self._refresh_view_navigation_actions()
        return bool(applied)

    def view_forward(self):
        _, can_forward = self._view_history_state()
        if not can_forward:
            self._refresh_view_navigation_actions()
            return False
        history = list(getattr(self, "_view_history", []) or [])
        index = int(getattr(self, "_view_history_index", -1))
        if not history or index < 0 or index >= (len(history) - 1):
            self._refresh_view_navigation_actions()
            return False
        previous_index = index
        index += 1
        self._view_history_index = index
        applied = False
        try:
            applied = self._apply_local_view_history_entry(history[index])
        except Exception:
            applied = False
        if not applied:
            self._view_history_index = previous_index
        self._refresh_view_navigation_actions()
        return bool(applied)

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
        self._flush_pending_annotation_commits()
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
        
        if self in self.fits_viewer.channel_map_windows:
            self.fits_viewer.channel_map_windows.remove(self)

        #print("Result window closed")
        if self.toolbar._subplot_dialog is not None:
            self.toolbar._subplot_dialog.close()
            self.toolbar._subplot_dialog = None
        super().closeEvent(event)

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
            if hasattr(self.fits_viewer, 'x_min_input'):
                x_min = self.fits_viewer.x_min_input.text()
                x_max = self.fits_viewer.x_max_input.text()
                self.x_min_ch_input.setText(x_min)
                self.x_max_ch_input.setText(x_max)
                self.set_x_range()
            
        if self.plane == 'xy' or self.plane == 'zy':
            if hasattr(self.fits_viewer, 'y_min_input'):
                y_min = self.fits_viewer.y_min_input.text()
                y_max = self.fits_viewer.y_max_input.text()
                self.y_min_ch_input.setText(y_min)
                self.y_max_ch_input.setText(y_max)
                self.set_y_range()
            
        if self.plane == 'xz' or self.plane == 'zy':
            z_min, z_max = None, None
            if self.plane == 'xz' and len(self.subwindows) > 0 and self.subwindows[0]:
                if hasattr(self.subwindows[0], 'z_min_input'):
                    z_min = self.subwindows[0].z_min_input.text()
                    z_max = self.subwindows[0].z_max_input.text()
            elif self.plane == 'zy' and len(self.subwindows) > 1 and self.subwindows[1]:
                if hasattr(self.subwindows[1], 'z_min_input'):
                    z_min = self.subwindows[1].z_min_input.text()
                    z_max = self.subwindows[1].z_max_input.text()

            if z_min is not None and z_max is not None:
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
            self.set_x_range(record_history=False)
            
        if self.plane == 'xy' or self.plane == 'zy':
            self.y_min_ch_input.setText(str(self.ymin_val_full))
            self.y_max_ch_input.setText(str(self.ymax_val_full))
            self.set_y_range(record_history=False)
            
        if self.plane == 'xz' or self.plane == 'zy':
            self.z_min_ch_input.setText(str(self.zmin_val_full))
            self.z_max_ch_input.setText(str(self.zmax_val_full))
            self.set_z_range(record_history=False)
        if not bool(getattr(self, "_suspend_view_history_recording", False)):
            self._record_local_view_history(reason="range:full")
            
    def set_x_range(self, record_history: bool = True):
        try:
            x_min = str(self.x_min_ch_input.text() or "").strip()
            x_max = str(self.x_max_ch_input.text() or "").strip()
            if not x_min or not x_max:
                raise ValueError

            def _fallback(text, default):
                value = str(text or "").strip()
                if value:
                    return value
                return str(default if default is not None else "").strip()

            y_min_ref = _fallback(self.y_min_ch_input.text() if hasattr(self, "y_min_ch_input") else "", self.ymin_val)
            y_max_ref = _fallback(self.y_max_ch_input.text() if hasattr(self, "y_max_ch_input") else "", self.ymax_val)
            z_min_ref = _fallback(self.z_min_ch_input.text() if hasattr(self, "z_min_ch_input") else "", self.zmin_val)
            z_max_ref = _fallback(self.z_max_ch_input.text() if hasattr(self, "z_max_ch_input") else "", self.zmax_val)
            if self.fits_viewer.data.ndim == 3:
                xp_min = float(self.converter.world_to_pix(x_min, y_min_ref, z_min_ref)[0])
                xp_max = float(self.converter.world_to_pix(x_max, y_max_ref, z_max_ref)[0])
            elif self.fits_viewer.data.ndim == 4:
                xp_min = float(self.converter.world_to_pix(x_min, y_min_ref, z_min_ref, 0)[0])
                xp_max = float(self.converter.world_to_pix(x_max, y_max_ref, z_max_ref, 0)[0])
                
            if xp_min > xp_max: xp_min, xp_max = xp_max, xp_min

            for ax in self.axes: ax.set_xlim(xp_min, xp_max)
            self.xmin_val = x_min
            self.xmax_val = x_max
            self._background = None
            self.canvas.draw_idle()
            if record_history and not bool(getattr(self, "_suspend_view_history_recording", False)):
                self._record_local_view_history(reason="range:x")
        except ValueError:
            if not getattr(self, "_suppress_range_warning", False):
                QMessageBox.warning(self, 'Invalid Input', 'Please enter valid numeric values for the X range.')
            
    def set_y_range(self, record_history: bool = True):
        try:
            y_min = str(self.y_min_ch_input.text() or "").strip()
            y_max = str(self.y_max_ch_input.text() or "").strip()
            if not y_min or not y_max:
                raise ValueError

            def _fallback(text, default):
                value = str(text or "").strip()
                if value:
                    return value
                return str(default if default is not None else "").strip()

            x_min_ref = _fallback(self.x_min_ch_input.text() if hasattr(self, "x_min_ch_input") else "", self.xmin_val)
            x_max_ref = _fallback(self.x_max_ch_input.text() if hasattr(self, "x_max_ch_input") else "", self.xmax_val)
            z_min_ref = _fallback(self.z_min_ch_input.text() if hasattr(self, "z_min_ch_input") else "", self.zmin_val)
            z_max_ref = _fallback(self.z_max_ch_input.text() if hasattr(self, "z_max_ch_input") else "", self.zmax_val)
            if self.fits_viewer.data.ndim == 3:
                yp_min = float(self.converter.world_to_pix(x_min_ref, y_min, z_min_ref)[1])
                yp_max = float(self.converter.world_to_pix(x_max_ref, y_max, z_max_ref)[1])
            elif self.fits_viewer.data.ndim == 4:
                yp_min = float(self.converter.world_to_pix(x_min_ref, y_min, z_min_ref, 0)[1])
                yp_max = float(self.converter.world_to_pix(x_max_ref, y_max, z_max_ref, 0)[1])
                
            if yp_min > yp_max: yp_min, yp_max = yp_max, yp_min

            for ax in self.axes: ax.set_ylim(yp_min, yp_max)
            self.ymin_val = y_min
            self.ymax_val = y_max
            self._background = None
            self.canvas.draw_idle()
            if record_history and not bool(getattr(self, "_suspend_view_history_recording", False)):
                self._record_local_view_history(reason="range:y")
        except ValueError:
            if not getattr(self, "_suppress_range_warning", False):
                QMessageBox.warning(self, 'Invalid Input', 'Please enter valid numeric values for the Y range.')


    def set_z_range(self, record_history: bool = True):
        try:
            z_min = str(self.z_min_ch_input.text() or "").strip()
            z_max = str(self.z_max_ch_input.text() or "").strip()
            if not z_min or not z_max:
                raise ValueError

            def _fallback(text, default):
                value = str(text or "").strip()
                if value:
                    return value
                return str(default if default is not None else "").strip()

            x_min_ref = _fallback(self.x_min_ch_input.text() if hasattr(self, "x_min_ch_input") else "", self.xmin_val)
            x_max_ref = _fallback(self.x_max_ch_input.text() if hasattr(self, "x_max_ch_input") else "", self.xmax_val)
            y_min_ref = _fallback(self.y_min_ch_input.text() if hasattr(self, "y_min_ch_input") else "", self.ymin_val)
            y_max_ref = _fallback(self.y_max_ch_input.text() if hasattr(self, "y_max_ch_input") else "", self.ymax_val)
            if self.fits_viewer.data.ndim == 3:
                zp_min = float(self.converter.world_to_pix(x_min_ref, y_min_ref, z_min)[2])
                zp_max = float(self.converter.world_to_pix(x_max_ref, y_max_ref, z_max)[2])
            elif self.fits_viewer.data.ndim == 4:
                zp_min = float(self.converter.world_to_pix(x_min_ref, y_min_ref, z_min, 0)[2])
                zp_max = float(self.converter.world_to_pix(x_max_ref, y_max_ref, z_max, 0)[2])
                
            if zp_min > zp_max: zp_min, zp_max = zp_max, zp_min
            
            if self.plane == 'xz':
                for ax in self.axes: ax.set_ylim(zp_min, zp_max)
            elif self.plane == 'zy':
                for ax in self.axes: ax.set_xlim(zp_min, zp_max)
            self.zmin_val = z_min
            self.zmax_val = z_max
            self._background = None
            self.canvas.draw_idle()
            if record_history and not bool(getattr(self, "_suspend_view_history_recording", False)):
                self._record_local_view_history(reason="range:z")
        except ValueError:
            if not getattr(self, "_suppress_range_warning", False):
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
            
            self.set_x_range(record_history=False)
            self.set_y_range(record_history=False)

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

            self.set_x_range(record_history=False)
            self.set_z_range(record_history=False)
            
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
            
            self.set_z_range(record_history=False)
            self.set_y_range(record_history=False)
        if not bool(getattr(self, "_suspend_view_history_recording", False)):
            self._record_local_view_history(reason=f"nav:{plane}")

    def resync_after_workspace_restore(self):
        refreshed = False
        if getattr(self, "im_list", None):
            try:
                self.update_images()
                refreshed = True
            except Exception:
                refreshed = False

        marker_manager = getattr(self, "marker_manager", None)
        marker_planes = list(getattr(self, "_marker_planes", []) or [])
        if marker_manager is not None and marker_planes:
            try:
                marker_manager.redraw_planes(marker_planes)
            except Exception:
                for plane_name in marker_planes:
                    try:
                        marker_manager.redraw_plane(plane_name)
                    except Exception:
                        continue

        if not refreshed:
            try:
                self._refresh_contours()
            except Exception:
                pass
            if getattr(self, "canvas", None) is not None:
                try:
                    self.canvas.draw_idle()
                except Exception:
                    pass
            
    @staticmethod
    def _clamp_float(value: float, low: float, high: float) -> float:
        return max(low, min(high, float(value)))

    def _event_to_figure_coords(self, event):
        fig = getattr(self, "fig", None)
        if fig is None:
            return None, None
        ex = getattr(event, "x", None)
        ey = getattr(event, "y", None)
        if ex is None or ey is None:
            return None, None
        
        # Robustly get figure dimensions
        bbox = getattr(fig, "bbox", None)
        fw = float(getattr(bbox, "width", 0.0) or 0.0)
        fh = float(getattr(bbox, "height", 0.0) or 0.0)
        
        if fw <= 0.0 or fh <= 0.0:
            return None, None
        return float(ex) / fw, float(ey) / fh

    def _set_colorbar_geometry(self, pos_x: float, pos_y: float, width: float, height: float):
        min_size = 0.01
        width = self._clamp_float(width, min_size, 1.0)
        height = self._clamp_float(height, min_size, 1.0)
        pos_x = self._clamp_float(pos_x, 0.0, max(0.0, 1.0 - width))
        pos_y = self._clamp_float(pos_y, 0.0, max(0.0, 1.0 - height))

        if getattr(self, "cax", None) is not None:
            try:
                self.cax.set_position([pos_x, pos_y, width, height])
                self.cax.set_gid("colorbar")
            except Exception:
                pass
            if getattr(self, "canvas", None) is not None:
                try:
                    self._request_canvas_redraw()
                except Exception:
                    pass

        # Update local config reference
        if isinstance(self.config, dict):
            self.config["cbar_pos_x"] = pos_x
            self.config["cbar_pos_y"] = pos_y
            self.config["cbar_width"] = width
            self.config["cbar_height"] = height

        # Explicitly update main viewer config for workspace persistence
        main_config_mgr = getattr(self.fits_viewer, "config_manager", None)
        if main_config_mgr is not None:
            main_config = getattr(main_config_mgr, "config", None)
            if isinstance(main_config, dict):
                main_config["cbar_pos_x"] = pos_x
                main_config["cbar_pos_y"] = pos_y
                main_config["cbar_width"] = width
                main_config["cbar_height"] = height

    def _begin_colorbar_drag(self, event) -> bool:
        if getattr(self, "toolbar", None) is not None:
            mode = getattr(self.toolbar, "mode", "")
            if mode in ("zoom rect", "pan/zoom"):
                return False
                
        if self._is_colorbar_auto_layout_enabled():
            return False
        if getattr(event, "dblclick", False):
            return False
        if getattr(event, "button", None) not in (1, 3):
            return False
            
        cax = getattr(event, "inaxes", None)
        if not self._is_colorbar_axes(cax):
            return False
            
        xfig, yfig = self._event_to_figure_coords(event)
        if xfig is None or yfig is None:
            return True
        try:
            x0, y0, w0, h0 = [float(v) for v in cax.get_position().bounds]
        except Exception:
            return False
            
        key_text = str(getattr(event, "key", "") or "").lower()
        mode = "resize" if (event.button == 3 or "shift" in key_text) else "move"
        self._colorbar_drag_state = {
            "mode": mode,
            "start_x": xfig,
            "start_y": yfig,
            "orig_x": x0,
            "orig_y": y0,
            "orig_w": w0,
            "orig_h": h0,
        }
        return True

    def _drag_colorbar(self, event) -> bool:
        drag_state = getattr(self, "_colorbar_drag_state", None)
        if not drag_state:
            return False
        xfig, yfig = self._event_to_figure_coords(event)
        if xfig is None or yfig is None:
            return True

        dx = xfig - drag_state["start_x"]
        dy = yfig - drag_state["start_y"]
        if drag_state["mode"] == "move":
            pos_x = drag_state["orig_x"] + dx
            pos_y = drag_state["orig_y"] + dy
            width = drag_state["orig_w"]
            height = drag_state["orig_h"]
        else:
            pos_x = drag_state["orig_x"]
            pos_y = drag_state["orig_y"]
            width = drag_state["orig_w"] + dx
            height = drag_state["orig_h"] + dy

        self._set_colorbar_geometry(pos_x, pos_y, width, height)
        return True

    def _end_colorbar_drag(self, event) -> bool:
        if getattr(self, "_colorbar_drag_state", None) is None:
            return False
        self._drag_colorbar(event)
        self._colorbar_drag_state = None
        return True

    def _is_colorbar_axes(self, ax) -> bool:
        if ax is None:
            return False
        # Check standard Matplotlib stored axes
        if getattr(self, "cax", None) is ax:
            return True
        # Also check the Axes of the colorbar object itself if distinct
        cb = getattr(self, "colorbar", None)
        if cb is not None and getattr(cb, "ax", None) is ax:
            return True
        # Fallback to gid check if set
        try:
            gid = ax.get_gid()
            if gid == "colorbar":
                return True
        except Exception:
            pass
        return False

    def _handle_colorbar_double_click(self, event) -> bool:
        if not getattr(event, "dblclick", False):
            return False
        if not self._is_colorbar_axes(getattr(event, "inaxes", None)):
            return False

        # Toggle window-local auto-layout mode so each window is independent.
        current_state = self._is_colorbar_auto_layout_enabled()
        new_state = not current_state
        self._colorbar_auto_layout_override = bool(new_state)

        state_str = "ON" if new_state else "OFF"
        print(f"Colorbar Auto-Layout: {state_str}")

        if new_state:
            # If turning ON, snap to the correct position immediately.
            self._schedule_colorbar_auto_layout(force=True)

        return True

    def on_click(self, event):
        if self._handle_colorbar_double_click(event):
            return
        if self._begin_colorbar_drag(event):
            return
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
        self._schedule_colorbar_auto_layout_if_anchor_changed(force=False)

###########
class ChannelMapSettingPanel(QDialog):
    def __init__(self, fits_viewer, subwindows):
        super().__init__()
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
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
        self._action_record_tag = "panel:channel_map"
        self.coord_wrap = self.fits_viewer.config_manager.config.get('coord_wrap')
        
        self.original_xlim = self.fits_viewer.ax.get_xlim()
        self.original_ylim = self.fits_viewer.ax.get_ylim()
        self.original_zlim = _resolve_z_view_limits(self.fits_viewer, self.subwindows)
        
        self.from_val =  0.5
        self.to_val = _axis_upper_pixel_edge(self.fits_viewer, 3)
        self.from_pix = 0.5
        self.to_pix = _axis_upper_pixel_edge(self.fits_viewer, 3)
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
        self.plane_xy_radio.toggled.connect(lambda checked: self.set_plane_num(0, checked))
        self.plane_xz_radio.toggled.connect(lambda checked: self.set_plane_num(1, checked))
        self.plane_zy_radio.toggled.connect(lambda checked: self.set_plane_num(2, checked))
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
        self.mode_integ_radio.toggled.connect(lambda checked: self.set_mode_num(0, checked))
        self.mode_avg_radio.toggled.connect(lambda checked: self.set_mode_num(1, checked))
        self.mode_slice_radio.toggled.connect(lambda checked: self.set_mode_num(2, checked))
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
        self.dir_l2r_radio.toggled.connect(lambda checked: self.set_dir_num(0))
        self.dir_t2b_radio.toggled.connect(lambda checked: self.set_dir_num(1))
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
        self.ch_radio.toggled.connect(lambda checked: self.set_worldch_num(0, checked))
        self.val_radio.toggled.connect(lambda checked: self.set_worldch_num(1, checked))
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
        self.chlabel_radio_middle.toggled.connect(lambda checked: self.set_chlabel_num(0))
        self.chlabel_radio_range.toggled.connect(lambda checked: self.set_chlabel_num(1))
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

    def set_plane_num(self, num, is_checked=None):
        """Set plane number based on selected plane radio button."""
        if is_checked is None:
            sender = self.sender()
            if sender:
                is_checked = sender.isChecked()
            else:
                is_checked = True

        if not is_checked:
            return
        self.plane_num = num
        if self.worldch_num == 0: #ch
            self.from_val =  0.5
            if self.plane_num == 0: #X-Y
                self.to_val = _axis_upper_pixel_edge(self.fits_viewer, 3)
            elif self.plane_num == 1:
                self.to_val = _axis_upper_pixel_edge(self.fits_viewer, 2)
            elif self.plane_num == 2:
                self.to_val = _axis_upper_pixel_edge(self.fits_viewer, 1)
            self.interval = 1
            
        elif self.worldch_num == 1: #world
            if self.plane_num == 0: #X-Y
                if self.fits_viewer.data.ndim == 3:
                    self.from_val = self.converter.pix_to_world(0, 0, -0.5)[2]
                    self.to_val = self.converter.pix_to_world(0, 0, _axis_last_pixel_center(self.fits_viewer, 3))[2]
                elif self.fits_viewer.data.ndim == 4:
                    self.from_val = self.converter.pix_to_world(0, 0, -0.5, 0)[2]
                    self.to_val = self.converter.pix_to_world(0, 0, _axis_last_pixel_center(self.fits_viewer, 3), 0)[2]
            elif self.plane_num == 1: #X-Z
                if self.fits_viewer.data.ndim == 3:
                    self.from_val = self.converter.pix_to_world(0, -0.5, 0)[1]
                    self.to_val = self.converter.pix_to_world(0, _axis_last_pixel_center(self.fits_viewer, 2), 0)[1]
                elif self.fits_viewer.data.ndim == 4:
                    self.from_val = self.converter.pix_to_world(0, -0.5, 0, 0)[1]
                    self.to_val = self.converter.pix_to_world(0, _axis_last_pixel_center(self.fits_viewer, 2), 0, 0)[1]
            elif self.plane_num == 2: #Z-Y
                if self.fits_viewer.data.ndim == 3:
                    self.from_val = self.converter.pix_to_world(-0.5, 0, 0)[0]
                    self.to_val = self.converter.pix_to_world(_axis_last_pixel_center(self.fits_viewer, 1), 0, 0)[0]
                elif self.fits_viewer.data.ndim == 4:
                    self.from_val = self.converter.pix_to_world(-0.5, 0, 0, 0)[0]
                    self.to_val = self.converter.pix_to_world(_axis_last_pixel_center(self.fits_viewer, 1), 0, 0, 0)[0]
                    
            interval_val = abs(self.wcs.wcs.cdelt[2-self.plane_num])
            axis_type = self.converter.get_axis_types()[2-self.plane_num]
            self.interval = self.converter.format_world_coordinate(interval_val, axis_type)
            
        self.from_edit.setText(str(self.from_val))
        self.to_edit.setText(str(self.to_val))
        self.interval_edit.setText(str(self.interval))
        if self.mode_num == 2:
            self.set_mode_num(2)

    def set_worldch_num(self, num, is_checked=None):
        """Set unit selection (0 for 'ch', 1 for 'world')."""
        if is_checked is None:
            sender = self.sender()
            if sender:
                is_checked = sender.isChecked()
            else:
                is_checked = True  # Fallback if no sender and no boolean passed

        # Ignore if sender is being unchecked.
        if not is_checked:
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

    def set_mode_num(self, num, is_checked=None):
        """Set mode number based on selected mode radio button."""
        if is_checked is None:
            sender = self.sender()
            if sender:
                is_checked = sender.isChecked()
            else:
                is_checked = True

        if not is_checked:
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
            self.chlabel_radio_middle.setChecked(True)
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

    def remove_window_reference(self, window):
        action_tag = str(getattr(window, "_workspace_action_tag", "") or "").strip() if window is not None else ""
        if action_tag:
            clear_action_preview_record(self.fits_viewer, action_tag)
        if window in self.channel_result_windows:
            self.channel_result_windows.remove(window)
        if hasattr(self.fits_viewer, 'channel_map_windows') and window in self.fits_viewer.channel_map_windows:
            self.fits_viewer.channel_map_windows.remove(window)
        self._clear_channel_map_action_if_no_windows()

    def _clear_channel_map_action_if_no_windows(self):
        live_windows = []
        for window in list(self.channel_result_windows):
            if window is None:
                continue
            try:
                if window.isVisible():
                    live_windows.append(window)
            except Exception:
                continue
        if not live_windows:
            clear_action_preview_record(
                self.fits_viewer,
                self._action_record_tag,
                action_name="compute_channel_map",
            )

    def _record_current_channel_map_action(self, payload: dict):
        if not isinstance(payload, dict):
            return
        action_tag = str(payload.get("_window_action_tag") or "").strip()
        if not action_tag:
            action_tag = f"{self._action_record_tag}:{uuid.uuid4().hex}"
            payload["_window_action_tag"] = action_tag
        record_action_preview(
            self.fits_viewer,
            "compute_channel_map",
            payload,
            replace_tag=action_tag,
        )

    def _labels_to_world_strings(self, labels, plane_num):
        str_labels = []
        for s_pix, c_pix, e_pix in labels:
            try:
                if plane_num == 0:
                    if self.fits_viewer.data.ndim == 3:
                        s_w = self.converter.pix_to_world(0, 0, s_pix)[2]
                        c_w = self.converter.pix_to_world(0, 0, c_pix)[2]
                        e_w = self.converter.pix_to_world(0, 0, e_pix)[2]
                    else:
                        s_w = self.converter.pix_to_world(0, 0, s_pix, 0)[2]
                        c_w = self.converter.pix_to_world(0, 0, c_pix, 0)[2]
                        e_w = self.converter.pix_to_world(0, 0, e_pix, 0)[2]
                elif plane_num == 1:
                    if self.fits_viewer.data.ndim == 3:
                        s_w = self.converter.pix_to_world(0, s_pix, 0)[1]
                        c_w = self.converter.pix_to_world(0, c_pix, 0)[1]
                        e_w = self.converter.pix_to_world(0, e_pix, 0)[1]
                    else:
                        s_w = self.converter.pix_to_world(0, s_pix, 0, 0)[1]
                        c_w = self.converter.pix_to_world(0, c_pix, 0, 0)[1]
                        e_w = self.converter.pix_to_world(0, e_pix, 0, 0)[1]
                else:
                    if self.fits_viewer.data.ndim == 3:
                        s_w = self.converter.pix_to_world(s_pix, 0, 0)[0]
                        c_w = self.converter.pix_to_world(c_pix, 0, 0)[0]
                        e_w = self.converter.pix_to_world(e_pix, 0, 0)[0]
                    else:
                        s_w = self.converter.pix_to_world(s_pix, 0, 0, 0)[0]
                        c_w = self.converter.pix_to_world(c_pix, 0, 0, 0)[0]
                        e_w = self.converter.pix_to_world(e_pix, 0, 0, 0)[0]
                str_labels.append([s_w, c_w, e_w])
            except Exception:
                str_labels.append([f"{s_pix:.4g}", f"{c_pix:.4g}", f"{e_pix:.4g}"])
        return str_labels

    @staticmethod
    def _normalize_unit_text(unit) -> str:
        return str(unit or "").strip().replace(" ", "")

    def _axis_label_for_integration_axis(self, axis_index: int) -> str:
        try:
            if axis_index == 2:
                xz_window = _resolve_xz_subwindow(self.subwindows)
                if xz_window is not None:
                    return str(xz_window.ax.get_ylabel() or "")
                return ""
            if axis_index == 1:
                return str(self.fits_viewer.ax.get_ylabel() or "")
            if axis_index == 0:
                return str(self.fits_viewer.ax.get_xlabel() or "")
        except Exception:
            return ""
        return ""

    def _extract_unit_from_axis_label(self, axis_label: str) -> str:
        match = re.search(r"\[(.*?)\]", str(axis_label or ""))
        if not match:
            return ""
        return self._normalize_unit_text(match.group(1))

    def _channel_map_intensity_unit(self, *, mode: str, plane_num: int) -> str:
        header = getattr(self.fits_viewer, "header", None)
        base_unit = ""
        if header is not None:
            try:
                base_unit = str(header.get("BUNIT", "") or "").strip()
            except Exception:
                base_unit = ""

        mode_name = str(mode or "").strip().lower()
        if mode_name != "integrate":
            return base_unit

        axis_index = {0: 2, 1: 1, 2: 0}.get(int(plane_num), 2)
        axis_unit = ""
        axis_label = self._axis_label_for_integration_axis(axis_index)
        axis_unit = self._extract_unit_from_axis_label(axis_label)

        if not axis_unit and header is not None:
            try:
                axis_unit = self._normalize_unit_text(header.get(f"CUNIT{axis_index + 1}", ""))
            except Exception:
                axis_unit = ""

        try:
            if not axis_unit:
                cunit = self.wcs.wcs.cunit[axis_index]
                axis_unit = self._normalize_unit_text(
                    cunit.to_string() if hasattr(cunit, "to_string") else cunit
                )
        except Exception:
            if not axis_unit:
                axis_unit = ""

        if base_unit and axis_unit:
            return f"{base_unit} {axis_unit}".strip()
        return (base_unit or axis_unit).strip()

    def _open_channel_map_window(
        self,
        images,
        str_labels,
        *,
        tiles_x: int,
        tiles_y: int,
        plane_num: int,
        dir_num: int,
        chlabel_num: int,
        title: str = "Channel Map",
        action_tag: Optional[str] = None,
        mode: str = "average",
    ):
        self.ch_imdata = images
        self.range_label = str_labels
        self.tiles_x = int(tiles_x)
        self.tiles_y = int(tiles_y)
        self.channel_map_window = ChannelMapWindow(
            fits_viewer=self.fits_viewer,
            subwindows=self.subwindows,
            ch_imdata=self.ch_imdata,
            range_label=self.range_label,
            tiles_x=self.tiles_x,
            tiles_y=self.tiles_y,
            dir_num=int(dir_num),
            chlabel_num=int(chlabel_num),
            wcs=self.wcs,
            plane_num=int(plane_num),
            title=title,
            intensity_unit=self._channel_map_intensity_unit(mode=mode, plane_num=plane_num),
        )
        tag = str(action_tag or "").strip()
        if not tag:
            tag = f"{self._action_record_tag}:{uuid.uuid4().hex}"
        self.channel_map_window._workspace_action_tag = tag
        self.channel_result_windows.append(self.channel_map_window)
        if hasattr(self.fits_viewer, 'channel_map_windows'):
            self.fits_viewer.channel_map_windows.append(self.channel_map_window)
        self.channel_map_window.show()
        self.channel_map_window.destroyed.connect(
            lambda *_args, win=self.channel_map_window: self.remove_window_reference(win)
        )
        return self.channel_map_window

    def restore_window_from_action_params(
        self,
        params: dict,
        app_state_override: Optional[AppState] = None,
    ) -> bool:
        if not isinstance(params, dict):
            return False
        if app_state_override is not None and getattr(app_state_override, "data", None) is not None:
            state = app_state_override
        else:
            state = create_app_state(
                data=self.fits_viewer.data,
                header=self.fits_viewer.header,
                wcs=self.wcs,
                filepath=getattr(self.fits_viewer, "filename", None),
            )
        try:
            axis = max(0, min(int(params.get("axis", 0)), 2))
            start_channel = float(params.get("start_channel", 0.0))
            end_channel_param = params.get("end_channel")
            end_channel = float(end_channel_param) if end_channel_param is not None else None
            interval = float(params.get("interval", 1.0))
            mode = str(params.get("mode") or "average").lower()
            if mode not in {"slice", "average", "integrate"}:
                mode = "average"
        except Exception:
            return False

        try:
            result = compute_channel_map(
                state=state,
                start_channel=start_channel,
                end_channel=end_channel,
                interval=interval,
                mode=mode,
                axis=axis,
            )
        except Exception:
            return False

        images = list(result.images)
        labels = list(getattr(result, "display_labels", result.labels))
        reverse = bool(params.get("reverse", False))
        if reverse:
            images = images[::-1]
            labels = labels[::-1]

        if mode == "integrate":
            try:
                wcs_axis = 2 - axis
                cdelt = abs(self.wcs.wcs.cdelt[wcs_axis])
                images = [img * cdelt for img in images]
            except Exception:
                pass

        try:
            plane_num = max(0, min(int(params.get("plane_num", axis)), 2))
        except Exception:
            plane_num = axis

        if plane_num == 2:
            images = [img.T for img in images]

        str_labels = self._labels_to_world_strings(labels, plane_num)
        try:
            tiles_x = max(1, int(params.get("tiles_x", self.tiles_x_combo.currentText())))
        except Exception:
            tiles_x = 4
        try:
            tiles_y = max(1, int(params.get("tiles_y", self.tiles_y_combo.currentText())))
        except Exception:
            tiles_y = 4
        try:
            dir_num = int(params.get("dir_num", self.dir_num))
        except Exception:
            dir_num = 0
        try:
            chlabel_num = int(params.get("chlabel_num", self.chlabel_num))
        except Exception:
            chlabel_num = 0
        title = str(params.get("title") or "Channel Map")
        action_tag = str(params.get("_window_action_tag") or "").strip()

        self._open_channel_map_window(
            images,
            str_labels,
            tiles_x=tiles_x,
            tiles_y=tiles_y,
            plane_num=plane_num,
            dir_num=dir_num,
            chlabel_num=chlabel_num,
            title=title,
            action_tag=action_tag,
            mode=mode,
        )
        return True


    def mkchmap(self):
        # Ensure we read pixel values
        restore_val_radio = False
        if self.val_radio.isChecked():
            self.ch_radio.setChecked(True) # This triggers conversion to pixels in text fields
            restore_val_radio = True
            
        try:
            from_pix = float(self.from_edit.text())
            to_pix = float(self.to_edit.text())
            interval_pix = float(self.interval_edit.text())
        except ValueError:
             QMessageBox.warning(self, 'Error', 'Invalid range values')
             if restore_val_radio: self.val_radio.setChecked(True)
             return

        if restore_val_radio:
            self.val_radio.setChecked(True)
        if interval_pix <= 0:
            QMessageBox.warning(self, 'Error', 'Interval must be greater than 0')
            return

        # Determine axis and AppState
        axis = self.plane_num 
        # plane_num: 0(XY)->Z(axis=0), 1(XZ)->Y(axis=1), 2(ZY)->X(axis=2) in some conventions?
        # Let's verify mapping against compute_channel_map expectations.
        # compute_channel_map uses `data.shape[axis]`. 
        # AppState.data is (Z, Y, X).
        # plane=0 (XY view) -> we scroll through Z -> axis=0.
        # plane=1 (XZ view) -> we scroll through Y -> axis=1.
        # plane=2 (ZY view) -> we scroll through X -> axis=2.
        # Matches.

        state = create_app_state(
            data=self.fits_viewer.data,
            header=self.fits_viewer.header,
            wcs=self.wcs,
            filepath=getattr(self.fits_viewer, 'filename', None)
        )

        # Handle direction
        reverse = False
        if from_pix > to_pix:
            from_pix, to_pix = to_pix, from_pix
            reverse = True
        
        # Determine mode
        # mode_num: 0=Integrate, 1=Average, 2=Slice
        mode_str = "average"
        if hasattr(self, 'mode_num'):
            if self.mode_num == 0: mode_str = "integrate"
            elif self.mode_num == 1: mode_str = "average"
            elif self.mode_num == 2: mode_str = "slice"
        else:
            # Fallback if mode_num not set (check radio buttons if accessible, or default)
            # Assuming mode_num is reliably set by radio buttons
             mode_str = "average"

        # compute_channel_map supports float interval/range for fractional logic
        
        # Adjust inputs to 1-based -> 0-based for compute_channel_map
        # (GUI uses 1-based indexing for user, logic uses 0-based)
        i_start_0based = from_pix - 1.0
        i_interval = interval_pix
        
        if mode_str == "slice":
            # For slice mode, original implementation used: count = int((to-from)/interval + 1)
            # This implies the end is inclusive.
            # We calculate end_0based such that the loop runs exactly 'count' times.
            count = int(abs(to_pix - from_pix)/interval_pix + 1)
            i_end_0based = i_start_0based + count * i_interval
            # Using exact math to cover the last element. 
        else:
            i_end_0based = to_pix - 1.0
        
        try:
             result = compute_channel_map(
                 state=state,
                 start_channel=i_start_0based,
                 end_channel=i_end_0based,
                 interval=i_interval,
                 mode=mode_str,
                 axis=axis
             )
        except Exception as e:
             QMessageBox.critical(self, "Error", f"Failed to compute channel map:\n{e}")
             return

        images = list(result.images)
        labels = getattr(result, "display_labels", result.labels) # list of (start, center, end) floats (0-based pixels)
        labels = list(labels)

        if reverse:
            images = images[::-1]
            labels = labels[::-1]

        # Handle 'Integrate' scaling (matches GUI logic of Sum * cdelt)
        # usecase 'integrate' is just Sum.
        if mode_str == "integrate":
            # cdelt for the slicing axis
            # axis=0 -> Z -> cdelt[2] (in wcs terms, index 2)
            # axis=1 -> Y -> cdelt[1]
            # axis=2 -> X -> cdelt[0]
            # WCS indexing is reverse of numpy usually?
            # WCS: 1=X, 2=Y, 3=Z.
            # python axis 0 (Z) -> WCS axis 3.
            # python axis 1 (Y) -> WCS axis 2.
            # python axis 2 (X) -> WCS axis 1.
            
            # wcs.wcs.cdelt is [dx, dy, dz] usually?
            # cdelt[0] is X, cdelt[1] is Y, cdelt[2] is Z.
            
            wcs_axis = 2 - axis # 0->2, 1->1, 2->0.
            try:
                cdelt = abs(self.wcs.wcs.cdelt[wcs_axis])
                images = [img * cdelt for img in images]
            except:
                pass

        # Transpose images if needed
        # GUI logic: 
        # if plane_num == 2 (ZY): `integ_result.T`.
        # Usecase returns (Y, X) for Z-slice (axis 0).
        # For axis 2 (X-slice), result is (Z, Y).
        # For axis 1 (Y-slice), result is (Z, X).
        # GUI expects images to be passed to `imshow(..., origin='lower')`.
        # We need to verify orientation.
        # If plane=0 (XY), image is (Y, X). Correct.
        # If plane=1 (XZ), image is (Z, X)? Or (Y, X) of the slice?
        # Usecase: slicing axis 1 (Y). Result is (Z, X) (dims 0 and 2).
        # If plane=2 (ZY), image is (Z, Y) (dims 0 and 1).
        
        # mkchmap line 2107: `if self.plane_num == 2: self.ch_imdata.append(integ_result.T)`
        # `integ_result` from `integration` likely preserves data shape order?
        # `integration` uses `np.take`. Returns array with one dim removed.
        # If axis=2 (X), result is (Z, Y).
        # Transpose -> (Y, Z).
        # Why transpose? Maybe `imshow` expects (row, col).
        # If we want to show Z on Y-axis (vertical) and Y on X-axis?
        # ZY plane usually means Z vertical, Y horizontal? Or Y vertical, Z horizontal?
        # Standard fits viewer ZY plane: Z vertical, Y horizontal?
        # Actually usually Y is horizontal in ZY view? 
        # Let's perform the transpose if UseCase output differs from GUI expectation.
        # GUI transposes for plane_num=2.
        # For plane_num=1 (XZ), it does NOT transpose.
        # UseCase axis=1 (Y) -> (Z, X).
        # UseCase axis=2 (X) -> (Z, Y).
        # If GUI transposes (Z, Y) -> (Y, Z).
        
        if self.plane_num == 2:
            images = [img.T for img in images]

        str_labels = self._labels_to_world_strings(labels, self.plane_num)
        try:
            self.tiles_x = max(1, int(self.tiles_x_combo.currentText()))
        except Exception:
            self.tiles_x = 4
        try:
            self.tiles_y = max(1, int(self.tiles_y_combo.currentText()))
        except Exception:
            self.tiles_y = 4

        window = self._open_channel_map_window(
            images,
            str_labels,
            tiles_x=self.tiles_x,
            tiles_y=self.tiles_y,
            plane_num=self.plane_num,
            dir_num=self.dir_num,
            chlabel_num=self.chlabel_num,
            title="Channel Map",
            mode=mode_str,
        )
        action_tag = str(getattr(window, "_workspace_action_tag", "") or "").strip() if window is not None else ""
        self._record_current_channel_map_action(
            {
                "axis": int(axis),
                "start_channel": float(i_start_0based),
                "end_channel": float(i_end_0based),
                "interval": float(i_interval),
                "mode": mode_str,
                "reverse": bool(reverse),
                "plane_num": int(self.plane_num),
                "tiles_x": int(self.tiles_x),
                "tiles_y": int(self.tiles_y),
                "dir_num": int(self.dir_num),
                "chlabel_num": int(self.chlabel_num),
                "title": "Channel Map",
                "_window_action_tag": action_tag,
            }
        )


    def move_to_default_position(self):
        # Get MainWindow geometry
        mainwindow_geometry = self.fits_viewer.geometry()
        mainwindow_x = mainwindow_geometry.x()
        mainwindow_y = mainwindow_geometry.y()
        mainwindow_width = mainwindow_geometry.width()

        # Move ControlPanel to the right of MainWindow
        self.move(mainwindow_x + mainwindow_width, mainwindow_y - 28)
