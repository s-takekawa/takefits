import copy
import json
from typing import Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QGroupBox, QFormLayout, QCheckBox, QComboBox, QDoubleSpinBox, QLabel, QButtonGroup, QRadioButton, QLineEdit, QGridLayout, QFileDialog, QMessageBox, QScrollArea, QSizePolicy, QAbstractScrollArea, QSlider, QMenu
import time
import math
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.ticker import FuncFormatter, AutoMinorLocator
from takefits.ui.navigation_toolbar import MyNavigationToolbar
from takefits.tools.color_scale import ColorSettingsPanel, ColorMode
from takefits.tools.pv_polyline import PolylinePathInteraction
from takefits.core.contour_manager import ContourManager, ContourItem
from takefits.core.coordinate import CoordinateConverter
from takefits.core.wcs_frames import (
    build_native_world_vector,
    native_celestial_frame,
    normalize_display_frame,
    transform_world_vector_between_frames_with_status,
)
from astropy.wcs.utils import proj_plane_pixel_scales
from takefits.core.history_provenance import build_processing_history_lines_with_action
from takefits.core.usecases import (
    EllipsePathGeometry,
    PolylinePathGeometry,
    POSITION_ORIGIN_CENTER,
    POSITION_ORIGIN_START,
    PV_SPLINE_BSPLINE,
    PV_SPLINE_CATMULL_ROM,
    PV_SPLINE_NONE,
    PV_X_AXIS_PHI,
    PV_X_AXIS_POSITION,
    anchored_straight_line,
    clamp_pv_smoothness,
    compute_pv,
    export_figure,
    export_pv_fits,
    fraction_from_position,
    normalize_position_origin,
    normalize_pv_spline_type,
    normalize_pv_x_axis_mode,
    position_axis_bounds,
    position_from_fraction,
    sample_count_from_spacing,
    sample_path_points,
    set_pv_endpoints,
    straight_line_from_center,
    StraightPathGeometry,
)
from takefits.tools.pv_slit_overlay import build_slit_overlay

DEFAULT_SAMPLE_SPACING_PIX = 1.0
PV_COLOR_HISTOGRAM_DEBOUNCE_MS = 150


def _sample_centered_image_extent(
    bounds,
    num_samples,
):
    """Expand centre-coordinate bounds into the pixel edges used by imshow."""
    start, end = float(bounds[0]), float(bounds[1])
    count = int(num_samples)
    if count <= 1:
        return start, end
    step = (end - start) / float(count - 1)
    return start - step / 2.0, end + step / 2.0


class PVNavigationToolbar(MyNavigationToolbar):
    """PV diagram window specific navigation toolbar.

    Overrides home and get_current_lim to interact with PVdiagram's specific methods
    (reset_pv_range and update_range_inputs) instead of FITSViewer's range panel.
    """

    def home(self, *args):
        """Override home button action to reset PV range."""
        if hasattr(self.parent, 'reset_pv_range'):
            self.parent.reset_pv_range()

    def get_current_lim(self, event):
        """Override zoom/pan action to update PV range inputs."""
        if event.inaxes and event.inaxes.get_gid() == "colorbar":
            return
        if self.ax is None:
            return

        if hasattr(self.parent, 'update_range_inputs'):
            self.parent.is_range_manual = True
            self.parent.update_range_inputs()

    def save_figure(self):
        # Get base filename logic from parent class (MyNavigationToolbar)
        current_dir = os.getcwd()
        desktop_dir = os.path.join(os.path.expanduser("~"), "Desktop")
        initial_dir = desktop_dir if os.access(desktop_dir, os.W_OK) else current_dir

        if not self.default_image_name:
            if hasattr(self.parent, "filename") and self.parent.filename:
                self.default_image_name = self.parent.filename
            else:
                self.default_image_name = "figure"

        # Remove .fits extension if present
        if self.default_image_name.endswith(".fits"):
            base_name = self.default_image_name[:-5]
        else:
            base_name = self.default_image_name

        # --- Customization for PV Diagram ---
        # Add ".pv" suffix and set default extension to "pdf"
        custom_suffix = ".pv"
        default_extension = "pdf" # Force pdf as primary extension for this specific case
        default_filename = f"{base_name}{custom_suffix}.{default_extension}"
        # --- End Customization ---

        path, selected_filter = QFileDialog.getSaveFileName(
            self.canvas.parent(),
            "Save Figure",
            os.path.join(initial_dir, default_filename),
            f"PDF Files (*.pdf);;EPS Files (*.eps);;SVG Files (*.svg);;PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;All Files (*)"
        )

        if path:
            # Standard saving procedure (copied from base class logic if needed, ensuring transparency/dpi settings)
            connections = self.canvas.callbacks.callbacks
            saved_connections = connections.copy()
            self.canvas.callbacks.callbacks = {}

            visibility = []
            if hasattr(self.ax, 'patches'):
                for patch in self.ax.patches:
                    visibility.append(patch.get_visible())
                    patch.set_visible(True)

            try:
                # Use headless usecase
                export_figure(self.canvas.figure, path, dpi=300, transparent=True)
            except Exception as e:
                print(f"Error saving figure: {e}")
            finally:
                # Restore visibility and connections
                if hasattr(self.ax, 'patches'):
                    for patch, vis in zip(self.ax.patches, visibility):
                        patch.set_visible(vis)
                self.canvas.callbacks.callbacks = saved_connections

            filename = os.path.basename(path)
            # Use show_save_success_message if available on parent, otherwise print
            if hasattr(self, 'show_save_success_message'):
                self.show_save_success_message(path, filename)
            else:
                print(f"Figure saved to {path}")



class PVdiagram(QMainWindow):
    # This window builds its own menu bar (the "Actions" menu, below), so it
    # opts out of the main window's menu-bar mirroring: mirroring would clear
    # and overwrite the bespoke "Actions" entries the moment the PV window is
    # focused. See takefits.ui.menu_bar.mirror_menu_bar_to_window.
    _owns_menu_bar = True

    # Slit (PV path) undo/redo keyboard shortcuts. Aligned with the app-wide
    # Undo/Redo convention (Option+Left / Option+Right == Alt+Left/Right, the
    # same keys as "Undo Analysis"), NOT Cmd+Z which is reserved for View Back.
    _SLIT_UNDO_SEQUENCES = ("Alt+Left",)
    _SLIT_REDO_SEQUENCES = ("Alt+Right",)

    def __init__(self, fits_viewer):
        super().__init__()
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.fits_viewer = fits_viewer
        self.wcs = self.fits_viewer.wcs
        if self.fits_viewer.data.ndim == 3:
            self.data = self.fits_viewer.data
        elif self.fits_viewer.data.ndim == 4:
            self.data = self.fits_viewer.data[0]

        # Initialize the coordinate converter
        self.coord_converter = CoordinateConverter(self.wcs, self.fits_viewer.config_manager.config)

        self._get_pixel_scale()
        self.length_unit = self._default_length_unit()

        self.color_settings_panel = None
        self._contour_layer_id: Optional[str] = None
        self._contour_title_connected = False
        pv_settings = ColorSettingsPanel.settings[ColorMode.PV]
        self.color_pattern = (
            pv_settings['color_pattern'] or
            ColorSettingsPanel.settings[ColorMode.MAIN]['color_pattern'] or
            self.fits_viewer.displaymap.config.get('colorscale')
        )
        self.min_val = pv_settings['min_val']
        self.max_val = pv_settings['max_val']
        self._color_panel_hint = dict(pv_settings or {})

        if self.min_val is not None and self.max_val is not None:
            self.is_clim_fixed = True
        else:
            self.is_clim_fixed = False


        # Initialize line endpoints and state flags
        self.line_start = None
        self.line_end = None
        self.line_fixed = False  # Flag indicating if the line is finalized
        self.edit_mode = None    # Modes: None, "endpoint", "move", "rotate"
        self.dragging_endpoint = None
        self.drag_start = None
        self.initial_line_start = None
        self.initial_line_end = None
        self._pending_precise_world_line = None

        # For rotate mode: center, initial angle, and line length
        self.center = None
        self.initial_angle = None
        self.initial_line_length = 0

        self.drag_mode = None  # For new drawing mode
        self.arrow_artist = None
        self.pos_indicator_on_arrow = None
        self.dragging_pos_indicator = False
        self.last_position_coord = None

        # Markers for endpoints and rotation center
        self.marker_artist_start = None
        self.marker_artist_end = None
        self.center_marker = None

        # Polyline-only live display toggle for the node-point markers. Turning
        # this off hides the polyline control-point dots (and selection ring),
        # e.g. to export clean figures for papers. Exposed via nodeMarkerCheck,
        # which is shown only in polyline mode.
        self.show_node_markers = True

        self.indicator_positions = {
            "start": True,
            "center": False,
            "end": False
        }
        # Ellipse uses an independent default: the center cross is on, start/end
        # (phi0/phi1) ticks are off. start/end overlap on a closed ellipse but are
        # kept as items for future arc support.
        self.ellipse_indicator_positions = {
            "start": False,
            "center": True,
            "end": False,
        }
        self.width_indicators = []

        self.slice_width = 1.0       # Default slice width
        self.sample_spacing_pix = DEFAULT_SAMPLE_SPACING_PIX
        self.x_axis_mode = PV_X_AXIS_POSITION
        self.position_axis_flipped = False
        self.ellipse_axis_unit = "pixel"
        self.position_origin = POSITION_ORIGIN_START
        self._endpoint_position_origin = POSITION_ORIGIN_START
        self.geometry_input_mode = "endpoints"
        self._updating_geometry_controls = False
        self.path_type = "straight"
        self.pv_path_items = []
        self.active_pv_path_id = None
        self._next_pv_path_index = 1
        self.inactive_path_artists = {}
        self._syncing_path_list_combo = False
        self._restoring_single_pv_path = False
        self.ellipse_geometry = None

        # Polyline (multi-segment) path state. The active polyline's vertices live
        # here (single source of truth for serialization); PolylinePathInteraction
        # owns the mouse interaction + overlay artists.
        self.polyline_vertices = []          # list[(x, y)] in main-image pixel coords
        self.polyline_finished = False       # True once the path is committed (not mid-draw)
        self.polyline_selected_index = None  # index of the selected/edited node
        # When re-extending a finished path from its start node, new nodes are
        # prepended (verts[0]) and the rubber band anchors at verts[0] instead of
        # verts[-1]. False = extend/append from the end (default add direction).
        self.polyline_extend_from_start = False
        self.polyline_line_artist = None     # Line2D through the vertices
        self.polyline_node_artist = None     # node markers
        self.polyline_rubber_artist = None   # rubber-band preview to the cursor
        self.polyline_select_artist = None   # highlight ring on the selected node
        self.polyline_indicator_artists = []  # start/center/end ticks + position marker
        self.polyline_spline_type = PV_SPLINE_NONE  # "none" | "catmull_rom"
        self.polyline_smoothness = 1.0       # spline roundness 0..1 (Catmull-Rom)
        self._polyline = PolylinePathInteraction(self)  # interaction delegate
        self.ellipse_artist = None
        self.ellipse_handle_artists = {}
        self.ellipse_indicator_artists = []
        self.ellipse_drag_anchor = None
        self.initial_ellipse_geometry = None
        self.ellipse_resize_handle = None
        self.ellipse_rotation_reference_angle = None

        # Key press states
        self.command_pressed = False
        self.shift_pressed = False

        self.pvarrow_color = 'yellow'
        self.pvmarker_color = 'None'
        self.arrow_size = 1.

        self.weight_mode = 0  # 0: bilinear interpolation, 1: gaussian weighting
        self.last_update_time = 0

        self.original_position_range = None # Store initial full range for position axis
        self.original_velocity_range = None # Store initial full range for velocity axis
        self.is_range_manual = False      # Flag to check if zoom/pan or manual input has occurred
        self.arrow_is_being_dragged = False


        self.initUI()
        self._register_contour_layer()

        try:
            range_panel = getattr(self.fits_viewer, 'range_panel', None)
            if range_panel and hasattr(range_panel, 'z_min_input'):
                initial_vel_min = range_panel.z_min_input.text()
                initial_vel_max = range_panel.z_max_input.text()

                if initial_vel_min and initial_vel_max:
                    self.vel_min_input.setText(initial_vel_min)
                    self.vel_max_input.setText(initial_vel_max)
                    #self.set_pv_range()
                    self.is_range_manual = True
        except AttributeError:
             print("Warning: Could not sync initial velocity range for PV diagram.")
        except Exception as e:
            print(f"Error during initial velocity range sync: {e}")

        try:
            slider = getattr(self.fits_viewer, 'slider', None)
            if slider:
                initial_channel = slider.value()
                if initial_channel >= 0:
                    self.update_cursor(initial_channel)
                    self.h_cursor_line.set_visible(not self.swapAxesCheck.isChecked())
                    self.v_cursor_line.set_visible(self.swapAxesCheck.isChecked())
        except AttributeError:
            pass


        try:
            range_panel = getattr(self.fits_viewer, 'range_panel', None)
            if range_panel and hasattr(range_panel, 'z_min_input'):
                initial_vel_min_str = range_panel.z_min_input.text()
                initial_vel_max_str = range_panel.z_max_input.text()

                if initial_vel_min_str and initial_vel_max_str:
                    float(initial_vel_min_str)
                    float(initial_vel_max_str)
                    self.vel_min_input.setText(initial_vel_min_str)
                    self.vel_max_input.setText(initial_vel_max_str)
                    #self.set_pv_range()
                    self.is_range_manual = True
        except (AttributeError, ValueError):
            pass

    def _get_pixel_scale(self):
        """Calculate pixel scale in degrees from WCS more robustly."""
        self.pixel_scale_deg = None
        if self.wcs:
            try:
                # First, try using the standard utility which is more accurate for projections
                if self.wcs.is_celestial:
                    scales = proj_plane_pixel_scales(self.wcs)
                    self.pixel_scale_deg = (abs(scales[0]) + abs(scales[1])) / 2.0

                # If that fails or it's not a celestial WCS, try reading CDELT keywords
                if self.pixel_scale_deg is None and self.wcs.wcs.cdelt is not None:
                    cdelt1 = abs(self.wcs.wcs.cdelt[0])
                    cdelt2 = abs(self.wcs.wcs.cdelt[1])

                    # Convert unit object to string before comparing
                    unit_str = str(self.wcs.wcs.cunit[0]).lower()

                    if unit_str in ('deg', 'degree', 'degrees'):
                        self.pixel_scale_deg = (cdelt1 + cdelt2) / 2.0

            except Exception as e:
                print(f"Could not determine pixel scale from WCS: {e}")

        if self.pixel_scale_deg is None:
            print("Warning: Pixel scale in degrees could not be determined. Angular unit conversion will be disabled.")

    def _default_length_unit(self):
        """Choose a readable angular unit for lengths spanning this image."""
        if self.pixel_scale_deg is None:
            return 'pixel'
        try:
            spatial_shape = self.data.shape[-2:]
            max_span_px = max(max(int(size) - 1, 1) for size in spatial_shape)
            field_span_deg = max_span_px * float(self.pixel_scale_deg)
        except Exception:
            return 'deg'
        if field_span_deg < (1.0 / 60.0):
            return 'arcsec'
        return 'deg'

    def _convert_length(self, length, from_unit, to_unit):
        """Convert length between different units."""
        if from_unit == to_unit or self.pixel_scale_deg is None:
            return length

        # Conversion factors to degrees
        factors_to_deg = {
            'pixel': self.pixel_scale_deg,
            'deg': 1.0,
            'arcmin': 1.0 / 60.0,
            'arcsec': 1.0 / 3600.0
        }

        # First, convert the input length to degrees
        length_deg = length * factors_to_deg[from_unit]

        # Then, convert from degrees to the target unit
        len_out = length_deg / factors_to_deg[to_unit]

        return len_out


    def _get_dynamic_step(self, value):
            """Calculate a reasonable step value based on the number's 3rd significant digit."""
            if value == 0:
                # If value is zero, use the current spinbox step or a small default
                return self.arrowLengthSpin.singleStep() or 0.01

            # Calculate the order of magnitude of the 3rd significant digit
            # For 0.021833, log10 is ~-1.66. floor(-1.66 - 2) = -4. 10**-4 = 0.0001
            power = 10**math.floor(math.log10(abs(value)) - 2)
            return power


    def initUI(self):
        self.setWindowTitle(f'PV Diagram: {self.fits_viewer.filename}')

        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # self.background = np.sum(self.data, axis=0) # This is too slow for large FITS
        data_shape_y = self.data.shape[-2]
        data_shape_x = self.data.shape[-1]

        self.fits_ax = self.fits_viewer.overlay_ax
        self.fits_ax.set_autoscale_on(False)
        self.fits_canvas = self.fits_viewer.canvas
        self.fits_canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.fits_canvas.setFocus()

        self.pvResultPanel = QWidget(main_widget)
        central_layout = QVBoxLayout(self.pvResultPanel)
        central_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.pvResultPanel, stretch=5)

        control_panel_right_widget = QWidget()
        control_layout_right = QVBoxLayout(control_panel_right_widget)
        control_layout_right.setContentsMargins(0, 0, 0, 0)
        control_layout_right.setSpacing(6)
        control_panel_right_widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self.controlScrollArea = QScrollArea()
        self.controlScrollArea.setWidgetResizable(True)
        self.controlScrollArea.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)
        self.controlScrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.controlScrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.controlScrollArea.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.controlScrollArea.setWidget(control_panel_right_widget)
        main_layout.addWidget(self.controlScrollArea, stretch=0)

        control_panel_widget = QWidget()
        control_layout = QGridLayout(control_panel_widget)
        control_layout.setContentsMargins(5, 0, 5, 0) # Tight margins [L, T, R, B]
        control_layout.setVerticalSpacing(0)   # Tight vertical spacing
        control_layout.setHorizontalSpacing(5) # Standard horizontal spacing


        self.pos_label = QLabel('P:')
        self.pos_min_input = QLineEdit(self)
        self.pos_max_input = QLineEdit(self)
        self.pos_button = QPushButton('Set P')

        self.pos_min_input.setFixedWidth(80)
        self.pos_max_input.setFixedWidth(80)

        control_layout.addWidget(self.pos_label, 0, 0, Qt.AlignmentFlag.AlignRight) # Col 0
        control_layout.addWidget(self.pos_min_input, 0, 1)                         # Col 1
        control_layout.addWidget(self.pos_max_input, 0, 2)                         # Col 2
        control_layout.addWidget(self.pos_button, 0, 3)                            # Col 3

        self.saveFitsButton = QPushButton("Save as FITS")
        self.saveFitsButton.clicked.connect(self.save_fits)
        control_layout.addWidget(self.saveFitsButton, 0, 5)


        self.vel_label = QLabel('V:')
        self.vel_min_input = QLineEdit(self)
        self.vel_max_input = QLineEdit(self)
        self.vel_button = QPushButton('Set V')

        self.vel_min_input.setFixedWidth(80)
        self.vel_max_input.setFixedWidth(80)

        control_layout.addWidget(self.vel_label, 1, 0, Qt.AlignmentFlag.AlignRight) # Col 0
        control_layout.addWidget(self.vel_min_input, 1, 1)                         # Col 1
        control_layout.addWidget(self.vel_max_input, 1, 2)                         # Col 2
        control_layout.addWidget(self.vel_button, 1, 3)                            # Col 3

        self.colorScaleButton = QPushButton("Color Scale")
        self.colorScaleButton.clicked.connect(self.open_color_settings)
        control_layout.addWidget(self.colorScaleButton, 1, 5)


        control_layout.setColumnStretch(4, 1) # Spacer column between input group (col 3) and buttons (col 5)
        central_layout.addWidget(control_panel_widget)


        # --- Center: PV Diagram Canvas ---
        self.pv_fig, self.pv_ax = plt.subplots()

        config = self.fits_viewer.config_manager.config
        tick_pos = config.get('default_ticks_position', 'btlr')
        fig_bg_color = config.get('fig_background_color', '#ececec')
        ax_bg_color = config.get('ax_background_color', 'white')
        bad_color = config.get('bad_color', 'black')


        lc = config.get('click_linecolor', 'cyan')
        lw = config.get('click_linewidth', 0.25)
        self.h_cursor_line = self.pv_ax.axhline(y=0, color=lc, lw=lw, visible=False)
        self.v_cursor_line = self.pv_ax.axvline(x=0, color=lc, lw=lw, visible=False)

        self.pv_v_cursor_line = self.pv_ax.axvline(x=0, color=lc, lw=lw, visible=False)
        self.pv_h_cursor_line = self.pv_ax.axhline(y=0, color=lc, lw=lw, visible=False)


        self.pv_fig.subplots_adjust(left=0.15, right=1, bottom=0.1, top=0.95)

        self.pv_fig.set_facecolor(fig_bg_color)
        self.pv_ax.set_facecolor(ax_bg_color)
        self.pv_ax.tick_params(
            axis='both', which='both',
            bottom='b' in tick_pos, top='t' in tick_pos, left='l' in tick_pos, right='r' in tick_pos,
            direction=config.get('tick_direction', 'out')
        )

        self.pv_canvas = FigureCanvas(self.pv_fig)
        central_layout.addWidget(self.pv_canvas)
        n_vel = self.data.shape[0]
        init_pv = np.zeros((n_vel, 10))

        cmap = plt.get_cmap(self.color_pattern)
        cmap.set_bad(color=bad_color)

        self.pv_im = self.pv_ax.imshow(init_pv, aspect='auto', origin='lower', interpolation='none', cmap=cmap, vmin=self.min_val, vmax=self.max_val, rasterized=True)
        self.im = self.pv_im
        self.canvas = self.pv_canvas
        self.pv_ax.set_xlabel('Position')

        self.pv_ax.xaxis.set_minor_locator(AutoMinorLocator(config.get('x_mtick_freq', 5)))
        self.pv_ax.yaxis.set_minor_locator(AutoMinorLocator(config.get('z_mtick_freq', 5)))

        #self.pv_ax.set_title('Position-Velocity Diagram')
        self.pv_fig.colorbar(self.pv_im, ax=self.pv_ax)
        self.pv_canvas.draw()

        self.pv_canvas.mpl_connect('button_press_event', self.on_pv_press)
        self.pv_canvas.mpl_connect('motion_notify_event', self.on_pv_motion)

        self.cid_press = self.fits_canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_key_press = self.fits_canvas.mpl_connect('key_press_event', self.on_key_press)
        self.cid_motion = self.fits_canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cid_release = self.fits_canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_key_release = self.fits_canvas.mpl_connect('key_release_event', self.on_key_release)

        self.toolbar = PVNavigationToolbar(self.pv_canvas, self, "pv",
                                           self.pv_ax, color_mode=None, default_image_name=self.fits_viewer.filename)
        central_layout.addWidget(self.toolbar)

        # --- Right Panel: Slit Controls ---
        arrow_control_group = QGroupBox("Slit Geometry")
        arrow_control_layout = QFormLayout(arrow_control_group)
        arrow_control_layout.setVerticalSpacing(4)
        arrow_control_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.pathTypeCombo = QComboBox()
        self.pathTypeCombo.addItem("Straight", "straight")
        self.pathTypeCombo.addItem("Polyline", "polyline")
        polyline_item = self.pathTypeCombo.model().item(self.pathTypeCombo.count() - 1)
        if polyline_item is not None:
            polyline_item.setToolTip(
                "Draw a multi-segment polyline PV path: click to add nodes, "
                "double-click or Enter to finish."
            )
        self.pathTypeCombo.addItem("Ellipse", "ellipse")
        ellipse_item = self.pathTypeCombo.model().item(self.pathTypeCombo.count() - 1)
        if ellipse_item is not None:
            ellipse_item.setToolTip("Draw and edit a full ellipse or elliptical arc PV path.")
        self.pathTypeCombo.currentIndexChanged.connect(self._on_path_type_changed)
        stable_label_texts = (
            "Path Type:", "Geometry Input:", "Position Origin:",
            "Straight Length:", "P Axis Unit:", "X Axis:",
        )
        stable_label_width = max(
            self.fontMetrics().horizontalAdvance(text) for text in stable_label_texts
        ) + 8
        self.pathTypeLabel = QLabel("Path Type:")
        self.pathTypeLabel.setFixedWidth(stable_label_width)
        self.pathTypeLabel.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        arrow_control_layout.addRow(self.pathTypeLabel, self.pathTypeCombo)

        self.xAxisModeCombo = QComboBox()
        self.xAxisModeCombo.addItem("Position", PV_X_AXIS_POSITION)
        self.xAxisModeCombo.addItem("Phi (ellipse)", PV_X_AXIS_PHI)
        phi_item = self.xAxisModeCombo.model().item(1)
        if phi_item is not None:
            phi_item.setEnabled(False)
            phi_item.setToolTip("Available when Path Type is Ellipse.")
        self.xAxisModeCombo.currentIndexChanged.connect(self._on_x_axis_mode_changed)
        self.xAxisModeLabel = QLabel("X Axis:")
        self.xAxisModeLabel.setFixedWidth(stable_label_width)
        self.xAxisModeLabel.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        arrow_control_layout.addRow(self.xAxisModeLabel, self.xAxisModeCombo)

        self.activePathCombo = QComboBox()
        self.activePathCombo.setToolTip("Select which PV path is active for editing and display.")
        self.activePathCombo.currentIndexChanged.connect(self._on_active_path_combo_changed)

        self.geometryInputCombo = QComboBox()
        self.geometryInputCombo.addItem("Endpoints", "endpoints")
        self.geometryInputCombo.addItem("Center + Length + PA", "center")
        self.geometryInputCombo.currentIndexChanged.connect(self._on_geometry_input_mode_changed)
        self.geometryInputLabel = QLabel("Geometry Input:")
        arrow_control_layout.addRow(self.geometryInputLabel, self.geometryInputCombo)

        self.endpointGeometryWidget = QWidget()
        endpoint_geometry_layout = QFormLayout(self.endpointGeometryWidget)
        endpoint_geometry_layout.setContentsMargins(0, 0, 0, 0)
        endpoint_geometry_layout.setVerticalSpacing(4)

        # Start coordinates
        spinbox_width = 90
        self.startXSpin = QDoubleSpinBox()
        self.startXSpin.setFixedWidth(spinbox_width)
        self.startXSpin.setRange(0, data_shape_x); self.startXSpin.setDecimals(2)
        self.startLonEdit = QLineEdit()
        start_x_layout = QHBoxLayout(); start_x_layout.addWidget(self.startXSpin); start_x_layout.addWidget(self.startLonEdit)
        endpoint_geometry_layout.addRow("Start X:", start_x_layout)

        self.startYSpin = QDoubleSpinBox()
        self.startYSpin.setFixedWidth(spinbox_width)
        self.startYSpin.setRange(0, data_shape_y); self.startYSpin.setDecimals(2)
        self.startLatEdit = QLineEdit()
        start_y_layout = QHBoxLayout(); start_y_layout.addWidget(self.startYSpin); start_y_layout.addWidget(self.startLatEdit)
        endpoint_geometry_layout.addRow("Start Y:", start_y_layout)

        # End coordinates
        self.endXSpin = QDoubleSpinBox()
        self.endXSpin.setFixedWidth(spinbox_width)
        self.endXSpin.setRange(0, data_shape_x); self.endXSpin.setDecimals(2)
        self.endLonEdit = QLineEdit()
        end_x_layout = QHBoxLayout(); end_x_layout.addWidget(self.endXSpin); end_x_layout.addWidget(self.endLonEdit)
        endpoint_geometry_layout.addRow("End X:", end_x_layout)

        self.endYSpin = QDoubleSpinBox()
        self.endYSpin.setFixedWidth(spinbox_width)
        self.endYSpin.setRange(0, data_shape_y); self.endYSpin.setDecimals(2)
        self.endLatEdit = QLineEdit()
        end_y_layout = QHBoxLayout(); end_y_layout.addWidget(self.endYSpin); end_y_layout.addWidget(self.endLatEdit)
        endpoint_geometry_layout.addRow("End Y:", end_y_layout)
        arrow_control_layout.addRow(self.endpointGeometryWidget)

        self.centerGeometryWidget = QWidget()
        center_geometry_layout = QFormLayout(self.centerGeometryWidget)
        center_geometry_layout.setContentsMargins(0, 0, 0, 0)
        center_geometry_layout.setVerticalSpacing(4)

        self.centerXSpin = QDoubleSpinBox()
        self.centerXSpin.setFixedWidth(spinbox_width)
        self.centerXSpin.setRange(0, data_shape_x); self.centerXSpin.setDecimals(2)
        self.centerLonEdit = QLineEdit()
        center_x_layout = QHBoxLayout(); center_x_layout.addWidget(self.centerXSpin); center_x_layout.addWidget(self.centerLonEdit)
        center_geometry_layout.addRow("Center X:", center_x_layout)

        self.centerYSpin = QDoubleSpinBox()
        self.centerYSpin.setFixedWidth(spinbox_width)
        self.centerYSpin.setRange(0, data_shape_y); self.centerYSpin.setDecimals(2)
        self.centerLatEdit = QLineEdit()
        self.useCursorCenterButton = QPushButton("Set Center from Cursor")
        self.useCursorCenterButton.setToolTip("Set the slit center to the current cursor position in the main image.")
        center_y_layout = QHBoxLayout()
        center_y_layout.addWidget(self.centerYSpin)
        center_y_layout.addWidget(self.centerLatEdit)
        center_geometry_layout.addRow("Center Y:", center_y_layout)
        center_button_layout = QHBoxLayout()
        center_button_layout.addWidget(self.useCursorCenterButton)
        center_button_layout.addStretch()
        center_geometry_layout.addRow("", center_button_layout)
        arrow_control_layout.addRow(self.centerGeometryWidget)

        self.ellipseGeometryWidget = QWidget()
        ellipse_geometry_layout = QFormLayout(self.ellipseGeometryWidget)
        ellipse_geometry_layout.setContentsMargins(0, 0, 0, 0)
        ellipse_geometry_layout.setVerticalSpacing(4)

        self.ellipseCenterXSpin = QDoubleSpinBox()
        self.ellipseCenterXSpin.setFixedWidth(spinbox_width)
        self.ellipseCenterXSpin.setRange(-99999.0, 99999.0)
        self.ellipseCenterXSpin.setDecimals(2)
        self.ellipseCenterLonEdit = QLineEdit()
        ellipse_center_x_layout = QHBoxLayout()
        ellipse_center_x_layout.addWidget(self.ellipseCenterXSpin)
        ellipse_center_x_layout.addWidget(self.ellipseCenterLonEdit)
        ellipse_geometry_layout.addRow("Center X:", ellipse_center_x_layout)

        self.ellipseCenterYSpin = QDoubleSpinBox()
        self.ellipseCenterYSpin.setFixedWidth(spinbox_width)
        self.ellipseCenterYSpin.setRange(-99999.0, 99999.0)
        self.ellipseCenterYSpin.setDecimals(2)
        self.ellipseCenterLatEdit = QLineEdit()
        ellipse_center_y_layout = QHBoxLayout()
        ellipse_center_y_layout.addWidget(self.ellipseCenterYSpin)
        ellipse_center_y_layout.addWidget(self.ellipseCenterLatEdit)
        ellipse_geometry_layout.addRow("Center Y:", ellipse_center_y_layout)

        self.ellipseMajorSpin = QDoubleSpinBox()
        self.ellipseMajorSpin.setFixedWidth(spinbox_width)
        self.ellipseMajorSpin.setRange(0.0, 99999.0)
        self.ellipseMajorSpin.setDecimals(2)
        self.ellipseMinorSpin = QDoubleSpinBox()
        self.ellipseMinorSpin.setFixedWidth(spinbox_width)
        self.ellipseMinorSpin.setRange(0.0, 99999.0)
        self.ellipseMinorSpin.setDecimals(2)
        self.ellipseAxisUnitCombo = QComboBox()
        self.ellipseAxisUnitCombo.addItems(['pixel', 'deg', 'arcmin', 'arcsec'])
        ellipse_axes_layout = QHBoxLayout()
        ellipse_axes_layout.addWidget(self.ellipseMajorSpin)
        ellipse_axes_layout.addWidget(self.ellipseMinorSpin)
        ellipse_axes_layout.addWidget(self.ellipseAxisUnitCombo)
        ellipse_axes_layout.addStretch()
        ellipse_geometry_layout.addRow("Semi a/b:", ellipse_axes_layout)

        if self.pixel_scale_deg is None:
            for i in range(1, 4):
                self.ellipseAxisUnitCombo.model().item(i).setEnabled(False)

        self.ellipsePASpin = QDoubleSpinBox()
        self.ellipsePASpin.setFixedWidth(spinbox_width)
        self.ellipsePASpin.setRange(-180.0, 180.0)
        self.ellipsePASpin.setDecimals(1)
        self.ellipsePASpin.setToolTip("Ellipse major-axis angle in display pixel coordinates, measured from +X.")
        ellipse_geometry_layout.addRow("Ellipse PA (°):", self.ellipsePASpin)

        self.ellipseSpanModeCombo = QComboBox()
        self.ellipseSpanModeCombo.addItem("Full", "full")
        self.ellipseSpanModeCombo.addItem("Arc", "arc")
        self.ellipseSpanModeCombo.setToolTip("Use the full ellipse or restrict extraction to Phi0..Phi1.")
        self.ellipseSpanModeCombo.currentIndexChanged.connect(self._on_ellipse_span_mode_changed)
        ellipse_geometry_layout.addRow("Span:", self.ellipseSpanModeCombo)

        self.ellipseStartPhiSpin = QDoubleSpinBox()
        self.ellipseStartPhiSpin.setFixedWidth(spinbox_width)
        self.ellipseStartPhiSpin.setRange(0.0, 360.0)
        self.ellipseStartPhiSpin.setDecimals(1)
        self.ellipseStartPhiSpin.setToolTip("Starting phase on the ellipse, measured from the major-axis direction after PA rotation.")
        ellipse_geometry_layout.addRow("Start Phi0 (°):", self.ellipseStartPhiSpin)
        self.ellipseEndPhiLabel = QLabel("End Phi1 (°):")
        self.ellipseEndPhiSpin = QDoubleSpinBox()
        self.ellipseEndPhiSpin.setFixedWidth(spinbox_width)
        self.ellipseEndPhiSpin.setRange(0.0, 360.0)
        self.ellipseEndPhiSpin.setDecimals(1)
        self.ellipseEndPhiSpin.setValue(180.0)
        self.ellipseEndPhiSpin.setToolTip("Ending phase for Arc spans, measured from the major-axis direction after PA rotation.")
        ellipse_geometry_layout.addRow(self.ellipseEndPhiLabel, self.ellipseEndPhiSpin)
        arrow_control_layout.addRow(self.ellipseGeometryWidget)

        # Angle
        self.rotationAngleSpin = QDoubleSpinBox()
        self.rotationAngleSpin.setRange(-180, 180)
        self.rotationAngleSpin.setDecimals(1)
        self.rotationAngleSpin.setSingleStep(1)
        self.rotationAngleSpin.setToolTip("Straight-slit angle in display pixel coordinates, measured from +X.")
        self.rotationAngleLabel = QLabel("Straight Angle (°):")
        arrow_control_layout.addRow(self.rotationAngleLabel, self.rotationAngleSpin)

        # Length
        length_layout = QHBoxLayout()
        length_layout.setContentsMargins(0, 0, 0, 0)
        self.arrowLengthSpin = QDoubleSpinBox()
        self.arrowLengthSpin.setRange(0.0, 99999.0)
        self.arrowLengthSpin.setValue(0)
        self.arrowLengthSpin.setFixedWidth(120)

        self.arrowLengthSpin.setDecimals(0)
        self.arrowLengthSpin.setSingleStep(1.0)

        length_layout.addWidget(self.arrowLengthSpin)

        self.lengthUnitCombo = QComboBox()
        self.lengthUnitCombo.addItems(['pixel', 'deg', 'arcmin', 'arcsec'])
        self.lengthUnitCombo.setCurrentText(self.length_unit)
        length_layout.addWidget(self.lengthUnitCombo)
        length_layout.addStretch()
        self.lengthWidget = QWidget()
        self.lengthWidget.setLayout(length_layout)
        self.lengthLabel = QLabel("Straight Length:")
        arrow_control_layout.addRow(self.lengthLabel, self.lengthWidget)

        if self.pixel_scale_deg is None:
            for i in range(1, 4): # deg, arcmin, arcsec
                self.lengthUnitCombo.model().item(i).setEnabled(False)

        self.sliceWidthSpin = QDoubleSpinBox()
        self.sliceWidthSpin.setRange(0, 99)
        self.sliceWidthSpin.setDecimals(2)
        self.sliceWidthSpin.setValue(self.slice_width)
        arrow_control_layout.addRow("Slice Width:", self.sliceWidthSpin)

        # Polyline-only curve controls. "Curve" picks straight vs an
        # interpolating spline; "Smoothness" sets its roundness. Both are hidden
        # unless Path Type = Polyline (see _sync_path_type_controls).
        self.polylineCurveCombo = QComboBox()
        self.polylineCurveCombo.addItem("Straight (none)", PV_SPLINE_NONE)
        self.polylineCurveCombo.addItem("Catmull-Rom (through nodes)", PV_SPLINE_CATMULL_ROM)
        self.polylineCurveCombo.addItem("B-spline (approximate)", PV_SPLINE_BSPLINE)
        self.polylineCurveCombo.setToolTip(
            "Straight: connect the nodes with line segments.\n"
            "Catmull-Rom: a smooth curve that passes through every node.\n"
            "B-spline: a smoother curve that approximates (smooths) the nodes "
            "instead of passing through them; it stays inside the node hull, so "
            "it never overshoots."
        )
        self.polylineCurveCombo.currentIndexChanged.connect(self._on_polyline_curve_type_changed)
        self.polylineCurveLabel = QLabel("Curve:")
        self.polylineCurveLabel.setFixedWidth(stable_label_width)
        self.polylineCurveLabel.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        arrow_control_layout.addRow(self.polylineCurveLabel, self.polylineCurveCombo)

        smoothness_layout = QHBoxLayout()
        smoothness_layout.setContentsMargins(0, 0, 0, 0)
        self.polylineSmoothnessSlider = QSlider(Qt.Orientation.Horizontal)
        self.polylineSmoothnessSlider.setRange(0, 100)
        self.polylineSmoothnessSlider.setValue(int(round(self.polyline_smoothness * 100)))
        self.polylineSmoothnessSlider.setToolTip(
            "Spline roundness: 0% = straight chords, 100% = full curve. "
            "The curve passes through every node at any value."
        )
        self.polylineSmoothnessSlider.valueChanged.connect(self._on_polyline_smoothness_changed)
        self.polylineSmoothnessSlider.sliderReleased.connect(self._on_polyline_smoothness_committed)
        smoothness_layout.addWidget(self.polylineSmoothnessSlider)
        self.polylineSmoothnessValueLabel = QLabel(f"{int(round(self.polyline_smoothness * 100))}%")
        self.polylineSmoothnessValueLabel.setFixedWidth(36)
        self.polylineSmoothnessValueLabel.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        smoothness_layout.addWidget(self.polylineSmoothnessValueLabel)
        self.polylineSmoothnessWidget = QWidget()
        self.polylineSmoothnessWidget.setLayout(smoothness_layout)
        self.polylineSmoothnessLabel = QLabel("Smoothness:")
        self.polylineSmoothnessLabel.setFixedWidth(stable_label_width)
        self.polylineSmoothnessLabel.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        arrow_control_layout.addRow(self.polylineSmoothnessLabel, self.polylineSmoothnessWidget)

        # Read-only path length: smoothing changes the arc length (and therefore
        # the PV position-axis scale), so keep it visible for reproducibility.
        self.polylinePathLengthValueLabel = QLabel("-")
        self.polylinePathLengthLabel = QLabel("Path length:")
        self.polylinePathLengthLabel.setFixedWidth(stable_label_width)
        self.polylinePathLengthLabel.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        arrow_control_layout.addRow(self.polylinePathLengthLabel, self.polylinePathLengthValueLabel)

        # Polyline-only toggle for the draggable node-dot markers. Lives in the
        # main panel (not the Options panel) and is shown only in polyline mode
        # (see _sync_path_type_controls); straight/ellipse markers are handled by
        # the Indicators checkboxes / always-on ellipse handles.
        self.nodeMarkerCheck = QCheckBox("Node Markers")
        self.nodeMarkerCheck.setChecked(self.show_node_markers)
        self.nodeMarkerCheck.setToolTip(
            "Show the polyline node-point markers. Turn off for clean paper figures."
        )
        arrow_control_layout.addRow(self.nodeMarkerCheck)

        position_origin_layout = QHBoxLayout()
        self.positionOriginCombo = QComboBox()
        self.positionOriginCombo.addItem("Start", POSITION_ORIGIN_START)
        self.positionOriginCombo.addItem("Center", POSITION_ORIGIN_CENTER)
        self.positionOriginCombo.currentIndexChanged.connect(self._on_position_origin_changed)
        position_origin_layout.addWidget(self.positionOriginCombo)
        self.reverseDirectionButton = QPushButton("Reverse")
        self.reverseDirectionButton.setToolTip("Swap the slit start/end points and reverse the positive position direction.")
        self.reverseDirectionButton.clicked.connect(self.reverse_direction)
        position_origin_layout.addWidget(self.reverseDirectionButton)
        position_origin_layout.addStretch()
        self.positionOriginWidget = QWidget()
        self.positionOriginWidget.setLayout(position_origin_layout)
        self.positionOriginLabel = QLabel("Position Origin:")
        arrow_control_layout.addRow(self.positionOriginLabel, self.positionOriginWidget)

        self.optionsToggleButton = QPushButton("Show Options")
        self.optionsToggleButton.setCheckable(True)
        self.optionsToggleButton.toggled.connect(self._on_options_toggled)
        arrow_control_layout.addRow(self.optionsToggleButton)

        self.optionsPanel = QWidget()
        options_layout = QFormLayout(self.optionsPanel)
        options_layout.setContentsMargins(0, 0, 0, 0)
        options_layout.setVerticalSpacing(4)

        options_layout.addRow("Active Path:", self.activePathCombo)

        indicators_layout = QHBoxLayout()
        self.startIndicatorCheck = QCheckBox("Start")
        self.centerIndicatorCheck = QCheckBox("Center")
        self.endIndicatorCheck = QCheckBox("End")
        self.startIndicatorCheck.setChecked(self.indicator_positions["start"])
        self.centerIndicatorCheck.setChecked(self.indicator_positions["center"])
        self.endIndicatorCheck.setChecked(self.indicator_positions["end"])
        self.startIndicatorCheck.setToolTip(
            "Straight: tick at the start endpoint. Ellipse: tick at Phi0 (arc start)."
        )
        self.centerIndicatorCheck.setToolTip(
            "Straight: tick at the slit center. Ellipse: cross at the ellipse center."
        )
        self.endIndicatorCheck.setToolTip(
            "Straight: tick at the end endpoint. Ellipse: tick at Phi1 (arc end; "
            "overlaps Phi0 on a closed ellipse)."
        )
        indicators_layout.addWidget(self.startIndicatorCheck)
        indicators_layout.addWidget(self.centerIndicatorCheck)
        indicators_layout.addWidget(self.endIndicatorCheck)
        options_layout.addRow("Indicators:", indicators_layout)

        self.interpGroup = QButtonGroup(self)
        self.bilinearRadio = QRadioButton("Bilinear")
        self.gaussianRadio = QRadioButton("Gaussian")
        self.bilinearRadio.setChecked(True)
        self.interpGroup.addButton(self.bilinearRadio, 0)
        self.interpGroup.addButton(self.gaussianRadio, 1)
        self.interpGroup.buttonClicked.connect(self._on_weight_mode_changed)
        interp_layout = QHBoxLayout()
        interp_layout.addWidget(self.bilinearRadio)
        interp_layout.addWidget(self.gaussianRadio)
        options_layout.addRow("Interpolation:", interp_layout)

        self.sampleSpacingSpin = QDoubleSpinBox()
        self.sampleSpacingSpin.setRange(0.25, 100.0)
        self.sampleSpacingSpin.setDecimals(2)
        self.sampleSpacingSpin.setSingleStep(0.25)
        self.sampleSpacingSpin.setValue(DEFAULT_SAMPLE_SPACING_PIX)
        self.sampleSpacingSpin.setSuffix(" pix")
        self.sampleSpacingSpin.setToolTip("Average sampling step along the PV path in image pixels.")
        self.sampleSpacingSpin.setFixedWidth(110)
        self.sampleSpacingSpin.valueChanged.connect(self._on_sample_spacing_changed)
        self.sampleSpacingSpin.editingFinished.connect(self._on_sample_spacing_editing_finished)
        options_layout.addRow("Sampling Step:", self.sampleSpacingSpin)

        self.arrowColorCombo = QComboBox()
        colors = ["blue", "red", "green", "cyan", "magenta", "black", "white", "gray", "orange", "purple", "yellow", "olive"]
        self.arrowColorCombo.addItems(colors)
        self.arrowColorCombo.setCurrentText(self.pvarrow_color)
        self.arrowColorCombo.currentTextChanged.connect(self._on_slit_color_changed)
        options_layout.addRow("Slit Color:", self.arrowColorCombo)

        self.markerColorCombo = QComboBox()
        markerColors = ["None", "blue", "red", "green", "cyan", "magenta", "black", "white", "gray", "orange", "purple", "yellow", "olive"]
        self.markerColorCombo.addItems(markerColors)
        self.markerColorCombo.setCurrentText(self.pvmarker_color)
        self.markerColorCombo.currentTextChanged.connect(lambda new_color: setattr(self, "pvmarker_color", new_color))
        #arrow_control_layout.addRow("Marker Color:", self.markerColorCombo)

        self.arrowSizeSpin = QDoubleSpinBox()
        self.arrowSizeSpin.setRange(0.1, 10.0)
        self.arrowSizeSpin.setSingleStep(0.1)
        self.arrowSizeSpin.setValue(self.arrow_size)
        #self.arrowSizeSpin.valueChanged.connect(lambda new_size: setattr(self, "arrow_size", new_size))
        self.arrowSizeSpin.valueChanged.connect(self._on_arrow_width_changed)
        options_layout.addRow("Slit Line Width:", self.arrowSizeSpin)

        self.swapAxesCheck = QCheckBox("Swap PV Axes")
        self.swapAxesCheck.stateChanged.connect(self.on_swap_axes_changed)
        options_layout.addRow(self.swapAxesCheck)
        self.flipPositionAxisCheck = QCheckBox("Flip Position Axis")
        self.flipPositionAxisCheck.setToolTip(
            "Reverse the displayed PV position axis without changing the slit "
            "geometry or FITS export data."
        )
        self.flipPositionAxisCheck.stateChanged.connect(self._on_position_axis_flip_changed)
        options_layout.addRow(self.flipPositionAxisCheck)
        arrow_control_layout.addRow(self.optionsPanel)
        self.optionsPanel.setVisible(False)

        self.autoUpdateCheck = QCheckBox("Auto Update")
        self.autoUpdateCheck.setChecked(False)
        self.autoUpdateCheck.stateChanged.connect(self._on_auto_update_changed)
        arrow_control_layout.addRow(self.autoUpdateCheck)

        action_buttons_widget = QWidget()
        action_buttons_layout = QVBoxLayout(action_buttons_widget)
        action_buttons_layout.setContentsMargins(0, 0, 0, 0)
        action_buttons_layout.setSpacing(3)

        arrow_button_layout = QHBoxLayout()
        arrow_button_layout.setContentsMargins(0, 0, 0, 0)
        arrow_button_layout.setSpacing(4)
        self.applyArrowButton = QPushButton("Apply")
        self.applyArrowButton.clicked.connect(self.apply_controls)
        self.clearArrowButton = QPushButton("Clear")
        self.clearArrowButton.clicked.connect(self.clear_arrow)
        arrow_button_layout.addWidget(self.applyArrowButton)
        arrow_button_layout.addWidget(self.clearArrowButton)

        path_button_layout = QHBoxLayout()
        path_button_layout.setContentsMargins(0, 0, 0, 0)
        path_button_layout.setSpacing(4)
        self.savePathButton = QPushButton("Save Path...")
        self.savePathButton.clicked.connect(self.save_path_recipe)
        self.loadPathButton = QPushButton("Load Path...")
        self.loadPathButton.clicked.connect(self.load_path_recipe)
        path_button_layout.addWidget(self.savePathButton)
        path_button_layout.addWidget(self.loadPathButton)

        copy_button_layout = QHBoxLayout()
        copy_button_layout.setContentsMargins(0, 0, 0, 0)
        copy_button_layout.setSpacing(4)
        self.copySlitButton = QPushButton("Copy Slit to Window")
        self.copySlitButton.setToolTip(
            "Drop a static copy of this slit onto an open xy result window "
            "(e.g. an integrated/moment map) for figures. The copy stays after "
            "this PV panel is closed; click it on that window and press Delete "
            "to remove it."
        )
        self._copy_slit_button_menu = QMenu(self.copySlitButton)
        self._copy_slit_button_menu.aboutToShow.connect(
            lambda: self._populate_copy_slit_menu(self._copy_slit_button_menu)
        )
        self.copySlitButton.setMenu(self._copy_slit_button_menu)
        copy_button_layout.addWidget(self.copySlitButton)

        action_buttons_layout.addLayout(arrow_button_layout)
        action_buttons_layout.addLayout(path_button_layout)
        action_buttons_layout.addLayout(copy_button_layout)
        arrow_control_layout.addRow(action_buttons_widget)

        control_layout_right.addWidget(arrow_control_group)
        control_layout_right.addStretch(1)

        self._endpoint_input_widgets = [
            self.startXSpin, self.startYSpin, self.startLonEdit, self.startLatEdit,
            self.endXSpin, self.endYSpin, self.endLonEdit, self.endLatEdit,
        ]
        self._center_input_widgets = [
            self.centerXSpin, self.centerYSpin, self.centerLonEdit, self.centerLatEdit,
            self.useCursorCenterButton,
        ]
        self._ellipse_input_widgets = [
            self.ellipseCenterXSpin, self.ellipseCenterYSpin,
            self.ellipseCenterLonEdit, self.ellipseCenterLatEdit,
            self.ellipseMajorSpin, self.ellipseMinorSpin,
            self.ellipseAxisUnitCombo, self.ellipsePASpin, self.ellipseSpanModeCombo,
            self.ellipseStartPhiSpin,
            self.ellipseEndPhiLabel, self.ellipseEndPhiSpin,
        ]

        # Connect signals for synchronization
        self.startXSpin.valueChanged.connect(self.update_arrow_from_gui)
        self.startYSpin.valueChanged.connect(self.update_arrow_from_gui)
        self.endXSpin.valueChanged.connect(self.update_arrow_from_gui)
        self.endYSpin.valueChanged.connect(self.update_arrow_from_gui)
        self.centerXSpin.valueChanged.connect(self.update_arrow_from_center)
        self.centerYSpin.valueChanged.connect(self.update_arrow_from_center)

        self.startLonEdit.editingFinished.connect(self._update_pixel_from_world)
        self.startLatEdit.editingFinished.connect(self._update_pixel_from_world)
        self.endLonEdit.editingFinished.connect(self._update_pixel_from_world)
        self.endLatEdit.editingFinished.connect(self._update_pixel_from_world)
        self.centerLonEdit.editingFinished.connect(self._update_center_pixel_from_world)
        self.centerLatEdit.editingFinished.connect(self._update_center_pixel_from_world)
        self.ellipseCenterLonEdit.editingFinished.connect(self._update_ellipse_pixel_from_world)
        self.ellipseCenterLatEdit.editingFinished.connect(self._update_ellipse_pixel_from_world)
        self.useCursorCenterButton.clicked.connect(self.use_cursor_as_center)
        for spin in (
            self.ellipseCenterXSpin, self.ellipseCenterYSpin,
            self.ellipseMajorSpin, self.ellipseMinorSpin,
            self.ellipsePASpin, self.ellipseStartPhiSpin, self.ellipseEndPhiSpin,
        ):
            spin.valueChanged.connect(self._update_ellipse_from_controls)
            # Pressing Enter commits and forces a PV redraw, mirroring straight.
            spin.editingFinished.connect(self.finalize_interactive_update)
        self.ellipseAxisUnitCombo.currentTextChanged.connect(self._on_ellipse_axis_unit_changed)

        self.rotationAngleSpin.valueChanged.connect(self.update_arrow_from_rotation)
        self.arrowLengthSpin.valueChanged.connect(self.update_arrow_from_length)
        self.lengthUnitCombo.currentTextChanged.connect(self._on_unit_changed)
        self.sliceWidthSpin.valueChanged.connect(self.update_arrow_from_gui)

        # editingFinishedシグナルは、インタラクティブモードを終了し、最終的な描画を行う
        self.startXSpin.editingFinished.connect(self.finalize_interactive_update)
        self.startYSpin.editingFinished.connect(self.finalize_interactive_update)
        self.endXSpin.editingFinished.connect(self.finalize_interactive_update)
        self.endYSpin.editingFinished.connect(self.finalize_interactive_update)
        self.centerXSpin.editingFinished.connect(self.finalize_interactive_update)
        self.centerYSpin.editingFinished.connect(self.finalize_interactive_update)
        self.rotationAngleSpin.editingFinished.connect(self.finalize_interactive_update)
        self.arrowLengthSpin.editingFinished.connect(self.finalize_interactive_update)
        self.sliceWidthSpin.editingFinished.connect(self.finalize_interactive_update)


        self.pos_button.clicked.connect(self.set_pv_range)
        self.vel_button.clicked.connect(self.set_pv_range)
        self.pos_min_input.returnPressed.connect(self.set_pv_range)
        self.pos_max_input.returnPressed.connect(self.set_pv_range)
        self.vel_min_input.returnPressed.connect(self.set_pv_range)
        self.vel_max_input.returnPressed.connect(self.set_pv_range)

        def update_indicator_positions():
            if self._is_ellipse_mode():
                self.ellipse_indicator_positions["start"] = self.startIndicatorCheck.isChecked()
                self.ellipse_indicator_positions["center"] = self.centerIndicatorCheck.isChecked()
                self.ellipse_indicator_positions["end"] = self.endIndicatorCheck.isChecked()
                self._draw_ellipse_overlay()
                self._invalidate_main_overlay_background()
                self._request_main_overlay_redraw()
                return
            self.indicator_positions["start"] = self.startIndicatorCheck.isChecked()
            self.indicator_positions["center"] = self.centerIndicatorCheck.isChecked()
            self.indicator_positions["end"] = self.endIndicatorCheck.isChecked()
            self.update_arrow_from_gui()
        self.startIndicatorCheck.stateChanged.connect(lambda _: update_indicator_positions())
        self.centerIndicatorCheck.stateChanged.connect(lambda _: update_indicator_positions())
        self.endIndicatorCheck.stateChanged.connect(lambda _: update_indicator_positions())

        def on_node_markers_toggled():
            # Polyline-only: the checkbox is hidden in other modes. Redraw goes
            # through update_arrow_from_gui, which calls _draw_polyline_overlay.
            self.show_node_markers = self.nodeMarkerCheck.isChecked()
            self.update_arrow_from_gui()
        self.nodeMarkerCheck.stateChanged.connect(lambda _: on_node_markers_toggled())

        self._on_unit_changed(self.lengthUnitCombo.currentText())
        self._sync_path_type_controls()
        self._sync_path_list_combo()
        self._sync_sampling_controls()
        self._sync_x_axis_controls()

        # --- Blit高速化のための変数 ---
        self.background_cache = None
        self.is_interactive_mode = False

        # Slit (PV path) undo/redo. Built after the controls exist because the
        # initial baseline snapshot reads the current widget state.
        self._build_slit_undo_menu()
        self._init_slit_undo_state()

        # Most sessions begin by placing a slit on the main image. Keep this
        # window compact until a usable path is committed; the PV range, canvas,
        # and toolbar are revealed together exactly once per window lifetime.
        self._pv_result_revealed = False
        self._pv_result_full_size_hint = self.sizeHint()
        self.pvResultPanel.setVisible(False)
        # QMainWindow may retain the width calculated while the result pane was
        # still visible. Re-apply the compact size hint so the initial window is
        # only as wide as the Slit Geometry scroll panel (plus layout margins).
        compact_layout = self.centralWidget().layout()
        compact_layout.invalidate()
        compact_layout.activate()
        compact_hint = self.centralWidget().sizeHint()
        minimum_hint = self.minimumSizeHint()
        compact_width = max(
            compact_hint.width(),
            minimum_hint.width(),
        )
        compact_height = max(
            self._pv_result_full_size_hint.height(),
            minimum_hint.height(),
        )
        screen = self.screen()
        if screen is not None:
            available = screen.availableGeometry()
            # Qt will not honor a geometry smaller than minimumSizeHint().
            # Re-apply that constraint after the screen clamp so our target
            # matches the geometry Qt can actually set on small displays.
            compact_width = max(
                min(compact_width, available.width()),
                minimum_hint.width(),
            )
            compact_height = max(
                min(compact_height, available.height()),
                minimum_hint.height(),
            )
        self.resize(compact_width, compact_height)
        self._pv_compact_positioned = False

    def _position_compact_panel_next_to_main(self):
        """Place the initial Slit Geometry panel beside the main image window."""
        if (
            not self.isVisible()
            or getattr(self, "_pv_result_revealed", False)
            or getattr(self, "_pv_compact_positioned", False)
        ):
            return

        main = getattr(self, "fits_viewer", None)
        if main is None:
            return
        try:
            main_geometry = main.frameGeometry()
        except Exception:
            return

        screen = None
        try:
            screen = main.screen()
        except Exception:
            pass
        if screen is None:
            screen = self.screen()
        if screen is None:
            return

        available = screen.availableGeometry()
        gap = 8
        right_x = main_geometry.right() + gap + 1
        left_x = main_geometry.left() - gap - self.width()
        if right_x + self.width() - 1 <= available.right():
            x = right_x
        elif left_x >= available.left():
            x = left_x
        else:
            x = max(available.left(), available.right() - self.width() + 1)
        y = max(
            available.top(),
            min(main_geometry.top(), available.bottom() - self.height() + 1),
        )
        self.move(x, y)
        self._pv_compact_positioned = True

    def _reveal_pv_result(self):
        """Reveal the PV result pane once and expand the window around its right edge."""
        panel = getattr(self, "pvResultPanel", None)
        if panel is None or getattr(self, "_pv_result_revealed", False):
            return False

        previous_geometry = self.geometry()
        self._pv_result_revealed = True
        if not self.isVisible():
            panel.setVisible(True)
            return True

        # Showing the result pane and deferring setGeometry() to the next event
        # loop exposes Qt's intermediate layout: the window first grows to the
        # right, then jumps back when its settings-side edge is restored. Apply
        # visibility and the final geometry in one update-suppressed operation
        # so only the settled layout is painted.
        updates_were_enabled = self.updatesEnabled()
        if updates_were_enabled:
            self.setUpdatesEnabled(False)
        try:
            panel.setVisible(True)
            self._expand_for_pv_result(previous_geometry)
        finally:
            if updates_were_enabled:
                self.setUpdatesEnabled(True)
        self.update()
        return True

    def _expand_for_pv_result(self, previous_geometry):
        """Resize a visible compact window without moving its settings-side edge."""
        if not self.isVisible() or not getattr(self, "_pv_result_revealed", False):
            return

        if getattr(self, "pvResultPanel", None) is None:
            return
        try:
            self.centralWidget().layout().activate()
        except Exception:
            pass

        full_hint = getattr(self, "_pv_result_full_size_hint", None)
        minimum_hint = self.minimumSizeHint()
        target_width = max(
            full_hint.width() if full_hint is not None else previous_geometry.width(),
            minimum_hint.width(),
        )
        target_height = max(
            full_hint.height() if full_hint is not None else previous_geometry.height(),
            minimum_hint.height(),
        )

        screen = self.screen()
        if screen is not None:
            available = screen.availableGeometry()
            target_width = max(
                min(target_width, available.width()),
                minimum_hint.width(),
            )
            target_height = max(
                min(target_height, available.height()),
                minimum_hint.height(),
            )
            right_edge = previous_geometry.right()
            x = max(
                available.left(),
                min(right_edge - target_width + 1, available.right() - target_width + 1),
            )
            y = max(
                available.top(),
                min(previous_geometry.top(), available.bottom() - target_height + 1),
            )
        else:
            x = previous_geometry.right() - target_width + 1
            y = previous_geometry.top()

        self.setGeometry(x, y, target_width, target_height)
        try:
            self.pv_canvas.draw_idle()
        except Exception:
            pass

    def _reveal_pv_result_if_ready(self):
        """Reveal only after a non-degenerate path has been committed/restored."""
        try:
            ready = self._current_path_length_px() > 1e-6
        except Exception:
            ready = False
        if ready:
            return self._reveal_pv_result()
        return False


    # ------------------------------------------------------------------
    # Slit (PV path) undo / redo
    #
    # Snapshot-based: every editing command captures the full workspace state
    # via export_workspace_state()/restore_workspace_state() -- the same
    # serialization used for Save/Load Workspace and path recipes. Undo/redo
    # walk a stack of those snapshots. The slit is drawn on the *main* canvas, so
    # while this window is open the main window routes its Undo/Redo here (see
    # MainWindow._active_slit_undo_tool); this stack is separate from the main
    # window's analysis ActionSession.
    # ------------------------------------------------------------------
    def _build_slit_undo_menu(self):
        # Key handling uses QShortcut with WidgetWithChildrenShortcut context
        # (mirroring integration.py's view-history shortcuts) so it fires when
        # the PV window itself is focused. On macOS a secondary window's native
        # menu-bar shortcuts are unreliable, so the shortcut -- not the menu
        # action -- owns the key sequence; the menu stays as a click affordance.
        self._slit_undo_shortcuts = self._create_slit_shortcuts(
            self._SLIT_UNDO_SEQUENCES, self.undo_slit
        )
        self._slit_redo_shortcuts = self._create_slit_shortcuts(
            self._SLIT_REDO_SEQUENCES, self.redo_slit
        )

        # Name the menu "Actions" (not "Edit") so macOS does not auto-inject
        # Writing Tools / Emoji entries, matching menu_bar.py's convention.
        actions_menu = self.menuBar().addMenu("Actions")
        self.undo_slit_action = QAction("Undo Slit", self)
        self.redo_slit_action = QAction("Redo Slit", self)
        self.undo_slit_action.triggered.connect(self.undo_slit)
        self.redo_slit_action.triggered.connect(self.redo_slit)
        actions_menu.addAction(self.undo_slit_action)
        actions_menu.addAction(self.redo_slit_action)
        actions_menu.addSeparator()
        self.copy_slit_menu = actions_menu.addMenu("Copy Slit to Window")
        self.copy_slit_menu.aboutToShow.connect(
            lambda: self._populate_copy_slit_menu(self.copy_slit_menu)
        )

    def _create_slit_shortcuts(self, sequences, callback):
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

    def _init_slit_undo_state(self):
        self._slit_undo_stack = []
        self._slit_redo_stack = []
        self._restoring_slit_state = False
        self._slit_undo_limit = 50
        self._slit_baseline = self._capture_slit_snapshot()
        self._update_slit_undo_actions()

    def _capture_slit_snapshot(self):
        try:
            return copy.deepcopy(self.export_workspace_state())
        except Exception:
            return None

    @staticmethod
    def _snapshot_has_geometry(state):
        if not isinstance(state, dict):
            return False
        polyline = state.get("polyline")
        has_polyline = isinstance(polyline, dict) and bool(polyline.get("vertices"))
        return bool(
            state.get("pv_paths")
            or state.get("line_pixel")
            or state.get("ellipse_geometry")
            or has_polyline
        )

    def _record_slit_change(self):
        """Commit the current slit state as a new undo step (if it changed)."""
        if getattr(self, "_restoring_slit_state", False):
            return
        if not hasattr(self, "_slit_undo_stack"):
            return  # called before undo state was initialized
        snapshot = self._capture_slit_snapshot()
        if snapshot is None or snapshot == self._slit_baseline:
            return
        if self._slit_baseline is not None:
            self._slit_undo_stack.append(self._slit_baseline)
            if len(self._slit_undo_stack) > self._slit_undo_limit:
                self._slit_undo_stack.pop(0)
        self._slit_redo_stack.clear()
        self._slit_baseline = snapshot
        self._update_slit_undo_actions()

    def _restore_slit_snapshot(self, state):
        self._restoring_slit_state = True
        try:
            if self._snapshot_has_geometry(state):
                self.restore_workspace_state(copy.deepcopy(state))
            else:
                # An empty snapshot means "no slit". restore_workspace_state
                # early-returns without clearing when there is no geometry, so
                # remove every path explicitly here.
                self._clear_inactive_path_artists()
                self.pv_path_items = []
                self.active_pv_path_id = None
                self.clear_arrow()
                self._sync_path_list_combo()
        finally:
            self._restoring_slit_state = False

    def undo_slit(self):
        if not getattr(self, "_slit_undo_stack", None):
            return
        state = self._slit_undo_stack.pop()
        if self._slit_baseline is not None:
            self._slit_redo_stack.append(self._slit_baseline)
        self._slit_baseline = state
        self._restore_slit_snapshot(state)
        self._update_slit_undo_actions()

    def redo_slit(self):
        if not getattr(self, "_slit_redo_stack", None):
            return
        state = self._slit_redo_stack.pop()
        if self._slit_baseline is not None:
            self._slit_undo_stack.append(self._slit_baseline)
        self._slit_baseline = state
        self._restore_slit_snapshot(state)
        self._update_slit_undo_actions()

    # Protocol consumed by MainWindow's Undo/Redo dispatch. While this window is
    # open, the main window routes Undo/Redo Analysis (Option+Left/Right) to the
    # slit instead of the analysis session -- see MainWindow._active_slit_undo_tool.
    def slit_can_undo(self):
        return bool(getattr(self, "_slit_undo_stack", None))

    def slit_can_redo(self):
        return bool(getattr(self, "_slit_redo_stack", None))

    def _update_slit_undo_actions(self):
        undo_action = getattr(self, "undo_slit_action", None)
        if undo_action is not None:
            undo_action.setEnabled(self.slit_can_undo())
        redo_action = getattr(self, "redo_slit_action", None)
        if redo_action is not None:
            redo_action.setEnabled(self.slit_can_redo())
        self._notify_main_window_undo_redo_state()

    def _notify_main_window_undo_redo_state(self):
        refresh = getattr(self.fits_viewer, "_refresh_undo_redo_actions", None)
        if callable(refresh):
            try:
                refresh()
            except Exception:
                pass

    def _on_unit_changed(self, new_unit):
        """
        Handle user changing the length unit, preserving the cursor's fractional position.
        """
        fractional_pos = self._get_cursor_fractional_position(unit=self.length_unit)

        self.length_unit = new_unit
        self.arrowLengthSpin.blockSignals(True)

        if new_unit == 'pixel':
            self.arrowLengthSpin.setDecimals(0)
            self.arrowLengthSpin.setSingleStep(1.0)
        else:
            if new_unit == 'arcsec': self.arrowLengthSpin.setDecimals(2)
            elif new_unit == 'arcmin': self.arrowLengthSpin.setDecimals(3)
            elif new_unit == 'deg':
                decimals = 4
                if self.pixel_scale_deg:
                    angular_width_deg = self.data.shape[-1] * self.pixel_scale_deg
                    if angular_width_deg < 0.1: decimals = 6
                    elif angular_width_deg < 1: decimals = 5
                self.arrowLengthSpin.setDecimals(decimals)

        if self._is_polyline_mode():
            new_display_length = self._current_path_length_in_unit(unit=new_unit)
            if fractional_pos is not None:
                self.last_position_coord = position_from_fraction(
                    fractional_pos,
                    new_display_length,
                    POSITION_ORIGIN_START,
                )
            self.arrowLengthSpin.blockSignals(False)
            self._update_polyline_length_label()
            self.update_pv_diagram(force_update=True)
            if self.last_position_coord is not None:
                self.update_pv_position_cursor(self.last_position_coord)
                self._update_main_window_marker(self.last_position_coord)
            return

        if self._is_ellipse_mode():
            new_display_length = self._current_path_length_in_unit(unit=new_unit)
            if (
                fractional_pos is not None
                and self._current_x_axis_mode() == PV_X_AXIS_POSITION
            ):
                self.last_position_coord = position_from_fraction(
                    fractional_pos,
                    new_display_length,
                    POSITION_ORIGIN_START,
                )
            self.arrowLengthSpin.blockSignals(False)
            self.update_pv_diagram(force_update=True)
            if self.last_position_coord is not None:
                self.update_pv_position_cursor(self.last_position_coord)
                self._update_main_window_marker(self.last_position_coord)
            return

        if self.line_start is None or self.line_end is None:
            self.arrowLengthSpin.blockSignals(False)
            return

        line_length_px = np.hypot(self.line_end[0] - self.line_start[0], self.line_end[1] - self.line_start[1])
        new_display_length = self._convert_length(line_length_px, 'pixel', new_unit)

        if new_unit == 'pixel':
            self.arrowLengthSpin.setValue(round(new_display_length))
        else:
            self.arrowLengthSpin.setSingleStep(self._get_dynamic_step(new_display_length))
            self.arrowLengthSpin.setValue(new_display_length)

        self.arrowLengthSpin.blockSignals(False)

        if fractional_pos is not None:
            self.last_position_coord = position_from_fraction(
                fractional_pos,
                new_display_length,
                self._current_position_origin(),
            )

        self.update_pv_diagram(force_update=True)

        if self.last_position_coord is not None:
            self.update_pv_position_cursor(self.last_position_coord)
            self._update_main_window_marker(self.last_position_coord)


    def _on_auto_update_changed(self, state):
        if state == Qt.CheckState.Checked.value:
            self.update_pv_diagram(force_update=True)

    def _coerce_sample_spacing_pix(self, value):
        if value is None:
            return DEFAULT_SAMPLE_SPACING_PIX
        try:
            spacing = float(value)
        except Exception:
            return DEFAULT_SAMPLE_SPACING_PIX
        if not np.isfinite(spacing) or spacing <= 0.0:
            return DEFAULT_SAMPLE_SPACING_PIX
        return max(0.25, min(100.0, spacing))

    def _current_sample_spacing_pix(self):
        try:
            return float(self.sampleSpacingSpin.value())
        except Exception:
            return self._coerce_sample_spacing_pix(getattr(self, "sample_spacing_pix", None))

    def _sync_sampling_controls(self):
        spacing = self._current_sample_spacing_pix()
        self.sample_spacing_pix = spacing
        spin = getattr(self, "sampleSpacingSpin", None)
        if spin is not None:
            spin.setEnabled(True)

    def _on_sample_spacing_changed(self):
        self.sample_spacing_pix = self._current_sample_spacing_pix()
        if self.autoUpdateCheck.isChecked():
            self.update_pv_diagram(force_update=True)

    def _on_sample_spacing_editing_finished(self):
        self.sample_spacing_pix = self._current_sample_spacing_pix()
        self.update_pv_diagram(force_update=True)

    def _on_weight_mode_changed(self, *_args):
        self.weight_mode = self.interpGroup.checkedId()
        self.update_pv_diagram(force_update=True)

    def _current_path_type(self):
        combo = getattr(self, "pathTypeCombo", None)
        if combo is not None:
            try:
                value = str(combo.currentData() or "straight")
                if value == "straight":
                    return "straight"
                if value == "polyline":
                    return "polyline"
                if value == "ellipse":
                    return "ellipse_arc" if self._ellipse_span_mode_is_arc() else "ellipse"
            except Exception:
                pass
        return self._normalize_path_type_value(getattr(self, "path_type", "straight"))

    def _ellipse_span_mode_is_arc(self):
        combo = getattr(self, "ellipseSpanModeCombo", None)
        if combo is not None:
            try:
                return str(combo.currentData() or "full") == "arc"
            except Exception:
                pass
        return self._normalize_path_type_value(getattr(self, "path_type", "straight")) == "ellipse_arc"

    def _set_ellipse_span_mode(self, arc_enabled):
        combo = getattr(self, "ellipseSpanModeCombo", None)
        if combo is None:
            return
        target = "arc" if arc_enabled else "full"
        try:
            idx = combo.findData(target)
            if idx >= 0:
                combo.blockSignals(True)
                combo.setCurrentIndex(idx)
        finally:
            try:
                combo.blockSignals(False)
            except Exception:
                pass

    def _normalize_path_type_value(self, path_type):
        value = str(path_type or "straight").strip().lower()
        if value in ("ellipse", "ellipse_arc", "polyline"):
            return value
        return "straight"

    def _is_ellipse_path_type(self, path_type):
        return self._normalize_path_type_value(path_type) in ("ellipse", "ellipse_arc")

    def _is_arc_path_type(self, path_type):
        return self._normalize_path_type_value(path_type) == "ellipse_arc"

    def _is_polyline_path_type(self, path_type):
        return self._normalize_path_type_value(path_type) == "polyline"

    def _is_polyline_mode(self):
        return self._is_polyline_path_type(self._current_path_type())

    def _is_ellipse_mode(self):
        return self._is_ellipse_path_type(self._current_path_type())

    def _is_arc_mode(self):
        return self._is_arc_path_type(self._current_path_type())

    def _active_indicator_positions(self):
        return (
            self.ellipse_indicator_positions
            if self._is_ellipse_mode()
            else self.indicator_positions
        )

    def _sync_indicator_checks_for_mode(self):
        """Reflect the active path type's indicator flags in the shared checkboxes."""
        positions = self._active_indicator_positions()
        for check, key in (
            (getattr(self, "startIndicatorCheck", None), "start"),
            (getattr(self, "centerIndicatorCheck", None), "center"),
            (getattr(self, "endIndicatorCheck", None), "end"),
        ):
            if check is None:
                continue
            check.blockSignals(True)
            try:
                check.setChecked(bool(positions.get(key, False)))
            finally:
                check.blockSignals(False)

    def _path_type_label(self, path_type):
        value = self._normalize_path_type_value(path_type)
        if value == "ellipse_arc":
            return "Ellipse Arc"
        if value == "ellipse":
            return "Ellipse"
        return "Straight"

    def _single_path_state_has_geometry(self, state):
        if not isinstance(state, dict):
            return False
        if self._is_polyline_path_type(state.get("path_type")):
            return len(self._polyline_vertices_from_state(state)) >= 2
        if self._is_ellipse_path_type(state.get("path_type")):
            return isinstance(state.get("ellipse_geometry"), dict)
        return isinstance(state.get("line_pixel"), dict)

    def _find_pv_path_item(self, path_id):
        for item in self.pv_path_items:
            if item.get("id") == path_id:
                return item
        return None

    def _new_pv_path_id(self):
        while True:
            path_id = f"pvpath-{self._next_pv_path_index}"
            self._next_pv_path_index += 1
            if self._find_pv_path_item(path_id) is None:
                return path_id

    def _new_pv_path_name(self, path_type):
        label = self._path_type_label(path_type)
        count = 1 + sum(1 for item in self.pv_path_items if item.get("path_type") == path_type)
        return f"{label} {count}"

    def _sync_path_list_combo(self):
        combo = getattr(self, "activePathCombo", None)
        if combo is None:
            return
        self._syncing_path_list_combo = True
        combo.blockSignals(True)
        try:
            combo.clear()
            if not self.pv_path_items:
                combo.addItem("New path", None)
                combo.setEnabled(False)
                combo.setCurrentIndex(0)
                return
            combo.setEnabled(True)
            active_index = 0
            for index, item in enumerate(self.pv_path_items):
                name = str(item.get("name") or self._path_type_label(item.get("path_type")))
                combo.addItem(name, item.get("id"))
                if item.get("id") == self.active_pv_path_id:
                    active_index = index
            combo.setCurrentIndex(active_index)
        finally:
            combo.blockSignals(False)
            self._syncing_path_list_combo = False

    def _on_active_path_combo_changed(self):
        if getattr(self, "_syncing_path_list_combo", False):
            return
        combo = getattr(self, "activePathCombo", None)
        if combo is None:
            return
        path_id = combo.currentData()
        if path_id is not None:
            self.activate_pv_path(path_id)

    def _coerce_pv_path_item(self, raw):
        if not isinstance(raw, dict):
            return None
        state = raw.get("state")
        if not isinstance(state, dict):
            state = {key: value for key, value in raw.items() if key not in {"id", "name", "visible"}}
        if not self._single_path_state_has_geometry(state):
            return None
        path_type = self._normalize_path_type_value(state.get("path_type") or raw.get("path_type"))
        path_id = str(raw.get("id") or self._new_pv_path_id())
        item = {
            "id": path_id,
            "name": str(raw.get("name") or self._new_pv_path_name(path_type)),
            "path_type": path_type,
            "visible": bool(raw.get("visible", True)),
            "state": dict(state),
        }
        item["state"]["path_type"] = path_type
        return item

    def _path_items_for_workspace(self):
        items = []
        for item in self.pv_path_items:
            state = item.get("state")
            if not self._single_path_state_has_geometry(state):
                continue
            items.append(
                {
                    "id": item.get("id"),
                    "name": item.get("name"),
                    "path_type": item.get("path_type"),
                    "visible": bool(item.get("visible", True)),
                    "state": dict(state),
                }
            )
        return items

    def _sync_active_path_item_from_current_state(self, *, sync_combo=True):
        if getattr(self, "_restoring_single_pv_path", False):
            return False
        state = self._export_single_path_state()
        if not self._single_path_state_has_geometry(state):
            return False
        path_type = str(state.get("path_type") or "straight").lower()
        item = self._find_pv_path_item(self.active_pv_path_id)
        if item is None:
            item = {
                "id": self._new_pv_path_id(),
                "name": self._new_pv_path_name(path_type),
                "path_type": path_type,
                "visible": True,
                "state": state,
            }
            self.pv_path_items.append(item)
            self.active_pv_path_id = item["id"]
        else:
            item["path_type"] = path_type
            item["state"] = state
            if not item.get("name"):
                item["name"] = self._new_pv_path_name(path_type)
        if sync_combo:
            self._sync_path_list_combo()
        self._reveal_pv_result_if_ready()
        return True

    def begin_new_path(self):
        """Keep existing paths and start a blank path in the current Path Type."""
        self._sync_active_path_item_from_current_state()
        self.active_pv_path_id = None
        self.last_position_coord = None
        if self.pos_indicator_on_arrow is not None:
            self.pos_indicator_on_arrow.set_visible(False)
        self._clear_straight_path()
        self._clear_ellipse_path()
        self._clear_polyline_path()
        self._sync_path_type_controls()
        self._sync_path_list_combo()
        self._redraw_inactive_path_artists()
        self._invalidate_main_overlay_background()
        self._request_main_overlay_redraw()
        # NOTE: deliberately not recording an undo step here. begin_new_path is a
        # transient "blank new path" boundary (active path committed, live
        # geometry cleared); the meaningful edit is recorded when the new path is
        # actually committed (apply_controls/on_release). Recording the transient
        # state produced a confusing extra undo step where an existing path
        # briefly vanished.

    def _remove_inactive_artists_for_path(self, path_id):
        for artist in list((self.inactive_path_artists or {}).pop(path_id, [])):
            self._remove_artist(artist)

    def _clear_inactive_path_artists(self):
        for path_id in list((self.inactive_path_artists or {}).keys()):
            self._remove_inactive_artists_for_path(path_id)
        self.inactive_path_artists = {}

    def _inactive_path_color(self):
        return str(getattr(self, "pvarrow_color", "yellow") or "yellow")

    def _ellipse_path_xy(self, path, *, min_points=96):
        if path is None:
            return None, None
        sweep = abs(float(path.end_phi_rad - path.start_phi_rad))
        points = max(int(min_points), int(math.ceil(max(sweep, 1e-6) / (2.0 * math.pi) * 192)))
        phi = np.linspace(float(path.start_phi_rad), float(path.end_phi_rad), points + 1)
        cos_pa = math.cos(float(path.pa_rad))
        sin_pa = math.sin(float(path.pa_rad))
        local_x = float(path.semi_major_px) * np.cos(phi)
        local_y = float(path.semi_minor_px) * np.sin(phi)
        cx, cy = path.center
        xs = float(cx) + local_x * cos_pa - local_y * sin_pa
        ys = float(cy) + local_x * sin_pa + local_y * cos_pa
        return xs, ys

    def _redraw_inactive_path_artists(self):
        self._clear_inactive_path_artists()
        color = self._inactive_path_color()
        for item in self.pv_path_items:
            if item.get("id") == self.active_pv_path_id or not item.get("visible", True):
                continue
            state = item.get("state") or {}
            artists = []
            item_type = self._normalize_path_type_value(item.get("path_type"))
            if self._is_ellipse_path_type(item_type):
                path = self._ellipse_path_geometry_from_state(state, item_type)
                if path is None:
                    continue
                if self._is_arc_path_type(item_type):
                    xs, ys = self._ellipse_path_xy(path)
                    if xs is None or ys is None:
                        continue
                    line, = self.fits_ax.plot(
                        xs,
                        ys,
                        color=color,
                        linewidth=max(1.0, 0.9 * float(self.arrow_size)),
                        alpha=0.35,
                        zorder=4,
                    )
                    artists.append(line)
                else:
                    cx, cy = path.center
                    patch = mpl.patches.Ellipse(
                        (cx, cy),
                        width=2.0 * float(path.semi_major_px),
                        height=2.0 * float(path.semi_minor_px),
                        angle=math.degrees(float(path.pa_rad)),
                        fill=False,
                        edgecolor=color,
                        linewidth=max(1.0, 0.9 * float(self.arrow_size)),
                        alpha=0.35,
                        zorder=4,
                    )
                    self.fits_ax.add_patch(patch)
                    artists.append(patch)
            elif self._is_polyline_path_type(item_type):
                verts = self._polyline_vertices_from_state(state)
                if len(verts) < 2:
                    continue
                spline_type, smoothness = self._polyline_spline_from_state(state)
                geom = self._polyline_path_geometry_from_vertices(
                    verts, spline_type=spline_type, smoothness=smoothness
                )
                curve = geom.effective_vertices if geom is not None else verts
                line, = self.fits_ax.plot(
                    [p[0] for p in curve],
                    [p[1] for p in curve],
                    color=color,
                    linewidth=max(1.0, 0.9 * float(self.arrow_size)),
                    alpha=0.35,
                    zorder=4,
                )
                artists.append(line)
            else:
                start, end = self._extract_line_from_workspace_state(state)
                if start is None or end is None:
                    continue
                patch = mpl.patches.FancyArrowPatch(
                    start,
                    end,
                    arrowstyle="Simple,tail_width=1.2,head_width=6,head_length=6",
                    shrinkA=0,
                    shrinkB=0,
                    color=color,
                    lw=0,
                    mutation_scale=max(0.7, 0.85 * float(self.arrow_size)),
                    alpha=0.35,
                    zorder=4,
                )
                self.fits_ax.add_patch(patch)
                artists.append(patch)
            if artists:
                self.inactive_path_artists[item.get("id")] = artists
        self._invalidate_main_overlay_background()

    def _path_item_hit_test(self, item, x, y, tol):
        state = item.get("state") or {}
        item_type = self._normalize_path_type_value(item.get("path_type"))
        if self._is_ellipse_path_type(item_type):
            path = self._ellipse_path_geometry_from_state(state, item_type)
            if path is None:
                return False
            semi_major = float(path.semi_major_px)
            semi_minor = float(path.semi_minor_px)
            if semi_major <= 0.0 or semi_minor <= 0.0:
                return False
            if self._is_arc_path_type(item_type):
                return self._ellipse_arc_curve_hit(path, x, y, tol)
            cx, cy = path.center
            dx = float(x) - cx
            dy = float(y) - cy
            pa = float(path.pa_rad)
            cos_a = math.cos(pa)
            sin_a = math.sin(pa)
            x_local = cos_a * dx + sin_a * dy
            y_local = -sin_a * dx + cos_a * dy
            norm = (x_local / semi_major) ** 2 + (y_local / semi_minor) ** 2
            if norm <= 1.0:
                return True
            edge_distance = abs(math.sqrt(norm) - 1.0) * max(semi_major, semi_minor)
            return edge_distance <= tol

        if self._is_polyline_path_type(item_type):
            verts = self._polyline_vertices_from_state(state)
            if len(verts) < 2:
                return False
            # Control nodes are always the grab handles, even in smooth mode.
            for (vx, vy) in verts:
                if math.hypot(float(x) - vx, float(y) - vy) <= tol:
                    return True
            # The connecting path follows the densified curve when Smooth is on.
            spline_type, smoothness = self._polyline_spline_from_state(state)
            geom = self._polyline_path_geometry_from_vertices(
                verts, spline_type=spline_type, smoothness=smoothness
            )
            curve = geom.effective_vertices if geom is not None else verts
            for i in range(len(curve) - 1):
                if self.point_line_distance((x, y), curve[i], curve[i + 1]) <= tol:
                    return True
            return False

        start, end = self._extract_line_from_workspace_state(state)
        if start is None or end is None:
            return False
        return self.point_line_distance((x, y), start, end) <= tol

    def _activate_path_at_point(self, x, y, tol):
        for item in reversed(self.pv_path_items):
            if item.get("id") == self.active_pv_path_id or not item.get("visible", True):
                continue
            if self._path_item_hit_test(item, x, y, tol):
                return self.activate_pv_path(item.get("id"))
        return False

    def _active_path_hit_test(self, x, y, tol):
        state = self._export_single_path_state()
        if not self._single_path_state_has_geometry(state):
            return False
        if self._is_arc_mode() and (
            self._ellipse_center_hit(x, y)
            or self._ellipse_phase_handle(x, y)
            or self._ellipse_resize_handle(x, y)
        ):
            return True
        item = {
            "id": self.active_pv_path_id,
            "path_type": state.get("path_type", self._current_path_type()),
            "state": state,
        }
        return self._path_item_hit_test(item, x, y, tol)

    def activate_pv_path(self, path_id):
        item = self._find_pv_path_item(path_id)
        if item is None:
            return False
        if path_id == self.active_pv_path_id and self._single_path_state_has_geometry(item.get("state")):
            return True

        self._sync_active_path_item_from_current_state()
        item = self._find_pv_path_item(path_id)
        if item is None:
            return False

        self._remove_inactive_artists_for_path(path_id)
        self._clear_straight_path()
        self._clear_ellipse_path()
        self._clear_polyline_path()
        self.active_pv_path_id = path_id

        self._restoring_single_pv_path = True
        try:
            restored = self.restore_workspace_state(dict(item.get("state") or {}))
        finally:
            self._restoring_single_pv_path = False
        self._sync_path_list_combo()
        self._redraw_inactive_path_artists()
        self._invalidate_main_overlay_background()
        self._request_main_overlay_redraw()
        return bool(restored)

    def _on_path_type_changed(self):
        previous = getattr(self, "path_type", "straight")
        new_type = self._current_path_type()
        if new_type != previous:
            self._sync_active_path_item_from_current_state()
            self.active_pv_path_id = None
            self._clear_straight_path()
            self._clear_ellipse_path()
            self._clear_polyline_path()
        self.path_type = new_type
        self._sync_path_type_controls()
        self._sync_path_list_combo()
        self._redraw_inactive_path_artists()
        self._request_main_overlay_redraw()

    def _on_ellipse_span_mode_changed(self):
        if not self._is_ellipse_mode():
            return
        self.path_type = self._current_path_type()
        self._sync_path_type_controls()
        if self.ellipse_geometry is not None:
            self._draw_ellipse_overlay()
            self.update_pv_diagram(force_update=True)
            self._sync_active_path_item_from_current_state()
            self._redraw_inactive_path_artists()
        self._request_main_overlay_redraw()

    def _sync_path_type_controls(self):
        ellipse_mode = self._is_ellipse_mode()
        polyline_mode = self._is_polyline_mode()
        # Straight = the only mode with editable endpoint/length/origin geometry.
        straight_mode = not ellipse_mode and not polyline_mode
        arc_mode = self._is_arc_mode()
        ellipse_widget = getattr(self, "ellipseGeometryWidget", None)
        if ellipse_widget is not None:
            ellipse_widget.setVisible(ellipse_mode)
        geometry_combo = getattr(self, "geometryInputCombo", None)
        if geometry_combo is not None:
            geometry_combo.setEnabled(straight_mode)
            geometry_combo.setVisible(straight_mode)
        geometry_label = getattr(self, "geometryInputLabel", None)
        if geometry_label is not None:
            geometry_label.setVisible(straight_mode)
        for widget in getattr(self, "_ellipse_input_widgets", []):
            try:
                widget.setEnabled(ellipse_mode)
            except Exception:
                pass
        for widget in (
            getattr(self, "ellipseEndPhiLabel", None),
            getattr(self, "ellipseEndPhiSpin", None),
        ):
            try:
                if widget is not None:
                    widget.setVisible(arc_mode)
                    widget.setEnabled(arc_mode)
            except Exception:
                pass
        for widget in (
            getattr(self, "rotationAngleLabel", None),
            getattr(self, "rotationAngleSpin", None),
            getattr(self, "positionOriginLabel", None),
            getattr(self, "positionOriginWidget", None),
            getattr(self, "positionOriginCombo", None),
            getattr(self, "reverseDirectionButton", None),
        ):
            try:
                if widget is not None:
                    widget.setEnabled(straight_mode)
                    widget.setVisible(straight_mode)
            except Exception:
                pass
        try:
            if getattr(self, "lengthLabel", None) is not None:
                if ellipse_mode:
                    length_text = "P Axis Unit:"
                elif polyline_mode:
                    length_text = "Path Unit:"
                else:
                    length_text = "Straight Length:"
                self.lengthLabel.setText(length_text)
                self.lengthLabel.setVisible(True)
                self.lengthLabel.setEnabled(True)
        except Exception:
            pass
        try:
            if getattr(self, "lengthWidget", None) is not None:
                self.lengthWidget.setVisible(True)
                self.lengthWidget.setEnabled(True)
        except Exception:
            pass
        try:
            if getattr(self, "arrowLengthSpin", None) is not None:
                # Editable only for straight; ellipse/polyline length is derived.
                self.arrowLengthSpin.setVisible(straight_mode)
                self.arrowLengthSpin.setEnabled(straight_mode)
        except Exception:
            pass
        try:
            if getattr(self, "lengthUnitCombo", None) is not None:
                self.lengthUnitCombo.setVisible(True)
                self.lengthUnitCombo.setEnabled(True)
        except Exception:
            pass
        # Curve type / smoothness / path-length / node-marker toggle are
        # polyline-only; show + sync.
        for widget in (
            getattr(self, "polylineCurveLabel", None),
            getattr(self, "polylineCurveCombo", None),
            getattr(self, "polylineSmoothnessLabel", None),
            getattr(self, "polylineSmoothnessWidget", None),
            getattr(self, "polylinePathLengthLabel", None),
            getattr(self, "polylinePathLengthValueLabel", None),
            getattr(self, "nodeMarkerCheck", None),
        ):
            try:
                if widget is not None:
                    widget.setVisible(polyline_mode)
            except Exception:
                pass
        if polyline_mode:
            self._sync_polyline_curve_controls()
            self._update_polyline_length_label()
        try:
            if getattr(self, "applyArrowButton", None) is not None:
                self.applyArrowButton.setEnabled(True)
        except Exception:
            pass
        try:
            if getattr(self, "savePathButton", None) is not None:
                self.savePathButton.setEnabled(True)
        except Exception:
            pass
        try:
            if getattr(self, "saveFitsButton", None) is not None:
                self.saveFitsButton.setEnabled(True)
        except Exception:
            pass
        self._sync_x_axis_controls()
        self._sync_geometry_input_enabled()
        self._sync_indicator_checks_for_mode()
        if ellipse_mode:
            self._sync_ellipse_controls_from_state()

    def _set_path_type_from_state(self, path_type):
        desired = self._normalize_path_type_value(path_type)
        combo = getattr(self, "pathTypeCombo", None)
        if combo is not None:
            try:
                combo_path_type = "ellipse" if self._is_ellipse_path_type(desired) else desired
                idx = combo.findData(combo_path_type)
                if idx >= 0:
                    combo.blockSignals(True)
                    combo.setCurrentIndex(idx)
                    combo.blockSignals(False)
            except Exception:
                try:
                    combo.blockSignals(False)
                except Exception:
                    pass
        self._set_ellipse_span_mode(self._is_arc_path_type(desired))
        self.path_type = desired
        self._sync_path_type_controls()
        return desired

    def _current_x_axis_mode(self):
        combo = getattr(self, "xAxisModeCombo", None)
        if combo is not None:
            try:
                return normalize_pv_x_axis_mode(combo.currentData())
            except Exception:
                pass
        return normalize_pv_x_axis_mode(getattr(self, "x_axis_mode", PV_X_AXIS_POSITION))

    def _sync_x_axis_controls(self):
        self.x_axis_mode = self._current_x_axis_mode()
        combo = getattr(self, "xAxisModeCombo", None)
        label = getattr(self, "xAxisModeLabel", None)
        phi_enabled = self._is_ellipse_mode()
        if label is not None:
            try:
                label.setVisible(phi_enabled)
            except Exception:
                pass
        if combo is not None:
            try:
                combo.setVisible(phi_enabled)
            except Exception:
                pass
            try:
                idx = combo.findData(PV_X_AXIS_PHI)
                if idx >= 0:
                    item = combo.model().item(idx)
                    if item is not None:
                        item.setEnabled(phi_enabled)
                        item.setToolTip("" if phi_enabled else "Available when Path Type is Ellipse.")
            except Exception:
                pass
        if not phi_enabled and self.x_axis_mode != PV_X_AXIS_POSITION:
            self.x_axis_mode = PV_X_AXIS_POSITION
            if combo is not None:
                idx = combo.findData(PV_X_AXIS_POSITION)
                if idx >= 0:
                    combo.blockSignals(True)
                    try:
                        combo.setCurrentIndex(idx)
                    finally:
                        combo.blockSignals(False)

    def _on_x_axis_mode_changed(self):
        self._sync_x_axis_controls()
        self.is_range_manual = False
        self.update_pv_diagram(force_update=True)

    def _current_position_origin(self):
        combo = getattr(self, "positionOriginCombo", None)
        if combo is not None:
            try:
                return normalize_position_origin(combo.currentData())
            except Exception:
                pass
        return normalize_position_origin(getattr(self, "position_origin", POSITION_ORIGIN_START))

    def _current_geometry_input_mode(self):
        combo = getattr(self, "geometryInputCombo", None)
        if combo is not None:
            try:
                value = str(combo.currentData() or "endpoints")
                if value in ("endpoints", "center"):
                    return value
            except Exception:
                pass
        return "center" if getattr(self, "geometry_input_mode", "endpoints") == "center" else "endpoints"

    def _sync_geometry_input_enabled(self):
        ellipse_mode = self._is_ellipse_mode()
        polyline_mode = self._is_polyline_mode()
        # Polyline has no endpoint/center spinbox geometry (it is mouse-driven).
        path_geom_mode = ellipse_mode or polyline_mode
        center_mode = self._current_geometry_input_mode() == "center"
        if center_mode and not path_geom_mode:
            self._set_position_origin_for_center_input()
        endpoint_section = getattr(self, "endpointGeometryWidget", None)
        if endpoint_section is not None:
            endpoint_section.setVisible(not center_mode and not path_geom_mode)
        center_section = getattr(self, "centerGeometryWidget", None)
        if center_section is not None:
            center_section.setVisible(center_mode and not path_geom_mode)
        for widget in getattr(self, "_endpoint_input_widgets", []):
            try:
                widget.setEnabled(not center_mode and not path_geom_mode)
            except Exception:
                pass
        for widget in getattr(self, "_center_input_widgets", []):
            try:
                widget.setEnabled(center_mode and not path_geom_mode)
            except Exception:
                pass
        position_combo = getattr(self, "positionOriginCombo", None)
        if position_combo is not None:
            try:
                position_combo.setEnabled(not center_mode and not ellipse_mode)
                if center_mode:
                    position_combo.setToolTip("Center + Length + PA uses Center origin so length/PA edits keep the center fixed.")
                elif ellipse_mode:
                    position_combo.setToolTip("Ellipse position starts at Phi0; Center origin is not used.")
                else:
                    position_combo.setToolTip("")
            except Exception:
                pass

    def _on_geometry_input_mode_changed(self):
        previous_mode = getattr(self, "geometry_input_mode", "endpoints")
        new_mode = self._current_geometry_input_mode()
        if new_mode == "center" and previous_mode != "center":
            self._endpoint_position_origin = self._current_position_origin()

        self.geometry_input_mode = new_mode
        self._sync_geometry_input_enabled()
        if new_mode == "endpoints" and previous_mode == "center":
            origin = normalize_position_origin(
                getattr(self, "_endpoint_position_origin", POSITION_ORIGIN_START)
            )
            combo = getattr(self, "positionOriginCombo", None)
            if combo is not None:
                idx = combo.findData(origin)
                if idx >= 0 and combo.currentIndex() != idx:
                    combo.setCurrentIndex(idx)
                else:
                    self.position_origin = origin
            else:
                self.position_origin = origin
        self.update_controls()

    def _set_position_origin_for_center_input(self):
        combo = getattr(self, "positionOriginCombo", None)
        if combo is None:
            self.position_origin = POSITION_ORIGIN_CENTER
            return
        try:
            idx = combo.findData(POSITION_ORIGIN_CENTER)
            if idx >= 0 and combo.currentIndex() != idx:
                combo.setCurrentIndex(idx)
            else:
                self.position_origin = POSITION_ORIGIN_CENTER
        except Exception:
            self.position_origin = POSITION_ORIGIN_CENTER

    def _on_options_toggled(self, checked):
        panel = getattr(self, "optionsPanel", None)
        if panel is not None:
            panel.setVisible(bool(checked))
        button = getattr(self, "optionsToggleButton", None)
        if button is not None:
            button.setText("Hide Options" if checked else "Show Options")

    def _line_center(self):
        if self.line_start is None or self.line_end is None:
            return None
        return (
            (self.line_start[0] + self.line_end[0]) / 2.0,
            (self.line_start[1] + self.line_end[1]) / 2.0,
        )

    def _current_line_angle_rad(self):
        if self.line_start is not None and self.line_end is not None:
            return np.arctan2(
                self.line_end[1] - self.line_start[1],
                self.line_end[0] - self.line_start[0],
            )
        return np.radians(self.rotationAngleSpin.value())

    def _current_line_length_px(self):
        if self.line_start is not None and self.line_end is not None:
            return np.hypot(
                self.line_end[0] - self.line_start[0],
                self.line_end[1] - self.line_start[1],
            )
        current_unit = self.lengthUnitCombo.currentText()
        return self._convert_length(self.arrowLengthSpin.value(), current_unit, 'pixel')

    def _set_center_controls_from_pixel(self, x, y):
        for spin, value in ((self.centerXSpin, x), (self.centerYSpin, y)):
            try:
                spin.blockSignals(True)
                spin.setValue(float(value))
            finally:
                spin.blockSignals(False)
        self._update_center_world_from_pixel()

    def _update_center_world_from_pixel(self):
        """Update center WCS QLineEdits from the center pixel spinboxes."""
        center_coords = self.coord_converter.pix_to_world(
            self.centerXSpin.value(), self.centerYSpin.value()
        )
        center_lon, center_lat = center_coords[0], center_coords[1]

        self.centerLonEdit.blockSignals(True)
        self.centerLatEdit.blockSignals(True)
        self.centerLonEdit.setText(center_lon)
        self.centerLatEdit.setText(center_lat)
        self.centerLonEdit.blockSignals(False)
        self.centerLatEdit.blockSignals(False)

    def _update_center_pixel_from_world(self):
        """Update center pixel spinboxes from the center WCS QLineEdits."""
        try:
            center_pix = self.coord_converter.world_to_pix(
                self.centerLonEdit.text(), self.centerLatEdit.text()
            )
            self._set_center_controls_from_pixel(center_pix[0], center_pix[1])
            self.update_arrow_from_center()
        except ValueError as e:
            print(f"Error parsing center coordinate string: {e}")
            center = self._line_center()
            if center is not None:
                self._set_center_controls_from_pixel(center[0], center[1])
        except Exception as e:
            print(f"Error updating center pixels from world coordinates: {e}")
            center = self._line_center()
            if center is not None:
                self._set_center_controls_from_pixel(center[0], center[1])

    def _current_main_cursor_xy(self):
        viewer = getattr(self, "fits_viewer", None)
        if viewer is None:
            return None
        try:
            x_getter = getattr(viewer, "_get_shared_xpix", None)
            y_getter = getattr(viewer, "_get_shared_ypix", None)
            if callable(x_getter) and callable(y_getter):
                return (float(x_getter()), float(y_getter()))
        except Exception:
            pass
        x_value = getattr(viewer, "xpix", None)
        y_value = getattr(viewer, "ypix", None)
        if x_value is None or y_value is None:
            cursor = getattr(viewer, "cursor", None)
            x_value = getattr(cursor, "xpix", x_value)
            y_value = getattr(cursor, "ypix", y_value)
        if x_value is None or y_value is None:
            return None
        try:
            return (float(x_value), float(y_value))
        except Exception:
            return None

    def _line_length_in_unit(self, unit=None):
        if self.line_start is None or self.line_end is None:
            return 0.0
        if unit is None:
            unit = self.length_unit
        line_length_px = np.hypot(
            self.line_end[0] - self.line_start[0],
            self.line_end[1] - self.line_start[1],
        )
        return self._convert_length(line_length_px, 'pixel', unit)

    def _position_coord_from_fraction(self, fraction, unit=None, origin=None):
        if unit is None:
            unit = self.length_unit
        if origin is None:
            origin = self._current_position_origin()
        length = self._line_length_in_unit(unit=unit)
        return position_from_fraction(fraction, length, origin)

    def _on_position_origin_changed(self):
        old_origin = normalize_position_origin(getattr(self, "position_origin", POSITION_ORIGIN_START))
        fraction = self._get_cursor_fractional_position(origin=old_origin)
        self.position_origin = self._current_position_origin()
        if self._current_geometry_input_mode() == "endpoints":
            self._endpoint_position_origin = self.position_origin
        if fraction is not None:
            self.last_position_coord = self._position_coord_from_fraction(
                fraction,
                origin=self.position_origin,
            )
        self.is_range_manual = False
        self.update_pv_diagram(force_update=True)
        if self.last_position_coord is not None:
            self.update_pv_position_cursor(self.last_position_coord)
            self._update_main_window_marker(self.last_position_coord)

    def _on_arrow_width_changed(self, new_size):
        """Handles changes to the arrow width spinbox."""
        self.arrow_size = new_size
        if self.arrow_artist:
            try:
                self.arrow_artist.set_mutation_scale(self.arrow_size)
            except Exception:
                pass
        for indicator in self.width_indicators or []:
            try:
                indicator.set_linewidth(1.5 * self.arrow_size)
            except Exception:
                pass
        if self.ellipse_artist is not None:
            try:
                self.ellipse_artist.set_linewidth(1.5 * self.arrow_size)
            except Exception:
                pass

        # Update the position indicator with the new size
        self._update_main_window_marker(self.last_position_coord)
        self._redraw_inactive_path_artists()
        if hasattr(self.fits_viewer, 'redraw_main_overlay_and_blit'):
            self.fits_viewer.redraw_main_overlay_and_blit()
        else:
            self.fits_canvas.draw_idle()

    def _on_slit_color_changed(self, new_color):
        """Apply slit color changes to existing slit artists immediately."""
        self.pvarrow_color = new_color
        artists = [self.arrow_artist, self.pos_indicator_on_arrow]
        artists.extend(self.width_indicators or [])
        artists.append(self.ellipse_artist)
        artists.extend((self.ellipse_handle_artists or {}).values())
        for artist in artists:
            if artist is None:
                continue
            try:
                artist.set_color(new_color)
            except Exception:
                pass
            try:
                artist.set_edgecolor(new_color)
            except Exception:
                pass
        self._redraw_inactive_path_artists()
        if hasattr(self.fits_viewer, 'redraw_main_overlay_and_blit'):
            self.fits_viewer.redraw_main_overlay_and_blit()
        else:
            self.fits_canvas.draw_idle()

    def _request_main_overlay_redraw(self):
        if hasattr(self.fits_viewer, 'redraw_main_overlay_and_blit'):
            self.fits_viewer.redraw_main_overlay_and_blit()
        else:
            self.fits_canvas.draw_idle()

    def _invalidate_main_overlay_background(self):
        viewer = getattr(self, "fits_viewer", None)
        if viewer is None:
            return
        invalidator = getattr(viewer, "_invalidate_plane_background", None)
        if callable(invalidator):
            try:
                invalidator("xy")
                return
            except Exception:
                pass
        state = getattr(viewer, "state", None)
        if state is not None:
            try:
                state._background = None
                state.image_background = None
            except Exception:
                pass
        try:
            viewer._background = None
            viewer._background_initialized = False
        except Exception:
            pass

    def main_overlay_artists(self):
        artists = [
            self.arrow_artist,
            self.pos_indicator_on_arrow,
            self.ellipse_artist,
            # Polyline overlay must be part of the dynamic blit list, otherwise a
            # blit triggered elsewhere (e.g. clicking the PV diagram) restores the
            # cached background without it and the polyline visually disappears.
            self.polyline_line_artist,
            self.polyline_node_artist,
            self.polyline_select_artist,
            self.polyline_rubber_artist,
        ]
        artists.extend(self.width_indicators or [])
        artists.extend((self.ellipse_handle_artists or {}).values())
        artists.extend(self.ellipse_indicator_artists or [])
        artists.extend(getattr(self, "polyline_indicator_artists", None) or [])
        # Inactive paths are kept out of the dynamic overlay list so their
        # translucent alpha is composited exactly once into the background.
        # They are redrawn on top of the image during the fast channel-change
        # blit via inactive_overlay_artists() (see ViewerBlitMixin), otherwise
        # the freshly drawn channel image would hide them.
        return [artist for artist in artists if artist is not None]

    def inactive_overlay_artists(self):
        """Flat list of inactive-path slit artists drawn on the main image.

        These live in the cached background, but the fast image blit on a channel
        change repaints the image over that background, so they must be redrawn on
        top of the new image to stay visible (mirrors how the active slit and
        contours are repainted)."""
        artists = []
        for arts in (self.inactive_path_artists or {}).values():
            for artist in (arts or []):
                if artist is not None:
                    artists.append(artist)
        return artists

    def _main_image_axes(self):
        axes = [self.fits_ax]
        viewer = getattr(self, "fits_viewer", None)
        for candidate in (
            getattr(viewer, "ax", None),
            getattr(getattr(viewer, "displaymap", None), "overlay_ax", None),
        ):
            if candidate is not None and all(candidate is not ax for ax in axes):
                axes.append(candidate)
        return tuple(axes)

    def _event_xy_on_main_image(self, event):
        if event is None:
            return None
        if getattr(event, "inaxes", None) not in self._main_image_axes():
            return None

        ex = getattr(event, "x", None)
        ey = getattr(event, "y", None)
        if ex is not None and ey is not None and self.fits_ax is not None:
            try:
                x, y = self.fits_ax.transData.inverted().transform((float(ex), float(ey)))
                if np.isfinite(x) and np.isfinite(y):
                    return float(x), float(y)
            except Exception:
                pass

        x = getattr(event, "xdata", None)
        y = getattr(event, "ydata", None)
        try:
            x = float(x)
            y = float(y)
        except Exception:
            return None
        if not np.isfinite(x) or not np.isfinite(y):
            return None
        return x, y

    def _event_xy_for_main_image_drag(self, event):
        """Return main-image data coordinates while an existing drag is active.

        Press handling remains restricted to the image axes, but a drag that
        started there must keep receiving useful coordinates after the pointer
        crosses the axes frame. Matplotlib still supplies display-pixel
        coordinates for those motion/release events even though ``inaxes`` and
        ``xdata``/``ydata`` become ``None``.
        """
        event_xy = self._event_xy_on_main_image(event)
        if event_xy is not None:
            return event_xy
        if event is None or self.fits_ax is None:
            return None

        ex = getattr(event, "x", None)
        ey = getattr(event, "y", None)
        if ex is None or ey is None:
            return None
        try:
            x, y = self.fits_ax.transData.inverted().transform((float(ex), float(ey)))
        except Exception:
            return None
        if not np.isfinite(x) or not np.isfinite(y):
            return None
        return float(x), float(y)

    def _straight_interaction_active(self):
        return self.drag_mode == "draw" or self.edit_mode in {"endpoint", "move", "rotate"}

    def _main_image_drag_bounds(self):
        """Return the visible, sampleable data rectangle for straight drags."""
        height, width = self.data.shape[-2], self.data.shape[-1]
        data_bounds = (
            0.0,
            max(0.0, float(width - 1)),
            0.0,
            max(0.0, float(height - 1)),
        )
        try:
            view_x0, view_x1 = sorted(float(value) for value in self.fits_ax.get_xlim())
            view_y0, view_y1 = sorted(float(value) for value in self.fits_ax.get_ylim())
        except Exception:
            return data_bounds

        x0 = max(data_bounds[0], view_x0)
        x1 = min(data_bounds[1], view_x1)
        y0 = max(data_bounds[2], view_y0)
        y1 = min(data_bounds[3], view_y1)
        if x0 > x1 or y0 > y1:
            return data_bounds
        return x0, x1, y0, y1

    @staticmethod
    def _project_drag_point_to_bounds(anchor, target, bounds):
        """Project an outside target onto a rectangle along the anchor ray."""
        x0, x1, y0, y1 = bounds
        ax, ay = float(anchor[0]), float(anchor[1])
        tx, ty = float(target[0]), float(target[1])
        if x0 <= tx <= x1 and y0 <= ty <= y1:
            return tx, ty

        anchor_is_inside = x0 <= ax <= x1 and y0 <= ay <= y1
        dx = tx - ax
        dy = ty - ay
        if not anchor_is_inside or (abs(dx) < 1e-12 and abs(dy) < 1e-12):
            return min(max(tx, x0), x1), min(max(ty, y0), y1)

        exit_fractions = []
        if dx > 0.0:
            exit_fractions.append((x1 - ax) / dx)
        elif dx < 0.0:
            exit_fractions.append((x0 - ax) / dx)
        if dy > 0.0:
            exit_fractions.append((y1 - ay) / dy)
        elif dy < 0.0:
            exit_fractions.append((y0 - ay) / dy)

        valid_fractions = [value for value in exit_fractions if 0.0 <= value <= 1.0]
        if not valid_fractions:
            return min(max(tx, x0), x1), min(max(ty, y0), y1)
        fraction = min(valid_fractions)
        return ax + fraction * dx, ay + fraction * dy

    def _constrain_straight_drag_xy(self, x, y):
        """Keep a straight-slit drag inside the visible image data rectangle."""
        bounds = self._main_image_drag_bounds()
        target = (float(x), float(y))

        if self.drag_mode == "draw" and self.line_start is not None:
            return self._project_drag_point_to_bounds(self.line_start, target, bounds)

        if self.edit_mode == "endpoint":
            anchor = self.locked_end if self.dragging_endpoint == 0 else self.locked_start
            if anchor is not None:
                return self._project_drag_point_to_bounds(anchor, target, bounds)

        if (
            self.edit_mode == "move"
            and self.drag_start is not None
            and self.initial_line_start is not None
            and self.initial_line_end is not None
        ):
            x0, x1, y0, y1 = bounds
            desired_dx = target[0] - self.drag_start[0]
            desired_dy = target[1] - self.drag_start[1]
            line_min_x = min(self.initial_line_start[0], self.initial_line_end[0])
            line_max_x = max(self.initial_line_start[0], self.initial_line_end[0])
            line_min_y = min(self.initial_line_start[1], self.initial_line_end[1])
            line_max_y = max(self.initial_line_start[1], self.initial_line_end[1])
            min_dx = x0 - line_min_x
            max_dx = x1 - line_max_x
            min_dy = y0 - line_min_y
            max_dy = y1 - line_max_y
            if min_dx <= max_dx:
                desired_dx = min(max(desired_dx, min_dx), max_dx)
            if min_dy <= max_dy:
                desired_dy = min(max(desired_dy, min_dy), max_dy)
            return self.drag_start[0] + desired_dx, self.drag_start[1] + desired_dy

        # Rotation only uses the pointer direction; its fixed-length geometry is
        # left unchanged rather than distorting the slit to fit the rectangle.
        return target

    @staticmethod
    def _motion_has_left_button(event):
        """Return whether a motion event reports the left button as held."""
        buttons = getattr(event, "buttons", None)
        if buttons is not None:
            try:
                return bool(buttons & mpl.backend_bases.MouseButton.LEFT)
            except (TypeError, ValueError):
                try:
                    return mpl.backend_bases.MouseButton.LEFT in buttons
                except TypeError:
                    pass
        button = getattr(event, "button", None)
        return button in (1, mpl.backend_bases.MouseButton.LEFT)

    def _update_world_from_pixel(self):
        """Update WCS QLineEdits from pixel QDoubleSpinBoxes."""
        if self.line_start is None or self.line_end is None:
            return

        # Update start coordinates
        start_coords = self.coord_converter.pix_to_world(
            self.startXSpin.value(), self.startYSpin.value()
        )
        start_lon, start_lat = start_coords[0], start_coords[1]

        self.startLonEdit.blockSignals(True)
        self.startLatEdit.blockSignals(True)
        self.startLonEdit.setText(start_lon)
        self.startLatEdit.setText(start_lat)
        self.startLonEdit.blockSignals(False)
        self.startLatEdit.blockSignals(False)

        # Update end coordinates
        end_coords = self.coord_converter.pix_to_world(
            self.endXSpin.value(), self.endYSpin.value()
        )
        end_lon, end_lat = end_coords[0], end_coords[1]

        self.endLonEdit.blockSignals(True)
        self.endLatEdit.blockSignals(True)
        self.endLonEdit.setText(end_lon)
        self.endLatEdit.setText(end_lat)
        self.endLonEdit.blockSignals(False)
        self.endLatEdit.blockSignals(False)

    def _update_pixel_from_world(self):
        """Update pixel QDoubleSpinBoxes from WCS QLineEdits."""
        try:
            # Update start pixels
            start_pix = self.coord_converter.world_to_pix(
                self.startLonEdit.text(), self.startLatEdit.text()
            )
            start_x, start_y = start_pix[0], start_pix[1]
            exact_start = (float(start_x), float(start_y))

            self.startXSpin.blockSignals(True)
            self.startYSpin.blockSignals(True)
            self.startXSpin.setValue(start_x)
            self.startYSpin.setValue(start_y)
            self.startXSpin.blockSignals(False)
            self.startYSpin.blockSignals(False)

            # Update end pixels
            end_pix = self.coord_converter.world_to_pix(
                self.endLonEdit.text(), self.endLatEdit.text()
            )
            end_x, end_y = end_pix[0], end_pix[1]
            exact_end = (float(end_x), float(end_y))

            self.endXSpin.blockSignals(True)
            self.endYSpin.blockSignals(True)
            self.endXSpin.setValue(end_x)
            self.endYSpin.setValue(end_y)
            self.endXSpin.blockSignals(False)
            self.endYSpin.blockSignals(False)

            # Preserve full-precision endpoints for this update; the spinboxes
            # remain rounded for display only.
            self._pending_precise_world_line = (exact_start, exact_end)

            # If successful, trigger arrow update
            self.apply_controls()

        except ValueError as e:            # Handle parsing errors, e.g., show a message to the user            print(f"Error parsing coordinate string: {e}")
            # Revert to the last valid pixel values
            self._update_world_from_pixel()

    def _update_pixel_from_world_for_end_point(self):
        """A helper to update just the end point controls after length/angle change."""
        self.endXSpin.blockSignals(True)
        self.endYSpin.blockSignals(True)
        self.endXSpin.setValue(self.line_end[0])
        self.endYSpin.setValue(self.line_end[1])
        self.endXSpin.blockSignals(False)
        self.endYSpin.blockSignals(False)

        end_coords = self.coord_converter.pix_to_world(self.line_end[0], self.line_end[1])
        self.endLonEdit.blockSignals(True)
        self.endLatEdit.blockSignals(True)
        self.endLonEdit.setText(end_coords[0])
        self.endLatEdit.setText(end_coords[1])
        self.endLonEdit.blockSignals(False)
        self.endLatEdit.blockSignals(False)

    def _seed_color_panel_settings_from_current_image(self):
        settings = {
            "min_val": None,
            "max_val": None,
            "log_scale": False,
            "gamma_value": 1.0,
            "invert": False,
            "color_pattern": None,
        }
        raw = dict(ColorSettingsPanel.settings.get(ColorMode.PV, {}) or {})
        if isinstance(raw, dict):
            settings.update(raw)
        hint = getattr(self, "_color_panel_hint", None)
        if isinstance(hint, dict):
            settings.update(hint)

        image = getattr(self, "im", None)
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
        ColorSettingsPanel.settings[ColorMode.PV] = dict(settings)
        return settings

    def open_color_settings(self):
        self._seed_color_panel_settings_from_current_image()
        if self.color_settings_panel is None:
            # Histogram/auto-range controls must describe the derived PV image,
            # not rescan the entire source cube.  Passing self.data here caused
            # >1 GiB boolean masks/copies when the PV color panel was opened.
            pv_data = self.pv_im.get_array()
            self.color_settings_panel = ColorSettingsPanel(
                mode=ColorMode.PV,
                fits_viewer=self,
                data=pv_data,
                config=self.fits_viewer.displaymap.config,
                color_pattern= self.color_pattern,
                filename = self.fits_viewer.filename,
                bad_color = self.fits_viewer.displaymap.bad_color
            )
            self.color_settings_panel.show()
            self.color_settings_panel.destroyed.connect(self.on_color_settings_closed)
        else:
            self.color_settings_panel.raise_()
            self.color_settings_panel.activateWindow()

    def _refresh_open_pv_color_histogram(self):
        """Refresh one visible histogram from the latest debounced PV image."""
        panel = self.color_settings_panel
        if panel is None:
            return
        toggle = getattr(panel, "hist_toggle_button", None)
        if toggle is not None and not toggle.isChecked():
            return
        panel.update_histogram()

    def _sync_open_pv_color_panel_data(self):
        """Point Color Settings at the latest PV and debounce histogram work."""
        panel = self.color_settings_panel
        if panel is None:
            return
        current_pv_data = self.pv_im.get_array()
        panel.data = current_pv_data
        panel._data_nbytes = getattr(current_pv_data, "nbytes", None)

        toggle = getattr(panel, "hist_toggle_button", None)
        if toggle is not None and not toggle.isChecked():
            return
        timer = getattr(self, "_pv_color_histogram_timer", None)
        if timer is None:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._refresh_open_pv_color_histogram)
            self._pv_color_histogram_timer = timer
        timer.start(PV_COLOR_HISTOGRAM_DEBOUNCE_MS)

    def on_color_settings_closed(self):
        timer = getattr(self, "_pv_color_histogram_timer", None)
        if timer is not None:
            timer.stop()
        try:
            self._color_panel_hint = dict(ColorSettingsPanel.settings.get(ColorMode.PV, {}) or {})
            pattern = str(self._color_panel_hint.get("color_pattern") or "")
            if pattern:
                if bool(self._color_panel_hint.get("invert")) and not pattern.endswith("_r"):
                    pattern = f"{pattern}_r"
                self.color_pattern = pattern
            self.min_val = self._color_panel_hint['min_val']
            self.max_val = self._color_panel_hint['max_val']
            if self.min_val is not None and self.max_val is not None:
                self.is_clim_fixed = True

            # Set colormap with bad color
            config = self.fits_viewer.config_manager.config
            bad_color = config.get('bad_color', 'black')
            cmap = plt.get_cmap(self.color_pattern)
            cmap.set_bad(color=bad_color)
            self.pv_im.set_cmap(cmap)
            self.pv_im.set_clim(self.min_val, self.max_val)
            self.pv_canvas.draw()
        except Exception:
            pass
        self.color_settings_panel = None

    def save_fits(self):
        """Saves the current PV diagram data using core usecase."""
        ellipse_mode = self._is_ellipse_mode()
        polyline_mode = self._is_polyline_mode()
        try:
            ellipse_path = self._current_ellipse_path_geometry() if ellipse_mode else None
            polyline_path = self._current_polyline_path_geometry() if polyline_mode else None
        except (MemoryError, ValueError) as exc:
            self._handle_pv_memory_error(exc, force_update=True)
            return
        if ellipse_mode:
            if ellipse_path is None:
                QMessageBox.warning(self, "Save Error", "Please draw an ellipse PV path before saving.")
                return
            if not self.update_pv_diagram(force_update=True):
                return
        elif polyline_mode:
            if polyline_path is None:
                QMessageBox.warning(self, "Save Error", "Please draw a polyline PV path before saving.")
                return
            if not self.update_pv_diagram(force_update=True):
                return
        elif self.line_start is None or self.line_end is None:
            QMessageBox.warning(self, "Save Error", "Please draw a PV slice on the main window first before saving.")
            return

        action_params = {
            "width": float(self.sliceWidthSpin.value()),
            "position_unit": str(self.lengthUnitCombo.currentText() or "pixel"),
        }
        sample_spacing_pix = self._current_sample_spacing_pix()
        if sample_spacing_pix is not None:
            action_params["sample_spacing_pix"] = float(sample_spacing_pix)
        if int(self.weight_mode) != 0:
            action_params["weight_mode"] = int(self.weight_mode)
        if ellipse_mode:
            geom = self._normalize_ellipse_geometry(self.ellipse_geometry)
            cx, cy = geom["center"]
            action_params.update(
                {
                    "path_type": "ellipse",
                    "center_pix": [float(cx), float(cy)],
                    "semi_major_px": float(geom["semi_major"]),
                    "semi_minor_px": float(geom["semi_minor"]),
                    "pa_deg": math.degrees(float(geom["pa_rad"])),
                    "phi0_deg": math.degrees(float(geom["start_phi_rad"])),
                    "phi1_deg": math.degrees(float(geom.get("end_phi_rad", float(geom["start_phi_rad"]) + 2.0 * math.pi))),
                    "x_axis_mode": self._current_x_axis_mode(),
                    "sample_axis": self._current_x_axis_mode(),
                }
            )
            if self._is_arc_mode():
                action_params["path_type"] = "ellipse_arc"
            try:
                center_world = self.coord_converter.pix_to_world(cx, cy)
                action_params["center_world"] = [f"{float(center_world[0]):.6f}", f"{float(center_world[1]):.6f}"]
            except Exception:
                pass
        elif polyline_mode:
            vertices = [[float(x), float(y)] for (x, y) in self._current_polyline_vertices()]
            action_params.update(
                {
                    "path_type": "polyline",
                    "vertices": vertices,
                    "spline_type": normalize_pv_spline_type(self.polyline_spline_type),
                    "smoothness": clamp_pv_smoothness(self.polyline_smoothness),
                    "position_origin": POSITION_ORIGIN_START,
                    "x_axis_mode": PV_X_AXIS_POSITION,
                    "sample_axis": PV_X_AXIS_POSITION,
                }
            )
            vertices_world = []
            for x, y in vertices:
                try:
                    world = self.coord_converter.pix_to_world(x, y)
                    vertices_world.append([f"{float(world[0]):.6f}", f"{float(world[1]):.6f}"])
                except Exception:
                    vertices_world = []
                    break
            if len(vertices_world) == len(vertices):
                action_params["vertices_world"] = vertices_world
        else:
            action_params["position_origin"] = self._current_position_origin()
            try:
                start_world = self.coord_converter.pix_to_world(self.line_start[0], self.line_start[1])
                end_world = self.coord_converter.pix_to_world(self.line_end[0], self.line_end[1])
                action_params["start_world"] = [f"{float(start_world[0]):.6f}", f"{float(start_world[1]):.6f}"]
                action_params["end_world"] = [f"{float(end_world[0]):.6f}", f"{float(end_world[1]):.6f}"]
            except Exception:
                action_params.update(
                    {
                        "x0": float(self.line_start[0]),
                        "y0": float(self.line_start[1]),
                        "x1": float(self.line_end[0]),
                        "y1": float(self.line_end[1]),
                    }
                )

        history = build_processing_history_lines_with_action(
            self.fits_viewer,
            "compute_pv",
            action_params,
        )

        # Get AppState
        app_state = self.get_app_state()
        if not app_state:
            QMessageBox.warning(self, "Error", "Application state not available for export.")
            return

        # Get output path
        base_name = os.path.splitext(os.path.basename(self.fits_viewer.filename))[0]
        default_filename = f"{base_name}.pv.fits"

        desktop_dir = os.path.join(os.path.expanduser("~"), "Desktop")
        initial_dir = desktop_dir if os.access(desktop_dir, os.W_OK) else os.getcwd()
        initial_path = os.path.join(initial_dir, default_filename)

        path, _ = QFileDialog.getSaveFileName(self, "Save PV FITS", initial_path, "FITS Files (*.fits);;All Files (*)")

        if not path:
            return

        # Call usecase
        try:
            from takefits.core.usecases import export_pv_fits
            if ellipse_mode:
                ref = self._ellipse_point_normal_from_fraction(0.0)
                x0 = y0 = x1 = y1 = 0.0
                if ref is not None:
                    x0, y0 = float(ref[0]), float(ref[1])
                    x1, y1 = x0, y0
                phi_start_deg, phi_end_deg = self._ellipse_phi_bounds_deg()
                export_pv_fits(
                    app_state,
                    self.pv_im.get_array(),
                    path,
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    is_swapped=self.swapAxesCheck.isChecked(),
                    history_entries=history,
                    position_origin=POSITION_ORIGIN_START,
                    path_type=self._current_path_type(),
                    path_length_px=float(ellipse_path.length_px),
                    x_axis_mode=self._current_x_axis_mode(),
                    phi_start_deg=phi_start_deg,
                    phi_end_deg=phi_end_deg,
                    position_unit=str(self.lengthUnitCombo.currentText() or "pixel"),
                )
                QMessageBox.information(self, "Save Successful", f"FITS successfully saved as: {path}")
                return
            if polyline_mode:
                vertices = self._current_polyline_vertices()
                x0, y0 = vertices[0]
                x1, y1 = vertices[-1]
                export_pv_fits(
                    app_state,
                    self.pv_im.get_array(),
                    path,
                    x0=float(x0),
                    y0=float(y0),
                    x1=float(x1),
                    y1=float(y1),
                    is_swapped=self.swapAxesCheck.isChecked(),
                    history_entries=history,
                    position_origin=POSITION_ORIGIN_START,
                    path_type="polyline",
                    path_length_px=float(polyline_path.length_px),
                    x_axis_mode=PV_X_AXIS_POSITION,
                    position_unit=str(self.lengthUnitCombo.currentText() or "pixel"),
                )
                QMessageBox.information(self, "Save Successful", f"FITS successfully saved as: {path}")
                return
            export_pv_fits(
                app_state,
                self.pv_im.get_array(),
                path,
                x0=self.line_start[0],
                y0=self.line_start[1],
                x1=self.line_end[0],
                y1=self.line_end[1],
                is_swapped=self.swapAxesCheck.isChecked(),
                history_entries=history,
                position_origin=self._current_position_origin(),
                x_axis_mode=self._current_x_axis_mode(),
                position_unit=str(self.lengthUnitCombo.currentText() or "pixel"),
            )
            QMessageBox.information(self, "Save Successful", f"FITS successfully saved as: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save FITS:\n{str(e)}")

    def _default_path_recipe_path(self):
        source_path = str(getattr(self.fits_viewer, "filename_path", "") or "")
        default_dir = os.path.dirname(source_path) if source_path else os.getcwd()
        if not default_dir:
            default_dir = os.getcwd()
        filename = str(getattr(self.fits_viewer, "filename", "") or "pv_path")
        base_name = os.path.splitext(os.path.basename(filename))[0]
        return os.path.join(default_dir, f"{base_name}.pvpath.json")

    def _wcs_values_for_path_recipe(self, attr_name):
        wcs_obj = getattr(self.wcs, "wcs", None)
        values = getattr(wcs_obj, attr_name, None)
        if values is None:
            return []
        try:
            return [str(value) for value in values]
        except Exception:
            return []

    def _current_path_world_frame(self):
        frame = normalize_display_frame(native_celestial_frame(self.wcs))
        return frame if frame != "native" else "native"

    def _frame_from_recipe_spatial_ctype(self, recipe):
        source = recipe.get("source") if isinstance(recipe, dict) else None
        if not isinstance(source, dict):
            return "native"
        ctype_values = source.get("wcs_ctype")
        if not isinstance(ctype_values, (list, tuple)):
            return "native"
        axis_text = " ".join(str(value or "").upper() for value in ctype_values[:2])
        if "GLON" in axis_text or "GLAT" in axis_text:
            return "galactic"
        if "RA" in axis_text or "DEC" in axis_text:
            return "icrs"
        return "native"

    def _path_recipe_world_frame(self, recipe, geometry):
        frame = "native"
        if isinstance(geometry, dict):
            frame = geometry.get("world_frame", frame)
        if (not frame or normalize_display_frame(frame) == "native") and isinstance(recipe, dict):
            source = recipe.get("source")
            if isinstance(source, dict):
                frame = source.get("world_frame", frame)
        normalized = normalize_display_frame(frame)
        if normalized != "native":
            return normalized
        return self._frame_from_recipe_spatial_ctype(recipe)

    def _path_recipe_frames_are_convertible(self, recipe, geometry):
        source_frame = self._path_recipe_world_frame(recipe, geometry)
        target_frame = self._current_path_world_frame()
        return self._path_recipe_world_geometry_is_usable(source_frame, target_frame)

    def _path_recipe_world_geometry_is_usable(self, source_frame, target_frame):
        if source_frame == target_frame:
            return True
        if source_frame == "native" or target_frame == "native":
            return False
        return True

    def _convert_recipe_world_point_to_current_native(self, point, source_frame, spectral_world=None):
        point = self._coerce_recipe_point(point)
        if point is None:
            return None
        target_frame = self._current_path_world_frame()
        if source_frame == target_frame:
            return point
        if not self._path_recipe_world_geometry_is_usable(source_frame, target_frame):
            return None

        vector = build_native_world_vector(self.wcs)
        if len(vector) < 2:
            return None
        vector[0] = float(point[0])
        vector[1] = float(point[1])
        if spectral_world is not None and len(vector) >= 3:
            try:
                vector[2] = float(spectral_world)
            except Exception:
                pass
        try:
            transformed, converted = transform_world_vector_between_frames_with_status(
                vector,
                self.wcs,
                source_frame,
                "native",
            )
            if not converted:
                return None
            return [float(transformed[0]), float(transformed[1])]
        except Exception:
            return None

    def _path_recipe_source_descriptor(self):
        header = getattr(self.fits_viewer, "header", None)
        object_name = None
        if header is not None:
            try:
                object_value = header.get("OBJECT", None)
                object_name = str(object_value) if object_value not in (None, "") else None
            except Exception:
                object_name = None
        return {
            "filename": str(getattr(self.fits_viewer, "filename", "") or ""),
            "object": object_name,
            "shape": [int(value) for value in getattr(self.data, "shape", ())],
            "world_frame": self._current_path_world_frame(),
            "wcs_ctype": self._wcs_values_for_path_recipe("ctype"),
            "wcs_cunit": self._wcs_values_for_path_recipe("cunit"),
        }

    def _build_straight_path_recipe_from_state(self, workspace_state):
        if not isinstance(workspace_state, dict):
            return None
        line_pixel = workspace_state.get("line_pixel")
        line_world_raw = workspace_state.get("line_world_raw")
        if not isinstance(line_pixel, dict) and not isinstance(line_world_raw, dict):
            return None

        geometry = {
            "world_frame": self._current_path_world_frame(),
            "start_world": None,
            "end_world": None,
            "pixel_cache": line_pixel,
        }
        if isinstance(line_world_raw, dict):
            geometry["start_world"] = line_world_raw.get("start")
            geometry["end_world"] = line_world_raw.get("end")
        if workspace_state.get("spectral_world") is not None:
            geometry["spectral_world"] = workspace_state.get("spectral_world")

        width_pix = float(workspace_state.get("slice_width", self.sliceWidthSpin.value()))
        width_world = None
        if self.pixel_scale_deg is not None:
            width_world = {
                "value": float(width_pix * self.pixel_scale_deg * 3600.0),
                "unit": "arcsec",
            }

        extraction = {
            "width_pix": width_pix,
            "width_world": width_world,
            "sample_spacing_pix": self._coerce_sample_spacing_pix(workspace_state.get("sample_spacing_pix")),
            "position_origin": workspace_state.get("position_origin", POSITION_ORIGIN_START),
            "geometry_input_mode": workspace_state.get("geometry_input_mode", "endpoints"),
            "direction": "forward",
            "weight_mode": int(workspace_state.get("weight_mode", self.weight_mode)),
        }

        return {
            "schema": 1,
            "kind": "takefits_pv_path",
            "path_type": "straight",
            "source": self._path_recipe_source_descriptor(),
            "geometry": geometry,
            "extraction": extraction,
            "display": {
                "length_unit": str(workspace_state.get("length_unit") or self.lengthUnitCombo.currentText() or "pixel"),
                "x_axis_mode": workspace_state.get("x_axis_mode", PV_X_AXIS_POSITION),
                "swap_axes": bool(workspace_state.get("swap_axes", self.swapAxesCheck.isChecked())),
                "position_axis_flipped": bool(workspace_state.get("position_axis_flipped", self._position_axis_flipped())),
            },
        }

    def _build_straight_path_recipe(self):
        if self.line_start is None or self.line_end is None:
            return None
        return self._build_straight_path_recipe_from_state(self._export_single_path_state())

    def _build_ellipse_path_recipe_from_state(self, workspace_state):
        if not isinstance(workspace_state, dict):
            return None
        geom = self._normalize_ellipse_geometry(workspace_state.get("ellipse_geometry"))
        if geom is None:
            return None
        semi_major = float(geom["semi_major"])
        semi_minor = float(geom["semi_minor"])
        if semi_major <= 0.0 or semi_minor <= 0.0:
            return None
        cx, cy = geom["center"]

        def _pixel_world(px, py):
            try:
                world = self._pixel_to_world_xy_for_workspace(px, py)
                if world is not None:
                    return [float(world[0]), float(world[1])]
            except Exception:
                pass
            return None

        # Axis tip points let the loader rebuild orientation (PA) and the
        # semi-axes from sky coordinates, so the ellipse reproduces correctly
        # across different celestial frames / projections (e.g. l,b -> RA/Dec),
        # mirroring how the straight slit stores both endpoints in world space.
        pa = float(geom["pa_rad"])
        cos_pa, sin_pa = math.cos(pa), math.sin(pa)
        major_tip_px = (cx + semi_major * cos_pa, cy + semi_major * sin_pa)
        minor_tip_px = (cx - semi_minor * sin_pa, cy + semi_minor * cos_pa)

        center_world = _pixel_world(cx, cy)
        major_axis_world = _pixel_world(*major_tip_px)
        minor_axis_world = _pixel_world(*minor_tip_px)

        def _angular(value_pix):
            if self.pixel_scale_deg is None:
                return None
            return {
                "value": float(self._convert_length(value_pix, "pixel", "deg")),
                "unit": "deg",
            }

        geometry = {
            "world_frame": self._current_path_world_frame(),
            "center_world": center_world,
            "center_pixel": [float(cx), float(cy)],
            "major_axis_world": major_axis_world,
            "minor_axis_world": minor_axis_world,
            "major_axis_pixel": [float(major_tip_px[0]), float(major_tip_px[1])],
            "minor_axis_pixel": [float(minor_tip_px[0]), float(minor_tip_px[1])],
            "semi_major_pix": semi_major,
            "semi_minor_pix": semi_minor,
            "semi_major_world": _angular(semi_major),
            "semi_minor_world": _angular(semi_minor),
            "pa_deg": math.degrees(float(geom["pa_rad"])),
            "start_phi_deg": self._normalize_phi_deg(float(geom["start_phi_rad"])),
            "end_phi_deg": self._normalize_phi_deg(float(geom.get("end_phi_rad", float(geom["start_phi_rad"]) + 2.0 * math.pi))),
        }
        spectral_world = workspace_state.get("spectral_world")
        if spectral_world is None:
            spectral_world = self._current_spectral_world_for_workspace()
        if spectral_world is not None:
            geometry["spectral_world"] = spectral_world

        width_pix = float(workspace_state.get("slice_width", self.sliceWidthSpin.value()))
        width_world = None
        if self.pixel_scale_deg is not None:
            width_world = {
                "value": float(width_pix * self.pixel_scale_deg * 3600.0),
                "unit": "arcsec",
            }

        extraction = {
            "width_pix": width_pix,
            "width_world": width_world,
            "sample_spacing_pix": self._coerce_sample_spacing_pix(workspace_state.get("sample_spacing_pix")),
            "weight_mode": int(workspace_state.get("weight_mode", self.weight_mode)),
        }

        return {
            "schema": 1,
            "kind": "takefits_pv_path",
            "path_type": self._normalize_path_type_value(workspace_state.get("path_type", "ellipse")),
            "source": self._path_recipe_source_descriptor(),
            "geometry": geometry,
            "extraction": extraction,
            "display": {
                "length_unit": str(workspace_state.get("length_unit") or self.lengthUnitCombo.currentText() or "pixel"),
                "ellipse_axis_unit": str(workspace_state.get("ellipse_axis_unit") or self._current_ellipse_axis_unit()),
                "x_axis_mode": normalize_pv_x_axis_mode(workspace_state.get("x_axis_mode", self._current_x_axis_mode())),
                "swap_axes": bool(workspace_state.get("swap_axes", self.swapAxesCheck.isChecked())),
                "position_axis_flipped": bool(workspace_state.get("position_axis_flipped", self._position_axis_flipped())),
            },
        }

    def _build_ellipse_path_recipe(self):
        if self._normalize_ellipse_geometry(self.ellipse_geometry) is None:
            return None
        return self._build_ellipse_path_recipe_from_state(self._export_single_path_state())

    def _build_polyline_path_recipe_from_state(self, workspace_state):
        if not isinstance(workspace_state, dict):
            return None
        block = workspace_state.get("polyline")
        if not isinstance(block, dict):
            return None
        pixel = block.get("vertices")
        world = block.get("vertices_world")
        has_pixel = isinstance(pixel, list) and len(pixel) >= 2
        has_world = isinstance(world, list) and len(world) >= 2
        if not has_pixel and not has_world:
            return None
        geometry = {
            "world_frame": self._current_path_world_frame(),
            "vertices_world": world if has_world else None,
            "pixel_cache": {"vertices": pixel} if has_pixel else None,
        }
        spline_type, smoothness = self._polyline_spline_from_state(workspace_state)
        if spline_type != PV_SPLINE_NONE:
            geometry["spline"] = {"type": spline_type, "smoothness": smoothness}
        if workspace_state.get("spectral_world") is not None:
            geometry["spectral_world"] = workspace_state.get("spectral_world")
        width_pix = float(workspace_state.get("slice_width", self.sliceWidthSpin.value()))
        width_world = None
        if self.pixel_scale_deg is not None:
            width_world = {"value": float(width_pix * self.pixel_scale_deg * 3600.0), "unit": "arcsec"}
        extraction = {
            "width_pix": width_pix,
            "width_world": width_world,
            "sample_spacing_pix": self._coerce_sample_spacing_pix(workspace_state.get("sample_spacing_pix")),
            "position_origin": POSITION_ORIGIN_START,
            "geometry_input_mode": "polyline",
            "direction": "forward",
            "weight_mode": int(workspace_state.get("weight_mode", self.weight_mode)),
        }
        return {
            "schema": 1,
            "kind": "takefits_pv_path",
            "path_type": "polyline",
            "source": self._path_recipe_source_descriptor(),
            "geometry": geometry,
            "extraction": extraction,
            "display": {
                "length_unit": str(workspace_state.get("length_unit") or self.lengthUnitCombo.currentText() or "pixel"),
                "x_axis_mode": PV_X_AXIS_POSITION,
                "swap_axes": bool(workspace_state.get("swap_axes", self.swapAxesCheck.isChecked())),
                "position_axis_flipped": bool(workspace_state.get("position_axis_flipped", self._position_axis_flipped())),
            },
        }

    def _build_polyline_path_recipe(self):
        if len(self._current_polyline_vertices()) < 2:
            return None
        return self._build_polyline_path_recipe_from_state(self._export_single_path_state())

    def _build_path_recipe_from_state(self, state):
        path_type = self._normalize_path_type_value((state or {}).get("path_type"))
        if self._is_polyline_path_type(path_type):
            return self._build_polyline_path_recipe_from_state(state)
        if self._is_ellipse_path_type(path_type):
            return self._build_ellipse_path_recipe_from_state(state)
        return self._build_straight_path_recipe_from_state(state)

    def _build_path_set_recipe(self):
        self._sync_active_path_item_from_current_state()
        paths = []
        for item in self.pv_path_items:
            recipe = self._build_path_recipe_from_state(item.get("state") or {})
            if recipe is None:
                continue
            paths.append(
                {
                    "id": item.get("id"),
                    "name": item.get("name"),
                    "visible": bool(item.get("visible", True)),
                    "path_type": recipe.get("path_type"),
                    "recipe": recipe,
                }
            )
        if not paths:
            return None
        return {
            "schema": 1,
            "kind": "takefits_pv_path_set",
            "source": self._path_recipe_source_descriptor(),
            "active_path_id": self.active_pv_path_id,
            "paths": paths,
        }

    def save_path_recipe(self):
        self._sync_active_path_item_from_current_state()
        if len(self.pv_path_items) > 1:
            recipe = self._build_path_set_recipe()
            if recipe is None:
                QMessageBox.warning(self, "Save Path", "Please draw a PV path before saving.")
                return
        elif self._is_polyline_mode():
            recipe = self._build_polyline_path_recipe()
            if recipe is None:
                QMessageBox.warning(self, "Save Path", "Please draw a polyline PV path before saving a path.")
                return
        elif self._is_ellipse_mode():
            recipe = self._build_ellipse_path_recipe()
            if recipe is None:
                QMessageBox.warning(self, "Save Path", "Please draw an ellipse PV path before saving a path.")
                return
        else:
            recipe = self._build_straight_path_recipe()
            if recipe is None:
                QMessageBox.warning(self, "Save Path", "Please draw a straight PV slit before saving a path.")
                return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Path",
            self._default_path_recipe_path(),
            "PV Path Files (*.pvpath.json *.json);;All Files (*)",
        )
        if not path:
            return
        if not path.lower().endswith((".pvpath.json", ".json")):
            path += ".pvpath.json"

        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(recipe, handle, ensure_ascii=True, indent=2)
                handle.write("\n")
            QMessageBox.information(self, "Save Path", f"Saved PV path:\n{path}")
        except Exception as exc:
            QMessageBox.warning(self, "Save Path", f"Failed to save PV path:\n{exc}")

    def _coerce_recipe_point(self, value):
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            return None
        try:
            return [float(value[0]), float(value[1])]
        except Exception:
            return None

    def _path_recipe_type(self, recipe):
        return str(recipe.get("path_type") or "").strip().lower()

    def _validate_path_recipe_header(self, recipe):
        if not isinstance(recipe, dict):
            raise ValueError("Path recipe must be a JSON object.")
        if recipe.get("kind") not in ("takefits_pv_path", "takefits_pv_path_set"):
            raise ValueError("Not a TakeFITS PV path recipe.")

    def _workspace_state_from_path_recipe(self, recipe):
        self._validate_path_recipe_header(recipe)
        if recipe.get("kind") == "takefits_pv_path_set":
            return self._workspace_state_from_path_set_recipe(recipe)
        path_type = self._path_recipe_type(recipe)
        if path_type == "straight":
            return self._workspace_state_from_straight_path_recipe(recipe)
        if path_type == "polyline":
            return self._workspace_state_from_polyline_path_recipe(recipe)
        if self._is_ellipse_path_type(path_type):
            return self._workspace_state_from_ellipse_path_recipe(recipe)
        raise ValueError("Only Straight, Polyline, Ellipse, and Ellipse Arc PV path recipes are supported.")

    def _workspace_state_from_path_set_recipe(self, recipe):
        paths_raw = recipe.get("paths")
        if not isinstance(paths_raw, list) or not paths_raw:
            raise ValueError("Path set recipe is missing paths.")

        paths = []
        active_path_id = recipe.get("active_path_id")
        for index, entry in enumerate(paths_raw):
            if not isinstance(entry, dict):
                continue
            child_recipe = entry.get("recipe")
            if not isinstance(child_recipe, dict):
                child_recipe = entry
            if child_recipe.get("kind") != "takefits_pv_path":
                continue
            state = self._workspace_state_from_path_recipe(child_recipe)
            if not self._single_path_state_has_geometry(state):
                continue
            path_type = self._normalize_path_type_value(state.get("path_type") or child_recipe.get("path_type"))
            path_id = str(entry.get("id") or f"pvpath-{index + 1}")
            paths.append(
                {
                    "id": path_id,
                    "name": str(entry.get("name") or self._path_type_label(path_type)),
                    "path_type": path_type,
                    "visible": bool(entry.get("visible", True)),
                    "state": state,
                }
            )

        if not paths:
            raise ValueError("Path set recipe has no usable paths.")
        if not any(item.get("id") == active_path_id for item in paths):
            active_path_id = paths[0].get("id")
        active_state = dict(paths[0]["state"])
        for item in paths:
            if item.get("id") == active_path_id:
                active_state = dict(item["state"])
                break
        active_state["pv_paths"] = paths
        active_state["active_pv_path_id"] = active_path_id
        return active_state

    def _workspace_state_from_straight_path_recipe(self, recipe):
        geometry = recipe.get("geometry")
        if not isinstance(geometry, dict):
            raise ValueError("Path recipe is missing geometry.")

        line_pixel = None
        pixel_cache = geometry.get("pixel_cache")
        if isinstance(pixel_cache, dict):
            start_pixel = self._coerce_recipe_point(pixel_cache.get("start"))
            end_pixel = self._coerce_recipe_point(pixel_cache.get("end"))
            if start_pixel is not None and end_pixel is not None:
                line_pixel = {"start": start_pixel, "end": end_pixel}

        line_world_raw = None
        source_frame = self._path_recipe_world_frame(recipe, geometry)
        target_frame = self._current_path_world_frame()
        raw_start_world = self._coerce_recipe_point(geometry.get("start_world"))
        raw_end_world = self._coerce_recipe_point(geometry.get("end_world"))
        world_geometry_available = raw_start_world is not None and raw_end_world is not None
        start_world = self._convert_recipe_world_point_to_current_native(
            raw_start_world,
            source_frame,
            spectral_world=geometry.get("spectral_world"),
        )
        end_world = self._convert_recipe_world_point_to_current_native(
            raw_end_world,
            source_frame,
            spectral_world=geometry.get("spectral_world"),
        )
        if start_world is not None and end_world is not None:
            line_world_raw = {"start": start_world, "end": end_world}

        if line_pixel is None and line_world_raw is None:
            raise ValueError("Path recipe has no usable pixel or world geometry.")

        extraction = recipe.get("extraction")
        if not isinstance(extraction, dict):
            extraction = {}
        display = recipe.get("display")
        if not isinstance(display, dict):
            display = {}
        x_axis_mode = normalize_pv_x_axis_mode(display.get("x_axis_mode", PV_X_AXIS_POSITION))
        if x_axis_mode != PV_X_AXIS_POSITION:
            x_axis_mode = PV_X_AXIS_POSITION

        position_origin = normalize_position_origin(extraction.get("position_origin", POSITION_ORIGIN_START))
        geometry_input_mode = str(extraction.get("geometry_input_mode") or "endpoints")
        if geometry_input_mode not in ("endpoints", "center"):
            geometry_input_mode = "endpoints"
        if geometry_input_mode == "center":
            position_origin = POSITION_ORIGIN_CENTER
        sample_spacing_pix = self._coerce_sample_spacing_pix(extraction.get("sample_spacing_pix"))

        state = {
            "schema": 1,
            "line_pixel": line_pixel,
            "line_world_raw": line_world_raw,
            "line_world": None,
            "spectral_world": geometry.get("spectral_world"),
            "slice_width": float(extraction.get("width_pix", self.sliceWidthSpin.value())),
            "sample_spacing_pix": sample_spacing_pix,
            "weight_mode": int(extraction.get("weight_mode", self.weight_mode)),
            "position_origin": position_origin,
            "geometry_input_mode": geometry_input_mode,
            "swap_axes": bool(display.get("swap_axes", self.swapAxesCheck.isChecked())),
            "position_axis_flipped": bool(display.get("position_axis_flipped", self._position_axis_flipped())),
            "auto_update": bool(self.autoUpdateCheck.isChecked()),
            "length_unit": str(display.get("length_unit") or self.lengthUnitCombo.currentText() or "pixel"),
            "x_axis_mode": x_axis_mode,
        }
        state["_path_recipe_load_meta"] = {
            "geometry_source": "world" if line_world_raw is not None else "pixel_cache",
            "world_geometry_available": bool(world_geometry_available),
            "world_geometry_failed": bool(world_geometry_available and line_world_raw is None),
            "source_frame": source_frame,
            "target_frame": target_frame,
        }
        if line_world_raw is not None:
            state["line_world"] = {
                "start": [f"{line_world_raw['start'][0]:.12g}", f"{line_world_raw['start'][1]:.12g}"],
                "end": [f"{line_world_raw['end'][0]:.12g}", f"{line_world_raw['end'][1]:.12g}"],
            }
        return state

    def _workspace_state_from_polyline_path_recipe(self, recipe):
        geometry = recipe.get("geometry")
        if not isinstance(geometry, dict):
            raise ValueError("Polyline path recipe is missing geometry.")

        # Prefer world vertices (portable across cubes); fall back to the pixel
        # cache. If any world vertex fails to convert, use the pixel cache wholesale.
        pixel_verts = []
        source_frame = self._path_recipe_world_frame(recipe, geometry)
        spectral_world = geometry.get("spectral_world")
        world_raw = geometry.get("vertices_world")
        if isinstance(world_raw, list) and len(world_raw) >= 2:
            ok = True
            for vw in world_raw:
                pt = self._coerce_recipe_point(vw)
                native = self._convert_recipe_world_point_to_current_native(
                    pt, source_frame, spectral_world=spectral_world
                ) if pt is not None else None
                px = None
                if native is not None:
                    try:
                        px = self._world_xy_to_pixel_for_workspace(
                            native[0], native[1], spectral_world=spectral_world
                        )
                    except Exception:
                        px = None
                if px is None:
                    ok = False
                    break
                pixel_verts.append([float(px[0]), float(px[1])])
            if not ok:
                pixel_verts = []

        if len(pixel_verts) < 2:
            pixel_verts = []
            cache = geometry.get("pixel_cache")
            raw = cache.get("vertices") if isinstance(cache, dict) else None
            if isinstance(raw, list):
                for v in raw:
                    pt = self._coerce_recipe_point(v)
                    if pt is not None:
                        pixel_verts.append([float(pt[0]), float(pt[1])])

        if len(pixel_verts) < 2:
            raise ValueError("Polyline path recipe has no usable pixel or world geometry.")

        extraction = recipe.get("extraction") if isinstance(recipe.get("extraction"), dict) else {}
        display = recipe.get("display") if isinstance(recipe.get("display"), dict) else {}
        polyline_block = {"vertices": pixel_verts}
        spline = geometry.get("spline")
        if isinstance(spline, dict):
            spline_type = normalize_pv_spline_type(spline.get("type"))
            if spline_type != PV_SPLINE_NONE:
                polyline_block["spline"] = {
                    "type": spline_type,
                    "smoothness": clamp_pv_smoothness(spline.get("smoothness", 1.0)),
                }
        elif geometry.get("smooth"):  # legacy boolean alias
            polyline_block["spline"] = {"type": PV_SPLINE_CATMULL_ROM, "smoothness": 1.0}
        return {
            "schema": 1,
            "path_type": "polyline",
            "polyline": polyline_block,
            "spectral_world": spectral_world,
            "slice_width": float(extraction.get("width_pix", self.sliceWidthSpin.value())),
            "sample_spacing_pix": self._coerce_sample_spacing_pix(extraction.get("sample_spacing_pix")),
            "weight_mode": int(extraction.get("weight_mode", self.weight_mode)),
            "position_origin": POSITION_ORIGIN_START,
            "geometry_input_mode": "endpoints",
            "swap_axes": bool(display.get("swap_axes", self.swapAxesCheck.isChecked())),
            "position_axis_flipped": bool(display.get("position_axis_flipped", self._position_axis_flipped())),
            "auto_update": bool(self.autoUpdateCheck.isChecked()),
            "length_unit": str(display.get("length_unit") or self.lengthUnitCombo.currentText() or "pixel"),
            "x_axis_mode": PV_X_AXIS_POSITION,
        }

    def _workspace_state_from_ellipse_path_recipe(self, recipe):
        geometry = recipe.get("geometry")
        if not isinstance(geometry, dict):
            raise ValueError("Path recipe is missing geometry.")

        pixel_center = self._coerce_recipe_point(geometry.get("center_pixel"))
        source_frame = self._path_recipe_world_frame(recipe, geometry)
        target_frame = self._current_path_world_frame()
        raw_center_world = self._coerce_recipe_point(geometry.get("center_world"))
        world_geometry_available = raw_center_world is not None
        spectral_world = geometry.get("spectral_world")
        try:
            spectral_value = float(spectral_world) if spectral_world is not None else None
        except Exception:
            spectral_value = None
        center_world = self._convert_recipe_world_point_to_current_native(
            raw_center_world,
            source_frame,
            spectral_world=spectral_world,
        )

        resolved_center = None
        geometry_source = "world"
        if center_world is not None:
            resolved_center = self._world_xy_to_pixel_for_workspace(
                center_world[0],
                center_world[1],
                spectral_world=spectral_value,
                reference=tuple(pixel_center) if pixel_center else None,
            )
        if resolved_center is None:
            if pixel_center is None:
                raise ValueError("Path recipe has no usable pixel or world geometry.")
            resolved_center = (float(pixel_center[0]), float(pixel_center[1]))
            geometry_source = "pixel_cache"

        def _resolve_tip_pixel(world_field, pixel_field):
            raw_world = self._coerce_recipe_point(geometry.get(world_field))
            converted = self._convert_recipe_world_point_to_current_native(
                raw_world,
                source_frame,
                spectral_world=spectral_world,
            )
            pix_cache = self._coerce_recipe_point(geometry.get(pixel_field))
            if converted is not None:
                tip = self._world_xy_to_pixel_for_workspace(
                    converted[0],
                    converted[1],
                    spectral_world=spectral_value,
                    reference=tuple(pix_cache) if pix_cache else None,
                )
                if tip is not None:
                    return (float(tip[0]), float(tip[1]))
            if pix_cache is not None:
                return (float(pix_cache[0]), float(pix_cache[1]))
            return None

        def _axis_pix(world_field, pix_field):
            info = geometry.get(world_field)
            if isinstance(info, dict) and self.pixel_scale_deg is not None:
                try:
                    value = float(info.get("value"))
                    unit = str(info.get("unit") or "deg")
                    return float(self._convert_length(value, unit, "pixel"))
                except Exception:
                    pass
            try:
                return float(geometry.get(pix_field))
            except Exception:
                return None

        # Prefer reconstructing PA and the semi-axes from the axis-tip points so
        # the orientation transfers across celestial frames. Fall back to the
        # stored angular sizes plus pixel PA when tips are unavailable.
        semi_major = None
        semi_minor = None
        pa_rad = None
        major_tip = _resolve_tip_pixel("major_axis_world", "major_axis_pixel")
        minor_tip = _resolve_tip_pixel("minor_axis_world", "minor_axis_pixel")
        if major_tip is not None:
            major_vec = (major_tip[0] - resolved_center[0], major_tip[1] - resolved_center[1])
            major_len = math.hypot(major_vec[0], major_vec[1])
            if major_len > 1e-9:
                semi_major = major_len
                pa_rad = math.atan2(major_vec[1], major_vec[0])
                if minor_tip is not None:
                    minor_len = math.hypot(
                        minor_tip[0] - resolved_center[0],
                        minor_tip[1] - resolved_center[1],
                    )
                    if minor_len > 1e-9:
                        semi_minor = minor_len

        if semi_major is None:
            semi_major = _axis_pix("semi_major_world", "semi_major_pix")
        if semi_minor is None:
            semi_minor = _axis_pix("semi_minor_world", "semi_minor_pix")
        if pa_rad is None:
            pa_rad = math.radians(float(geometry.get("pa_deg", 0.0) or 0.0))
        if (
            semi_major is None or semi_minor is None
            or semi_major <= 0.0 or semi_minor <= 0.0
        ):
            raise ValueError("Path recipe has invalid ellipse semi-axes.")

        path_type = self._path_recipe_type(recipe)
        start_phi_rad = math.radians(float(geometry.get("start_phi_deg", 0.0) or 0.0))
        end_phi_rad = math.radians(float(geometry.get("end_phi_deg", 360.0) or 360.0))
        if not self._is_arc_path_type(path_type):
            end_phi_rad = start_phi_rad + 2.0 * math.pi

        ellipse_geometry = {
            "center": [float(resolved_center[0]), float(resolved_center[1])],
            "semi_major": float(semi_major),
            "semi_minor": float(semi_minor),
            "pa_rad": float(pa_rad),
            "start_phi_rad": float(start_phi_rad),
            "end_phi_rad": float(end_phi_rad),
        }

        extraction = recipe.get("extraction")
        if not isinstance(extraction, dict):
            extraction = {}
        display = recipe.get("display")
        if not isinstance(display, dict):
            display = {}

        x_axis_mode = normalize_pv_x_axis_mode(display.get("x_axis_mode", PV_X_AXIS_POSITION))

        sample_spacing_pix = self._coerce_sample_spacing_pix(extraction.get("sample_spacing_pix"))

        state = {
            "schema": 1,
            "path_type": path_type,
            "ellipse_geometry": ellipse_geometry,
            "line_pixel": None,
            "line_world_raw": None,
            "line_world": None,
            "spectral_world": spectral_world,
            "slice_width": float(extraction.get("width_pix", self.sliceWidthSpin.value())),
            "sample_spacing_pix": sample_spacing_pix,
            "weight_mode": int(extraction.get("weight_mode", self.weight_mode)),
            "position_origin": POSITION_ORIGIN_START,
            "geometry_input_mode": "endpoints",
            "swap_axes": bool(display.get("swap_axes", self.swapAxesCheck.isChecked())),
            "position_axis_flipped": bool(display.get("position_axis_flipped", self._position_axis_flipped())),
            "auto_update": bool(self.autoUpdateCheck.isChecked()),
            "length_unit": str(display.get("length_unit") or self.lengthUnitCombo.currentText() or "pixel"),
            "ellipse_axis_unit": str(display.get("ellipse_axis_unit") or self._current_ellipse_axis_unit() or "pixel"),
            "x_axis_mode": x_axis_mode,
        }
        state["_path_recipe_load_meta"] = {
            "geometry_source": geometry_source,
            "world_geometry_available": bool(world_geometry_available),
            "world_geometry_failed": bool(world_geometry_available and center_world is None),
            "source_frame": source_frame,
            "target_frame": target_frame,
        }
        return state

    def _path_recipe_candidate_line_pixels(self, state):
        start = None
        end = None
        line_pixel = state.get("line_pixel")
        reference_start = None
        reference_end = None
        if isinstance(line_pixel, dict):
            reference_start = self._coerce_recipe_point(line_pixel.get("start"))
            reference_end = self._coerce_recipe_point(line_pixel.get("end"))

        spectral_world = state.get("spectral_world")
        try:
            spectral_world = float(spectral_world) if spectral_world is not None else None
        except Exception:
            spectral_world = None

        world_line_raw = state.get("line_world_raw")
        if isinstance(world_line_raw, dict):
            start_world = self._coerce_recipe_point(world_line_raw.get("start"))
            end_world = self._coerce_recipe_point(world_line_raw.get("end"))
            if start_world is not None and end_world is not None:
                start = self._world_xy_to_pixel_for_workspace(
                    start_world[0],
                    start_world[1],
                    spectral_world=spectral_world,
                    reference=reference_start,
                )
                end = self._world_xy_to_pixel_for_workspace(
                    end_world[0],
                    end_world[1],
                    spectral_world=spectral_world,
                    reference=reference_end,
                )

        if start is None or end is None:
            if reference_start is not None and reference_end is not None:
                start = (float(reference_start[0]), float(reference_start[1]))
                end = (float(reference_end[0]), float(reference_end[1]))
        return start, end

    def _path_recipe_load_warnings(self, state):
        warnings = []
        meta = state.get("_path_recipe_load_meta")
        if not isinstance(meta, dict):
            meta = {}

        if meta.get("geometry_source") == "pixel_cache":
            if meta.get("world_geometry_available"):
                warnings.append(
                    f"World coordinates could not be converted from {meta.get('source_frame', 'unknown')} "
                    f"to the current {meta.get('target_frame', 'unknown')} frame. Pixel cache will be used."
                )
            else:
                warnings.append("This path has no saved world coordinates. Pixel cache will be used.")

        ellipse_geometry = state.get("ellipse_geometry")
        if isinstance(ellipse_geometry, dict):
            center = self._coerce_recipe_point(ellipse_geometry.get("center"))
            if center is not None:
                max_x = max(0.0, float(self.data.shape[-1] - 1))
                max_y = max(0.0, float(self.data.shape[-2] - 1))
                if center[0] < 0.0 or center[0] > max_x or center[1] < 0.0 or center[1] > max_y:
                    warnings.append("The ellipse center is outside the current image.")
            return warnings

        start, end = self._path_recipe_candidate_line_pixels(state)
        if start is not None and end is not None:
            max_x = max(0.0, float(self.data.shape[-1] - 1))
            max_y = max(0.0, float(self.data.shape[-2] - 1))
            out_of_bounds = False
            for x, y in (start, end):
                if x < 0.0 or x > max_x or y < 0.0 or y > max_y:
                    out_of_bounds = True
                    break
            if out_of_bounds:
                warnings.append("One or both path endpoints are outside the current image and will be clipped.")
        return warnings

    def _confirm_path_recipe_load_warnings(self, warnings):
        if not warnings:
            return True

        message = "The path may not reproduce exactly in the current cube:\n\n"
        message += "\n".join(f"- {item}" for item in warnings)
        message += "\n\nLoad it anyway?"
        reply = QMessageBox.question(
            self,
            "Load Path",
            message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return reply == QMessageBox.StandardButton.Yes

    def load_path_recipe(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Path",
            os.path.dirname(self._default_path_recipe_path()) or os.getcwd(),
            "PV Path Files (*.pvpath.json *.json);;All Files (*)",
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as handle:
                recipe = json.load(handle)
            state = self._workspace_state_from_path_recipe(recipe)
            if not self._confirm_path_recipe_load_warnings(self._path_recipe_load_warnings(state)):
                return
            restored = self.restore_workspace_state(state)
            if not restored:
                QMessageBox.warning(self, "Load Path", "The path recipe could not be restored in the current cube.")
                return
            self._record_slit_change()
            QMessageBox.information(self, "Load Path", f"Loaded PV path:\n{path}")
        except Exception as exc:
            QMessageBox.warning(self, "Load Path", f"Failed to load PV path:\n{exc}")


    def _remove_artist(self, artist):
        if artist is None:
            return
        try:
            if getattr(artist, "axes", None) is not None:
                artist.remove()
        except Exception as e:
            print(f"Error removing artist: {e}")

    def _clear_straight_path(self):
        """Clear straight-line artists and state without touching ellipse geometry."""
        artists = [
            self.arrow_artist,
            self.pos_indicator_on_arrow,
            self.marker_artist_start,
            self.marker_artist_end,
            self.center_marker,
        ]
        artists.extend(self.width_indicators or [])
        for artist in artists:
            self._remove_artist(artist)
        self.arrow_artist = None
        self.pos_indicator_on_arrow = None
        self.marker_artist_start = None
        self.marker_artist_end = None
        self.center_marker = None
        self.width_indicators = []
        self.line_start = None
        self.line_end = None
        self.line_fixed = False
        self.edit_mode = None
        self.drag_mode = None
        self.dragging_endpoint = None
        self.dragging_pos_indicator = False

    def _clear_ellipse_path(self):
        """Clear ellipse artists and state without touching straight-line geometry."""
        self._remove_artist(self.ellipse_artist)
        self.ellipse_artist = None
        for artist in list((self.ellipse_handle_artists or {}).values()):
            self._remove_artist(artist)
        self.ellipse_handle_artists = {}
        self._clear_ellipse_indicator_ticks()
        self.ellipse_geometry = None
        self.ellipse_drag_anchor = None
        self.initial_ellipse_geometry = None
        self.ellipse_resize_handle = None
        self.ellipse_rotation_reference_angle = None
        if self.edit_mode and str(self.edit_mode).startswith("ellipse"):
            self.edit_mode = None
        if self.drag_mode == "ellipse_draw":
            self.drag_mode = None

    _POLYLINE_ARTIST_ATTRS = (
        "polyline_line_artist",
        "polyline_node_artist",
        "polyline_rubber_artist",
        "polyline_select_artist",
    )

    def _clear_polyline_path(self):
        """Clear polyline artists and state without touching straight/ellipse geometry."""
        for attr in self._POLYLINE_ARTIST_ATTRS:
            self._remove_artist(getattr(self, attr, None))
            setattr(self, attr, None)
        for art in (self.polyline_indicator_artists or []):
            self._remove_artist(art)
        self.polyline_indicator_artists = []
        self.polyline_vertices = []
        self.polyline_finished = False
        self.polyline_selected_index = None
        self.polyline_extend_from_start = False

    def _draw_polyline_overlay(self, rubber_xy=None):
        """Render the active polyline (line + node markers + selection + rubber band)."""
        for attr in self._POLYLINE_ARTIST_ATTRS:
            self._remove_artist(getattr(self, attr, None))
            setattr(self, attr, None)
        for art in (self.polyline_indicator_artists or []):
            self._remove_artist(art)
        self.polyline_indicator_artists = []
        verts = self._current_polyline_vertices()
        color = str(getattr(self, "pvarrow_color", "yellow") or "yellow")
        lw = max(1.0, 0.9 * float(self.arrow_size))
        if verts:
            # Node markers always sit on the raw control points; the connecting
            # line follows the densified spline curve when Smooth is on.
            xs = [v[0] for v in verts]
            ys = [v[1] for v in verts]
            if len(verts) >= 2:
                geom = self._current_polyline_path_geometry()
                if geom is not None and geom.is_smooth:
                    curve = geom.effective_vertices
                    line_xs = [p[0] for p in curve]
                    line_ys = [p[1] for p in curve]
                else:
                    line_xs, line_ys = xs, ys
                (self.polyline_line_artist,) = self.fits_ax.plot(
                    line_xs, line_ys, color=color, linewidth=lw, zorder=5,
                    solid_capstyle="round",
                )
            if self.show_node_markers:
                (self.polyline_node_artist,) = self.fits_ax.plot(
                    xs, ys, linestyle="None", marker="o",
                    markersize=max(3.0, 1.6 * float(self.arrow_size)),
                    markerfacecolor=color, markeredgecolor=color, zorder=6,
                )
                sel = self.polyline_selected_index
                if sel is not None and 0 <= sel < len(verts):
                    sx, sy = verts[sel]
                    (self.polyline_select_artist,) = self.fits_ax.plot(
                        [sx], [sy], linestyle="None", marker="o",
                        markersize=max(6.0, 2.6 * float(self.arrow_size)),
                        markerfacecolor="none", markeredgecolor=color,
                        markeredgewidth=1.8, zorder=7,
                    )
        # Start/center/end indicator ticks (perpendicular marks), toggled by the
        # shared start/center/end checkboxes. Sized exactly like the straight
        # path's indicators: total length == slice width, lw == 1.5 * arrow_size.
        if len(verts) >= 2:
            half = float(self.sliceWidthSpin.value()) / 2.0
            ind_lw = 1.5 * float(self.arrow_size)
            positions = getattr(self, "indicator_positions", {}) or {}
            for key, frac in (("start", 0.0), ("center", 0.5), ("end", 1.0)):
                if not positions.get(key):
                    continue
                pn = self._polyline_point_normal_from_fraction(frac)
                if pn is None:
                    continue
                px, py, nx, ny = pn
                tick, = self.fits_ax.plot(
                    [px - nx * half, px + nx * half],
                    [py - ny * half, py + ny * half],
                    color=color, linewidth=ind_lw, zorder=6,
                )
                self.polyline_indicator_artists.append(tick)
        if rubber_xy is not None and verts and not self.polyline_finished:
            rx, ry = float(rubber_xy[0]), float(rubber_xy[1])
            # Anchor the preview at the end being extended (start node when
            # re-extending a path from its first vertex, else the last vertex).
            lx, ly = verts[0] if self.polyline_extend_from_start else verts[-1]
            (self.polyline_rubber_artist,) = self.fits_ax.plot(
                [lx, rx], [ly, ry], color=color, linewidth=max(0.8, 0.7 * lw),
                linestyle="--", alpha=0.7, zorder=5,
            )
        self._update_polyline_length_label()
        try:
            self.fits_canvas.draw_idle()
        except Exception:
            pass

    def _normalize_ellipse_geometry(self, geometry):
        if not isinstance(geometry, dict):
            return None
        center = geometry.get("center", (0.0, 0.0))
        try:
            cx, cy = float(center[0]), float(center[1])
            semi_major = max(0.0, float(geometry.get("semi_major", 0.0)))
            semi_minor = max(0.0, float(geometry.get("semi_minor", 0.0)))
            pa_rad = float(geometry.get("pa_rad", 0.0))
            start_phi_rad = float(geometry.get("start_phi_rad", 0.0))
            end_phi_rad = float(geometry.get("end_phi_rad", start_phi_rad + 2.0 * math.pi))
        except Exception:
            return None
        if not all(np.isfinite(value) for value in (cx, cy, semi_major, semi_minor, pa_rad, start_phi_rad, end_phi_rad)):
            return None
        return {
            "center": (cx, cy),
            "semi_major": semi_major,
            "semi_minor": semi_minor,
            "pa_rad": pa_rad,
            "start_phi_rad": start_phi_rad,
            "end_phi_rad": end_phi_rad,
        }

    def _normalize_phi_rad(self, phi_rad):
        value = math.fmod(float(phi_rad), 2.0 * math.pi)
        if value < 0.0:
            value += 2.0 * math.pi
        if abs(value - 2.0 * math.pi) <= 1e-12:
            value = 0.0
        return value

    def _normalize_phi_deg(self, phi_rad):
        return math.degrees(self._normalize_phi_rad(phi_rad))

    def _canonical_arc_phi_pair(self, start_phi_rad, end_phi_rad, *, full_if_equal=True):
        start = self._normalize_phi_rad(start_phi_rad)
        end_display = self._normalize_phi_rad(end_phi_rad)
        sweep = end_display - start
        if sweep < 0.0 or (full_if_equal and abs(sweep) <= 1e-9):
            sweep += 2.0 * math.pi
        if not full_if_equal and sweep <= math.radians(0.1):
            sweep = math.radians(0.1)
        sweep = min(max(sweep, math.radians(0.1)), 2.0 * math.pi)
        return start, start + sweep

    def _set_ellipse_geometry(self, geometry, *, sync_controls=True, redraw=True):
        normalized = self._normalize_ellipse_geometry(geometry)
        if normalized is None:
            return False
        self.ellipse_geometry = normalized
        if sync_controls:
            self._sync_ellipse_controls_from_state()
        else:
            self._update_ellipse_world_from_pixel()
        if redraw:
            self._draw_ellipse_overlay()
        if self._is_ellipse_mode() and self.last_position_coord is not None:
            self._update_main_window_marker(self.last_position_coord)
        return True

    def _ellipse_path_geometry_from_state(self, state, path_type=None):
        if isinstance(state, dict) and "ellipse_geometry" in state:
            geom = self._normalize_ellipse_geometry(state.get("ellipse_geometry"))
            resolved_type = self._normalize_path_type_value(path_type or state.get("path_type") or self._current_path_type())
        else:
            geom = self._normalize_ellipse_geometry(state)
            resolved_type = self._normalize_path_type_value(path_type or self._current_path_type())
        if geom is None:
            return None
        semi_major = float(geom["semi_major"])
        semi_minor = float(geom["semi_minor"])
        if semi_major <= 0.0 or semi_minor <= 0.0:
            return None
        start_phi = float(geom.get("start_phi_rad", 0.0))
        end_phi = float(geom.get("end_phi_rad", start_phi + 2.0 * math.pi))
        if not self._is_arc_path_type(resolved_type):
            end_phi = start_phi + 2.0 * math.pi
        else:
            start_phi, end_phi = self._canonical_arc_phi_pair(start_phi, end_phi, full_if_equal=True)
        return EllipsePathGeometry(
            center=geom["center"],
            semi_major_px=semi_major,
            semi_minor_px=semi_minor,
            pa_rad=float(geom.get("pa_rad", 0.0)),
            start_phi_rad=start_phi,
            end_phi_rad=end_phi,
        )

    def _current_ellipse_path_geometry(self):
        return self._ellipse_path_geometry_from_state(self.ellipse_geometry, self._current_path_type())

    # --- Polyline path helpers -------------------------------------------
    def _current_polyline_vertices(self):
        verts = []
        for v in (self.polyline_vertices or []):
            try:
                verts.append((float(v[0]), float(v[1])))
            except Exception:
                continue
        return verts

    def _polyline_vertices_from_state(self, state):
        if not isinstance(state, dict):
            return []
        block = state.get("polyline")
        verts = []
        if isinstance(block, dict):
            raw = block.get("vertices")
            if isinstance(raw, (list, tuple)):
                for v in raw:
                    if isinstance(v, (list, tuple)) and len(v) >= 2:
                        try:
                            verts.append((float(v[0]), float(v[1])))
                        except Exception:
                            continue
        return verts

    def _polyline_path_geometry_from_vertices(self, vertices, spline_type=PV_SPLINE_NONE, smoothness=1.0):
        pts = []
        for v in (vertices or []):
            try:
                pts.append((float(v[0]), float(v[1])))
            except Exception:
                continue
        if len(pts) < 2:
            return None
        try:
            return PolylinePathGeometry.from_points(
                pts, spline_type=spline_type, smoothness=smoothness
            )
        except Exception:
            return None

    def _current_polyline_path_geometry(self):
        return self._polyline_path_geometry_from_vertices(
            self._current_polyline_vertices(),
            spline_type=normalize_pv_spline_type(getattr(self, "polyline_spline_type", PV_SPLINE_NONE)),
            smoothness=clamp_pv_smoothness(getattr(self, "polyline_smoothness", 1.0)),
        )

    def _polyline_spline_from_state(self, state):
        """Read (spline_type, smoothness) from a workspace/path state polyline block.

        Accepts the new ``spline: {type, smoothness}`` form and the legacy
        ``smooth: bool`` flag; absent => straight (back-compatible default).
        """
        if isinstance(state, dict):
            block = state.get("polyline")
            if isinstance(block, dict):
                spline = block.get("spline")
                if isinstance(spline, dict):
                    return (
                        normalize_pv_spline_type(spline.get("type")),
                        clamp_pv_smoothness(spline.get("smoothness", 1.0)),
                    )
                if block.get("smooth"):  # legacy boolean alias
                    return (PV_SPLINE_CATMULL_ROM, 1.0)
        return (PV_SPLINE_NONE, 1.0)

    def _polyline_point_normal_from_fraction(self, fraction):
        """Point + slice normal at an arc-length fraction along the active polyline.

        Uses ``effective_vertices`` so the indicator ticks ride the densified
        curve when Smooth is on, and the raw nodes when it is off.
        """
        geom = self._current_polyline_path_geometry()
        if geom is None:
            return None
        pts = np.asarray(geom.effective_vertices, dtype=float)
        if len(pts) < 2:
            return None
        seg = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
        total = float(seg.sum())
        if total <= 1e-12:
            return None
        target = max(0.0, min(1.0, float(fraction))) * total
        cum = np.concatenate(([0.0], np.cumsum(seg)))
        i = int(np.clip(np.searchsorted(cum, target, side="right") - 1, 0, len(seg) - 1))
        seg_len = float(seg[i])
        t = (target - cum[i]) / seg_len if seg_len > 1e-12 else 0.0
        p0, p1 = pts[i], pts[i + 1]
        x, y = p0 + (p1 - p0) * t
        dx, dy = float(p1[0] - p0[0]), float(p1[1] - p0[1])
        norm = math.hypot(dx, dy)
        if norm > 1e-12:
            nx, ny = -dy / norm, dx / norm
        else:
            nx, ny = 0.0, 1.0
        return float(x), float(y), float(nx), float(ny)

    def _current_path_length_px(self):
        if self._is_polyline_mode():
            path = self._current_polyline_path_geometry()
            return 0.0 if path is None else float(path.length_px)
        if self._is_ellipse_mode():
            path = self._current_ellipse_path_geometry()
            return 0.0 if path is None else float(path.length_px)
        if self.line_start is None or self.line_end is None:
            return 0.0
        return float(np.hypot(
            self.line_end[0] - self.line_start[0],
            self.line_end[1] - self.line_start[1],
        ))

    def _current_path_length_in_unit(self, unit=None):
        if unit is None:
            unit = self.length_unit
        return self._convert_length(self._current_path_length_px(), 'pixel', unit)

    def _ellipse_phi_bounds_deg(self):
        path = self._current_ellipse_path_geometry()
        if path is None:
            return 0.0, 360.0
        return math.degrees(path.start_phi_rad), math.degrees(path.end_phi_rad)

    def _ellipse_phi_display_bounds_deg(self):
        phi0, phi1 = self._ellipse_phi_bounds_deg()
        return 0.0, phi1 - phi0

    def _ellipse_point_normal_from_fraction_for_path(self, path, fraction):
        if path is None:
            return None
        frac = max(0.0, min(1.0, float(fraction)))
        phi = path.start_phi_rad + frac * (path.end_phi_rad - path.start_phi_rad)
        cos_pa = math.cos(path.pa_rad)
        sin_pa = math.sin(path.pa_rad)
        cos_phi = math.cos(phi)
        sin_phi = math.sin(phi)
        local_x = path.semi_major_px * cos_phi
        local_y = path.semi_minor_px * sin_phi
        cx, cy = path.center
        x = cx + local_x * cos_pa - local_y * sin_pa
        y = cy + local_x * sin_pa + local_y * cos_pa

        direction = 1.0 if path.sweep_phi_rad >= 0.0 else -1.0
        tangent_local_x = -path.semi_major_px * sin_phi * direction
        tangent_local_y = path.semi_minor_px * cos_phi * direction
        tangent_x = tangent_local_x * cos_pa - tangent_local_y * sin_pa
        tangent_y = tangent_local_x * sin_pa + tangent_local_y * cos_pa
        norm = math.hypot(tangent_x, tangent_y)
        if norm <= 1e-12:
            normal_x, normal_y = 0.0, 1.0
        else:
            normal_x, normal_y = -tangent_y / norm, tangent_x / norm
        return x, y, normal_x, normal_y

    def _ellipse_point_normal_from_fraction(self, fraction):
        return self._ellipse_point_normal_from_fraction_for_path(
            self._current_ellipse_path_geometry(),
            fraction,
        )

    def _ellipse_fraction_for_phi(self, phi, path):
        span = path.end_phi_rad - path.start_phi_rad
        if abs(span) <= 1e-12:
            return 0.0
        if abs(abs(span) - 2.0 * math.pi) <= 1e-6:
            return ((float(phi) - path.start_phi_rad) / span) % 1.0
        best = None
        best_error = None
        for offset in range(-3, 4):
            candidate = float(phi) + 2.0 * math.pi * offset
            fraction = (candidate - path.start_phi_rad) / span
            clamped = max(0.0, min(1.0, fraction))
            error = abs(fraction - clamped)
            if best_error is None or error < best_error:
                best = clamped
                best_error = error
        return 0.0 if best is None else float(best)

    def _ellipse_fraction_from_point_for_path(self, path, x, y):
        """Return the path fraction (0..1) of the ellipse point nearest (x, y)."""
        if path is None:
            return None
        a = float(path.semi_major_px)
        b = float(path.semi_minor_px)
        if a <= 1e-9 or b <= 1e-9:
            return None
        cx, cy = path.center
        cos_pa = math.cos(path.pa_rad)
        sin_pa = math.sin(path.pa_rad)
        dx = float(x) - cx
        dy = float(y) - cy
        x_local = cos_pa * dx + sin_pa * dy
        y_local = -sin_pa * dx + cos_pa * dy
        # Eccentric-anomaly estimate of the nearest parametric angle, matching the
        # phi parametrization used by _ellipse_point_normal_from_fraction.
        phi = math.atan2(y_local / b, x_local / a)
        return self._ellipse_fraction_for_phi(phi, path)

    def _ellipse_fraction_from_point(self, x, y):
        return self._ellipse_fraction_from_point_for_path(
            self._current_ellipse_path_geometry(),
            x,
            y,
        )

    def _ellipse_position_coord_from_fraction(self, fraction):
        """Map a path fraction (0..1) to the current ellipse position-axis value."""
        frac = max(0.0, min(1.0, float(fraction)))
        if self._current_x_axis_mode() == PV_X_AXIS_PHI:
            phi0, phi1 = self._ellipse_phi_display_bounds_deg()
            return phi0 + frac * (phi1 - phi0)
        length = self._current_path_length_in_unit()
        return position_from_fraction(frac, length, POSITION_ORIGIN_START)

    def _ellipse_geometry_for_workspace(self):
        geom = self._normalize_ellipse_geometry(self.ellipse_geometry)
        if geom is None:
            return None
        cx, cy = geom["center"]
        return {
            "center": [float(cx), float(cy)],
            "semi_major": float(geom["semi_major"]),
            "semi_minor": float(geom["semi_minor"]),
            "pa_rad": float(geom["pa_rad"]),
            "start_phi_rad": float(geom["start_phi_rad"]),
            "end_phi_rad": float(geom.get("end_phi_rad", float(geom["start_phi_rad"]) + 2.0 * math.pi)),
        }

    def _polyline_block_for_workspace(self):
        verts = self._current_polyline_vertices()
        if len(verts) < 2:
            return None
        pixel = [[float(x), float(y)] for (x, y) in verts]
        world = []
        for (x, y) in verts:
            try:
                wxy = self._pixel_to_world_xy_for_workspace(x, y)
            except Exception:
                wxy = None
            if wxy is None:
                world = None
                break
            world.append([float(wxy[0]), float(wxy[1])])
        block = {"vertices": pixel}
        spline_type = normalize_pv_spline_type(getattr(self, "polyline_spline_type", PV_SPLINE_NONE))
        if spline_type != PV_SPLINE_NONE:
            # Only emit a spline block for a curve; absence == straight (legacy).
            block["spline"] = {
                "type": spline_type,
                "smoothness": clamp_pv_smoothness(getattr(self, "polyline_smoothness", 1.0)),
            }
        if world is not None:
            block["vertices_world"] = world
        return block

    def _current_ellipse_axis_unit(self):
        combo = getattr(self, "ellipseAxisUnitCombo", None)
        if combo is not None:
            try:
                unit = str(combo.currentText() or "pixel")
                if unit in ("pixel", "deg", "arcmin", "arcsec"):
                    return unit
            except Exception:
                pass
        return str(getattr(self, "ellipse_axis_unit", "pixel") or "pixel")

    def _ellipse_axis_to_display(self, value_pix):
        return self._convert_length(float(value_pix), "pixel", self._current_ellipse_axis_unit())

    def _ellipse_axis_to_pixel(self, value):
        return self._convert_length(float(value), self._current_ellipse_axis_unit(), "pixel")

    def _update_ellipse_world_from_pixel(self):
        """Update ellipse center world coordinate fields from pixel spinboxes."""
        try:
            center_coords = self.coord_converter.pix_to_world(
                self.ellipseCenterXSpin.value(), self.ellipseCenterYSpin.value()
            )
            center_lon, center_lat = center_coords[0], center_coords[1]
        except Exception:
            center_lon, center_lat = "", ""

        self.ellipseCenterLonEdit.blockSignals(True)
        self.ellipseCenterLatEdit.blockSignals(True)
        try:
            self.ellipseCenterLonEdit.setText(center_lon)
            self.ellipseCenterLatEdit.setText(center_lat)
        finally:
            self.ellipseCenterLonEdit.blockSignals(False)
            self.ellipseCenterLatEdit.blockSignals(False)

    def _update_ellipse_pixel_from_world(self):
        """Update ellipse center pixel spinboxes from world coordinate fields."""
        try:
            center_pix = self.coord_converter.world_to_pix(
                self.ellipseCenterLonEdit.text(), self.ellipseCenterLatEdit.text()
            )
            self.ellipseCenterXSpin.blockSignals(True)
            self.ellipseCenterYSpin.blockSignals(True)
            try:
                self.ellipseCenterXSpin.setValue(float(center_pix[0]))
                self.ellipseCenterYSpin.setValue(float(center_pix[1]))
            finally:
                self.ellipseCenterXSpin.blockSignals(False)
                self.ellipseCenterYSpin.blockSignals(False)
            self._update_ellipse_from_controls()
        except ValueError as e:
            print(f"Error parsing ellipse center coordinate string: {e}")
            self._sync_ellipse_controls_from_state()
        except Exception as e:
            print(f"Error updating ellipse center pixels from world coordinates: {e}")
            self._sync_ellipse_controls_from_state()

    def _on_ellipse_axis_unit_changed(self, *_args):
        self.ellipse_axis_unit = self._current_ellipse_axis_unit()
        self._sync_ellipse_controls_from_state()

    def _ellipse_unit_vectors(self, geometry=None):
        geom = self.ellipse_geometry if geometry is None else geometry
        if not geom:
            return (1.0, 0.0), (0.0, 1.0)
        pa = float(geom.get("pa_rad", 0.0))
        major = (math.cos(pa), math.sin(pa))
        minor = (-math.sin(pa), math.cos(pa))
        return major, minor

    def _ellipse_handle_positions(self, geometry=None, *, include_hidden_phase=False):
        geom = self.ellipse_geometry if geometry is None else geometry
        if not geom:
            return {}
        positions = {}
        if not self.ellipse_indicator_positions.get("center", True):
            pass
        else:
            cx, cy = geom["center"]
            positions["center"] = (cx, cy)
        if self._is_arc_mode():
            path = self._ellipse_path_geometry_from_state(geom, "ellipse_arc")
            start = self._ellipse_point_normal_from_fraction_for_path(path, 0.0)
            end = self._ellipse_point_normal_from_fraction_for_path(path, 1.0)
            if start is not None and (
                include_hidden_phase or self.ellipse_indicator_positions.get("start", False)
            ):
                positions["arc_start"] = (start[0], start[1])
            if end is not None and (
                include_hidden_phase or self.ellipse_indicator_positions.get("end", False)
            ):
                positions["arc_end"] = (end[0], end[1])
        return positions

    def _ellipse_to_local(self, x, y, geometry=None):
        geom = self.ellipse_geometry if geometry is None else geometry
        if not geom:
            return None
        cx, cy = geom["center"]
        dx = float(x) - cx
        dy = float(y) - cy
        pa = float(geom.get("pa_rad", 0.0))
        cos_a = math.cos(pa)
        sin_a = math.sin(pa)
        return cos_a * dx + sin_a * dy, -sin_a * dx + cos_a * dy

    def _ellipse_to_global(self, x_local, y_local, geometry=None):
        geom = self.ellipse_geometry if geometry is None else geometry
        if not geom:
            return None
        cx, cy = geom["center"]
        pa = float(geom.get("pa_rad", 0.0))
        cos_a = math.cos(pa)
        sin_a = math.sin(pa)
        return (
            cx + cos_a * float(x_local) - sin_a * float(y_local),
            cy + sin_a * float(x_local) + cos_a * float(y_local),
        )

    def _ellipse_contains(self, x, y, geometry=None):
        geom = self.ellipse_geometry if geometry is None else geometry
        if not geom:
            return False
        semi_major = float(geom.get("semi_major", 0.0))
        semi_minor = float(geom.get("semi_minor", 0.0))
        if semi_major <= 0.0 or semi_minor <= 0.0:
            return False
        local = self._ellipse_to_local(x, y, geom)
        if local is None:
            return False
        x_local, y_local = local
        return (x_local / semi_major) ** 2 + (y_local / semi_minor) ** 2 <= 1.0

    def _ellipse_arc_curve_hit(self, path, x, y, tol):
        if path is None:
            return False
        fraction = self._ellipse_fraction_from_point_for_path(path, x, y)
        point = self._ellipse_point_normal_from_fraction_for_path(path, fraction)
        if point is None:
            return False
        return math.hypot(float(x) - point[0], float(y) - point[1]) <= float(tol)

    def _ellipse_move_hit(self, x, y, *, tol=None):
        if self._is_arc_mode():
            if tol is None:
                tol = self.get_tolerance()
            return self._ellipse_arc_curve_hit(self._current_ellipse_path_geometry(), x, y, tol)
        return self._ellipse_contains(x, y)

    def _ellipse_resize_tolerance_px(self):
        canvas = getattr(self, "fits_canvas", None)
        if canvas is None:
            return 6.0
        try:
            return max(6.0, min(float(canvas.width()), float(canvas.height())) * 0.015)
        except Exception:
            return 6.0

    def _ellipse_display_distance(self, x0, y0, x1, y1):
        try:
            p0 = self.fits_ax.transData.transform((float(x0), float(y0)))
            p1 = self.fits_ax.transData.transform((float(x1), float(y1)))
            return math.hypot(float(p0[0]) - float(p1[0]), float(p0[1]) - float(p1[1]))
        except Exception:
            return math.hypot(float(x0) - float(x1), float(y0) - float(y1))

    def _ellipse_center_hit(self, x, y):
        geom = self.ellipse_geometry
        if not geom:
            return False
        cx, cy = geom["center"]
        return self._ellipse_display_distance(x, y, cx, cy) <= self._ellipse_resize_tolerance_px()

    def _ellipse_outline_disp(self, semi_major, semi_minor, phis, geom):
        """Display-space coords of outline points at parametric *phis* (data
        coords in, screen pixels out), or ``None`` if the transform fails."""
        cx, cy = geom["center"]
        pa = float(geom.get("pa_rad", 0.0))
        cos_a, sin_a = math.cos(pa), math.sin(pa)
        xl = semi_major * np.cos(phis)
        yl = semi_minor * np.sin(phis)
        gx = cx + cos_a * xl - sin_a * yl
        gy = cy + sin_a * xl + cos_a * yl
        try:
            return self.fits_ax.transData.transform(np.column_stack([gx, gy]))
        except Exception:
            return None

    def _nearest_ellipse_outline_phi(self, x, y, semi_major, semi_minor, geom):
        """Parametric angle and screen distance of the outline point nearest to
        the (data-coord) cursor. Two-stage sampling keeps the grab band gap-free
        on large ellipses without an iterative closest-point solve."""
        try:
            cursor = self.fits_ax.transData.transform((float(x), float(y)))
        except Exception:
            return None, float("inf")
        coarse = np.linspace(0.0, 2.0 * math.pi, 121)[:-1]
        disp = self._ellipse_outline_disp(semi_major, semi_minor, coarse, geom)
        if disp is None:
            return None, float("inf")
        d = np.hypot(disp[:, 0] - cursor[0], disp[:, 1] - cursor[1])
        i = int(np.argmin(d))
        step = coarse[1] - coarse[0]
        fine = np.linspace(coarse[i] - step, coarse[i] + step, 41)
        fdisp = self._ellipse_outline_disp(semi_major, semi_minor, fine, geom)
        if fdisp is None:
            return float(coarse[i] % (2.0 * math.pi)), float(d[i])
        fd = np.hypot(fdisp[:, 0] - cursor[0], fdisp[:, 1] - cursor[1])
        j = int(np.argmin(fd))
        return float(fine[j] % (2.0 * math.pi)), float(fd[j])

    def _ellipse_outline_cursor(self, phi, geom):
        """Resize cursor chosen from the outline point's screen-space direction,
        so feedback stays correct when the ellipse is rotated."""
        try:
            cx, cy = geom["center"]
            c_disp = self.fits_ax.transData.transform((cx, cy))
            p = self._ellipse_outline_disp(
                float(geom["semi_major"]), float(geom["semi_minor"]), np.array([phi]), geom
            )[0]
            ang = math.degrees(math.atan2(p[1] - c_disp[1], p[0] - c_disp[0])) % 180.0
        except Exception:
            return Qt.CursorShape.SizeAllCursor
        if ang < 22.5 or ang >= 157.5:
            return Qt.CursorShape.SizeHorCursor
        if ang < 67.5:
            return Qt.CursorShape.SizeBDiagCursor
        if ang < 112.5:
            return Qt.CursorShape.SizeVerCursor
        return Qt.CursorShape.SizeFDiagCursor

    def _ellipse_resize_handle(self, x, y):
        geom = self.ellipse_geometry
        if not geom:
            return None
        semi_major = float(geom["semi_major"])
        semi_minor = float(geom["semi_minor"])
        if semi_major <= 0.0 or semi_minor <= 0.0:
            return None
        # Arc slits keep the original affordance: the curve itself is for moving,
        # and only the major/minor axis tips resize. Whole-outline resize is for
        # closed ellipses, where pinning the opposite outline point is meaningful
        # (on an arc that anchor would sit on the hidden side).
        if self._is_arc_mode():
            return self._ellipse_tip_resize_handle(x, y, semi_major, semi_minor, geom)
        # Keep an interior move zone: only the outer band (near the curve) is a
        # resize target, so clicking well inside moves the ellipse instead of
        # resizing it.
        local = self._ellipse_to_local(x, y, geom)
        if local is not None:
            norm = (local[0] / semi_major) ** 2 + (local[1] / semi_minor) ** 2
            if norm < 0.5:
                return None
        # The whole outline is grabbable for resize, so a rotated ellipse can be
        # resized without locating its major/minor axis tips.
        phi, distance = self._nearest_ellipse_outline_phi(x, y, semi_major, semi_minor, geom)
        if phi is None or distance > self._ellipse_resize_tolerance_px():
            return None
        return {"type": "ellipse", "phi": phi, "cursor": self._ellipse_outline_cursor(phi, geom)}

    def _ellipse_tip_resize_handle(self, x, y, semi_major, semi_minor, geom):
        """Resize handle at the four axis tips only (used for arc slits, so the
        rest of the curve stays a move target). Returns the tip's parametric
        angle so the shared placement-style resize motion still applies."""
        tol = self._ellipse_resize_tolerance_px()
        tips = (
            (0.0, (semi_major, 0.0)),
            (math.pi, (-semi_major, 0.0)),
            (math.pi / 2.0, (0.0, semi_minor)),
            (3.0 * math.pi / 2.0, (0.0, -semi_minor)),
        )
        best_phi, best_dist = None, tol
        for phi, (lx, ly) in tips:
            point = self._ellipse_to_global(lx, ly, geom)
            if point is None:
                continue
            d = self._ellipse_display_distance(x, y, point[0], point[1])
            if d <= best_dist:
                best_phi, best_dist = phi, d
        if best_phi is None:
            return None
        return {"type": "ellipse", "phi": best_phi, "cursor": self._ellipse_outline_cursor(best_phi, geom)}

    def _ellipse_phase_handle(self, x, y):
        if not self._is_arc_mode():
            return None
        positions = self._ellipse_handle_positions(self.ellipse_geometry, include_hidden_phase=True)
        tolerance = self._ellipse_resize_tolerance_px()
        for key in ("arc_start", "arc_end"):
            point = positions.get(key)
            if point is None:
                continue
            if self._ellipse_display_distance(x, y, point[0], point[1]) <= tolerance:
                return key
        return None

    def _sync_ellipse_controls_from_state(self):
        geom = self.ellipse_geometry
        controls = [
            self.ellipseCenterXSpin, self.ellipseCenterYSpin,
            self.ellipseCenterLonEdit, self.ellipseCenterLatEdit,
            self.ellipseMajorSpin, self.ellipseMinorSpin,
            self.ellipseAxisUnitCombo, self.ellipsePASpin, self.ellipseStartPhiSpin,
            self.ellipseEndPhiSpin,
        ]
        for spin in controls:
            spin.blockSignals(True)
        try:
            if geom is None:
                self._update_ellipse_world_from_pixel()
                return
            cx, cy = geom["center"]
            self.ellipseCenterXSpin.setValue(cx)
            self.ellipseCenterYSpin.setValue(cy)
            self._update_ellipse_world_from_pixel()
            self.ellipseMajorSpin.setValue(self._ellipse_axis_to_display(float(geom["semi_major"])))
            self.ellipseMinorSpin.setValue(self._ellipse_axis_to_display(float(geom["semi_minor"])))
            self.ellipsePASpin.setValue(math.degrees(float(geom["pa_rad"])))
            self.ellipseStartPhiSpin.setValue(self._normalize_phi_deg(float(geom["start_phi_rad"])))
            self.ellipseEndPhiSpin.setValue(self._normalize_phi_deg(float(geom.get("end_phi_rad", float(geom["start_phi_rad"]) + 2.0 * math.pi))))
        finally:
            for spin in controls:
                spin.blockSignals(False)

    def _update_ellipse_from_controls(self):
        if getattr(self, "_updating_geometry_controls", False):
            return
        if not self._is_ellipse_mode():
            return
        start_phi = math.radians(self.ellipseStartPhiSpin.value())
        end_phi = math.radians(self.ellipseEndPhiSpin.value())
        if self._is_arc_mode():
            start_phi, end_phi = self._canonical_arc_phi_pair(start_phi, end_phi, full_if_equal=True)
        geometry = {
            "center": (self.ellipseCenterXSpin.value(), self.ellipseCenterYSpin.value()),
            "semi_major": self._ellipse_axis_to_pixel(self.ellipseMajorSpin.value()),
            "semi_minor": self._ellipse_axis_to_pixel(self.ellipseMinorSpin.value()),
            "pa_rad": math.radians(self.ellipsePASpin.value()),
            "start_phi_rad": start_phi,
            "end_phi_rad": end_phi,
        }
        if self._set_ellipse_geometry(geometry, sync_controls=False, redraw=True):
            self._request_main_overlay_redraw()
            # Real-time PV redraw on parameter edits when Auto Update is enabled.
            self.update_pv_diagram()

    def _draw_ellipse_overlay(self):
        geom = self.ellipse_geometry
        if geom is None:
            return
        cx, cy = geom["center"]
        semi_major = float(geom["semi_major"])
        semi_minor = float(geom["semi_minor"])
        if semi_major <= 0.0 or semi_minor <= 0.0:
            self._remove_artist(self.ellipse_artist)
            self.ellipse_artist = None
            for artist in list((self.ellipse_handle_artists or {}).values()):
                self._remove_artist(artist)
            self.ellipse_handle_artists = {}
            self._clear_ellipse_indicator_ticks()
            return
        if self._is_arc_mode():
            path = self._current_ellipse_path_geometry()
            xs, ys = self._ellipse_path_xy(path)
            if xs is None or ys is None:
                return
            if not isinstance(self.ellipse_artist, mpl.lines.Line2D):
                self._remove_artist(self.ellipse_artist)
                self.ellipse_artist = None
            if self.ellipse_artist is None:
                self.ellipse_artist, = self.fits_ax.plot(
                    xs,
                    ys,
                    color=self.pvarrow_color,
                    linewidth=1.5 * self.arrow_size,
                    zorder=5,
                )
            else:
                self.ellipse_artist.set_data(xs, ys)
                self.ellipse_artist.set_color(self.pvarrow_color)
                self.ellipse_artist.set_linewidth(1.5 * self.arrow_size)
        else:
            if not isinstance(self.ellipse_artist, mpl.patches.Ellipse):
                self._remove_artist(self.ellipse_artist)
                self.ellipse_artist = None
            if self.ellipse_artist is None:
                self.ellipse_artist = mpl.patches.Ellipse(
                    (cx, cy),
                    width=2.0 * semi_major,
                    height=2.0 * semi_minor,
                    angle=math.degrees(float(geom["pa_rad"])),
                    fill=False,
                    edgecolor=self.pvarrow_color,
                    linewidth=1.5 * self.arrow_size,
                    zorder=5,
                )
                self.fits_ax.add_patch(self.ellipse_artist)
            else:
                self.ellipse_artist.center = (cx, cy)
                self.ellipse_artist.width = 2.0 * semi_major
                self.ellipse_artist.height = 2.0 * semi_minor
                self.ellipse_artist.angle = math.degrees(float(geom["pa_rad"]))
                self.ellipse_artist.set_edgecolor(self.pvarrow_color)
                self.ellipse_artist.set_linewidth(1.5 * self.arrow_size)

        valid_handle_keys = set(self._ellipse_handle_positions(geom))
        for key in list(self.ellipse_handle_artists):
            if key not in valid_handle_keys:
                self._remove_artist(self.ellipse_handle_artists.pop(key, None))

        marker_styles = {"center": "+", "arc_start": "o", "arc_end": "s"}
        for key, (x, y) in self._ellipse_handle_positions(geom).items():
            artist = self.ellipse_handle_artists.get(key)
            if artist is None:
                artist, = self.fits_ax.plot(
                    [x], [y],
                    marker=marker_styles.get(key, "o"),
                    markersize=7,
                    color=self.pvarrow_color,
                    markerfacecolor="none",
                    linestyle="None",
                    zorder=6,
                )
                self.ellipse_handle_artists[key] = artist
            else:
                artist.set_data([x], [y])
                artist.set_color(self.pvarrow_color)
                artist.set_marker(marker_styles.get(key, "o"))
                artist.set_markerfacecolor("none")
            # Ellipse handles are always shown (the Node Markers toggle is
            # polyline-only); indicators control the ellipse annotation ticks.
            artist.set_visible(True)

        self._draw_ellipse_indicator_ticks()

    def _clear_ellipse_indicator_ticks(self):
        for artist in list(self.ellipse_indicator_artists or []):
            self._remove_artist(artist)
        self.ellipse_indicator_artists = []

    def _draw_ellipse_indicator_ticks(self):
        """Draw short perpendicular ticks at the Phi0 (start) / Phi1 (end) points."""
        self._clear_ellipse_indicator_ticks()
        path = self._current_ellipse_path_geometry()
        if path is None:
            return
        half = self.sliceWidthSpin.value() / 2.0
        if half <= 0.0:
            return
        tick_fractions = []
        if self.ellipse_indicator_positions.get("start", False):
            tick_fractions.append(0.0)
        if self.ellipse_indicator_positions.get("end", False):
            tick_fractions.append(1.0)
        for fraction in tick_fractions:
            point = self._ellipse_point_normal_from_fraction(fraction)
            if point is None:
                continue
            base_x, base_y, normal_x, normal_y = point
            tick = mpl.lines.Line2D(
                [base_x - half * normal_x, base_x + half * normal_x],
                [base_y - half * normal_y, base_y + half * normal_y],
                color=self.pvarrow_color,
                lw=1.5 * self.arrow_size,
                zorder=6,
            )
            self.fits_ax.add_line(tick)
            self.ellipse_indicator_artists.append(tick)

    def clear_arrow(self):
        """Clear the active PV path from the canvas and reset active state."""
        if self.active_pv_path_id is not None:
            self.pv_path_items = [
                item for item in self.pv_path_items
                if item.get("id") != self.active_pv_path_id
            ]
            self._remove_inactive_artists_for_path(self.active_pv_path_id)
            self.active_pv_path_id = None
        self._clear_straight_path()
        self._clear_ellipse_path()
        self._clear_polyline_path()
        self.last_position_coord = None
        if self.pos_indicator_on_arrow is not None:
            self.pos_indicator_on_arrow.set_visible(False)
        self._sync_path_list_combo()
        self._redraw_inactive_path_artists()
        self._invalidate_main_overlay_background()
        self.fits_canvas.draw_idle()
        self._record_slit_change()

    def get_tolerance(self):
        """Return tolerance based on image diagonal"""
        height, width = self.data.shape[-2], self.data.shape[-1]
        diag = np.hypot(width, height)
        return diag * 0.01 if diag < 500 else 10

    def _set_canvas_cursor(self, cursor):
        if getattr(self, 'fits_canvas', None) is None:
            return
        if cursor is None:
            cursor = Qt.CursorShape.ArrowCursor
        self.fits_canvas.setCursor(cursor)

    def _update_hover_cursor(self, x, y):
        if getattr(self, 'fits_canvas', None) is None:
            return
        if self.fits_viewer.toolbar.mode != '' or self.dragging_pos_indicator:
            return
        if not self.line_fixed or self.line_start is None or self.line_end is None:
            self._set_canvas_cursor(Qt.CursorShape.ArrowCursor)
            return

        if x is None or y is None:
            self._set_canvas_cursor(Qt.CursorShape.ArrowCursor)
            return

        tol = self.get_tolerance()
        d0 = math.hypot(x - self.line_start[0], y - self.line_start[1])
        d1 = math.hypot(x - self.line_end[0], y - self.line_end[1])

        if min(d0, d1) < tol:
            self._set_canvas_cursor(Qt.CursorShape.CrossCursor)
            return

        dist = self.point_line_distance((x, y), self.line_start, self.line_end)
        if dist < tol:
            self._set_canvas_cursor(Qt.CursorShape.OpenHandCursor)
        else:
            self._set_canvas_cursor(Qt.CursorShape.ArrowCursor)

    def update_controls(self):
        """Update GUI controls with current line endpoints and rotation angle."""
        if self.line_start is not None and self.line_end is not None:
            line_length_px = 0.0
            geometry_spins = [
                self.startXSpin, self.startYSpin, self.endXSpin, self.endYSpin,
                self.centerXSpin, self.centerYSpin,
            ]
            self._updating_geometry_controls = True
            try:
                # Block signals to prevent loops while updating programmatically
                for spin in geometry_spins:
                    spin.blockSignals(True)

                self.startXSpin.setValue(self.line_start[0])
                self.startYSpin.setValue(self.line_start[1])
                self.endXSpin.setValue(self.line_end[0])
                self.endYSpin.setValue(self.line_end[1])
                center = self._line_center()
                if center is not None:
                    self.centerXSpin.setValue(center[0])
                    self.centerYSpin.setValue(center[1])

                # Unblock signals after setting values
                for spin in geometry_spins:
                    spin.blockSignals(False)

                # Update WCS fields from the new pixel values
                self._update_world_from_pixel()
                self._update_center_world_from_pixel()

                angle_rad = np.arctan2(self.line_end[1] - self.line_start[1],
                                    self.line_end[0] - self.line_start[0])
                self.rotationAngleSpin.blockSignals(True)
                self.rotationAngleSpin.setValue(np.degrees(angle_rad))
                self.rotationAngleSpin.blockSignals(False)

                line_length_px = np.hypot(self.line_end[0] - self.line_start[0],
                                        self.line_end[1] - self.line_start[1])

                # Convert pixel length to the currently selected unit for display
                current_unit = self.lengthUnitCombo.currentText()
                display_length = self._convert_length(line_length_px, 'pixel', current_unit)

                self.arrowLengthSpin.blockSignals(True)
                # Set singleStep according to the current unit
                if current_unit == 'pixel':
                    display_length = round(display_length)
                    self.arrowLengthSpin.setSingleStep(1.0)
                else:
                    self.arrowLengthSpin.setSingleStep(self._get_dynamic_step(display_length))

                self.arrowLengthSpin.setValue(display_length)
                self.arrowLengthSpin.blockSignals(False)
            finally:
                for spin in geometry_spins:
                    spin.blockSignals(False)
                self.rotationAngleSpin.blockSignals(False)
                self.arrowLengthSpin.blockSignals(False)
                self._updating_geometry_controls = False

            if line_length_px < 1:
                if self.marker_artist_end is None:
                    if self.line_end==(0, 0): return
                    self.marker_artist_end, = self.fits_ax.plot(
                        [self.line_end[0]], [self.line_end[1]],
                        marker='o', markersize=5, color=self.pvmarker_color, linestyle='None')
                else:
                    self.marker_artist_end.set_data([self.line_end[0]], [self.line_end[1]])
        self.slice_width = self.sliceWidthSpin.value()
        self.fits_ax.get_xlim()

    def create_arrow_patch(self):
        """Create a FancyArrowPatch with fixed thickness and add a width indicator."""
        # Create arrow with fixed parameters.
        arrow = mpl.patches.FancyArrowPatch(
            self.line_start, self.line_end,
            arrowstyle="Simple,tail_width=1.5,head_width=7,head_length=7", shrinkA = 0, shrinkB = 0,
            color=self.pvarrow_color, lw = 0, mutation_scale=self.arrow_size, animated=self.is_interactive_mode)
        # Compute arrow direction vector:
        dx = self.line_end[0] - self.line_start[0]
        dy = self.line_end[1] - self.line_start[1]
        mag = np.hypot(dx, dy)
        if mag == 0:
            perp = (0, 0)
        else:
            # Perpendicular unit vector (rotate 90°):
            perp = (-dy/mag, dx/mag)
        # Use slice width (in pixel units) as the total length of the indicator.
        half = self.sliceWidthSpin.value() / 2.0

        if self.width_indicators:
            for indicator in self.width_indicators:
                try:
                    indicator.remove()
                except Exception:
                    pass
            self.width_indicators = []

        for pos, flag in self.indicator_positions.items():
            if flag:
                if pos == "start":
                    base = self.line_start
                elif pos == "center":
                    cx = (self.line_start[0] + self.line_end[0]) / 2.0
                    cy = (self.line_start[1] + self.line_end[1]) / 2.0
                    base = (cx, cy)
                elif pos == "end":
                    base = self.line_end
                else:
                    base = self.line_start
                x0, y0 = base
                ind_start = (x0 - half * perp[0], y0 - half * perp[1])
                ind_end   = (x0 + half * perp[0], y0 + half * perp[1])
                indicator = mpl.lines.Line2D(
                    [ind_start[0], ind_end[0]],
                    [ind_start[1], ind_end[1]],
                    color=self.pvarrow_color, lw=1.5*self.arrow_size, animated=self.is_interactive_mode)
                self.fits_ax.add_line(indicator)
                self.width_indicators.append(indicator)
        return arrow

    def update_arrow_from_gui(self):
        if getattr(self, "_updating_geometry_controls", False):
            return
        if self._is_polyline_mode():
            # Shared controls (slice width) feed here in polyline mode; sync the
            # width and recompute PV without touching straight/ellipse geometry.
            self.slice_width = self.sliceWidthSpin.value()
            self._draw_polyline_overlay()
            self._update_main_window_marker(self.last_position_coord)
            self._invalidate_main_overlay_background()
            self._request_main_overlay_redraw()
            self.update_pv_diagram(force_update=True)
            return
        if self._is_ellipse_mode():
            # Shared controls (e.g. slice width) feed here in ellipse mode; refresh
            # the overlay (ticks/indicator scale with slice width) and PV without
            # touching the straight geometry.
            self.slice_width = self.sliceWidthSpin.value()
            self._draw_ellipse_overlay()
            self._update_main_window_marker(self.last_position_coord)
            self._invalidate_main_overlay_background()
            self._request_main_overlay_redraw()
            self.update_pv_diagram()
            return
        if not self.is_interactive_mode:
            self.start_interactive_update()

        new_start = (self.startXSpin.value(), self.startYSpin.value())
        new_end = (self.endXSpin.value(), self.endYSpin.value())

        self.line_start, self.line_end = new_start, new_end

        if self.arrow_artist is not None:
            self.arrow_artist.remove()

        self.arrow_artist = self.create_arrow_patch()
        self.fits_ax.add_patch(self.arrow_artist)

        self.update_controls()
        self._update_main_window_marker(self.last_position_coord)

        self.do_interactive_update()

    def update_arrow_from_center(self):
        if getattr(self, "_updating_geometry_controls", False):
            return
        if not self.is_interactive_mode:
            self.start_interactive_update()

        center = (self.centerXSpin.value(), self.centerYSpin.value())
        length_px = self._current_line_length_px()
        angle_rad = self._current_line_angle_rad()
        self.line_start, self.line_end = straight_line_from_center(center, length_px, angle_rad)

        if self.arrow_artist is not None:
            self.arrow_artist.remove()
        self.arrow_artist = self.create_arrow_patch()
        self.fits_ax.add_patch(self.arrow_artist)

        self.update_controls()
        self._update_main_window_marker(self.last_position_coord)
        self.do_interactive_update()
        self.update_pv_diagram()

    def use_cursor_as_center(self):
        center = self._current_main_cursor_xy()
        if center is None:
            QMessageBox.information(self, "Center", "Current cursor position is not available.")
            return
        self._set_center_controls_from_pixel(center[0], center[1])
        self.update_arrow_from_center()

    def update_arrow_from_rotation(self):
        if getattr(self, "_updating_geometry_controls", False):
            return
        if self.line_start is None or self.line_end is None: return
        if not self.is_interactive_mode: self.start_interactive_update()

        length = np.hypot(self.line_end[0] - self.line_start[0], self.line_end[1] - self.line_start[1])
        new_angle = np.radians(self.rotationAngleSpin.value())
        self.line_start, self.line_end = anchored_straight_line(
            self.line_start,
            self.line_end,
            length,
            new_angle,
            self._current_position_origin(),
        )

        if self.arrow_artist is not None: self.arrow_artist.remove()
        self.arrow_artist = self.create_arrow_patch()
        self.fits_ax.add_patch(self.arrow_artist)

        self.update_controls()
        self._update_main_window_marker(self.last_position_coord)
        self.do_interactive_update()
        self.update_pv_diagram()

    def update_arrow_from_length(self):
        if getattr(self, "_updating_geometry_controls", False):
            return
        if self.line_start is None or self.line_end is None: return
        if not self.is_interactive_mode: self.start_interactive_update()

        angle_rad = np.radians(self.rotationAngleSpin.value())
        current_unit = self.lengthUnitCombo.currentText()
        gui_length = self.arrowLengthSpin.value()
        length_px = self._convert_length(gui_length, current_unit, 'pixel')

        self.arrow_length = length_px
        self.line_start, self.line_end = anchored_straight_line(
            self.line_start,
            self.line_end,
            length_px,
            angle_rad,
            self._current_position_origin(),
        )

        if self.arrow_artist is not None:
            self.arrow_artist.remove()
        self.arrow_artist = self.create_arrow_patch()
        self.fits_ax.add_patch(self.arrow_artist)

        self.update_controls()
        self._update_main_window_marker(self.last_position_coord)
        self.do_interactive_update()
        self.update_pv_diagram()

    def reverse_direction(self):
        """Reverse the straight slit direction while preserving the selected physical position."""
        if self.line_start is None or self.line_end is None:
            return

        fraction = self._get_cursor_fractional_position()
        self.line_start, self.line_end = self.line_end, self.line_start
        if fraction is not None:
            self.last_position_coord = self._position_coord_from_fraction(1.0 - fraction)

        if self.arrow_artist is not None:
            self.arrow_artist.remove()
        self.arrow_artist = self.create_arrow_patch()
        self.fits_ax.add_patch(self.arrow_artist)
        self.line_fixed = True
        self.is_range_manual = False
        self.update_controls()
        self.update_pv_diagram(force_update=True)
        if self.last_position_coord is not None:
            self.update_pv_position_cursor(self.last_position_coord)
        self._update_main_window_marker(self.last_position_coord)
        if hasattr(self.fits_viewer, 'redraw_main_overlay_and_blit'):
            self.fits_viewer.redraw_main_overlay_and_blit()
        else:
            self.fits_canvas.draw_idle()
        self._record_slit_change()

    def _apply_polyline(self):
        """Commit/recompute the active polyline path (mirrors the ellipse apply branch)."""
        if len(self._current_polyline_vertices()) < 2:
            return
        self.polyline_finished = True
        self.polyline_extend_from_start = False
        # NB: keep polyline_selected_index so a node stays selected after a
        # move/insert (the delegate clears it on delete or new-path).
        self._draw_polyline_overlay()
        self.update_pv_diagram(force_update=True)
        self._update_main_window_marker(self.last_position_coord)
        self._sync_active_path_item_from_current_state()
        self._redraw_inactive_path_artists()
        self._request_main_overlay_redraw()
        self._record_slit_change()

    def _current_polyline_spline_type(self):
        combo = getattr(self, "polylineCurveCombo", None)
        if combo is not None:
            try:
                return normalize_pv_spline_type(combo.currentData())
            except Exception:
                pass
        return normalize_pv_spline_type(getattr(self, "polyline_spline_type", PV_SPLINE_NONE))

    def _sync_polyline_curve_controls(self):
        """Reflect polyline_spline_type/polyline_smoothness into the widgets."""
        combo = getattr(self, "polylineCurveCombo", None)
        if combo is not None:
            try:
                idx = combo.findData(normalize_pv_spline_type(self.polyline_spline_type))
                if idx >= 0:
                    combo.blockSignals(True)
                    combo.setCurrentIndex(idx)
                    combo.blockSignals(False)
            except Exception:
                try:
                    combo.blockSignals(False)
                except Exception:
                    pass
        slider = getattr(self, "polylineSmoothnessSlider", None)
        if slider is not None:
            try:
                slider.blockSignals(True)
                slider.setValue(int(round(clamp_pv_smoothness(self.polyline_smoothness) * 100)))
            finally:
                slider.blockSignals(False)
        self._sync_polyline_smoothness_enabled()

    def _sync_polyline_smoothness_enabled(self):
        """Smoothness only matters for a curve type; grey it out for Straight."""
        is_curve = normalize_pv_spline_type(self.polyline_spline_type) != PV_SPLINE_NONE
        for widget in (
            getattr(self, "polylineSmoothnessLabel", None),
            getattr(self, "polylineSmoothnessWidget", None),
        ):
            try:
                if widget is not None:
                    widget.setEnabled(is_curve)
            except Exception:
                pass
        label = getattr(self, "polylineSmoothnessValueLabel", None)
        if label is not None:
            try:
                label.setText(f"{int(round(clamp_pv_smoothness(self.polyline_smoothness) * 100))}%")
            except Exception:
                pass

    def _update_polyline_length_label(self):
        label = getattr(self, "polylinePathLengthValueLabel", None)
        if label is None:
            return
        try:
            if len(self._current_polyline_vertices()) < 2:
                label.setText("-")
                return
            unit = str(self.lengthUnitCombo.currentText() or "pixel")
            length = self._current_path_length_in_unit(unit=unit)
            decimals = 0 if unit == "pixel" else 3
            label.setText(f"{length:.{decimals}f} {unit}")
        except Exception:
            label.setText("-")

    def _apply_polyline_curve_change(self, *, record_undo=True):
        """Shared redraw/recompute after a curve-type or smoothness edit."""
        if not self._is_polyline_mode():
            return
        self._draw_polyline_overlay()
        self._update_polyline_length_label()
        if len(self._current_polyline_vertices()) >= 2:
            self.update_pv_diagram(force_update=True)
            self._update_main_window_marker(self.last_position_coord)
            self._sync_active_path_item_from_current_state()
            self._redraw_inactive_path_artists()
        self._request_main_overlay_redraw()
        if record_undo:
            self._record_slit_change()

    def _on_polyline_curve_type_changed(self):
        new_type = self._current_polyline_spline_type()
        if new_type == normalize_pv_spline_type(self.polyline_spline_type):
            return
        self.polyline_spline_type = new_type
        self._sync_polyline_smoothness_enabled()
        # Curve-type change is a discrete edit -> recompute + one undo step.
        self._apply_polyline_curve_change(record_undo=True)

    def _on_polyline_smoothness_changed(self, value):
        """Live preview while dragging; undo is recorded on release/commit."""
        self.polyline_smoothness = clamp_pv_smoothness(int(value) / 100.0)
        label = getattr(self, "polylineSmoothnessValueLabel", None)
        if label is not None:
            try:
                label.setText(f"{int(value)}%")
            except Exception:
                pass
        if not self._is_polyline_mode():
            return
        # Redraw the curve live; only recompute PV live when Auto Update is on
        # (a full PV recompute per slider tick is otherwise too heavy).
        self._draw_polyline_overlay()
        self._update_polyline_length_label()
        if self.autoUpdateCheck.isChecked() and len(self._current_polyline_vertices()) >= 2:
            self.update_pv_diagram(force_update=True)
            self._update_main_window_marker(self.last_position_coord)
        self._request_main_overlay_redraw()

    def _on_polyline_smoothness_committed(self):
        """Slider released: force the PV recompute and record one undo step."""
        self._apply_polyline_curve_change(record_undo=True)

    def apply_controls(self, *, preserve_straight_geometry=False):
        if self._is_polyline_mode():
            self._apply_polyline()
            return
        if self._is_ellipse_mode():
            self._update_ellipse_from_controls()
            self._draw_ellipse_overlay()
            self.update_pv_diagram(force_update=True)
            self._update_main_window_marker(self.last_position_coord)
            self._sync_active_path_item_from_current_state()
            self._redraw_inactive_path_artists()
            self._request_main_overlay_redraw()
            self._record_slit_change()
            return
        preserve_straight_geometry = (
            preserve_straight_geometry
            and self.line_start is not None
            and self.line_end is not None
        )
        if preserve_straight_geometry:
            # Mouse drags already produced the authoritative endpoints. In pixel
            # display mode arrowLengthSpin is intentionally rounded for the UI;
            # rebuilding from that rounded value would move a boundary-clipped
            # endpoint back outside the image during release finalization.
            pass
        elif self._current_geometry_input_mode() == "center":
            center = (self.centerXSpin.value(), self.centerYSpin.value())
            length_px = self._convert_length(
                self.arrowLengthSpin.value(),
                self.lengthUnitCombo.currentText(),
                'pixel',
            )
            angle_rad = np.radians(self.rotationAngleSpin.value())
            self.line_start, self.line_end = straight_line_from_center(center, length_px, angle_rad)
        else:
            precise_line = self._pending_precise_world_line
            if precise_line is not None:
                gui_start, gui_end = precise_line
                self._pending_precise_world_line = None
            else:
                gui_start = (self.startXSpin.value(), self.startYSpin.value())
                gui_end = (self.endXSpin.value(), self.endYSpin.value())
            tol_coord = 1e-2
            if (self.line_start is None or self.line_end is None or
                np.hypot(gui_start[0] - self.line_start[0], gui_start[1] - self.line_start[1]) > tol_coord or
                np.hypot(gui_end[0] - self.line_end[0], gui_end[1] - self.line_end[1]) > tol_coord):
                self.line_start, self.line_end = gui_start, gui_end
            else:
                length_px = self._convert_length(
                    self.arrowLengthSpin.value(),
                    self.lengthUnitCombo.currentText(),
                    'pixel',
                )
                angle_rad = np.radians(self.rotationAngleSpin.value())
                self.line_start, self.line_end = anchored_straight_line(
                    self.line_start,
                    self.line_end,
                    length_px,
                    angle_rad,
                    self._current_position_origin(),
                )
        self.line_fixed = True
        if self.arrow_artist is not None:
            self.arrow_artist.remove()
        self.arrow_artist = self.create_arrow_patch()
        self.fits_ax.add_patch(self.arrow_artist)
        self.update_controls()
        self.update_pv_diagram(force_update=True)
        self._update_main_window_marker(self.last_position_coord)
        self._sync_active_path_item_from_current_state()
        self._redraw_inactive_path_artists()
        self._request_main_overlay_redraw()
        self._record_slit_change()


    def point_line_distance(self, point, start, end):
        """Return the shortest distance from a point to a line segment"""
        x, y = point
        x0, y0 = start
        x1, y1 = end
        dx = x1 - x0
        dy = y1 - y0
        if dx == 0 and dy == 0:
            return np.hypot(x - x0, y - y0)
        t = ((x - x0) * dx + (y - y0) * dy) / (dx * dx + dy * dy)
        t = max(0.0, min(1.0, t))
        nearest_x = x0 + t * dx
        nearest_y = y0 + t * dy
        return np.hypot(x - nearest_x, y - nearest_y)

    def update_endpoint_markers(self, m_x, m_y):
        """Update blue markers for endpoints based on mouse position"""
        tol = self.get_tolerance()
        if m_x is None or m_y is None: return
        d0 = np.hypot(m_x - self.line_start[0], m_y - self.line_start[1])
        if d0 < tol:
            if self.marker_artist_start is None:
                self.marker_artist_start, = self.fits_ax.plot(
                    [self.line_start[0]], [self.line_start[1]],
                    marker='o', markersize=5, color=self.pvmarker_color, linestyle='None', zorder = 5)
            else:
                self.marker_artist_start.set_data([self.line_start[0]], [self.line_start[1]])

        else:
            if self.marker_artist_start is not None:
                self.marker_artist_start.remove()
                self.marker_artist_start = None


        d1 = np.hypot(m_x - self.line_end[0], m_y - self.line_end[1])
        if d1 < tol:
            if self.marker_artist_end is None:
                self.marker_artist_end, = self.fits_ax.plot(
                    [self.line_end[0]], [self.line_end[1]],
                    marker='o', markersize=5, color=self.pvmarker_color, linestyle='None', zorder = 5)
            else:
                self.marker_artist_end.set_data([self.line_end[0]], [self.line_end[1]])
        else:
            if self.marker_artist_end is not None:
                self.marker_artist_end.remove()
                self.marker_artist_end = None


    def start_rotate_mode(self, x, y):
        """Initialize rotate mode using mouse press coordinates"""
        self.edit_mode = "rotate"
        self.drag_start = (x, y)
        d_start = np.hypot(x - self.line_start[0], y - self.line_start[1])
        d_end = np.hypot(x - self.line_end[0], y - self.line_end[1])
        self.rotating_endpoint = 0 if d_start < d_end else 1
        cx = (self.line_start[0] + self.line_end[0]) / 2.0
        cy = (self.line_start[1] + self.line_end[1]) / 2.0
        self.center = (cx, cy)
        self.initial_angle = np.arctan2(self.line_end[1] - cy, self.line_end[0] - cx)
        self.initial_line_length = np.hypot(self.line_end[0] - self.line_start[0],
                                            self.line_end[1] - self.line_start[1])
        self._set_canvas_cursor(Qt.CursorShape.CrossCursor)

    def _on_ellipse_press(self, x, y):
        if self.ellipse_geometry is not None:
            phase_handle = self._ellipse_phase_handle(x, y)
            if phase_handle is not None:
                self.edit_mode = f"ellipse_phi_{'start' if phase_handle == 'arc_start' else 'end'}"
                self.drag_start = (x, y)
                self.initial_ellipse_geometry = dict(self.ellipse_geometry or {})
                self._invalidate_main_overlay_background()
                self._set_canvas_cursor(Qt.CursorShape.CrossCursor)
                return True

            if self.shift_pressed and self._ellipse_contains(x, y):
                cx, cy = self.ellipse_geometry["center"]
                dx = float(x) - cx
                dy = float(y) - cy
                if math.hypot(dx, dy) > 1e-6:
                    self.edit_mode = "ellipse_rotate"
                    self.drag_start = (x, y)
                    self.initial_ellipse_geometry = dict(self.ellipse_geometry or {})
                    self.ellipse_rotation_reference_angle = math.atan2(dy, dx)
                    self._invalidate_main_overlay_background()
                    self._set_canvas_cursor(Qt.CursorShape.CrossCursor)
                    return True

            # The centre "+" is the move grip: check it before the outline so a
            # thin/elongated slit (whose whole body sits within the resize band)
            # can still be moved rather than only resized.
            if self._ellipse_center_hit(x, y):
                self.edit_mode = "ellipse_move"
                self.drag_start = (x, y)
                self.initial_ellipse_geometry = dict(self.ellipse_geometry or {})
                self._invalidate_main_overlay_background()
                self._set_canvas_cursor(Qt.CursorShape.ClosedHandCursor)
                return True

            resize_handle = self._ellipse_resize_handle(x, y)
            if resize_handle is not None:
                self.edit_mode = "ellipse_resize"
                self.drag_start = (x, y)
                self.initial_ellipse_geometry = dict(self.ellipse_geometry or {})
                self.ellipse_resize_handle = resize_handle
                self._invalidate_main_overlay_background()
                self._set_canvas_cursor(resize_handle.get("cursor", Qt.CursorShape.SizeAllCursor))
                return True

            if self._ellipse_move_hit(x, y):
                self.edit_mode = "ellipse_move"
                self.drag_start = (x, y)
                self.initial_ellipse_geometry = dict(self.ellipse_geometry or {})
                self._invalidate_main_overlay_background()
                self._set_canvas_cursor(Qt.CursorShape.ClosedHandCursor)
                return True

        if self.ellipse_geometry is None:
            self.edit_mode = None
            self.drag_start = (x, y)
            self.drag_mode = "ellipse_draw"
            self.ellipse_drag_anchor = (x, y)
            self._set_ellipse_geometry(
                {
                    "center": (x, y),
                    "semi_major": 0.0,
                    "semi_minor": 0.0,
                    "pa_rad": 0.0,
                    "start_phi_rad": 0.0,
                    "end_phi_rad": math.pi,
                },
                sync_controls=True,
                redraw=False,
            )
            self._invalidate_main_overlay_background()
            self._set_canvas_cursor(Qt.CursorShape.CrossCursor)
            return True

        self.edit_mode = None
        self._update_ellipse_hover_cursor(x, y)
        return True

    def _on_ellipse_motion(self, x, y):
        if self.drag_mode == "ellipse_draw":
            anchor = self.ellipse_drag_anchor
            if anchor is None:
                return False
            ax, ay = anchor
            semi_major = abs(float(x) - ax) / 2.0
            semi_minor = abs(float(y) - ay) / 2.0
            geometry = {
                "center": ((float(x) + ax) / 2.0, (float(y) + ay) / 2.0),
                "semi_major": semi_major,
                "semi_minor": semi_minor,
                "pa_rad": 0.0,
                "start_phi_rad": 0.0,
                "end_phi_rad": math.pi,
            }
            self._set_ellipse_geometry(geometry, sync_controls=True, redraw=True)
            self._set_canvas_cursor(Qt.CursorShape.CrossCursor)
            self._request_main_overlay_redraw()
            # Real-time PV redraw while drawing when Auto Update is enabled.
            self.update_pv_diagram()
            return True

        if not self.edit_mode or not str(self.edit_mode).startswith("ellipse_"):
            self._update_ellipse_hover_cursor(x, y)
            return False

        initial = self.initial_ellipse_geometry
        if not initial:
            return False
        geometry = dict(initial)
        cx, cy = geometry["center"]
        mode = str(self.edit_mode).replace("ellipse_", "", 1)
        if mode == "move":
            dx = float(x) - self.drag_start[0]
            dy = float(y) - self.drag_start[1]
            geometry["center"] = (cx + dx, cy + dy)
            self._set_canvas_cursor(Qt.CursorShape.ClosedHandCursor)
        elif mode == "rotate":
            dx = float(x) - cx
            dy = float(y) - cy
            if math.hypot(dx, dy) > 1e-6 and self.ellipse_rotation_reference_angle is not None:
                current_angle = math.atan2(dy, dx)
                delta = math.atan2(
                    math.sin(current_angle - self.ellipse_rotation_reference_angle),
                    math.cos(current_angle - self.ellipse_rotation_reference_angle),
                )
                geometry["pa_rad"] = float(initial.get("pa_rad", 0.0)) + delta
            self._set_canvas_cursor(Qt.CursorShape.CrossCursor)
        elif mode == "resize":
            resize_handle = self.ellipse_resize_handle or {}
            phi = resize_handle.get("phi")
            if phi is None:
                return False
            # Outline drag (placement-style): pin the outline point opposite the
            # grabbed one and let the grabbed point (at parametric angle ``phi``)
            # follow the cursor, so the centre slides to the midpoint exactly
            # like drawing a fresh ellipse. Near an axis tip the perpendicular
            # semi-axis is frozen, recovering a single-axis drag.
            pa = float(initial.get("pa_rad", 0.0))
            cos_a = math.cos(pa)
            sin_a = math.sin(pa)
            a0 = float(initial.get("semi_major", 0.0))
            b0 = float(initial.get("semi_minor", 0.0))
            cos_p, sin_p = math.cos(phi), math.sin(phi)
            min_size = 1e-3
            freeze = 0.15

            anchor_x = cx + cos_a * (-a0 * cos_p) - sin_a * (-b0 * sin_p)
            anchor_y = cy + sin_a * (-a0 * cos_p) + cos_a * (-b0 * sin_p)
            wx = float(x) - anchor_x
            wy = float(y) - anchor_y
            u = (cos_a * wx + sin_a * wy) / 2.0
            v = (-sin_a * wx + cos_a * wy) / 2.0

            new_a = a0 if abs(cos_p) < freeze else max(u / cos_p, min_size)
            new_b = b0 if abs(sin_p) < freeze else max(v / sin_p, min_size)

            geometry["center"] = (
                anchor_x + cos_a * (new_a * cos_p) - sin_a * (new_b * sin_p),
                anchor_y + sin_a * (new_a * cos_p) + cos_a * (new_b * sin_p),
            )
            geometry["semi_major"] = max(new_a, min_size)
            geometry["semi_minor"] = max(new_b, min_size)
            geometry["pa_rad"] = pa
            self._set_canvas_cursor(resize_handle.get("cursor", Qt.CursorShape.SizeAllCursor))
        elif mode in ("phi_start", "phi_end"):
            semi_major = float(geometry.get("semi_major", 0.0))
            semi_minor = float(geometry.get("semi_minor", 0.0))
            if semi_major <= 1e-9 or semi_minor <= 1e-9:
                return False
            local = self._ellipse_to_local(x, y, geometry)
            if local is None:
                return False
            phi = math.atan2(local[1] / semi_minor, local[0] / semi_major)
            if mode == "phi_start":
                end_phi = float(initial.get("end_phi_rad", float(initial.get("start_phi_rad", 0.0)) + math.pi))
                new_start, new_end = self._canonical_arc_phi_pair(phi, end_phi, full_if_equal=False)
                geometry["start_phi_rad"] = new_start
                geometry["end_phi_rad"] = new_end
            else:
                start_phi = float(initial.get("start_phi_rad", 0.0))
                new_start, new_end = self._canonical_arc_phi_pair(start_phi, phi, full_if_equal=False)
                geometry["start_phi_rad"] = new_start
                geometry["end_phi_rad"] = new_end
            self._set_canvas_cursor(Qt.CursorShape.CrossCursor)
        else:
            return False

        self._set_ellipse_geometry(geometry, sync_controls=True, redraw=True)
        self._request_main_overlay_redraw()
        # Real-time PV redraw while editing when Auto Update is enabled.
        self.update_pv_diagram()
        return True

    def _on_ellipse_release(self, event_xy):
        handled = self.drag_mode == "ellipse_draw" or (
            self.edit_mode is not None and str(self.edit_mode).startswith("ellipse_")
        )
        if handled:
            geom = self.ellipse_geometry
            cleared = False
            if geom is not None and (geom["semi_major"] < 1e-6 or geom["semi_minor"] < 1e-6):
                self._clear_ellipse_path()
                cleared = True
            self.drag_mode = None
            self.edit_mode = None
            self.ellipse_drag_anchor = None
            self.initial_ellipse_geometry = None
            self.ellipse_resize_handle = None
            self.ellipse_rotation_reference_angle = None
            self._invalidate_main_overlay_background()
            if event_xy is not None:
                self._update_ellipse_hover_cursor(*event_xy)
            else:
                self._set_canvas_cursor(Qt.CursorShape.ArrowCursor)
            self._request_main_overlay_redraw()
            # Settle the PV diagram without requiring the Apply button, mirroring
            # the straight-slit finalize behavior.
            if not cleared:
                self.update_pv_diagram(force_update=True)
                self._update_main_window_marker(self.last_position_coord)
                self._sync_active_path_item_from_current_state()
                self._redraw_inactive_path_artists()
            self._record_slit_change()
        return handled

    def _update_ellipse_hover_cursor(self, x, y):
        if not self._is_ellipse_mode():
            return
        if x is None or y is None:
            self._set_canvas_cursor(Qt.CursorShape.ArrowCursor)
            return
        if self.shift_pressed and self._ellipse_contains(x, y):
            self._set_canvas_cursor(Qt.CursorShape.CrossCursor)
            return
        if self._ellipse_phase_handle(x, y):
            self._set_canvas_cursor(Qt.CursorShape.CrossCursor)
            return
        if self._ellipse_center_hit(x, y):
            self._set_canvas_cursor(Qt.CursorShape.OpenHandCursor)
            return
        resize_handle = self._ellipse_resize_handle(x, y)
        if resize_handle:
            self._set_canvas_cursor(resize_handle.get("cursor", Qt.CursorShape.SizeAllCursor))
            return
        if self._ellipse_move_hit(x, y):
            self._set_canvas_cursor(Qt.CursorShape.OpenHandCursor)
        else:
            self._set_canvas_cursor(Qt.CursorShape.ArrowCursor)

    def _finish_straight_interaction(self, event_xy=None):
        """Commit the current straight interaction and always clear drag state."""
        interaction_was_active = self._straight_interaction_active()
        try:
            # Preserve the existing release behavior for control-driven preview
            # updates too; only the drag-state cleanup depends on an active
            # pointer interaction.
            if self.is_interactive_mode:
                self.finalize_interactive_update(
                    preserve_straight_geometry=interaction_was_active,
                )
        finally:
            self.drag_mode = None
            self.edit_mode = None
            self.dragging_endpoint = None
            self.arrow_is_being_dragged = False

        if event_xy is not None:
            self._update_hover_cursor(*event_xy)
        else:
            self._set_canvas_cursor(Qt.CursorShape.ArrowCursor)

        if interaction_was_active:
            # No-op when finalize_interactive_update/apply_controls already
            # recorded the same geometry.
            self._record_slit_change()

    def on_press(self, event):
        self.arrow_is_being_dragged = False
        # --- Toolbar Mode Check ---
        # If the main window's toolbar is in Pan or Zoom mode, do nothing.
        # This allows the toolbar's event handlers to take precedence.
        if self.fits_viewer.toolbar.mode != '':
            return
        # --- End Check ---

        event_xy = self._event_xy_on_main_image(event)
        if event_xy is None:
            return

        x, y = event_xy
        tol = self.get_tolerance()

        # Polyline mode delegates all mouse interaction to PolylinePathInteraction
        # (roadmap: a delegate, not inline branches in the straight-line handlers).
        if self._is_polyline_mode():
            self._polyline.on_press(x, y, event)
            return

        # Check if clicking on the position indicator. This must run before the
        # ellipse branch so the indicator can be dragged in ellipse mode too.
        if self.pos_indicator_on_arrow and self.pos_indicator_on_arrow.get_visible():
            # Get indicator's current position (midpoint of the line)
            ind_x_data, ind_y_data = self.pos_indicator_on_arrow.get_data()
            ind_x = (ind_x_data[0] + ind_x_data[1]) / 2
            ind_y = (ind_y_data[0] + ind_y_data[1]) / 2

            # Check distance from click to indicator's center
            if np.hypot(x - ind_x, y - ind_y) < tol:
                self.dragging_pos_indicator = True
                # Invalidate the cached overlay background so the first drag frame
                # rebuilds it cleanly (with the indicator hidden during capture).
                # Without this a stale background can leave an indicator ghost.
                if self._is_ellipse_mode():
                    self._invalidate_main_overlay_background()
                self._set_canvas_cursor(Qt.CursorShape.ClosedHandCursor)
                return # Exit to avoid triggering other actions

        if self._activate_path_at_point(x, y, tol):
            return

        if self._is_ellipse_mode():
            if self.ellipse_geometry is not None and not self._active_path_hit_test(x, y, tol):
                self.begin_new_path()
            self._on_ellipse_press(x, y)
            return

        if self.line_fixed and self.line_start is not None and self.line_end is not None:
            active_hit = self._active_path_hit_test(x, y, tol)
            if (
                not active_hit
                and not (event.dblclick or self.command_pressed or self.shift_pressed or event.button == 3)
            ):
                self.begin_new_path()

        if not self.line_fixed:
            # New drawing mode: left-click to set start and end
            self.line_start = (x, y)
            self.line_end = (x, y)
            self.drag_mode = "draw"
            if self.arrow_artist is not None:
                self.arrow_artist.remove()
            self.arrow_artist = self.create_arrow_patch()
            self.fits_ax.add_patch(self.arrow_artist)
            # # Start blit mode for dragging
        else:
            # Use double-click, shift/command key, or right-click to start rotate mode
            if event.dblclick or self.command_pressed or self.shift_pressed or event.button == 3:
                self.start_rotate_mode(x, y)
                #self.start_interactive_update()
            else:
                # Left-click: decide between endpoint editing and move mode
                d0 = np.hypot(x - self.line_start[0], y - self.line_start[1])
                d1 = np.hypot(x - self.line_end[0], y - self.line_end[1])
                if d0 < tol:
                    self.edit_mode = "endpoint"
                    self.dragging_endpoint = 0
                    self.locked_end = self.line_end
                    self._set_canvas_cursor(Qt.CursorShape.CrossCursor)
                    #self.start_interactive_update()
                elif d1 < tol:
                    self.edit_mode = "endpoint"
                    self.dragging_endpoint = 1
                    self.locked_start = self.line_start
                    self._set_canvas_cursor(Qt.CursorShape.CrossCursor)
                    #self.start_interactive_update()
                else:
                    dist = self.point_line_distance((x, y), self.line_start, self.line_end)
                    if dist < tol:
                        self.edit_mode = "move"
                        self.drag_start = (x, y)
                        self.initial_line_start = self.line_start
                        self.initial_line_end = self.line_end
                        self._set_canvas_cursor(Qt.CursorShape.ClosedHandCursor)
                        #self.start_interactive_update()
                    else:
                        self.edit_mode = None
                        self._update_hover_cursor(x, y)

    def on_motion(self, event):
        # --- Toolbar Mode Check ---
        # If the main window's toolbar is in Pan or Zoom mode, do nothing.
        if self.fits_viewer.toolbar.mode != '':
            return
        # --- End Check ---

        event_xy = self._event_xy_on_main_image(event)
        if self._straight_interaction_active() and not self._motion_has_left_button(event):
            # A release outside the canvas may not reach Matplotlib. If the next
            # motion reports no left button, settle the last valid geometry
            # instead of reviving a stale drag when the pointer re-enters.
            self._finish_straight_interaction(event_xy)
            return
        if event_xy is None and self._straight_interaction_active():
            event_xy = self._event_xy_for_main_image_drag(event)
        if event_xy is None:
            self._set_canvas_cursor(Qt.CursorShape.ArrowCursor)
            return
        x, y = event_xy

        if self._is_polyline_mode():
            self._polyline.on_motion(x, y, event)
            return

        if self.dragging_pos_indicator:
            self._set_canvas_cursor(Qt.CursorShape.ClosedHandCursor)
            if self._is_ellipse_mode():
                # Move the cursor along the ellipse by projecting the mouse onto
                # the nearest path fraction, mirroring the straight-slit drag.
                fraction = self._ellipse_fraction_from_point(x, y)
                if fraction is not None:
                    position_coord = self._ellipse_position_coord_from_fraction(fraction)
                    self.update_pv_position_cursor(position_coord)
                    self._update_main_window_marker(position_coord)
                    self._request_main_overlay_redraw()
                return
            # Project mouse position onto the arrow line
            x0, y0 = self.line_start
            x1, y1 = self.line_end
            dx, dy = x1 - x0, y1 - y0
            line_len_sq = dx*dx + dy*dy

            if line_len_sq > 1e-9: # Avoid division by zero
                t = ((x - x0) * dx + (y - y0) * dy) / line_len_sq
                t = max(0, min(1, t)) # Clamp between 0 and 1

                # Calculate position_coord from the fraction
                line_length_px = np.hypot(dx, dy)
                current_unit = self.lengthUnitCombo.currentText()
                line_length_in_unit = self._convert_length(line_length_px, 'pixel', current_unit)
                position_coord = position_from_fraction(
                    t,
                    line_length_in_unit,
                    self._current_position_origin(),
                )

                # Update PV diagram cursor and main window marker
                self.update_pv_position_cursor(position_coord)
                self._update_main_window_marker(position_coord)
                self.do_interactive_update() # Use blit for indicator update
            return

        if self._is_ellipse_mode():
            self._on_ellipse_motion(x, y)
            return

        if self._straight_interaction_active():
            x, y = self._constrain_straight_drag_xy(x, y)

        # Rotate preview when line is fixed and shift key is pressed
        if self.line_fixed and self.edit_mode is None and self.shift_pressed:
            dist_line = self.point_line_distance((x, y), self.line_start, self.line_end)
            tol_val = self.get_tolerance() * 2
            if dist_line < tol_val:
                self._set_canvas_cursor(Qt.CursorShape.CrossCursor)
                cx = (self.line_start[0] + self.line_end[0]) / 2.0
                cy = (self.line_start[1] + self.line_end[1]) / 2.0
                self.center = (cx, cy)
                if self.center_marker is None:
                    self.center_marker, = self.fits_ax.plot([cx], [cy], marker='o', markersize=5,
                                                            color=self.pvmarker_color, linestyle='None', zorder = 5)
                else:
                    self.center_marker.set_data([cx], [cy])
                self.fits_canvas.draw_idle()
            else:
                if self.center_marker is not None:
                    self.center_marker.remove()
                    self.center_marker = None
                self._update_hover_cursor(x, y)

        if self.drag_mode == "draw" or self.edit_mode is not None:
            if not self.is_interactive_mode:
                self.start_interactive_update()

            self.arrow_is_being_dragged = True

            if self.drag_mode == "draw":
                self.line_end = (x, y)
                self._set_canvas_cursor(Qt.CursorShape.CrossCursor)

            elif self.edit_mode == "endpoint":
                self._set_canvas_cursor(Qt.CursorShape.CrossCursor)
                if self.dragging_endpoint == 0: self.line_start = (x, y)
                else: self.line_end = (x, y)

            elif self.edit_mode == "move":
                self._set_canvas_cursor(Qt.CursorShape.ClosedHandCursor)
                dx = x - self.drag_start[0]
                dy = y - self.drag_start[1]
                self.line_start = (self.initial_line_start[0] + dx, self.initial_line_start[1] + dy)
                self.line_end = (self.initial_line_end[0] + dx, self.initial_line_end[1] + dy)

            elif self.edit_mode == "rotate":
                self._set_canvas_cursor(Qt.CursorShape.CrossCursor)
                cx, cy = self.center
                current_angle = np.arctan2(y - cy, x - cx)
                new_angle = current_angle + np.pi if self.rotating_endpoint == 0 else current_angle
                half_length = self.initial_line_length / 2.0
                self.line_start = (cx - half_length * np.cos(new_angle), cy - half_length * np.sin(new_angle))
                self.line_end = (cx + half_length * np.cos(new_angle), cy + half_length * np.sin(new_angle))

            if self.arrow_artist is not None:
                self.arrow_artist.remove()
            self.arrow_artist = self.create_arrow_patch()
            self.fits_ax.add_patch(self.arrow_artist)
            self.update_controls()
            self._update_main_window_marker(self.last_position_coord)
            self.do_interactive_update()
            self.update_pv_diagram()
            return

        self._update_hover_cursor(x, y)

    def on_release(self, event):
        """Handle mouse release events for drawing and interaction."""
        event_xy = self._event_xy_on_main_image(event)
        if self._is_polyline_mode():
            self._polyline.on_release(event_xy, event)
            return
        if self.dragging_pos_indicator:
            self.dragging_pos_indicator = False
            if self._is_ellipse_mode():
                # The indicator drag only moves the position cursor; the ellipse
                # geometry is unchanged, so just settle the overlay/cursor with a
                # clean background rebuild to avoid leaving an indicator ghost.
                self._invalidate_main_overlay_background()
                self._request_main_overlay_redraw()
                if event_xy is not None:
                    self._update_ellipse_hover_cursor(*event_xy)
                else:
                    self._set_canvas_cursor(Qt.CursorShape.ArrowCursor)
                return
            self.finalize_interactive_update()
            if event_xy is not None:
                self._update_hover_cursor(*event_xy)
            else:
                self._set_canvas_cursor(Qt.CursorShape.ArrowCursor)
            return

        if self._is_ellipse_mode():
            if self._on_ellipse_release(event_xy):
                return

        # Straight interactions must terminate even when the pointer is outside
        # the axes and no data coordinates are available on release.
        self._finish_straight_interaction(event_xy)


    def set_pv_range(self):
        """Apply the range from QLineEdit inputs to the PV plot axes."""
        try:
            # Get values from inputs for position axis
            pos_min_val = float(self.pos_min_input.text())
            pos_max_val = float(self.pos_max_input.text())

            # Get values from inputs for velocity axis
            vel_min_val = float(self.vel_min_input.text())
            vel_max_val = float(self.vel_max_input.text())

            if pos_min_val >= pos_max_val:
                print("Warning: Position Min value must be less than Max value.")
                # Restore previous valid values from plot
                self.update_range_inputs()
                return

            if vel_min_val >= vel_max_val:
                print("Warning: Velocity Min value must be less than Max value.")
                # Restore previous valid values from plot
                self.update_range_inputs()
                return

            # Apply based on axis swap state
            pos_limits = self._display_position_limits((pos_min_val, pos_max_val))
            if self.swapAxesCheck.isChecked():
                self.pv_ax.set_ylim(*pos_limits) # Position on Y axis
                self.pv_ax.set_xlim(vel_min_val, vel_max_val) # Velocity on X axis
            else:
                self.pv_ax.set_xlim(*pos_limits) # Position on X axis
                self.pv_ax.set_ylim(vel_min_val, vel_max_val) # Velocity on Y axis

            self.is_range_manual = True # Mark that range was set manually
            self.pv_canvas.draw_idle()

        except ValueError:
            print("Invalid input for range. Please enter numeric values.")
            # Restore previous valid values from plot
            self.update_range_inputs()

    def _position_axis_flipped(self, *, fallback=None):
        check = getattr(self, "flipPositionAxisCheck", None)
        if check is not None:
            try:
                return bool(check.isChecked())
            except Exception:
                pass
        if fallback is not None:
            return bool(fallback)
        return bool(getattr(self, "position_axis_flipped", False))

    def _set_position_axis_flip_state(self, flipped):
        self.position_axis_flipped = bool(flipped)
        check = getattr(self, "flipPositionAxisCheck", None)
        if check is None:
            return
        try:
            check.blockSignals(True)
            check.setChecked(self.position_axis_flipped)
        finally:
            try:
                check.blockSignals(False)
            except Exception:
                pass

    def _display_position_limits(self, limits, *, flipped=None):
        if limits is None:
            return None
        left, right = float(limits[0]), float(limits[1])
        use_flipped = self._position_axis_flipped() if flipped is None else bool(flipped)
        if use_flipped:
            return (right, left)
        return (left, right)

    def _logical_position_limits_from_display(self, limits, *, flipped=None):
        if limits is None:
            return None
        left, right = float(limits[0]), float(limits[1])
        use_flipped = self._position_axis_flipped() if flipped is None else bool(flipped)
        if use_flipped:
            return (right, left)
        return (left, right)

    def _current_position_axis_limits(self):
        if self.pv_ax is None:
            return None
        if self.swapAxesCheck.isChecked():
            return self.pv_ax.get_ylim()
        return self.pv_ax.get_xlim()

    def _set_position_axis_limits(self, logical_limits, *, flipped=None):
        display_limits = self._display_position_limits(logical_limits, flipped=flipped)
        if display_limits is None or self.pv_ax is None:
            return
        if self.swapAxesCheck.isChecked():
            self.pv_ax.set_ylim(*display_limits)
        else:
            self.pv_ax.set_xlim(*display_limits)

    def _on_position_axis_flip_changed(self):
        new_flipped = self._position_axis_flipped()
        old_flipped = not new_flipped
        raw_limits = self._current_position_axis_limits()
        self.position_axis_flipped = new_flipped
        if raw_limits is not None:
            logical_limits = self._logical_position_limits_from_display(
                raw_limits,
                flipped=old_flipped,
            )
            self._set_position_axis_limits(logical_limits, flipped=new_flipped)
        self.update_range_inputs()
        try:
            self.pv_canvas.draw_idle()
        except Exception:
            pass

    def update_range_inputs(self):
        """Update QLineEdit inputs based on current plot axes limits."""
        if self.pv_ax is None or self.original_position_range is None:
            return

        xlim = self.pv_ax.get_xlim()
        ylim = self.pv_ax.get_ylim()

        # Determine which axis corresponds to Position and Velocity based on swap state
        if self.swapAxesCheck.isChecked():
            pos_range = ylim
            vel_range = xlim
        else:
            pos_range = xlim
            vel_range = ylim
        pos_range = self._logical_position_limits_from_display(pos_range)

        # Update Position inputs (format to reasonable precision)
        self.pos_min_input.blockSignals(True)
        self.pos_max_input.blockSignals(True)
        self.pos_min_input.setText(f"{pos_range[0]:.6g}")
        self.pos_max_input.setText(f"{pos_range[1]:.6g}")
        self.pos_min_input.blockSignals(False)
        self.pos_max_input.blockSignals(False)

        # Update Velocity inputs (format to reasonable precision)
        self.vel_min_input.blockSignals(True)
        self.vel_max_input.blockSignals(True)
        self.vel_min_input.setText(f"{vel_range[0]:.6g}")
        self.vel_max_input.setText(f"{vel_range[1]:.6g}")
        self.vel_min_input.blockSignals(False)
        self.vel_max_input.blockSignals(False)

    def get_app_state(self):
        """Get AppState from MainWindow for usecase layer access."""
        # Try to find the main window hosting the app state
        if hasattr(self.fits_viewer, 'app_state'):
            return self.fits_viewer.app_state

        # If fits_viewer is a subwindow, it might have a pointer to main_window
        main_window = getattr(self.fits_viewer, 'main_window', None)
        if main_window and hasattr(main_window, 'app_state'):
            return main_window.app_state

        return None

    def sync_pv_state(self, x0, y0, x1, y1, width):
        """Sync current PV slice parameters to app_state."""
        app_state = self.get_app_state()
        if app_state:
            set_pv_endpoints(app_state, x0, y0, x1, y1, width)

    def reset_pv_range(self):
        """Reset PV plot axes to the original full range by recalculating a new full extent."""
        self.is_range_manual = False
        self.update_pv_diagram(force_update=True)

    def _handle_pv_memory_error(self, exc, *, force_update=False):
        """Report a PV input/allocation failure without modal-dialog storms."""
        message = str(exc)
        self._set_pv_status_message(message)
        now = time.monotonic()
        previous_message = getattr(self, "_last_pv_memory_error_message", None)
        previous_time = float(getattr(self, "_last_pv_memory_error_time", 0.0) or 0.0)
        should_show = (
            message != previous_message
            or now - previous_time >= 5.0
        )
        self._last_pv_memory_error_message = message
        if should_show:
            self._last_pv_memory_error_time = now
            title = (
                "PV Memory Limit"
                if isinstance(exc, MemoryError)
                else "Invalid PV Path"
            )
            QMessageBox.warning(self, title, message)

    def _clear_pv_error_state(self):
        self._last_pv_memory_error_message = None
        self._last_pv_memory_error_time = 0.0

    def update_pv_diagram(self, force_update=False):
        """Update the PV diagram based on sampled points along the line."""
        if not self.autoUpdateCheck.isChecked() and not force_update:
            return False

        try:
            ellipse_mode = self._is_ellipse_mode()
            polyline_mode = self._is_polyline_mode()
            ellipse_path = self._current_ellipse_path_geometry() if ellipse_mode else None
            polyline_path = self._current_polyline_path_geometry() if polyline_mode else None
            # Ellipse and polyline are both driven by a path_geometry object (vs
            # the straight path's line_start/line_end), so they share most
            # branches here.
            path_geometry_mode = ellipse_mode or polyline_mode
            active_path = ellipse_path if ellipse_mode else polyline_path
            if path_geometry_mode:
                if active_path is None:
                    return False
            elif self.line_start is None or self.line_end is None:
                return False
        except (MemoryError, ValueError) as exc:
            self._handle_pv_memory_error(exc, force_update=force_update)
            return False

        current_time = time.time()
        if current_time - self.last_update_time < 0.1 and not force_update:
            return False
        self.last_update_time = current_time

        # --- Proportional Zoom Calculation (Step 1) ---
        # If already zoomed/panned (is_range_manual=True), calculate the current view's fractional state
        # BEFORE recalculating the full range based on potentially new units.
        pos_zoom_frac = None
        vel_zoom_frac = None
        if self.is_range_manual and self.original_position_range is not None and self.original_velocity_range is not None:
            is_swapped = self.swapAxesCheck.isChecked()
            if is_swapped:
                pos_view_lim = self._logical_position_limits_from_display(self.pv_ax.get_ylim())
                vel_view_lim = self.pv_ax.get_xlim()
            else:
                pos_view_lim = self._logical_position_limits_from_display(self.pv_ax.get_xlim())
                vel_view_lim = self.pv_ax.get_ylim()

            old_pos_full_lim = self.original_position_range
            pos_range_width_old = old_pos_full_lim[1] - old_pos_full_lim[0]
            if abs(pos_range_width_old) > 1e-9: # Avoid division by zero
                pos_zoom_frac = (
                    (pos_view_lim[0] - old_pos_full_lim[0]) / pos_range_width_old,
                    (pos_view_lim[1] - old_pos_full_lim[0]) / pos_range_width_old
                )

            old_vel_full_lim = self.original_velocity_range
            vel_range_width_old = old_vel_full_lim[1] - old_vel_full_lim[0]
            if abs(vel_range_width_old) > 1e-9:
                vel_zoom_frac = (
                    (vel_view_lim[0] - old_vel_full_lim[0]) / vel_range_width_old,
                    (vel_view_lim[1] - old_vel_full_lim[0]) / vel_range_width_old
                )
        # --- End Proportional Zoom Calculation ---

        # --- PV Data Calculation (Step 2) ---
        try:
            if path_geometry_mode:
                line_length_px = float(active_path.length_px)
            else:
                x0, y0 = self.line_start
                x1, y1 = self.line_end
                line_length_px = np.hypot(x1 - x0, y1 - y0)
        except (MemoryError, ValueError) as exc:
            self._handle_pv_memory_error(exc, force_update=force_update)
            return False
        sample_spacing_pix = self._current_sample_spacing_pix()
        x_axis_mode = self._current_x_axis_mode()

        # Use app_state if available
        app_state = self.get_app_state()
        if app_state:
            if path_geometry_mode:
                try:
                    pv = compute_pv(
                        app_state,
                        path_geometry=active_path,
                        width=self.slice_width,
                        sample_spacing_pix=sample_spacing_pix,
                        weight_mode=self.weight_mode,
                        sample_axis=x_axis_mode,
                    )
                except (MemoryError, ValueError) as exc:
                    self._handle_pv_memory_error(
                        exc,
                        force_update=force_update,
                    )
                    return False
            else:
                # Sync standard params
                self.sync_pv_state(x0, y0, x1, y1, self.slice_width)

                # Compute using headless usecase
                try:
                    pv = compute_pv(
                        app_state,
                        x0=x0, y0=y0, x1=x1, y1=y1,
                        width=self.slice_width,
                        sample_spacing_pix=sample_spacing_pix,
                        weight_mode=self.weight_mode
                    )
                except (MemoryError, ValueError) as exc:
                    self._handle_pv_memory_error(
                        exc,
                        force_update=force_update,
                    )
                    return False
            num_samples = pv.shape[1]
        else:
            # Fallback (should normally not happen)
            n_vel = self.data.shape[0]
            if line_length_px == 0:
                num_samples = 1
                pv = np.full((n_vel, num_samples), np.nan)
            else:
                 num_samples = sample_count_from_spacing(line_length_px, sample_spacing_pix)
                 pv = np.full((n_vel, num_samples), np.nan)
                 print("Warning: AppState not available for PV calculation.")

        # --- End PV Data Calculation ---
        self._clear_pv_error_state()

        # Preserve the historical all-NaN update path: replace the image, but
        # leave the current axes/range controls untouched.  It is still a
        # successful refresh (and therefore safe to export), not a stale result.
        if np.all(np.isnan(pv)):
            self.pv_im.set_data(pv)
            self._sync_open_pv_color_panel_data()
            self._refresh_contours()
            self.pv_canvas.draw()
            return True

        # --- New Full Range Calculation and Axis Formatting (Step 3) ---
        current_unit = self.lengthUnitCombo.currentText()
        line_length_in_unit = self._convert_length(line_length_px, 'pixel', current_unit)
        use_phi_axis = ellipse_mode and x_axis_mode == PV_X_AXIS_PHI
        position_label = 'Phi - Phi0 [deg]' if use_phi_axis else f'Position [{current_unit}]'

        n_vel = pv.shape[0]
        vel_label = "Velocity"
        try:
            v_xz = self.fits_viewer.get_viewer_by_plane('xz')
            if v_xz and v_xz.state and v_xz.state.ax_coord:
                 vel_label = v_xz.state.ax_coord[1].get_axislabel()
        except Exception:
            pass
        if self.wcs and self.wcs.wcs.naxis >= 3:
            wcs_spec = self.wcs.sub(['spectral'])
            v_coords = wcs_spec.wcs_pix2world(np.arange(n_vel), 0)[0]
            if n_vel > 1:
                dv = v_coords[1] - v_coords[0]
            else:
                spec_axis_index = next((i for i, ct in enumerate(self.wcs.wcs.ctype) if 'VEL' in ct.upper() or 'VRAD' in ct.upper() or 'VOPT' in ct.upper() or 'FREQ' in ct.upper()), -1)
                dv = self.wcs.wcs.cdelt[spec_axis_index] if spec_axis_index != -1 else 1
            y_min, y_max = v_coords[0] - dv / 2.0, v_coords[-1] + dv / 2.0
        else:
            y_min, y_max = -0.5, n_vel - 0.5

        if use_phi_axis:
            x_min_pos, x_max_pos = self._ellipse_phi_display_bounds_deg()
            self.pv_ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
            self.pv_ax.xaxis.set_major_locator(mpl.ticker.AutoLocator())
        else:
            position_origin = POSITION_ORIGIN_START if path_geometry_mode else self._current_position_origin()
            if num_samples <= 1:
                x_min_pos, x_max_pos = position_axis_bounds(line_length_in_unit, position_origin, num_samples)
                def physical_length_formatter(x, pos):
                    if position_origin == POSITION_ORIGIN_CENTER:
                        val = line_length_in_unit * x
                    else:
                        val = line_length_in_unit * (x + 0.5)
                    return f'{val:.3g}'
                self.pv_ax.xaxis.set_major_formatter(FuncFormatter(physical_length_formatter))
                self.pv_ax.set_xticks([-0.5, 0, 0.5])
            else:
                x_min_pos, x_max_pos = position_axis_bounds(line_length_in_unit, position_origin, num_samples)
                self.pv_ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
                self.pv_ax.xaxis.set_major_locator(mpl.ticker.AutoLocator())

        # Define new full ranges based on current units
        new_pos_full_range = (x_min_pos, x_max_pos)
        new_vel_full_range = (y_min, y_max)
        image_pos_extent = _sample_centered_image_extent(
            new_pos_full_range,
            num_samples,
        )
        # --- End Full Range Calculation ---


        # --- Calculate Final View Limits (Step 4) ---
        # Calculate final position limits using proportional zoom fraction
        if self.is_range_manual and pos_zoom_frac:
            pos_range_width_new = new_pos_full_range[1] - new_pos_full_range[0]
            pos_lim_final = (new_pos_full_range[0] + pos_zoom_frac[0] * pos_range_width_new,
                             new_pos_full_range[0] + pos_zoom_frac[1] * pos_range_width_new)
        else:
            pos_lim_final = new_pos_full_range

        # Calculate final velocity limits using proportional zoom fraction (or initial synced range)
        # Check against original_velocity_range which holds the initial state or manual changes from UI
        if self.is_range_manual:
            if vel_zoom_frac:
                vel_range_width_new = new_vel_full_range[1] - new_vel_full_range[0]
                vel_lim_final = (new_vel_full_range[0] + vel_zoom_frac[0] * vel_range_width_new,
                                 new_vel_full_range[0] + vel_zoom_frac[1] * vel_range_width_new)
            else:
                # If no fraction calculated (e.g., first run after initial sync), use current axis limits from UI input if possible
                try:
                    vel_lim_final = (float(self.vel_min_input.text()), float(self.vel_max_input.text()))
                except ValueError:
                    vel_lim_final = new_vel_full_range # Fallback
        else:
            vel_lim_final = new_vel_full_range

        # Update stored full ranges for next iteration
        self.original_position_range = new_pos_full_range
        if not self.is_range_manual or self.original_velocity_range is None:
             self.original_velocity_range = new_vel_full_range

        # --- Apply Plot Settings and Limits (Step 5) ---
        # Set image data and extent (full range)
        if self.swapAxesCheck.isChecked():
            pv_display = np.transpose(pv)
            self.pv_im.set_data(pv_display)
            self.pv_im.set_extent((new_vel_full_range[0], new_vel_full_range[1], image_pos_extent[0], image_pos_extent[1]))
            self._refresh_contours()
            self.pv_ax.set_ylim(*self._display_position_limits(pos_lim_final))
            self.pv_ax.set_xlim(vel_lim_final)
            self.pv_ax.set_xlabel(vel_label)
            self.pv_ax.set_ylabel(position_label)
        else:
            self.pv_im.set_data(pv)
            self.pv_im.set_extent((image_pos_extent[0], image_pos_extent[1], new_vel_full_range[0], new_vel_full_range[1]))
            self._refresh_contours()
            self.pv_ax.set_xlim(*self._display_position_limits(pos_lim_final))
            self.pv_ax.set_ylim(vel_lim_final)
            self.pv_ax.set_xlabel(position_label)
            self.pv_ax.set_ylabel(vel_label)

        self._sync_open_pv_color_panel_data()

        # Update color scale limits
        pv_settings = ColorSettingsPanel.settings[ColorMode.PV]
        if pv_settings['min_val'] is not None and pv_settings['max_val'] is not None:
            if not self.is_clim_fixed or self.min_val != pv_settings['min_val'] or self.max_val != pv_settings['max_val']:
                self.min_val = pv_settings['min_val']
                self.max_val = pv_settings['max_val']
                self.is_clim_fixed = True

        if self.is_clim_fixed:
            self.pv_im.set_clim(self.min_val, self.max_val)
        else:
            if not np.all(np.isnan(pv)):
                self.pv_im.set_clim(np.nanmin(pv), np.nanmax(pv))

        # Update UI input fields based on the final calculated limits
        self.update_range_inputs()
        self.pv_canvas.draw()
        self.pv_canvas.flush_events()
        return True

    def get_channel_from_velocity(self, velocity_coord):
        """Convert velocity coordinate from PV diagram to channel index."""
        if self.wcs and self.wcs.wcs.naxis >= 3:
            try:
                # Use the spectral part of the WCS for conversion
                spec_wcs = self.wcs.sub(['spectral'])
                # The input is already in world coordinates (e.g., km/s), so we convert it to pixel
                channel_pix = np.atleast_1d(spec_wcs.world_to_pixel_values(velocity_coord))
                return int(round(channel_pix[0]))
            except Exception as e:
                print(f"Error converting velocity to channel: {e}")
                return None
        return None

    def update_cursor(self, channel):
        """Update the cursor line position based on the channel index from MainWindow."""
        if self.wcs and self.wcs.wcs.naxis >= 3:
            try:
                # Use the spectral part of the WCS for conversion
                spec_wcs = self.wcs.sub(['spectral'])
                # Convert channel index (pixel) to velocity (world)
                velocity_coord = spec_wcs.pixel_to_world_values(channel)

                # Update cursor position and make it visible
                is_swapped = self.swapAxesCheck.isChecked()

                if is_swapped:
                    self.v_cursor_line.set_xdata([velocity_coord])
                    self.h_cursor_line.set_visible(False)
                    self.v_cursor_line.set_visible(True)
                else:
                    self.h_cursor_line.set_ydata([velocity_coord])
                    self.v_cursor_line.set_visible(False)
                    self.h_cursor_line.set_visible(True)

                self.pv_canvas.draw_idle()
            except Exception as e:
                print(f"Error updating PV cursor: {e}")

            if self.last_position_coord is not None:
                self._update_main_window_marker(self.last_position_coord)
                #self.fits_canvas.draw_idle()

    def update_pv_position_cursor(self, position_coord):
        """Updates only the position cursor on the PV diagram."""
        if self.pv_ax is None:
            return

        is_swapped = self.swapAxesCheck.isChecked()
        if is_swapped:
            self.pv_h_cursor_line.set_ydata([position_coord])
            self.pv_v_cursor_line.set_visible(False)
            self.pv_h_cursor_line.set_visible(True)
        else:
            self.pv_v_cursor_line.set_xdata([position_coord])
            self.pv_h_cursor_line.set_visible(False)
            self.pv_v_cursor_line.set_visible(True)

        self.last_position_coord = position_coord
        self.pv_canvas.draw_idle()


    def on_pv_press(self, event):
        """Handle mouse press events on the PV canvas, especially double-clicks."""
        if event.dblclick and event.inaxes is None:
            self.h_cursor_line.set_visible(False)
            self.v_cursor_line.set_visible(False)
            self.pv_v_cursor_line.set_visible(False)
            self.pv_h_cursor_line.set_visible(False)
            if self.pos_indicator_on_arrow:
                self.pos_indicator_on_arrow.set_visible(False)
                # Rebuild the cached overlay background through the blit pipeline so
                # the hidden indicator does not leave a ghost on the main image.
                self._invalidate_main_overlay_background()
                self._request_main_overlay_redraw()
            self.pv_canvas.draw_idle()
            self.last_position_coord = None
            return

        if event.dblclick:
            current_mode = self.toolbar.mode
            if current_mode == 'pan/zoom':
                # Exit pan mode on double click
                self.toolbar.pan(False)
                self.toolbar._active = None
                # Simulate a release event to properly terminate pan action in matplotlib backend
                release_event = mpl.backend_bases.MouseEvent(
                    'button_release_event', self.canvas, event.x, event.y,
                    button=event.button, dblclick=True, guiEvent=event.guiEvent
                )
                self.toolbar.release_pan(release_event)
                self.toolbar._update_buttons_checked()
                self.canvas.draw_idle()
                return

            elif current_mode == 'zoom rect':
                # Exit zoom mode on double click
                self.toolbar.zoom(False)
                self.toolbar._active = None
                # Simulate a release event to properly terminate zoom action in matplotlib backend
                release_event = mpl.backend_bases.MouseEvent(
                    'button_release_event', self.canvas, event.x, event.y,
                    button=event.button, dblclick=True, guiEvent=event.guiEvent
                )
                self.toolbar.release_zoom(release_event)
                self.toolbar._update_buttons_checked()
                self.canvas.draw_idle()
                return

        if self.toolbar.mode in ['zoom rect', 'pan/zoom']:
            return

        # Handle single left-click to update MainWindow
        if event.button == 1 and event.inaxes == self.pv_ax:
            is_swapped = self.swapAxesCheck.isChecked()
            velocity_coord = event.xdata if is_swapped else event.ydata
            position_coord = event.ydata if is_swapped else event.xdata

            self.last_position_coord = position_coord

            channel = self.get_channel_from_velocity(velocity_coord)
            if channel is not None:
                main_slider = getattr(self.fits_viewer, 'slider', None)
                if main_slider and 0 <= channel <= main_slider.maximum():
                    main_slider.setValue(channel)

            if is_swapped:
                self.pv_h_cursor_line.set_ydata([position_coord])
                self.pv_v_cursor_line.set_visible(False)
                self.pv_h_cursor_line.set_visible(True)
            else:
                self.pv_v_cursor_line.set_xdata([position_coord])
                self.pv_h_cursor_line.set_visible(False)
                self.pv_v_cursor_line.set_visible(True)


            self._update_main_window_marker(position_coord)
            if hasattr(self.fits_viewer, 'redraw_main_overlay_and_blit'):
                self.fits_viewer.redraw_main_overlay_and_blit()
            else:
                self.fits_canvas.draw_idle()
            self.pv_canvas.draw_idle()




    def on_pv_motion(self, event):
        """Handle mouse motion events on the PV canvas."""
        if self.toolbar.mode in ['zoom rect', 'pan/zoom']:
            return
        if event.button == 1 and event.inaxes == self.pv_ax:
            is_swapped = self.swapAxesCheck.isChecked()
            velocity_coord = event.xdata if self.swapAxesCheck.isChecked() else event.ydata
            position_coord = event.ydata if is_swapped else event.xdata

            self.last_position_coord = position_coord
            channel = self.get_channel_from_velocity(velocity_coord)

            if channel is not None:
                # Use broadcast_channel_update if available to sync all viewers
                if hasattr(self.fits_viewer, 'control_panel') and hasattr(self.fits_viewer.control_panel, 'broadcast_channel_update'):
                    self.fits_viewer.control_panel.broadcast_channel_update(channel)
                else:
                    # Fallback to simple slider update
                    main_slider = getattr(self.fits_viewer, 'slider', None)
                    if main_slider and 0 <= channel <= main_slider.maximum():
                        main_slider.setValue(channel)

            if is_swapped:
                self.pv_h_cursor_line.set_ydata([position_coord])
                self.pv_v_cursor_line.set_visible(False)
                self.pv_h_cursor_line.set_visible(True)
            else:
                self.pv_v_cursor_line.set_xdata([position_coord])
                self.pv_h_cursor_line.set_visible(False)
                self.pv_v_cursor_line.set_visible(True)

            self._update_main_window_marker(position_coord)
            if hasattr(self.fits_viewer, 'redraw_main_overlay_and_blit'):
                self.fits_viewer.redraw_main_overlay_and_blit()
            else:
                self.fits_canvas.draw_idle()
            self.pv_canvas.draw_idle()


    def _update_main_window_marker(self, position_coord):
        """Creates or updates the position indicator on the main window's arrow."""
        fraction = self._get_cursor_fractional_position()

        if self._is_polyline_mode():
            pn = self._polyline_point_normal_from_fraction(fraction) if fraction is not None else None
            if pn is None:
                if self.pos_indicator_on_arrow:
                    self.pos_indicator_on_arrow.set_visible(False)
                return
            base_x, base_y, normal_x, normal_y = pn
            half = self.sliceWidthSpin.value() / 2.0
            ind_start = (base_x - half * normal_x, base_y - half * normal_y)
            ind_end = (base_x + half * normal_x, base_y + half * normal_y)
            if self.pos_indicator_on_arrow is None:
                self.pos_indicator_on_arrow = mpl.lines.Line2D(
                    [ind_start[0], ind_end[0]], [ind_start[1], ind_end[1]],
                    color=self.pvarrow_color, lw=1.5 * self.arrow_size,
                    animated=self.is_interactive_mode)
                self.fits_ax.add_line(self.pos_indicator_on_arrow)
            else:
                self.pos_indicator_on_arrow.set_data([ind_start[0], ind_end[0]], [ind_start[1], ind_end[1]])
                self.pos_indicator_on_arrow.set_linewidth(1.5 * self.arrow_size)
                self.pos_indicator_on_arrow.set_animated(self.is_interactive_mode)
            self.pos_indicator_on_arrow.set_visible(True)
            return

        if self._is_ellipse_mode():
            if fraction is None:
                if self.pos_indicator_on_arrow:
                    self.pos_indicator_on_arrow.set_visible(False)
                return
            ellipse_point = self._ellipse_point_normal_from_fraction(fraction)
            if ellipse_point is None:
                if self.pos_indicator_on_arrow:
                    self.pos_indicator_on_arrow.set_visible(False)
                return
            base_x, base_y, normal_x, normal_y = ellipse_point
            half = self.sliceWidthSpin.value() / 2.0
            ind_start = (base_x - half * normal_x, base_y - half * normal_y)
            ind_end = (base_x + half * normal_x, base_y + half * normal_y)
            if self.pos_indicator_on_arrow is None:
                self.pos_indicator_on_arrow = mpl.lines.Line2D(
                    [ind_start[0], ind_end[0]], [ind_start[1], ind_end[1]],
                    color=self.pvarrow_color, lw=1.5 * self.arrow_size, animated=self.is_interactive_mode)
                self.fits_ax.add_line(self.pos_indicator_on_arrow)
            else:
                self.pos_indicator_on_arrow.set_data([ind_start[0], ind_end[0]], [ind_start[1], ind_end[1]])
                self.pos_indicator_on_arrow.set_linewidth(1.5 * self.arrow_size)
                self.pos_indicator_on_arrow.set_animated(self.is_interactive_mode)
            self.pos_indicator_on_arrow.set_visible(True)
            return

        if self.line_start is None or self.line_end is None:
             if self.pos_indicator_on_arrow:
                self.pos_indicator_on_arrow.set_visible(False)
             return

        if fraction is None:
            if self.pos_indicator_on_arrow:
                self.pos_indicator_on_arrow.set_visible(False)
            return

        fraction = max(0, min(1, fraction))

        dx = self.line_end[0] - self.line_start[0]
        dy = self.line_end[1] - self.line_start[1]

        base_x = self.line_start[0] + fraction * dx
        base_y = self.line_start[1] + fraction * dy

        mag = np.hypot(dx, dy)
        if mag > 1e-9:
            perp = (-dy / mag, dx / mag)
            half = self.sliceWidthSpin.value() / 2.0
            ind_start = (base_x - half * perp[0], base_y - half * perp[1])
            ind_end = (base_x + half * perp[0], base_y + half * perp[1])

            if self.pos_indicator_on_arrow is None:
                self.pos_indicator_on_arrow = mpl.lines.Line2D(
                    [ind_start[0], ind_end[0]], [ind_start[1], ind_end[1]],
                    color=self.pvarrow_color, lw=1.5 * self.arrow_size, animated=self.is_interactive_mode)
                self.fits_ax.add_line(self.pos_indicator_on_arrow)
            else:
                self.pos_indicator_on_arrow.set_data([ind_start[0], ind_end[0]], [ind_start[1], ind_end[1]])
                self.pos_indicator_on_arrow.set_linewidth(1.5 * self.arrow_size)
                self.pos_indicator_on_arrow.set_animated(self.is_interactive_mode)
            self.pos_indicator_on_arrow.set_visible(True)


    def get_rotation_angle(self):
        """Return the rotation angle (in degrees) with horizontal as 0°"""
        if self.line_start is None or self.line_end is None:
            return None
        angle_rad = np.arctan2(self.line_end[1] - self.line_start[1],
                                self.line_end[0] - self.line_start[0])
        return np.degrees(angle_rad)

    def compute_arrow_center(self):
        """Return the center of the arrow (midpoint of start and end)"""
        cx = (self.line_start[0] + self.line_end[0]) / 2.0
        cy = (self.line_start[1] + self.line_end[1]) / 2.0
        return (cx, cy)

    def on_key_press(self, event):
        """Handle key press events"""
        key = event.key.lower()
        if self._is_polyline_mode() and self._polyline.on_key(key):
            return
        if key in ['delete', 'backspace', 'escape']:
            self.clear_arrow()

        elif key in ['command', 'meta']:
            self.command_pressed = True
        elif key == 'shift':
            self.shift_pressed = True

    def on_key_release(self, event):
        """Handle key release events"""
        key = event.key.lower()
        if key in ['command', 'meta']:
            self.command_pressed = False
        elif key == 'shift':
            self.shift_pressed = False
            if self.center_marker is not None and self.edit_mode is None:
                self.center_marker.remove()
                self.center_marker = None
            self.fits_canvas.draw_idle()


    def on_swap_axes_changed(self):
        """Handles axis swapping, preserving cursor positions and working after clear."""
        is_swapped = self.swapAxesCheck.isChecked()

        vel_coord, pos_coord = None, None
        if self.h_cursor_line.get_visible(): vel_coord = self.h_cursor_line.get_ydata()[0]
        elif self.v_cursor_line.get_visible(): vel_coord = self.v_cursor_line.get_xdata()[0]
        if self.pv_v_cursor_line.get_visible(): pos_coord = self.pv_v_cursor_line.get_xdata()[0]
        elif self.pv_h_cursor_line.get_visible(): pos_coord = self.pv_h_cursor_line.get_ydata()[0]

        self.h_cursor_line.set_visible(False)
        self.v_cursor_line.set_visible(False)
        self.pv_h_cursor_line.set_visible(False)
        self.pv_v_cursor_line.set_visible(False)

        if self.line_start is None and not self._is_ellipse_mode():
            current_xlim = self.pv_ax.get_xlim()
            current_ylim = self.pv_ax.get_ylim()
            current_xlabel = self.pv_ax.get_xlabel()
            current_ylabel = self.pv_ax.get_ylabel()

            current_data = self.pv_im.get_array()
            left, right, bottom, top = self.pv_im.get_extent()
            self.pv_im.set_data(current_data.T)
            self._refresh_contours()

            self.pv_ax.set_xlim(current_ylim)
            self.pv_ax.set_ylim(current_xlim)
            self.pv_ax.set_xlabel(current_ylabel)
            self.pv_ax.set_ylabel(current_xlabel)

            self.pv_im.set_extent([bottom, top, left, right])

            self.pv_ax.set_aspect('auto')
        else:
            self.is_range_manual = False
            self.update_pv_diagram(force_update=True)

        if vel_coord is not None:
            if is_swapped: self.v_cursor_line.set_xdata([vel_coord]); self.v_cursor_line.set_visible(True)
            else: self.h_cursor_line.set_ydata([vel_coord]); self.h_cursor_line.set_visible(True)

        if pos_coord is not None:
            if is_swapped: self.pv_h_cursor_line.set_ydata([pos_coord]); self.pv_h_cursor_line.set_visible(True)
            else: self.pv_v_cursor_line.set_xdata([pos_coord]); self.pv_v_cursor_line.set_visible(True)

        self.pv_canvas.draw_idle()

    def _get_cursor_fractional_position(self, unit=None, origin=None):
        """
        Calculates the fractional position of the cursor along the arrow.
        If a unit is provided, it calculates the fraction relative to that unit.
        """
        if self.last_position_coord is None:
            return None

        if self._is_polyline_mode():
            if unit is None:
                unit = self.length_unit
            length = self._current_path_length_in_unit(unit=unit)
            if length > 1e-9:
                return fraction_from_position(
                    self.last_position_coord, length, POSITION_ORIGIN_START
                )
            return 0.0

        if self._is_ellipse_mode():
            if self._current_ellipse_path_geometry() is None:
                return None
            if self._current_x_axis_mode() == PV_X_AXIS_PHI:
                phi0, phi1 = self._ellipse_phi_display_bounds_deg()
                span = phi1 - phi0
                if abs(span) <= 1e-9:
                    return 0.0
                return (float(self.last_position_coord) - phi0) / span

            if unit is None:
                unit = self.length_unit
            length = self._current_path_length_in_unit(unit=unit)
            if length > 1e-9:
                return fraction_from_position(
                    self.last_position_coord,
                    length,
                    POSITION_ORIGIN_START,
                )
            return 0.0

        if self.line_start is None or self.line_end is None:
            return None

        line_length_px = np.hypot(self.line_end[0] - self.line_start[0],
                                  self.line_end[1] - self.line_start[1])

        if unit is None:
            unit = self.length_unit
        if origin is None:
            origin = self._current_position_origin()

        line_length_in_unit = self._convert_length(line_length_px, 'pixel', unit)

        if line_length_in_unit > 1e-9:
            return fraction_from_position(
                self.last_position_coord,
                line_length_in_unit,
                origin,
            )

        return 0.0

    def start_interactive_update(self):
        if self.is_interactive_mode:
            return
        self.is_interactive_mode = True

        artists = [self.arrow_artist] + self.width_indicators
        if self.pos_indicator_on_arrow:
            artists.append(self.pos_indicator_on_arrow)

        for artist in artists:
            if artist: artist.set_visible(False)

        self.fits_canvas.draw()
        self.background_cache = self.fits_canvas.copy_from_bbox(self.fits_ax.bbox)

        for artist in artists:
            if artist: artist.set_visible(True)

    def do_interactive_update(self):
        if not self.is_interactive_mode or self.background_cache is None:
            self.start_interactive_update()

        self.fits_canvas.restore_region(self.background_cache)

        artists_to_draw = [self.arrow_artist] + self.width_indicators
        if self.pos_indicator_on_arrow:
            artists_to_draw.append(self.pos_indicator_on_arrow)

        for artist in artists_to_draw:
            if artist:
                self.fits_ax.draw_artist(artist)

        self.fits_canvas.blit(self.fits_ax.bbox)
        self.fits_canvas.flush_events()

    def finalize_interactive_update(self, *, preserve_straight_geometry=False):
        if not self.is_interactive_mode:
            if self._is_ellipse_mode():
                if self.ellipse_geometry is not None:
                    self.apply_controls()
            elif self.line_start is not None:
                self.apply_controls(
                    preserve_straight_geometry=preserve_straight_geometry,
                )
            return

        self.is_interactive_mode = False
        self.background_cache = None

        artists = [self.arrow_artist, self.pos_indicator_on_arrow] + self.width_indicators
        for artist in artists:
            if artist:
                artist.set_animated(False)

        self.apply_controls(
            preserve_straight_geometry=preserve_straight_geometry,
        )

    def _default_contour_label(self) -> str:
        title = self.windowTitle() or "PV Diagram"
        return title

    def _contour_items_provider(self):
        if not hasattr(self, "pv_ax") or self.pv_ax is None:
            return []
        if not hasattr(self, "pv_im") or self.pv_im is None:
            return []
        arr = self.pv_im.get_array()
        if arr is None:
            return []
        if np.ma.isMaskedArray(arr):
            arr = arr.filled(np.nan)
        data = np.asarray(arr)
        extent = self.pv_im.get_extent() if hasattr(self.pv_im, "get_extent") else None
        metadata = {}
        try:
            clim = self.pv_im.get_clim()
        except Exception:
            clim = None
        if clim is not None:
            metadata["clim"] = tuple(clim)
        return [ContourItem(ax=self.pv_ax, data=data, label=self._default_contour_label(), extent=extent, metadata=metadata)]

    def _register_contour_layer(self):
        if self._contour_layer_id is not None:
            return
        manager = ContourManager.instance()
        layer_id = f"pv-{id(self)}"
        try:
            manager.register_layer(
                layer_id=layer_id,
                label=self._default_contour_label(),
                plane="pv",
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

    def _current_spectral_world_for_workspace(self):
        if self.wcs is None or getattr(self.wcs, "naxis", 0) < 3:
            return None
        zpix = 0.0
        try:
            getter = getattr(self.fits_viewer, "_get_shared_zpix", None)
            if callable(getter):
                zpix = float(getter())
            else:
                current = getattr(self.fits_viewer, "current_channel_index", None)
                if callable(current):
                    zpix = float(current())
        except Exception:
            zpix = 0.0
        try:
            spectral_wcs = self.wcs.sub(["spectral"])
            world = spectral_wcs.wcs_pix2world([zpix], 0)
            return float(np.atleast_1d(world)[0])
        except Exception:
            return None

    def _pixel_to_world_xy_for_workspace(self, xpix, ypix):
        if self.wcs is None:
            return None
        try:
            naxis = int(getattr(self.wcs, "naxis", 0) or 0)
        except Exception:
            naxis = 0
        if naxis < 2:
            return None
        pixel = [float(xpix), float(ypix)]
        if naxis >= 3:
            zpix = 0.0
            try:
                getter = getattr(self.fits_viewer, "_get_shared_zpix", None)
                if callable(getter):
                    zpix = float(getter())
                else:
                    current = getattr(self.fits_viewer, "current_channel_index", None)
                    if callable(current):
                        zpix = float(current())
            except Exception:
                zpix = 0.0
            pixel.append(zpix)
        while len(pixel) < naxis:
            pixel.append(0.0)
        try:
            world = self.wcs.wcs_pix2world([pixel], 0)[0]
            return float(world[0]), float(world[1])
        except Exception:
            return None

    def _longitude_candidates_for_workspace(self, lon_value):
        try:
            lon = float(lon_value)
        except Exception:
            return []
        ctype0 = ""
        try:
            ctype0 = str(self.wcs.wcs.ctype[0] or "").upper()
        except Exception:
            ctype0 = ""
        looks_wrapped = any(token in ctype0 for token in ("RA", "GLON", "LON"))
        if not looks_wrapped:
            return [lon]
        return [lon, lon - 360.0, lon + 360.0]

    def _world_xy_to_pixel_for_workspace(self, world_x, world_y, *, spectral_world=None, reference=None):
        if self.wcs is None:
            return None
        try:
            naxis = int(getattr(self.wcs, "naxis", 0) or 0)
        except Exception:
            naxis = 0
        if naxis < 2:
            return None

        try:
            wy = float(world_y)
        except Exception:
            return None

        if spectral_world is None:
            spectral_world = self._current_spectral_world_for_workspace()

        try:
            max_x = max(0.0, float(self.data.shape[-1] - 1))
            max_y = max(0.0, float(self.data.shape[-2] - 1))
        except Exception:
            max_x = 0.0
            max_y = 0.0

        ref = None
        if isinstance(reference, (list, tuple)) and len(reference) >= 2:
            try:
                ref = (float(reference[0]), float(reference[1]))
            except Exception:
                ref = None

        best = None
        best_score = None
        for wx in self._longitude_candidates_for_workspace(world_x):
            world = [float(wx), wy]
            if naxis >= 3:
                try:
                    world.append(float(spectral_world if spectral_world is not None else 0.0))
                except Exception:
                    world.append(0.0)
            while len(world) < naxis:
                world.append(0.0)
            try:
                pix = self.wcs.wcs_world2pix([world], 0)[0]
                x = float(pix[0])
                y = float(pix[1])
            except Exception:
                continue
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            in_bounds = (0.0 <= x <= max_x and 0.0 <= y <= max_y)
            if ref is not None:
                score = (x - ref[0]) ** 2 + (y - ref[1]) ** 2
            else:
                score = (x - max_x / 2.0) ** 2 + (y - max_y / 2.0) ** 2
            if not in_bounds:
                score += 1.0e9
            if best_score is None or score < best_score:
                best = (x, y)
                best_score = score
        return best

    def _slit_overlay_color(self):
        combo = getattr(self, "arrowColorCombo", None)
        if combo is not None:
            try:
                text = str(combo.currentText() or "").strip()
                if text:
                    return text
            except Exception:
                pass
        return getattr(self, "pvarrow_color", "yellow")

    def _build_slit_overlay_payload(self):
        """Sample the current slit into a static overlay dict (pixel coords).

        Returns ``None`` when no usable slit geometry exists. The result is a
        plain dict (see ``pv_slit_overlay.build_slit_overlay``) that can be drawn
        on any window sharing the main image's xy pixel grid.
        """
        path_type = self._current_path_type()
        geometry = None
        closed = False
        try:
            if path_type in ("ellipse", "ellipse_arc"):
                geometry = self._current_ellipse_path_geometry()
                closed = path_type == "ellipse"
            elif path_type == "polyline":
                geometry = self._current_polyline_path_geometry()
            else:
                if self.line_start is None or self.line_end is None:
                    return None
                geometry = StraightPathGeometry.from_endpoints(
                    float(self.line_start[0]), float(self.line_start[1]),
                    float(self.line_end[0]), float(self.line_end[1]),
                )
        except Exception:
            geometry = None
        if geometry is None:
            return None

        try:
            samples = sample_path_points(
                geometry,
                sample_spacing_pix=self._current_sample_spacing_pix(),
                sample_axis=self._current_x_axis_mode(),
            )
        except Exception:
            return None

        try:
            width_px = float(self.sliceWidthSpin.value())
        except Exception:
            width_px = 0.0
        try:
            linewidth = max(0.5, 1.5 * float(self.arrowSizeSpin.value()))
        except Exception:
            linewidth = 1.5

        return build_slit_overlay(
            samples.xs,
            samples.ys,
            samples.normal_x,
            samples.normal_y,
            width_px=width_px,
            color=self._slit_overlay_color(),
            linewidth=linewidth,
            closed=closed,
            label=f"PV slit ({self.fits_viewer.filename})",
        )

    def _live_xy_overlay_targets(self):
        """Return [(window, title)] of live xy result windows that can host a slit."""
        targets = []
        seen = set()
        refs = getattr(self.fits_viewer, "integ_result_windows", None) or []
        for ref in list(refs):
            try:
                window = ref() if callable(ref) else ref
            except Exception:
                window = None
            if window is None or id(window) in seen:
                continue
            if getattr(window, "plane", None) != "xy":
                continue
            if not hasattr(window, "add_slit_overlay"):
                continue
            seen.add(id(window))
            try:
                title = window.windowTitle() or "Result window"
            except Exception:
                title = "Result window"
            targets.append((window, title))
        return targets

    def _copy_slit_to_window(self, window):
        overlay = self._build_slit_overlay_payload()
        if overlay is None:
            self._set_pv_status_message(
                "No slit to copy — draw or place a slit first."
            )
            return
        try:
            window.add_slit_overlay(overlay)
        except Exception as exc:
            self._set_pv_status_message(f"Could not copy slit: {exc}")
            return
        self._set_pv_status_message("Slit copied to the selected window.")

    def _populate_copy_slit_menu(self, menu):
        menu.clear()
        targets = self._live_xy_overlay_targets()
        if not targets:
            action = menu.addAction("No xy result windows open")
            action.setEnabled(False)
            return
        for window, title in targets:
            action = menu.addAction(title)
            action.triggered.connect(
                lambda _checked=False, w=window: self._copy_slit_to_window(w)
            )
        windows_with_slit = [
            (window, title)
            for window, title in targets
            if getattr(window, "has_slit_overlay", None) and window.has_slit_overlay()
        ]
        if windows_with_slit:
            menu.addSeparator()
            for window, title in windows_with_slit:
                action = menu.addAction(f"Clear copied slit: {title}")
                action.triggered.connect(
                    lambda _checked=False, w=window: self._clear_slit_from_window(w)
                )

    def _clear_slit_from_window(self, window):
        try:
            window.clear_slit_overlays()
        except Exception:
            return
        self._set_pv_status_message("Copied slit cleared.")

    def _set_pv_status_message(self, message):
        # Do not call QMainWindow.statusBar() here. Qt creates the bottom status
        # bar on first use, which slightly changes the PV/integration window size.
        pass

    def _export_single_path_state(self):
        state = {
            "schema": 1,
            "path_type": self._current_path_type(),
            "line_pixel": None,
            "line_world": None,
            "line_world_raw": None,
            "ellipse_geometry": None,
            "polyline": None,
            "spectral_world": self._current_spectral_world_for_workspace(),
            "slice_width": float(self.sliceWidthSpin.value()),
            "sample_spacing_pix": self._current_sample_spacing_pix(),
            "weight_mode": int(self.weight_mode),
            "position_origin": self._current_position_origin(),
            "endpoint_position_origin": normalize_position_origin(
                getattr(self, "_endpoint_position_origin", POSITION_ORIGIN_START)
            ),
            "geometry_input_mode": self._current_geometry_input_mode(),
            "swap_axes": bool(self.swapAxesCheck.isChecked()),
            "position_axis_flipped": self._position_axis_flipped(),
            "x_axis_mode": self._current_x_axis_mode(),
            "auto_update": bool(self.autoUpdateCheck.isChecked()),
            "length_unit": str(self.lengthUnitCombo.currentText() or "pixel"),
            "ellipse_axis_unit": self._current_ellipse_axis_unit(),
            "arrow_color": str(self.arrowColorCombo.currentText() or self.pvarrow_color),
            "arrow_size": float(self.arrowSizeSpin.value()),
            "indicator_positions": {
                "start": bool(self.indicator_positions.get("start", False)),
                "center": bool(self.indicator_positions.get("center", False)),
                "end": bool(self.indicator_positions.get("end", False)),
            },
            "ellipse_indicator_positions": {
                "start": bool(self.ellipse_indicator_positions.get("start", False)),
                "center": bool(self.ellipse_indicator_positions.get("center", True)),
                "end": bool(self.ellipse_indicator_positions.get("end", False)),
            },
            "range_inputs": {
                "pos_min": str(self.pos_min_input.text() or "").strip(),
                "pos_max": str(self.pos_max_input.text() or "").strip(),
                "vel_min": str(self.vel_min_input.text() or "").strip(),
                "vel_max": str(self.vel_max_input.text() or "").strip(),
            },
            "is_range_manual": bool(self.is_range_manual),
            "last_position_coord": None,
        }

        if self.last_position_coord is not None:
            try:
                state["last_position_coord"] = float(self.last_position_coord)
            except Exception:
                pass

        state["ellipse_geometry"] = self._ellipse_geometry_for_workspace()
        state["polyline"] = self._polyline_block_for_workspace()

        if self.line_start is None or self.line_end is None:
            return state

        try:
            x0, y0 = float(self.line_start[0]), float(self.line_start[1])
            x1, y1 = float(self.line_end[0]), float(self.line_end[1])
            state["line_pixel"] = {"start": [x0, y0], "end": [x1, y1]}
        except Exception:
            state["line_pixel"] = None

        try:
            start_world_raw = self._pixel_to_world_xy_for_workspace(self.line_start[0], self.line_start[1])
            end_world_raw = self._pixel_to_world_xy_for_workspace(self.line_end[0], self.line_end[1])
            if start_world_raw is not None and end_world_raw is not None:
                state["line_world_raw"] = {
                    "start": [float(start_world_raw[0]), float(start_world_raw[1])],
                    "end": [float(end_world_raw[0]), float(end_world_raw[1])],
                }
                state["line_world"] = {
                    "start": [f"{float(start_world_raw[0]):.12g}", f"{float(start_world_raw[1]):.12g}"],
                    "end": [f"{float(end_world_raw[0]):.12g}", f"{float(end_world_raw[1]):.12g}"],
                }
        except Exception:
            pass

        return state

    def export_workspace_state(self):
        self._sync_active_path_item_from_current_state(sync_combo=False)
        state = self._export_single_path_state()
        state["pv_paths"] = self._path_items_for_workspace()
        state["active_pv_path_id"] = self.active_pv_path_id
        self._sync_path_list_combo()
        return state

    def _extract_line_from_workspace_state(self, state):
        start = None
        end = None
        line_pixel = state.get("line_pixel")
        reference_start = None
        reference_end = None
        if isinstance(line_pixel, dict):
            start_pixel = line_pixel.get("start")
            end_pixel = line_pixel.get("end")
            if isinstance(start_pixel, (list, tuple)) and len(start_pixel) >= 2:
                try:
                    reference_start = (float(start_pixel[0]), float(start_pixel[1]))
                except Exception:
                    reference_start = None
            if isinstance(end_pixel, (list, tuple)) and len(end_pixel) >= 2:
                try:
                    reference_end = (float(end_pixel[0]), float(end_pixel[1]))
                except Exception:
                    reference_end = None

        spectral_world = state.get("spectral_world")
        try:
            spectral_world = float(spectral_world) if spectral_world is not None else None
        except Exception:
            spectral_world = None

        world_line_raw = state.get("line_world_raw")
        if isinstance(world_line_raw, dict):
            start_world = world_line_raw.get("start")
            end_world = world_line_raw.get("end")
            if (
                isinstance(start_world, (list, tuple)) and len(start_world) >= 2
                and isinstance(end_world, (list, tuple)) and len(end_world) >= 2
            ):
                start = self._world_xy_to_pixel_for_workspace(
                    start_world[0],
                    start_world[1],
                    spectral_world=spectral_world,
                    reference=reference_start,
                )
                end = self._world_xy_to_pixel_for_workspace(
                    end_world[0],
                    end_world[1],
                    spectral_world=spectral_world,
                    reference=reference_end,
                )

        world_line = state.get("line_world")
        if (start is None or end is None) and isinstance(world_line, dict):
            start_world = world_line.get("start")
            end_world = world_line.get("end")
            if (
                isinstance(start_world, (list, tuple)) and len(start_world) >= 2
                and isinstance(end_world, (list, tuple)) and len(end_world) >= 2
            ):
                start = self._world_xy_to_pixel_for_workspace(
                    start_world[0],
                    start_world[1],
                    spectral_world=spectral_world,
                    reference=reference_start,
                )
                end = self._world_xy_to_pixel_for_workspace(
                    end_world[0],
                    end_world[1],
                    spectral_world=spectral_world,
                    reference=reference_end,
                )

        if start is None or end is None:
            if isinstance(line_pixel, dict):
                start_pixel = line_pixel.get("start")
                end_pixel = line_pixel.get("end")
                if (
                    isinstance(start_pixel, (list, tuple)) and len(start_pixel) >= 2
                    and isinstance(end_pixel, (list, tuple)) and len(end_pixel) >= 2
                ):
                    try:
                        start = (float(start_pixel[0]), float(start_pixel[1]))
                        end = (float(end_pixel[0]), float(end_pixel[1]))
                    except Exception:
                        start = None
                        end = None

        if start is None or end is None:
            return None, None

        max_x = max(0.0, float(self.data.shape[-1] - 1))
        max_y = max(0.0, float(self.data.shape[-2] - 1))
        x0 = min(max(float(start[0]), 0.0), max_x)
        y0 = min(max(float(start[1]), 0.0), max_y)
        x1 = min(max(float(end[0]), 0.0), max_x)
        y1 = min(max(float(end[1]), 0.0), max_y)
        if np.hypot(x1 - x0, y1 - y0) < 1e-6:
            return None, None
        return (x0, y0), (x1, y1)

    def restore_workspace_state(self, state):
        if not isinstance(state, dict):
            return False

        if not getattr(self, "_restoring_single_pv_path", False):
            raw_items = state.get("pv_paths")
            if isinstance(raw_items, list):
                items = []
                for raw_item in raw_items:
                    item = self._coerce_pv_path_item(raw_item)
                    if item is not None:
                        items.append(item)
                if items:
                    self._clear_inactive_path_artists()
                    self.pv_path_items = items
                    requested_id = state.get("active_pv_path_id")
                    matched_active = self._find_pv_path_item(requested_id) is not None
                    if not matched_active:
                        requested_id = items[0].get("id")
                    self.active_pv_path_id = requested_id
                    active_item = self._find_pv_path_item(self.active_pv_path_id)
                    active_state = dict(active_item.get("state") or {})
                    if matched_active:
                        # Overlay the snapshot's live (top-level) fields only when
                        # they describe the active path. When active_pv_path_id was
                        # None/unmatched -- e.g. an undo snapshot captured with no
                        # active path -- those fields are empty and must not clobber
                        # the stored geometry of the fallback path.
                        for key, value in state.items():
                            if key in {"pv_paths", "active_pv_path_id"}:
                                continue
                            if key in active_state:
                                active_state[key] = value
                    active_item["state"] = active_state
                    self._restoring_single_pv_path = True
                    try:
                        restored = self.restore_workspace_state(active_state)
                    finally:
                        self._restoring_single_pv_path = False
                    self._sync_path_list_combo()
                    self._redraw_inactive_path_artists()
                    self._invalidate_main_overlay_background()
                    self._request_main_overlay_redraw()
                    if restored:
                        self._reveal_pv_result_if_ready()
                    return bool(restored)

        desired_unit = str(state.get("length_unit") or self.lengthUnitCombo.currentText() or "pixel")
        if self.lengthUnitCombo.findText(desired_unit) >= 0:
            self.lengthUnitCombo.setCurrentText(desired_unit)

        desired_ellipse_unit = str(state.get("ellipse_axis_unit") or self._current_ellipse_axis_unit() or "pixel")
        if self.ellipseAxisUnitCombo.findText(desired_ellipse_unit) >= 0:
            try:
                self.ellipseAxisUnitCombo.blockSignals(True)
                self.ellipseAxisUnitCombo.setCurrentText(desired_ellipse_unit)
                self.ellipse_axis_unit = desired_ellipse_unit
            finally:
                self.ellipseAxisUnitCombo.blockSignals(False)

        desired_path_type = self._set_path_type_from_state(state.get("path_type", "straight"))

        desired_x_axis_mode = normalize_pv_x_axis_mode(state.get("x_axis_mode", PV_X_AXIS_POSITION))
        try:
            idx = self.xAxisModeCombo.findData(desired_x_axis_mode)
            if idx >= 0:
                self.xAxisModeCombo.blockSignals(True)
                self.xAxisModeCombo.setCurrentIndex(idx)
                self.x_axis_mode = desired_x_axis_mode
                self.xAxisModeCombo.blockSignals(False)
        except Exception:
            try:
                self.xAxisModeCombo.blockSignals(False)
            except Exception:
                pass
        self._sync_x_axis_controls()
        self._set_position_axis_flip_state(bool(state.get("position_axis_flipped", False)))

        try:
            self.autoUpdateCheck.setChecked(bool(state.get("auto_update", False)))
        except Exception:
            pass
        try:
            self.weight_mode = int(state.get("weight_mode", self.weight_mode))
        except Exception:
            pass
        if self.weight_mode not in (0, 1):
            self.weight_mode = 0
        self.bilinearRadio.blockSignals(True)
        self.gaussianRadio.blockSignals(True)
        self.bilinearRadio.setChecked(self.weight_mode == 0)
        self.gaussianRadio.setChecked(self.weight_mode == 1)
        self.bilinearRadio.blockSignals(False)
        self.gaussianRadio.blockSignals(False)
        if self.arrowColorCombo.findText(str(state.get("arrow_color", ""))) >= 0:
            self.arrowColorCombo.setCurrentText(str(state.get("arrow_color")))
        try:
            self.arrowSizeSpin.setValue(float(state.get("arrow_size", self.arrowSizeSpin.value())))
        except Exception:
            pass
        try:
            width = float(state.get("slice_width", self.sliceWidthSpin.value()))
            self.sliceWidthSpin.setValue(max(0.0, min(99.0, width)))
        except Exception:
            pass

        sample_spacing = self._coerce_sample_spacing_pix(state.get("sample_spacing_pix"))
        try:
            self.sampleSpacingSpin.blockSignals(True)
            self.sampleSpacingSpin.setValue(sample_spacing)
        except Exception:
            pass
        finally:
            try:
                self.sampleSpacingSpin.blockSignals(False)
            except Exception:
                pass
        self._sync_sampling_controls()

        desired_origin = normalize_position_origin(state.get("position_origin", POSITION_ORIGIN_START))
        stored_geometry_input = str(state.get("geometry_input_mode") or "endpoints")
        endpoint_origin_fallback = (
            desired_origin
            if stored_geometry_input == "endpoints"
            else POSITION_ORIGIN_START
        )
        self._endpoint_position_origin = normalize_position_origin(
            state.get("endpoint_position_origin", endpoint_origin_fallback)
        )
        try:
            idx = self.positionOriginCombo.findData(desired_origin)
            if idx >= 0:
                self.positionOriginCombo.blockSignals(True)
                self.positionOriginCombo.setCurrentIndex(idx)
                self.position_origin = desired_origin
                self.positionOriginCombo.blockSignals(False)
        except Exception:
            try:
                self.positionOriginCombo.blockSignals(False)
            except Exception:
                pass

        desired_geometry_input = str(state.get("geometry_input_mode") or "endpoints")
        if desired_geometry_input not in ("endpoints", "center"):
            desired_geometry_input = "endpoints"
        try:
            idx = self.geometryInputCombo.findData(desired_geometry_input)
            if idx >= 0:
                self.geometryInputCombo.blockSignals(True)
                self.geometryInputCombo.setCurrentIndex(idx)
                self.geometry_input_mode = desired_geometry_input
                self.geometryInputCombo.blockSignals(False)
                self._sync_geometry_input_enabled()
        except Exception:
            try:
                self.geometryInputCombo.blockSignals(False)
            except Exception:
                pass

        indicator_state = state.get("indicator_positions")
        if isinstance(indicator_state, dict):
            for key in ("start", "center", "end"):
                self.indicator_positions[key] = bool(
                    indicator_state.get(key, self.indicator_positions.get(key, False))
                )
        ellipse_indicator_state = state.get("ellipse_indicator_positions")
        if isinstance(ellipse_indicator_state, dict):
            for key in ("start", "center", "end"):
                self.ellipse_indicator_positions[key] = bool(
                    ellipse_indicator_state.get(key, self.ellipse_indicator_positions.get(key, False))
                )
        # Reflect the restored flags for whichever path type is now active.
        self._sync_indicator_checks_for_mode()

        if self._is_polyline_path_type(desired_path_type):
            self._clear_straight_path()
            self._clear_ellipse_path()
            self._clear_polyline_path()
            # Restore the spline type/smoothness before drawing so overlay+PV match.
            self.polyline_spline_type, self.polyline_smoothness = self._polyline_spline_from_state(state)
            self._sync_polyline_curve_controls()
            verts = self._polyline_vertices_from_state(state)
            restored_polyline = len(verts) >= 2
            if restored_polyline:
                self.polyline_vertices = [(float(x), float(y)) for (x, y) in verts]
                self.polyline_finished = True
                self.polyline_selected_index = None
                self._draw_polyline_overlay()
            desired_swap = bool(state.get("swap_axes", False))
            try:
                if bool(self.swapAxesCheck.isChecked()) != desired_swap:
                    self.swapAxesCheck.setChecked(desired_swap)
            except Exception:
                pass
            range_inputs = state.get("range_inputs")
            if isinstance(range_inputs, dict):
                self.pos_min_input.setText(str(range_inputs.get("pos_min", self.pos_min_input.text() or "")))
                self.pos_max_input.setText(str(range_inputs.get("pos_max", self.pos_max_input.text() or "")))
                self.vel_min_input.setText(str(range_inputs.get("vel_min", self.vel_min_input.text() or "")))
                self.vel_max_input.setText(str(range_inputs.get("vel_max", self.vel_max_input.text() or "")))
            if restored_polyline:
                try:
                    self.update_pv_diagram(force_update=True)
                except Exception:
                    pass
            if restored_polyline and bool(state.get("is_range_manual", False)):
                try:
                    self.set_pv_range()
                except Exception:
                    pass
            try:
                self.pv_canvas.draw_idle()
            except Exception:
                pass
            try:
                if hasattr(self.fits_viewer, "redraw_main_overlay_and_blit"):
                    self.fits_viewer.redraw_main_overlay_and_blit()
                else:
                    self.fits_canvas.draw_idle()
            except Exception:
                pass
            if restored_polyline and not getattr(self, "_restoring_single_pv_path", False):
                self._sync_active_path_item_from_current_state()
                self._redraw_inactive_path_artists()
            if restored_polyline:
                self._reveal_pv_result_if_ready()
            return restored_polyline

        if self._is_ellipse_path_type(desired_path_type):
            self._clear_straight_path()
            self._clear_ellipse_path()
            restored_ellipse = False
            ellipse_geometry = state.get("ellipse_geometry")
            if ellipse_geometry is not None:
                restored_ellipse = self._set_ellipse_geometry(
                    ellipse_geometry,
                    sync_controls=True,
                    redraw=True,
                )
            desired_swap = bool(state.get("swap_axes", False))
            try:
                if bool(self.swapAxesCheck.isChecked()) != desired_swap:
                    self.swapAxesCheck.setChecked(desired_swap)
            except Exception:
                pass
            range_inputs = state.get("range_inputs")
            if isinstance(range_inputs, dict):
                self.pos_min_input.setText(str(range_inputs.get("pos_min", self.pos_min_input.text() or "")))
                self.pos_max_input.setText(str(range_inputs.get("pos_max", self.pos_max_input.text() or "")))
                self.vel_min_input.setText(str(range_inputs.get("vel_min", self.vel_min_input.text() or "")))
                self.vel_max_input.setText(str(range_inputs.get("vel_max", self.vel_max_input.text() or "")))
            if restored_ellipse:
                try:
                    self.update_pv_diagram(force_update=True)
                except Exception:
                    pass
            if restored_ellipse and bool(state.get("is_range_manual", False)):
                try:
                    self.set_pv_range()
                except Exception:
                    pass
            if restored_ellipse and state.get("last_position_coord") is not None:
                try:
                    position_coord = float(state.get("last_position_coord"))
                    self.update_pv_position_cursor(position_coord)
                    self._update_main_window_marker(position_coord)
                except Exception:
                    pass
            try:
                self.pv_canvas.draw_idle()
            except Exception:
                pass
            try:
                if hasattr(self.fits_viewer, "redraw_main_overlay_and_blit"):
                    self.fits_viewer.redraw_main_overlay_and_blit()
                else:
                    self.fits_canvas.draw_idle()
            except Exception:
                pass
            if restored_ellipse and not getattr(self, "_restoring_single_pv_path", False):
                self._sync_active_path_item_from_current_state()
                self._redraw_inactive_path_artists()
            if restored_ellipse:
                self._reveal_pv_result_if_ready()
            return restored_ellipse

        self._clear_ellipse_path()

        start_end = self._extract_line_from_workspace_state(state)
        line_start, line_end = start_end
        restored_line = False
        if line_start is not None and line_end is not None:
            x0, y0 = line_start
            x1, y1 = line_end
            for spin, value in (
                (self.startXSpin, x0),
                (self.startYSpin, y0),
                (self.endXSpin, x1),
                (self.endYSpin, y1),
            ):
                try:
                    spin.blockSignals(True)
                    spin.setValue(float(value))
                finally:
                    spin.blockSignals(False)
            self.line_start = (x0, y0)
            self.line_end = (x1, y1)
            self.line_fixed = True
            self._update_world_from_pixel()
            try:
                if self.arrow_artist is not None:
                    self.arrow_artist.remove()
            except Exception:
                pass
            try:
                self.arrow_artist = self.create_arrow_patch()
                self.fits_ax.add_patch(self.arrow_artist)
            except Exception:
                self.arrow_artist = None
            try:
                self.update_controls()
            except Exception:
                pass
            try:
                self.update_pv_diagram(force_update=True)
            except Exception:
                pass
            try:
                self._update_main_window_marker(self.last_position_coord)
            except Exception:
                pass
            restored_line = True

        desired_swap = bool(state.get("swap_axes", False))
        try:
            if bool(self.swapAxesCheck.isChecked()) != desired_swap:
                self.swapAxesCheck.setChecked(desired_swap)
        except Exception:
            pass

        range_inputs = state.get("range_inputs")
        if isinstance(range_inputs, dict):
            self.pos_min_input.setText(str(range_inputs.get("pos_min", self.pos_min_input.text() or "")))
            self.pos_max_input.setText(str(range_inputs.get("pos_max", self.pos_max_input.text() or "")))
            self.vel_min_input.setText(str(range_inputs.get("vel_min", self.vel_min_input.text() or "")))
            self.vel_max_input.setText(str(range_inputs.get("vel_max", self.vel_max_input.text() or "")))

        if restored_line and bool(state.get("is_range_manual", False)):
            try:
                self.set_pv_range()
            except Exception:
                pass

        if restored_line and state.get("last_position_coord") is not None:
            try:
                position_coord = float(state.get("last_position_coord"))
                self.update_pv_position_cursor(position_coord)
                self._update_main_window_marker(position_coord)
            except Exception:
                pass

        try:
            self.pv_canvas.draw_idle()
        except Exception:
            pass
        try:
            if hasattr(self.fits_viewer, "redraw_main_overlay_and_blit"):
                self.fits_viewer.redraw_main_overlay_and_blit()
            else:
                self.fits_canvas.draw_idle()
        except Exception:
            pass

        if restored_line and not getattr(self, "_restoring_single_pv_path", False):
            self._sync_active_path_item_from_current_state()
            self._redraw_inactive_path_artists()

        if restored_line:
            self._reveal_pv_result_if_ready()
        return restored_line

    def showEvent(self, event):
        super().showEvent(event)
        # While the PV window is open, the main window routes Undo/Redo to the
        # slit (the slit is edited on the main canvas, so the main window is the
        # active window during editing and must serve the shortcut).
        self._register_slit_undo_tool(True)
        if (
            not getattr(self, "_pv_result_revealed", False)
            and not getattr(self, "_pv_compact_positioned", False)
        ):
            QTimer.singleShot(0, self._position_compact_panel_next_to_main)

    def _register_slit_undo_tool(self, active):
        main = getattr(self, "fits_viewer", None)
        if main is None:
            return
        try:
            if active:
                main._pv_slit_undo_tool = self
            elif getattr(main, "_pv_slit_undo_tool", None) is self:
                main._pv_slit_undo_tool = None
        except Exception:
            return
        refresh = getattr(main, "_refresh_undo_redo_actions", None)
        if callable(refresh):
            try:
                refresh()
            except Exception:
                pass

    def closeEvent(self, event):
        # Suppress undo recording while the window tears itself down, and hand
        # Undo/Redo back to the analysis session.
        self._restoring_slit_state = True
        self._register_slit_undo_tool(False)
        self._unregister_contour_layer()
        super().closeEvent(event)

        self.is_interactive_mode = False
        self.background_cache = None

        self._clear_inactive_path_artists()
        self.pv_path_items = []
        self.active_pv_path_id = None
        self.clear_arrow()

        self.fits_canvas.mpl_disconnect(self.cid_press)
        self.fits_canvas.mpl_disconnect(self.cid_key_press)
        self.fits_canvas.mpl_disconnect(self.cid_motion)
        self.fits_canvas.mpl_disconnect(self.cid_release)
        self.fits_canvas.mpl_disconnect(self.cid_key_release)

        if self.color_settings_panel is not None:
            self.color_settings_panel.close()
            self.color_settings_panel = None
        ColorSettingsPanel.settings[ColorMode.PV]['min_val'] = None
        ColorSettingsPanel.settings[ColorMode.PV]['max_val'] = None

        try:
            plt.close(self.pv_fig)
        except Exception:
            pass

        self._set_canvas_cursor(Qt.CursorShape.ArrowCursor)
