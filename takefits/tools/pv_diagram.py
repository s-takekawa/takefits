from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QGroupBox, QFormLayout, QCheckBox, QComboBox, QDoubleSpinBox, QLabel, QButtonGroup, QRadioButton, QLineEdit, QGridLayout, QFileDialog, QMessageBox
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
from takefits.core.contour_manager import ContourManager, ContourItem
from takefits.core.coordinate import CoordinateConverter
from astropy.wcs.utils import proj_plane_pixel_scales
from takefits.core.history_provenance import build_processing_history_lines_with_action
from takefits.core.usecases import compute_pv, set_pv_endpoints, export_pv_fits, export_figure


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
        self.length_unit = 'pixel'  # Default unit

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

        self.indicator_positions = {
            "start": True,
            "center": False,
            "end": False
        }
        self.width_indicators = []

        self.slice_width = 1.0       # Default slice width

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

        central_layout = QVBoxLayout()
        main_layout.addLayout(central_layout, stretch=5)

        control_layout_right = QVBoxLayout()
        main_layout.addLayout(control_layout_right, stretch=1)

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

        # --- Right Panel: Arrow Controls ---
        arrow_control_group = QGroupBox("Arrow Controls")
        arrow_control_layout = QFormLayout(arrow_control_group)

        # Start coordinates
        spinbox_width = 90
        self.startXSpin = QDoubleSpinBox()
        self.startXSpin.setFixedWidth(spinbox_width)
        self.startXSpin.setRange(0, data_shape_x); self.startXSpin.setDecimals(2)
        self.startLonEdit = QLineEdit()
        start_x_layout = QHBoxLayout(); start_x_layout.addWidget(self.startXSpin); start_x_layout.addWidget(self.startLonEdit)
        arrow_control_layout.addRow("Start X:", start_x_layout)

        self.startYSpin = QDoubleSpinBox()
        self.startYSpin.setFixedWidth(spinbox_width)
        self.startYSpin.setRange(0, data_shape_y); self.startYSpin.setDecimals(2)
        self.startLatEdit = QLineEdit()
        start_y_layout = QHBoxLayout(); start_y_layout.addWidget(self.startYSpin); start_y_layout.addWidget(self.startLatEdit)
        arrow_control_layout.addRow("Start Y:", start_y_layout)

        # End coordinates
        self.endXSpin = QDoubleSpinBox()
        self.endXSpin.setFixedWidth(spinbox_width)
        self.endXSpin.setRange(0, data_shape_x); self.endXSpin.setDecimals(2)
        self.endLonEdit = QLineEdit()
        end_x_layout = QHBoxLayout(); end_x_layout.addWidget(self.endXSpin); end_x_layout.addWidget(self.endLonEdit)
        arrow_control_layout.addRow("End X:", end_x_layout)

        self.endYSpin = QDoubleSpinBox()
        self.endYSpin.setFixedWidth(spinbox_width)
        self.endYSpin.setRange(0, data_shape_y); self.endYSpin.setDecimals(2)
        self.endLatEdit = QLineEdit()
        end_y_layout = QHBoxLayout(); end_y_layout.addWidget(self.endYSpin); end_y_layout.addWidget(self.endLatEdit)
        arrow_control_layout.addRow("End Y:", end_y_layout)

        # Angle
        self.rotationAngleSpin = QDoubleSpinBox()
        self.rotationAngleSpin.setRange(-180, 180)
        self.rotationAngleSpin.setDecimals(1)
        self.rotationAngleSpin.setSingleStep(1)
        arrow_control_layout.addRow("Angle (°):", self.rotationAngleSpin)

        # Length
        length_layout = QHBoxLayout()
        self.arrowLengthSpin = QDoubleSpinBox()
        self.arrowLengthSpin.setRange(0.0, 99999.0)
        self.arrowLengthSpin.setValue(0)
        self.arrowLengthSpin.setFixedWidth(120)

        self.arrowLengthSpin.setDecimals(0)
        self.arrowLengthSpin.setSingleStep(1.0)

        length_layout.addWidget(self.arrowLengthSpin)

        self.lengthUnitCombo = QComboBox()
        self.lengthUnitCombo.addItems(['pixel', 'deg', 'arcmin', 'arcsec'])
        length_layout.addWidget(self.lengthUnitCombo)
        length_layout.addStretch()
        arrow_control_layout.addRow("Length:", length_layout)

        if self.pixel_scale_deg is None:
            for i in range(1, 4): # deg, arcmin, arcsec
                self.lengthUnitCombo.model().item(i).setEnabled(False)

        self.sliceWidthSpin = QDoubleSpinBox()
        self.sliceWidthSpin.setRange(0, 99)
        self.sliceWidthSpin.setDecimals(2)
        self.sliceWidthSpin.setValue(self.slice_width)
        arrow_control_layout.addRow("Slice Width:", self.sliceWidthSpin)

        indicators_layout = QHBoxLayout()
        self.startIndicatorCheck = QCheckBox("Start")
        self.centerIndicatorCheck = QCheckBox("Center")
        self.endIndicatorCheck = QCheckBox("End")
        self.startIndicatorCheck.setChecked(self.indicator_positions["start"])
        self.centerIndicatorCheck.setChecked(self.indicator_positions["center"])
        self.endIndicatorCheck.setChecked(self.indicator_positions["end"])
        indicators_layout.addWidget(self.startIndicatorCheck)
        indicators_layout.addWidget(self.centerIndicatorCheck)
        indicators_layout.addWidget(self.endIndicatorCheck)
        arrow_control_layout.addRow("Indicators:", indicators_layout)

        self.interpGroup = QButtonGroup(self)
        self.bilinearRadio = QRadioButton("Bilinear")
        self.gaussianRadio = QRadioButton("Gaussian")
        self.bilinearRadio.setChecked(True)
        self.interpGroup.addButton(self.bilinearRadio, 0)
        self.interpGroup.addButton(self.gaussianRadio, 1)
        self.interpGroup.buttonClicked.connect(
            lambda _btn: setattr(self, "weight_mode", self.interpGroup.checkedId())
        )
        interp_layout = QHBoxLayout()
        interp_layout.addWidget(self.bilinearRadio)
        interp_layout.addWidget(self.gaussianRadio)
        arrow_control_layout.addRow("Interpolation:", interp_layout)

        self.arrowColorCombo = QComboBox()
        colors = ["blue", "red", "green", "cyan", "magenta", "black", "white", "gray", "orange", "purple", "yellow", "olive"]
        self.arrowColorCombo.addItems(colors)
        self.arrowColorCombo.setCurrentText(self.pvarrow_color)
        self.arrowColorCombo.currentTextChanged.connect(lambda new_color: setattr(self, "pvarrow_color", new_color))
        arrow_control_layout.addRow("Arrow Color:", self.arrowColorCombo)

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
        arrow_control_layout.addRow("Arrow Width:", self.arrowSizeSpin)

        self.swapAxesCheck = QCheckBox("Swap PV Axes")
        self.swapAxesCheck.stateChanged.connect(self.on_swap_axes_changed)
        arrow_control_layout.addRow(self.swapAxesCheck)

        self.autoUpdateCheck = QCheckBox("Auto Update")
        self.autoUpdateCheck.setChecked(False)
        self.autoUpdateCheck.stateChanged.connect(self._on_auto_update_changed)
        arrow_control_layout.addRow(self.autoUpdateCheck)

        arrow_button_layout = QHBoxLayout()
        self.applyArrowButton = QPushButton("Apply")
        self.applyArrowButton.clicked.connect(self.apply_controls)
        self.clearArrowButton = QPushButton("Clear")
        self.clearArrowButton.clicked.connect(self.clear_arrow)
        arrow_button_layout.addWidget(self.applyArrowButton)
        arrow_button_layout.addWidget(self.clearArrowButton)
        arrow_control_layout.addRow(arrow_button_layout)

        control_layout_right.addWidget(arrow_control_group)

        # Connect signals for synchronization
        self.startXSpin.valueChanged.connect(self.update_arrow_from_gui)
        self.startYSpin.valueChanged.connect(self.update_arrow_from_gui)
        self.endXSpin.valueChanged.connect(self.update_arrow_from_gui)
        self.endYSpin.valueChanged.connect(self.update_arrow_from_gui)

        self.startLonEdit.editingFinished.connect(self._update_pixel_from_world)
        self.startLatEdit.editingFinished.connect(self._update_pixel_from_world)
        self.endLonEdit.editingFinished.connect(self._update_pixel_from_world)
        self.endLatEdit.editingFinished.connect(self._update_pixel_from_world)

        self.rotationAngleSpin.valueChanged.connect(self.update_arrow_from_rotation)
        self.arrowLengthSpin.valueChanged.connect(self.update_arrow_from_length)
        self.lengthUnitCombo.currentTextChanged.connect(self._on_unit_changed)
        self.sliceWidthSpin.valueChanged.connect(self.update_arrow_from_gui)

        # editingFinishedシグナルは、インタラクティブモードを終了し、最終的な描画を行う
        self.startXSpin.editingFinished.connect(self.finalize_interactive_update)
        self.startYSpin.editingFinished.connect(self.finalize_interactive_update)
        self.endXSpin.editingFinished.connect(self.finalize_interactive_update)
        self.endYSpin.editingFinished.connect(self.finalize_interactive_update)
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
            self.indicator_positions["start"] = self.startIndicatorCheck.isChecked()
            self.indicator_positions["center"] = self.centerIndicatorCheck.isChecked()
            self.indicator_positions["end"] = self.endIndicatorCheck.isChecked()
            self.update_arrow_from_gui()
        self.startIndicatorCheck.stateChanged.connect(lambda _: update_indicator_positions())
        self.centerIndicatorCheck.stateChanged.connect(lambda _: update_indicator_positions())
        self.endIndicatorCheck.stateChanged.connect(lambda _: update_indicator_positions())

        self._on_unit_changed(self.lengthUnitCombo.currentText())

        # --- Blit高速化のための変数 ---
        self.background_cache = None
        self.is_interactive_mode = False
        

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
            self.last_position_coord = fractional_pos * new_display_length
        
        self.update_pv_diagram(force_update=True)

        if self.last_position_coord is not None:
            self.update_pv_position_cursor(self.last_position_coord)
            self._update_main_window_marker(self.last_position_coord)


    def _on_auto_update_changed(self, state):
        if state == Qt.CheckState.Checked.value:
            self.update_pv_diagram(force_update=True)

    def _on_arrow_width_changed(self, new_size):
        """Handles changes to the arrow width spinbox."""
        self.arrow_size = new_size
        if self.arrow_artist:
            # Re-create the main arrow and width indicators with the new size
            self.arrow_artist.remove()
            self.arrow_artist = self.create_arrow_patch()
            self.fits_ax.add_patch(self.arrow_artist)

        # Update the position indicator with the new size
        self._update_main_window_marker(self.last_position_coord)
        self.fits_canvas.draw_idle()

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
            self.color_settings_panel = ColorSettingsPanel(
                mode=ColorMode.PV,
                fits_viewer=self,
                data=self.data,
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

    def on_color_settings_closed(self):
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
        if self.line_start is None or self.line_end is None:
            QMessageBox.warning(self, "Save Error", "Please draw a PV slice on the main window first before saving.")
            return

        action_params = {
            "width": float(self.sliceWidthSpin.value()),
        }
        if int(self.weight_mode) != 0:
            action_params["weight_mode"] = int(self.weight_mode)
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
            export_pv_fits(
                app_state,
                self.pv_im.get_array(),
                path,
                x0=self.line_start[0],
                y0=self.line_start[1],
                x1=self.line_end[0],
                y1=self.line_end[1],
                is_swapped=self.swapAxesCheck.isChecked(),
                history_entries=history
            )
            QMessageBox.information(self, "Save Successful", f"FITS successfully saved as: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save FITS:\n{str(e)}")


    def clear_arrow(self):
        """Clear all arrow-related artists from the canvas and reset the state."""
        # --- 1. Collect all artists to remove ---
        artists_to_remove = [
            self.arrow_artist,
            self.pos_indicator_on_arrow,
            self.marker_artist_start,
            self.marker_artist_end,
            self.center_marker
        ]
        if self.width_indicators:
            artists_to_remove.extend(self.width_indicators)

        # --- 2. Safely remove each artist ---
        for artist in artists_to_remove:
            if artist and artist.axes:
                try:
                    artist.remove()
                except Exception as e:
                    print(f"Error removing artist: {e}")

        # --- 3. Reset all state variables ---
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
        
        # --- 4. Request a redraw of the main canvas ---
        self.fits_canvas.draw_idle()

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
            # Block signals to prevent loops while updating programmatically
            for spin in [self.startXSpin, self.startYSpin, self.endXSpin, self.endYSpin]:
                spin.blockSignals(True)

            self.startXSpin.setValue(self.line_start[0])
            self.startYSpin.setValue(self.line_start[1])
            self.endXSpin.setValue(self.line_end[0])
            self.endYSpin.setValue(self.line_end[1])
            
            # Unblock signals after setting values
            for spin in [self.startXSpin, self.startYSpin, self.endXSpin, self.endYSpin]:
                spin.blockSignals(False)

            # Update WCS fields from the new pixel values
            self._update_world_from_pixel()

            angle_rad = np.arctan2(self.line_end[1] - self.line_start[1],
                                self.line_end[0] - self.line_start[0])
            self.rotationAngleSpin.setValue(np.degrees(angle_rad))

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

    def update_arrow_from_rotation(self):
        if self.line_start is None or self.line_end is None: return
        if not self.is_interactive_mode: self.start_interactive_update()
            
        cx = (self.line_start[0] + self.line_end[0]) / 2.0
        cy = (self.line_start[1] + self.line_end[1]) / 2.0
        length = np.hypot(self.line_end[0] - self.line_start[0], self.line_end[1] - self.line_start[1])
        half = length / 2.0
        new_angle = np.radians(self.rotationAngleSpin.value())
        self.line_start = (cx - half * np.cos(new_angle), cy - half * np.sin(new_angle))
        self.line_end = (cx + half * np.cos(new_angle), cy + half * np.sin(new_angle))
        
        if self.arrow_artist is not None: self.arrow_artist.remove()
        self.arrow_artist = self.create_arrow_patch()
        self.fits_ax.add_patch(self.arrow_artist)

        self.update_controls()
        self._update_main_window_marker(self.last_position_coord)
        self.do_interactive_update()
        self.update_pv_diagram()

    def update_arrow_from_length(self):
        if self.line_start is None: return
        if not self.is_interactive_mode: self.start_interactive_update()

        angle_rad = np.radians(self.rotationAngleSpin.value())
        current_unit = self.lengthUnitCombo.currentText()
        gui_length = self.arrowLengthSpin.value()
        length_px = self._convert_length(gui_length, current_unit, 'pixel')
        
        self.arrow_length = length_px
        self.line_end = (self.line_start[0] + length_px * np.cos(angle_rad),
                        self.line_start[1] + length_px * np.sin(angle_rad))

        if self.arrow_artist is not None:
            self.arrow_artist.remove()
        self.arrow_artist = self.create_arrow_patch()
        self.fits_ax.add_patch(self.arrow_artist)

        self._update_pixel_from_world_for_end_point()
        self._update_main_window_marker(self.last_position_coord)
        self.do_interactive_update()
        self.update_pv_diagram()

    def apply_controls(self):
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
            cx = (self.line_start[0] + self.line_end[0]) / 2.0
            cy = (self.line_start[1] + self.line_end[1]) / 2.0
            length = np.hypot(self.line_end[0] - self.line_start[0],
                              self.line_end[1] - self.line_start[1])
            half = length / 2.0
            new_angle = np.radians(self.rotationAngleSpin.value())
            self.line_start = (cx - half * np.cos(new_angle), cy - half * np.sin(new_angle))
            self.line_end = (cx + half * np.cos(new_angle), cy + half * np.sin(new_angle))
        self.line_fixed = True
        if self.arrow_artist is not None:
            self.arrow_artist.remove()
        self.arrow_artist = self.create_arrow_patch()
        self.fits_ax.add_patch(self.arrow_artist)
        self.update_controls()
        self.update_pv_diagram(force_update=True)
        self._update_main_window_marker(self.last_position_coord)


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

    def on_press(self, event):
        self.arrow_is_being_dragged = False
        # --- Toolbar Mode Check ---
        # If the main window's toolbar is in Pan or Zoom mode, do nothing.
        # This allows the toolbar's event handlers to take precedence.
        if self.fits_viewer.toolbar.mode != '':
            return
        # --- End Check ---
        
        if event.inaxes != self.fits_ax:
            return

        x, y = event.xdata, event.ydata
        tol = self.get_tolerance()

        # Check if clicking on the position indicator
        if self.pos_indicator_on_arrow and self.pos_indicator_on_arrow.get_visible():
            # Get indicator's current position (midpoint of the line)
            ind_x_data, ind_y_data = self.pos_indicator_on_arrow.get_data()
            ind_x = (ind_x_data[0] + ind_x_data[1]) / 2
            ind_y = (ind_y_data[0] + ind_y_data[1]) / 2

            # Check distance from click to indicator's center
            if np.hypot(x - ind_x, y - ind_y) < tol:
                self.dragging_pos_indicator = True
                self._set_canvas_cursor(Qt.CursorShape.ClosedHandCursor)
                return # Exit to avoid triggering other actions

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

        if event.xdata is None or event.ydata is None:
            self._set_canvas_cursor(Qt.CursorShape.ArrowCursor)
            return
        x, y = event.xdata, event.ydata

        if self.dragging_pos_indicator:
            self._set_canvas_cursor(Qt.CursorShape.ClosedHandCursor)
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
                position_coord = t * line_length_in_unit

                # Update PV diagram cursor and main window marker
                self.update_pv_position_cursor(position_coord)
                self._update_main_window_marker(position_coord)
                self.do_interactive_update() # Use blit for indicator update
            return

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
        if self.dragging_pos_indicator:
            self.dragging_pos_indicator = False
            self.finalize_interactive_update()
            self._update_hover_cursor(event.xdata, event.ydata)
            return

        if event.inaxes != self.fits_ax:
            self._set_canvas_cursor(Qt.CursorShape.ArrowCursor)
            return

        # This now handles finalization for ALL mouse operations
        if self.is_interactive_mode:
            self.finalize_interactive_update()

        self.drag_mode = None
        self.edit_mode = None
        self.dragging_endpoint = None
        self.arrow_is_being_dragged = False
        
        self._update_hover_cursor(event.xdata, event.ydata)


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
            if self.swapAxesCheck.isChecked():
                self.pv_ax.set_ylim(pos_min_val, pos_max_val) # Position on Y axis
                self.pv_ax.set_xlim(vel_min_val, vel_max_val) # Velocity on X axis
            else:
                self.pv_ax.set_xlim(pos_min_val, pos_max_val) # Position on X axis
                self.pv_ax.set_ylim(vel_min_val, vel_max_val) # Velocity on Y axis
            
            self.is_range_manual = True # Mark that range was set manually
            self.pv_canvas.draw_idle()

        except ValueError:
            print("Invalid input for range. Please enter numeric values.")
            # Restore previous valid values from plot
            self.update_range_inputs()

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

 
    def update_pv_diagram(self, force_update=False):
        """Update the PV diagram based on sampled points along the line."""
        if not self.autoUpdateCheck.isChecked() and not force_update:
            return

        if self.line_start is None or self.line_end is None:
            return

        current_time = time.time()
        if current_time - self.last_update_time < 0.1 and not force_update:
            return
        self.last_update_time = current_time

        # --- Proportional Zoom Calculation (Step 1) ---
        # If already zoomed/panned (is_range_manual=True), calculate the current view's fractional state
        # BEFORE recalculating the full range based on potentially new units.
        pos_zoom_frac = None
        vel_zoom_frac = None
        if self.is_range_manual and self.original_position_range is not None and self.original_velocity_range is not None:
            is_swapped = self.swapAxesCheck.isChecked()
            if is_swapped:
                pos_view_lim = self.pv_ax.get_ylim()
                vel_view_lim = self.pv_ax.get_xlim()
            else:
                pos_view_lim = self.pv_ax.get_xlim()
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
        x0, y0 = self.line_start
        x1, y1 = self.line_end
        line_length_px = np.hypot(x1 - x0, y1 - y0)

        # Use app_state if available
        app_state = self.get_app_state()
        if app_state:
            # Sync standard params
            self.sync_pv_state(x0, y0, x1, y1, self.slice_width)
            
            # Compute using headless usecase
            pv = compute_pv(
                app_state,
                x0=x0, y0=y0, x1=x1, y1=y1,
                width=self.slice_width,
                weight_mode=self.weight_mode
            )
            num_samples = pv.shape[1]
        else:
            # Fallback (should normally not happen)
            n_vel = self.data.shape[0]
            if line_length_px == 0:
                num_samples = 1
                pv = np.full((n_vel, num_samples), np.nan)
            else:
                 num_samples = max(1, int(round(line_length_px)))
                 pv = np.full((n_vel, num_samples), np.nan)
                 print("Warning: AppState not available for PV calculation.")

        # --- End PV Data Calculation ---

        if np.all(np.isnan(pv)):
            self.pv_im.set_data(pv)
            self._refresh_contours()
            self.pv_canvas.draw()
            return

        # --- New Full Range Calculation and Axis Formatting (Step 3) ---
        current_unit = self.lengthUnitCombo.currentText()
        line_length_in_unit = self._convert_length(line_length_px, 'pixel', current_unit)
        position_label = f'Position [{current_unit}]'

        n_vel = self.data.shape[0]
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

        if num_samples <= 1:
            x_min_pos, x_max_pos = -0.5, 0.5 # Nominal extent for sub-pixel drawing
            def physical_length_formatter(x, pos):
                val = line_length_in_unit * (x + 0.5)
                return f'{val:.3g}'
            self.pv_ax.xaxis.set_major_formatter(FuncFormatter(physical_length_formatter))
            self.pv_ax.set_xticks([-0.5, 0, 0.5])
        else:
            x_min_pos, x_max_pos = 0, line_length_in_unit
            self.pv_ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
            self.pv_ax.xaxis.set_major_locator(mpl.ticker.AutoLocator())

        # Define new full ranges based on current units
        new_pos_full_range = (x_min_pos, x_max_pos)
        new_vel_full_range = (y_min, y_max)
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
            self.pv_im.set_extent((new_vel_full_range[0], new_vel_full_range[1], new_pos_full_range[0], new_pos_full_range[1])) # ★ Use FULL range for extent
            self._refresh_contours()
            self.pv_ax.set_ylim(pos_lim_final)
            self.pv_ax.set_xlim(vel_lim_final)
            self.pv_ax.set_xlabel(vel_label)
            self.pv_ax.set_ylabel(position_label)
        else:
            self.pv_im.set_data(pv)
            self.pv_im.set_extent((new_pos_full_range[0], new_pos_full_range[1], new_vel_full_range[0], new_vel_full_range[1])) # ★ Use FULL range for extent
            self._refresh_contours()
            self.pv_ax.set_xlim(pos_lim_final)
            self.pv_ax.set_ylim(vel_lim_final)
            self.pv_ax.set_xlabel(position_label)
            self.pv_ax.set_ylabel(vel_label)

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
                self.fits_canvas.draw_idle()
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

        if self.line_start is None:
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

    def _get_cursor_fractional_position(self, unit=None):
        """
        Calculates the fractional position of the cursor along the arrow.
        If a unit is provided, it calculates the fraction relative to that unit.
        """
        if self.last_position_coord is None or self.line_start is None or self.line_end is None:
            return None

        line_length_px = np.hypot(self.line_end[0] - self.line_start[0],
                                  self.line_end[1] - self.line_start[1])
        
        if unit is None:
            unit = self.length_unit
        
        line_length_in_unit = self._convert_length(line_length_px, 'pixel', unit)

        if line_length_in_unit > 1e-9:
            return self.last_position_coord / line_length_in_unit
        
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

    def finalize_interactive_update(self):
        if not self.is_interactive_mode:
            if self.line_start is not None:
                self.apply_controls()
            return
            
        self.is_interactive_mode = False
        self.background_cache = None
        
        artists = [self.arrow_artist, self.pos_indicator_on_arrow] + self.width_indicators
        for artist in artists:
            if artist:
                artist.set_animated(False)
        
        self.apply_controls()

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

    def export_workspace_state(self):
        state = {
            "schema": 1,
            "line_pixel": None,
            "line_world": None,
            "line_world_raw": None,
            "spectral_world": self._current_spectral_world_for_workspace(),
            "slice_width": float(self.sliceWidthSpin.value()),
            "weight_mode": int(self.weight_mode),
            "swap_axes": bool(self.swapAxesCheck.isChecked()),
            "auto_update": bool(self.autoUpdateCheck.isChecked()),
            "length_unit": str(self.lengthUnitCombo.currentText() or "pixel"),
            "arrow_color": str(self.arrowColorCombo.currentText() or self.pvarrow_color),
            "arrow_size": float(self.arrowSizeSpin.value()),
            "indicator_positions": {
                "start": bool(self.startIndicatorCheck.isChecked()),
                "center": bool(self.centerIndicatorCheck.isChecked()),
                "end": bool(self.endIndicatorCheck.isChecked()),
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

        desired_unit = str(state.get("length_unit") or self.lengthUnitCombo.currentText() or "pixel")
        if self.lengthUnitCombo.findText(desired_unit) >= 0:
            self.lengthUnitCombo.setCurrentText(desired_unit)

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

        indicator_state = state.get("indicator_positions")
        if isinstance(indicator_state, dict):
            for checkbox, key in (
                (self.startIndicatorCheck, "start"),
                (self.centerIndicatorCheck, "center"),
                (self.endIndicatorCheck, "end"),
            ):
                try:
                    checkbox.blockSignals(True)
                    checkbox.setChecked(bool(indicator_state.get(key, checkbox.isChecked())))
                finally:
                    checkbox.blockSignals(False)
            self.indicator_positions["start"] = bool(self.startIndicatorCheck.isChecked())
            self.indicator_positions["center"] = bool(self.centerIndicatorCheck.isChecked())
            self.indicator_positions["end"] = bool(self.endIndicatorCheck.isChecked())

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

        return restored_line

    def closeEvent(self, event):
        self._unregister_contour_layer()
        super().closeEvent(event)

        self.is_interactive_mode = False
        self.background_cache = None

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
