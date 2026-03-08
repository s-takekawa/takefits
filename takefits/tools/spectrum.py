from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGridLayout, QLineEdit, QCheckBox, QLabel,
    QFrame, QSizePolicy, QMessageBox, QFileDialog, QSpacerItem,
    QPushButton, QHBoxLayout, QSpinBox, QPlainTextEdit
)
from PySide6.QtCore import Qt, QTimer, Signal as pyqtSignal
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import time
import os


from takefits.core.coordinate import CoordinateConverter
from takefits.core.wcs_frames import (
    normalize_display_frame,
    plane_values_for_display,
    axis_is_longitude,
    axis_is_latitude,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from takefits.core.region import CircleRegion, RectangleRegion, EllipseRegion
from takefits.core.app_state import RegionSpec
from takefits.core.usecases import (
    get_averaged_spectrum,
    export_spectrum,
    export_figure,
    get_spectrum,
    fit_gaussian_spectrum,
)

class SpecWindow(QWidget):
    is_on = False
    channel_changed = pyqtSignal(int)

    
    def __init__(self, fits_viewer):
        super().__init__()
        self.fits_viewer = fits_viewer
        self.wcs = self.fits_viewer.wcs
        self.config = self.fits_viewer.displaymap.config
        self.converter = CoordinateConverter(self.wcs, self.config)

        # Initialize coordinates
        self.x = 0
        self.y = 0
        self.z = 0
        self.world_x = 0.0
        self.world_y = 0.0


        if hasattr(self.fits_viewer, 'region_manager'):
            self.fits_viewer.region_manager.selected_region_changed.connect(self.on_region_changed)

        z_axis_size = self.fits_viewer.data.shape[self.fits_viewer.data.ndim-3] 
        self.initial_x_range = (0, z_axis_size - 1)
        self.initial_y_range = (0, 1)
        
        if self.fits_viewer.data.ndim == 3:
            self.slices = (0, 0, 'x')
        elif self.fits_viewer.data.ndim == 4:
            self.slices = (0, 0, 'x', 0)

        self.auto_y_axis = True
        self.last_update_time = 0
        self.spec_axis = self.wcs.wcs.spec
        
        
        n_channels = self.fits_viewer.header[f'NAXIS{self.spec_axis + 1}']
        crval = self.fits_viewer.header[f'CRVAL{self.spec_axis + 1}']
        cdelt = self.fits_viewer.header[f'CDELT{self.spec_axis + 1}'] 
        crpix = self.fits_viewer.header[f'CRPIX{self.spec_axis + 1}'] 
        self.velocity_values = crval + (np.arange(n_channels) - (crpix - 1)) * cdelt
        
        self.is_dragging = False
        #self.pick_radius = 5  # pixels within which a click is considered 'on the line'
        self.active_region = None
        self.fit_result = None
        self.fit_signature = None
        self.fit_x_full = None
        self.fit_view_xlim = None
        self.fit_component_lines = []
        self.fit_legend = None
        self._fit_resample_in_progress = False
        self._fit_resample_timer = QTimer(self)
        self._fit_resample_timer.setSingleShot(True)
        self._fit_resample_timer.setInterval(40)
        self._fit_resample_timer.timeout.connect(self._resample_fit_for_view)

        self.initUI()
        self._initialized_on_show = False
        self._suppress_next_viewer_initialize = False

    def _current_display_frame(self):
        getter = getattr(self.fits_viewer, "_get_shared_display_frame", None)
        if callable(getter):
            try:
                return normalize_display_frame(getter())
            except Exception:
                return "native"
        return "native"

    def _shared_native_world_values(self):
        values = []
        for getter_name in (
            "_get_shared_world_x",
            "_get_shared_world_y",
            "_get_shared_world_z",
            "_get_shared_world_s",
        ):
            getter = getattr(self.fits_viewer, getter_name, None)
            if not callable(getter):
                values.append(None)
                continue
            try:
                values.append(float(getter()))
            except Exception:
                values.append(None)
        return values

    def _sync_coordinate_context(self):
        self.wcs = getattr(self.fits_viewer, "wcs", self.wcs)
        displaymap = getattr(self.fits_viewer, "displaymap", None)
        config = getattr(displaymap, "config", None)
        if isinstance(config, dict):
            self.config = config
        self.converter.wcs = self.wcs
        self.converter.config = self.config

    def _format_title_world_coordinates(self, world_native):
        self._sync_coordinate_context()
        if world_native is None or len(world_native) < 2:
            raise ValueError("Insufficient world coordinates for title formatting.")

        native_x = float(world_native[0])
        native_y = float(world_native[1])
        axis_types = self.converter.get_axis_types()
        axis_x = axis_types[0] if len(axis_types) > 0 else ""
        axis_y = axis_types[1] if len(axis_types) > 1 else ""

        display_x = native_x
        display_y = native_y
        display_axis_x = axis_x
        display_axis_y = axis_y

        display_frame = self._current_display_frame()
        if display_frame != "native":
            fallback_native_world = list(world_native)
            shared_world = self._shared_native_world_values()
            for idx, value in enumerate(shared_world):
                if idx >= len(fallback_native_world):
                    break
                if value is not None:
                    fallback_native_world[idx] = value
            try:
                transformed = plane_values_for_display(
                    self.wcs,
                    "xy",
                    native_x,
                    native_y,
                    frame=display_frame,
                    fallback_native_world=fallback_native_world,
                )
                display_x, display_y, axis_x_t, axis_y_t = transformed
                if axis_x_t:
                    display_axis_x = axis_x_t
                if axis_y_t:
                    display_axis_y = axis_y_t
            except Exception:
                pass

        def _apply_longitude_wrap(value, axis_type):
            axis_upper = str(axis_type or "").upper()
            wrapped = float(value)
            if axis_upper.startswith("RA"):
                if wrapped < 0.0:
                    wrapped += 360.0
                elif wrapped > 360.0:
                    wrapped -= 360.0
                return wrapped
            if "GLON" in axis_upper or axis_upper.endswith("LON"):
                wrap_mode = int(self.config.get("coord_wrap", 180))
                if wrap_mode == 180:
                    if wrapped < -180.0:
                        wrapped += 360.0
                    elif wrapped > 180.0:
                        wrapped -= 360.0
                else:
                    if wrapped < 0.0:
                        wrapped += 360.0
                    elif wrapped > 360.0:
                        wrapped -= 360.0
            return wrapped

        display_x = _apply_longitude_wrap(display_x, display_axis_x)
        display_y = _apply_longitude_wrap(display_y, display_axis_y)

        world_x_str = self.converter.format_world_coordinate(display_x, display_axis_x)
        world_y_str = self.converter.format_world_coordinate(display_y, display_axis_y)

        # Append " deg" unit if configured for decimal display and the axis is angular
        is_decimal = self.config.get("decimal", True)
        if is_decimal:
            if axis_is_longitude(display_axis_x) or axis_is_latitude(display_axis_x):
                world_x_str += " deg"
            if axis_is_longitude(display_axis_y) or axis_is_latitude(display_axis_y):
                world_y_str += " deg"

        return world_x_str, world_y_str

    def initUI(self):
        layout = QVBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(10, 0, 10, 0)

        self.setLayout(layout)
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection=self.fits_viewer.wcs, slices=self.slices)
        self.ax.callbacks.connect("xlim_changed", self._on_fit_xlim_changed)

        self.canvas = FigureCanvas(self.fig)
        self.fig.subplots_adjust(bottom=0.15) 

        # CODE ADDED: Connect mouse events for dragging
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('button_release_event', self.on_release)

        self.toolbar = SpecNavigationToolbar(self.canvas, self)

        range_layout = QGridLayout()
        range_layout.setHorizontalSpacing(5)
        range_layout.setVerticalSpacing(0)
    
        # Horizontal Range
        range_layout.addWidget(QLabel("Horizontal (X):"), 0, 0)
        self.x_min_input = QLineEdit(str(self.initial_x_range[0]))
        self.x_min_input.setFixedWidth(60)
        range_layout.addWidget(self.x_min_input, 0, 1)
        self.x_min_input.setText(f"{np.nanmin(self.velocity_values):.4g}")
    
        range_layout.addWidget(QLabel("to"), 0, 2)
    
        self.x_max_input = QLineEdit(str(self.initial_x_range[1]))
        self.x_max_input.setFixedWidth(60)
        range_layout.addWidget(self.x_max_input, 0, 3)
        self.x_max_input.setText(f"{np.nanmax(self.velocity_values):.4g}")

        vline1 = QFrame()
        vline1.setFrameShape(QFrame.Shape.VLine)
        vline1.setFrameShadow(QFrame.Shadow.Sunken)
        range_layout.addWidget(vline1, 0, 4, 1, 1)
    
        # Vertical Range
        range_layout.addWidget(QLabel("Vertical (Y):"), 0, 5)
        self.y_min_input = QLineEdit(str(self.initial_y_range[0]))
        self.y_min_input.setFixedWidth(60)
        range_layout.addWidget(self.y_min_input, 0, 6)
    
        range_layout.addWidget(QLabel("to"), 0, 7)
    
        self.y_max_input = QLineEdit(str(self.initial_y_range[1]))
        self.y_max_input.setFixedWidth(60)
        range_layout.addWidget(self.y_max_input, 0, 8)
    
        layout.addLayout(range_layout) 
        layout.addWidget(self.canvas)

        extract_layout = QHBoxLayout()
        extract_layout.setSpacing(4)

        self.fit_button = QPushButton("Gaussian Fit")
        self.fit_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.fit_button.setFixedWidth(108)
        self.fit_button.clicked.connect(self.fit_gaussian_components)
        extract_layout.addWidget(self.fit_button)

        self.fit_n_label = QLabel("N")
        self.fit_n_label.setFixedWidth(24)
        self.fit_n_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        extract_layout.addWidget(self.fit_n_label)

        self.fit_n_spinbox = QSpinBox()
        self.fit_n_spinbox.setRange(1, 20)
        self.fit_n_spinbox.setValue(3)
        self.fit_n_spinbox.setFixedWidth(52)
        spin_line_edit = self.fit_n_spinbox.lineEdit()
        if spin_line_edit is not None:
            spin_line_edit.returnPressed.connect(self.fit_gaussian_components)
        extract_layout.addWidget(self.fit_n_spinbox)

        self.fit_baseline_checkbox = QCheckBox("Fit baseline")
        self.fit_baseline_checkbox.setChecked(False)
        extract_layout.addWidget(self.fit_baseline_checkbox)
        extract_layout.addSpacing(8)

        self.clear_fit_button = QPushButton("Clear Fit")
        self.clear_fit_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.clear_fit_button.setFixedWidth(74)
        self.clear_fit_button.clicked.connect(self.clear_fit)
        extract_layout.addWidget(self.clear_fit_button)

        self.show_fit_result_checkbox = QCheckBox("Show Fit Result")
        self.show_fit_result_checkbox.setChecked(False)
        self.show_fit_result_checkbox.stateChanged.connect(self._toggle_fit_result_visibility)
        extract_layout.addWidget(self.show_fit_result_checkbox)

        extract_layout.addStretch(1)

        self.extract_button = QPushButton("Extract")
        self.extract_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.extract_button.setFixedWidth(100)
        self.extract_button.clicked.connect(self.extract_spectrum)
        extract_layout.addWidget(self.extract_button)

        layout.addLayout(extract_layout)

        self.fit_status_label = QPlainTextEdit()
        self.fit_status_label.setReadOnly(True)
        self.fit_status_label.setFixedHeight(96)
        self.fit_status_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.fit_status_label.setStyleSheet(
            "QPlainTextEdit { "
            "color: palette(text); "
            "background-color: palette(base); "
            "selection-background-color: palette(highlight); "
            "selection-color: palette(highlighted-text); "
            "}"
        )
        self.fit_status_label.setPlaceholderText("Gaussian fit results appear here.")
        self.fit_status_label.setVisible(False)
        layout.addWidget(self.fit_status_label)

        layout.addWidget(self.toolbar)
    
        # Auto checkbox
        self.auto_checkbox = QCheckBox("Auto Y-axis")
        self.auto_checkbox.setChecked(True)
        self.auto_checkbox.stateChanged.connect(self.toggle_auto_y_axis)
        range_layout.addWidget(self.auto_checkbox, 0, 9)
        range_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum), 0, 10)
        
        self.y_min_input.setEnabled(False)
        self.y_max_input.setEnabled(False)
        self._toggle_fit_result_visibility()
    
        self.x_min_input.returnPressed.connect(self.set_axis_range)
        self.x_max_input.returnPressed.connect(self.set_axis_range)
        self.y_min_input.returnPressed.connect(self.set_axis_range)
        self.y_max_input.returnPressed.connect(self.set_axis_range)
    
        self.ax.axhline(y=0, color='gray', linewidth=0.5)
        self.line, = self.ax.step([], [], where='mid', color='blue', linewidth=1.2)
        self.line.set_drawstyle("steps-mid")
        self.line.set_antialiased(False)
        self.line.set_snap(True)
        self.line.set_solid_capstyle("butt")
        self.line.set_solid_joinstyle("miter")
        try:
            path = self.line.get_path()
            path.should_simplify = False
            path.simplify_threshold = 0.0
        except Exception:
            pass
        self.cursor_line = self.ax.axvline(x=0, color='cyan', linestyle='-', linewidth=0.75)
        self.fit_line, = self.ax.plot([], [], color='crimson', linestyle='-', linewidth=1.9, zorder=20)
        self.fit_line.set_antialiased(True)
        self.fit_line.set_snap(False)
        self.fit_line.set_solid_capstyle("round")
        self.fit_line.set_solid_joinstyle("round")
        self.fit_line.set_visible(False)

    
        #Set label
        self.ax.set_xlabel(f"{self.fits_viewer.displaymap.third_axis_label}")
        self.ax.set_ylabel(f"Intensity [{self.fits_viewer.bunit}]", labelpad=10)
    
        SpecWindow.is_on = True

        self.setWindowTitle(f'Spec Panel: {self.fits_viewer.filename}')


        ### x ticks ###
        x_coord_helper = self.ax.coords[2]
        x_coord_helper.set_minor_frequency(5)
        x_coord_helper.display_minor_ticks(True)
        x_coord_helper.tick_params(
            axis='x', 
            width=1.,
            length=5,
            direction='in', 
            which='major'
        )
        x_coord_helper.tick_params(
            length=3.,
            which='minor'
        )
        
        ### y ticks ###
        self.ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
        self.ax.yaxis.set_tick_params(
            width=1.,
            length=5,
            direction='in',
            which='major',
            color='black'
        )
    
        self.ax.yaxis.set_tick_params(
            length=3,
            width=1.0,
            direction='in',
            which='minor',
            color='black'
        )
        
        # Use initial position from viewer/header if available or default to 0
        # This will be updated immediately by the caller or on show


        if self.fits_viewer.region_mode_enabled:
            selected_region = self.fits_viewer.region_manager.selected_region
            if selected_region:
                # Use QTimer.singleShot to ensure the window is fully loaded before updating
                QTimer.singleShot(0, lambda: self.on_region_changed(selected_region))

    def showEvent(self, event):
        super().showEvent(event)
        if self._initialized_on_show:
            return
        self._initialized_on_show = True
        # Defer one tick so the canvas is fully realized before plotting.
        QTimer.singleShot(0, self._initialize_from_viewer_state)

    def _initialize_from_viewer_state(self):
        """Initialize the spectrum from the current viewer state."""
        if self._suppress_next_viewer_initialize:
            self._suppress_next_viewer_initialize = False
            return
        try:
            coord_state = self.fits_viewer.get_coord_state()
            has_click = False
            if coord_state is not None:
                has_click = any(coord_state.get_clicked(p) for p in ("xy", "xz", "zy"))

            try:
                shared_x = int(self.fits_viewer._get_shared_xpix())
            except Exception:
                shared_x = int(round(getattr(self, "x", 0)))
            try:
                shared_y = int(self.fits_viewer._get_shared_ypix())
            except Exception:
                shared_y = int(round(getattr(self, "y", 0)))
            try:
                shared_z = int(self.fits_viewer._get_shared_zpix())
            except Exception:
                shared_z = int(round(getattr(self, "z", self.fits_viewer.current_channel_index())))

            if has_click:
                x = shared_x
                y = shared_y
                z = shared_z
            else:
                # Keep shared cursor position for workspace restores even when click flags are clear.
                x = shared_x
                y = shared_y
                z = int(self.fits_viewer.current_channel_index())

            # Clamp to cube bounds to avoid out-of-range empty spectra.
            cube = self.fits_viewer.data
            if cube.ndim == 4:
                cube = cube[0]
            z_max, y_max, x_max = cube.shape[0] - 1, cube.shape[1] - 1, cube.shape[2] - 1
            x = max(0, min(x, x_max))
            y = max(0, min(y, y_max))
            z = max(0, min(z, z_max))

            try:
                self.fits_viewer._update_shared_pix(x, y, z)
            except Exception:
                pass

            # Ensure the very first draw is not skipped by the 0.01s throttle.
            self.last_update_time = 0
            self.update_spectrum(x, y, z)
        except Exception:
            # Last-resort fallback
            self.last_update_time = 0
            self.update_spectrum(0, 0, 0)

    # CODE ADDED: Mouse event handlers for dragging the cursor line
    def on_press(self, event):
        """ Handles mouse button press event. """
        # Ignore clicks outside the axes or when a toolbar tool is active
        if event.inaxes != self.ax or self.toolbar.mode != '':
            return
        
        # Check if the click is close to the cursor line
        contains, _ = self.cursor_line.contains(event)
        if contains:
            self.is_dragging = True
            return

    def on_motion(self, event):
        """ Handles mouse motion event. """
        # If dragging is active, handle the line movement
        if self.is_dragging and event.inaxes == self.ax:
            new_channel = int(round(event.xdata))

            # Ensure the new channel is within the valid range
            n_channels = self.fits_viewer.header[f'NAXIS{self.spec_axis + 1}']
            if 0 <= new_channel < n_channels:
                # Update the main window's slider, which triggers all other updates
                self.channel_changed.emit(new_channel)
            return

        # If not dragging, handle the cursor change on hover
        is_over_line = False
        if event.inaxes == self.ax and self.toolbar.mode == '':
            contains, _ = self.cursor_line.contains(event)
            if contains:
                is_over_line = True

        if is_over_line:
            self.canvas.setCursor(Qt.CursorShape.SizeHorCursor)
        else:
            self.canvas.setCursor(Qt.CursorShape.ArrowCursor)

    def on_release(self, event):
        """ Handles mouse button release event. """
        self.is_dragging = False

    def toggle_auto_y_axis(self):
        self.auto_y_axis = self.auto_checkbox.isChecked()
        self.y_min_input.setEnabled(not self.auto_y_axis)
        self.y_max_input.setEnabled(not self.auto_y_axis)

    def _toggle_fit_result_visibility(self, _state=None):
        visible = bool(self.show_fit_result_checkbox.isChecked())
        self.fit_status_label.setVisible(visible)

    def _spectrum_source_key(self):
        if self.active_region is not None:
            region_id = getattr(self.active_region, "region_id", None)
            return ("region", region_id if region_id is not None else id(self.active_region))

        app_state = getattr(self.fits_viewer, "app_state", None)
        current_s = int(getattr(app_state, "current_s", 0)) if app_state is not None else 0
        return ("pixel", int(self.x), int(self.y), current_s)

    def _spectrum_fingerprint(self, spectrum):
        arr = np.asarray(spectrum, dtype=float).reshape(-1)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return (int(arr.size), None, None)
        mean_val = float(np.nanmean(finite))
        std_val = float(np.nanstd(finite))
        return (int(arr.size), round(mean_val, 6), round(std_val, 6))

    def _build_fit_signature(self, spectrum):
        return (self._spectrum_source_key(), self._spectrum_fingerprint(spectrum))

    def _clear_fit_artists(self, *, clear_result=True, reset_status=True, redraw=False):
        if hasattr(self, "fit_line") and self.fit_line is not None:
            try:
                self.fit_line.set_data([], [])
                self.fit_line.set_visible(False)
            except Exception:
                pass

        for line in list(self.fit_component_lines):
            try:
                line.remove()
            except Exception:
                try:
                    line.set_visible(False)
                except Exception:
                    pass
        self.fit_component_lines = []
        if self.fit_legend is not None:
            try:
                self.fit_legend.remove()
            except Exception:
                pass
            self.fit_legend = None

        if clear_result:
            self.fit_result = None
            self.fit_signature = None
            self.fit_x_full = None
            self.fit_view_xlim = None
            if self._fit_resample_timer.isActive():
                self._fit_resample_timer.stop()
        if reset_status and hasattr(self, "fit_status_label"):
            self.fit_status_label.setPlainText("")

        if redraw:
            self.canvas.draw_idle()

    def clear_fit(self):
        self._clear_fit_artists(clear_result=True, reset_status=True, redraw=True)

    def _residual_rms(self, fit_result):
        residual = np.asarray(getattr(fit_result, "residual", []), dtype=float).reshape(-1)
        finite_residual = residual[np.isfinite(residual)]
        if finite_residual.size <= 0:
            return None
        return float(np.sqrt(np.mean(finite_residual ** 2)))

    def _fit_sample_range(self):
        if self.spectrum is None:
            raise ValueError("No spectrum data to fit.")

        spectrum = np.asarray(self.spectrum, dtype=float).reshape(-1)
        x_full = np.arange(spectrum.size, dtype=float)

        x0, x1 = self.ax.get_xlim()
        x_min = min(float(x0), float(x1))
        x_max = max(float(x0), float(x1))

        finite = np.isfinite(spectrum)
        in_range = finite & (x_full >= x_min) & (x_full <= x_max)
        if np.count_nonzero(in_range) < 8:
            in_range = finite

        x_fit = x_full[in_range]
        y_fit = spectrum[in_range]
        if x_fit.size < 8:
            raise ValueError("Not enough valid points in the selected range for fitting.")

        return x_fit, y_fit, x_full

    def _channel_to_world_value(self, channel):
        velocity = np.asarray(self.velocity_values, dtype=float).reshape(-1)
        if velocity.size == 0 or not np.all(np.isfinite(velocity)):
            return None
        channels = np.arange(velocity.size, dtype=float)
        ch = float(np.clip(channel, channels[0], channels[-1]))
        return float(np.interp(ch, channels, velocity))

    def _format_fit_status(self, fit_result):
        def _fmt_unc(value, err, value_fmt=".4g", err_fmt=".2g", show_percent=True):
            v = float(value)
            e = float(err) if err is not None else float("nan")
            if not np.isfinite(e) or e < 0.0:
                return format(v, value_fmt)
            value_text = format(v, value_fmt)
            err_text = format(e, err_fmt)
            if show_percent:
                denom = abs(v)
                if denom > 0:
                    pct = 100.0 * e / denom
                    if np.isfinite(pct):
                        return f"{value_text} +/-{err_text} ({pct:.2g}%)"
            return f"{value_text} +/-{err_text}"

        baseline_mode = "fit" if fit_result.fit_baseline else f"fixed({fit_result.baseline_fixed:.4g})"
        residual_rms = self._residual_rms(fit_result)
        rms_text = f"{residual_rms:.3g}" if residual_rms is not None else "nan"
        lines = [
            f"Gaussian fit: {fit_result.n_components} component(s), "
            f"baseline={baseline_mode}, residual_rms={rms_text}"
        ]

        velocity = np.asarray(self.velocity_values, dtype=float).reshape(-1)
        dv = None
        if velocity.size > 1 and np.all(np.isfinite(velocity)):
            dv_raw = float(np.nanmedian(np.diff(velocity)))
            if np.isfinite(dv_raw):
                dv = abs(dv_raw)

        for idx, comp in enumerate(fit_result.components, start=1):
            center_world = self._channel_to_world_value(comp.center)
            if center_world is not None and dv is not None:
                sigma_world = comp.sigma * dv
                fwhm_world = comp.fwhm * dv
                center_err_world = comp.center_err * dv
                sigma_err_world = comp.sigma_err * dv
                fwhm_err_world = comp.fwhm_err * dv
                lines.append(
                    f"#{idx}: A={_fmt_unc(comp.amplitude, comp.amplitude_err)}, "
                    f"mu={_fmt_unc(center_world, center_err_world, show_percent=False)}, "
                    f"sigma={_fmt_unc(sigma_world, sigma_err_world, value_fmt='.3g', err_fmt='.2g')}, "
                    f"FWHM={_fmt_unc(fwhm_world, fwhm_err_world, value_fmt='.3g', err_fmt='.2g')}"
                )
            else:
                lines.append(
                    f"#{idx}: A={_fmt_unc(comp.amplitude, comp.amplitude_err)}, "
                    f"mu={_fmt_unc(comp.center, comp.center_err, value_fmt='.3f', err_fmt='.2g', show_percent=False)} ch, "
                    f"sigma={_fmt_unc(comp.sigma, comp.sigma_err, value_fmt='.3f', err_fmt='.2g')} ch, "
                    f"FWHM={_fmt_unc(comp.fwhm, comp.fwhm_err, value_fmt='.3f', err_fmt='.2g')} ch"
                )

        if fit_result.fit_baseline:
            lines.append(
                f"baseline_result={_fmt_unc(fit_result.baseline, fit_result.baseline_err)}"
            )
        return "\n".join(lines)

    def _fit_curve_samples(self, x_min, x_max):
        span = max(0.0, float(x_max) - float(x_min))
        canvas_width = 0
        try:
            canvas_width = int(max(0, self.canvas.width()))
        except Exception:
            canvas_width = 0
        by_channel = int(span * 120 + 1)
        by_pixel = int(max(1, canvas_width) * 12)
        # Keep curves smooth at high zoom while avoiding extreme memory usage.
        return int(np.clip(max(by_channel, by_pixel), 6000, 800000))

    def _normalized_view_xlim(self):
        x0, x1 = self.ax.get_xlim()
        lo = min(float(x0), float(x1))
        hi = max(float(x0), float(x1))
        if not np.isfinite(lo) or not np.isfinite(hi):
            return None
        return (lo, hi)

    def _fit_draw_domain(self, x_full):
        data_min = float(np.nanmin(x_full))
        data_max = float(np.nanmax(x_full))
        if not np.isfinite(data_min) or not np.isfinite(data_max) or data_max <= data_min:
            return data_min, data_max

        view_xlim = self._normalized_view_xlim()
        if view_xlim is None:
            return data_min, data_max

        lo, hi = view_xlim
        lo = max(lo, data_min)
        hi = min(hi, data_max)
        if hi <= lo:
            return data_min, data_max

        # Keep a small margin beyond the visible area to avoid edge clipping artifacts.
        pad = max(1.0, 0.02 * (hi - lo))
        return max(data_min, lo - pad), min(data_max, hi + pad)

    def _xlim_close(self, lhs, rhs, tol=1e-6):
        if lhs is None or rhs is None:
            return False
        return (
            abs(float(lhs[0]) - float(rhs[0])) <= tol
            and abs(float(lhs[1]) - float(rhs[1])) <= tol
        )

    def _on_fit_xlim_changed(self, _axes):
        if self.fit_result is None or self.fit_x_full is None:
            return
        current_xlim = self._normalized_view_xlim()
        if self._xlim_close(current_xlim, self.fit_view_xlim):
            return
        if self._fit_resample_in_progress:
            return
        if self._fit_resample_timer.isActive():
            self._fit_resample_timer.stop()
        self._fit_resample_timer.start()

    def _resample_fit_for_view(self):
        if self.fit_result is None or self.fit_x_full is None:
            return
        if self._fit_resample_in_progress:
            return
        try:
            self._fit_resample_in_progress = True
            self._draw_fit_result(self.fit_result, self.fit_x_full)
        finally:
            self._fit_resample_in_progress = False

    def _draw_fit_result(self, fit_result, x_full):
        self._clear_fit_artists(clear_result=False, reset_status=False, redraw=False)

        self.fit_view_xlim = self._normalized_view_xlim()

        x_min, x_max = self._fit_draw_domain(x_full)
        n_dense = self._fit_curve_samples(x_min, x_max)
        x_dense = np.linspace(x_min, x_max, n_dense)

        total_model = np.full_like(x_dense, float(fit_result.baseline), dtype=float)
        # Keep component colors distinct from the red total-fit line.
        component_palette = (
            "#ff7f0e",  # orange
            "#2ca02c",  # green
            "#17becf",  # cyan
            "#9467bd",  # purple
            "#1f77b4",  # blue
            "#8c564b",  # brown
            "#7f7f7f",  # gray
            "#bcbd22",  # olive
            "#4c78a8",  # steel blue
            "#72b7b2",  # muted cyan
        )

        for idx, comp in enumerate(fit_result.components):
            sigma = max(float(comp.sigma), 1e-8)
            gaussian = comp.amplitude * np.exp(-0.5 * ((x_dense - comp.center) / sigma) ** 2)
            total_model += gaussian
            color = component_palette[idx % len(component_palette)]
            comp_line, = self.ax.plot(
                x_dense,
                fit_result.baseline + gaussian,
                linestyle='-',
                linewidth=1.0,
                color=color,
                alpha=0.9,
                zorder=12,
            )
            comp_line.set_antialiased(True)
            comp_line.set_snap(False)
            comp_line.set_solid_capstyle("round")
            comp_line.set_solid_joinstyle("round")
            try:
                path = comp_line.get_path()
                path.should_simplify = False
                path.simplify_threshold = 0.0
            except Exception:
                pass
            self.fit_component_lines.append(comp_line)

        n_components = len(self.fit_component_lines)
        if n_components <= 1:
            self.fit_line.set_data([], [])
            self.fit_line.set_visible(False)
            legend_handles = list(self.fit_component_lines)
            legend_labels = [f"#{i + 1}" for i in range(n_components)]
        else:
            self.fit_line.set_data(x_dense, total_model)
            self.fit_line.set_zorder(20)
            self.fit_line.set_visible(True)
            try:
                path = self.fit_line.get_path()
                path.should_simplify = False
                path.simplify_threshold = 0.0
            except Exception:
                pass
            legend_handles = [self.fit_line] + list(self.fit_component_lines)
            legend_labels = ["Total fit"] + [f"#{i + 1}" for i in range(n_components)]

        self.fit_legend = self.ax.legend(
            legend_handles,
            legend_labels,
            loc="upper right",
            fontsize=8,
            framealpha=0.9,
            borderpad=0.4,
            labelspacing=0.3,
            handlelength=1.8,
            title="Gaussians",
            title_fontsize=8,
        )
        self.fit_status_label.setPlainText(self._format_fit_status(fit_result))
        self.canvas.draw_idle()

    def fit_gaussian_components(self, silent=False):
        if self.spectrum is None:
            if not silent:
                QMessageBox.warning(self, "No Data", "No spectrum data to fit.")
            return False

        try:
            x_fit, y_fit, x_full = self._fit_sample_range()
        except ValueError as exc:
            if not silent:
                QMessageBox.warning(self, "Fit Error", str(exc))
            return False

        n_value = int(self.fit_n_spinbox.value())
        fit_baseline = bool(self.fit_baseline_checkbox.isChecked())

        try:
            fit_result = fit_gaussian_spectrum(
                x_fit,
                y_fit,
                n_components=n_value,
                auto_components=False,
                fit_baseline=fit_baseline,
                baseline_fixed=0.0,
                allow_negative=False,
            )
        except Exception as exc:
            if not silent:
                QMessageBox.critical(self, "Fit Error", f"Gaussian fit failed.\nError: {exc}")
            return False

        self.fit_result = fit_result
        self.fit_signature = self._build_fit_signature(self.spectrum)
        self.fit_x_full = np.asarray(x_full, dtype=float)
        self._draw_fit_result(fit_result, x_full)
        return True

    def set_axis_range(self):
        try:
            x_min = float(self.x_min_input.text())
            x_max = float(self.x_max_input.text())
            if  self.fits_viewer.data.ndim == 3:
                v_min = self.converter.world_to_pix(self.world_x, self.world_y, x_min)[2]
                v_max = self.converter.world_to_pix(self.world_x, self.world_y, x_max)[2]
    
            elif  self.fits_viewer.data.ndim == 4:
                v_min = self.converter.world_to_pix(self.world_x, self.world_y, x_min, 0)[2]
                v_max = self.converter.world_to_pix(self.world_x, self.world_y, x_max, 0)[2]
            
            self.ax.set_xlim(v_min, v_max)
            
            y_min = float(self.y_min_input.text())
            y_max = float(self.y_max_input.text())
            self.ax.set_ylim(y_min, y_max)
            
            self.canvas.draw_idle()
        except ValueError:
            QMessageBox.warning(self, 'Invalid Input', ' Please enter numeric values.')
            return

    def update_position(self, x, y, z):
        """Public slot to update position from external source (e.g. FITSViewer)"""
        self.update_spectrum(x, y, z)

    def refresh_coordinate_display(self):
        """Refresh title/labels using current cursor with latest display-frame settings."""
        self._sync_coordinate_context()
        self.last_update_time = 0
        self.update_spectrum(self.x, self.y, self.z)

    def update_spectrum(self, x, y, z):
        current_time = time.time()
        if current_time - self.last_update_time < 0.01:
            return
        self.last_update_time = current_time

        if not self.fits_viewer.region_mode_enabled:
            self.active_region = None

        title = ""
        if self.active_region:
            self.spectrum, title = self._calculate_average_spectrum(self.active_region)

        else:
            x_pix = int(round(x))
            y_pix = int(round(y))
            self.x = x_pix
            self.y = y_pix
            self.z = z
            self.spectrum = None

            # Check bounds and use usecase
            data = self.fits_viewer.data
            # Determine dimensions
            if data.ndim == 4:
                max_y, max_x = data.shape[2], data.shape[3]
            else:
                max_y, max_x = data.shape[1], data.shape[2]

            if (0 <= x_pix < max_x) and (0 <= y_pix < max_y):
                # Fast path for drag updates: direct cube slice matches dist behavior.
                try:
                    if data.ndim == 4:
                        s_idx = 0
                        app_state = getattr(self.fits_viewer, 'app_state', None)
                        if app_state is not None:
                            try:
                                s_idx = int(getattr(app_state, 'current_s', 0))
                            except Exception:
                                s_idx = 0
                        s_idx = max(0, min(s_idx, data.shape[0] - 1))
                        data_cube = data[s_idx]
                    else:
                        data_cube = data
                    self.spectrum = data_cube[:, y_pix, x_pix]
                except Exception:
                    self.spectrum = None

                # Fallback for edge cases where direct slicing fails.
                if self.spectrum is None and hasattr(self.fits_viewer, 'app_state') and self.fits_viewer.app_state:
                    try:
                        self.spectrum = get_spectrum(self.fits_viewer.app_state, x=x_pix, y=y_pix)
                    except Exception:
                        self.spectrum = None


            try:
                if self.fits_viewer.data.ndim == 3:
                    world_native = self.wcs.wcs_pix2world(
                        [[float(x_pix), float(y_pix), float(z)]],
                        0,
                    )[0]
                else:  # ndim == 4
                    world_native = self.wcs.wcs_pix2world(
                        [[float(x_pix), float(y_pix), float(z), 0.0]],
                        0,
                    )[0]

                # Keep native coordinates for world->pixel conversion in axis range handling.
                self.world_x = float(world_native[0])
                self.world_y = float(world_native[1])

                world_x_str, world_y_str = self._format_title_world_coordinates(world_native)
                title = f"Spectrum at ({world_x_str}, {world_y_str})"
            except Exception:
                title = "Spectrum"

        if self.spectrum is None:
            self._clear_fit_artists(clear_result=True, reset_status=True, redraw=False)
        else:
            if self.fit_signature is not None:
                current_fit_signature = self._build_fit_signature(self.spectrum)
                if current_fit_signature != self.fit_signature:
                    self._clear_fit_artists(clear_result=True, reset_status=True, redraw=False)

        self.ax.set_title(title, loc='left')
        if self.spectrum is not None:
            self.update_plot(self.spectrum, x, y, z)
            
    def update_plot(self, spectrum, x, y, z):
        y_data = np.asarray(spectrum, dtype=float).reshape(-1)
        y_data = np.ma.masked_invalid(y_data)
        self.line.set_data(np.arange(len(y_data), dtype=float), y_data)
        self.line.set_drawstyle("steps-mid")
        self.line.set_antialiased(False)
        self.line.set_snap(True)
        self.line.set_solid_capstyle("butt")
        self.line.set_solid_joinstyle("miter")
        try:
            path = self.line.get_path()
            path.should_simplify = False
            path.simplify_threshold = 0.0
        except Exception:
            pass
        
        self.cursor_line.set_visible(True)
        self.cursor_line.set_xdata([z, z])
        try:
            x_min = float(self.x_min_input.text())
            x_max = float(self.x_max_input.text())
        except ValueError:
            QMessageBox.warning(self, 'Invalid Input', 'Incorrect value entered for the X-axis range.')
            return
        if  self.fits_viewer.data.ndim == 3:
            v_min = self.converter.world_to_pix(self.world_x, self.world_y, x_min)[2]
            v_max = self.converter.world_to_pix(self.world_x, self.world_y, x_max)[2]
            
        elif  self.fits_viewer.data.ndim == 4:
            v_min = self.converter.world_to_pix(self.world_x, self.world_y, x_min, 0)[2]
            v_max = self.converter.world_to_pix(self.world_x, self.world_y, x_max, 0)[2]
        
        self.ax.set_xlim(v_min, v_max)
        if self.auto_y_axis:
            try:
                x_min_pix, x_max_pix = self.ax.get_xlim()
                x_start = max(int(np.floor(x_min_pix)), 0)
                x_end = min(int(np.ceil(x_max_pix)), len(spectrum))
                if x_start >= x_end:
                     spectrum_in_range = spectrum
                else:
                     spectrum_in_range = spectrum[x_start:x_end]

                if spectrum_in_range.size > 0 and not np.all(np.isnan(spectrum_in_range)):
                    y_min, y_max = np.nanmin(spectrum_in_range), np.nanmax(spectrum_in_range)
                    y_range = y_max - y_min
                    if y_range == 0:
                        y_range = 1e-6
                    self.ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
                    self.update_range_textboxes()
            except (ValueError, IndexError):
                pass
        else:
            self.ax.set_ylim(float(self.y_min_input.text()), float(self.y_max_input.text()))
        
        self.canvas.draw_idle()

    def update_range_textboxes(self):
        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()
        if self.fits_viewer.data.ndim == 3:
            v_min = self.converter.pix_to_world(self.x, self.y, x_min)[2]
            v_max = self.converter.pix_to_world(self.x, self.y, x_max)[2]
        elif self.fits_viewer.data.ndim == 4:
            v_min = self.converter.pix_to_world(self.x, self.y, x_min, 0)[2]
            v_max = self.converter.pix_to_world(self.x, self.y, x_max, 0)[2]

        self.x_min_input.setText(f"{float(v_min):.4g}")
        self.x_max_input.setText(f"{float(v_max):.4g}")
        self.y_min_input.setText(f"{y_min:.4g}")
        self.y_max_input.setText(f"{y_max:.4g}")


    def extract_spectrum(self):
        """
        Extracts the current spectrum data to a text file using core usecase.
        """
        if self.spectrum is None:
            QMessageBox.warning(self, "No Data", "No spectrum data to extract.")
            return

        base_filename = os.path.basename(self.fits_viewer.filename)
        default_name = os.path.splitext(base_filename)[0] + ".spec.txt"

        desktop_dir = os.path.join(os.path.expanduser("~"), "Desktop")
        initial_path = os.path.join(desktop_dir, default_name)

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Spectrum", initial_path, "Text Files (*.txt);;All Files (*)"
        )

        if not path:
            return

        # Build Metadata
        metadata = {'filename': base_filename}
        
        if self.active_region:
            region = self.active_region
            state = region.get_state()
            shape = region.__class__.__name__.replace('Region', '')
            metadata['region_type'] = shape
            
            params = {}
            if state.get('label'):
                params['Region Label'] = state['label']

            center_pix = state.get('center')
            if center_pix:
                try:
                    if self.fits_viewer.data.ndim == 3:
                        world = self.converter.pix_to_world(center_pix[0], center_pix[1], 0)
                    else:
                        world = self.converter.pix_to_world(center_pix[0], center_pix[1], 0, 0)
                    params['Center World'] = f"({world[0]}, {world[1]})"
                except Exception:
                    params['Center Pix'] = f"({center_pix[0]:.2f}, {center_pix[1]:.2f})"

            editor = self.fits_viewer.region_manager.region_editors.get(region)
            if isinstance(region, CircleRegion):
                unit = editor._field_units.get('radius', 'pix') if editor else 'pix'
                value = editor.radius_spin.value() if editor else state['radius']
                params[f'Radius [{unit}]'] = f"{value:.3f}"
            elif isinstance(region, (RectangleRegion, EllipseRegion)):
                unit = editor._field_units.get('width', 'pix') if editor else 'pix'
                width = editor.width_spin.value() if editor else state['width']
                height = editor.height_spin.value() if editor else state['height']
                params[f'Width [{unit}]'] = f"{width:.3f}"
                params[f'Height [{unit}]'] = f"{height:.3f}"
                if 'angle' in state:
                    params['Angle [deg]'] = f"{state['angle']:.2f}"
            
            metadata['region_params'] = params
        
        else:
            # Single pixel mode
            try:
                if self.fits_viewer.data.ndim == 3:
                     world = self.converter.pix_to_world(self.x, self.y, 0)
                else: 
                     world = self.converter.pix_to_world(self.x, self.y, 0, 0)
                metadata['world_coord'] = f"({world[0]}, {world[1]})"
            except Exception: pass
            
            metadata['pixel_coord'] = f"({self.x}, {self.y})"

        fit_is_current = False
        if self.fit_result is not None and self.fit_signature is not None:
            try:
                fit_is_current = (self._build_fit_signature(self.spectrum) == self.fit_signature)
            except Exception:
                fit_is_current = False
        if fit_is_current:
            fit_text = self._format_fit_status(self.fit_result)
            fit_lines = [line.strip() for line in str(fit_text).splitlines() if line.strip()]
            if fit_lines:
                metadata['fit_info_lines'] = fit_lines

        try:
            export_spectrum(
                spectrum_data=self.spectrum,
                velocity_values=self.velocity_values,
                output_path=path,
                xlabel=self.ax.get_xlabel(),
                ylabel=self.ax.get_ylabel(),
                metadata=metadata
            )
            QMessageBox.information(self, "Success", f"Spectrum data successfully saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save spectrum data.\nError: {e}")

    def on_region_changed(self, region):
        """
        Slot for the signal from RegionManager. Updates the active region
        and refreshes the spectrum plot.
        """
        self.active_region = region
        self.active_region = region
        # Trigger an update using the current position.
        self.update_spectrum(self.x, self.y, self.z)

    def _region_center_pixel(self, region):
        if region is None or not hasattr(region, "get_state"):
            return None
        try:
            state = region.get_state() or {}
        except Exception:
            return None
        if not isinstance(state, dict):
            return None
        if "center" in state:
            try:
                center = state.get("center")
                return float(center[0]), float(center[1])
            except Exception:
                return None
        if "xy" in state and "width" in state and "height" in state:
            try:
                x0, y0 = state.get("xy")
                width = float(state.get("width", 0.0))
                height = float(state.get("height", 0.0))
                return float(x0) + width / 2.0, float(y0) + height / 2.0
            except Exception:
                return None
        return None

    def _region_kind(self, region):
        name = getattr(region, "__class__", type(region)).__name__
        return str(name).replace("Region", "").lower()

    def _resolve_workspace_region(self, region_state):
        if not isinstance(region_state, dict):
            return None
        manager = getattr(self.fits_viewer, "region_manager", None)
        if manager is None:
            return None
        regions = list(getattr(manager, "regions", []) or [])
        if not regions:
            return None

        region_id = region_state.get("id")
        if region_id is not None:
            for region in regions:
                candidate = getattr(region, "region_id", None)
                if candidate is None:
                    continue
                try:
                    if int(candidate) == int(region_id):
                        return region
                except Exception:
                    if str(candidate) == str(region_id):
                        return region

        label = str(region_state.get("label") or "").strip()
        kind = str(region_state.get("kind") or "").strip().lower()
        if label:
            for region in regions:
                candidate_label = str(getattr(region, "label_text", "") or "").strip()
                if candidate_label != label:
                    continue
                if kind and self._region_kind(region) != kind:
                    continue
                return region

        center_world = region_state.get("center_world")
        if isinstance(center_world, (list, tuple)) and len(center_world) >= 2:
            try:
                target = self.converter.world_to_pix(center_world[0], center_world[1])
                tx, ty = float(target[0]), float(target[1])
            except Exception:
                tx = ty = None
            if tx is not None and ty is not None:
                best_region = None
                best_dist = None
                for region in regions:
                    if kind and self._region_kind(region) != kind:
                        continue
                    center = self._region_center_pixel(region)
                    if center is None:
                        continue
                    dist = (float(center[0]) - tx) ** 2 + (float(center[1]) - ty) ** 2
                    if best_dist is None or dist < best_dist:
                        best_dist = dist
                        best_region = region
                if best_region is not None:
                    return best_region

        return None

    def export_workspace_state(self):
        fit_active = bool(self.fit_result is not None and self.fit_signature is not None)
        fit_summary = None
        if fit_active and self.fit_result is not None:
            fit_summary = {
                "n_components": int(self.fit_result.n_components),
                "fit_baseline": bool(self.fit_result.fit_baseline),
                "baseline": float(self.fit_result.baseline),
                "baseline_fixed": float(self.fit_result.baseline_fixed),
                "residual_rms": self._residual_rms(self.fit_result),
            }
        state = {
            "schema": 1,
            "auto_y_axis": bool(self.auto_checkbox.isChecked()),
            "axis_ranges": {
                "x_min": str(self.x_min_input.text() or "").strip(),
                "x_max": str(self.x_max_input.text() or "").strip(),
                "y_min": str(self.y_min_input.text() or "").strip(),
                "y_max": str(self.y_max_input.text() or "").strip(),
            },
            "pixel": {
                "x": int(round(float(self.x))),
                "y": int(round(float(self.y))),
                "z": int(round(float(self.z))),
            },
            "world": None,
            "region_mode_enabled": bool(getattr(self.fits_viewer, "region_mode_enabled", False)),
            "active_region": None,
            "fit_state": {
                "n_components": int(self.fit_n_spinbox.value()),
                "fit_baseline": bool(self.fit_baseline_checkbox.isChecked()),
                "show_fit_result": bool(self.show_fit_result_checkbox.isChecked()),
                "fit_active": fit_active,
                "fit_summary": fit_summary,
            },
        }

        try:
            if self.fits_viewer.data.ndim == 3:
                world = self.converter.pix_to_world(self.x, self.y, self.z)
            else:
                world = self.converter.pix_to_world(self.x, self.y, self.z, 0)
            if isinstance(world, (list, tuple)) and len(world) >= 2:
                state["world"] = {
                    "x": str(world[0]),
                    "y": str(world[1]),
                    "z": str(world[2]) if len(world) >= 3 else "",
                }
        except Exception:
            pass

        region = self.active_region
        if region is not None:
            region_payload = {
                "id": getattr(region, "region_id", None),
                "label": str(getattr(region, "label_text", "") or "").strip(),
                "kind": self._region_kind(region),
                "center_world": None,
            }
            center = self._region_center_pixel(region)
            if center is not None:
                region_payload["center_pixel"] = [float(center[0]), float(center[1])]
                try:
                    center_world = self.converter.pix_to_world(float(center[0]), float(center[1]), 0)
                    if isinstance(center_world, (list, tuple)) and len(center_world) >= 2:
                        region_payload["center_world"] = [str(center_world[0]), str(center_world[1])]
                except Exception:
                    pass
            state["active_region"] = region_payload

        return state

    def restore_workspace_state(self, state):
        if not isinstance(state, dict):
            return False

        data = self.fits_viewer.data
        if data.ndim == 4:
            cube = data[0]
        else:
            cube = data
        if cube.ndim < 3:
            return False

        max_z = max(0, cube.shape[0] - 1)
        max_y = max(0, cube.shape[1] - 1)
        max_x = max(0, cube.shape[2] - 1)

        target_x = None
        target_y = None
        target_z = None

        world = state.get("world")
        if isinstance(world, dict):
            wx = world.get("x")
            wy = world.get("y")
            wz = world.get("z")
            if wx not in (None, "") and wy not in (None, ""):
                try:
                    if data.ndim == 4:
                        pix = self.converter.world_to_pix(wx, wy, wz if wz not in (None, "") else 0, 0)
                    else:
                        pix = self.converter.world_to_pix(wx, wy, wz if wz not in (None, "") else 0)
                    target_x = float(pix[0])
                    target_y = float(pix[1])
                    if len(pix) >= 3:
                        target_z = float(pix[2])
                except Exception:
                    target_x = None
                    target_y = None
                    target_z = None

        pixel = state.get("pixel")
        if isinstance(pixel, dict):
            if target_x is None and pixel.get("x") is not None:
                try:
                    target_x = float(pixel.get("x"))
                except Exception:
                    pass
            if target_y is None and pixel.get("y") is not None:
                try:
                    target_y = float(pixel.get("y"))
                except Exception:
                    pass
            if target_z is None and pixel.get("z") is not None:
                try:
                    target_z = float(pixel.get("z"))
                except Exception:
                    pass

        if target_x is None:
            target_x = float(self.x)
        if target_y is None:
            target_y = float(self.y)
        if target_z is None:
            target_z = float(self.z)

        x = max(0, min(int(round(target_x)), max_x))
        y = max(0, min(int(round(target_y)), max_y))
        z = max(0, min(int(round(target_z)), max_z))
        # Workspace load can open this panel and queue show-time initialization.
        # Suppress that one-time initializer so restored coordinates are not overwritten.
        self._suppress_next_viewer_initialize = True

        try:
            self.fits_viewer.region_mode_enabled = bool(state.get("region_mode_enabled", False))
        except Exception:
            pass

        restored_region = False
        region_state = state.get("active_region")
        region = self._resolve_workspace_region(region_state)
        manager = getattr(self.fits_viewer, "region_manager", None)
        if region is not None and manager is not None:
            try:
                manager.select_region(region)
                self.active_region = region
                restored_region = True
                self.fits_viewer.region_mode_enabled = True
            except Exception:
                self.active_region = None
        elif not self.fits_viewer.region_mode_enabled:
            self.active_region = None

        try:
            self.channel_changed.emit(int(z))
        except Exception:
            pass
        self.last_update_time = 0
        self.update_spectrum(x, y, z)

        axis_ranges = state.get("axis_ranges")
        if isinstance(axis_ranges, dict):
            self.x_min_input.setText(str(axis_ranges.get("x_min", self.x_min_input.text() or "")))
            self.x_max_input.setText(str(axis_ranges.get("x_max", self.x_max_input.text() or "")))
            self.y_min_input.setText(str(axis_ranges.get("y_min", self.y_min_input.text() or "")))
            self.y_max_input.setText(str(axis_ranges.get("y_max", self.y_max_input.text() or "")))

        auto_y_axis = bool(state.get("auto_y_axis", True))
        self.auto_checkbox.setChecked(auto_y_axis)
        try:
            self.set_axis_range()
        except Exception:
            pass
        if auto_y_axis:
            self.last_update_time = 0
            self.update_spectrum(x, y, z)
        fit_state = state.get("fit_state")
        restored_fit = False
        if isinstance(fit_state, dict):
            try:
                n_components = int(fit_state.get("n_components", self.fit_n_spinbox.value()))
                n_components = max(int(self.fit_n_spinbox.minimum()), min(int(self.fit_n_spinbox.maximum()), n_components))
                self.fit_n_spinbox.setValue(n_components)
            except Exception:
                pass
            try:
                self.fit_baseline_checkbox.setChecked(bool(fit_state.get("fit_baseline", self.fit_baseline_checkbox.isChecked())))
            except Exception:
                pass
            try:
                self.show_fit_result_checkbox.setChecked(bool(fit_state.get("show_fit_result", self.show_fit_result_checkbox.isChecked())))
            except Exception:
                pass

            if bool(fit_state.get("fit_active", False)):
                restored_fit = bool(self.fit_gaussian_components(silent=True))
            else:
                self._clear_fit_artists(clear_result=True, reset_status=True, redraw=False)

        return bool(restored_region or axis_ranges or world or pixel or restored_fit or fit_state)


    def _ui_region_to_spec(self, region):
        """Convert UI region object to RegionSpec for usecases."""
        state = region.get_state()
        if isinstance(region, CircleRegion):
            cx, cy = state['center']
            r = state['radius']
            return RegionSpec(type="circle", center_x=cx, center_y=cy, params={'radius': r})
        elif isinstance(region, RectangleRegion):
            cx, cy = state['center']
            w, h = state['width'], state['height']
            angle = state.get('angle', 0.0)
            return RegionSpec(type="rectangle", center_x=cx, center_y=cy, 
                              params={'width': w, 'height': h, 'angle': angle})
        elif isinstance(region, EllipseRegion):
            cx, cy = state['center']
            w, h = state['width'], state['height']
            angle = state.get('angle', 0.0)
            return RegionSpec(type="ellipse", center_x=cx, center_y=cy, 
                              params={'width': w, 'height': h, 'angle': angle})
        return None



    def _calculate_average_spectrum(self, region):
        """
        Calculates the average spectrum within a region using core usecase.
        """
        state = self._ui_region_to_spec(region)
        if not state:
             return None, "Unsupported region shape"

        if hasattr(self.fits_viewer, 'app_state') and self.fits_viewer.app_state:
            try:
                # Use headless usecase
                velocity, spectrum, unit = get_averaged_spectrum(self.fits_viewer.app_state, state)
                self.velocity_values = velocity
                # self.current_unit = unit # Maybe store this if needed for plotting labels
                
                # --- Generate Title (same as before) ---
                region_label = getattr(region, 'label_text', '').strip()
                title_part1 = f"Average Spectrum ({region_label})" if region_label else f"Average Spectrum (Region {getattr(region, 'region_id', '?')})"
                title_part2 = ""
                center_pixel = self._region_center_pixel(region)
                if center_pixel is not None:
                    try:
                        z_value = float(self.z)
                        if self.fits_viewer.data.ndim == 3:
                            world_native = self.wcs.wcs_pix2world(
                                [[float(center_pixel[0]), float(center_pixel[1]), z_value]],
                                0,
                            )[0]
                        else:
                            world_native = self.wcs.wcs_pix2world(
                                [[float(center_pixel[0]), float(center_pixel[1]), z_value, 0.0]],
                                0,
                            )[0]
                        world_x_str, world_y_str = self._format_title_world_coordinates(world_native)
                        title_part2 = f"around ({world_x_str}, {world_y_str})"
                    except Exception:
                        pass
                title = f"{title_part1}\n{title_part2}"

                return spectrum, title

            except Exception as e:
                return None, f"Error calculating spectrum: {e}"

        else:
             return None, "Application state not initialized"

    def closeEvent(self, event):
        try:
            # Signal disconnection is handled automatically by PySide/PyQt usually,
            # but explicit disconnect is fine if specific cleanup is needed.
            # self.common_instance usage is removed.
            if hasattr(self.fits_viewer, 'region_manager'):
                self.fits_viewer.region_manager.selected_region_changed.disconnect(self.on_region_changed)
        except (TypeError, RuntimeError):
            pass

        SpecWindow.is_on = False
        super().closeEvent(event)
        self.destroyed.emit()



class SpecNavigationToolbar(NavigationToolbar):
    def __init__(self, canvas, parent=None):
        super().__init__(canvas, parent)
        self.parent = parent
        self._zoom_mode = False
        self._pan_mode = False

        self.cid_release = self.canvas.mpl_connect("button_release_event", self.on_release)
        self.cid_click = self.canvas.mpl_connect("button_press_event", self.on_click)

    def on_release(self, event):
        if self._zoom_mode or self._pan_mode:
            self.parent.update_range_textboxes()
        self.parent.on_release(event)

    def on_click(self, event):
        if event.dblclick:
            self.parent.cursor_line.set_visible(False)
            self.parent.canvas.draw_idle()
            current_mode = self.mode
            if current_mode == 'pan/zoom':
                self.pan() 
                self._active = None 
                release_event = mpl.backend_bases.MouseEvent(
                    name='button_release_event', canvas=self.canvas,
                    x=event.x, y=event.y, button=event.button,
                    key=event.key, step=event.step, dblclick=event.dblclick,
                    guiEvent=event.guiEvent
                )
                self.release_pan(release_event)
    
            elif current_mode == 'zoom rect':
                self.zoom()
                self._active = None
                release_event = mpl.backend_bases.MouseEvent(
                    name='button_release_event', canvas=self.canvas,
                    x=event.x, y=event.y, button=event.button,
                    key=event.key, step=event.step, dblclick=event.dblclick,
                    guiEvent=event.guiEvent
                )
                self.release_zoom(release_event)
            else:
                return
            self._update_buttons_checked()
            self.set_message('')

    def zoom(self, *args):
        self._zoom_mode = not self._zoom_mode
        super().zoom(*args)

    def pan(self, *args):
        self._pan_mode = not self._pan_mode
        super().pan(*args)
        
    def home(self, *args):
        super().home(*args)
        if self.parent.spectrum is not None and self.parent.spectrum.size > 0:
            y_min, y_max = np.nanmin(self.parent.spectrum), np.nanmax(self.parent.spectrum)
            y_range = y_max - y_min
            if y_range == 0: y_range = 1e-6
            self.parent.ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
            self.parent.update_range_textboxes()
        self.parent.canvas.draw_idle()

    def save_figure(self):
        current_dir = os.getcwd()
        desktop_dir = os.path.join(os.path.expanduser("~"), "Desktop")
        initial_dir = desktop_dir if os.access(desktop_dir, os.W_OK) else current_dir
        self.default_image_name = self.parent.fits_viewer.filename
        
        if self.default_image_name.endswith(".fits"):
            self.default_image_name = self.default_image_name[:-5]
            
        default_filename = f"{self.default_image_name}.spec.pdf"

        path, _ = QFileDialog.getSaveFileName(
            self.canvas.parent(), "Save Figure",
            os.path.join(initial_dir, default_filename),
            "PDF Files (*.pdf);;EPS Files (*.eps);;SVG Files (*.svg);;PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;All Files (*)"
        )
        if path:
            connections = self.canvas.callbacks.callbacks
            saved_connections = connections.copy()
            self.canvas.callbacks.callbacks = {}
            self.canvas.draw()
            
            try:
                # Use headless usecase
                export_figure(self.canvas.figure, path, dpi=300, transparent=False)
            except Exception as e:
                # Error handling could be improved, but for now simple print or message
                print(f"Error saving figure: {e}")
            
            self.canvas.callbacks.callbacks = saved_connections
            filename = os.path.basename(path) 
            self.show_save_success_message(path, filename)
            
    def show_save_success_message(self, path, filename):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setText(f'"{filename}" was saved successfully at:\n{path}')
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()
