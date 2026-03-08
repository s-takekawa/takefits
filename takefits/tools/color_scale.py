from PySide6.QtWidgets import QWidget, QVBoxLayout, QGridLayout, QComboBox, QLineEdit, QPushButton, QCheckBox, QLabel, QToolButton, QSizePolicy, QMessageBox, QDoubleSpinBox
from PySide6.QtGui import QPalette
from PySide6.QtCore import Qt, QSignalBlocker, QTimer, QSize
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from takefits.core.color import ColorMode
from takefits.core.custom_colormap import CustomColormap, ColorDefinitions
from takefits.logic.data_tools import (
    estimate_array_nbytes,
    fast_nanminmax,
    MEMMAP_THRESHOLD_BYTES,
)


class RegisterColor:
    rainbow_cdict = ColorDefinitions.rainbow()
    cool_cdict = ColorDefinitions.cool()
    rainbow = CustomColormap('Rainbow', rainbow_cdict)
    cool = CustomColormap('Cool', cool_cdict)
    rainbow_r = rainbow.reversed_colormap()
    cool_r = cool.reversed_colormap()
    colormaps.register(rainbow.get_colormap())
    colormaps.register(cool.get_colormap())
    colormaps.register(rainbow_r.get_colormap())
    colormaps.register(cool_r.get_colormap())



class ColorSettingsPanel(QWidget):
    bad_color = None
    filename = None
    
    settings = {
        ColorMode.MAIN: {"min_val": None, "max_val": None, "log_scale": False,
                    "gamma_value": 1.0, "invert": False, "color_pattern": None},
        ColorMode.INTEG: {"min_val": None, "max_val": None, "log_scale": False,
                     "gamma_value": 1.0, "invert": False, "color_pattern": None},
        ColorMode.CHANNEL: {"min_val": None, "max_val": None, "log_scale": False,
                       "gamma_value": 1.0, "invert": False, "color_pattern": None}
    }

    def __init__(self, fits_viewer, subwindows=None, data=None, config=None, color_pattern=None, bad_color=None, filename = None, mode = ColorMode.MAIN):
        super().__init__()
        self.fits_viewer = fits_viewer
        self.subwindows = subwindows
        self.mode = mode
        self._shared_history_record_pending = False
        self._shared_history_record_reason = ""
        self._suppress_shared_history_recording = True
        self.current_settings = ColorSettingsPanel.settings[self.mode]
        
        if config is None: self.config = self.fits_viewer.displaymap.config
        else: self.config = config
        self.dragging_min = False
        self.dragging_max = False
        if data is None: self.data = self.fits_viewer.data
        else: self.data = data
        self._data_nbytes = estimate_array_nbytes(self.data)
        if  color_pattern is None: self.color_pattern = self.fits_viewer.displaymap.colorscale
        elif self.current_settings["color_pattern"] is None: self.color_pattern = color_pattern
        if self.current_settings["color_pattern"] is not None:
            self.color_pattern = self.current_settings["color_pattern"]

        if bad_color is None: 
            self.bad_color = self.fits_viewer.displaymap.bad_color
        else: self.bad_color = bad_color
        if filename is None:
            self.filename = self.fits_viewer.filename
        else: self.filename = filename
        
        if mode == ColorMode.MAIN:
            self.hist_color = 'gold'
            self.vline_color = 'orange'
        elif mode == ColorMode.INTEG:
            self.hist_color = 'skyblue'
            self.vline_color = 'deepskyblue'
        elif mode == ColorMode.CHANNEL:
            self.hist_color = 'yellowgreen'
            self.vline_color = 'olivedrab'
        self.initUI()


    def initUI(self):
        self.layout = QVBoxLayout()
        palette = self.palette()
        background_color = palette.color(QPalette.ColorRole.Window).name()

        self.fig, self.ax = plt.subplots(figsize=(4., 2.), facecolor=background_color)
        self.canvas = FigureCanvas(self.fig)
        self.tick_color = 'white' if palette.color(QPalette.ColorRole.Window).lightness() < 128 else 'black'
        self.ax.tick_params(axis='both', which='major', labelsize=8,  colors=self.tick_color)
        self.ax.tick_params(axis='both', which='minor', labelsize=8, colors=self.tick_color) 
        for spine in self.ax.spines.values():
            spine.set_edgecolor(self.tick_color)
        self.ax.set_xlabel('Intensity', fontsize = 8, color=self.tick_color)
        self.ax.set_ylabel('Count', fontsize = 8, color=self.tick_color)
        
        
        # Controls
        self.controls_layout = QGridLayout()
        self.controls_layout.setHorizontalSpacing(6)
        self.controls_layout.setVerticalSpacing(3)
        self.controls_layout.setContentsMargins(0, 0, 0, 0)
        
        self.colorscale_combo = QComboBox(self)
        self.colorscale_combo.setEditable(True)
        colormap_names = [name for name in colormaps.keys() if not name.endswith('_r')]
        self.colorscale_combo.addItems(sorted(colormap_names))
        
        self.colorscale_combo.currentIndexChanged.connect(self.change_color_scale)
        self.colorscale_combo.setFixedWidth(100)
        self.controls_layout.addWidget(self.colorscale_combo, 0, 0)
        
        self.intensity_min = QLineEdit(self)
        self.intensity_min.setPlaceholderText('min')
        self.intensity_min.returnPressed.connect(self.update_intensity_range)
        self.controls_layout.addWidget(self.intensity_min, 0, 1, 1, 2)
        
        self.intensity_max = QLineEdit(self)
        self.intensity_max.setPlaceholderText('max')
        self.intensity_max.returnPressed.connect(self.update_intensity_range)
        self.controls_layout.addWidget(self.intensity_max, 0, 3, 1, 2)

        if self.current_settings["min_val"] is None or self.current_settings["max_val"] is None:
            display_min, display_max = self._current_display_range()
            if display_min is not None and display_max is not None:
                self.current_settings["min_val"] = display_min
                self.current_settings["max_val"] = display_max

        if self.current_settings["min_val"] is not None:
            self.intensity_min.setText(f"{self.current_settings['min_val']:.3g}")
        if self.current_settings["max_val"] is not None:
            self.intensity_max.setText(f"{self.current_settings['max_val']:.3g}")
        
        self.set_button = QPushButton('Set', self)
        self.set_button.clicked.connect(self.update_intensity_range)
        self.controls_layout.addWidget(self.set_button, 2, 3, 1, 2)
        
        self.min_max_button = QPushButton('Min・Max', self)
        self.min_max_button.clicked.connect(self.set_min_max)
        self.controls_layout.addWidget(self.min_max_button, 1, 1, 1, 2)
        
        self.auto_button = QPushButton('Auto', self)
        self.auto_button.clicked.connect(self.auto_intensity)
        self.controls_layout.addWidget(self.auto_button, 1, 3, 1, 2)  
        

        
        self.invert_checkbox = QCheckBox('Invert', self)
        self.invert_checkbox.stateChanged.connect(self.change_color_scale)
        self.controls_layout.addWidget(self.invert_checkbox, 1, 0)
        
        self.log_checkbox = QCheckBox('Log', self)
        self.log_checkbox.clicked.connect(self.toggle_log_scale)
        
        
        self.controls_layout.addWidget(self.log_checkbox, 2, 0, 1, 2)
        
        
        
        self.gamma_label = QLabel('gamma:', self)
        self.controls_layout.addWidget(self.gamma_label, 2, 1, 1, 1, alignment=Qt.AlignmentFlag.AlignRight)
        self.gamma_spinbox = QDoubleSpinBox(self)
        self.gamma_spinbox.setRange(0.0, 100.0)
        self.gamma_spinbox.setFixedWidth(50)
        self.gamma_spinbox.setSingleStep(0.1)
        self.gamma_spinbox.setDecimals(1)
        self.gamma_spinbox.setValue(self.current_settings["gamma_value"])
        
        self.controls_layout.addWidget(self.gamma_spinbox, 2, 2, 1, 1, alignment=Qt.AlignmentFlag.AlignLeft)
        self.gamma_spinbox.editingFinished.connect(self.update_gamma_from_spinbox)
        self.set_button.clicked.connect(self.update_gamma_from_spinbox)
        
        self.hist_toggle_button = QToolButton(self)
        self.hist_toggle_button.setCheckable(True)
        self.hist_toggle_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.hist_toggle_button.setArrowType(Qt.ArrowType.RightArrow)
        self.hist_toggle_button.setIconSize(QSize(8, 8))
        self.hist_toggle_button.setText(" Histogram")
        self.hist_toggle_button.clicked.connect(self.on_toggle)
        self.hist_toggle_button.setStyleSheet("QToolButton { border: none; padding: 0px; font-size: 13px;}")

        
        self.content_area = QWidget(self)
        self.content_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        

        self.content_area_layout = QVBoxLayout()
        self.content_area_layout.setContentsMargins(0, 0, 0, 0)
        self.content_area.setLayout(self.content_area_layout)
        self.content_area_layout.addWidget(self.canvas)

        self.layout.addWidget(self.hist_toggle_button)
        self.layout.addWidget(self.content_area)
        self.layout.addLayout(self.controls_layout)

        self.layout.setContentsMargins(10, 10, 10, 0)

        self.setLayout(self.layout)
        self.setWindowTitle(f'Color Settings:{self.filename}')
        
        allow_auto_histogram = (
            self.data.size < 5e8
            and (
                self._data_nbytes is None
                or self._data_nbytes <= MEMMAP_THRESHOLD_BYTES
            )
        )
        if allow_auto_histogram:
            self.hist_toggle_button.setChecked(True)
            self.hist_toggle_button.setArrowType(Qt.ArrowType.DownArrow)
            self.content_area.setVisible(True)
            QTimer.singleShot(0, self.on_toggle)
        else:
            self.hist_toggle_button.setChecked(False)
            self.content_area.setVisible(False)
        
        self.adjustSize()
        
        
        # Initial histogram update
        blockers = [
            QSignalBlocker(self.colorscale_combo),
            QSignalBlocker(self.invert_checkbox),
            QSignalBlocker(self.log_checkbox),
            QSignalBlocker(self.gamma_spinbox),
        ]
        try:
            self.colorscale_combo.setCurrentText(self.color_pattern)
            self.init_vertical_lines()
            self.invert_checkbox.setChecked(self.current_settings["invert"])
            self.log_checkbox.setChecked(self.current_settings["log_scale"])
            self.gamma_spinbox.setValue(self.current_settings["gamma_value"])
        finally:
            del blockers

        # Connect canvas events for dragging
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('motion_notify_event', self.on_drag)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self._suppress_shared_history_recording = False


        
    def on_toggle(self):
        checked = self.hist_toggle_button.isChecked()
        if checked:
            self.hist_toggle_button.setArrowType(Qt.ArrowType.DownArrow)
            self.update_histogram()
            self.content_area.setVisible(True)
        else:
            self.hist_toggle_button.setArrowType(Qt.ArrowType.RightArrow)
            self.content_area.setVisible(False)

    def _resolve_main_viewer(self):
        viewer = getattr(self, "fits_viewer", None)
        if viewer is None:
            return None
        main_viewer = getattr(viewer, "main_viewer", None)
        if main_viewer is not None:
            return main_viewer
        root = getattr(viewer, "fits_viewer", None)
        if root is None:
            return viewer
        nested = getattr(root, "main_viewer", None)
        return nested if nested is not None else root

    def _iter_history_record_targets(self):
        root_viewer = self._resolve_main_viewer()
        targets = []
        if self.mode == ColorMode.MAIN:
            if root_viewer is not None:
                targets.append(root_viewer)
            return targets

        if self.mode == ColorMode.INTEG and root_viewer is not None:
            for ref in list(getattr(root_viewer, "integ_result_windows", []) or []):
                window = ref() if callable(ref) else ref
                if window is None:
                    continue
                targets.append(window)
        elif self.mode == ColorMode.CHANNEL and root_viewer is not None:
            for window in list(getattr(root_viewer, "channel_map_windows", []) or []):
                if window is None:
                    continue
                targets.append(window)

        viewer = getattr(self, "fits_viewer", None)
        if viewer is not None:
            targets.insert(0, viewer)

        deduped = []
        seen = set()
        for target in targets:
            marker = id(target)
            if marker in seen:
                continue
            seen.add(marker)
            deduped.append(target)
        return deduped

    def _schedule_shared_color_history(self, reason: str = "color"):
        if bool(getattr(self, "_suppress_shared_history_recording", False)):
            return
        self._shared_history_record_reason = str(reason or "color")
        if self._shared_history_record_pending:
            return
        self._shared_history_record_pending = True
        QTimer.singleShot(0, self._flush_shared_color_history_record)

    def _flush_shared_color_history_record(self):
        self._shared_history_record_pending = False
        if bool(getattr(self, "_suppress_shared_history_recording", False)):
            return
        reason = f"color:{self._shared_history_record_reason or 'color'}"
        targets = self._iter_history_record_targets()
        if not targets:
            return

        for target in targets:
            if self.mode == ColorMode.MAIN:
                recorder = getattr(target, "_record_shared_view_history", None)
            else:
                recorder = getattr(target, "_record_local_view_history", None)
            if not callable(recorder):
                continue
            try:
                recorder(reason=reason)
            except Exception:
                continue
    
    def move_to_default_position(self):
        control_panel_geometry = self.fits_viewer.control_panel.geometry()
        control_panel_x = control_panel_geometry.x()
        control_panel_y = control_panel_geometry.y()
        control_panel_width = control_panel_geometry.width()

        # Move ColorSettingsPanel to the right of ControlPanel
        self.move(control_panel_x + control_panel_width, control_panel_y)

    def _current_display_range(self):
        try:
            if self.mode == ColorMode.CHANNEL:
                im = self.fits_viewer.im_list[0] if getattr(self.fits_viewer, 'im_list', None) else None
            else:
                im = getattr(self.fits_viewer, 'im', None)
            if im is None:
                return (None, None)

            clim = im.get_clim()
            if clim is None:
                return (None, None)

            vmin, vmax = clim
            if vmin is None or vmax is None:
                return (None, None)

            if not (np.isfinite(vmin) and np.isfinite(vmax)):
                return (None, None)

            return (float(vmin), float(vmax))
        except Exception:
            return (None, None)

    def _is_large_data_mode(self):
        viewers = [
            getattr(self, "fits_viewer", None),
            getattr(getattr(self, "fits_viewer", None), "main_viewer", None),
            getattr(getattr(self, "fits_viewer", None), "fits_viewer", None),
        ]
        seen = set()
        for viewer in viewers:
            if viewer is None:
                continue
            marker = id(viewer)
            if marker in seen:
                continue
            seen.add(marker)
            checker = getattr(viewer, "is_large_data_mode", None)
            if not callable(checker):
                continue
            try:
                return bool(checker())
            except Exception:
                continue
        return False

    def _current_image_data_range(self):
        try:
            if self.mode == ColorMode.CHANNEL:
                im = self.fits_viewer.im_list[0] if getattr(self.fits_viewer, 'im_list', None) else None
            else:
                im = getattr(self.fits_viewer, 'im', None)
            if im is None:
                return (None, None)

            data = im.get_array()
            if data is None:
                return (None, None)
            if np.ma.isMaskedArray(data):
                data = data.filled(np.nan)
            data = np.asarray(data)
            if data.size == 0:
                return (None, None)

            with np.errstate(all='ignore'):
                finite = data[np.isfinite(data)]
                if finite.size == 0:
                    return (None, None)
                data_min = float(np.min(finite))
                data_max = float(np.max(finite))

            if not np.isfinite(data_min) or not np.isfinite(data_max):
                return (None, None)
            if data_max < data_min:
                data_max = data_min
            return (data_min, data_max)
        except Exception:
            return (None, None)

    def _data_range(self):
        if self._is_large_data_mode():
            display_data_min, display_data_max = self._current_image_data_range()
            if display_data_min is not None and display_data_max is not None:
                return (display_data_min, display_data_max)

            display_min, display_max = self._current_display_range()
            if display_min is not None and display_max is not None:
                return (display_min, display_max)

        allow_full_scan = (
            self._data_nbytes is None
            or self._data_nbytes <= MEMMAP_THRESHOLD_BYTES
        )
        if allow_full_scan:
            try:
                with np.errstate(all='ignore'):
                    data_min = float(np.nanmin(self.data))
                    data_max = float(np.nanmax(self.data))
            except ValueError:
                data_min = 0.0
                data_max = 0.0
        else:
            data_min, data_max = fast_nanminmax(self.data)

        if not np.isfinite(data_min):
            data_min = 0.0

        if not np.isfinite(data_max):
            data_max = data_min

        if data_max < data_min:
            data_max = data_min

        return (data_min, data_max)

    def update_intensity_range(self):
        if self.mode == ColorMode.CHANNEL:
            if self.fits_viewer.im_list[0].get_cmap().name != 'from_list':
                self.change_color_scale()
        else:
            if self.fits_viewer.im.get_cmap().name != 'from_list':
                self.change_color_scale()
        try:
            data_min, data_max = self._data_range()
            min_val = float(self.intensity_min.text()) if self.intensity_min.text() else data_min
            max_val = float(self.intensity_max.text()) if self.intensity_max.text() else data_max
            if min_val > max_val:
                min_val, max_val = max_val, min_val
                self.intensity_min.setText(str(f"{min_val:.3g}"))
                self.intensity_max.setText(str(f"{max_val:.3g}"))
                    
            self.current_settings["min_val"] = min_val
            self.current_settings["max_val"] = max_val
            
            self.apply_to_all_windows(lambda im: im.set_clim(min_val, max_val))
            
            # Update histogram lines
            self.update_histogram_lines(min_val, max_val)
            self._schedule_shared_color_history("intensity")
            
            """
            # Save the state
            if not self.integ:
                ColorSettingsPanel.min_val = min_val
                ColorSettingsPanel.max_val = max_val
            else:
                ColorSettingsPanel.min_val_integ = min_val
                ColorSettingsPanel.max_val_integ = max_val
            """
            
        except ValueError:
            QMessageBox.warning(self, 'Error', 'Invalid intensity range values')
    
    def set_min_max(self):
        min_val, max_val = self._data_range()
        self.intensity_min.setText(str(f"{min_val:.3g}"))
        self.intensity_max.setText(str(f"{max_val:.3g}"))
        self.update_intensity_range()

    def auto_intensity(self):
        data_min, data_max = self._data_range()
        max_val = data_max * 0.8 if np.isfinite(data_max) else data_max
        min_val = max(data_min, 0.0) if np.isfinite(data_min) else data_min

        #max_val = np.nanpercentile(self.data[self.data > 0], 99.99)
        self.intensity_min.setText(str(f"{min_val:.3g}"))
        self.intensity_max.setText(str(f"{max_val:.3g}"))
        self.update_intensity_range()
    
    def toggle_log_scale(self):
        try:
            min_val = float(self.intensity_min.text())
            max_val = float(self.intensity_max.text())
    
            if self.log_checkbox.isChecked():
                if min_val <= 0 or max_val <= 0:
                    raise ValueError("Intensity range values must be positive for log scale.")
                
                self.apply_to_all_windows(lambda im: im.set_norm(mpl.colors.LogNorm(vmin=min_val, vmax=max_val)))
                self.apply_to_all_windows(lambda im: im.cmap.set_bad(color=self.config.get('bad_color', 'black')))
                self.auto_button.setEnabled(False)
                self.min_max_button.setEnabled(False)
                #ColorSettingsPanel.log_scale = True
            else:
                norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)
                self.apply_to_all_windows(lambda im: im.set_norm(norm))
                self.auto_button.setEnabled(True)
                self.min_max_button.setEnabled(True)
                #ColorSettingsPanel.log_scale = False    
        except ValueError as e:
            QMessageBox.warning(self, 'Error', str(e))
            self.log_checkbox.setChecked(False)
            
            #ColorSettingsPanel.log_scale = False
        self.current_settings["log_scale"] = self.log_checkbox.isChecked()
        self.update_intensity_range()
    
    def apply_to_all_windows(self, func):        
        self._suppress_colorbar_layout_for_pending_update()
        try:
            if self.mode == ColorMode.CHANNEL:
                for i in range(len(self.fits_viewer.im_list)):
                    im = self.fits_viewer.im_list[i]
                    func(im)
                    self._invalidate_viewer_cache(self.fits_viewer)
                    if getattr(self.fits_viewer, 'canvas', None):
                        self.fits_viewer.canvas.draw_idle()
                    if self.subwindows is not None:
                        for window in self.subwindows:
                            if window:
                                func(window.im_list[i])       
                                self._invalidate_viewer_cache(window)
                                if getattr(window, 'canvas', None):
                                    window.canvas.draw_idle()
            else:
                func(self.fits_viewer.im)
                self._invalidate_viewer_cache(self.fits_viewer)
                if getattr(self.fits_viewer, 'canvas', None):
                    self.fits_viewer.canvas.draw_idle()
                if self.subwindows is not None:
                    for window in self.subwindows:
                        if window:
                            func(window.im)       
                            self._invalidate_viewer_cache(window)
                            if getattr(window, 'canvas', None):
                                window.canvas.draw_idle()

            self._reapply_colorbar_style_for_changed_viewers(self._current_colorbar_config())
            self._refresh_overlay_after_color_change()
        finally:
            self._defer_restore_colorbar_layout_after_pending_update()

    def _invalidate_viewer_cache(self, viewer):
        if viewer is None:
            return
        state = getattr(viewer, 'state', None)
        if state is not None:
            state.image_background = None
            state._background = None
        try:
            viewer._background = None
        except Exception:
            pass

    def _suppress_colorbar_layout_for_pending_update(self):
        if getattr(self, "_pending_layout_suppression", None) is not None:
            return
        targets = []
        viewer = getattr(self, "fits_viewer", None)
        if viewer is not None:
            targets.append(viewer)
        for window in list(getattr(self, "subwindows", None) or []):
            if window is not None:
                targets.append(window)

        records = []
        seen = set()
        for target in targets:
            marker = id(target)
            if marker in seen:
                continue
            seen.add(marker)
            previous = self._set_colorbar_layout_suppressed(target, True)
            if previous is not None:
                records.append((target, previous))
        self._pending_layout_suppression = records

    def _restore_colorbar_layout_after_pending_update(self):
        records = list(getattr(self, "_pending_layout_suppression", None) or [])
        self._pending_layout_suppression = None
        for target, previous in records:
            try:
                setattr(target, "_suspend_colorbar_auto_layout", bool(previous))
            except Exception:
                continue

    def _iter_changed_viewers_for_color_update(self):
        viewers = []
        root = getattr(self, "fits_viewer", None)
        if root is not None:
            viewers.append(root)
        for window in list(getattr(self, "subwindows", None) or []):
            if window is not None:
                viewers.append(window)

        deduped = []
        seen = set()
        for viewer in viewers:
            marker = id(viewer)
            if marker in seen:
                continue
            seen.add(marker)
            deduped.append(viewer)
        return deduped

    def _iter_colorbar_targets_for_viewer(self, viewer):
        targets = []
        seen = set()

        def _append(cax, colorbar, canvas):
            if cax is None or colorbar is None:
                return
            marker = id(cax)
            if marker in seen:
                return
            seen.add(marker)
            targets.append((cax, colorbar, canvas))

        get_state = getattr(viewer, "get_viewer_state", None)
        if callable(get_state):
            for plane in ("xy", "xz", "zy"):
                try:
                    state = get_state(plane)
                except Exception:
                    state = None
                if state is None:
                    continue
                _append(
                    getattr(state, "cax", None),
                    getattr(state, "colorbar", None),
                    getattr(state, "canvas", None),
                )

        _append(
            getattr(viewer, "cax", None),
            getattr(viewer, "colorbar", None),
            getattr(viewer, "canvas", None),
        )
        return targets

    def _reapply_colorbar_style_for_changed_viewers(self, config):
        viewers = self._iter_changed_viewers_for_color_update()
        for viewer in viewers:
            targets = self._iter_colorbar_targets_for_viewer(viewer)
            for cax, colorbar, canvas in targets:
                original_bounds = None
                try:
                    original_bounds = tuple(float(v) for v in cax.get_position().bounds)
                except Exception:
                    original_bounds = None

                try:
                    self.apply_colorbar_settings(cax, colorbar, config, caller=self)
                except Exception:
                    continue

                if original_bounds is not None:
                    try:
                        current_bounds = tuple(float(v) for v in cax.get_position().bounds)
                        if any(abs(cur - org) > 1e-9 for cur, org in zip(current_bounds, original_bounds)):
                            cax.set_position(list(original_bounds))
                    except Exception:
                        pass

                if canvas is not None:
                    try:
                        canvas.draw_idle()
                    except Exception:
                        pass

        for viewer in viewers:
            sig_fn = getattr(viewer, "_colorbar_layout_anchor_signature", None)
            if callable(sig_fn):
                try:
                    viewer._colorbar_auto_anchor_sig = sig_fn()
                except Exception:
                    pass

    def _defer_restore_colorbar_layout_after_pending_update(self):
        if bool(getattr(self, "_layout_restore_pending", False)):
            return
        self._layout_restore_pending = True

        def _run_restore():
            self._layout_restore_pending = False
            self._restore_colorbar_layout_after_pending_update()

        # Keep suppression for one UI frame so draw_idle callbacks triggered by
        # color updates do not re-schedule colorbar auto-layout.
        QTimer.singleShot(30, _run_restore)

    def _current_colorbar_config(self):
        viewer = getattr(self, "fits_viewer", None)
        config = None

        if self.mode == ColorMode.MAIN:
            config = getattr(getattr(viewer, "config_manager", None), "config", None)
        elif self.mode == ColorMode.CHANNEL:
            config = getattr(viewer, "config", None)
            if not isinstance(config, dict):
                root = getattr(viewer, "fits_viewer", None)
                config = getattr(getattr(root, "config_manager", None), "config", None)
        elif self.mode == ColorMode.INTEG:
            getter = getattr(viewer, "_get_colorbar_config", None)
            if callable(getter):
                try:
                    config = getter()
                except Exception:
                    config = None
            if not isinstance(config, dict):
                config = getattr(viewer, "config", None)
            if not isinstance(config, dict):
                root = getattr(viewer, "fits_viewer", None)
                config = getattr(getattr(root, "config_manager", None), "config", None)

        if not isinstance(config, dict):
            config = self.config if isinstance(self.config, dict) else {}
        self.config = config
        return config

    @staticmethod
    def _set_colorbar_layout_suppressed(viewer, suppressed):
        if viewer is None:
            return None
        previous = bool(getattr(viewer, "_suspend_colorbar_auto_layout", False))
        try:
            setattr(viewer, "_suspend_colorbar_auto_layout", bool(suppressed))
        except Exception:
            return None
        return previous

    def _refresh_overlay_after_color_change(self):
        def _redraw_viewer_overlay(viewer):
            if viewer is None:
                return
            redraw_for_plane = getattr(viewer, "redraw_overlay_for_plane", None)
            if callable(redraw_for_plane):
                plane = str(getattr(viewer, "plane", "") or "")
                try:
                    if plane in ("xy", "xz", "zy"):
                        redraw_for_plane(plane)
                    else:
                        redraw_for_plane()
                    return
                except Exception:
                    pass
            redraw_main = getattr(viewer, "redraw_main_overlay_and_blit", None)
            if callable(redraw_main):
                try:
                    redraw_main()
                    return
                except Exception:
                    pass
            canvas = getattr(viewer, "canvas", None)
            if canvas is not None:
                try:
                    canvas.draw_idle()
                except Exception:
                    pass

        root = self._resolve_main_viewer()
        if self.mode == ColorMode.MAIN and root is not None:
            redraw_for_plane = getattr(root, "redraw_overlay_for_plane", None)
            if callable(redraw_for_plane):
                viewer_for_plane = getattr(root, "_viewer_for_plane", None)
                for plane in ("xy", "xz", "zy"):
                    if plane != "xy":
                        viewer = viewer_for_plane(plane) if callable(viewer_for_plane) else None
                        if viewer is None:
                            continue
                    try:
                        redraw_for_plane(plane)
                    except Exception:
                        continue
                return
            _redraw_viewer_overlay(root)
            return

        _redraw_viewer_overlay(self.fits_viewer)
        for window in list(self.subwindows or []):
            _redraw_viewer_overlay(window)

    def update_histogram(self):
        data = np.asanyarray(self.data).ravel()
        if self._data_nbytes is not None and self._data_nbytes > MEMMAP_THRESHOLD_BYTES:
            stride = max(1, int(np.ceil(data.size / 1_000_000)))
            data = data[::stride]
        with np.errstate(invalid='ignore'):
            data = data[np.isfinite(data)]
        self.ax.hist(data, bins=100, log=True, color=self.hist_color)
        self.fig.tight_layout()
        self.fig.subplots_adjust(bottom=0.2) 
        self.canvas.draw_idle()
    

    def init_vertical_lines(self):
        data_min, data_max = self._data_range()
        display_min, display_max = self._current_display_range()
        try:        
            if self.current_settings["min_val"] is not None and self.current_settings["max_val"] is not None:
                min_val = self.current_settings["min_val"]
                max_val = self.current_settings["max_val"]
            else:
                if self.intensity_min.text():
                    min_val = float(self.intensity_min.text())
                elif display_min is not None:
                    min_val = display_min
                else:
                    min_val = data_min

                if self.intensity_max.text():
                    max_val = float(self.intensity_max.text())
                elif display_max is not None:
                    max_val = display_max
                else:
                    max_val = data_max

        except ValueError:
            min_val = data_min
            max_val = data_max

        if min_val > max_val:
            min_val, max_val = max_val, min_val
        if min_val < data_min or min_val > data_max: min_val = data_min
        if max_val > data_max or max_val < data_min: max_val = data_max
        
        self.intensity_min.setText(str(f"{min_val:.3g}"))
        self.intensity_max.setText(str(f"{max_val:.3g}"))
        
        self.min_line = self.ax.axvline(min_val, color=self.vline_color, linestyle='-', linewidth=1, animated = False)
        self.max_line = self.ax.axvline(max_val, color=self.vline_color, linestyle='-', linewidth=1, animated = False)
        self.fig.tight_layout()
        self.canvas.draw_idle()
    
    def update_histogram_lines(self, min_val, max_val):
        self.min_line.set_xdata([min_val])
        self.max_line.set_xdata([max_val])
        self.canvas.draw_idle()
    
    def on_click(self, event):
        if event.inaxes != self.ax:
            return

        min_x = self.min_line.get_xdata()[0]
        max_x = self.max_line.get_xdata()[0]
        
        min_pixel_x = self.ax.transData.transform([min_x, 0])[0]
        max_pixel_x = self.ax.transData.transform([max_x, 0])[0]
        event_pixel_x = event.x
        if abs(event_pixel_x - min_pixel_x) < abs(event_pixel_x - max_pixel_x):
            if abs(event_pixel_x - min_pixel_x) < 15:  # 15 pixels threshold
                self.dragging_min = True
                self.min_line.set_animated(True)
        else:
            if abs(event_pixel_x - max_pixel_x) < 15:  # 15 pixels threshold
                self.dragging_max = True
                self.max_line.set_animated(True)
        self.canvas.draw()
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)

        self.min_line.set_animated(False)
        self.max_line.set_animated(False)
        self.canvas.draw_idle()
        
    
    def on_motion(self, event):
        if event.inaxes != self.ax:
            return
        
        min_x = self.min_line.get_xdata()[0]
        max_x = self.max_line.get_xdata()[0]
        
        min_pixel_x = self.ax.transData.transform([min_x, 0])[0]
        max_pixel_x = self.ax.transData.transform([max_x, 0])[0]
        event_pixel_x = event.x
        
        if abs(event_pixel_x - min_pixel_x) < abs(event_pixel_x - max_pixel_x):
            if abs(event_pixel_x - min_pixel_x) <= 15:  # 15 pixels threshold
                self.canvas.setCursor(Qt.CursorShape.SizeHorCursor)
                self.min_line.set_linewidth(3)  # Increase line width
                self.max_line.set_linewidth(1)  # Reset line width
            else:
                self.canvas.setCursor(Qt.CursorShape.ArrowCursor)
                self.min_line.set_linewidth(1)  # Reset line width
                self.max_line.set_linewidth(1)  # Reset line width
            if event.button != 1: self.canvas.draw_idle()
        else:
            if abs(event_pixel_x - max_pixel_x) <= 15:  # 15 pixels threshold
                self.canvas.setCursor(Qt.CursorShape.SizeHorCursor)
                self.max_line.set_linewidth(3)  # Increase line width
                self.min_line.set_linewidth(1)  # Reset line width
            else:
                self.canvas.setCursor(Qt.CursorShape.ArrowCursor)
                self.min_line.set_linewidth(1)  # Reset line width
                self.max_line.set_linewidth(1)  # Reset line width
            if event.button != 1: self.canvas.draw_idle()
        

    
    def on_drag(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return
            
        self.canvas.restore_region(self.background)
        
        if self.dragging_min:
            self.min_line.set_xdata([event.xdata])
            self.intensity_min.setText(str(f"{event.xdata:.3g}"))
            self.ax.draw_artist(self.min_line)
        elif self.dragging_max:
            self.max_line.set_xdata([event.xdata])
            self.intensity_max.setText(str(f"{event.xdata:.3g}"))
            self.ax.draw_artist(self.max_line)
        
        self.canvas.blit(self.ax.bbox)

    
    def on_release(self, event):
        if self.dragging_min or self.dragging_max:
            self.update_intensity_range()
        
        self.dragging_min = False
        self.dragging_max = False
        self.canvas.setCursor(Qt.CursorShape.ArrowCursor)
        self.min_line.set_linewidth(1)  # Reset line width
        self.max_line.set_linewidth(1)  # Reset line width
        self.min_line.set_animated(False)
        self.max_line.set_animated(False)


     
    def change_color_scale(self):
        pattern = self.colorscale_combo.currentText()
        if not pattern:
            QMessageBox.warning(self, 'Error', 'Color pattern cannot be empty.')
            return
        if pattern not in colormaps:
            QMessageBox.warning(self, 'Error', 'Invalid color pattern selected')
            self.colorscale_combo.setCurrentText(pattern.replace('_r', ''))  # Reset to last valid pattern
            return  # Exit the function if the pattern is invalid
        if self.invert_checkbox.isChecked():
            pattern += '_r'
            self.current_settings["invert"] = True
        #ColorSettingsPanel.color_pattern = pattern
        self.current_settings["color_pattern"] = pattern
        self.color_pattern = pattern

        try:
            def update_cmap(im):
                #im.set_cmap(pattern)
                im.set_cmap(self.create_gamma_cmap(pattern))
                im.cmap.set_bad(color=self.bad_color)
            self.apply_to_all_windows(update_cmap)
        except ValueError as ve:
            QMessageBox.warning(self, 'Error', str(ve))
            return
        self._schedule_shared_color_history("cmap")
        

    @classmethod
    def apply_colorbar_settings(cls, cax, colorbar, config, caller = None):
        if not isinstance(config, dict):
            config = {}

        tick_left = bool(config.get('colorbar_tick_left', False))
        tick_right = bool(config.get('colorbar_tick_right', True))
        tick_top = bool(config.get('colorbar_tick_top', False))
        tick_bottom = bool(config.get('colorbar_tick_bottom', True))
        label_left = bool(config.get('colorbar_tick_labelleft', False))
        label_right = not label_left
        label_top = bool(config.get('colorbar_tick_labeltop', False))
        label_bottom = not label_top
        tick_width = float(config.get('colorbar_tick_width', 1))
        tick_length = float(config.get('colorbar_tick_length', 4))
        mtick_length = float(config.get('colorbar_mtick_length', 2))
        tick_color = config.get('colorbar_tick_color')
        tick_labelcolor = config.get('colorbar_tick_labelcolor')
        tick_direction = config.get('colorbar_tick_direction')

        try:
            cax.set_zorder(300)
        except Exception:
            pass
        try:
            colorbar.ax.set_zorder(300)
        except Exception:
            pass

        cax.tick_params(
            axis='y',
            which='major',
            left=tick_left,
            right=tick_right,
            labelleft=label_left,
            labelright=label_right,
            width=tick_width,
            length=tick_length,
            color=tick_color,
            direction=tick_direction,
            labelcolor=tick_labelcolor,
        )
        cax.tick_params(
            axis='x',
            which='major',
            top=tick_top,
            bottom=tick_bottom,
            labeltop=label_top,
            labelbottom=label_bottom,
            width=tick_width,
            length=tick_length,
            color=tick_color,
            direction=tick_direction,
            labelcolor=tick_labelcolor,
        )
        cax.tick_params(axis='y', which='minor', left=tick_left, right=tick_right, length=mtick_length, color=tick_color, width=tick_width, direction=tick_direction)
        cax.tick_params(axis='x', which='minor', top=tick_top, bottom=tick_bottom, length=mtick_length, color=tick_color, width=tick_width, direction=tick_direction)

        try:
            if tick_left and tick_right:
                cax.yaxis.set_ticks_position('both')
            elif tick_left:
                cax.yaxis.set_ticks_position('left')
            elif tick_right:
                cax.yaxis.set_ticks_position('right')
            else:
                cax.yaxis.set_ticks_position('none')
        except Exception:
            pass
        try:
            if tick_top and tick_bottom:
                cax.xaxis.set_ticks_position('both')
            elif tick_top:
                cax.xaxis.set_ticks_position('top')
            elif tick_bottom:
                cax.xaxis.set_ticks_position('bottom')
            else:
                cax.xaxis.set_ticks_position('none')
        except Exception:
            pass
        try:
            cax.yaxis.set_label_position('left' if label_left else 'right')
        except Exception:
            pass
        try:
            cax.xaxis.set_label_position('top' if label_top else 'bottom')
        except Exception:
            pass

        colorbar.outline.set_color(tick_color)
        colorbar.outline.set_linewidth(tick_width)
        label_text = config.get('colorbar_label', '')
        if label_text is None:
            label_text = ''
        colorbar.set_label(str(label_text), fontsize=config.get('colorbar_label_fontsize'), color=config.get('colorbar_label_color'))
    
        colorbar.ax.minorticks_on()
        colorbar.ax.tick_params(which='minor', length=mtick_length, color=tick_color, width=tick_width, direction=tick_direction)
    
        if caller and hasattr(caller, 'log_checkbox') and caller.log_checkbox.isChecked():
            return
        try:
            mtick_freq = max(1, int(config.get('colorbar_mtick_freq', 2)))
        except Exception:
            mtick_freq = 2
        colorbar.ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(mtick_freq))
        colorbar.ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(mtick_freq))

    
    def create_gamma_cmap(self, pattern):
        cmap = plt.get_cmap(pattern)
        if pattern in ['Rainbow', 'Cool']:
            custom_cmap = CustomColormap(pattern, cmap._segmentdata)
            gamma_cmap = custom_cmap.apply_gamma(self.current_settings["gamma_value"])
            try:
                gamma_cmap.set_bad(color=self.config.get('bad_color', 'black'))
            except Exception:
                pass
            return gamma_cmap
        gamma_cmap = cmap(np.linspace(0, 1, cmap.N) ** self.current_settings["gamma_value"])
        listed = mpl.colors.ListedColormap(gamma_cmap)
        try:
            listed.set_bad(color=self.config.get('bad_color', 'black'))
        except Exception:
            pass
        return listed
    
    def update_gamma_from_spinbox(self):
        value = self.gamma_spinbox.value()
        self.current_settings["gamma_value"] = value
        self.apply_to_all_windows(lambda im: im.set_cmap(self.create_gamma_cmap(self.current_settings["color_pattern"])))
        self._schedule_shared_color_history("gamma")


    
    def closeEvent(self,event):
        self.current_settings["color_pattern"] = self.colorscale_combo.currentText()
        ColorSettingsPanel.settings[self.mode] = dict(self.current_settings)
        pass
        try:
            plt.close(self.fig)
        except Exception:
            pass
        super().closeEvent(event)
