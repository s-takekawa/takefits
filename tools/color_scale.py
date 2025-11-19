from PyQt6.QtWidgets import QWidget, QVBoxLayout, QGridLayout, QComboBox, QLineEdit, QPushButton, QCheckBox, QLabel, QToolButton, QWidget, QSizePolicy, QMessageBox, QDoubleSpinBox
from PyQt6.QtGui import QPalette
from PyQt6.QtCore import Qt, QTimer, QSize
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from enum import Enum
from core.common import Common
from core.custom_colormap import CustomColormap, ColorDefinitions
from logic.data_tools import (
    estimate_array_nbytes,
    fast_nanminmax,
    MEMMAP_THRESHOLD_BYTES,
)


class RegisterColor:
    #Add colormaps
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

class ColorMode(Enum):
    MAIN = "main"
    INTEG = "integ"
    CHANNEL = "channel"


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
        self.current_settings = ColorSettingsPanel.settings[self.mode]
        
        if config is None: self.config = self.fits_viewer.displaymap.config
        else: self.config = config
        self.dragging_min = False
        self.dragging_max = False
        self.near_line = False
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
        self._histogram_computed = False
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
        #self.colorscale_combo.addItems(sorted(colormaps.keys()))
        # exclude '_r'
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
        self.controls_layout.addWidget(self.min_max_button, 1, 1, 1, 2)  # Moved to the right side
        
        self.auto_button = QPushButton('Auto', self)
        self.auto_button.clicked.connect(self.auto_intensity)
        self.controls_layout.addWidget(self.auto_button, 1, 3, 1, 2)  
        

        
        self.invert_checkbox = QCheckBox('Invert', self)  # Name changed from reverse_checkbox to invert_checkbox
        self.invert_checkbox.stateChanged.connect(self.change_color_scale)
        self.controls_layout.addWidget(self.invert_checkbox, 1, 0)  # Moved below the combo box
        
        self.log_checkbox = QCheckBox('Log', self)
        #self.log_checkbox.stateChanged.connect(self.toggle_log_scale)
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
        self.hist_toggle_button.setText(" Histgram")
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
        
        if self.data.size < 5e8: #Threshold 
            self.hist_toggle_button.setChecked(True)
            self.hist_toggle_button.setArrowType(Qt.ArrowType.DownArrow)
            self.content_area.setVisible(True)
            QTimer.singleShot(0, self.on_toggle)
        else:
            self.hist_toggle_button.setChecked(False)
            self.content_area.setVisible(False)
        
        self.adjustSize()
#        self.move_to_default_position()
        
        
        # Initial histogram update
        self.colorscale_combo.setCurrentText(self.color_pattern)
        self.init_vertical_lines()

        self.invert_checkbox.setChecked(self.current_settings["invert"])
        self.log_checkbox.setChecked(self.current_settings["log_scale"])
        self.gamma_spinbox.setValue(self.current_settings["gamma_value"])

        # Connect canvas events for dragging
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('motion_notify_event', self.on_drag)
        self.canvas.mpl_connect('button_release_event', self.on_release)


        
    def on_toggle(self):
        checked = self.hist_toggle_button.isChecked()
        if checked:
            self.hist_toggle_button.setArrowType(Qt.ArrowType.DownArrow)
            self.update_histogram()
            self.content_area.setVisible(True)
        else:
            self.hist_toggle_button.setArrowType(Qt.ArrowType.RightArrow)
            self.content_area.setVisible(False)
    
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

    def _data_range(self):
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
                
                log_norm = mpl.colors.LogNorm(vmin=min_val, vmax=max_val)
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
            #self.canvas.draw()
        except ValueError as e:
            QMessageBox.warning(self, 'Error', str(e))
            self.log_checkbox.setChecked(False)
            
            #ColorSettingsPanel.log_scale = False
        self.current_settings["log_scale"] = self.log_checkbox.isChecked()
        self.update_intensity_range()
    
    def apply_to_all_windows(self, func):        
        if self.mode == ColorMode.CHANNEL:
            for i in range(len(self.fits_viewer.im_list)):
                im = self.fits_viewer.im_list[i]
                func(im)
                if self.subwindows is not None:
                    for window in self.subwindows:
                        if window:
                            func(window.im_list[i])       
                if not hasattr(self, '_color_update_pending'):
                    self._color_update_pending = False
                if not self._color_update_pending:
                    self._color_update_pending = True
                    QTimer.singleShot(0, self._do_apply_colorbar_settings)
        
        else:
            func(self.fits_viewer.im)
            if self.subwindows is not None:
                for window in self.subwindows:
                    if window:
                        func(window.im)       
            if not hasattr(self, '_color_update_pending'):
                self._color_update_pending = False
            if not self._color_update_pending:
                self._color_update_pending = True
                QTimer.singleShot(0, self._do_apply_colorbar_settings)

    def _do_apply_colorbar_settings(self):
        self.apply_colorbar_settings_to_all(self.config)
        self._color_update_pending = False

    def update_histogram(self):
        #self.ax.clear()
        data = np.asanyarray(self.data).ravel()
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
        

    def apply_colorbar_settings_to_all(self, config):
        if self.mode == ColorMode.INTEG:
            for cax, colorbar in zip(Common.integ_cax, Common.integ_colorbar):
                self.apply_colorbar_settings(cax, colorbar, config, caller=self)
        elif self.mode == ColorMode.MAIN:
            self.apply_colorbar_settings(Common.cax_xy, Common.colorbar_xy, config, caller=self)
        elif self.mode == ColorMode.CHANNEL:
            self.apply_colorbar_settings(Common.ch_cax, Common.ch_colorbar, config, caller=self)
        
        self.fits_viewer.canvas.draw()

        if self.subwindows is not None:
            for window, cax, colorbar in zip(self.subwindows, [Common.cax_xz, Common.cax_zy], [Common.colorbar_xz, Common.colorbar_zy]):
                if window:
                    self.apply_colorbar_settings(cax, colorbar, config, caller=self)
                    window.canvas.draw()
    
    @classmethod
    def apply_colorbar_settings(cls, cax, colorbar, config, caller = None):
        cax.tick_params(axis='y', which='both', left=config.get('colorbar_tick_left'), right=config.get('colorbar_tick_right'), 
                        labelleft=config.get('colorbar_tick_labelleft'), labelright=(not config.get('colorbar_tick_labelleft')),
                        width=config.get('colorbar_tick_width'), length=config.get('colorbar_tick_length'), 
                        color=config.get('colorbar_tick_color'), direction=config.get('colorbar_tick_direction'), 
                        labelcolor=config.get('colorbar_tick_labelcolor'))
    
        cax.tick_params(axis='x', which='both', top=config.get('colorbar_tick_top'), bottom=config.get('colorbar_tick_bottom'), 
                        labeltop=config.get('colorbar_tick_labeltop'), labelbottom=(not config.get('colorbar_tick_labeltop')),
                        width=config.get('colorbar_tick_width'), length=config.get('colorbar_tick_length'), 
                        color=config.get('colorbar_tick_color'), direction=config.get('colorbar_tick_direction'), 
                        labelcolor=config.get('colorbar_tick_labelcolor'))
    
        colorbar.outline.set_color(config.get('colorbar_tick_color'))
        colorbar.outline.set_linewidth(config.get('colorbar_tick_width'))
        colorbar.set_label(config.get('colorbar_label', 'Label'), fontsize=config.get('colorbar_label_fontsize'), color=config.get('colorbar_label_color'))
    
        colorbar.ax.minorticks_on()
        colorbar.ax.tick_params(which='minor', length=config.get('colorbar_mtick_length'), color=config.get('colorbar_tick_color'))
    
        if caller and hasattr(caller, 'log_checkbox') and caller.log_checkbox.isChecked():
            return
        colorbar.ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(config.get('colorbar_mtick_freq')))
        colorbar.ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(config.get('colorbar_mtick_freq')))

    
    def create_gamma_cmap(self, pattern):
        cmap = plt.get_cmap(pattern)
        if pattern in ['Rainbow', 'Cool']:
            custom_cmap = CustomColormap(pattern, cmap._segmentdata)
            return custom_cmap.apply_gamma(self.current_settings["gamma_value"])
        gamma_cmap = cmap(np.linspace(0, 1, cmap.N) ** self.current_settings["gamma_value"])
        self.apply_to_all_windows(lambda im: im.cmap.set_bad(color=self.config.get('bad_color', 'black')))
        return mpl.colors.ListedColormap(gamma_cmap)
    
    def update_gamma_from_spinbox(self):
        value = self.gamma_spinbox.value()
        self.current_settings["gamma_value"] = value
        self.apply_to_all_windows(lambda im: im.set_cmap(self.create_gamma_cmap(self.current_settings["color_pattern"])))
        self.apply_to_all_windows(lambda im: im.cmap.set_bad(color=self.config.get('bad_color', 'black')))


    
    def closeEvent(self,event):
        self.current_settings["color_pattern"] = self.colorscale_combo.currentText()
        ColorSettingsPanel.settings[self.mode] = self.current_settings
        self.colorscale_combo.currentIndexChanged.disconnect(self.change_color_scale)
        plt.close(self.fig)
        self.close()
        super().closeEvent(event)
        self.destroyed.emit()

        
    
        self.set_button.clicked.connect(self.update_intensity_range)
        self.controls_layout.addWidget(self.set_button, 2, 3, 1, 2)
        
        self.min_max_button = QPushButton('Min・Max', self)
        self.min_max_button.clicked.connect(self.set_min_max)
        self.controls_layout.addWidget(self.min_max_button, 1, 1, 1, 2)  # Moved to the right side
        
        self.auto_button = QPushButton('Auto', self)
        self.auto_button.clicked.connect(self.auto_intensity)
        self.controls_layout.addWidget(self.auto_button, 1, 3, 1, 2)  
        

        
        self.invert_checkbox = QCheckBox('Invert', self)  # Name changed from reverse_checkbox to invert_checkbox
        self.invert_checkbox.stateChanged.connect(self.change_color_scale)
        self.controls_layout.addWidget(self.invert_checkbox, 1, 0)  # Moved below the combo box
        
        self.log_checkbox = QCheckBox('Log', self)
        #self.log_checkbox.stateChanged.connect(self.toggle_log_scale)
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
        self.hist_toggle_button.setText(" Histgram")
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
        
        if self.data.size < 5e8: #Threshold 
            self.hist_toggle_button.setChecked(True)
            self.hist_toggle_button.setArrowType(Qt.ArrowType.DownArrow)
            self.update_histogram()
            self.content_area.setVisible(True)
        else:
            self.hist_toggle_button.setChecked(False)
            self.content_area.setVisible(False)
        
        self.adjustSize()
#        self.move_to_default_position()
        
        
        # Initial histogram update
        self.colorscale_combo.setCurrentText(self.color_pattern)
        self.init_vertical_lines()

        self.invert_checkbox.setChecked(self.current_settings["invert"])
        self.log_checkbox.setChecked(self.current_settings["log_scale"])
        self.gamma_spinbox.setValue(self.current_settings["gamma_value"])

        # Connect canvas events for dragging
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('motion_notify_event', self.on_drag)
        self.canvas.mpl_connect('button_release_event', self.on_release)


        
    def on_toggle(self):
        checked = self.hist_toggle_button.isChecked()
        if checked:
            self.hist_toggle_button.setArrowType(Qt.ArrowType.DownArrow)
            self.update_histogram()
            self.content_area.setVisible(True)
        else:
            self.hist_toggle_button.setArrowType(Qt.ArrowType.RightArrow)
            self.content_area.setVisible(False)
    
    def move_to_default_position(self):
        control_panel_geometry = self.fits_viewer.control_panel.geometry()
        control_panel_x = control_panel_geometry.x()
        control_panel_y = control_panel_geometry.y()
        control_panel_width = control_panel_geometry.width()
        
        # Move ColorSettingsPanel to the right of ControlPanel
        self.move(control_panel_x + control_panel_width, control_panel_y)

    def update_intensity_range(self):
        if self.mode == ColorMode.CHANNEL:
            if self.fits_viewer.im_list[0].get_cmap().name != 'from_list':
                self.change_color_scale()
        else:
            if self.fits_viewer.im.get_cmap().name != 'from_list':
                self.change_color_scale()
        try:
            min_val = float(self.intensity_min.text() or np.nanmin(self.data))
            max_val = float(self.intensity_max.text() or np.nanmax(self.data))
            if min_val > max_val:
                min_val, max_val = max_val, min_val
                self.intensity_min.setText(str(f"{min_val:.3g}"))
                self.intensity_max.setText(str(f"{max_val:.3g}"))
            
            if self.log_checkbox.isChecked():
                if min_val <= 0 or max_val <= 0:
                    raise ValueError
                
            self.current_settings["min_val"] = min_val
            self.current_settings["max_val"] = max_val
            
            self.apply_to_all_windows(lambda im: im.set_clim(min_val, max_val))
            
            # Update histogram lines
            self.update_histogram_lines(min_val, max_val)
            
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
        min_val = np.nanmin(self.data)
        max_val = np.nanmax(self.data)
        self.intensity_min.setText(str(f"{min_val:.3g}"))
        self.intensity_max.setText(str(f"{max_val:.3g}"))
        self.update_intensity_range()
    
    def auto_intensity(self):
        max_val = np.nanmax(self.data) * 0.8
        min_val = 0

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
                
                log_norm = mpl.colors.LogNorm(vmin=min_val, vmax=max_val)
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
            #self.canvas.draw()
        except ValueError as e:
            QMessageBox.warning(self, 'Error', str(e))
            self.log_checkbox.setChecked(False)
            
            #ColorSettingsPanel.log_scale = False
        self.current_settings["log_scale"] = self.log_checkbox.isChecked()
        self.update_intensity_range()
    
    def apply_to_all_windows(self, func):        
        if self.mode == ColorMode.CHANNEL:
            for i in range(len(self.fits_viewer.im_list)):
                im = self.fits_viewer.im_list[i]
                func(im)
                if self.subwindows is not None:
                    for window in self.subwindows:
                        if window:
                            func(window.im_list[i])       
                if not hasattr(self, '_color_update_pending'):
                    self._color_update_pending = False
                if not self._color_update_pending:
                    self._color_update_pending = True
                    QTimer.singleShot(0, self._do_apply_colorbar_settings)
        
        else:
            func(self.fits_viewer.im)
            if self.subwindows is not None:
                for window in self.subwindows:
                    if window:
                        func(window.im)       
            if not hasattr(self, '_color_update_pending'):
                self._color_update_pending = False
            if not self._color_update_pending:
                self._color_update_pending = True
                QTimer.singleShot(0, self._do_apply_colorbar_settings)

    def _do_apply_colorbar_settings(self):
        self.apply_colorbar_settings_to_all(self.config)
        self._color_update_pending = False

    def update_histogram(self):
        #self.ax.clear()
        data = self.data.flatten()
        self.ax.hist(data, bins=100, log=True, color=self.hist_color)
        self.ax.set_xlabel('Intensity', fontsize = 8)
        self.ax.set_ylabel('Count', fontsize = 8)
        self.fig.tight_layout()
        self.fig.subplots_adjust(bottom=0.2) 
        self.canvas.draw_idle()
    

    def init_vertical_lines(self):
        data_min = np.nanmin(self.data)
        data_max = np.nanmax(self.data)
        try:        
            if self.current_settings["min_val"] is not None and self.current_settings["max_val"] is not None:
                min_val = self.current_settings["min_val"]
                max_val = self.current_settings["max_val"]
            else:
                min_val = float(self.intensity_min.text()) if self.intensity_min.text() else data_min
                max_val = float(self.intensity_max.text()) if self.intensity_max.text() else data_max

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
        

    def apply_colorbar_settings_to_all(self, config):
        if self.mode == ColorMode.INTEG:
            for cax, colorbar in zip(Common.integ_cax, Common.integ_colorbar):
                self.apply_colorbar_settings(cax, colorbar, config, caller=self)
        elif self.mode == ColorMode.MAIN:
            self.apply_colorbar_settings(Common.cax_xy, Common.colorbar_xy, config, caller=self)
        elif self.mode == ColorMode.CHANNEL:
            self.apply_colorbar_settings(Common.ch_cax, Common.ch_colorbar, config, caller=self)
        
        self.fits_viewer.canvas.draw()

        if self.subwindows is not None:
            for window, cax, colorbar in zip(self.subwindows, [Common.cax_xz, Common.cax_zy], [Common.colorbar_xz, Common.colorbar_zy]):
                if window:
                    self.apply_colorbar_settings(cax, colorbar, config, caller=self)
                    window.canvas.draw()
    
    @classmethod
    def apply_colorbar_settings(cls, cax, colorbar, config, caller = None):
        cax.tick_params(axis='y', which='both', left=config.get('colorbar_tick_left'), right=config.get('colorbar_tick_right'), 
                        labelleft=config.get('colorbar_tick_labelleft'), labelright=(not config.get('colorbar_tick_labelleft')),
                        width=config.get('colorbar_tick_width'), length=config.get('colorbar_tick_length'), 
                        color=config.get('colorbar_tick_color'), direction=config.get('colorbar_tick_direction'), 
                        labelcolor=config.get('colorbar_tick_labelcolor'))
    
        cax.tick_params(axis='x', which='both', top=config.get('colorbar_tick_top'), bottom=config.get('colorbar_tick_bottom'), 
                        labeltop=config.get('colorbar_tick_labeltop'), labelbottom=(not config.get('colorbar_tick_labeltop')),
                        width=config.get('colorbar_tick_width'), length=config.get('colorbar_tick_length'), 
                        color=config.get('colorbar_tick_color'), direction=config.get('colorbar_tick_direction'), 
                        labelcolor=config.get('colorbar_tick_labelcolor'))
    
        colorbar.outline.set_color(config.get('colorbar_tick_color'))
        colorbar.outline.set_linewidth(config.get('colorbar_tick_width'))
        colorbar.set_label(config.get('colorbar_label', 'Label'), fontsize=config.get('colorbar_label_fontsize'), color=config.get('colorbar_label_color'))
    
        colorbar.ax.minorticks_on()
        colorbar.ax.tick_params(which='minor', length=config.get('colorbar_mtick_length'), color=config.get('colorbar_tick_color'))
    
        if caller and hasattr(caller, 'log_checkbox') and caller.log_checkbox.isChecked():
            return
        colorbar.ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(config.get('colorbar_mtick_freq')))
        colorbar.ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(config.get('colorbar_mtick_freq')))

    
    def create_gamma_cmap(self, pattern):
        cmap = plt.get_cmap(pattern)
        if pattern in ['Rainbow', 'Cool']:
            custom_cmap = CustomColormap(pattern, cmap._segmentdata)
            return custom_cmap.apply_gamma(self.current_settings["gamma_value"])
        gamma_cmap = cmap(np.linspace(0, 1, cmap.N) ** self.current_settings["gamma_value"])
        self.apply_to_all_windows(lambda im: im.cmap.set_bad(color=self.config.get('bad_color', 'black')))
        return mpl.colors.ListedColormap(gamma_cmap)
    
    def update_gamma_from_spinbox(self):
        value = self.gamma_spinbox.value()
        self.current_settings["gamma_value"] = value
        self.apply_to_all_windows(lambda im: im.set_cmap(self.create_gamma_cmap(self.current_settings["color_pattern"])))
        self.apply_to_all_windows(lambda im: im.cmap.set_bad(color=self.config.get('bad_color', 'black')))


    
    def closeEvent(self,event):
        self.current_settings["color_pattern"] = self.colorscale_combo.currentText()
        ColorSettingsPanel.settings[self.mode] = self.current_settings
        self.colorscale_combo.currentIndexChanged.disconnect(self.change_color_scale)
        plt.close(self.fig)
        self.close()
        super().closeEvent(event)
        self.destroyed.emit()

        
    
