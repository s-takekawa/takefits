import numpy as np
import matplotlib as mpl
import warnings
import os
from typing import List, Optional
from PyQt6.QtWidgets import QWidget, QMainWindow, QDialog, QGridLayout, QGroupBox, QVBoxLayout, QComboBox, QLineEdit, QPushButton, QRadioButton, QCheckBox, QLabel, QButtonGroup, QMessageBox
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QGuiApplication
from core.coordinate import CoordinateConverter
from core.common import Common
from core.save_fits import SaveFITS
from core.region_manager import RegionManager
from core.contour_manager import ContourManager, ContourItem
from core.marker_manager import MarkerManager
from matplotlib.figure import Figure
from tools.color_scale import ColorSettingsPanel, ColorMode
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from ui.navigation_toolbar import MyNavigationToolbar
from ui.display_map import TransparentOverlayAxes
from core.coordinate import Format_pix_to_wcs
from logic.add_hpbw import AddHPBW
from astropy.io import fits
from astropy.wcs import WCS
import weakref

from core.region import CircleRegion, RectangleRegion, EllipseRegion, CubeRegion


class IntegSettingsPanel(QDialog):
    def __init__(self, fits_viewer, subwindows):
        super().__init__()
        self.filename = fits_viewer.filename
        self.fits_viewer = fits_viewer
        self.color_settings_panel = None
        self.wcs = self.fits_viewer.wcs
        self.subwindows = subwindows
        
        self.original_xlim = self.fits_viewer.ax.get_xlim()
        self.original_ylim = self.fits_viewer.ax.get_ylim()
        self.original_zlim = self.subwindows[0].ax.get_ylim()

        self.integ_result_windows = []
        self.coord_wrap = self.fits_viewer.config_manager.config.get('coord_wrap')
        self.initUI()
        
    def initUI(self):
        self.converter = CoordinateConverter(self.wcs, self.fits_viewer.config_manager.config)
    
        layout = QGridLayout()
        layout.setHorizontalSpacing(3)
        layout.setVerticalSpacing(3)
        layout.setContentsMargins(15, 0, 10, 13)
        
    
        mode_group_box = QGroupBox("Mode")
        mode_layout = QVBoxLayout()
    
        self.integration_radio = QRadioButton("Integration")
        self.integration_radio.setChecked(True)
        self.moment1_radio = QRadioButton("Moment 1")
        self.moment2_radio = QRadioButton("Moment 2")
    
        mode_layout.addWidget(self.integration_radio)
        mode_layout.addWidget(self.moment1_radio)
        mode_layout.addWidget(self.moment2_radio)

        # Group for radio buttons to manage mutual exclusion
        self.mode_radio_group = QButtonGroup(self)
        self.mode_radio_group.addButton(self.integration_radio)
        self.mode_radio_group.addButton(self.moment1_radio)
        self.mode_radio_group.addButton(self.moment2_radio)
        self.mode_radio_group.setExclusive(True)
        
        self.others_combo = QComboBox()
        self.others_combo.setPlaceholderText("Others")
        self.others_combo.addItems([
            "Average", 
            "Peak Int.", 
            "Peak Coord.", 
            "Median", 
            "RMS"
        ])
        
        self.others_combo.setFixedWidth(95)
        self.others_combo.setCurrentIndex(-1) # No selection by default
        mode_layout.addWidget(self.others_combo)
        mode_group_box.setLayout(mode_layout)
        layout.addWidget(mode_group_box, 0, 8, 4, 1)

        self.setLayout(layout)

        # Connect signals to slots for new logic
        self.mode_radio_group.buttonClicked.connect(self.on_radio_mode_selected)
        self.others_combo.currentIndexChanged.connect(self.on_combo_mode_selected)

        self.x_radio = QRadioButton()
        self.xr_label = QLabel('X:')
        self.x_min_input = QLineEdit(self)
        self.x_min_input.setPlaceholderText("X min")
        self.x_max_input = QLineEdit(self)
        self.x_max_input.setPlaceholderText("X max")
        self.x_min_input.setFixedWidth(90)
        self.x_max_input.setFixedWidth(90)

        layout.addWidget(self.x_radio, 0, 0, 1, 1)
        layout.addWidget(self.xr_label, 0, 1, 1, 1)
        layout.addWidget(self.x_min_input, 0, 2, 1, 3)
        layout.addWidget(self.x_max_input, 0, 5, 1, 3)


        self.y_radio = QRadioButton()
        self.yr_label = QLabel('Y:')
        self.y_min_input = QLineEdit(self)
        self.y_min_input.setPlaceholderText("Y min")
        self.y_max_input = QLineEdit(self)
        self.y_max_input.setPlaceholderText("Y max")
        self.y_min_input.setFixedWidth(90)
        self.y_max_input.setFixedWidth(90)
        

        layout.addWidget(self.y_radio, 1, 0, 1, 1)
        layout.addWidget(self.yr_label, 1, 1, 1, 1)
        layout.addWidget(self.y_min_input, 1, 2, 1, 3) 
        layout.addWidget(self.y_max_input, 1, 5, 1, 3) 

        self.z_radio = QRadioButton()
        self.zr_label = QLabel('Z:')
        self.z_min_input = QLineEdit(self)
        self.z_min_input.setPlaceholderText("Z min")
        self.z_max_input = QLineEdit(self)
        self.z_max_input.setPlaceholderText("Z max")
        self.z_min_input.setFixedWidth(90)
        self.z_max_input.setFixedWidth(90)


        layout.addWidget(self.z_radio, 2, 0, 1, 1)
        layout.addWidget(self.zr_label, 2, 1, 1, 1)
        layout.addWidget(self.z_min_input, 2, 2, 1, 3) 
        layout.addWidget(self.z_max_input, 2, 5, 1, 3)


        self.clip_checkbox = QCheckBox('', self)
        self.clip_checkbox.stateChanged.connect(self.toggle_clipping)
        self.clip_input = QLineEdit(self)
        self.clip_input.setPlaceholderText("Clipping")
        self.clip_input.setFixedWidth(60)
        self.clip_input.setEnabled(False)

        layout.addWidget(self.clip_checkbox, 3, 0, 1, 1)
        layout.addWidget(self.clip_input, 3, 1, 1, 2) 

        self.execute_button = QPushButton("Execute")
        self.execute_button.clicked.connect(self.execute_integration)
        layout.addWidget(self.execute_button, 3, 3, 1, 5) 
        self.execute_button.setAutoDefault(True) 
        self.execute_button.setDefault(True)

    
        self.radio_group = QButtonGroup(self)
        self.radio_group.addButton(self.x_radio, 0)
        self.radio_group.addButton(self.y_radio, 1)
        self.radio_group.addButton(self.z_radio, 2)
        self.radio_group.buttonClicked.connect(self.on_radio_button_clicked)
    
        self.z_radio.setChecked(True)
        self.z_min_input.setFocus()
    
        self.initialize_ranges()
    
        self.setLayout(layout)
        self.setWindowTitle(f'Integ Panel: {self.fits_viewer.filename}')
        self.move_to_default_position()

    def on_radio_mode_selected(self):
        # When a radio button is selected, clear the combo box selection.
        self.others_combo.blockSignals(True)
        self.others_combo.setCurrentIndex(-1)
        self.others_combo.blockSignals(False)

    def on_combo_mode_selected(self, index):
        # When a combo box item is selected, deselect any active radio button.
        if index > -1:
            # A valid item is selected, so deselect radio buttons
            # Temporarily disable mutual exclusion to uncheck the button
            self.mode_radio_group.setExclusive(False)
            checked_button = self.mode_radio_group.checkedButton()
            if checked_button:
                checked_button.setChecked(False)
            self.mode_radio_group.setExclusive(True)

    def initialize_ranges(self):
        if self.fits_viewer.data.ndim == 3:
            self.xmin_val = self.converter.pix_to_world(self.original_xlim[0], 0, 0)[0]
            self.xmax_val = self.converter.pix_to_world(self.original_xlim[1], 0, 0)[0]
            self.ymin_val = self.converter.pix_to_world(0, self.original_ylim[0], 0)[1]
            self.ymax_val = self.converter.pix_to_world(0, self.original_ylim[1], 0)[1]
            self.zmin_val = self.converter.pix_to_world(0, 0, self.original_zlim[0])[2]
            self.zmax_val = self.converter.pix_to_world(0, 0, self.original_zlim[1])[2]
        elif self.fits_viewer.data.ndim == 4:
            self.xmin_val = self.converter.pix_to_world(self.original_xlim[0], 0, 0, 0)[0]
            self.xmax_val = self.converter.pix_to_world(self.original_xlim[1], 0, 0, 0)[0]
            self.ymin_val = self.converter.pix_to_world(0, self.original_ylim[0], 0, 0)[1]
            self.ymax_val = self.converter.pix_to_world(0, self.original_ylim[1], 0, 0)[1]
            self.zmin_val = self.converter.pix_to_world(0, 0, self.original_zlim[0], 0)[2]
            self.zmax_val = self.converter.pix_to_world(0, 0, self.original_zlim[1], 0)[2]
            
        self.x_min_input.setText(str(self.xmin_val))
        self.x_max_input.setText(str(self.xmax_val))
        self.y_min_input.setText(str(self.ymin_val))
        self.y_max_input.setText(str(self.ymax_val))
        self.z_min_input.setText(str(self.zmin_val))
        self.z_max_input.setText(str(self.zmax_val))
    
    def on_radio_button_clicked(self, button):
        if button == self.x_radio:
            self.x_min_input.setFocus()
        elif button == self.y_radio:
            self.y_min_input.setFocus()
        elif button == self.z_radio:
            self.z_min_input.setFocus()
            

    def toggle_clipping(self):
        self.clip_input.setEnabled(self.clip_checkbox.isChecked())
        if self.clip_checkbox.isChecked():
            self.clip_input.setFocus()
        else:
            if self.radio_group.checkedId() == 0: self.x_min_input.setFocus()
            elif self.radio_group.checkedId() == 1: self.y_min_input.setFocus()
            elif self.radio_group.checkedId() == 2: self.z_min_input.setFocus()


    def _set_integ_data(self):
        self.data = self.fits_viewer.data.copy()
        if self.wcs.naxis == 4: self.data = self.data[0]
    
        self.znpix = self.data.shape[0]-1
        self.ynpix = self.data.shape[1]-1
        self.xnpix = self.data.shape[2]-1
        
        if self.z_radio.isChecked():
            self.plane = 'xy'
            if self.wcs.naxis == 4: self.integ_slice = ('x', 'y', 0, 0)
            elif self.wcs.naxis == 3: self.integ_slice = ('x', 'y', 0)
            self.integ_axis = 0
        elif self.y_radio.isChecked():
            self.plane = 'xz'
            if self.wcs.naxis == 4: self.integ_slice = ('x', 0, 'y', 0)
            elif self.wcs.naxis == 3: self.integ_slice = ('x', 0, 'y')
            self.integ_axis = 1
        elif self.x_radio.isChecked():
            self.plane = 'zy'
            if self.wcs.naxis == 4: self.integ_slice = (0, 'y', 'x', 0)
            elif self.wcs.naxis == 3: self.integ_slice = (0, 'y', 'x')
            self.integ_axis = 2
        
        if self.clip_checkbox.isChecked():
            try:
                self.clip = float(self.clip_input.text())
                self.data[self.data < self.clip] = np.nan
            except ValueError:
                QMessageBox.warning(self, 'Error', 'Invalid clipping value provided!')
                return

    def execute_integration(self):
        self._set_integ_data()
        
        # Check which mode is active based on combo box or radio buttons
        if self.others_combo.currentIndex() > -1:
            # Combo box is selected
            mode_text = self.others_combo.currentText()
            if mode_text == "Average":
                self.integ_mode = 'average'
                self.integrated_data = self.average_fits_along_axis(self.data, axis=self.integ_axis)
            elif mode_text == "Peak Int.":
                self.integ_mode = 'peak_int'
                self.integrated_data = self.peak_int_fits_along_axis(self.data, axis=self.integ_axis)
            elif mode_text == "Peak Coord.":
                self.integ_mode = 'peak_corrd'
                self.integrated_data = self.peak_coord_fits_along_axis(self.data, axis=self.integ_axis)
            elif mode_text == "Median":
                self.integ_mode = 'median_int'
                self.integrated_data = self.median_int_fits_along_axis(self.data, axis=self.integ_axis)
            elif mode_text == "RMS":
                self.integ_mode = 'rms'
                self.integrated_data = self.rms_fits_along_axis(self.data, axis=self.integ_axis)
        
        elif self.mode_radio_group.checkedButton() is not None:
            # A radio button is selected
            if self.integration_radio.isChecked():
                self.integ_mode = 'int'
                self.integrated_data = self.integrate_fits_along_axis(self.data, axis=self.integ_axis)
            elif self.moment1_radio.isChecked():
                self.integ_mode = 'mom1'
                self.integrated_data = self.moment1_fits_along_axis(self.data, axis=self.integ_axis)
            elif self.moment2_radio.isChecked():
                self.integ_mode = 'mom2'
                self.integrated_data = self.moment2_fits_along_axis(self.data, axis=self.integ_axis)
        else:
            QMessageBox.warning(self, 'Mode Error', 'Please select an integration mode.')
            return


    def show_in_new_window(self,window_title):
        try:
            if np.all(np.isnan(self.integrated_data)):
                raise ValueError
        except (ValueError, AttributeError):
            QMessageBox.warning(self, 'Error', 'All pixel values are NaN or data is invalid!')
            return
            
        new_window = IntegResultWindow(
            integrated_data=self.integrated_data,
            plane=self.plane,
            slice=self.integ_slice,
            fits_viewer=self.fits_viewer,
            subwindows=self.subwindows,
            window_title=window_title,
            config=self.fits_viewer.config_manager.config,
            data = self.data,
            wcs = self.wcs,
            mode = self.integ_mode
        )
        new_window.integ_axis = getattr(self, 'integ_axis', None)
        self.fits_viewer.integ_result_windows.append(weakref.ref(new_window))
        self.integ_result_windows.append(new_window)
        new_window.show()
        new_window.destroyed.connect(lambda: self.remove_window_reference(new_window))

    def remove_window_reference(self, window):
        if window in self.integ_result_windows:
            self.integ_result_windows.remove(window)

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
        
    def _get_integ_range(self, axis):
        x_min = self.x_min_input.text()
        x_max = self.x_max_input.text()
        y_min = self.y_min_input.text()
        y_max = self.y_max_input.text()
        z_min = self.z_min_input.text()
        z_max = self.z_max_input.text()

        try:
            if self.fits_viewer.data.ndim == 3:
                if axis == 2:
                    min_pixel_float = float(self.converter.world_to_pix(x_min,y_min, z_min)[0])
                    max_pixel_float = float(self.converter.world_to_pix(x_max,y_min, z_min)[0])
                    self.min_input, self.max_input = x_min, x_max
                elif axis == 1:
                    min_pixel_float = float(self.converter.world_to_pix(x_min,y_min, z_min)[1])
                    max_pixel_float = float(self.converter.world_to_pix(x_min,y_max, z_min)[1])
                    self.min_input, self.max_input = y_min, y_max
                elif axis == 0:
                    min_pixel_float = float(self.converter.world_to_pix(x_min,y_min, z_min)[2])
                    max_pixel_float = float(self.converter.world_to_pix(x_min,y_min, z_max)[2])
                    self.min_input, self.max_input = z_min, z_max
            elif self.fits_viewer.data.ndim == 4:
                if axis == 2:
                    min_pixel_float = float(self.converter.world_to_pix(x_min,y_min, z_min,0)[0])
                    max_pixel_float = float(self.converter.world_to_pix(x_max,y_min, z_min,0)[0])
                    self.min_input, self.max_input = x_min, x_max
                elif axis == 1:
                    min_pixel_float = float(self.converter.world_to_pix(x_min,y_min, z_min,0)[1])
                    max_pixel_float = float(self.converter.world_to_pix(x_min,y_max, z_min,0)[1])
                    self.min_input, self.max_input = y_min, y_max
                elif axis == 0:
                    min_pixel_float = float(self.converter.world_to_pix(x_min,y_min, z_min,0)[2])
                    max_pixel_float = float(self.converter.world_to_pix(x_min,y_min, z_max,0)[2])
                    self.min_input, self.max_input = z_min, z_max
        except ValueError:
            QMessageBox.warning(self, 'Invalid Input', 'Please enter valid numeric values!')
            return
        
        if min_pixel_float > max_pixel_float:
            min_pixel_float, max_pixel_float = max_pixel_float, min_pixel_float
        if max_pixel_float == self.data.shape[axis] - 0.5: max_pixel_float -= 0.00001

        min_pixel = int(round(min_pixel_float))
        max_pixel = int(round(max_pixel_float))
        if min_pixel > max_pixel: max_pixel, min_pixel = min_pixel, max_pixel
        min_fraction = min_pixel - min_pixel_float - 0.5
        max_fraction = max_pixel_float - max_pixel + 0.5


        if min_pixel < 0:
            min_pixel = 0
            if axis == 2:
                if self.wcs.naxis == 4:
                    self.xmin_val = self.converter.pix_to_world(-0.5, 0, 0, 0)[0]
                elif self.wcs.naxis == 3:
                    self.xmin_val = self.converter.pix_to_world(-0.5, 0, 0)[0]
                self.x_min_input.setText(str(self.xmin_val))
                self.min_input = self.xmin_val
            elif axis == 1: # y-axis
                if self.wcs.naxis == 4:
                    self.ymin_val = self.converter.pix_to_world(0, -0.5, 0, 0)[1]
                elif self.wcs.naxis == 3:
                    self.ymin_val = self.converter.pix_to_world(0, -0.5, 0)[1]
                self.y_min_input.setText(str(self.ymin_val))
                self.min_input = self.ymin_val
            elif axis == 0: # z-axis
                if self.wcs.naxis == 4:
                    self.zmin_val = self.converter.pix_to_world(0, 0, -0.5, 0)[2]
                elif self.wcs.naxis == 3:
                    self.zmin_val = self.converter.pix_to_world(0, 0, -0.5)[2]
                self.z_min_input.setText(str(self.zmin_val))
                self.min_input = self.zmin_val
        
        if max_pixel >= self.data.shape[axis] or max_pixel < 0:
            max_pixel = self.data.shape[axis] - 1
            if axis == 2: # x-axis
                if self.wcs.naxis == 4:
                    self.xmax_val = self.converter.pix_to_world(self.xnpix+0.5, 0, 0, 0)[0]
                elif self.wcs.naxis == 3:
                    self.xmax_val = self.converter.pix_to_world(self.xnpix+0.5, 0, 0)[0]
                self.x_max_input.setText(str(self.xmax_val))
                self.max_input = self.xmax_val
            elif axis == 1: # y-axis
                if self.wcs.naxis == 4:
                    self.ymax_val = self.converter.pix_to_world(0, self.ynpix+0.5, 0, 0)[1]
                elif self.wcs.naxis == 3:
                    self.ymax_val = self.converter.pix_to_world(0, self.ynpix+0.5, 0)[1]
                self.y_max_input.setText(str(self.ymax_val))
                self.max_input = self.ymax_val
            elif axis == 0: # z-axis
                if self.wcs.naxis == 4:
                    self.zmax_val = self.converter.pix_to_world(0, 0, self.znpix+0.5, 0)[2]
                elif self.wcs.naxis == 3:
                    self.zmax_val = self.converter.pix_to_world(0, 0, self.znpix+0.5)[2]
                self.z_max_input.setText(str(self.zmax_val))
                self.max_input = self.zmax_val
        return min_pixel, max_pixel, min_fraction, max_fraction


    def integrate_fits_along_axis(self, data, axis):
        try:
            min_pixel, max_pixel, min_fraction, max_fraction = self._get_integ_range(axis)
        except: return
            
        sliced_data = np.take(data, indices=range(min_pixel, max_pixel), axis=axis)
        total_sum = self.custom_nansum(sliced_data, axis=axis)

        first_pixel_value = np.take(data, indices=[min_pixel], axis=axis)
        total_sum = self.nan_sum(total_sum, np.squeeze(first_pixel_value, axis=axis) * min_fraction)
        last_pixel_value = np.take(data, indices=[max_pixel], axis=axis)
        total_sum = self.nan_sum(total_sum, np.squeeze(last_pixel_value, axis=axis) * max_fraction)
        self.integrated_data = total_sum * abs(self.wcs.wcs.cdelt[2 - axis])

        self.show_in_new_window(f"Integration: {self.min_input} to {self.max_input}")
    
        return self.integrated_data


    def moment1_fits_along_axis(self, data, axis):
        try:
            min_pixel, max_pixel, min_fraction, max_fraction = self._get_integ_range(axis)
        except:
            return

        indices = np.arange(min_pixel, max_pixel + 1)
    
        slices = [slice(None)] * data.ndim
        slices[axis] = slice(min_pixel, max_pixel + 1)
        sliced_data = data[tuple(slices)]
    
        data_to_wcs_axis = {0: 2, 1: 1, 2: 0}
        wcs_axis = data_to_wcs_axis.get(axis, axis)
    
        num_pixels = len(indices)
        num_wcs_axes = self.wcs.naxis
        pixel_coords = np.zeros((num_pixels, num_wcs_axes))
    
        pixel_coords[:, wcs_axis] = indices
    
        for i in range(num_wcs_axes):
            if i != wcs_axis:
                pixel_coords[:, i] = self.wcs.wcs.crpix[i] - 1
    
        world_coords = self.wcs.wcs_pix2world(pixel_coords, 0)

        world_coords_axis = world_coords[:, wcs_axis]
    
        world_min = np.nanmin(world_coords_axis)
        world_max = np.nanmax(world_coords_axis)
    
        if world_min > world_max:
            world_min, world_max = world_max, world_min

        shape = [1] * data.ndim
        shape[axis] = -1
        world_coords_axis = world_coords_axis.reshape(shape)
    
        weighted_sum = self.custom_nansum(sliced_data * world_coords_axis, axis=axis)
        intensity_sum = self.custom_nansum(sliced_data, axis=axis)
    
        with np.errstate(divide='ignore', invalid='ignore'):
            moment1_map = weighted_sum / intensity_sum
    
        outside_range = (moment1_map < world_min) | (moment1_map > world_max)
        moment1_map[outside_range] = np.nan
    
        self.integrated_data = moment1_map
        self.show_in_new_window(f"Moment 1: {self.min_input} to {self.max_input}")
    
        return self.integrated_data


    def moment2_fits_along_axis(self, data, axis):
        try:
            min_pixel, max_pixel, min_fraction, max_fraction = self._get_integ_range(axis)
        except Exception as e:
            print(f"Error getting integration range: {e}")
            return

        indices = np.arange(min_pixel, max_pixel + 1)

        slices = [slice(None)] * data.ndim
        slices[axis] = slice(min_pixel, max_pixel + 1)
        sliced_data = data[tuple(slices)]

        data_to_wcs_axis = {0: 2, 1: 1, 2: 0}
        wcs_axis = data_to_wcs_axis.get(axis, axis)

        num_pixels = len(indices)
        num_wcs_axes = self.wcs.naxis
        pixel_coords = np.zeros((num_pixels, num_wcs_axes))

        pixel_coords[:, wcs_axis] = indices

        for i in range(num_wcs_axes):
            if i != wcs_axis:
                pixel_coords[:, i] = self.wcs.wcs.crpix[i] - 1

        world_coords = self.wcs.wcs_pix2world(pixel_coords, 0)

        world_coords_axis = world_coords[:, wcs_axis]

        world_min = np.nanmin(world_coords_axis)
        world_max = np.nanmax(world_coords_axis)

        if world_min > world_max:
            world_min, world_max = world_max, world_min

        shape = [1] * data.ndim
        shape[axis] = -1
        world_coords_axis = world_coords_axis.reshape(shape)

        intensity_sum = self.custom_nansum(sliced_data, axis=axis)

        weighted_sum = self.custom_nansum(sliced_data * world_coords_axis, axis=axis)
        with np.errstate(divide='ignore', invalid='ignore'):
            moment1_map = weighted_sum / intensity_sum

        weighted_sum_sq = self.custom_nansum(sliced_data * world_coords_axis**2, axis=axis)
        with np.errstate(divide='ignore', invalid='ignore'):
            moment2_map = np.sqrt((weighted_sum_sq / intensity_sum) - moment1_map**2)

        outside_range = (moment2_map < 0) | (moment2_map > (world_max - world_min))
        moment2_map[outside_range] = np.nan

        self.integrated_data = moment2_map
        self.show_in_new_window(f"Moment 2: {self.min_input} to {self.max_input}")

        return self.integrated_data
        
        

    def average_fits_along_axis(self, data, axis):
        try:
            min_pixel, max_pixel, min_fraction, max_fraction = self._get_integ_range(axis)
        except: return
            
        sliced_data = np.take(data, indices=range(min_pixel, max_pixel), axis=axis)
        total_sum = self.custom_nansum(sliced_data, axis=axis)

        first_pixel_value = np.take(data, indices=[min_pixel], axis=axis)
        total_sum = self.nan_sum(total_sum, np.squeeze(first_pixel_value, axis=axis) * (1-min_fraction))
        
        last_pixel_value = np.take(data, indices=[max_pixel], axis=axis)
        total_sum = self.nan_sum(total_sum, np.squeeze(last_pixel_value, axis=axis) * max_fraction)
        
        pix_number = max_fraction + max_pixel - min_pixel - min_fraction + 1.
        
        with np.errstate(divide='ignore', invalid='ignore'):
            averaged_map = total_sum / pix_number
        self.integrated_data = averaged_map
        self.show_in_new_window(f"Average: {self.min_input} to {self.max_input}")
    
        return self.integrated_data
        

    def peak_int_fits_along_axis(self, data, axis):
        try:
            min_pixel, max_pixel, min_fraction, max_fraction = self._get_integ_range(axis)
        except: return
            
        sliced_data = np.take(data, indices=range(min_pixel, max_pixel), axis=axis)
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            peak_map = np.nanmax(sliced_data, axis=axis)
            
        self.integrated_data = peak_map
        self.show_in_new_window(f"Peak Int.: {self.min_input} to {self.max_input}")
    
        return self.integrated_data



    def peak_coord_fits_along_axis(self, data, axis):
        try:
            min_pixel, max_pixel, min_fraction, max_fraction = self._get_integ_range(axis)
        except: return

        indices = np.arange(min_pixel, max_pixel + 1)

        slices = [slice(None)] * data.ndim
        slices[axis] = slice(min_pixel, max_pixel + 1)
        sliced_data = data[tuple(slices)]

        data_to_wcs_axis = {0: 2, 1: 1, 2: 0}
        wcs_axis = data_to_wcs_axis.get(axis, axis)

        num_pixels = len(indices)
        num_wcs_axes = self.wcs.naxis
        pixel_coords = np.zeros((num_pixels, num_wcs_axes))

        pixel_coords[:, wcs_axis] = indices

        for i in range(num_wcs_axes):
            if i != wcs_axis:
                pixel_coords[:, i] = self.wcs.wcs.crpix[i] - 1

        try:
            world_coords = self.wcs.wcs_pix2world(pixel_coords, 0)
        except Exception as e:
            print(f"Error in wcs_pix2world: {e}")
            return

        world_coords_axis = world_coords[:, wcs_axis]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            max_vals = np.nanmax(sliced_data, axis=axis)
            all_nan_mask = np.isnan(max_vals)

        peak_indices = np.full(max_vals.shape, np.nan)

        # Reshape sliced_data to (N, axis_dim)
        # where N is the number of valid (non-NaN) slices
        if axis == 0:
            # sliced_data.shape = (Z, Y, X)
            # Reshape to (Y*X, Z)
            reshaped_data = sliced_data.reshape(sliced_data.shape[0], -1).T  # (Y*X, Z)
        elif axis == 1:
            # sliced_data.shape = (Y, Z, X)
            # Reshape to (Y*X, Z)
            reshaped_data = sliced_data.transpose(0, 2, 1).reshape(-1, sliced_data.shape[1])  # (Y*X, Z)
        elif axis == 2:
            # sliced_data.shape = (Y, X, Z)
            # Reshape to (Y*X, Z)
            reshaped_data = sliced_data.reshape(-1, sliced_data.shape[2])  # (Y*X, Z)
        else:
            print(f"Unsupported axis: {axis}")
            return

        mask_flat = all_nan_mask.flatten()

        peak_indices_flat = np.full(reshaped_data.shape[0], np.nan)

        valid_rows = ~mask_flat

        if np.any(valid_rows):
            peak_indices_flat[valid_rows] = np.nanargmax(reshaped_data[valid_rows], axis=1)

        YX_shape = max_vals.shape
        peak_indices = peak_indices_flat.reshape(YX_shape)

        peak_world_map = np.full(YX_shape, np.nan, dtype=float)

        valid_mask_flat = valid_rows
        peak_indices_flat = peak_indices_flat

        valid_peak_indices = peak_indices_flat[valid_mask_flat].astype(int)
        
        peak_world_coords = world_coords_axis[valid_peak_indices]

        peak_world_map_flat = peak_world_map.flatten()
        peak_world_map_flat[valid_mask_flat] = peak_world_coords
        peak_world_map = peak_world_map_flat.reshape(YX_shape)
        
        if axis == 2:
            if self.coord_wrap == 180:
                peak_world_map[peak_world_map < -180] += 360
                peak_world_map[peak_world_map > 180] -= 360
            elif self.coord_wrap == 360:
                peak_world_map[peak_world_map < 0] += 360
                peak_world_map[peak_world_map > 360] -= 360
                
        self.integrated_data = peak_world_map
        self.show_in_new_window(f"Peak Coord.: {self.min_input} to {self.max_input}")

        return self.integrated_data


    def median_int_fits_along_axis(self, data, axis):
        try:
            min_pixel, max_pixel, min_fraction, max_fraction = self._get_integ_range(axis)
        except: return
            
        sliced_data = np.take(data, indices=range(min_pixel, max_pixel), axis=axis)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median_map = np.nanmedian(sliced_data, axis=axis)

        
        self.integrated_data = median_map
        self.show_in_new_window(f"Median: {self.min_input} to {self.max_input}")
    
        return self.integrated_data

    def rms_fits_along_axis(self, data, axis):
        try:
            min_pixel, max_pixel, min_fraction, max_fraction = self._get_integ_range(axis)
        except: return
            
        sliced_data = np.take(data, indices=range(min_pixel, max_pixel), axis=axis)
        
        squared = np.square(sliced_data)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_squared = np.nanmean(squared, axis=axis)
            mean_squared_all = np.nanmean(mean_squared)
        rms_map = np.sqrt(mean_squared)
        rms_value = np.sqrt(mean_squared_all)
        if 'BUNIT' in self.fits_viewer.header:
            bunit = self.fits_viewer.header['BUNIT']
        else: bunit = ''
        rms_message = f"\n\nRMS from {self.min_input} to {self.max_input} = {rms_value:.3g} {bunit}"
        if self.clip_checkbox.isChecked():
            clip_level = self.clip_input.text()
            rms_message += f'\n(using values > {clip_level} {bunit})'
        print(rms_message)

        self.integrated_data = rms_map
        self.show_in_new_window(f"RMS: {self.min_input} to {self.max_input}")

        return self.integrated_data
        
    def move_to_default_position(self):
        # Get MainWindow geometry
        mainwindow_geometry = self.fits_viewer.geometry()
        mainwindow_x = mainwindow_geometry.x()
        mainwindow_y = mainwindow_geometry.y()
        mainwindow_width = mainwindow_geometry.width()

        # Move ControlPanel to the right of MainWindow
        self.move(mainwindow_x + mainwindow_width, mainwindow_y - 28)


    def closeEvent(self, event):
        super().closeEvent(event)
        self.destroyed.emit()


class IntegResultWindow(QMainWindow):
    def __init__(self, data, wcs, integrated_data, plane, slice, fits_viewer, subwindows, window_title, config, mode, parent=None):
        super().__init__(parent)
        self.integrated_data = integrated_data
        self.plane = plane
        self.fits_viewer = fits_viewer
        self.subwindows = subwindows
        self.color_settings_panel = None
        self.integ_slice = slice
        self.data = data
        self.wcs = wcs
        self.integ_mode = mode
        self.config = config
        self.converter = CoordinateConverter(self.wcs, config)
        self.decimal =  config.get('decimal')
        self.number_decimals = config.get('number_decimals')
        self.coord_wrap = config.get('coord_wrap')
        base_filename = getattr(self.fits_viewer, 'filename', 'takefits.fits')
        base_root = os.path.splitext(base_filename)[0]
        mode_suffix = self.integ_mode if isinstance(self.integ_mode, str) else 'integ'
        self.filename = f"{base_root}.{mode_suffix}.fits"
        self.config_manager = getattr(self.fits_viewer, 'config_manager', None)
        self.setWindowTitle(window_title)
        self.original_window_title = window_title
        self.region_mode_enabled = self.fits_viewer.region_mode_enabled
        self.dragging = False
        self.label = QLabel(self) # For intensity value
        self._contour_layer_id: Optional[str] = None
        self._contour_title_connected = False
        self.marker_manager = MarkerManager(self)
        self.marker_manager.set_active_plane(self.plane)
        self.marker_panel = None
        self.marker_mode_enabled = False

        self.color_pattern = (
            ColorSettingsPanel.settings[ColorMode.MAIN]['color_pattern'] or 
            self.fits_viewer.displaymap.config.get('colorscale')
        )
        if ColorSettingsPanel.settings[ColorMode.INTEG]['color_pattern']:
            self.color_pattern = ColorSettingsPanel.settings[ColorMode.INTEG]['color_pattern']
            
        self.original_xlim = self.fits_viewer.ax.get_xlim()
        self.original_ylim = self.fits_viewer.ax.get_ylim()
        self.original_zlim = self.subwindows[0].ax.get_ylim()
        self.initialize_ranges()
        self.znpix = self.data.shape[0]-1
        self.ynpix = self.data.shape[1]-1
        self.xnpix = self.data.shape[2]-1
        
        self.initUI(config)
        self._set_bunit()

        self.region_manager = RegionManager(self)
        try:
            self.region_manager.selected_region_changed.connect(self._on_region_selection_changed)
        except Exception:
            pass
        self.cutout_dialog = None
        
        if hasattr(self.fits_viewer, 'region_manager'):
            for region in self.fits_viewer.region_manager.regions:
                new_region = None
                if isinstance(region, CubeRegion):
                    new_region = RectangleRegion(
                        xy=region.xy,
                        width=region.width,
                        height=region.height
                    )
                    new_region.set_angle(getattr(region, 'angle', 0.0))
                elif isinstance(region, RectangleRegion):
                    new_region = RectangleRegion(xy=region.xy, width=region.width, height=region.height)
                    new_region.set_angle(getattr(region, 'angle', 0.0))
                elif isinstance(region, EllipseRegion):
                    new_region = EllipseRegion(center=region.center, width=region.width, height=region.height)
                    new_region.set_angle(getattr(region, 'angle', 0.0))
                elif isinstance(region, CircleRegion):
                    new_region = CircleRegion(center=region.center, radius=region.radius)
                
                if new_region:
                    new_region.region_id = region.region_id
                    if hasattr(region, 'label_text') and region.label_text:
                        new_region.set_label_text(region.label_text)
                    
                    new_region.add_to_axes(self.overlay_ax)
                    self.region_manager.regions.append(new_region)
                    
                    if self.fits_viewer.region_manager.selected_region is region:
                        self.region_manager.select_region(new_region)

        self.region_mode_enabled = self.fits_viewer.region_mode_enabled
        current_shape = self.fits_viewer.region_manager.region_mode
        if self.region_mode_enabled and current_shape:
            shape_for_integ = 'rectangle' if current_shape == 'cube' else current_shape
            self.region_manager.set_region_mode(shape_for_integ)
            self.setWindowTitle(f"[REGION MODE: {shape_for_integ.upper()}] {self.original_window_title}")

        self._register_contour_layer()


    def open_cutout_dialog(self, region=None, use_view_bounds=False):
        from tools.cutout import CutoutSettingsDialog

        dialog = self.cutout_dialog
        roles = self.get_axis_roles()
        collapsed_axes = [idx for idx, role in enumerate(roles) if role == 'collapsed']
        axis_mapping = self._compute_cutout_axis_mapping()
        if dialog is None:
            header = self.wcs.to_header()
            header['NAXIS'] = self.integrated_data.ndim
            for axis, size in enumerate(reversed(self.integrated_data.shape), start=1):
                header[f'NAXIS{axis}'] = size
            if hasattr(self, 'bunit') and self.bunit:
                header['BUNIT'] = self.bunit

            dialog = CutoutSettingsDialog(
                self,
                region if region is not None else None,
                self,
                data_override=self.integrated_data,
                header_override=header,
                wcs_override=self.wcs,
                dialog_title=f"Cut Out ({self.windowTitle()})",
                collapsed_axes=collapsed_axes,
                wcs_to_data_axis=axis_mapping,
            )
            dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
            dialog.destroyed.connect(lambda *_: setattr(self, 'cutout_dialog', None))
            self.cutout_dialog = dialog
        else:
            dialog.collapsed_axes.update(collapsed_axes)
            dialog._wcs_to_data_axis = dialog._initialize_axis_mapping(axis_mapping)
            dialog._data_to_wcs_axis = {
                data_axis: wcs_axis
                for wcs_axis, data_axis in enumerate(dialog._wcs_to_data_axis)
                if data_axis is not None
            }

        if region is not None:
            dialog.reset_region(region)
        elif use_view_bounds or getattr(dialog, 'region', None) is None:
            dialog.reset_to_view()

        dialog.show()
        dialog.raise_()
        dialog.activateWindow()


    def _on_region_selection_changed(self, region):
        if self.cutout_dialog is None:
            return
        if region is not None:
            self.cutout_dialog.reset_region(region)
        else:
            self.cutout_dialog.reset_to_view()

    def get_axis_roles(self):
        if self.wcs is None:
            return []
        naxis = self.wcs.naxis
        roles = ['depth'] * naxis
        collapsed_idx = None
        if hasattr(self, 'integ_axis') and self.integ_axis is not None:
            collapsed_idx = naxis - self.integ_axis - 1 if naxis > self.integ_axis else None

        plane = getattr(self, 'plane', 'xy')
        candidate_order = []
        if plane == 'xy':
            candidate_order = [0, 1]
        elif plane == 'xz':
            candidate_order = [0, 2]
        elif plane == 'zy':
            candidate_order = [2, 1]
        else:
            candidate_order = list(range(naxis))

        if collapsed_idx is not None and 0 <= collapsed_idx < naxis:
            roles[collapsed_idx] = 'collapsed'

        display_names = ['display_x', 'display_y']
        di = 0
        for idx in candidate_order:
            if di >= len(display_names):
                break
            if idx == collapsed_idx or idx < 0 or idx >= naxis:
                continue
            roles[idx] = display_names[di]
            di += 1

        for idx in range(naxis):
            if roles[idx] not in ('display_x', 'display_y', 'collapsed'):
                roles[idx] = 'depth'

        return roles

    def _compute_cutout_axis_mapping(self):
        if self.wcs is None:
            return []

        naxis = self.wcs.naxis
        data_ndim = getattr(self.integrated_data, 'ndim', 0)
        original_ndim = getattr(self.data, 'ndim', data_ndim)
        if original_ndim <= 0:
            original_ndim = data_ndim

        base_mapping: List[Optional[int]] = []
        for axis in range(naxis):
            data_axis = original_ndim - axis - 1
            if 0 <= data_axis < original_ndim:
                base_mapping.append(data_axis)
            else:
                base_mapping.append(None)

        collapsed_axis = getattr(self, 'integ_axis', None)
        if collapsed_axis is None or data_ndim == original_ndim:
            adjusted: List[Optional[int]] = []
            for mapped in base_mapping:
                if mapped is None or mapped >= data_ndim:
                    adjusted.append(None)
                else:
                    adjusted.append(mapped)
            return adjusted

        adjusted: List[Optional[int]] = []
        for mapped in base_mapping:
            if mapped is None:
                adjusted.append(None)
            elif mapped == collapsed_axis:
                adjusted.append(None)
            elif mapped > collapsed_axis:
                adjusted.append(mapped - 1)
            else:
                adjusted.append(mapped)

        final: List[Optional[int]] = []
        for mapped in adjusted:
            if mapped is None or mapped >= data_ndim:
                final.append(None)
            else:
                final.append(mapped)
        return final

    def current_z_pixel_bounds(self):
        return (0, 1)


    def _set_bunit(self):
            # Set the correct brightness unit based on the integration mode.
            import re
            original_header = self.fits_viewer.header
            original_bunit = original_header.get('BUNIT', '')
            
            plane_to_axis = {'xy': 2, 'xz': 1, 'zy': 0}
            axis_to_drop = plane_to_axis.get(self.plane)
            
            full_axis_label = ''
            display_axis_unit = ''

            # Get the full text label from the corresponding displayed axis
            # and parse the unit from it. This is the most reliable method.
            try:
                if axis_to_drop == 2: # Z-axis (e.g., Velocity)
                    # Get label from the vertical axis of the XZ-plane subwindow
                    full_axis_label = self.subwindows[0].ax.get_ylabel()
                elif axis_to_drop == 1: # Y-axis (e.g., Declination)
                    # Get label from the vertical axis of the XY-plane main window
                    full_axis_label = self.fits_viewer.ax.get_ylabel()
                elif axis_to_drop == 0: # X-axis (e.g., Right Ascension)
                    # Get label from the horizontal axis of the XY-plane main window
                    full_axis_label = self.fits_viewer.ax.get_xlabel()

                # Use regex to find text within square brackets, e.g., "LSR Velocity [km/s]" -> "km/s"
                match = re.search(r'\[(.*?)\]', full_axis_label)
                if match:
                    display_axis_unit = match.group(1)
                else:
                    # Fallback for labels without brackets
                    display_axis_unit = self.wcs.wcs.cunit[axis_to_drop].to_string()

            except Exception:
                # General fallback if getting the label fails
                display_axis_unit = self.wcs.wcs.cunit[axis_to_drop].to_string()

            # Clean up the unit string by removing spaces
            display_axis_unit = display_axis_unit.replace(' ', '')

            if self.integ_mode == 'int':
                self.bunit = f"{original_bunit} {display_axis_unit}"
            elif self.integ_mode in ['mom1', 'mom2', 'peak_corrd']:
                self.bunit = display_axis_unit
            else: # average, peak_int, median, rms
                self.bunit = original_bunit

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

    def formatter(self, x, y):
        xstr, ystr = self.format_pix.convert(self.plane, x, y)
        xstr = ("{:>.%ds}" % (self.number_decimals+6)).format(xstr)
        ystr = ("{:>.%ds}" % (self.number_decimals+6)).format(ystr)
        if self.plane == 'xy': return f'x={xstr}, y={ystr}'
        elif  self.plane == 'xz': return f'x={xstr}, z={ystr}'
        elif  self.plane == 'zy': return f'z={xstr}, y={ystr}'


    def initUI(self, config):
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self._overlay_updates_enabled = True

        self.ax = self.fig.add_subplot(111, projection=self.fits_viewer.wcs, slices=self.integ_slice)
        self.resize(config.get('figure_width'), config.get('figure_height'))
        self.format_pix = Format_pix_to_wcs(self.wcs, self.integ_slice, self.ax, self.plane, self.decimal, self.number_decimals, self.coord_wrap)
        self.ax.format_coord = self.formatter
        self.fig.subplots_adjust( left = config.get('ax_pos_l'), 
                                right = config.get('ax_pos_r'),
                                bottom = config.get('ax_pos_b'), 
                                top = config.get('ax_pos_t'))
        
        if self.plane == 'zy': im_data = self.integrated_data.T
        else: im_data = self.integrated_data
        
        if self.plane == 'xy': aspect = 'equal'
        else: aspect = 'auto'
        self.im = self.ax.imshow(im_data, aspect = aspect, origin='lower', cmap=self.color_pattern)

        # Overlay axis for efficient cursor/region drawing
        self.overlay_ax = self.fig.add_axes(self.ax.get_position(), sharex=self.ax, sharey=self.ax, frameon=False)
        self.overlay_ax.__class__ = TransparentOverlayAxes
        self.overlay_ax.patch.set_alpha(0)
        self.overlay_ax.set_zorder(self.ax.get_zorder() + 1)
        self.overlay_ax.set_xticks([])
        self.overlay_ax.set_yticks([])
        self.overlay_ax.set_navigate(False)
        self._background = None
        self._updating_overlay = False
        if self.plane == 'xy':
            self.overlay_ax_xy = self.overlay_ax
            self.ax_xy = self.ax
        elif self.plane == 'xz':
            self.overlay_ax_xz = self.overlay_ax
            self.ax_xz = self.ax
        elif self.plane == 'zy':
            self.overlay_ax_zy = self.overlay_ax
            self.ax_zy = self.ax
            
        axis_unit = []
        axis_type = []
        for i in range(self.data.ndim): 
            axis_unit.append(self.ax.coords[i].get_format_unit())
            if self.wcs.world_axis_physical_types[i] is None: axis_type.append(None)
            else: axis_type.append(self.wcs.world_axis_physical_types[i].split('.')[-1])
        if 'glon' in self.ax.coords:
            self.ax.coords['glon'].set_coord_type(coord_wrap = config.get('coord_wrap'), coord_type = 'longitude')
        axis_format_decimal = np.isin(axis_type, ['lon', 'lat'])
        if config.get('decimal') == False: axis_format_decimal = [False for _ in axis_format_decimal]
        
        for idx in np.where(axis_format_decimal)[0]:
            self.ax.coords[idx].set_format_unit(axis_unit[idx] , decimal=axis_format_decimal[idx])

        disp_idx = tuple(True if i else False for i in self.integ_slice)
        for idx in np.where(np.logical_not(disp_idx))[0]:
            self.ax.coords[idx].set_ticklabel_visible(False)
            self.ax.coords[idx].set_ticks_visible(False) 
            

        for idx in np.where(disp_idx)[0]:
            self.ax.coords[idx].set_ticks_position(config.get('default_ticks_position'))
            self.ax.coords[idx].set_ticklabel(exclude_overlapping=True)
            self.ax.coords[idx].set_ticklabel_visible(True)
            self.ax.coords[idx].display_minor_ticks(True)

            
        self.xlabel = Common.ax_coord_xy[0].get_axislabel()
        self.ylabel = Common.ax_coord_xy[1].get_axislabel()
        self.zlabel = Common.ax_coord_xz[1].get_axislabel()
            
        self.fig.set_facecolor(config.get('fig_background_color'))
        self.ax.set_facecolor(config.get('ax_background_color'))
        self.im.cmap.set_bad(config.get('bad_color'))
        
        xtick_label_position = config.get('xticklabel_position')
        ytick_label_position = config.get('yticklabel_position')
        
        self.ax.coords[0].set_axislabel(self.xlabel, fontsize=config.get('axislabel_fontsize'),
                           fontfamily=config.get('axislabel_fontfamily'),
                           color=config.get('axislabel_color'))
        self.ax.coords[0].set_axislabel_position(xtick_label_position)
        self.ax.coords[0].set_ticklabel(rotation = config.get('tick_xlabelrotation'), pad = config.get('tick_pad_x'), ha='right', va='top')
        self.ax.coords[0].set_ticklabel_position(xtick_label_position)
        self.ax.coords[0].set_ticks_position(config.get('default_ticks_position'))
        self.ax.coords[0].set_minor_frequency(config.get('x_mtick_freq', 5))
        
        
        self.ax.coords[1].set_axislabel(self.ylabel, fontsize=config.get('axislabel_fontsize'),
                           fontfamily=config.get('axislabel_fontfamily'),
                           color=config.get('axislabel_color'))
        self.ax.coords[1].set_axislabel_position(ytick_label_position)
        self.ax.coords[1].set_ticklabel(rotation = config.get('tick_ylabelrotation'), pad = config.get('tick_pad_y'), ha='center', va='top')
        self.ax.coords[1].set_ticklabel_position(ytick_label_position)
        self.ax.coords[1].set_ticks_position(config.get('default_ticks_position'))
        self.ax.coords[1].set_minor_frequency(config.get('y_mtick_freq', 5))
        
        if self.plane == 'xz':
            self.ax.coords[2].set_axislabel_position(ytick_label_position)
            self.ax.coords[2].set_ticklabel(rotation = config.get('tick_ylabelrotation'), pad = config.get('tick_pad_y'), ha='center', va='top')
            self.ax.coords[2].set_ticklabel_position(ytick_label_position)
    
        elif self.plane == 'zy':
            self.zlabel = Common.ax_coord_zy[0].get_axislabel()
            self.ax.coords[2].set_axislabel_position(xtick_label_position)
            self.ax.coords[2].set_ticklabel(rotation = config.get('tick_xlabelrotation'), pad = config.get('tick_pad_x'), ha='center', va='top')
            self.ax.coords[2].set_ticklabel_position(xtick_label_position)

        self.ax.coords[2].set_axislabel(self.zlabel, fontsize=config.get('axislabel_fontsize'),
                    fontfamily=config.get('axislabel_fontfamily'),
                    color=config.get('axislabel_color'))
        self.ax.coords[2].set_ticks_position(config.get('default_ticks_position'))
        self.ax.coords[2].set_minor_frequency(config.get('z_mtick_freq', 5))
        self.ax.tick_params(axis='both', which = 'major', direction=config.get('tick_direction'), length=config.get('tick_length'),
                                color=config.get('tick_color'), width = config.get('tick_width'), labelsize = config.get('tick_labelsize'),
                                labelcolor = config.get('tick_labelcolor'))
        for spine in self.ax.spines.values():
                spine.set_visible(True)
                spine.set_zorder(5)
                spine.set_linewidth(config.get('tick_width'))
                spine.set_color(config.get('tick_color'))
        self.ax.tick_params(which = 'minor', length=config.get('mtick_length'))
    
        self.cax = self.fig.add_axes([config.get('cbar_pos_x'), config.get('cbar_pos_y'), config.get('cbar_width'), config.get('cbar_height')])
        self.cax.set_gid('colorbar')
        Common.update_integ_cax(self.cax)
        self.colorbar = self.fig.colorbar(self.im, cax = self.cax, orientation = config.get('colorbar_orientation') )
        Common.update_integ_colorbar(self.colorbar)
        self.colorbar.ax.minorticks_on()
        ColorSettingsPanel.apply_colorbar_settings(cax = self.cax, colorbar = self.colorbar, config=config)
        
        self.destroyed.connect(self.remove_colorbar)

        self.toolbar = MyNavigationToolbar(self.canvas, self, self.plane, self.ax, color_mode = ColorMode.INTEG, default_image_name = self.fits_viewer.filename)

        # Connect mouse events to handlers
        self.canvas.mpl_connect('draw_event', self.update_overlay_position)
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.canvas.mpl_connect('key_release_event', self.on_key_release)

        # Initialize click cursor lines
        self.click_v_line = self.overlay_ax.axvline(
            0, 0, 1, visible=False, lw=config.get('click_linewidth', 0.75),
            c=config.get('click_linecolor', 'greenyellow'), animated=True
        )
        self.click_h_line = self.overlay_ax.axhline(
            0, 0, 1, visible=False, lw=config.get('click_linewidth', 0.75),
            c=config.get('click_linecolor', 'greenyellow'), animated=True
        )

        # Initialize and configure the coordinate label (for intensity)
        #self.label.setParent(self.canvas)
        self.label.setStyleSheet(f"QLabel {{ color : {self.config.get('click_label_color', 'grey')}; }}")
        self.label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.label.setVisible(False)
        self.label.raise_()

        self.colorscale_button = QPushButton("Colorscale")
        self.colorscale_button.clicked.connect(self.open_color_settings)

        layout = QGridLayout()
        layout.setHorizontalSpacing(5)
        layout.setVerticalSpacing(5)
        layout.setContentsMargins(12, 0, 12, 12)
        
        if self.plane == 'xy' or self.plane == 'xz':
            self.xr_int_label = QLabel('X:')
            self.xr_int_label.setFixedWidth(20)
            self.x_min_int_input = QLineEdit(self)
            self.x_min_int_input.setPlaceholderText("X min value")
            self.x_min_int_input.setFixedWidth(80)
            self.x_min_int_input.returnPressed.connect(self.set_x_range)
            self.x_max_int_input = QLineEdit(self)
            self.x_max_int_input.setPlaceholderText("X max value")
            self.x_max_int_input.setFixedWidth(80)
            self.x_max_int_input.returnPressed.connect(self.set_x_range)
            self.x_int_button = QPushButton('Set X', self)
            self.x_int_button.clicked.connect(self.set_x_range)

        if self.plane == 'xy' or self.plane == 'zy':
            self.yr_int_label = QLabel('Y:')
            self.yr_int_label.setFixedWidth(20)
            self.y_min_int_input = QLineEdit(self)
            self.y_min_int_input.setPlaceholderText("Y min value")
            self.y_min_int_input.setFixedWidth(80)
            self.y_min_int_input.returnPressed.connect(self.set_y_range)
            self.y_max_int_input = QLineEdit(self)
            self.y_max_int_input.setPlaceholderText("Y max value")
            self.y_max_int_input.setFixedWidth(80)
            self.y_max_int_input.returnPressed.connect(self.set_y_range)
            self.y_int_button = QPushButton('Set Y', self)
            self.y_int_button.clicked.connect(self.set_y_range)
            
        if self.plane == 'xz' or self.plane == 'zy':
            self.zr_int_label = QLabel('Z:')
            self.zr_int_label.setFixedWidth(20)
            self.z_min_int_input = QLineEdit(self)
            self.z_min_int_input.setPlaceholderText("Z min value")
            self.z_min_int_input.setFixedWidth(80)
            self.z_min_int_input.returnPressed.connect(self.set_z_range)
            self.z_max_int_input = QLineEdit(self)
            self.z_max_int_input.setPlaceholderText("Z max value")
            self.z_max_int_input.setFixedWidth(80)
            self.z_max_int_input.returnPressed.connect(self.set_z_range)
            self.z_int_button = QPushButton('Set Z', self)
            self.z_int_button.clicked.connect(self.set_z_range)
        
        self.full_int_button = QPushButton('Full', self)
        self.full_int_button.clicked.connect(self.set_full_range)
        self.sync_int_button = QPushButton('Sync', self)
        self.sync_int_button.clicked.connect(self.sync_range)
        self.save_int_button = QPushButton('Save as FITS', self)
        self.save_int_button.clicked.connect(self.save_fits)
        self.marker_button = QPushButton("Markers", self)
        self.marker_button.clicked.connect(self.open_marker_panel)

        self.sync_range()
        
        self.hpbw = None
        if self.plane == 'xy':
            layout.addWidget(self.xr_int_label, 0, 0, 1, 1)
            layout.addWidget(self.x_min_int_input, 0, 1, 1, 1)
            layout.addWidget(self.x_max_int_input, 0, 2, 1, 1)
            layout.addWidget(self.x_int_button, 0, 3, 1, 1)
            
            layout.addWidget(self.yr_int_label, 1, 0, 1, 1)
            layout.addWidget(self.y_min_int_input, 1, 1, 1, 1)
            layout.addWidget(self.y_max_int_input, 1, 2, 1, 1)
            layout.addWidget(self.y_int_button, 1, 3, 1, 1)
            
            self.hpbw = AddHPBW(self.overlay_ax, self.fits_viewer.header, config)
            
        elif self.plane == 'xz':
            layout.addWidget(self.xr_int_label, 0, 0, 1, 1)
            layout.addWidget(self.x_min_int_input, 0, 1, 1, 1)
            layout.addWidget(self.x_max_int_input, 0, 2, 1, 1)
            layout.addWidget(self.x_int_button, 0, 3, 1, 1)
            
            layout.addWidget(self.zr_int_label, 1, 0, 1, 1)
            layout.addWidget(self.z_min_int_input, 1, 1, 1, 1)
            layout.addWidget(self.z_max_int_input, 1, 2, 1, 1)
            layout.addWidget(self.z_int_button, 1, 3, 1, 1)

        elif self.plane == 'zy':
            layout.addWidget(self.zr_int_label, 0, 0, 1, 1)
            layout.addWidget(self.z_min_int_input, 0, 1, 1, 1)
            layout.addWidget(self.z_max_int_input, 0, 2, 1, 1)
            layout.addWidget(self.z_int_button, 0, 3, 1, 1)
            
            layout.addWidget(self.yr_int_label, 1, 0, 1, 1)
            layout.addWidget(self.y_min_int_input, 1, 1, 1, 1)
            layout.addWidget(self.y_max_int_input, 1, 2, 1, 1)
            layout.addWidget(self.y_int_button, 1, 3, 1, 1)
        
        layout.addWidget(self.full_int_button, 0, 4, 1, 3)
        layout.addWidget(self.sync_int_button, 1, 4, 1, 3)

        layout.addWidget(self.marker_button, 0, 14, 1, 1)
        layout.addWidget(self.colorscale_button, 1, 15, 1,1)
        layout.addWidget(self.save_int_button, 0, 15, 1,1)
    
        layout.addWidget(self.canvas, 2, 0, 14, 16)
        layout.addWidget(self.toolbar, 16, 0, 1, 16)
    
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
        self.label.raise_()
        self.show()
        # Prime background for overlay-based draws
        self.canvas.draw()
        

    def update_overlay_position(self, event):
        if event.canvas is not self.canvas:
            return
        if not getattr(self, '_overlay_updates_enabled', True):
            return
        if getattr(self, '_updating_overlay', False):
            return
        if not hasattr(self, 'overlay_ax'):
            return
        region_manager = getattr(self, 'region_manager', None)
        marker_manager = getattr(self, 'marker_manager', None)
        if marker_manager is not None and marker_manager.is_dragging():
            return
        self._updating_overlay = True

        # Sync overlay with main axes and refresh background
        self.overlay_ax.set_position(self.ax.get_position())
        vline_visible = self.click_v_line.get_visible()
        hline_visible = self.click_h_line.get_visible()
        hidden_regions = []
        hidden_markers = []
        if region_manager is not None:
            hidden_regions = region_manager.prepare_for_background_capture()
        if marker_manager is not None:
            hidden_markers = marker_manager.prepare_for_background_capture(self.plane)

        self.click_v_line.set_visible(False)
        self.click_h_line.set_visible(False)
        self._background = self.canvas.copy_from_bbox(self.overlay_ax.bbox)
        self.click_v_line.set_visible(vline_visible)
        self.click_h_line.set_visible(hline_visible)

        if region_manager is not None:
            region_manager.restore_after_background_capture(hidden_regions)
        if marker_manager is not None and hidden_markers:
            marker_manager.restore_after_background_capture(hidden_markers)

        has_regions = bool(region_manager and region_manager.regions)
        has_markers = bool(marker_manager and marker_manager.markers_for_plane(self.plane))
        needs_redraw = (
            vline_visible
            or hline_visible
            or has_regions
            or has_markers
            or self.hpbw is not None
        )

        self._updating_overlay = False

        if needs_redraw:
            QTimer.singleShot(0, self.redraw_main_overlay_and_blit)

    def set_overlay_updates_enabled(self, enabled: bool):
        self._overlay_updates_enabled = bool(enabled)

    def _reset_navigation_mode(self):
        toolbar = getattr(self, 'toolbar', None)
        if toolbar is None:
            return
        try:
            toolbar.mode = ''
        except Exception:
            pass
        actions = getattr(toolbar, '_actions', None)
        if isinstance(actions, dict):
            for key in ('zoom', 'pan'):
                action = actions.get(key)
                if action is not None:
                    action.setChecked(False)
        try:
            toolbar._clear_cursor_override()
        except Exception:
            pass

    def open_marker_panel(self):
        if self.marker_panel is None or not self.marker_panel.isVisible():
            from tools.marker_panel import MarkerPanel
            self.marker_panel = MarkerPanel(self, self.marker_manager)
            try:
                self.marker_panel.destroyed.connect(lambda: setattr(self, 'marker_panel', None))
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
        previous = getattr(self, 'marker_mode_enabled', False)
        self.marker_mode_enabled = enabled
        marker_manager = getattr(self, 'marker_manager', None)
        if not enabled:
            if marker_manager is not None:
                marker_manager.cancel_placement()
            panel = getattr(self, 'marker_panel', None)
            if panel is not None and getattr(panel, 'placement_toggle', None) is not None:
                if panel.placement_toggle.isChecked():
                    panel.placement_toggle.blockSignals(True)
                    panel.placement_toggle.setChecked(False)
                    panel.placement_toggle.blockSignals(False)
                    try:
                        panel._on_placement_toggled(False)
                    except Exception:
                        pass
        else:
            if marker_manager is not None:
                marker_manager.set_active_plane(self.plane)
            if getattr(self, 'region_mode_enabled', False):
                self.set_region_mode(False)
            if getattr(self, 'canvas', None) is not None:
                self.canvas.setFocus()
            self._reset_navigation_mode()
        if enabled and self.toolbar.mode in ('zoom rect', 'pan/zoom'):
            self._reset_navigation_mode()
        if previous != enabled:
            if not enabled and self.canvas is not None:
                try:
                    self.canvas.setCursor(Qt.CursorShape.ArrowCursor)
                except Exception:
                    pass


    def _update_toolbar_message(self, event, intensity=None):
        if not getattr(self, 'toolbar', None):
            return
        if event is None or event.xdata is None or event.ydata is None:
            self.toolbar.set_message('')
            return
        message = self.ax.format_coord(event.xdata, event.ydata)
        value = intensity
        if value is None:
            value = self._sample_intensity(event.xdata, event.ydata)
        if value is not None:
            message = f"{message}\n[{value:.4g}]"
        self.toolbar.set_message(message)

    def _sample_intensity(self, x, y):
        xp, yp = int(round(x)), int(round(y))
        try:
            if self.plane == 'zy':
                return self.integrated_data.T[yp, xp]
            return self.integrated_data[yp, xp]
        except (IndexError, TypeError):
            return None


    def redraw_main_overlay_and_blit(self):
        canvas = getattr(self, 'canvas', None)
        if canvas is None or getattr(self, '_background', None) is None:
            return

        canvas.restore_region(self._background)

        region_manager = getattr(self, 'region_manager', None)
        if region_manager is not None:
            region_manager.draw_regions_for_blit()

        if self.hpbw is not None:
            self.hpbw.update_position()

        marker_manager = getattr(self, 'marker_manager', None)
        if marker_manager is not None:
            marker_manager.draw_markers_for_blit(self.plane)

        if self.click_v_line.get_visible():
            self.overlay_ax.draw_artist(self.click_v_line)
        if self.click_h_line.get_visible():
            self.overlay_ax.draw_artist(self.click_h_line)

        canvas.blit(self.overlay_ax.bbox)


    def remove_colorbar(self):
        Common.remove_integ_colorbar(self.colorbar)
        Common.remove_integ_cax(self.cax)

    def open_color_settings(self):
        if self.color_settings_panel is None:
            self.color_settings_panel = ColorSettingsPanel(
                mode=ColorMode.INTEG,
                fits_viewer=self,
                data=self.integrated_data,
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
        if self.color_settings_panel is not None:
            self.color_settings_panel.close()
            self.color_settings_panel = None

    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._background = None
        if hasattr(self, 'label'):
            pos_x = self.config.get('poslabel_x', 0.75)
            pos_y = self.config.get('poslabel_y', 0.9)
            width = self.config.get('poslabel_w', 250)
            height = self.config.get('poslabel_h', 30)
            self.label.setGeometry(int(self.canvas.width()*pos_x - width/2), int(self.canvas.height() - self.canvas.height()*pos_y + height/2) , width, height)
            
        if self.toolbar._subplot_dialog is not None:
            self.toolbar._subplot_dialog.width_spin.setValue(int(self.window().width()))
            self.toolbar._subplot_dialog.height_spin.setValue(int(self.window().height()))

    
    def closeEvent(self, event):
        self.remove_colorbar()
        if self.color_settings_panel is not None:
            self.color_settings_panel.close()
            self.color_settings_panel = None
        if getattr(self, 'cutout_dialog', None) is not None:
            try:
                self.cutout_dialog.close()
            finally:
                self.cutout_dialog = None

        if self.canvas is not None:
            self.canvas.close()
            self.canvas = None
        if self.fig is not None:
            self.fig.clear()
            self.fig = None
        if self.toolbar._subplot_dialog is not None:
            self.toolbar._subplot_dialog.close()
        super().closeEvent(event)

        
    
    def sync_range(self):
        if self.plane == 'xy' or self.plane == 'xz':
            x_min = Common.xmin_input_xy.text()
            x_max = Common.xmax_input_xy.text()
            self.x_min_int_input.setText(x_min)
            self.x_max_int_input.setText(x_max)
            self.set_x_range()
            
        if self.plane == 'xy' or self.plane == 'zy':
            y_min = Common.ymin_input_xy.text()
            y_max = Common.ymax_input_xy.text()
            self.y_min_int_input.setText(y_min)
            self.y_max_int_input.setText(y_max)
            self.set_y_range()
            
        if self.plane == 'xz' or self.plane == 'zy':
            z_min = Common.zmin_input_xz.text()
            z_max = Common.zmax_input_xz.text()
            self.z_min_int_input.setText(z_min)
            self.z_max_int_input.setText(z_max)
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
            self.x_min_int_input.setText(str(self.xmin_val_full))
            self.x_max_int_input.setText(str(self.xmax_val_full))
            self.set_x_range()
            
        if self.plane == 'xy' or self.plane == 'zy':
            self.y_min_int_input.setText(str(self.ymin_val_full))
            self.y_max_int_input.setText(str(self.ymax_val_full))
            self.set_y_range()
            
        if self.plane == 'xz' or self.plane == 'zy':
            self.z_min_int_input.setText(str(self.zmin_val_full))
            self.z_max_int_input.setText(str(self.zmax_val_full))
            self.set_z_range()
            
    def set_x_range(self):
        try:
            x_min = self.x_min_int_input.text()
            x_max = self.x_max_int_input.text()
            if self.fits_viewer.data.ndim == 3:
                xp_min = float(self.converter.world_to_pix(x_min, self.ymin_val, self.zmin_val)[0])
                xp_max = float(self.converter.world_to_pix(x_max, self.ymax_val, self.zmax_val)[0])
            elif self.fits_viewer.data.ndim == 4:
                xp_min = float(self.converter.world_to_pix(x_min, self.ymin_val, self.zmin_val, 0)[0])
                xp_max = float(self.converter.world_to_pix(x_max, self.ymax_val, self.zmax_val, 0)[0])
                
            if xp_min > xp_max: xp_min, xp_max = xp_max, xp_min
                
            self.ax.set_xlim(xp_min, xp_max)
            self._background = None
            self.canvas.draw_idle()
        except ValueError:
            QMessageBox.warning(self, 'Invalid Input', 'Please enter valid numeric values for the X range.')
            
    def set_y_range(self):
        try:
            y_min = self.y_min_int_input.text()
            y_max = self.y_max_int_input.text()
            if self.fits_viewer.data.ndim == 3:
                yp_min = float(self.converter.world_to_pix(self.xmin_val, y_min, self.zmin_val)[1])
                yp_max = float(self.converter.world_to_pix(self.xmax_val, y_max, self.zmax_val)[1])
            elif self.fits_viewer.data.ndim == 4:
                yp_min = float(self.converter.world_to_pix(self.xmin_val, y_min, self.zmin_val, 0)[1])
                yp_max = float(self.converter.world_to_pix(self.xmax_val, y_max, self.zmax_val, 0)[1])
                
            if yp_min > yp_max: yp_min, yp_max = yp_max, yp_min

            self.ax.set_ylim(yp_min, yp_max)
            self._background = None
            self.canvas.draw_idle()
        except ValueError:
            QMessageBox.warning(self, 'Invalid Input', 'Please enter valid numeric values for the Y range.')


    def set_z_range(self):
        try:
            z_min = self.z_min_int_input.text()
            z_max = self.z_max_int_input.text()
            if self.fits_viewer.data.ndim == 3:
                zp_min = float(self.converter.world_to_pix(self.xmin_val, self.ymin_val, z_min)[2])
                zp_max = float(self.converter.world_to_pix(self.xmax_val, self.ymax_val, z_max)[2])
            elif self.fits_viewer.data.ndim == 4:
                zp_min = float(self.converter.world_to_pix(self.xmin_val, self.ymin_val, z_min, 0)[2])
                zp_max = float(self.converter.world_to_pix(self.xmax_val, self.ymax_val, z_max, 0)[2])
                
            if zp_min > zp_max: zp_min, zp_max = zp_max, zp_min
            
            if self.plane == 'xz':
                self.ax.set_ylim(zp_min, zp_max)
            elif self.plane == 'zy':
                self.ax.set_xlim(zp_min, zp_max)
            self._background = None
            self.canvas.draw_idle()
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
            
            
            self.x_min_int_input.setText(str(xmin_val))
            self.x_max_int_input.setText(str(xmax_val))
            self.y_min_int_input.setText(str(ymin_val))
            self.y_max_int_input.setText(str(ymax_val))
    
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
                
            self.x_min_int_input.setText(str(xmin_val))
            self.x_max_int_input.setText(str(xmax_val))
            self.z_min_int_input.setText(str(zmin_val))
            self.z_max_int_input.setText(str(zmax_val))
                
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
            
            self.z_min_int_input.setText(str(zmin_val))
            self.z_max_int_input.setText(str(zmax_val))
            self.y_min_int_input.setText(str(ymin_val))
            self.y_max_int_input.setText(str(ymax_val))
            

    def on_click(self, event):
        # Handle all double-click events first
        if event.dblclick:
            current_mode = self.toolbar.mode
            # Case 1: If in pan/zoom mode, disable it
            if current_mode == 'pan/zoom':
                self.toolbar.pan(False)
                self.toolbar._active = None
                release_event = mpl.backend_bases.MouseEvent('button_release_event', self.canvas, event.x, event.y, button=event.button, dblclick=True, guiEvent=event.guiEvent)
                self.toolbar.release_pan(release_event)
                self.toolbar._update_buttons_checked()
                self.canvas.draw_idle()
                return
            # Case 2: If in zoom rect mode, disable it
            elif current_mode == 'zoom rect':
                self.toolbar.zoom(False)
                self.toolbar._active = None
                release_event = mpl.backend_bases.MouseEvent('button_release_event', self.canvas, event.x, event.y, button=event.button, dblclick=True, guiEvent=event.guiEvent)
                self.toolbar.release_zoom(release_event)
                self.toolbar._update_buttons_checked()
                self.canvas.draw_idle()
                return
            # Case 3: If not in a special mode and click is outside axes, hide lines/label
            elif event.inaxes not in (self.ax, self.overlay_ax):
                self.label.setVisible(False)
                self.click_v_line.set_visible(False)
                self.click_h_line.set_visible(False)
                self.redraw_main_overlay_and_blit()
                return

        if (
            getattr(self, 'marker_mode_enabled', False)
            and self.toolbar.mode == ''
            and event.button == 1
            and event.inaxes in (self.ax, self.overlay_ax)
        ):
            marker_manager = getattr(self, 'marker_manager', None)
            if marker_manager is not None:
                self.canvas.setFocus()
                marker_manager.set_active_plane(self.plane)
                marker_manager.handle_press(event)
                self.redraw_main_overlay_and_blit()
            return

        # Handle single left-click events for showing coordinates
        if self.region_mode_enabled and self.toolbar.mode == '' and event.button == 1 and event.inaxes in (self.ax, self.overlay_ax):
            self.dragging = False
            self.canvas.setFocus()
            self.region_manager.handle_press(event)
            self.redraw_main_overlay_and_blit()
            return

        if event.button == 1 and self.toolbar.mode == '' and event.inaxes in (self.ax, self.overlay_ax) and event.xdata is not None and event.ydata is not None:
            self.dragging = True
            x, y = event.xdata, event.ydata
            xstr, ystr = self.format_pix.convert(self.plane, x, y)

            if self.plane == 'xy': text_coord_tuple = (xstr, ystr)
            elif self.plane == 'xz': text_coord_tuple = (xstr, ystr)
            elif self.plane == 'zy': text_coord_tuple = (ystr, xstr)

            try:
                if self.plane == 'zy':
                    intensity = self.integrated_data.T[int(round(y)), int(round(x))]
                else:
                    intensity = self.integrated_data[int(round(y)), int(round(x))]
                print(f'\r Clicked at ({", ".join(map(str, text_coord_tuple))})              \n Intensity = {intensity:.5g} {self.bunit}            \033[1A', end='     ')
                
                intensity_str = '{:.4g}'.format(intensity)
                self.label.setText(f'{xstr}, {ystr} \n[{intensity_str}]')
            except (IndexError, TypeError):
                print(f'\r Clicked at ({", ".join(map(str, text_coord_tuple))})              \n               \033[1A', end='     ')
                self.label.setText(f'{xstr}, {ystr}')

            # Position and show the main label
            pos_x = self.config.get('poslabel_x', 0.75)
            pos_y = self.config.get('poslabel_y', 0.9)
            width = self.config.get('poslabel_w', 250)
            height = self.config.get('poslabel_h', 30)
            self.label.setGeometry(int(self.canvas.width()*pos_x - width/2), int(self.canvas.height() - self.canvas.height()*pos_y)+20, width, height)
            self.label.setVisible(True)

            # Update and show cursor lines
            self.click_v_line.set_data([x, x], [0, 1])
            self.click_h_line.set_data([0, 1], [y, y])
            self.click_v_line.set_visible(True)
            self.click_h_line.set_visible(True)
            self.redraw_main_overlay_and_blit()
            if self.region_mode_enabled and self.toolbar.mode == '' and event.inaxes in (self.ax, self.overlay_ax):
                self.canvas.setFocus()
                self.region_manager.handle_press(event)
                self.redraw_main_overlay_and_blit()


                

    def on_release(self, event):
        # Stop dragging when the mouse button is released
        if event.button == 1:
            self.dragging = False
            if self.region_mode_enabled:
                self.region_manager.handle_release(event)
                self.redraw_main_overlay_and_blit()
            marker_manager = getattr(self, 'marker_manager', None)
            if getattr(self, 'marker_mode_enabled', False) and marker_manager is not None:
                marker_manager.handle_release(event)


    def on_motion(self, event):
        marker_manager = getattr(self, 'marker_manager', None)
        if marker_manager is not None and getattr(self, 'marker_mode_enabled', False):
            if marker_manager.is_dragging():
                marker_manager.handle_motion(event)
            else:
                marker_manager.handle_hover(event)
            return

        if self.toolbar:
            if event.inaxes in (self.ax, self.overlay_ax) and event.xdata is not None and event.ydata is not None:
                self._update_toolbar_message(event)
            else:
                self._update_toolbar_message(None)

        if self.region_mode_enabled:
            if (self.region_manager.is_drawing or self.region_manager.is_dragging or
                    self.region_manager.is_resizing or self.region_manager.is_rotating):
                if event.inaxes in (self.ax, self.overlay_ax):
                    self.region_manager.handle_motion(event)
                    self.redraw_main_overlay_and_blit()
            else:
                if self.region_manager.update_hover_cursor(event):
                    if self.toolbar and event.xdata is not None and event.ydata is not None:
                        self._update_toolbar_message(event)
                    return
            return

        # If dragging, update the cursor and coordinate label
        if self.dragging and event.inaxes in (self.ax, self.overlay_ax) and event.xdata is not None and event.ydata is not None:
            x, y = event.xdata, event.ydata
            xstr, ystr = self.format_pix.convert(self.plane, x, y)
            xp, yp = int(round(x)), int(round(y))

            try:
                if self.plane == 'zy':
                    intensity = self.integrated_data.T[yp, xp]
                else:
                    intensity = self.integrated_data[yp, xp]
                intensity_str = '{:.4g}'.format(intensity)
                self.label.setText(f'{xstr}, {ystr} \n[{intensity_str}]')
            except (IndexError, TypeError):
                self.label.setText(f'{xstr}, {ystr}')
                intensity = None

            # Update cursor lines
            self.click_v_line.set_data([x, x], [0, 1])
            self.click_h_line.set_data([0, 1], [y, y])
            self.redraw_main_overlay_and_blit()
            if self.region_mode_enabled:
                self.region_manager.handle_motion(event)
                self.redraw_main_overlay_and_blit()
            else:
                self._update_toolbar_message(event, intensity)

        elif event.inaxes in (self.ax, self.overlay_ax) and event.xdata is not None and event.ydata is not None:
            self._update_toolbar_message(event)
        else:
            self._update_toolbar_message(None)


    def on_key_press(self, event):
        """
        Handles key press events from the Matplotlib canvas.
        """
        region_manager = getattr(self, 'region_manager', None)
        if region_manager is not None:
            region_manager.handle_key_press(event)
        marker_manager = getattr(self, 'marker_manager', None)
        if marker_manager is not None:
            marker_manager.handle_key_press(event)
        if event.key == 'backspace' or event.key == 'delete':
            if self.region_mode_enabled:
                self.region_manager.delete_selected_region()
            elif getattr(self, 'marker_mode_enabled', False) and marker_manager is not None:
                marker = marker_manager.selected_marker()
                if marker is not None:
                    plane = marker.plane
                    marker_manager.remove_marker(marker.marker_id, plane)
                    marker_manager.redraw_plane(plane)

    def on_key_release(self, event):
        region_manager = getattr(self, 'region_manager', None)
        if region_manager is not None:
            region_manager.handle_key_release(event)
        marker_manager = getattr(self, 'marker_manager', None)
        if marker_manager is not None:
            marker_manager.handle_key_release(event)

    def _default_contour_label(self) -> str:
        plane_tag = self.plane.upper() if self.plane else ""
        title = self.windowTitle() or ""
        if plane_tag:
            if title:
                return f"{title} [{plane_tag}]"
            return plane_tag
        return title or "Integration Map"

    def _contour_items_provider(self):
        if not hasattr(self, "ax") or self.ax is None:
            return []
        image = getattr(self, "im", None)
        if image is None:
            return []
        arr = image.get_array()
        if arr is None:
            return []
        if np.ma.isMaskedArray(arr):
            arr = arr.filled(np.nan)
        data = np.asarray(arr)
        metadata = {}
        try:
            clim = self.im.get_clim()
        except Exception:
            clim = None
        if clim is not None:
            metadata["clim"] = tuple(clim)
        return [ContourItem(ax=self.ax, data=data, label=self._default_contour_label(), metadata=metadata)]

    def _register_contour_layer(self):
        if self._contour_layer_id is not None:
            return
        manager = ContourManager.instance()
        layer_id = f"integ-{id(self)}"
        try:
            manager.register_layer(
                layer_id=layer_id,
                label=self._default_contour_label(),
                plane=self.plane,
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
        self._unregister_contour_layer()
        marker_panel = getattr(self, 'marker_panel', None)
        if marker_panel is not None:
            try:
                marker_panel.close()
            except Exception:
                pass
            self.marker_panel = None
        marker_manager = getattr(self, 'marker_manager', None)
        if marker_manager is not None:
            marker_manager.clear_plane(self.plane)
        super().closeEvent(event)

    def save_fits(self):
        data = self.integrated_data
        data_min =  float(np.nanmin(data))
        data_max =  float(np.nanmax(data))
        original_header = self.fits_viewer.header
        new_header = self.fits_viewer.header.copy()
        original_filename = self.fits_viewer.filename
        

        new_header['DATAMIN'] = data_min
        new_header['DATAMAX'] = data_max



        plane_to_axis = {'xy': 2, 'xz': 1, 'zy': 0}
        axis_to_drop = plane_to_axis.get(self.plane)
        
        new_wcs = WCS(naxis=data.ndim)
        axes_to_keep = [i for i in range(3) if i != axis_to_drop]
        if self.plane == 'zy':
            axes_to_keep = axes_to_keep[::-1]

        for new_axis_index, old_axis_index in enumerate(axes_to_keep):
            new_wcs.wcs.crpix[new_axis_index] = self.wcs.wcs.crpix[old_axis_index]
            new_wcs.wcs.cdelt[new_axis_index] = self.wcs.wcs.cdelt[old_axis_index]
            new_wcs.wcs.crval[new_axis_index] = self.wcs.wcs.crval[old_axis_index]
            new_wcs.wcs.ctype[new_axis_index] = self.wcs.wcs.ctype[old_axis_index]
            new_wcs.wcs.cunit[new_axis_index] = self.wcs.wcs.cunit[old_axis_index]
            if self.wcs.wcs.has_pc():
                new_wcs.wcs.pc[new_axis_index, :] = self.wcs.wcs.pc[old_axis_index, axes_to_keep]
            if self.wcs.wcs.has_cd():
                new_wcs.wcs.cd[new_axis_index, :] = self.wcs.wcs.cd[old_axis_index, axes_to_keep]


        new_header = new_wcs.to_header()
        
        for key in ['BSCALE', 'BZERO',
                    'BMAJ', 'BMIN', 'BPA', 'BTYPE', 'OBJECT', 'BUNIT', 'RADESYS',
                    'LONPOLE', 'LATPOLE', 'TELESCOP', 'INSTRUME', 'OBSERVER',
                    'DATE-OBS', 'DATE', 'TIMESYS', 'OBSRA', 'OBSDEC',
                    'OBSGEO-X', 'OBSGEO-Y', 'OBSGEO-Z', 'SPECSYS', 'RESTFRQ', 
                    'VELREF', 'ALTRVAL', 'ALTRPIX']:
            if key in original_header:
                new_header[key] = original_header[key]
        
        #velocity unit conversion [Note: Subject to change in the future.]
        unit_wcs = self.wcs.wcs.cunit[2].to_string().replace(' ', '').lower()
        unit_header = original_header.get('CUNIT3', '').replace(' ', '').lower()            
        if unit_wcs and unit_header:
            if unit_wcs != unit_header:
                if axis_to_drop == 1:
                    new_header['CUNIT2'] = original_header['CUNIT3']
                elif axis_to_drop == 0:
                    new_header['CUNIT1'] = original_header['CUNIT3']
            elif ('m/s' or 'ms-1' in unit_header and  original_header['CDELT3'] > 100):
                if axis_to_drop == 1:
                    new_header['CUNIT2'] = 'km/s'
                elif axis_to_drop == 0:
                    new_header['CUNIT1'] = 'km/s'
        elif not 'CUNIT3' in original_header:
            if axis_to_drop == 1:
                if new_header['CUNIT2']=='m s-1': new_header['CUNIT2'] = 'km/s'
            elif axis_to_drop == 0:
                if new_header['CUNIT1']=='m s-1': new_header['CUNIT1'] = 'km/s'
        
        if axis_to_drop == 0:
            data = data.T
        new_header = self.reorder_fits_header(new_header)
        """
        if 'BUNIT' in original_header and f'CUNIT{axis_to_drop+1}' in original_header:
            if self.integ_mode == 'int':
                if axis_to_drop == 0 or axis_to_drop == 1:
                    new_header['BUNIT'] = original_header['BUNIT'] + ' ' + original_header[f'CUNIT{axis_to_drop+1}']
                else:
                    new_header['BUNIT'] = original_header['BUNIT'] + ' km/s'                    
            elif self.integ_mode == 'mom1' or self.integ_mode == 'mom2' or self.integ_mode == 'peak_coord':
                if axis_to_drop == 0 or axis_to_drop == 1:
                    new_header['BUNIT'] = original_header[f'CUNIT{axis_to_drop+1}']
                else:
                    new_header['BUNIT'] = 'km/s'
        """
        if self.bunit:
            new_header['BUNIT'] = self.bunit

        save_fits = SaveFITS(data, new_header, original_filename)
        save_fits.save(suffix=self.integ_mode)
        
    def reorder_fits_header(self, header):
        preferred_order = [
            'SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'NAXIS3', 'NAXIS4', 'EXTEND',
            'BSCALE', 'BZERO',
    
            'BMAJ', 'BMIN', 'BPA', 'BTYPE', 'OBJECT', 'BUNIT', 'RADESYS',
            'LONPOLE', 'LATPOLE', 'TELESCOP', 'INSTRUME', 'OBSERVER',
            'DATE-OBS', 'DATE', 'TIMESYS', 'OBSRA', 'OBSDEC',
            'OBSGEO-X', 'OBSGEO-Y', 'OBSGEO-Z', 'SPECSYS', 'RESTFRQ', 
            'VELREF', 'ALTRVAL', 'ALTRPIX',
    
            'CTYPE1', 'CRVAL1', 'CDELT1', 'CRPIX1', 'CUNIT1', 'CROTA1',
            'CTYPE2', 'CRVAL2', 'CDELT2', 'CRPIX2', 'CUNIT2', 'CROTA2',
            'CTYPE3', 'CRVAL3', 'CDELT3', 'CRPIX3', 'CUNIT3', 'CROTA3',
            'CTYPE4', 'CRVAL4', 'CDELT4', 'CRPIX4', 'CUNIT4', 'CROTA4',
    
            'PC1_1', 'PC2_1', 'PC3_1', 'PC4_1',
            'PC1_2', 'PC2_2', 'PC3_2', 'PC4_2', 
            'PC1_3', 'PC2_3', 'PC3_3', 'PC4_3',
            'PC1_4', 'PC2_4', 'PC3_4', 'PC4_4',
        ]

        new_header = fits.Header()
    
        for key in preferred_order:
            if key in header:
                try:
                    value = header[key]
                    new_header[key] = value
                except ValueError as e:
                    print(f"Skipping invalid key-value pair: {key} -> {value}")
    
        for key, value in header.items():
            if key not in new_header:
                try:
                    new_header[key] = value
                except ValueError as e:
                    print(f"Skipping invalid key-value pair: {key} -> {value}")
    
        return new_header

    def set_region_mode(self, checked):
        checked = bool(checked)
        if checked:
            self.set_marker_mode(False)
        self.region_mode_enabled = checked
        main_window_shape = self.fits_viewer.region_manager.region_mode
        if self.region_mode_enabled and main_window_shape:
            shape_for_integ = 'rectangle' if main_window_shape == 'cube' else main_window_shape
            self.region_manager.set_region_mode(shape_for_integ)
            self.setWindowTitle(f"[REGION MODE: {shape_for_integ.upper()}] {self.original_window_title}")
            self.redraw_main_overlay_and_blit()
        else:
            self.setWindowTitle(self.original_window_title)
            self.region_manager.deselect_all()
            if self.canvas:
                self._background = None
                self.canvas.draw_idle()
