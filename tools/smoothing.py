from PyQt6.QtWidgets import QDialog, QGridLayout, QGroupBox, QButtonGroup, QVBoxLayout, QRadioButton, QCheckBox, QLabel, QPushButton, QLineEdit, QDoubleSpinBox, QMessageBox
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QDoubleValidator
import numpy as np
from astropy.convolution import Gaussian1DKernel, Gaussian2DKernel, CustomKernel
from scipy.signal import fftconvolve
from core.save_fits import SaveFITS
import time

class SmoothSettingsPanel(QDialog): 
    def __init__(self, fits_viewer, subwindows):
        super().__init__()
        self.fits_viewer = fits_viewer        
        self.header = self.fits_viewer.header
        self.wcs = self.fits_viewer.wcs
        self.subwindows = subwindows
        
        self.data = self.fits_viewer.data
        self.original_data = None  # Defer copying until first execution
        
        self.initUI()
        

        self.bmaj_exists = 'BMAJ' in self.header
        self.bmin_exists = 'BMIN' in self.header
        self.bpa_exists = 'BPA' in self.header
            
        if self.bmaj_exists:
            self.original_bmaj = self.header['BMAJ']
            self.bmaj_in_arcsec = self.header['BMAJ'] * 3600
            self.bmaj_current.setText(f'{self.bmaj_in_arcsec:.3g}')
        if self.bmin_exists:
            self.original_bmin = self.header['BMIN']
            self.bmin_in_arcsec = self.header['BMIN'] * 3600
            self.bmin_current.setText(f'{self.bmin_in_arcsec:.3g}')
        if self.bpa_exists:
            self.original_bpa = self.header['BPA']
            self.bpa_in_deg = self.header['BPA']
            self.bpa_current.setText(f'{self.bpa_in_deg:.3g}')
        
    def initUI(self):    
        grid_layout = QGridLayout()
        grid_layout.setHorizontalSpacing(10)
        grid_layout.setVerticalSpacing(10)
        grid_layout.setContentsMargins(15, 0, 10, 13)
        
        self.smoothness_groupbox = QGroupBox("Smoothness")
        smoothness_layout = QGridLayout()

        smoothness_layout.addWidget(QLabel("X"), 0, 0)
        self.x_spinbox = QDoubleSpinBox()
        self.x_spinbox.setValue(1.00)
        self.x_spinbox.setSingleStep(1.)
        smoothness_layout.addWidget(self.x_spinbox, 0, 1)

        smoothness_layout.addWidget(QLabel("Y"), 1, 0)
        self.y_spinbox = QDoubleSpinBox()
        self.y_spinbox.setValue(1.00)
        self.y_spinbox.setSingleStep(1.)
        smoothness_layout.addWidget(self.y_spinbox, 1, 1)
        
        if  self.subwindows:
            smoothness_layout.addWidget(QLabel("Z"), 2, 0)
            self.z_spinbox = QDoubleSpinBox()
            self.z_spinbox.setValue(0.00)
            self.z_spinbox.setSingleStep(1.)
            smoothness_layout.addWidget(self.z_spinbox, 2, 1)

        self.smoothness_groupbox.setLayout(smoothness_layout)

        grid_layout.addWidget(self.smoothness_groupbox, 0, 0, 4, 1)

        kernel_groupbox = QGroupBox("Kernel")
        kernel_layout = QVBoxLayout()

        self.gaussian_radio = QRadioButton("Gaussian")
        self.boxcar_radio = QRadioButton("Boxcar")
        self.gaussian_radio.setChecked(True)

        self.kernel_button_group = QButtonGroup()
        self.kernel_button_group.addButton(self.gaussian_radio)
        self.kernel_button_group.addButton(self.boxcar_radio)

        kernel_layout.addWidget(self.gaussian_radio)
        kernel_layout.addWidget(self.boxcar_radio)
        kernel_groupbox.setLayout(kernel_layout)

        grid_layout.addWidget(kernel_groupbox, 0, 1, 3, 1)

        self.checkbox = QCheckBox("Target res.")
        grid_layout.addWidget(self.checkbox, 3, 1, alignment=Qt.AlignmentFlag.AlignCenter)

        self.target_res_groupbox = QGroupBox("Target resolution")
        target_res_layout = QGridLayout()
        target_res_layout.setSpacing(2)

        target_res_layout.addWidget(QLabel("BMAJ"), 0, 0)
        self.bmaj_target = QLineEdit()
        self.bmaj_target.setFixedWidth(50)
        target_res_layout.addWidget(self.bmaj_target, 0, 1)
        target_res_layout.addWidget(QLabel("arcsec"), 0, 2)
        self.bmaj_target.setValidator(QDoubleValidator())
        
        target_res_layout.addWidget(QLabel("BMIN"), 1, 0)
        self.bmin_target = QLineEdit()
        self.bmin_target.setFixedWidth(50)
        target_res_layout.addWidget(self.bmin_target, 1, 1)
        target_res_layout.addWidget(QLabel("arcsec"), 1, 2)
        self.bmin_target.setValidator(QDoubleValidator())

        target_res_layout.addWidget(QLabel("BPA"), 2, 0)
        self.bpa_target = QLineEdit()
        self.bpa_target.setFixedWidth(50)
        target_res_layout.addWidget(self.bpa_target, 2, 1)
        target_res_layout.addWidget(QLabel("deg"), 2, 2)
        self.bpa_target.setValidator(QDoubleValidator())

        self.target_res_groupbox.setLayout(target_res_layout)
        grid_layout.addWidget(self.target_res_groupbox, 0, 2, 4, 1)

        current_res_groupbox = QGroupBox("Current resolution")
        current_res_layout = QGridLayout()
        current_res_layout.setSpacing(2)

        current_res_layout.addWidget(QLabel("BMAJ"), 0, 0)
        self.bmaj_current = QLineEdit()
        self.bmaj_current.setFixedWidth(50)
        self.bmaj_current.setReadOnly(True)
        current_res_layout.addWidget(self.bmaj_current, 0, 1)
        current_res_layout.addWidget(QLabel("arcsec"), 0, 2)

        current_res_layout.addWidget(QLabel("BMIN"), 1, 0)
        self.bmin_current = QLineEdit()
        self.bmin_current.setFixedWidth(50)
        self.bmin_current.setReadOnly(True)
        current_res_layout.addWidget(self.bmin_current, 1, 1)
        current_res_layout.addWidget(QLabel("arcsec"), 1, 2)

        current_res_layout.addWidget(QLabel("BPA"), 2, 0)
        self.bpa_current = QLineEdit()
        self.bpa_current.setFixedWidth(50)
        self.bpa_current.setReadOnly(True)
        current_res_layout.addWidget(self.bpa_current, 2, 1)
        current_res_layout.addWidget(QLabel("deg"), 2, 2)

        current_res_groupbox.setLayout(current_res_layout)
        grid_layout.addWidget(current_res_groupbox, 0, 3, 4, 1)

        self.execute_button = QPushButton("Execute")
        self.execute_button.clicked.connect(self.execute_smoothing)
        grid_layout.addWidget(self.execute_button, 4, 0, 1, 1) 
        self.execute_button.setAutoDefault(True) 
        self.execute_button.setDefault(True)

        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_smoothing)
        grid_layout.addWidget(self.reset_button, 4, 1, 1, 1) 
        

        self.save_button = QPushButton("Save as FITS")
        self.save_button.clicked.connect(self.save_fits)
        grid_layout.addWidget(self.save_button, 4, 2, 1, 1) 


        self.nan_checkbox = QCheckBox("Handle NaNs") #Time-consuming
        self.nan_checkbox.setChecked(False)
        grid_layout.addWidget(self.nan_checkbox, 4, 3, alignment=Qt.AlignmentFlag.AlignCenter)
        self.nan_checkbox.setEnabled(False) #### if dsired, change it to True

        self.setLayout(grid_layout)

        self.checkbox.stateChanged.connect(self.toggle_target_res_group)
        self.checkbox.setChecked(False)
        self.target_res_groupbox.setEnabled(False)

        self.smoothness_groupbox.setEnabled(True)

        self.gaussian_radio.toggled.connect(self.kernel_selection_changed)

        self.checkbox.setEnabled(True)
        
        self.setWindowTitle(f'Smooth Panel: {self.fits_viewer.filename}')
        self.move_to_default_position()

    def toggle_target_res_group(self, state=None):
        is_checked = self.checkbox.isChecked()
        self.target_res_groupbox.setEnabled(is_checked)
        self.bmaj_target.setFocus()
        self.smoothness_groupbox.setEnabled(not is_checked)
        
        if is_checked:
            if not self.bmaj_exists:
                self.bmaj_current.setReadOnly(False)
            if not self.bmin_exists:
                self.bmin_current.setReadOnly(False)
            if not self.bpa_exists:
                self.bpa_current.setReadOnly(False)
        else:
            self.bmaj_current.setReadOnly(True)
            self.bmin_current.setReadOnly(True)
            self.bpa_current.setReadOnly(True)

    def kernel_selection_changed(self, checked):
        if checked:
            self.checkbox.setEnabled(True)
            self.toggle_target_res_group()
        else:
            self.checkbox.setChecked(False)
            self.checkbox.setEnabled(False)
            self.target_res_groupbox.setEnabled(False)
            self.smoothness_groupbox.setEnabled(True)

    def execute_smoothing(self):
        kernel = self.get_kernel()
        if kernel is None:
            return

        if self.original_data is None:
            self.original_data = self.fits_viewer.data.copy()

        self.perform_convolution(kernel)
        self.fits_viewer.update_cube()

        for window in self.subwindows:
            if window:
                window.data = self.fits_viewer.data
                window.update_cube()

        self.update_all_displays()

    def get_kernel(self):
        if self.gaussian_radio.isChecked():
            kernel_type = 'gaussian'
        elif self.boxcar_radio.isChecked():
            kernel_type = 'boxcar'
        else:
            kernel_type = 'gaussian'
    
        if not self.checkbox.isChecked():
            smoothness_x = self.x_spinbox.value()
            smoothness_y = self.y_spinbox.value()
            if self.subwindows:
                smoothness_z = self.z_spinbox.value()
            else:
                smoothness_z = 0

            if smoothness_x == 0 and smoothness_y == 0 and smoothness_z == 0:
                #QMessageBox.warning(self, 'Invalid Input', 'Smoothness of 0 is not acceptable for Gaussian kernel.')
                self.reset_smoothing()
                return None
                
            if kernel_type == 'gaussian':
                kx = Gaussian1DKernel(smoothness_x).array if smoothness_x > 0 else np.array([1.0])
                ky = Gaussian1DKernel(smoothness_y).array if smoothness_y > 0 else np.array([1.0])

                if self.data.ndim == 2 or smoothness_z == 0:
                    if smoothness_x > 0 and smoothness_y > 0:
                        kernel = Gaussian2DKernel(x_stddev=smoothness_x, y_stddev=smoothness_y)
                    elif smoothness_x > 0 and smoothness_y == 0:
                        kx = Gaussian1DKernel(smoothness_x).array
                        kernel_array = np.outer(np.array([1]), kx)
                        kernel_array /= kernel_array.sum()
                        kernel = CustomKernel(kernel_array)
                    elif smoothness_x == 0 and smoothness_y > 0:
                        ky = Gaussian1DKernel(smoothness_y).array
                        kernel_array = np.outer(ky, np.array([1]))
                        kernel_array /= kernel_array.sum()
                        kernel = CustomKernel(kernel_array)
                    
                    print(f"\n\nGaussian kernel size (FWHM): ({2.0 * np.sqrt(2.0 * np.log(2))*smoothness_x:.3g}, {2.0 * np.sqrt(2.0 * np.log(2))*smoothness_y:.3g}) pixels")
                
                elif self.data.ndim >= 3:
                    if smoothness_z > 0:
                        kz = Gaussian1DKernel(smoothness_z).array
                    else:
                        kz = np.array([1.0])

                    kernel_array = kz[:, None, None] * ky[None, :, None] * kx[None, None, :]
                    kernel_array /= kernel_array.sum()
                    kernel = CustomKernel(kernel_array)
                    print(f"\n\nGaussian kernel size (FWHM): ({2.0 * np.sqrt(2.0 * np.log(2))*smoothness_x:.3g}, {2.0 * np.sqrt(2.0 * np.log(2))*smoothness_y:.3g}, {2.0 * np.sqrt(2.0 * np.log(2))*smoothness_z:.3g}) pixels")


            elif kernel_type == 'boxcar':
                size_x = int(2 * smoothness_x + 1) if smoothness_x > 0 else 1
                size_y = int(2 * smoothness_y + 1) if smoothness_y > 0 else 1
                if self.data.ndim >= 3:
                    size_z = int(2 * smoothness_z + 1) if smoothness_z > 0 else 1
                else:
                    size_z = 1

                if size_x % 2 == 0:
                    size_x += 1
                if size_y % 2 == 0:
                    size_y += 1
                if self.data.ndim >= 3 and size_z % 2 == 0:
                    size_z += 1
        
                if self.data.ndim == 2:
                    kernel_array = np.ones((size_y, size_x))
                    print(f"\n\nBoxcar kernel size: ({size_x}, {size_y}) pixels")
                elif self.data.ndim >= 3:
                    kernel_array = np.ones((size_z, size_y, size_x))
                    print(f"\n\nBoxcar kernel size: ({size_x}, {size_y}, {size_z}) pixels")
                
                kernel_array /= kernel_array.sum()
                kernel = CustomKernel(kernel_array)
                
        elif self.checkbox.isChecked() and kernel_type == 'gaussian':
            kernel = self.calculate_gaussian_kernel()
            if kernel is None:
                return None
        else:
            QMessageBox.warning(self, 'Invalid Input', 'Boxcar kernel does not support target resolution smoothing.')
            return None
            
        return kernel

    def calculate_gaussian_kernel(self):
        try:
            current_bmaj = float(self.bmaj_current.text())  # in arcsec
            current_bmin = float(self.bmin_current.text())
            current_bpa = float(self.bpa_current.text())  # in degrees
            target_bmaj = float(self.bmaj_target.text())
            target_bmin = float(self.bmin_target.text())
            target_bpa = float(self.bpa_target.text())
        except ValueError:
            QMessageBox.warning(self, 'Invalid Input', 'Please enter valid numeric values for the Beam parameters.')
            return None
    
        pixel_scale_x = abs(self.header['CDELT1']) * 3600  # arcsec/pixel
        pixel_scale_y = abs(self.header['CDELT2']) * 3600
    
        if target_bmaj < target_bmin:
            target_bmaj, target_bmin = target_bmin, target_bmaj
    
        FWHM_to_sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2)))
        sigma_current_maj = current_bmaj * FWHM_to_sigma
        sigma_current_min = current_bmin * FWHM_to_sigma
        sigma_target_maj = target_bmaj * FWHM_to_sigma
        sigma_target_min = target_bmin * FWHM_to_sigma
    
        sigma_kernel_maj_sq = sigma_target_maj ** 2 - sigma_current_maj ** 2
        sigma_kernel_min_sq = sigma_target_min ** 2 - sigma_current_min ** 2
    
        if sigma_kernel_maj_sq <= 0 or sigma_kernel_min_sq <= 0:
            QMessageBox.warning(self, 'Invalid Input', "Target resolution must be larger than current resolution.")
            return None
    
        sigma_kernel_maj = np.sqrt(sigma_kernel_maj_sq)
        sigma_kernel_min = np.sqrt(sigma_kernel_min_sq)
    
        sigma_kernel_x = sigma_kernel_min / pixel_scale_x
        sigma_kernel_y = sigma_kernel_maj / pixel_scale_y
        theta = np.deg2rad(target_bpa - current_bpa)
    
        fwhm_kernel_x = sigma_kernel_x / FWHM_to_sigma
        fwhm_kernel_y = sigma_kernel_y / FWHM_to_sigma
    
        print(f"\n\nCalculated Gaussian kernel size (FWHM): ({fwhm_kernel_y:.3g}, {fwhm_kernel_x:.3g}) pixels")
        print(f"Kernel rotation angle: {np.rad2deg(theta):.3g} deg")
        kernel = Gaussian2DKernel(x_stddev=sigma_kernel_x, y_stddev=sigma_kernel_y, theta=theta)
    
        # Update the header with new beam parameters
        self.new_bmaj = target_bmaj / 3600  # Convert back to degrees
        self.new_bmin = target_bmin / 3600
        self.new_bpa = target_bpa
    
        self.header['BMAJ'] = float(self.new_bmaj)
        self.header['BMIN'] = float(self.new_bmin)
        self.header['BPA'] = float(self.new_bpa)
    
        return kernel


    def perform_convolution(self, kernel):
        if self.data.ndim in [2, 3]:
            data = self.data
            if self.nan_checkbox.isChecked():
                smoothed_data = self.convolve_with_nan(data, kernel)
            else:
                smoothed_data = self.convolve_without_nan(data, kernel)
            self.fits_viewer.data = smoothed_data
        elif self.data.ndim == 4:
            data = self.data
            smoothed_data = np.empty_like(data)
            for i in range(data.shape[0]):
                if self.nan_checkbox.isChecked():
                    smoothed_data[i] = self.convolve_with_nan(data[i], kernel)
                else:
                    smoothed_data[i] = self.convolve_without_nan(data[i], kernel)
            self.fits_viewer.data = smoothed_data
        else:
            QMessageBox.warning(self, 'Invalid Data', 'Data dimensionality not supported.')
            return
    
    def convolve_with_nan(self, data, kernel):
        num_nonspatial = data.ndim - len(kernel.shape)
        pad_width = [(0, 0)] * num_nonspatial + [(s // 2, s // 2) for s in kernel.shape]
        
        data_padded = np.pad(data, pad_width=pad_width, mode='reflect')
    
        # Extend kernel dimensions to match data dimensions
        kernel_nd = kernel.array
        while kernel_nd.ndim < data_padded.ndim:
            kernel_nd = kernel_nd[None, ...]
    
        # Perform convolution with NaN handling
        current_time = time.time()
        print("Convolving kernel...")
        smoothed_data_padded = convolve_fft(
            data_padded,
            kernel_nd,
            nan_treatment='interpolate',
            preserve_nan=True,
            normalize_kernel=True,
            boundary='fill',
            fill_value=0.0,
            allow_huge=True
        )
        elupsetime = time.time()-current_time
        print(f"Smoothing done in {elupsetime:.3g} sec")
        # Remove padding
        slices = [slice(None)] * num_nonspatial + [slice(pad[0], -pad[1] if pad[1] != 0 else None) 
                                                for pad in pad_width[num_nonspatial:]]
        smoothed_data = smoothed_data_padded[tuple(slices)]
        
        return smoothed_data
        
    
    def convolve_without_nan(self, data, kernel):
        # NaN to Zero
        data = np.nan_to_num(data, nan=0.0)
    
        if data.ndim != len(kernel.shape):
            pad_width = [(0, 0)] * (data.ndim - len(kernel.shape)) + [(s // 2, s // 2) for s in kernel.shape]
        else:
            pad_width = [(s // 2, s // 2) for s in kernel.shape]
    
        data_padded = np.pad(data, pad_width=pad_width, mode='reflect')
    
        kernel_nd = kernel.array
        while kernel_nd.ndim < data_padded.ndim:
            kernel_nd = kernel_nd[None, ...]
        
        current_time = time.time()
        print("Convolving kernel...")
        smoothed_data_padded = fftconvolve(data_padded, kernel_nd, mode='same')
        elupsetime = time.time()-current_time
        print(f"Smoothing done in {elupsetime:.3g} sec")
        slices = [slice(p[0], -p[1] if p[1] != 0 else None) for p in pad_width]
        smoothed_data = smoothed_data_padded[tuple(slices)]
        
        return smoothed_data
        

    def reset_smoothing(self):
        # Reset to the state when the panel was opened
        if self.original_data is not None:
            self.fits_viewer.data = self.original_data.copy() # Reset data to original
            self.fits_viewer.update_cube()
            for window in self.subwindows:
                if window:
                    window.data = self.fits_viewer.data # Reset data for each subwindow
                    window.update_cube()

            # Update all displays
            self.update_all_displays()
        
        
        if self.bmaj_exists:
            self.header['BMAJ'] = self.original_bmaj
        if self.bmin_exists:
            self.header['BMIN'] = self.original_bmin
        if self.bpa_exists:
            self.header['BPA'] = self.original_bpa
            
        if not self.bmaj_exists and 'BMAJ' in self.header:
             del self.header['BMAJ']
        if not self.bmin_exists and 'BMIN' in self.header:
             del self.header['BMIN']
        if not self.bpa_exists and 'BPA' in self.header:
             del self.header['BPA']
        
        print("\n\nReset")
            

    def update_all_displays(self):
        # Get the current channel index for the main viewer
        current_channel_main = self.fits_viewer.current_channel_index()
    
        # Update the display for the main viewer
        if self.fits_viewer.data.ndim == 4:
            if self.fits_viewer.plane == 'xy':
                data_slice = self.fits_viewer.data[0, current_channel_main, :, :] 
            elif self.fits_viewer.plane == 'xz':
                data_slice = self.fits_viewer.data[0, :, current_channel_main, :]
            elif self.fits_viewer.plane == 'zy':
                data_slice = self.fits_viewer.data[0, :, :, current_channel_main].T
            else:
                data_slice = self.fits_viewer.data[0, 0]  
    
        elif self.fits_viewer.data.ndim == 3:
            if self.fits_viewer.plane == 'xy':
                data_slice = self.fits_viewer.data[current_channel_main, :, :]
            elif self.fits_viewer.plane == 'xz':
                data_slice = self.fits_viewer.data[:, current_channel_main, :]
            elif self.fits_viewer.plane == 'zy':
                data_slice = self.fits_viewer.data[:, :, current_channel_main].T
            else:
                data_slice = self.fits_viewer.data[0]
    
        elif self.fits_viewer.data.ndim == 2:
            data_slice = self.fits_viewer.data
        else:
            raise TypeError(f"Invalid number of dimensions: {self.fits_viewer.data.ndim}")
    
        # Update the image data and color limits
        if data_slice.ndim == 2:
            self.fits_viewer.im.set_data(data_slice)
        else:
            raise TypeError(f"Invalid data shape {data_slice.shape} for 2D image display.")
    
        self.fits_viewer.canvas.draw()
    
        # Update the displays for all subwindows
        for window in self.subwindows:
            if window:
                current_channel = window.current_channel_index()
                if window.data.ndim == 4:
                    if window.plane == 'xy':
                        data_slice = window.data[0, current_channel, :, :]
                    elif window.plane == 'xz':
                        data_slice = window.data[0, :, current_channel, :]
                    elif window.plane == 'zy':
                        data_slice = window.data[0, :, :, current_channel].T
                    else:
                        data_slice = window.data[0, 0]
                        
                elif window.data.ndim == 3:
                    if window.plane == 'xy':
                        data_slice = window.data[current_channel, :, :]
                    elif window.plane == 'xz':
                        data_slice = window.data[:, current_channel, :]
                    elif window.plane == 'zy':
                        data_slice = window.data[:, :, current_channel].T
                    else:
                        data_slice = window.data[0]
    
                elif window.data.ndim == 2:
                    data_slice = window.data
                else:
                    raise TypeError(f"Invalid number of dimensions: {window.data.ndim}")
    
                if data_slice.ndim == 2:
                    window.im.set_data(data_slice)
                else:
                    raise TypeError(f"Invalid data shape {data_slice.shape} for 2D image display.")
                    
                window.canvas.draw()

    def save_fits(self):
        self.execute_smoothing()
        data_min =  float(np.nanmin(self.fits_viewer.data))
        data_max =  float(np.nanmax(self.fits_viewer.data))
        new_header = self.fits_viewer.header.copy()
        original_filename = self.fits_viewer.filename
        
        new_header['DATAMIN'] = data_min
        new_header['DATAMAX'] = data_max

        if  self.checkbox.isChecked():
            new_header['BMAJ'] = float(self.new_bmaj)
            new_header['BMIN'] = float(self.new_bmin)
            new_header['BPA'] = float(self.new_bpa)

        save_fits = SaveFITS(self.fits_viewer.data, new_header, original_filename)
        save_fits.save(suffix="sm")

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
