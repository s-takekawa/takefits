from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QGridLayout, QLineEdit, 
                             QPushButton, QLabel, QHBoxLayout, QMessageBox, 
                             QRadioButton, QGroupBox, QFileDialog)
from astropy.io import fits
from astropy.wcs import WCS
from core.save_fits import SaveFITS
import numpy as np
import os
from datetime import datetime

class MaskSettingsPanel(QWidget):
    def __init__(self, fits_viewer, subwindows):
        super().__init__()
        self.fits_viewer = fits_viewer
        self.subwindows = subwindows
        self.original_data = None  # Defer copying
        self.setWindowTitle(f'Mask Settings: {self.fits_viewer.filename}')
        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()
        grid_layout = QGridLayout()
        threshold_group = QGroupBox("Create Mask by Threshold")
        threshold_grid = QGridLayout(threshold_group)
        self.threshold_label = QLabel('Value:')
        self.threshold_input = QLineEdit(self)
        self.threshold_input.setPlaceholderText("Threshold")
        self.mask_below_radio = QRadioButton("Mask < Value")
        self.mask_above_radio = QRadioButton("Mask > Value")
        self.mask_below_radio.setChecked(True)
        threshold_grid.addWidget(self.threshold_label, 0, 0)
        threshold_grid.addWidget(self.threshold_input, 0, 1)
        threshold_grid.addWidget(self.mask_below_radio, 1, 0)
        threshold_grid.addWidget(self.mask_above_radio, 1, 1)
        grid_layout.addWidget(threshold_group, 0, 0, 1, 2)
        external_mask_group = QGroupBox("Apply External Mask")
        external_mask_layout = QHBoxLayout(external_mask_group)
        self.load_mask_button = QPushButton('Load Mask FITS', self)
        external_mask_layout.addWidget(self.load_mask_button)
        grid_layout.addWidget(external_mask_group, 1, 0, 1, 2)
        self.apply_button = QPushButton('Apply Threshold', self)
        self.reset_button = QPushButton('Reset All Masks', self)
        grid_layout.addWidget(self.apply_button, 2, 0)
        grid_layout.addWidget(self.reset_button, 2, 1)
        self.save_masked_button = QPushButton('Save Masked FITS', self)
        self.save_mask_01_button = QPushButton('Save Mask as FITS', self)
        grid_layout.addWidget(self.save_masked_button, 3, 0)
        grid_layout.addWidget(self.save_mask_01_button, 3, 1)
        self.apply_button.clicked.connect(self.apply_threshold_mask)
        self.load_mask_button.clicked.connect(self.load_and_apply_mask)
        self.reset_button.clicked.connect(self.reset_mask)
        self.save_masked_button.clicked.connect(self.save_masked_fits)
        self.save_mask_01_button.clicked.connect(self.save_mask_as_fits)
        self.threshold_input.returnPressed.connect(self.apply_threshold_mask)

        self.apply_button.setAutoDefault(True) 
        self.apply_button.setDefault(True)
        self.threshold_input.setFocus()

        main_layout.addLayout(grid_layout)
        self.setLayout(main_layout)
        self.adjustSize()
        self.move_to_default_position()

    def move_to_default_position(self):
        if hasattr(self.fits_viewer, 'control_panel'):
            cp_geom = self.fits_viewer.control_panel.geometry()
            self.move(cp_geom.x() + cp_geom.width(), cp_geom.y())

    def apply_threshold_mask(self):
        if self.original_data is None:
            self.original_data = self.fits_viewer.data.copy()
        try:
            threshold = float(self.threshold_input.text())
        except ValueError:
            QMessageBox.warning(self, 'Input Error', 'Please enter a valid numeric threshold.')
            return
        try:
            masked_data = self.original_data.copy()
            if self.mask_below_radio.isChecked():
                masked_data[masked_data < threshold] = np.nan
            else:
                masked_data[masked_data > threshold] = np.nan
            if np.all(np.isnan(masked_data)):
                QMessageBox.warning(self, 'Display Warning', 
                    'All pixels were masked by the threshold.\nDisplay might be empty, but the mask is applied.')
            self._update_data_and_displays(masked_data, "[MASK ACTIVE]")
        except Exception as e:
            QMessageBox.critical(self, 'Processing Error', f'An error occurred: {e}')

    def load_and_apply_mask(self):
        if self.original_data is None:
            self.original_data = self.fits_viewer.data.copy()
        filename, _ = QFileDialog.getOpenFileName(self, 'Open Mask FITS', '', 'FITS Files (*.fits *.fit)')
        if not filename: return
        try:
            with fits.open(filename) as hdul:
                mask_data, mask_header = hdul[0].data, hdul[0].header
            mask_wcs = WCS(mask_header)
            if mask_data.shape != self.original_data.shape:
                QMessageBox.critical(self, 'Shape Mismatch', f'Mask dimensions ({mask_data.shape}) do not match data dimensions ({self.original_data.shape}).')
                return
            if self.fits_viewer.wcs.to_header_string() != mask_wcs.to_header_string():
                reply = QMessageBox.warning(self, 'WCS Mismatch', "Mask coordinates may not match data.\nApply anyway (pixel-to-pixel)?",
                                            QMessageBox.StandardButton.Apply | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Cancel)
                if reply == QMessageBox.StandardButton.Cancel: return
            masked_data = self.original_data.copy()
            masked_data[mask_data == 0] = np.nan
            self._update_data_and_displays(masked_data, f"[EXTERNAL MASK: {os.path.basename(filename)}]")
        except Exception as e:
            QMessageBox.critical(self, 'File Error', f'Failed to load or apply mask: {e}')

    def reset_mask(self):
        if self.original_data is not None:
            self._update_data_and_displays(self.original_data)

    def _update_data_and_displays(self, new_data, status_message=None):
        self.fits_viewer.data = new_data
        self.fits_viewer.update_cube()
        for window in self.subwindows:
            if window:
                window.data = new_data
                window.update_cube()
        self.update_all_displays()
        if status_message:
            self.fits_viewer.setWindowTitle(f"{status_message} {self.fits_viewer.original_window_title}")
        else:
            self.fits_viewer.setWindowTitle(self.fits_viewer.original_window_title)

    def update_all_displays(self):
        all_windows = [self.fits_viewer] + self.subwindows
        for window in all_windows:
            if not window: continue
            current_channel = window.current_channel_index()
            data_slice = None
            if window.data.ndim == 4:
                if window.plane == 'xy': data_slice = window.data[0, current_channel, :, :]
                elif window.plane == 'xz': data_slice = window.data[0, :, current_channel, :]
                elif window.plane == 'zy': data_slice = window.data[0, :, :, current_channel].T
            elif window.data.ndim == 3:
                if window.plane == 'xy': data_slice = window.data[current_channel, :, :]
                elif window.plane == 'xz': data_slice = window.data[:, current_channel, :]
                elif window.plane == 'zy': data_slice = window.data[:, :, current_channel].T
            elif window.data.ndim == 2:
                data_slice = window.data
            if data_slice is not None:
                window.im.set_data(data_slice)
                window.canvas.draw()
                
    def _get_sanitized_header(self):
        new_header = self.fits_viewer.header.copy()
        try:
            original_header_on_disk = fits.getheader(self.fits_viewer.filename)
            if 'CUNIT3' not in original_header_on_disk and 'CUNIT3' in new_header:
                del new_header['CUNIT3']
        except Exception as e:
            print(f"Could not check original header on disk. Error: {e}")
        return new_header

    def save_masked_fits(self):
        try:
            threshold_text = self.threshold_input.text()
            if not threshold_text:
                raise ValueError("Threshold input is empty.")
            threshold = float(threshold_text)
            condition = "<" if self.mask_below_radio.isChecked() else ">"
            
            new_header = self._get_sanitized_header()
            new_header['DATAMIN'] = float(np.nanmin(self.fits_viewer.data))
            new_header['DATAMAX'] = float(np.nanmax(self.fits_viewer.data))
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            new_header.add_history(f"Data masked using takefits on {timestamp}")
            new_header.add_history(f"Threshold applied: mask pixels with value {condition} {threshold}")

            save_dialog = SaveFITS(self.fits_viewer.data, new_header, self.fits_viewer.filename)
            save_dialog.save(suffix="masked")
        except ValueError:
            QMessageBox.warning(self, 'Input Error', 'A valid numeric threshold must be in the input box to save with history.')
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'An error occurred while saving the masked FITS: {e}')

    def save_mask_as_fits(self):
        if self.original_data is None:
            QMessageBox.warning(self, 'No Mask', 'Please apply a threshold mask first before saving it.')
            return
        try:
            threshold = float(self.threshold_input.text())
            mask_data = np.ones(self.original_data.shape, dtype=np.float32)

            if self.mask_below_radio.isChecked():
                condition = f"< {threshold}"
                mask_data[self.original_data < threshold] = 0
            else:
                condition = f"> {threshold}"
                mask_data[self.original_data > threshold] = 0

            new_header = self.fits_viewer.header.copy()
            
            for key in ['SIMPLE', 'BITPIX', 'NAXIS', 'EXTEND', 'BSCALE', 'BZERO', 'DATAMIN', 'DATAMAX', 'BUNIT']:
                if key in new_header:
                    del new_header[key]

            for i in range(1, new_header.get('NAXIS', 0) + 1):
                if f'NAXIS{i}' in new_header:
                    del new_header[f'NAXIS{i}']

            hdu = fits.PrimaryHDU(data=mask_data, header=new_header)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            hdu.header.add_history("Mask file (1=unmasked, 0=masked) generated by takefits")
            hdu.header.add_history(f"on {timestamp}")
            hdu.header.add_history(f"Source file: {os.path.basename(self.fits_viewer.filename)}")
            hdu.header.add_history(f"Threshold condition: mask pixels with value {condition}")

            save_dialog = SaveFITS(hdu.data, hdu.header, self.fits_viewer.filename)
            save_dialog.save(suffix="_mask_")

        except ValueError:
            QMessageBox.warning(self, 'Input Error', 'A valid numeric threshold is required to generate the mask file.')
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'An error occurred while saving the mask file: {e}')

    def closeEvent(self, event):
        self.destroyed.emit()
        super().closeEvent(event)
