from PySide6.QtWidgets import QDialog, QGridLayout, QGroupBox, QButtonGroup, QVBoxLayout, QRadioButton, QCheckBox, QLabel, QPushButton, QLineEdit, QDoubleSpinBox, QMessageBox
from PySide6.QtCore import Qt
from PySide6.QtGui import QDoubleValidator
import numpy as np
from takefits.ui.save_fits_dialog import SaveFITS
from takefits.core import usecases
from takefits.core.history_provenance import build_processing_history_lines
from takefits.tools.base_panel import clear_action_preview_record, confirm_pending_close, has_action_record_tag, record_action_preview
import time
import os

class SmoothSettingsPanel(QDialog):
    def __init__(self, fits_viewer, subwindows):
        super().__init__()
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.fits_viewer = fits_viewer
        self.header = self.fits_viewer.header
        self.wcs = self.fits_viewer.wcs
        self.subwindows = subwindows
        
        self.data = self.fits_viewer.data
        self.original_data = None  # Defer copying until first execution
        self._has_pending_changes = False
        self._action_record_tag = "panel:smoothing"
        self._last_action_name = None
        self._last_action_params = None
        
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
        
        self.kernel_selection_changed()
        self.resync_after_workspace_restore()
        
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
        self.hanning_radio = QRadioButton("Hanning")
        self.gaussian_radio.setChecked(True)
        self.hanning_radio.setEnabled(self.data.ndim == 3)
        self.hanning_radio.setToolTip("Available only for 3D cubes")

        self.kernel_button_group = QButtonGroup()
        self.kernel_button_group.addButton(self.gaussian_radio)
        self.kernel_button_group.addButton(self.boxcar_radio)
        self.kernel_button_group.addButton(self.hanning_radio)

        kernel_layout.addWidget(self.gaussian_radio)
        kernel_layout.addWidget(self.boxcar_radio)
        kernel_layout.addWidget(self.hanning_radio)
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


        self.setLayout(grid_layout)

        self.checkbox.stateChanged.connect(self.toggle_target_res_group)
        self.checkbox.setChecked(False)
        self.target_res_groupbox.setEnabled(False)

        self.smoothness_groupbox.setEnabled(True)

        self.gaussian_radio.toggled.connect(self.kernel_selection_changed)
        self.boxcar_radio.toggled.connect(self.kernel_selection_changed)
        self.hanning_radio.toggled.connect(self.kernel_selection_changed)

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

    def kernel_selection_changed(self, _checked=None):
        if self.gaussian_radio.isChecked():
            self.checkbox.setEnabled(True)
            self.toggle_target_res_group()
            return

        self.checkbox.setChecked(False)
        self.checkbox.setEnabled(False)
        self.target_res_groupbox.setEnabled(False)
        self.bmaj_current.setReadOnly(True)
        self.bmin_current.setReadOnly(True)
        self.bpa_current.setReadOnly(True)

        if self.boxcar_radio.isChecked():
            self.smoothness_groupbox.setEnabled(True)
        elif self.hanning_radio.isChecked():
            self.smoothness_groupbox.setEnabled(False)
        else:
            self.smoothness_groupbox.setEnabled(True)

    def execute_smoothing(self):
        if self.original_data is None:
            self.original_data = self.fits_viewer.data.copy()

        # Determine kernel type
        if self.gaussian_radio.isChecked():
            kernel_type = 'gaussian'
        elif self.boxcar_radio.isChecked():
            kernel_type = 'boxcar'
        else:
            kernel_type = 'hanning'

        if self.checkbox.isChecked() and kernel_type == 'gaussian':
            # Target resolution smoothing
            smoothed_data = self._execute_target_resolution_smoothing()
            if smoothed_data is None:
                return
        else:
            # Normal smoothing with pixel-based kernel size
            smoothed_data = self._execute_normal_smoothing(kernel_type)
            if smoothed_data is None:
                return

        # Update data in viewer and subwindows
        self.fits_viewer.data = smoothed_data
        self.fits_viewer.update_cube()

        for window in self.subwindows:
            if window:
                window.data = self.fits_viewer.data
                window.update_cube()

        self._has_pending_changes = True
        self._sync_app_state_data()
        self.update_all_displays()
        self._refresh_main_hpbw_overlay()
        self._record_preview_action()

    def _execute_normal_smoothing(self, kernel_type: str):
        """Execute normal smoothing using usecase layer."""
        self._last_action_name = "apply_smoothing"
        smoothness_x = self.x_spinbox.value()
        smoothness_y = self.y_spinbox.value()
        smoothness_z = self.z_spinbox.value() if self.subwindows else 0

        if kernel_type == 'hanning':
            if self.original_data.ndim != 3:
                QMessageBox.warning(self, 'Unsupported Data', 'Hanning is available only for 3D cubes.')
                return None
            self._last_action_params = {
                "kernel_type": kernel_type,
            }
            print("\n\nHanning kernel (velocity axis): [0.25, 0.5, 0.25] channels")
        else:
            self._last_action_params = {
                "kernel_type": kernel_type,
                "smoothness_x": float(smoothness_x),
                "smoothness_y": float(smoothness_y),
                "smoothness_z": float(smoothness_z),
            }

            if smoothness_x == 0 and smoothness_y == 0 and smoothness_z == 0:
                self.reset_smoothing()
                return None

            # Log kernel info
            if kernel_type == 'gaussian':
                fwhm_factor = 2.0 * np.sqrt(2.0 * np.log(2))
                print(f"\n\nGaussian kernel size (FWHM): ({fwhm_factor*smoothness_x:.3g}, {fwhm_factor*smoothness_y:.3g}, {fwhm_factor*smoothness_z:.3g}) pixels")
            else:
                size_x = int(2 * smoothness_x + 1) if smoothness_x > 0 else 1
                size_y = int(2 * smoothness_y + 1) if smoothness_y > 0 else 1
                size_z = int(2 * smoothness_z + 1) if smoothness_z > 0 else 1
                print(f"\n\nBoxcar kernel size: ({size_x}, {size_y}, {size_z}) pixels")

        # Call usecase layer
        current_time = time.time()
        print("Convolving kernel...")

        # Handle 4D data by applying to each slice
        data = self.original_data
        if data.ndim == 4:
            smoothed_data = np.empty_like(data)
            for i in range(data.shape[0]):
                smoothed_data[i] = usecases.compute_smoothed(
                    data[i],
                    kernel_type=kernel_type,
                    smoothness_x=smoothness_x,
                    smoothness_y=smoothness_y,
                    smoothness_z=smoothness_z,
                    handle_nan=True,
                )
        else:
            smoothed_data = usecases.compute_smoothed(
                data,
                kernel_type=kernel_type,
                smoothness_x=smoothness_x,
                smoothness_y=smoothness_y,
                smoothness_z=smoothness_z,
                handle_nan=True,
            )

        elapsed_time = time.time() - current_time
        print(f"Smoothing done in {elapsed_time:.3g} sec")
        return smoothed_data

    def _execute_target_resolution_smoothing(self):
        """Execute target resolution smoothing using usecase layer."""
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

        # Swap if needed
        if target_bmaj < target_bmin:
            target_bmaj, target_bmin = target_bmin, target_bmaj
            self.bmaj_target.setText(f"{target_bmaj:.3g}")
            self.bmin_target.setText(f"{target_bmin:.3g}")

        self._last_action_name = "apply_smoothing_to_resolution"
        self._last_action_params = {
            "target_bmaj": float(target_bmaj),
            "target_bmin": float(target_bmin),
            "target_bpa": float(target_bpa),
            "current_bmaj": float(current_bmaj),
            "current_bmin": float(current_bmin),
            "current_bpa": float(current_bpa),
        }

        current_time = time.time()
        print("Convolving kernel...")

        try:
            smoothed_data, new_beam = usecases.compute_smoothed_to_resolution(
                self.original_data,
                self.header,
                target_bmaj=target_bmaj,
                target_bmin=target_bmin,
                target_bpa=target_bpa,
                current_bmaj=current_bmaj,
                current_bmin=current_bmin,
                current_bpa=current_bpa
            )
        except ValueError as e:
            QMessageBox.warning(self, 'Invalid Input', str(e))
            return None

        elapsed_time = time.time() - current_time
        print(f"Smoothing done in {elapsed_time:.3g} sec")

        # Store new beam parameters for save_fits
        self.new_bmaj = new_beam['BMAJ']
        self.new_bmin = new_beam['BMIN']
        self.new_bpa = new_beam['BPA']

        # Update header
        self.header['BMAJ'] = float(self.new_bmaj)
        self.header['BMIN'] = float(self.new_bmin)
        self.header['BPA'] = float(self.new_bpa)

        # Log kernel info
        FWHM_to_sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2)))
        pixel_scale_x = abs(self.header['CDELT1']) * 3600
        pixel_scale_y = abs(self.header['CDELT2']) * 3600

        sigma_kernel_maj_sq = (target_bmaj * FWHM_to_sigma) ** 2 - (current_bmaj * FWHM_to_sigma) ** 2
        sigma_kernel_min_sq = (target_bmin * FWHM_to_sigma) ** 2 - (current_bmin * FWHM_to_sigma) ** 2
        sigma_kernel_x = np.sqrt(sigma_kernel_min_sq) / pixel_scale_x
        sigma_kernel_y = np.sqrt(sigma_kernel_maj_sq) / pixel_scale_y
        theta = target_bpa - current_bpa

        fwhm_kernel_x = sigma_kernel_x / FWHM_to_sigma
        fwhm_kernel_y = sigma_kernel_y / FWHM_to_sigma
        print(f"\n\nCalculated Gaussian kernel size (FWHM): ({fwhm_kernel_y:.3g}, {fwhm_kernel_x:.3g}) pixels")
        print(f"Kernel rotation angle: {theta:.3g} deg")

        return smoothed_data

    def reset_smoothing(self):
        preferred_cursor = self._capture_preferred_cursor_snapshot()
        removed_preview = self._clear_preview_action()
        restored_from_history = False
        if removed_preview:
            restored_from_history = self._restore_state_from_action_history(preferred_cursor=preferred_cursor)

        # Fallback path: reset to the state when the panel was opened.
        if not restored_from_history and self.original_data is not None:
            self.fits_viewer.data = self.original_data.copy()
            self.fits_viewer.update_cube()
            for window in self.subwindows:
                if window:
                    window.data = self.fits_viewer.data
                    window.update_cube()

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

            self._sync_app_state_data()
            self.update_all_displays()
            self._restore_preferred_cursor_snapshot(preferred_cursor)

        # Keep panel references synchronized with the latest viewer state.
        self.data = self.fits_viewer.data
        self.header = self.fits_viewer.header
        self.wcs = self.fits_viewer.wcs
        self._refresh_current_resolution_display()
        self._refresh_main_hpbw_overlay()

        self._has_pending_changes = False
        self._last_action_name = None
        self._last_action_params = None
        
        print("\n\nReset")

    def _record_preview_action(self):
        if not self._last_action_name or not isinstance(self._last_action_params, dict):
            return
        record_action_preview(
            self.fits_viewer,
            self._last_action_name,
            dict(self._last_action_params),
            replace_tag=self._action_record_tag,
        )

    def _clear_preview_action(self):
        removed = clear_action_preview_record(
            self.fits_viewer,
            self._action_record_tag,
            action_name="apply_smoothing",
        )
        if removed:
            return True
        return bool(
            clear_action_preview_record(
                self.fits_viewer,
                self._action_record_tag,
                action_name="apply_smoothing_to_resolution",
            )
        )

    def _capture_preferred_cursor_snapshot(self):
        main_window = getattr(self.fits_viewer, 'main_window', None) or self.fits_viewer
        capture = getattr(main_window, "_capture_shared_cursor_snapshot", None)
        if callable(capture):
            try:
                snapshot = capture()
                if isinstance(snapshot, dict) and snapshot:
                    return dict(snapshot)
            except Exception:
                pass
        try:
            return {"zpix": int(self.fits_viewer.current_channel_index())}
        except Exception:
            return None

    def _restore_preferred_cursor_snapshot(self, snapshot) -> None:
        if not isinstance(snapshot, dict) or not snapshot:
            return
        main_window = getattr(self.fits_viewer, 'main_window', None) or self.fits_viewer
        zpix = snapshot.get("zpix")
        if zpix is None:
            return
        try:
            target_z = int(round(float(zpix)))
        except Exception:
            return
        update_channel = getattr(main_window, "update_channel", None)
        if callable(update_channel):
            try:
                update_channel("xy", target_z)
            except Exception:
                pass

    def _restore_state_from_action_history(self, *, preferred_cursor=None) -> bool:
        main_window = getattr(self.fits_viewer, 'main_window', None) or self.fits_viewer
        session = getattr(main_window, 'action_session', None)
        if session is None:
            return False
        replay_to_cursor = getattr(session, "_replay_to_cursor", None)
        apply_state = getattr(main_window, "_apply_action_session_state_to_viewers", None)
        if not callable(replay_to_cursor) or not callable(apply_state):
            return False
        try:
            history = list(getattr(session, "history", []) or [])
            cursor = int(getattr(session, "cursor", len(history)))
            cursor = max(0, min(cursor, len(history)))
        except Exception:
            return False
        try:
            replay_to_cursor(cursor)
            if isinstance(preferred_cursor, dict) and preferred_cursor:
                apply_state(preferred_cursor=dict(preferred_cursor))
            else:
                apply_state()
            return True
        except Exception:
            return False

    def _refresh_current_resolution_display(self):
        header = getattr(self, "header", None)
        if header is None:
            return
        try:
            if 'BMAJ' in header:
                self.bmaj_current.setText(f"{float(header['BMAJ']) * 3600:.3g}")
            else:
                self.bmaj_current.clear()
        except Exception:
            pass
        try:
            if 'BMIN' in header:
                self.bmin_current.setText(f"{float(header['BMIN']) * 3600:.3g}")
            else:
                self.bmin_current.clear()
        except Exception:
            pass
        try:
            if 'BPA' in header:
                self.bpa_current.setText(f"{float(header['BPA']):.3g}")
            else:
                self.bpa_current.clear()
        except Exception:
            pass

    def _sync_app_state_data(self):
        """Keep MainWindow.app_state data in sync with current cube after smoothing."""
        main_window = getattr(self.fits_viewer, 'main_window', None) or self.fits_viewer
        if hasattr(main_window, 'sync_app_state_data'):
            main_window.sync_app_state_data(
                data=self.fits_viewer.data,
                header=self.fits_viewer.header,
                wcs=self.fits_viewer.wcs,
            )
            return
        app_state = getattr(main_window, 'app_state', None)
        if app_state is not None:
            app_state.data = self.fits_viewer.data
            app_state.header = self.fits_viewer.header
            app_state.wcs = self.fits_viewer.wcs

    def _refresh_main_hpbw_overlay(self):
        hpbw = getattr(self.fits_viewer, 'hpbw', None)
        if hpbw is None:
            return
        try:
            hpbw.header = self.fits_viewer.header
        except Exception:
            pass
        refresh = getattr(hpbw, 'refresh_geometry_from_header', None)
        if callable(refresh):
            try:
                refresh()
            except Exception:
                pass
        redraw_overlay = getattr(self.fits_viewer, 'redraw_main_overlay_and_blit', None)
        if callable(redraw_overlay):
            try:
                redraw_overlay()
                return
            except Exception:
                pass
        canvas = getattr(self.fits_viewer, 'canvas', None)
        if canvas is not None:
            try:
                canvas.draw_idle()
            except Exception:
                pass
            

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

        if self.checkbox.isChecked():
            new_header['BMAJ'] = float(self.new_bmaj)
            new_header['BMIN'] = float(self.new_bmin)
            new_header['BPA'] = float(self.new_bpa)

        # Use common centralized processing history

        for entry in build_processing_history_lines(self.fits_viewer):
            new_header.add_history(entry)

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
        if self._has_pending_changes:
            choice = confirm_pending_close(
                self,
                "Close Smoothing Panel",
                "There are unapplied smoothing changes.",
            )
            if choice == "cancel":
                event.ignore()
                return
            if choice == "discard":
                self.reset_smoothing()
        super().closeEvent(event)

    def resync_after_workspace_restore(self):
        self._has_pending_changes = bool(has_action_record_tag(self.fits_viewer, self._action_record_tag))
