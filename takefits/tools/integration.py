import numpy as np
import weakref
import matplotlib as mpl
import os
import json
import uuid
from typing import List, Optional
from PySide6.QtWidgets import QWidget, QMainWindow, QDialog, QGridLayout, QGroupBox, QVBoxLayout, QComboBox, QLineEdit, QPushButton, QRadioButton, QCheckBox, QLabel, QButtonGroup, QMessageBox, QFileDialog
from PySide6.QtCore import Qt, QTimer, QSignalBlocker
from PySide6.QtGui import QGuiApplication, QKeySequence, QShortcut
from takefits.core.coordinate import CoordinateConverter
from takefits.core.region_manager import RegionManager
from takefits.core.contour_manager import ContourManager, ContourItem
from takefits.core.marker_manager import MarkerManager
from matplotlib.figure import Figure
from takefits.tools.color_scale import ColorSettingsPanel, ColorMode
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from takefits.ui.navigation_toolbar import MyNavigationToolbar
from takefits.core.plotting.display_map import TransparentOverlayAxes
from takefits.core.coordinate import Format_pix_to_wcs
from takefits.core.colorbar_layout import compute_colorbar_geometry, orientation_for_placement
from takefits.core.click_label_layout import compute_click_label_geometry
from takefits.logic.add_hpbw import AddHPBW
from astropy.io import fits
from takefits.core.usecases import compute_moment, export_moment_fits, AppState
from takefits.core.app_state import MarkerSpec, RegionSpec, create_app_state
from takefits.core.action_session import ActionSession
from takefits.core.actions import ActionRegistry, register_default_actions
from takefits.core.annotation_serialization import (
    build_marker_payload_from_specs,
    build_region_payload_from_specs,
    snapshot_marker_specs,
    snapshot_region_specs,
)
from takefits.core.history_provenance import build_processing_history_lines
from takefits.tools.base_panel import (
    clear_action_preview_record,
    confirm_pending_close,
    record_action_preview,
)

from takefits.core.region import CircleRegion, RectangleRegion, EllipseRegion, CubeRegion


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
        self._action_record_tag = "panel:integration"
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
            "RMS",
            "Sigma (Std Dev)"
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
        
        # Set pixel dimensions for range validation
        self.znpix = self.data.shape[0] - 1
        self.ynpix = self.data.shape[1] - 1
        self.xnpix = self.data.shape[2] - 1
    
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


    def execute_integration(self):
        self._set_integ_data()
        # Sync integration range to app_state (usecase layer bridge)
        if not self.sync_integration_range_to_app_state():
            return
        
        # Prepare clipping threshold
        clip_threshold = None
        if self.clip_checkbox.isChecked():
            try:
                clip_threshold = float(self.clip_input.text())
            except ValueError:
                QMessageBox.warning(self, 'Error', 'Invalid clipping value provided!')
                return
        
        # Prepare params
        app_state = self.get_app_state()

        # Check which mode is active based on combo box or radio buttons
        if self.others_combo.currentIndex() > -1:
            # Combo box is selected
            mode_text = self.others_combo.currentText()
            
            if mode_text == "Average":
                self.integ_mode = 'average'
                self.integrated_data = compute_moment(app_state, moment_type="average", axis=self.integ_axis, clip_threshold=clip_threshold)
                window = self.show_in_new_window(f"Average: {self.min_input} to {self.max_input}")
                if window is not None:
                    self._record_current_integration_action(
                        clip_threshold,
                        action_tag=getattr(window, "_workspace_action_tag", None),
                    )
            elif mode_text == "Peak Int.":
                self.integ_mode = 'peak_int'
                self.integrated_data = compute_moment(app_state, moment_type="peak", axis=self.integ_axis, clip_threshold=clip_threshold)
                window = self.show_in_new_window(f"Peak Int.: {self.min_input} to {self.max_input}")
                if window is not None:
                    self._record_current_integration_action(
                        clip_threshold,
                        action_tag=getattr(window, "_workspace_action_tag", None),
                    )
            elif mode_text == "Peak Coord.":
                self.integ_mode = 'peak_corrd'
                self.integrated_data = compute_moment(app_state, moment_type="peak_coord", axis=self.integ_axis, clip_threshold=clip_threshold)
                window = self.show_in_new_window(f"Peak Coord.: {self.min_input} to {self.max_input}")
                if window is not None:
                    self._record_current_integration_action(
                        clip_threshold,
                        action_tag=getattr(window, "_workspace_action_tag", None),
                    )
            elif mode_text == "Median":
                self.integ_mode = 'median_int'
                self.integrated_data = compute_moment(app_state, moment_type="median", axis=self.integ_axis, clip_threshold=clip_threshold)
                window = self.show_in_new_window(f"Median: {self.min_input} to {self.max_input}")
                if window is not None:
                    self._record_current_integration_action(
                        clip_threshold,
                        action_tag=getattr(window, "_workspace_action_tag", None),
                    )
            elif mode_text == "RMS":
                self.integ_mode = 'rms'
                self.integrated_data = compute_moment(app_state, moment_type="rms", axis=self.integ_axis, clip_threshold=clip_threshold)
                # Note: Legacy RMS had a print message about RMS value.
                # Ideally we should show it in UI, but `integrated_data` is a map.
                # Usecase computes MAP. Global RMS value needs separate call if desired.
                # For now, we focus on the map display.
                window = self.show_in_new_window(f"RMS: {self.min_input} to {self.max_input}")
                if window is not None:
                    self._record_current_integration_action(
                        clip_threshold,
                        action_tag=getattr(window, "_workspace_action_tag", None),
                    )
            elif mode_text == "Sigma (Std Dev)":
                self.integ_mode = 'sigma'
                self.integrated_data = compute_moment(app_state, moment_type="sigma", axis=self.integ_axis, clip_threshold=clip_threshold)
                window = self.show_in_new_window(f"Sigma: {self.min_input} to {self.max_input}")
                if window is not None:
                    self._record_current_integration_action(
                        clip_threshold,
                        action_tag=getattr(window, "_workspace_action_tag", None),
                    )
        
        elif self.mode_radio_group.checkedButton() is not None:
            # A radio button is selected
            if self.integration_radio.isChecked():
                self.integ_mode = 'int'
                self.integrated_data = compute_moment(app_state, moment_type="moment0", axis=self.integ_axis, clip_threshold=clip_threshold)
                window = self.show_in_new_window(f"Integration: {self.min_input} to {self.max_input}")
                if window is not None:
                    self._record_current_integration_action(
                        clip_threshold,
                        action_tag=getattr(window, "_workspace_action_tag", None),
                    )
            elif self.moment1_radio.isChecked():
                self.integ_mode = 'mom1'
                self.integrated_data = compute_moment(app_state, moment_type="moment1", axis=self.integ_axis, clip_threshold=clip_threshold)
                window = self.show_in_new_window(f"Moment 1: {self.min_input} to {self.max_input}")
                if window is not None:
                    self._record_current_integration_action(
                        clip_threshold,
                        action_tag=getattr(window, "_workspace_action_tag", None),
                    )
            elif self.moment2_radio.isChecked():
                self.integ_mode = 'mom2'
                self.integrated_data = compute_moment(app_state, moment_type="moment2", axis=self.integ_axis, clip_threshold=clip_threshold)
                window = self.show_in_new_window(f"Moment 2: {self.min_input} to {self.max_input}")
                if window is not None:
                    self._record_current_integration_action(
                        clip_threshold,
                        action_tag=getattr(window, "_workspace_action_tag", None),
                    )
        else:
            QMessageBox.warning(self, 'Mode Error', 'Please select an integration mode.')
            return


    def show_in_new_window(self,window_title):
        try:
            if np.all(np.isnan(self.integrated_data)):
                raise ValueError
        except (ValueError, AttributeError):
            QMessageBox.warning(self, 'Error', 'All pixel values are NaN or data is invalid!')
            return None
            
            
        metadata = {
            'mode': self.integ_mode,
            'axis': self.integ_axis,
            'range': f"{self.min_input} to {self.max_input}",
            'clipping': f"Values < {self.clip_input.text()}" if self.clip_checkbox.isChecked() else "None"
        }
            
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
            mode = self.integ_mode,
            history_metadata=metadata
        )
        new_window._workspace_action_tag = f"{self._action_record_tag}:{uuid.uuid4().hex}"
        new_window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        new_window.integ_axis = getattr(self, 'integ_axis', None)
        self.fits_viewer.integ_result_windows.append(weakref.ref(new_window))
        #self.fits_viewer.integ_result_windows.append(new_window)
        self.integ_result_windows.append(new_window)
        new_window.show()
        new_window.destroyed.connect(lambda: self.remove_window_reference(new_window))
        return new_window

    def remove_window_reference(self, window):
        action_tag = str(getattr(window, "_workspace_action_tag", "") or "").strip() if window is not None else ""
        if action_tag:
            clear_action_preview_record(self.fits_viewer, action_tag)
        if window in self.integ_result_windows:
            self.integ_result_windows.remove(window)
        cleaned_refs = []
        for window_ref in list(getattr(self.fits_viewer, "integ_result_windows", []) or []):
            target = window_ref() if callable(window_ref) else window_ref
            if target is None or target is window:
                continue
            cleaned_refs.append(window_ref)
        self.fits_viewer.integ_result_windows = cleaned_refs
        self._clear_integration_action_if_no_windows()

    def _record_current_integration_action(self, clip_threshold, *, action_tag: Optional[str] = None):
        mode_to_moment = {
            "int": "moment0",
            "mom1": "moment1",
            "mom2": "moment2",
            "average": "average",
            "peak_int": "peak",
            "peak_corrd": "peak_coord",
            "median_int": "median",
            "rms": "rms",
            "sigma": "sigma",
        }
        moment_type = mode_to_moment.get(getattr(self, "integ_mode", None))
        if not moment_type:
            return
        app_state = self.get_app_state()
        if app_state is None:
            return

        payload = {
            "moment_type": moment_type,
            "axis": int(self.integ_axis),
        }

        if app_state.integ_min_pix is not None and app_state.integ_max_pix is not None:
            payload["pixel_range"] = [
                float(app_state.integ_min_pix),
                float(app_state.integ_max_pix),
            ]
        min_world = getattr(self, "min_input", None)
        max_world = getattr(self, "max_input", None)
        if min_world is not None and max_world is not None:
            payload["world_range"] = [str(min_world), str(max_world)]
        if clip_threshold is not None:
            payload["clip_threshold"] = float(clip_threshold)
        tag = str(action_tag or "").strip()
        if not tag:
            tag = f"{self._action_record_tag}:{uuid.uuid4().hex}"
        payload["_window_action_tag"] = tag

        record_action_preview(
            self.fits_viewer,
            "compute_moment",
            payload,
            replace_tag=tag,
        )

    def _clear_integration_action_if_no_windows(self):
        live_windows = [w for w in self.integ_result_windows if w is not None]
        if not live_windows:
            clear_action_preview_record(
                self.fits_viewer,
                self._action_record_tag,
                action_name="compute_moment",
            )

    def _mode_from_moment_type(self, moment_type: str) -> str:
        lookup = {
            "moment0": "int",
            "moment1": "mom1",
            "moment2": "mom2",
            "average": "average",
            "peak": "peak_int",
            "peak_coord": "peak_corrd",
            "median": "median_int",
            "rms": "rms",
            "sigma": "sigma",
        }
        return lookup.get(str(moment_type or "").lower(), "int")

    def _title_from_moment_type(self, moment_type: str) -> str:
        lookup = {
            "moment0": "Integration",
            "moment1": "Moment 1",
            "moment2": "Moment 2",
            "average": "Average",
            "peak": "Peak Int.",
            "peak_coord": "Peak Coord.",
            "median": "Median",
            "rms": "RMS",
            "sigma": "Sigma",
        }
        return lookup.get(str(moment_type or "").lower(), "Integration")

    def _plane_and_slice_for_axis(self, axis: int):
        axis_int = int(axis)
        if axis_int <= 0:
            plane = "xy"
            integ_slice = ("x", "y", 0, 0) if self.wcs.naxis == 4 else ("x", "y", 0)
        elif axis_int == 1:
            plane = "xz"
            integ_slice = ("x", 0, "y", 0) if self.wcs.naxis == 4 else ("x", 0, "y")
        else:
            plane = "zy"
            integ_slice = (0, "y", "x", 0) if self.wcs.naxis == 4 else (0, "y", "x")
        return plane, integ_slice

    def _world_value_from_axis_pixel(self, axis: int, pix: float):
        axis_int = int(axis)
        try:
            pixel = float(pix)
        except Exception:
            return None

        try:
            if self.fits_viewer.data.ndim == 4:
                if axis_int <= 0:
                    return self.converter.pix_to_world(0, 0, pixel, 0)[2]
                if axis_int == 1:
                    return self.converter.pix_to_world(0, pixel, 0, 0)[1]
                return self.converter.pix_to_world(pixel, 0, 0, 0)[0]
            if axis_int <= 0:
                return self.converter.pix_to_world(0, 0, pixel)[2]
            if axis_int == 1:
                return self.converter.pix_to_world(0, pixel, 0)[1]
            return self.converter.pix_to_world(pixel, 0, 0)[0]
        except Exception:
            return None

    def _resolve_world_range_for_display(self, axis: int, pixel_range, world_range):
        if isinstance(world_range, (list, tuple)) and len(world_range) == 2:
            return str(world_range[0]), str(world_range[1])
        if isinstance(pixel_range, (list, tuple)) and len(pixel_range) == 2:
            world_min = self._world_value_from_axis_pixel(axis, pixel_range[0])
            world_max = self._world_value_from_axis_pixel(axis, pixel_range[1])
            if world_min is not None and world_max is not None:
                return str(world_min), str(world_max)
        return None

    def restore_window_from_action_params(
        self,
        params: dict,
        app_state_override: Optional[AppState] = None,
    ) -> bool:
        if not isinstance(params, dict):
            return False
        app_state = app_state_override if app_state_override is not None else self.get_app_state()
        if app_state is None:
            return False

        moment_type = str(params.get("moment_type") or "moment0").lower()
        try:
            axis = int(params.get("axis", 0))
        except Exception:
            axis = 0
        axis = max(0, min(axis, 2))

        pixel_range = None
        pixel_payload = params.get("pixel_range")
        if isinstance(pixel_payload, (list, tuple)) and len(pixel_payload) == 2:
            try:
                pixel_range = (float(pixel_payload[0]), float(pixel_payload[1]))
            except Exception:
                pixel_range = None

        world_range = None
        world_payload = params.get("world_range")
        if isinstance(world_payload, (list, tuple)) and len(world_payload) == 2:
            world_range = (world_payload[0], world_payload[1])

        clip_threshold = params.get("clip_threshold")
        if clip_threshold is not None:
            try:
                clip_threshold = float(clip_threshold)
            except Exception:
                clip_threshold = None

        try:
            integrated_data = compute_moment(
                app_state,
                moment_type=moment_type,
                axis=axis,
                clip_threshold=clip_threshold,
                pixel_range=pixel_range,
                world_range=world_range,
            )
        except Exception:
            return False

        try:
            if np.all(np.isnan(integrated_data)):
                return False
        except Exception:
            pass

        data = np.asarray(self.fits_viewer.data)
        if data.ndim == 4:
            data = data[0]

        plane, integ_slice = self._plane_and_slice_for_axis(axis)
        integ_mode = self._mode_from_moment_type(moment_type)
        title_prefix = self._title_from_moment_type(moment_type)

        world_display_range = self._resolve_world_range_for_display(axis, pixel_range, world_range)
        if world_display_range is not None:
            range_text = f"{world_display_range[0]} to {world_display_range[1]}"
        elif pixel_range is not None:
            range_text = f"{pixel_range[0]:g} to {pixel_range[1]:g}"
        else:
            range_text = "full range"

        metadata = {
            "mode": integ_mode,
            "axis": axis,
            "range": range_text,
            "clipping": f"Values < {clip_threshold:g}" if clip_threshold is not None else "None",
        }

        new_window = IntegResultWindow(
            integrated_data=integrated_data,
            plane=plane,
            slice=integ_slice,
            fits_viewer=self.fits_viewer,
            subwindows=self.subwindows,
            window_title=f"{title_prefix}: {range_text}",
            config=self.fits_viewer.config_manager.config,
            data=data,
            wcs=self.wcs,
            mode=integ_mode,
            history_metadata=metadata,
        )
        action_tag = str(params.get("_window_action_tag") or "").strip()
        if not action_tag:
            action_tag = f"{self._action_record_tag}:{uuid.uuid4().hex}"
        new_window._workspace_action_tag = action_tag
        new_window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        new_window.integ_axis = axis
        self.fits_viewer.integ_result_windows.append(weakref.ref(new_window))
        self.integ_result_windows.append(new_window)
        new_window.show()
        new_window.destroyed.connect(lambda: self.remove_window_reference(new_window))
        return True

    def get_app_state(self):
        """Get AppState from MainWindow for usecase layer access."""
        main_window = getattr(self.fits_viewer, 'main_window', None) or self.fits_viewer
        return getattr(main_window, 'app_state', None)

    def sync_integration_range_to_app_state(self):
        """Sync current integration range to app_state (usecase layer bridge)."""
        app_state = self.get_app_state()
        if app_state is None:
            return False
        try:
            result = self._get_integ_range(self.integ_axis)
            if result is None:
                return False
            min_pix, max_pix, _, _ = result
            app_state.integ_min_pix = min_pix
            app_state.integ_max_pix = max_pix
            return True
        except Exception as e:
            QMessageBox.warning(self, 'Error', f'Failed to parse integration range: {e}')
            return False



        
    def _get_integ_range(self, axis):
        x_min_text = str(self.x_min_input.text() or "").strip()
        x_max_text = str(self.x_max_input.text() or "").strip()
        y_min_text = str(self.y_min_input.text() or "").strip()
        y_max_text = str(self.y_max_input.text() or "").strip()
        z_min_text = str(self.z_min_input.text() or "").strip()
        z_max_text = str(self.z_max_input.text() or "").strip()

        def _optional_float(text: str, fallback: float) -> float:
            if not text:
                return float(fallback)
            try:
                return float(text)
            except ValueError:
                return float(fallback)

        def _required_float(text: str) -> float:
            if not text:
                raise ValueError
            return float(text)

        try:
            # For coordinate conversion context, non-selected axis values are optional.
            x_anchor = _optional_float(x_min_text, getattr(self, "xmin_val", 0.0))
            y_anchor = _optional_float(y_min_text, getattr(self, "ymin_val", 0.0))
            z_anchor = _optional_float(z_min_text, getattr(self, "zmin_val", 0.0))

            if self.fits_viewer.data.ndim == 3:
                if axis == 2:
                    x_min = _required_float(x_min_text)
                    x_max = _required_float(x_max_text)
                    min_pixel_float = float(self.converter.world_to_pix(x_min, y_anchor, z_anchor)[0])
                    max_pixel_float = float(self.converter.world_to_pix(x_max, y_anchor, z_anchor)[0])
                    self.min_input, self.max_input = x_min_text, x_max_text
                elif axis == 1:
                    y_min = _required_float(y_min_text)
                    y_max = _required_float(y_max_text)
                    min_pixel_float = float(self.converter.world_to_pix(x_anchor, y_min, z_anchor)[1])
                    max_pixel_float = float(self.converter.world_to_pix(x_anchor, y_max, z_anchor)[1])
                    self.min_input, self.max_input = y_min_text, y_max_text
                elif axis == 0:
                    z_min = _required_float(z_min_text)
                    z_max = _required_float(z_max_text)
                    min_pixel_float = float(self.converter.world_to_pix(x_anchor, y_anchor, z_min)[2])
                    max_pixel_float = float(self.converter.world_to_pix(x_anchor, y_anchor, z_max)[2])
                    self.min_input, self.max_input = z_min_text, z_max_text
                else:
                    raise ValueError
            elif self.fits_viewer.data.ndim == 4:
                if axis == 2:
                    x_min = _required_float(x_min_text)
                    x_max = _required_float(x_max_text)
                    min_pixel_float = float(self.converter.world_to_pix(x_min, y_anchor, z_anchor, 0)[0])
                    max_pixel_float = float(self.converter.world_to_pix(x_max, y_anchor, z_anchor, 0)[0])
                    self.min_input, self.max_input = x_min_text, x_max_text
                elif axis == 1:
                    y_min = _required_float(y_min_text)
                    y_max = _required_float(y_max_text)
                    min_pixel_float = float(self.converter.world_to_pix(x_anchor, y_min, z_anchor, 0)[1])
                    max_pixel_float = float(self.converter.world_to_pix(x_anchor, y_max, z_anchor, 0)[1])
                    self.min_input, self.max_input = y_min_text, y_max_text
                elif axis == 0:
                    z_min = _required_float(z_min_text)
                    z_max = _required_float(z_max_text)
                    min_pixel_float = float(self.converter.world_to_pix(x_anchor, y_anchor, z_min, 0)[2])
                    max_pixel_float = float(self.converter.world_to_pix(x_anchor, y_anchor, z_max, 0)[2])
                    self.min_input, self.max_input = z_min_text, z_max_text
                else:
                    raise ValueError
            else:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, 'Invalid Input', 'Please enter valid numeric values for the selected axis range.')
            return
        
        if min_pixel_float > max_pixel_float:
            min_pixel_float, max_pixel_float = max_pixel_float, min_pixel_float
        if max_pixel_float == self.data.shape[axis] - 0.5: max_pixel_float -= 0.00001
        
        # We now return the raw float values for the usecase layer to handle fractional logic.
        # But we still need ints for local clamping checks below?
        min_pixel = int(round(min_pixel_float))
        max_pixel = int(round(max_pixel_float))
        
        # Legacy fraction calc (kept if needed by deleted methods, but we are deleting them)
        # min_fraction = ...
        
        # Return floats



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
        return min_pixel_float, max_pixel_float, 0, 0









        
        


        










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
        live_windows = [w for w in self.integ_result_windows if w is not None]
        if live_windows:
            choice = confirm_pending_close(
                self,
                "Close Integration Panel",
                "Integration result windows are open.",
                keep_label="Keep and Close",
                discard_label="Close Results and Close",
            )
            if choice == "cancel":
                event.ignore()
                return
            if choice == "discard":
                for window in list(live_windows):
                    try:
                        window.close()
                    except Exception:
                        pass
                clear_action_preview_record(
                    self.fits_viewer,
                    self._action_record_tag,
                    action_name="compute_moment",
                )
        else:
            clear_action_preview_record(
                self.fits_viewer,
                self._action_record_tag,
                action_name="compute_moment",
            )
        super().closeEvent(event)
        self.destroyed.emit()


class IntegResultWindow(QMainWindow):
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

    def __init__(self, data, wcs, integrated_data, plane, slice, fits_viewer, subwindows, window_title, config, mode, parent=None, history_metadata: Optional[dict] = None):
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
        self.history_metadata = history_metadata
        self.converter = CoordinateConverter(self.wcs, config)
        self.decimal =  config.get('decimal')
        self.auto_precision_digits = bool(config.get('auto_precision_digits', True))
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
        self._view_history = []
        self._view_history_index = -1
        self._suspend_view_history_recording = True
        self.label = QLabel(self) # For intensity value
        self._contour_layer_id: Optional[str] = None
        self._contour_title_connected = False
        self.marker_manager = MarkerManager(self)
        self.marker_manager.set_active_plane(self.plane)
        self.marker_panel = None
        self.marker_mode_enabled = False
        self._colorbar_autolayout_pending = False
        self._colorbar_auto_layout_override = None
        self._colorbar_auto_anchor_sig = None
        self._setup_marker_action_bridge()

        self.color_pattern = (
            ColorSettingsPanel.settings[ColorMode.MAIN]['color_pattern'] or 
            self.fits_viewer.displaymap.config.get('colorscale')
        )
        if ColorSettingsPanel.settings[ColorMode.INTEG]['color_pattern']:
            self.color_pattern = ColorSettingsPanel.settings[ColorMode.INTEG]['color_pattern']
        self._color_panel_hint = dict(ColorSettingsPanel.settings.get(ColorMode.INTEG, {}) or {})
            
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

        self._setup_region_action_bridge()
        self._initialize_annotation_history_seed()
        self._setup_undo_redo_shortcuts()
        self._view_history = []
        self._view_history_index = -1
        self._suspend_view_history_recording = False
        self._record_local_view_history(reason="init", force=True)
        self._refresh_view_navigation_actions()

        self._register_contour_layer()
        self._schedule_colorbar_auto_layout_if_anchor_changed(force=True)


    def open_cutout_dialog(self, region=None, use_view_bounds=False):
        from takefits.tools.cutout import CutoutSettingsDialog

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

    @staticmethod
    def _normalize_label_text(text) -> str:
        return " ".join(str(text).split()).strip()

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
        label_text = ""
        try:
            if axis == xkey:
                label_text = str(self.ax.get_xlabel() or "")
            elif axis == ykey:
                label_text = str(self.ax.get_ylabel() or "")
        except Exception:
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

    def _format_cursor_pair_text(self, x_text: str, y_text: str) -> str:
        xkey, ykey = self._plane_coord_keys(self.plane)
        x_with_unit = self._format_value_with_axis_unit(xkey, x_text)
        y_with_unit = self._format_value_with_axis_unit(ykey, y_text)
        return f"{x_with_unit}, {y_with_unit}"

    def _compose_click_label_text(self, x_text: str, y_text: str, intensity=None) -> str:
        coord_line = self._format_cursor_pair_text(x_text, y_text)
        if intensity is None:
            return coord_line
        try:
            intensity_line = self._format_intensity_with_unit(intensity)
        except Exception:
            return coord_line
        if intensity_line:
            return f"{coord_line}\n[{self._normalize_label_text(intensity_line)}]"
        return coord_line

    def _toolbar_message_text(self, x: float, y: float, intensity=None) -> str:
        xstr, ystr = self.format_pix.convert(self.plane, x, y)
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
        value = intensity
        if value is None:
            value = self._sample_intensity(x, y)
        if value is not None:
            line2 = f"[{self._normalize_label_text(self._format_intensity_with_unit(value))}] {frame}"
        else:
            line2 = frame
        return f"{line1}\n{line2}"

    def formatter(self, x, y):
        return self._toolbar_message_text(x, y)


    def initUI(self, config):
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self._overlay_updates_enabled = True

        self.ax = self.fig.add_subplot(111, projection=self.fits_viewer.wcs, slices=self.integ_slice)
        self.resize(config.get('figure_width'), config.get('figure_height'))
        self.format_pix = Format_pix_to_wcs(
            self.wcs,
            self.integ_slice,
            self.ax,
            self.plane,
            self.decimal,
            self.number_decimals,
            self.coord_wrap,
            fits_viewer=self.fits_viewer,
            auto_precision_digits=self.auto_precision_digits,
        )
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
            # Get zlabel from zy plane
            coord = fv.get_coordinator() if hasattr(fv, 'get_coordinator') else None
            zy_state = coord.get_state('zy') if coord else None
            if zy_state is not None and zy_state.ax_coord is not None:
                self.zlabel = zy_state.ax_coord[0].get_axislabel()
            else:
                self.zlabel = ""
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
        self.cax.set_zorder(300)

        #Common.update_integ_cax(self.cax)
        self.colorbar = self.fig.colorbar(self.im, cax = self.cax, orientation = config.get('colorbar_orientation') )
        self.colorbar.ax.set_zorder(300)
        #Common.update_integ_colorbar(self.colorbar)
        self.colorbar.ax.minorticks_on()
        ColorSettingsPanel.apply_colorbar_settings(cax = self.cax, colorbar = self.colorbar, config=config)
        
        self.destroyed.connect(self.remove_colorbar)

        self.toolbar = MyNavigationToolbar(self.canvas, self, self.plane, self.ax, color_mode = ColorMode.INTEG, default_image_name = self.fits_viewer.filename)
        self.canvas.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

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
        self.label.setParent(self.canvas)
        self.label.setStyleSheet(f"QLabel {{ color : {self.config.get('click_label_color', 'grey')}; }}")
        self.label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._position_click_label()
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
        # Prime background for overlay-based draws
        self.canvas.draw()
        QTimer.singleShot(0, self._focus_canvas_initial)

    def _focus_canvas_initial(self):
        canvas = getattr(self, "canvas", None)
        if canvas is None:
            return
        try:
            canvas.setFocus()
        except Exception:
            pass

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self._focus_canvas_initial)
        

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
        self._position_click_label()
        self._schedule_colorbar_auto_layout_if_anchor_changed(force=False)
        vline_visible = self.click_v_line.get_visible()
        hline_visible = self.click_h_line.get_visible()
        self._capture_overlay_background()

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

    def _position_click_label(self):
        label = getattr(self, "label", None)
        canvas = getattr(self, "canvas", None)
        if label is None or canvas is None:
            return
        x, y, width, height = compute_click_label_geometry(
            canvas.width(),
            canvas.height(),
            pos_x=self.config.get('poslabel_x', 0.99),
            pos_y=self.config.get('poslabel_y', 0.99),
            requested_width=self.config.get('poslabel_w', 250),
            requested_height=self.config.get('poslabel_h', 30),
        )
        if width <= 0 or height <= 0:
            return
        label.setGeometry(x, y, width, height)

    def _capture_overlay_background(self):
        canvas = getattr(self, "canvas", None)
        overlay_ax = getattr(self, "overlay_ax", None)
        if canvas is None or overlay_ax is None:
            return None

        region_manager = getattr(self, "region_manager", None)
        marker_manager = getattr(self, "marker_manager", None)

        vline_visible = self.click_v_line.get_visible()
        hline_visible = self.click_h_line.get_visible()
        hidden_regions = []
        hidden_markers = []
        if region_manager is not None:
            hidden_regions = region_manager.prepare_for_background_capture()
        if marker_manager is not None:
            hidden_markers = marker_manager.prepare_for_background_capture(self.plane)

        try:
            self.click_v_line.set_visible(False)
            self.click_h_line.set_visible(False)
            self._background = canvas.copy_from_bbox(overlay_ax.bbox)
        except Exception:
            self._background = None
        finally:
            self.click_v_line.set_visible(vline_visible)
            self.click_h_line.set_visible(hline_visible)
            if region_manager is not None:
                region_manager.restore_after_background_capture(hidden_regions)
            if marker_manager is not None and hidden_markers:
                marker_manager.restore_after_background_capture(hidden_markers)

        return self._background

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
            from takefits.tools.marker_panel import MarkerPanel
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

    # Marker plane helpers -------------------------------------------------
    def has_marker_plane(self, plane: str) -> bool:
        current = (getattr(self, "plane", "xy") or "xy").lower()
        base = self.marker_plane_base(plane)
        return base == current and getattr(self, "overlay_ax", None) is not None

    def marker_axes_for_plane(self, plane: str):
        if self.has_marker_plane(plane):
            return getattr(self, "overlay_ax", None)
        return None

    def marker_canvas_for_plane(self, plane: str):
        if self.has_marker_plane(plane):
            return getattr(self, "canvas", None)
        return None

    def marker_plane_base(self, plane: str) -> str:
        if not plane:
            return (getattr(self, "plane", "xy") or "xy").lower()
        plane_lower = plane.lower()
        if "xz" in plane_lower:
            return "xz"
        if "zy" in plane_lower:
            return "zy"
        if "xy" in plane_lower:
            return "xy"
        return plane_lower

    def remap_loaded_marker_state(self, state, *, source_plane: Optional[str] = None, world_frame: Optional[str] = None):
        """
        Remap loaded marker planes into this single-plane window.

        Prefer exact/base plane if supported; otherwise, fall back to current plane
        so marker files from other viewers can still be loaded here.
        """
        plane_name = (getattr(state, "plane", None) or source_plane or "").lower()
        base = self.marker_plane_base(plane_name)
        if self.has_marker_plane(plane_name):
            return [base]
        if self.has_marker_plane(base):
            return [base]
        current = (getattr(self, "plane", "xy") or "xy").lower()
        if self.has_marker_plane(current):
            return [current]
        return []

    # Annotation ActionSession bridge ---------------------------------------
    def _setup_marker_action_bridge(self):
        data = np.asarray(self.integrated_data)
        header = None
        if self.wcs is not None:
            try:
                header = self.wcs.to_header()
                header["NAXIS"] = int(data.ndim)
                for axis, size in enumerate(reversed(data.shape), start=1):
                    header[f"NAXIS{axis}"] = int(size)
            except Exception:
                header = None
        if header is None:
            try:
                header = self.fits_viewer.header.copy()
            except Exception:
                header = {}

        self.app_state = create_app_state(
            data=data,
            header=header,
            wcs=self.wcs,
            filepath=getattr(self, "filename", None),
        )
        registry = ActionRegistry()
        register_default_actions(registry)
        self.action_session = ActionSession(registry=registry, state=self.app_state)
        self._suspend_action_recording = False
        self._last_markers_fingerprint = None
        self._last_regions_fingerprint = None
        self._markers_commit_timer = QTimer(self)
        self._markers_commit_timer.setSingleShot(True)
        self._markers_commit_timer.timeout.connect(self._commit_markers_to_session)
        self._regions_commit_timer = None
        try:
            self.marker_manager.markers_changed.connect(lambda *_: self._schedule_markers_commit())
        except Exception:
            pass

    def _setup_region_action_bridge(self):
        self._last_regions_fingerprint = None
        self._regions_commit_timer = QTimer(self)
        self._regions_commit_timer.setSingleShot(True)
        self._regions_commit_timer.timeout.connect(self._commit_regions_to_session)
        try:
            self.region_manager.selected_region_changed.connect(lambda *_: self._schedule_regions_commit())
        except Exception:
            pass

    def _initialize_annotation_history_seed(self):
        try:
            self.app_state.markers = [MarkerSpec.from_dict(entry) for entry in self._marker_specs_snapshot()]
        except Exception:
            pass
        try:
            self.app_state.regions = [RegionSpec.from_dict(entry) for entry in self._region_specs_snapshot()]
        except Exception:
            pass
        try:
            self.action_session.set_initial_state_seed()
        except Exception:
            pass
        self._last_markers_fingerprint = json.dumps(
            self._marker_specs_snapshot(), sort_keys=True, separators=(",", ":")
        )
        self._last_regions_fingerprint = json.dumps(
            self._region_specs_snapshot(), sort_keys=True, separators=(",", ":")
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
        if getattr(self, "_regions_commit_timer", None) is not None and self._regions_commit_timer.isActive():
            self._regions_commit_timer.stop()
            self._commit_regions_to_session()
        if getattr(self, "_markers_commit_timer", None) is not None and self._markers_commit_timer.isActive():
            self._markers_commit_timer.stop()
            self._commit_markers_to_session()

    def _schedule_markers_commit(self, delay_ms: int = 200) -> None:
        if getattr(self, "_suspend_action_recording", False):
            return
        if hasattr(self, "_markers_commit_timer") and self._markers_commit_timer is not None:
            self._markers_commit_timer.start(int(delay_ms))

    def _schedule_regions_commit(self, delay_ms: int = 200) -> None:
        if getattr(self, "_suspend_action_recording", False):
            return
        if hasattr(self, "_regions_commit_timer") and self._regions_commit_timer is not None:
            self._regions_commit_timer.start(int(delay_ms))

    def _marker_specs_snapshot(self):
        return snapshot_marker_specs(getattr(self, "marker_manager", None))

    def _region_specs_snapshot(self):
        return snapshot_region_specs(
            getattr(self, "region_manager", None),
            default_plane=(getattr(self, "plane", "xy") or "xy"),
        )

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

    def _commit_regions_to_session(self) -> None:
        if getattr(self, "_suspend_action_recording", False):
            return
        specs = self._region_specs_snapshot()
        fingerprint = json.dumps(specs, sort_keys=True, separators=(",", ":"))
        if fingerprint == self._last_regions_fingerprint:
            return
        self._last_regions_fingerprint = fingerprint
        try:
            self.action_session.execute("set_regions", regions=specs)
            self._notify_main_window_undo_redo_state()
        except Exception:
            try:
                self.app_state.regions = [RegionSpec.from_dict(entry) for entry in specs]
            except Exception:
                pass

    def _build_region_payload_from_state(self):
        state = getattr(self.action_session, "state", None)
        region_specs = list(getattr(state, "regions", []) or []) if state is not None else []
        return build_region_payload_from_specs(region_specs, default_plane=(self.plane or "xy"))

    def _build_marker_payload_from_state(self):
        state = getattr(self.action_session, "state", None)
        marker_specs = list(getattr(state, "markers", []) or []) if state is not None else []
        return build_marker_payload_from_specs(marker_specs)

    def _apply_action_session_state_to_viewer(self):
        state = getattr(self.action_session, "state", None)
        if state is None:
            return
        self._suspend_action_recording = True
        try:
            self.app_state = state

            region_manager = getattr(self, "region_manager", None)
            if region_manager is not None:
                try:
                    region_manager.delete_all_regions()
                except Exception:
                    pass
                region_payload = self._build_region_payload_from_state()
                if region_payload.get("regions"):
                    try:
                        region_manager.import_regions_from_dict(region_payload, clear_existing=True)
                    except Exception:
                        pass

            marker_manager = getattr(self, "marker_manager", None)
            if marker_manager is not None:
                planes_to_redraw = set()
                marker_layers = list((getattr(marker_manager, "_layers", None) or {}).keys())
                for plane in marker_layers:
                    plane_name = str(self.marker_plane_base(str(plane or "")) or "").lower()
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
                    plane_name = str(self.marker_plane_base(str(marker_entry.get("plane") or "")) or "").lower()
                    if plane_name:
                        planes_to_redraw.add(plane_name)
                if marker_entries:
                    try:
                        imported_plane = marker_manager.import_from_dict(marker_payload)
                        imported_name = str(self.marker_plane_base(str(imported_plane or "")) or "").lower()
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
                    # Rebuild background after marker restore so previous marker strokes
                    # are not baked into the blit buffer.
                    self._background = None
                    if getattr(self, "canvas", None) is not None:
                        try:
                            self.canvas.draw()
                        except Exception:
                            self.canvas.draw_idle()
            try:
                self.redraw_main_overlay_and_blit()
            except Exception:
                if getattr(self, "canvas", None) is not None:
                    self.canvas.draw_idle()

            self._last_regions_fingerprint = json.dumps(
                self._region_specs_snapshot(), sort_keys=True, separators=(",", ":")
            )
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
            try:
                self.click_v_line.set_visible(False)
                self.click_h_line.set_visible(False)
                self.label.setVisible(False)
            except Exception:
                pass
            try:
                self.redraw_main_overlay_and_blit()
            except Exception:
                pass
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
        self.toolbar.set_message(self._toolbar_message_text(event.xdata, event.ydata, intensity=intensity))

    def _sample_intensity(self, x, y):
        xp, yp = int(round(x)), int(round(y))
        try:
            if self.plane == 'zy':
                return self.integrated_data.T[yp, xp]
            return self.integrated_data[yp, xp]
        except (IndexError, TypeError):
            return None

    def refresh_coordinate_display(self):
        if self.toolbar is not None and self.ax is not None:
            try:
                xmid = sum(self.ax.get_xlim()) / 2.0
                ymid = sum(self.ax.get_ylim()) / 2.0
                self.toolbar.set_message(self._toolbar_message_text(xmid, ymid))
            except Exception:
                pass

        if not getattr(self, "label", None):
            return
        if not self.label.isVisible():
            return
        try:
            xdata = self.click_v_line.get_xdata()
            ydata = self.click_h_line.get_ydata()
            if xdata is None or ydata is None or len(xdata) == 0 or len(ydata) == 0:
                return
            x = float(xdata[0])
            y = float(ydata[0])
        except Exception:
            return

        try:
            xstr, ystr = self.format_pix.convert(self.plane, x, y)
            intensity = self._sample_intensity(x, y)
            self.label.setText(self._compose_click_label_text(xstr, ystr, intensity))
        except Exception:
            return


    def redraw_main_overlay_and_blit(self):
        canvas = getattr(self, 'canvas', None)
        if canvas is None:
            return

        if getattr(self, "_background", None) is None:
            self._capture_overlay_background()
            if getattr(self, "_background", None) is None:
                try:
                    canvas.draw()
                except Exception:
                    canvas.draw_idle()
                self._capture_overlay_background()
            if getattr(self, "_background", None) is None:
                canvas.draw_idle()
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
        self._blit_colorbar_foreground(force=False)


    def remove_colorbar(self):
        #Common.remove_integ_colorbar(self.colorbar)
        #Common.remove_integ_cax(self.cax)
        pass

    def _seed_color_panel_settings_from_current_image(self):
        settings = {
            "min_val": None,
            "max_val": None,
            "log_scale": False,
            "gamma_value": 1.0,
            "invert": False,
            "color_pattern": None,
        }
        raw = dict(ColorSettingsPanel.settings.get(ColorMode.INTEG, {}) or {})
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
        ColorSettingsPanel.settings[ColorMode.INTEG] = dict(settings)

    @staticmethod
    def _bbox_overlaps(a, b) -> bool:
        if a is None or b is None:
            return False
        try:
            return bool(a.overlaps(b))
        except Exception:
            pass
        try:
            ax0, ay0, ax1, ay1 = float(a.x0), float(a.y0), float(a.x1), float(a.y1)
            bx0, by0, bx1, by1 = float(b.x0), float(b.y0), float(b.x1), float(b.y1)
        except Exception:
            return False
        return (ax0 < bx1) and (ax1 > bx0) and (ay0 < by1) and (ay1 > by0)

    def _set_colorbar_zorder(self, zorder: float = 300.0):
        cax = getattr(self, "cax", None)
        if cax is None:
            return
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

    def _colorbar_needs_foreground_blit(self) -> bool:
        cax = getattr(self, "cax", None)
        ax = getattr(self, "ax", None)
        overlay_ax = getattr(self, "overlay_ax", None)
        if cax is None:
            return False
        cbar_bbox = getattr(cax, "bbox", None)
        ax_bbox = getattr(ax, "bbox", None)
        overlay_bbox = getattr(overlay_ax, "bbox", None)
        return self._bbox_overlaps(cbar_bbox, ax_bbox) or self._bbox_overlaps(cbar_bbox, overlay_bbox)

    def _blit_colorbar_foreground(self, force: bool = False) -> bool:
        canvas = getattr(self, "canvas", None)
        fig = getattr(self, "fig", None)
        cax = getattr(self, "cax", None)
        if canvas is None or fig is None or cax is None:
            return False

        self._set_colorbar_zorder()
        if not force and not self._colorbar_needs_foreground_blit():
            return False

        try:
            fig.draw_artist(cax)
            canvas.blit(cax.bbox)
            return True
        except Exception:
            if force:
                try:
                    canvas.draw_idle()
                except Exception:
                    pass
            return False

    def _is_colorbar_auto_layout_enabled(self) -> bool:
        override = getattr(self, "_colorbar_auto_layout_override", None)
        if override is not None:
            return bool(override)
        return bool(self._get_colorbar_config().get("colorbar_auto_layout", False))

    def _get_colorbar_config(self):
        config_mgr = getattr(self, "config_manager", None)
        config = getattr(config_mgr, "config", None) if config_mgr is not None else None
        if isinstance(config, dict):
            return config
        local_config = getattr(self, "config", None)
        if isinstance(local_config, dict):
            return local_config
        return {}

    def _set_colorbar_orientation_config(self, orientation: str):
        orientation = str(orientation or "").strip().lower()
        if orientation not in ("vertical", "horizontal"):
            return
        for cfg in (getattr(self, "config", None), getattr(getattr(self, "config_manager", None), "config", None)):
            if isinstance(cfg, dict):
                cfg["colorbar_orientation"] = orientation

    def _current_colorbar_orientation(self) -> str:
        orientation = str(getattr(getattr(self, "colorbar", None), "orientation", "") or "").lower()
        if orientation in ("vertical", "horizontal"):
            return orientation
        config = self._get_colorbar_config()
        cfg_orientation = str(config.get("colorbar_orientation", "") or "").lower()
        if cfg_orientation in ("vertical", "horizontal"):
            return cfg_orientation
        return "vertical"

    def _rebuild_colorbar(self, orientation: str):
        config = self._get_colorbar_config()
        old_cax = getattr(self, "cax", None)
        if old_cax is not None:
            try:
                bounds = [float(v) for v in old_cax.get_position().bounds]
            except Exception:
                bounds = [
                    float(config.get("cbar_pos_x", 0.9)),
                    float(config.get("cbar_pos_y", 0.11)),
                    float(config.get("cbar_width", 0.04)),
                    float(config.get("cbar_height", 0.77)),
                ]
        else:
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
        self.colorbar = self.fig.colorbar(self.im, cax=self.cax, orientation=orientation)
        self.colorbar.ax.set_zorder(300)
        self.colorbar.ax.minorticks_on()
        ColorSettingsPanel.apply_colorbar_settings(cax=self.cax, colorbar=self.colorbar, config=config)
        self._set_colorbar_zorder()

    def _apply_colorbar_auto_layout(self, force: bool = False) -> bool:
        if not force and not self._is_colorbar_auto_layout_enabled():
            return False
        if bool(getattr(self, "_applying_colorbar_auto_layout", False)):
            return False
        cax = getattr(self, "cax", None)
        ax = getattr(self, "ax", None)
        if cax is None or ax is None:
            return False

        try:
            ax_bounds = tuple(float(v) for v in ax.get_position().bounds)
            cbar_bounds = tuple(float(v) for v in cax.get_position().bounds)
            fig = getattr(self, "fig", None)
            fig_w_px = float(getattr(getattr(fig, "bbox", None), "width", 0.0) or 0.0)
            fig_h_px = float(getattr(getattr(fig, "bbox", None), "height", 0.0) or 0.0)
        except Exception:
            return False

        if len(ax_bounds) != 4 or len(cbar_bounds) != 4:
            return False
        if ax_bounds[2] <= 0.0 or ax_bounds[3] <= 0.0:
            return False
        if fig_w_px <= 0.0:
            fig_w_px = 1.0
        if fig_h_px <= 0.0:
            fig_h_px = 1.0

        config = self._get_colorbar_config()
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
            self._rebuild_colorbar(target_orientation)
            cax = getattr(self, "cax", None)
            if cax is None:
                return False
            try:
                cbar_bounds = tuple(float(v) for v in cax.get_position().bounds)
            except Exception:
                return False

        target = compute_colorbar_geometry(
            ax_bounds,
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
            if getattr(self, "canvas", None) is not None:
                self.canvas.draw_idle()
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
        if bool(getattr(self, "_colorbar_autolayout_pending", False)):
            return
        self._colorbar_autolayout_pending = True

        def _run():
            self._colorbar_autolayout_pending = False
            if not force and bool(getattr(self, "_suspend_colorbar_auto_layout", False)):
                return
            if self._is_colorbar_auto_layout_enabled():
                self._apply_colorbar_auto_layout(force=force)

        QTimer.singleShot(0, _run)

    def _colorbar_layout_anchor_signature(self):
        ax = getattr(self, "ax", None)
        fig = getattr(self, "fig", None)
        if ax is None or fig is None:
            return None
        try:
            ax_bounds = tuple(round(float(v), 8) for v in ax.get_position().bounds)
            fig_w = round(float(getattr(getattr(fig, "bbox", None), "width", 0.0) or 0.0), 3)
            fig_h = round(float(getattr(getattr(fig, "bbox", None), "height", 0.0) or 0.0), 3)
        except Exception:
            return None
        return ax_bounds + (fig_w, fig_h)

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

    def open_color_settings(self):
        self._seed_color_panel_settings_from_current_image()
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
        try:
            self._color_panel_hint = dict(ColorSettingsPanel.settings.get(ColorMode.INTEG, {}) or {})
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
        try:
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            return {
                "xlim": [float(xlim[0]), float(xlim[1])],
                "ylim": [float(ylim[0]), float(ylim[1])],
            }
        except Exception:
            return None

    def _capture_view_history_color(self):
        fallback = dict(ColorSettingsPanel.settings.get(ColorMode.INTEG, {}) or {})
        hint = getattr(self, "_color_panel_hint", None)
        if isinstance(hint, dict):
            fallback.update(hint)

        settings = self._normalize_view_history_color_settings(
            self._extract_live_color_settings(),
            fallback=fallback,
        )
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
        bad_color = self.config.get("bad_color", "black") if isinstance(self.config, dict) else "black"
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
        ColorSettingsPanel.settings[ColorMode.INTEG] = dict(merged)
        panel.current_settings = ColorSettingsPanel.settings[ColorMode.INTEG]

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
            if min_val is not None and max_val is not None and getattr(panel, "min_line", None) is not None and getattr(panel, "max_line", None) is not None:
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
        ColorSettingsPanel.settings[ColorMode.INTEG] = dict(settings)

        try:
            cmap = self._build_view_history_gamma_cmap(display_pattern, settings.get("gamma_value", 1.0))
        except Exception:
            return False

        image = getattr(self, "im", None)
        if image is None:
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
                colorbar.update_normal(image)
            except Exception:
                pass
            try:
                ColorSettingsPanel.apply_colorbar_settings(
                    cax=self.cax,
                    colorbar=colorbar,
                    config=self._get_colorbar_config(),
                )
            except Exception:
                pass
        self._set_colorbar_zorder()

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
                        self.ax.set_xlim(x0, x1)
                        self.ax.set_ylim(y0, y1)
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

    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._background = None
        if hasattr(self, 'label'):
            self._position_click_label()
            
        if self.toolbar._subplot_dialog is not None:
            self.toolbar._subplot_dialog.width_spin.setValue(int(self.window().width()))
            self.toolbar._subplot_dialog.height_spin.setValue(int(self.window().height()))
        self._schedule_colorbar_auto_layout_if_anchor_changed(force=False)

    
    def closeEvent(self, event):
        self._flush_pending_annotation_commits()
        # Close color settings panel
        if self.color_settings_panel is not None:
            self.color_settings_panel.close()
            self.color_settings_panel = None
        
        # Close cutout dialog
        if getattr(self, 'cutout_dialog', None) is not None:
            try:
                self.cutout_dialog.close()
            finally:
                self.cutout_dialog = None

        # Cleanup canvas and figure
        self.remove_colorbar()
        if self.canvas is not None:
            self.canvas.close()
            self.canvas = None
        if self.fig is not None:
            self.fig.clear()
            self.fig = None
        if self.toolbar._subplot_dialog is not None:
            self.toolbar._subplot_dialog.close()
        
        # Cleanup contour layer
        self._unregister_contour_layer()
        
        # Cleanup marker panel
        marker_panel = getattr(self, 'marker_panel', None)
        if marker_panel is not None:
            try:
                marker_panel.close()
            except Exception:
                pass
            self.marker_panel = None
        
        # Clear marker manager plane
        marker_manager = getattr(self, 'marker_manager', None)
        if marker_manager is not None:
            marker_manager.clear_plane(self.plane)
        
        super().closeEvent(event)
        try:
            self.destroyed.emit()
        except Exception:
            pass


        
    
    def sync_range(self):
        if self.plane == 'xy' or self.plane == 'xz':
            if hasattr(self.fits_viewer, 'x_min_input'):
                x_min = self.fits_viewer.x_min_input.text()
                x_max = self.fits_viewer.x_max_input.text()
                self.x_min_int_input.setText(x_min)
                self.x_max_int_input.setText(x_max)
                self.set_x_range()
            
        if self.plane == 'xy' or self.plane == 'zy':
            if hasattr(self.fits_viewer, 'y_min_input'):
                y_min = self.fits_viewer.y_min_input.text()
                y_max = self.fits_viewer.y_max_input.text()
                self.y_min_int_input.setText(y_min)
                self.y_max_int_input.setText(y_max)
                self.set_y_range()
            
        if self.plane == 'xz' or self.plane == 'zy':
            z_min, z_max = self._resolve_sync_z_inputs()
            if not z_min and self.zmin_val is not None:
                z_min = str(self.zmin_val)
            if not z_max and self.zmax_val is not None:
                z_max = str(self.zmax_val)
            if z_min and z_max:
                self.z_min_int_input.setText(z_min)
                self.z_max_int_input.setText(z_max)
                self.set_z_range()

    def _resolve_sync_z_inputs(self):
        preferred_indices = [0, 1] if self.plane == "xz" else [1, 0]
        for idx in preferred_indices:
            if idx >= len(self.subwindows):
                continue
            window = self.subwindows[idx]
            if window is None:
                continue
            if not (hasattr(window, "z_min_input") and hasattr(window, "z_max_input")):
                continue
            z_min = str(window.z_min_input.text() or "").strip()
            z_max = str(window.z_max_input.text() or "").strip()
            if z_min and z_max:
                return z_min, z_max

        range_panel = getattr(self.fits_viewer, "range_panel", None)
        if range_panel is not None and hasattr(range_panel, "z_min_input") and hasattr(range_panel, "z_max_input"):
            z_min = str(range_panel.z_min_input.text() or "").strip()
            z_max = str(range_panel.z_max_input.text() or "").strip()
            if z_min and z_max:
                return z_min, z_max
        return "", ""


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
            self.set_x_range(record_history=False)
            
        if self.plane == 'xy' or self.plane == 'zy':
            self.y_min_int_input.setText(str(self.ymin_val_full))
            self.y_max_int_input.setText(str(self.ymax_val_full))
            self.set_y_range(record_history=False)
            
        if self.plane == 'xz' or self.plane == 'zy':
            self.z_min_int_input.setText(str(self.zmin_val_full))
            self.z_max_int_input.setText(str(self.zmax_val_full))
            self.set_z_range(record_history=False)
        if not bool(getattr(self, "_suspend_view_history_recording", False)):
            self._record_local_view_history(reason="range:full")
            
    def set_x_range(self, record_history: bool = True):
        try:
            x_min = str(self.x_min_int_input.text() or "").strip()
            x_max = str(self.x_max_int_input.text() or "").strip()
            if not x_min or not x_max:
                raise ValueError

            def _fallback(text, default):
                value = str(text or "").strip()
                if value:
                    return value
                return str(default if default is not None else "").strip()

            y_min_ref = _fallback(self.y_min_int_input.text() if hasattr(self, "y_min_int_input") else "", self.ymin_val)
            y_max_ref = _fallback(self.y_max_int_input.text() if hasattr(self, "y_max_int_input") else "", self.ymax_val)
            z_min_ref = _fallback(self.z_min_int_input.text() if hasattr(self, "z_min_int_input") else "", self.zmin_val)
            z_max_ref = _fallback(self.z_max_int_input.text() if hasattr(self, "z_max_int_input") else "", self.zmax_val)
            if self.fits_viewer.data.ndim == 3:
                xp_min = float(self.converter.world_to_pix(x_min, y_min_ref, z_min_ref)[0])
                xp_max = float(self.converter.world_to_pix(x_max, y_max_ref, z_max_ref)[0])
            elif self.fits_viewer.data.ndim == 4:
                xp_min = float(self.converter.world_to_pix(x_min, y_min_ref, z_min_ref, 0)[0])
                xp_max = float(self.converter.world_to_pix(x_max, y_max_ref, z_max_ref, 0)[0])
                
            if xp_min > xp_max: xp_min, xp_max = xp_max, xp_min
                
            self.ax.set_xlim(xp_min, xp_max)
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
            y_min = str(self.y_min_int_input.text() or "").strip()
            y_max = str(self.y_max_int_input.text() or "").strip()
            if not y_min or not y_max:
                raise ValueError

            def _fallback(text, default):
                value = str(text or "").strip()
                if value:
                    return value
                return str(default if default is not None else "").strip()

            x_min_ref = _fallback(self.x_min_int_input.text() if hasattr(self, "x_min_int_input") else "", self.xmin_val)
            x_max_ref = _fallback(self.x_max_int_input.text() if hasattr(self, "x_max_int_input") else "", self.xmax_val)
            z_min_ref = _fallback(self.z_min_int_input.text() if hasattr(self, "z_min_int_input") else "", self.zmin_val)
            z_max_ref = _fallback(self.z_max_int_input.text() if hasattr(self, "z_max_int_input") else "", self.zmax_val)
            if self.fits_viewer.data.ndim == 3:
                yp_min = float(self.converter.world_to_pix(x_min_ref, y_min, z_min_ref)[1])
                yp_max = float(self.converter.world_to_pix(x_max_ref, y_max, z_max_ref)[1])
            elif self.fits_viewer.data.ndim == 4:
                yp_min = float(self.converter.world_to_pix(x_min_ref, y_min, z_min_ref, 0)[1])
                yp_max = float(self.converter.world_to_pix(x_max_ref, y_max, z_max_ref, 0)[1])
                
            if yp_min > yp_max: yp_min, yp_max = yp_max, yp_min

            self.ax.set_ylim(yp_min, yp_max)
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
            z_min = str(self.z_min_int_input.text() or "").strip()
            z_max = str(self.z_max_int_input.text() or "").strip()
            if not z_min or not z_max:
                raise ValueError

            def _fallback(text, default):
                value = str(text or "").strip()
                if value:
                    return value
                return str(default if default is not None else "").strip()

            x_min_ref = _fallback(self.x_min_int_input.text() if hasattr(self, "x_min_int_input") else "", self.xmin_val)
            x_max_ref = _fallback(self.x_max_int_input.text() if hasattr(self, "x_max_int_input") else "", self.xmax_val)
            y_min_ref = _fallback(self.y_min_int_input.text() if hasattr(self, "y_min_int_input") else "", self.ymin_val)
            y_max_ref = _fallback(self.y_max_int_input.text() if hasattr(self, "y_max_int_input") else "", self.ymax_val)
            if self.fits_viewer.data.ndim == 3:
                zp_min = float(self.converter.world_to_pix(x_min_ref, y_min_ref, z_min)[2])
                zp_max = float(self.converter.world_to_pix(x_max_ref, y_max_ref, z_max)[2])
            elif self.fits_viewer.data.ndim == 4:
                zp_min = float(self.converter.world_to_pix(x_min_ref, y_min_ref, z_min, 0)[2])
                zp_max = float(self.converter.world_to_pix(x_max_ref, y_max_ref, z_max, 0)[2])
                
            if zp_min > zp_max: zp_min, zp_max = zp_max, zp_min
            
            if self.plane == 'xz':
                self.ax.set_ylim(zp_min, zp_max)
            elif self.plane == 'zy':
                self.ax.set_xlim(zp_min, zp_max)
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
        if not bool(getattr(self, "_suspend_view_history_recording", False)):
            self._record_local_view_history(reason=f"nav:{plane}")

    def resync_after_workspace_restore(self):
        try:
            self._refresh_contours()
        except Exception:
            pass

        marker_manager = getattr(self, "marker_manager", None)
        if marker_manager is not None:
            try:
                marker_manager.redraw_plane(self.plane)
            except Exception:
                pass

        self._background = None
        redraw_overlay = getattr(self, "redraw_main_overlay_and_blit", None)
        if callable(redraw_overlay):
            try:
                redraw_overlay()
                return
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
                    self.canvas.draw_idle()
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

    def _begin_colorbar_drag(self, event) -> bool:
        # Check standard toolbar modes
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

    def _handle_colorbar_double_click(self, event) -> bool:
        if not getattr(event, "dblclick", False):
            return False
        if not self._is_colorbar_axes(getattr(event, "inaxes", None)):
            return False

        # Toggle window-local auto-layout mode so each result window is independent.
        current_state = self._is_colorbar_auto_layout_enabled()
        new_state = not current_state
        self._colorbar_auto_layout_override = bool(new_state)

        state_str = "ON" if new_state else "OFF"
        print(f"Colorbar Auto-Layout: {state_str}")

        if new_state:
            # If turning ON, snap to the correct position immediately.
            self._schedule_colorbar_auto_layout(force=True)
        else:
            # If turning OFF, clear pending auto-layout requests.
            self._colorbar_autolayout_pending = False

        return True


    def on_click(self, event):
        if self._handle_colorbar_double_click(event):
            return
        if self._begin_colorbar_drag(event):
            return

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
                self.label.setText(self._compose_click_label_text(xstr, ystr, intensity))
            except (IndexError, TypeError):
                print(f'\r Clicked at ({", ".join(map(str, text_coord_tuple))})              \n               \033[1A', end='     ')
                self.label.setText(self._compose_click_label_text(xstr, ystr, None))

            # Position and show the main label
            self._position_click_label()
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
        if self._end_colorbar_drag(event):
            return
        # Stop dragging when the mouse button is released
        if event.button == 1:
            self.dragging = False
            if self.region_mode_enabled:
                self.region_manager.handle_release(event)
                self.redraw_main_overlay_and_blit()
                self._schedule_regions_commit(0)
            marker_manager = getattr(self, 'marker_manager', None)
            if getattr(self, 'marker_mode_enabled', False) and marker_manager is not None:
                marker_manager.handle_release(event)


    def on_motion(self, event):
        if self._drag_colorbar(event):
            return
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
                self.label.setText(self._compose_click_label_text(xstr, ystr, intensity))
            except (IndexError, TypeError):
                self.label.setText(self._compose_click_label_text(xstr, ystr, None))
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
        key = str(getattr(event, "key", "") or "").lower()
        if key in self._VIEW_BACK_KEY_TOKENS:
            self.view_back()
            return
        if key in self._VIEW_FORWARD_KEY_TOKENS:
            self.view_forward()
            return

        region_manager = getattr(self, 'region_manager', None)
        if region_manager is not None:
            region_manager.handle_key_press(event)
        marker_manager = getattr(self, 'marker_manager', None)
        if marker_manager is not None:
            marker_manager.handle_key_press(event)
        if event.key == 'backspace' or event.key == 'delete':
            if self.region_mode_enabled:
                self.region_manager.delete_selected_region()
                self._schedule_regions_commit(0)
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
        from takefits.core.contour_manager import ContourManager
        if self._contour_layer_id is not None:
            return
        manager = ContourManager.instance()
        # Use a unique ID for this window
        layer_id = f"Integ_{id(self)}"
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

        try:
            manager.contour_updated.connect(self._on_contour_updated)
        except Exception:
            pass

        if not self._contour_title_connected:
            try:
                self.windowTitleChanged.connect(self._handle_title_change_for_contours)
                self._contour_title_connected = True
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Region Editor Compatibility Methods
    # ------------------------------------------------------------------
    def _get_shared_xpix(self) -> int:
        return 0

    def _get_shared_ypix(self) -> int:
        return 0

    def _get_shared_zpix(self) -> int:
        return 0
        
    def _get_shared_spix(self) -> int:
        return 0

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


    def save_fits(self):
        # Generate default filename
        base_filename = getattr(self.fits_viewer, 'filename', "takefits.fits")
        base_root = os.path.splitext(base_filename)[0]
        mode_suffix = self.integ_mode if isinstance(self.integ_mode, str) else 'integ'
        default_filename = f"{base_root}.{mode_suffix}.fits"

        filename, _ = QFileDialog.getSaveFileName(
            None, "Save FITS File", default_filename, "FITS Files (*.fits);;All Files (*)")

        if not filename:
             return

        # Prepare History
        history = []
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        history.append(f"Integration executed by takefits on {timestamp}")
        history.append(f"Source file: {os.path.basename(self.fits_viewer.filename)}")

        if self.history_metadata:
             mode_map = {
                'int': 'Integration', 'mom1': 'Moment 1', 'mom2': 'Moment 2',
                'average': 'Average', 'peak_int': 'Peak Intensity', 
                'peak_corrd': 'Peak Coordinate', 'median_int': 'Median', 'rms': 'RMS',
                'sigma': 'Sigma (Std Dev)'
            }
             mode_str = mode_map.get(self.history_metadata.get('mode'), self.history_metadata.get('mode'))
             history.append(f"Mode: {mode_str}")
             
             axis_idx = self.history_metadata.get('axis')
             axis_name = "Unknown"
             if axis_idx == 0: axis_name = "Z (Depth/Spectral)" 
             elif axis_idx == 1: axis_name = "Y (Lat/Dec)"
             elif axis_idx == 2: axis_name = "X (Lon/RA)"
             history.append(f"Axis: {axis_name}")
             history.append(f"Range: {self.history_metadata.get('range')}")
             history.append(f"Clipping: {self.history_metadata.get('clipping')}")

        history.extend(build_processing_history_lines(self.fits_viewer))

        # Robustly get spectral metadata
        spec_meta = getattr(self.fits_viewer, 'spectral_metadata', {})

        # Construct AppState for export
        state = AppState(
             data=None, # Not needed for export meta
             wcs=self.wcs, # Original WCS
             header=self.fits_viewer.header,
             spectral_metadata=spec_meta
        )
        plane_axes_map = {
            'xy': (1, 2),
            'xz': (1, 3),
            'zy': (3, 2),
        }
        display_fits_axes = plane_axes_map.get(str(self.plane or '').lower(), (1, 2))
        export_data = self.integrated_data.T if str(self.plane or '').lower() == 'zy' else self.integrated_data
        
        try:
             outfile = export_moment_fits(
                  state=state,
                  moment_data=export_data,
                  output_path=filename,
                  moment_type=self.integ_mode,
                  history_entries=history,
                  display_fits_axes=display_fits_axes,
                  integration_axis=int(getattr(self, "integ_axis", 0) or 0),
             )
             QMessageBox.information(None, "Save Successful", f"FITS successfully saved as: {outfile}")
             print(f"File successfully saved as: {outfile}\n")
        except Exception as e:
             QMessageBox.warning(None, "Save Error", f"Failed to save FITS:\n{e}")
             print(f"Error saving FITS: {e}")

        
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
                except ValueError:
                    print(f"Skipping invalid key-value pair: {key} -> {value}")
    
        for key, value in header.items():
            if key not in new_header:
                try:
                    new_header[key] = value
                except ValueError:
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
