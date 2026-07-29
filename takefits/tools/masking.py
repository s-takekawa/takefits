from datetime import datetime
import os

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QDoubleSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut

from takefits.core import usecases
from takefits.core.app_state import create_app_state
from takefits.core.history_provenance import build_processing_history_lines
from takefits.core.usecases.mask import (
    compute_masked_from_keep_mask,
    compute_moment_masked,
)
from takefits.logic.data_tools import create_preview_snapshot
from takefits.tools.base_panel import BaseToolPanel, clear_action_preview_record, has_action_record_tag, record_action_preview
from takefits.tools.base_panel import capture_preferred_cursor_snapshot, replay_action_history_to_current_cursor
from takefits.ui.save_fits_dialog import SaveFITS


class MaskSettingsPanel(BaseToolPanel):
    def __init__(self, fits_viewer, subwindows):
        self.original_data = None
        self.current_mask = None
        self.last_mask_history_entries = []
        self._mask_export_metadata = {}
        self._has_pending_changes = False
        self._action_record_tag = "panel:mask"
        super().__init__(fits_viewer, subwindows)
        self.setWindowTitle(f"Mask Settings: {self.fits_viewer.filename}")
        self.resync_after_workspace_restore()

    def initUI(self):
        main_layout = QVBoxLayout()
        self.tab_widget = QTabWidget()
        self.data_mask_tab = self._build_data_mask_tab()
        self.moment_mask_tab = self._build_moment_mask_tab()
        self.tab_widget.addTab(self.data_mask_tab, "Data Mask")
        self.tab_widget.addTab(self.moment_mask_tab, "Moment Mask")
        self.tab_widget.currentChanged.connect(self._on_tab_changed)
        main_layout.addWidget(self.tab_widget)

        bottom_layout = QGridLayout()
        self.reset_button = QPushButton("Reset All Masks", self)
        self.save_masked_button = QPushButton("Save Masked FITS", self)
        self.save_mask_01_button = QPushButton("Save Mask as FITS", self)
        bottom_layout.addWidget(self.reset_button, 0, 0)
        bottom_layout.addWidget(self.save_masked_button, 0, 1)
        bottom_layout.addWidget(self.save_mask_01_button, 1, 0, 1, 2)

        self.reset_button.clicked.connect(self.reset_mask)
        self.save_masked_button.clicked.connect(self.save_masked_fits)
        self.save_mask_01_button.clicked.connect(self.save_mask_as_fits)

        self.return_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Return), self)
        self.return_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.return_shortcut.activated.connect(self._handle_enter_shortcut)
        self.enter_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Enter), self)
        self.enter_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.enter_shortcut.activated.connect(self._handle_enter_shortcut)

        main_layout.addLayout(bottom_layout)
        self.setLayout(main_layout)
        self._update_moment_mask_availability()
        self._set_default_action_for_current_tab()
        self.adjustSize()
        self.move_to_default_position()

    def _moment_mask_available(self) -> bool:
        data = getattr(self.fits_viewer, "data", None)
        return bool(getattr(data, "ndim", 0) >= 3)

    def _handle_enter_shortcut(self):
        if not self.isVisible():
            return
        if (
            hasattr(self, "advanced_dialog")
            and self.advanced_dialog is not None
            and self.advanced_dialog.isVisible()
            and self.advanced_dialog.isActiveWindow()
        ):
            return
        current_tab = self.tab_widget.currentWidget()
        if current_tab is self.moment_mask_tab and self._moment_mask_available():
            self.apply_moment_mask()
            return
        if current_tab is self.data_mask_tab:
            self.apply_threshold_mask()

    def _update_moment_mask_availability(self):
        enabled = self._moment_mask_available()
        moment_index = self.tab_widget.indexOf(self.moment_mask_tab)
        if moment_index >= 0:
            self.tab_widget.setTabEnabled(moment_index, enabled)
        self.moment_mask_tab.setEnabled(enabled)
        if not enabled:
            data_index = self.tab_widget.indexOf(self.data_mask_tab)
            if data_index >= 0:
                self.tab_widget.setCurrentIndex(data_index)
            self.tab_widget.setTabToolTip(
                moment_index,
                "Moment Mask requires 3D/4D cube data.",
            )
        else:
            self.tab_widget.setTabToolTip(moment_index, "")
        self._set_default_action_for_current_tab()

    def _on_tab_changed(self, _index: int):
        self._set_default_action_for_current_tab()
        QTimer.singleShot(0, self._defocus_moment_controls_if_needed)

    def _set_default_action_for_current_tab(self):
        self.apply_threshold_button.setDefault(False)
        self.apply_moment_button.setDefault(False)
        self.apply_threshold_button.setAutoDefault(True)
        self.apply_moment_button.setAutoDefault(False)

        current_tab = self.tab_widget.currentWidget()
        if current_tab is self.data_mask_tab:
            self.apply_threshold_button.setDefault(True)

    def _defocus_moment_controls_if_needed(self):
        if self.tab_widget.currentWidget() is not self.moment_mask_tab:
            return
        focused_widget = self.focusWidget()
        if focused_widget is not None and focused_widget is not self.tab_widget:
            focused_widget.clearFocus()
        self.tab_widget.setFocus(Qt.FocusReason.OtherFocusReason)

    def _build_data_mask_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        grid_layout = QGridLayout()

        threshold_group = QGroupBox("Create Mask by Threshold")
        threshold_grid = QGridLayout(threshold_group)
        self.threshold_label = QLabel("Value:")
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
        self.load_mask_button = QPushButton("Load Mask FITS", self)
        external_mask_layout.addWidget(self.load_mask_button)
        grid_layout.addWidget(external_mask_group, 1, 0, 1, 2)

        self.apply_threshold_button = QPushButton("Apply Threshold", self)
        grid_layout.addWidget(self.apply_threshold_button, 2, 0, 1, 2)

        self.apply_threshold_button.clicked.connect(self.apply_threshold_mask)
        self.load_mask_button.clicked.connect(self.load_and_apply_mask)

        self.apply_threshold_button.setAutoDefault(True)
        self.apply_threshold_button.setDefault(True)
        self.threshold_input.setFocus()

        layout.addLayout(grid_layout)
        return tab

    def _make_double_spin(
        self,
        minimum: float,
        maximum: float,
        step: float,
        decimals: int,
        parent: QWidget | None = None,
    ) -> QDoubleSpinBox:
        widget = QDoubleSpinBox(parent if parent is not None else self)
        widget.setRange(minimum, maximum)
        widget.setSingleStep(step)
        widget.setDecimals(decimals)
        widget.setKeyboardTracking(False)
        return widget

    def _make_int_spin(
        self,
        minimum: int,
        maximum: int,
        step: int = 1,
        parent: QWidget | None = None,
    ) -> QSpinBox:
        widget = QSpinBox(parent if parent is not None else self)
        widget.setRange(minimum, maximum)
        widget.setSingleStep(step)
        widget.setKeyboardTracking(False)
        return widget

    def _build_moment_mask_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        grid = QGridLayout()

        algorithm_group = QGroupBox("Automatic Moment Mask")
        algorithm_layout = QGridLayout(algorithm_group)

        algorithm_layout.addWidget(QLabel("Algorithm:"), 0, 0)
        self.moment_algorithm_combo = QComboBox(self)
        self.moment_algorithm_combo.addItem("Dilated Masking (Seed+Grow)", "smoothed_hysteresis")
        self.moment_algorithm_combo.addItem("Moment Masking", "moment_masking")
        algorithm_layout.addWidget(self.moment_algorithm_combo, 0, 1)

        algorithm_layout.addWidget(QLabel("Polarity:"), 1, 0)
        self.moment_polarity_combo = QComboBox(self)
        self.moment_polarity_combo.addItem("Emission (+)", "emission")
        self.moment_polarity_combo.addItem("Absorption (-)", "absorption")
        algorithm_layout.addWidget(self.moment_polarity_combo, 1, 1)

        algorithm_layout.addWidget(QLabel("Preset:"), 2, 0)
        self.moment_preset_combo = QComboBox(self)
        self.moment_preset_combo.addItem("Faint", "faint")
        self.moment_preset_combo.addItem("Normal", "normal")
        self.moment_preset_combo.addItem("Strict", "strict")
        self.moment_preset_combo.setCurrentIndex(1)
        algorithm_layout.addWidget(self.moment_preset_combo, 2, 1)

        self.advanced_button = QPushButton("Advanced Settings...", self)
        self.advanced_button.clicked.connect(self._open_moment_advanced_dialog)
        algorithm_layout.addWidget(self.advanced_button, 3, 0, 1, 2)

        grid.addWidget(algorithm_group, 0, 0, 1, 2)

        self.apply_moment_button = QPushButton("Apply Moment Mask", self)
        grid.addWidget(self.apply_moment_button, 1, 0, 1, 2)
        self.apply_moment_button.clicked.connect(self.apply_moment_mask)
        self.apply_moment_button.setAutoDefault(False)

        layout.addLayout(grid)

        self.moment_preset_combo.currentIndexChanged.connect(self._on_moment_preset_changed)
        self.moment_algorithm_combo.currentIndexChanged.connect(self._update_algorithm_specific_controls)

        self._create_moment_advanced_dialog()
        self._apply_moment_tooltips()

        self._on_moment_preset_changed()
        self._update_algorithm_specific_controls()

        return tab

    def _create_moment_advanced_dialog(self):
        self.advanced_dialog = QDialog(self)
        self.advanced_dialog.setWindowTitle("Moment Mask Advanced Settings")

        dialog_layout = QVBoxLayout(self.advanced_dialog)
        self.advanced_group = QGroupBox("Advanced Parameters", self.advanced_dialog)
        adv_layout = QGridLayout(self.advanced_group)

        self.smooth_xy_label = QLabel("Smooth XY [pix]", self.advanced_group)
        self.smooth_xy_spin = self._make_double_spin(0.0, 20.0, 0.1, 2, parent=self.advanced_group)
        adv_layout.addWidget(self.smooth_xy_label, 0, 0)
        adv_layout.addWidget(self.smooth_xy_spin, 0, 1)

        self.smooth_v_label = QLabel("Smooth V [ch]", self.advanced_group)
        self.smooth_v_spin = self._make_double_spin(0.0, 20.0, 0.1, 2, parent=self.advanced_group)
        adv_layout.addWidget(self.smooth_v_label, 1, 0)
        adv_layout.addWidget(self.smooth_v_spin, 1, 1)

        self.seed_sigma_label = QLabel("Seed sigma", self.advanced_group)
        self.seed_sigma_spin = self._make_double_spin(0.1, 20.0, 0.1, 2, parent=self.advanced_group)
        adv_layout.addWidget(self.seed_sigma_label, 2, 0)
        adv_layout.addWidget(self.seed_sigma_spin, 2, 1)

        self.grow_sigma_label = QLabel("Grow sigma", self.advanced_group)
        self.grow_sigma_spin = self._make_double_spin(0.1, 20.0, 0.1, 2, parent=self.advanced_group)
        adv_layout.addWidget(self.grow_sigma_label, 3, 0)
        adv_layout.addWidget(self.grow_sigma_spin, 3, 1)

        self.clip_sigma_label = QLabel("Clip sigma", self.advanced_group)
        self.clip_sigma_spin = self._make_double_spin(0.1, 20.0, 0.1, 2, parent=self.advanced_group)
        adv_layout.addWidget(self.clip_sigma_label, 4, 0)
        adv_layout.addWidget(self.clip_sigma_spin, 4, 1)

        self.expand_xy_label = QLabel("Expand XY [pix]", self.advanced_group)
        self.expand_xy_spin = self._make_int_spin(0, 20, 1, parent=self.advanced_group)
        adv_layout.addWidget(self.expand_xy_label, 5, 0)
        adv_layout.addWidget(self.expand_xy_spin, 5, 1)

        self.expand_v_label = QLabel("Expand V [ch]", self.advanced_group)
        self.expand_v_spin = self._make_int_spin(0, 20, 1, parent=self.advanced_group)
        adv_layout.addWidget(self.expand_v_label, 6, 0)
        adv_layout.addWidget(self.expand_v_spin, 6, 1)

        self.min_channels_label = QLabel("Min channels", self.advanced_group)
        self.min_channels_spin = self._make_int_spin(1, 256, 1, parent=self.advanced_group)
        adv_layout.addWidget(self.min_channels_label, 7, 0)
        adv_layout.addWidget(self.min_channels_spin, 7, 1)

        self.min_voxels_label = QLabel("Min voxels", self.advanced_group)
        self.min_voxels_spin = self._make_int_spin(0, 100000000, 10, parent=self.advanced_group)
        adv_layout.addWidget(self.min_voxels_label, 8, 0)
        adv_layout.addWidget(self.min_voxels_spin, 8, 1)

        self.connectivity_label = QLabel("Connectivity", self.advanced_group)
        self.connectivity_combo = QComboBox(self.advanced_group)
        self.connectivity_combo.addItem("6", 6)
        self.connectivity_combo.addItem("18", 18)
        self.connectivity_combo.addItem("26", 26)
        adv_layout.addWidget(self.connectivity_label, 9, 0)
        adv_layout.addWidget(self.connectivity_combo, 9, 1)

        dialog_layout.addWidget(self.advanced_group)

        button_layout = QHBoxLayout()
        self.advanced_reset_button = QPushButton("Reset to Preset", self.advanced_dialog)
        self.advanced_close_button = QPushButton("Close", self.advanced_dialog)
        self.advanced_reset_button.clicked.connect(self._on_moment_preset_changed)
        self.advanced_close_button.clicked.connect(self.advanced_dialog.close)
        button_layout.addWidget(self.advanced_reset_button)
        button_layout.addWidget(self.advanced_close_button)
        dialog_layout.addLayout(button_layout)

        self.advanced_dialog.setLayout(dialog_layout)
        self.advanced_dialog.adjustSize()

    def _set_tooltip(self, widgets: list[QWidget], text: str):
        for widget in widgets:
            widget.setToolTip(text)

    def _apply_moment_tooltips(self):
        self.moment_algorithm_combo.setToolTip(
            "Choose the automatic masking recipe used to detect signal voxels."
        )
        self.moment_polarity_combo.setToolTip(
            "Select emission (+) or absorption (-) feature detection."
        )
        self.moment_preset_combo.setToolTip(
            "Preset values for smoothing and threshold parameters."
        )
        self.advanced_button.setToolTip(
            "Open advanced parameter settings for the selected recipe."
        )
        self.apply_moment_button.setToolTip(
            "Generate and apply an automatic mask with current settings."
        )

        self._set_tooltip(
            [self.smooth_xy_label, self.smooth_xy_spin],
            "Gaussian smoothing sigma along spatial axes, in pixels.",
        )
        self._set_tooltip(
            [self.smooth_v_label, self.smooth_v_spin],
            "Gaussian smoothing sigma along spectral axis, in channels.",
        )
        self._set_tooltip(
            [self.seed_sigma_label, self.seed_sigma_spin],
            "Seed threshold in sigma on the smoothed cube (Seed+Grow).",
        )
        self._set_tooltip(
            [self.grow_sigma_label, self.grow_sigma_spin],
            "Grow threshold in sigma on the original cube (Seed+Grow).",
        )
        self._set_tooltip(
            [self.clip_sigma_label, self.clip_sigma_spin],
            "Threshold in sigma on smoothed cube (Moment Masking).",
        )
        self._set_tooltip(
            [self.expand_xy_label, self.expand_xy_spin],
            "Spatial expansion radius in pixels (Moment Masking).",
        )
        self._set_tooltip(
            [self.expand_v_label, self.expand_v_spin],
            "Spectral expansion radius in channels (Moment Masking).",
        )
        self._set_tooltip(
            [self.min_channels_label, self.min_channels_spin],
            "Minimum spectral extent for keeping a connected component.",
        )
        self._set_tooltip(
            [self.min_voxels_label, self.min_voxels_spin],
            "Minimum voxel count for keeping a connected component.",
        )
        self._set_tooltip(
            [self.connectivity_label, self.connectivity_combo],
            "Connectivity used in 3D component labeling (6/18/26).",
        )
        self.advanced_reset_button.setToolTip(
            "Reset advanced values to the currently selected preset."
        )
        self.advanced_close_button.setToolTip("Close this settings dialog.")

    def _open_moment_advanced_dialog(self):
        self._update_algorithm_specific_controls()
        self.advanced_dialog.adjustSize()
        self.advanced_dialog.show()
        self.advanced_dialog.raise_()
        self.advanced_dialog.activateWindow()

    def _on_moment_preset_changed(self):
        preset_key = self.moment_preset_combo.currentData()
        params = usecases.get_moment_mask_preset(preset_key)
        self.smooth_xy_spin.setValue(float(params["smooth_xy_pix"]))
        self.smooth_v_spin.setValue(float(params["smooth_v_chan"]))
        self.seed_sigma_spin.setValue(float(params["seed_sigma"]))
        self.grow_sigma_spin.setValue(float(params["grow_sigma"]))
        self.clip_sigma_spin.setValue(float(params["clip_sigma"]))
        self.expand_xy_spin.setValue(int(params["expand_xy_pix"]))
        self.expand_v_spin.setValue(int(params["expand_v_chan"]))
        self.min_channels_spin.setValue(int(params["min_channels"]))
        self.min_voxels_spin.setValue(int(params["min_voxels"]))
        self._set_connectivity_combo_value(int(params["connectivity"]))

    def _set_connectivity_combo_value(self, connectivity: int):
        for index in range(self.connectivity_combo.count()):
            if int(self.connectivity_combo.itemData(index)) == int(connectivity):
                self.connectivity_combo.setCurrentIndex(index)
                return
        self.connectivity_combo.setCurrentIndex(self.connectivity_combo.count() - 1)

    def _update_algorithm_specific_controls(self):
        algorithm = self.moment_algorithm_combo.currentData()
        use_a = algorithm == "smoothed_hysteresis"
        use_b = algorithm == "moment_masking"

        self.seed_sigma_label.setVisible(use_a)
        self.seed_sigma_spin.setVisible(use_a)
        self.grow_sigma_label.setVisible(use_a)
        self.grow_sigma_spin.setVisible(use_a)

        self.clip_sigma_label.setVisible(use_b)
        self.clip_sigma_spin.setVisible(use_b)
        self.expand_xy_label.setVisible(use_b)
        self.expand_xy_spin.setVisible(use_b)
        self.expand_v_label.setVisible(use_b)
        self.expand_v_spin.setVisible(use_b)

    def move_to_default_position(self):
        if hasattr(self.fits_viewer, "control_panel"):
            cp_geom = self.fits_viewer.control_panel.geometry()
            self.move(cp_geom.x() + cp_geom.width(), cp_geom.y())

    def _ensure_original_data(self):
        if self.original_data is None:
            self.original_data = create_preview_snapshot(
                self.fits_viewer.data,
                operation_name="Masking",
            )

    def _collect_moment_params(self) -> dict:
        return {
            "algorithm": self.moment_algorithm_combo.currentData(),
            "polarity": self.moment_polarity_combo.currentData(),
            "preset": self.moment_preset_combo.currentData(),
            "smooth_xy_pix": float(self.smooth_xy_spin.value()),
            "smooth_v_chan": float(self.smooth_v_spin.value()),
            "seed_sigma": float(self.seed_sigma_spin.value()),
            "grow_sigma": float(self.grow_sigma_spin.value()),
            "clip_sigma": float(self.clip_sigma_spin.value()),
            "expand_xy_pix": int(self.expand_xy_spin.value()),
            "expand_v_chan": int(self.expand_v_spin.value()),
            "min_channels": int(self.min_channels_spin.value()),
            "min_voxels": int(self.min_voxels_spin.value()),
            "connectivity": int(self.connectivity_combo.currentData()),
            "noise_method": "diff_mad",
        }

    def apply_threshold_mask(self):
        self._ensure_original_data()
        try:
            threshold = float(self.threshold_input.text())
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Please enter a valid numeric threshold.")
            return

        condition = "less_than" if self.mask_below_radio.isChecked() else "greater_than"
        try:
            masked_data = usecases.compute_masked(self.original_data, threshold, condition)
            current_mask = np.isfinite(masked_data)
            if np.all(~current_mask):
                QMessageBox.warning(
                    self,
                    "Display Warning",
                    "All pixels were masked by the threshold.\nDisplay might be empty, but the mask is applied.",
                )
            self.current_mask = current_mask
            symbol = "<" if condition == "less_than" else ">"
            self.last_mask_history_entries = [f"Threshold mask: value {symbol} {threshold}"]
            self._mask_export_metadata = {
                "threshold": float(threshold),
                "condition": condition,
            }
            self._update_data_and_displays(masked_data, "[MASK ACTIVE]")
            self._record_preview_action(
                "apply_mask_threshold",
                {"threshold": float(threshold), "condition": condition},
            )
            self._has_pending_changes = True
        except Exception as exc:
            QMessageBox.critical(self, "Processing Error", f"An error occurred: {exc}")

    def apply_moment_mask(self):
        if not self._moment_mask_available():
            QMessageBox.information(
                self,
                "Moment Mask Unavailable",
                "Moment mask is available only for 3D/4D cube data.",
            )
            return
        self._ensure_original_data()
        params = self._collect_moment_params()
        try:
            masked_data, mask = compute_moment_masked(
                self.original_data,
                **params,
            )

            self.current_mask = mask.astype(bool)
            algorithm = params["algorithm"]
            preset = params["preset"]
            polarity = params["polarity"]
            self.last_mask_history_entries = [
                f"Moment mask algorithm={algorithm}",
                f"Moment mask preset={preset}, polarity={polarity}",
                (
                    "smooth_xy_pix={smooth_xy_pix}, smooth_v_chan={smooth_v_chan}, "
                    "seed_sigma={seed_sigma}, grow_sigma={grow_sigma}, clip_sigma={clip_sigma}, "
                    "expand_xy_pix={expand_xy_pix}, expand_v_chan={expand_v_chan}, "
                    "min_channels={min_channels}, min_voxels={min_voxels}, connectivity={connectivity}, "
                    "noise_method={noise_method}"
                ).format(**params),
            ]
            self._mask_export_metadata = {}
            status = f"[MOMENT MASK: {preset}/{polarity}]"
            self._update_data_and_displays(masked_data, status)
            self._record_preview_action("apply_mask_moment_recipe", params)
            self._has_pending_changes = True
        except Exception as exc:
            QMessageBox.critical(self, "Processing Error", f"Failed to apply moment mask: {exc}")

    def load_and_apply_mask(self):
        self._ensure_original_data()
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Mask FITS", "", "FITS Files (*.fits *.fit)"
        )
        if not filename:
            return

        try:
            with fits.open(filename) as hdul:
                mask_data = hdul[0].data
                mask_header = hdul[0].header
            mask_wcs = WCS(mask_header)
            if mask_data.shape != self.original_data.shape:
                QMessageBox.critical(
                    self,
                    "Shape Mismatch",
                    f"Mask dimensions ({mask_data.shape}) do not match data dimensions ({self.original_data.shape}).",
                )
                return
            if self.fits_viewer.wcs.to_header_string() != mask_wcs.to_header_string():
                reply = QMessageBox.warning(
                    self,
                    "WCS Mismatch",
                    "Mask coordinates may not match data.\nApply anyway (pixel-to-pixel)?",
                    QMessageBox.StandardButton.Apply | QMessageBox.StandardButton.Cancel,
                    QMessageBox.StandardButton.Cancel,
                )
                if reply == QMessageBox.StandardButton.Cancel:
                    return

            current_mask = np.isfinite(mask_data) & (mask_data != 0)
            masked_data = compute_masked_from_keep_mask(
                self.original_data,
                current_mask,
                operation_name="External masking",
            )

            self.current_mask = current_mask
            self.last_mask_history_entries = [f"External mask applied: {os.path.basename(filename)}"]
            self._mask_export_metadata = {}
            self._update_data_and_displays(masked_data, f"[EXTERNAL MASK: {os.path.basename(filename)}]")
            self._record_preview_action(
                "apply_mask_external",
                {"mask_path": filename, "mask_value": 0.0},
            )
            self._has_pending_changes = True
        except Exception as exc:
            QMessageBox.critical(self, "File Error", f"Failed to load or apply mask: {exc}")

    def reset_mask(self):
        preferred_cursor = capture_preferred_cursor_snapshot(self.fits_viewer)
        removed_preview = self._clear_preview_action()
        restored_from_history = False
        if removed_preview:
            restored_from_history = replay_action_history_to_current_cursor(
                self.fits_viewer,
                preferred_cursor=preferred_cursor,
            )

        if not restored_from_history and self.original_data is not None:
            self._update_data_and_displays(self.original_data)
        else:
            self.fits_viewer.setWindowTitle(self.fits_viewer.original_window_title)
        self.current_mask = None
        self.last_mask_history_entries = []
        self._mask_export_metadata = {}
        self._has_pending_changes = False

    def _record_preview_action(self, action_name: str, params: dict) -> None:
        record_action_preview(
            self.fits_viewer,
            action_name,
            params,
            replace_tag=self._action_record_tag,
        )

    def _clear_preview_action(self) -> bool:
        for action_name in (
            "apply_mask_threshold",
            "apply_mask_external",
            "apply_mask_moment_recipe",
        ):
            removed = clear_action_preview_record(
                self.fits_viewer,
                self._action_record_tag,
                action_name=action_name,
            )
            if removed:
                return True
        return False

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
            if not window:
                continue
            current_channel = window.current_channel_index()
            data_slice = None
            if window.data.ndim == 4:
                if window.plane == "xy":
                    data_slice = window.data[0, current_channel, :, :]
                elif window.plane == "xz":
                    data_slice = window.data[0, :, current_channel, :]
                elif window.plane == "zy":
                    data_slice = window.data[0, :, :, current_channel].T
            elif window.data.ndim == 3:
                if window.plane == "xy":
                    data_slice = window.data[current_channel, :, :]
                elif window.plane == "xz":
                    data_slice = window.data[:, current_channel, :]
                elif window.plane == "zy":
                    data_slice = window.data[:, :, current_channel].T
            elif window.data.ndim == 2:
                data_slice = window.data
            if data_slice is not None:
                window.im.set_data(data_slice)
                window.canvas.draw()

    def _get_sanitized_header(self):
        new_header = self.fits_viewer.header.copy()
        try:
            original_header_on_disk = fits.getheader(self.fits_viewer.filename)
            if "CUNIT3" not in original_header_on_disk and "CUNIT3" in new_header:
                del new_header["CUNIT3"]
        except Exception as exc:
            print(f"Could not check original header on disk. Error: {exc}")
        return new_header

    def save_masked_fits(self):
        if self.current_mask is None:
            QMessageBox.warning(self, "No Mask", "Please apply a mask first.")
            return

        try:
            new_header = self._get_sanitized_header()
            new_header["DATAMAX"] = float(np.nanmax(self.fits_viewer.data))

            for entry in build_processing_history_lines(self.fits_viewer):
                new_header.add_history(entry)

            save_dialog = SaveFITS(self.fits_viewer.data, new_header, self.fits_viewer.filename)
            save_dialog.save(suffix="masked")
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"An error occurred while saving the masked FITS: {exc}")

    def save_mask_as_fits(self):
        if self.current_mask is None:
            QMessageBox.warning(self, "No Mask", "Please apply a mask first before saving it.")
            return

        try:
            mask_data = np.asarray(self.current_mask, dtype=bool).astype(
                np.float32,
                copy=False,
            )
            save_dialog = SaveFITS(mask_data, self.fits_viewer.header, self.fits_viewer.filename)
            default_filename = save_dialog.generate_new_filename("mask")
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Save FITS File",
                default_filename,
                "FITS Files (*.fits);;All Files (*)",
            )
            if not filename:
                return

            state = getattr(self.fits_viewer, "app_state", None)
            if state is None:
                state = create_app_state(
                    data=getattr(self.fits_viewer, "data", None),
                    header=getattr(self.fits_viewer, "header", None),
                    wcs=getattr(self.fits_viewer, "wcs", None),
                    filepath=getattr(self.fits_viewer, "filename", None),
                )

            usecases.export_mask_fits(
                state,
                mask_data,
                filename,
                history_entries=build_processing_history_lines(self.fits_viewer),
                mask_as_float=True,
                **dict(self._mask_export_metadata),
            )
            QMessageBox.information(self, "Save Successful", f"FITS successfully saved as: {filename}")
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"An error occurred while saving the mask file: {exc}")

    def closeEvent(self, event):
        super().closeEvent(event)

    def has_pending_changes(self) -> bool:
        return self._has_pending_changes

    def pending_close_title(self) -> str:
        return "Close Mask Panel"

    def pending_close_text(self) -> str:
        return "There are unapplied mask changes."

    def discard_pending_changes(self) -> None:
        self.reset_mask()

    def resync_after_workspace_restore(self) -> None:
        pending = bool(has_action_record_tag(self.fits_viewer, self._action_record_tag))
        self._has_pending_changes = pending
        if pending and self.current_mask is None:
            try:
                self.current_mask = np.isfinite(np.asarray(self.fits_viewer.data))
            except Exception:
                self.current_mask = None
