from functools import partial
from typing import Dict, List, Optional, Tuple
from astropy import units as u
from astropy.coordinates import Angle
from PySide6.QtCore import Qt, QSize, Signal as pyqtSignal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from takefits.tools.base_panel import confirm_pending_close


class RegridPanel(QWidget):
    """UI panel for configuring the regrid feature."""

    regrid_requested = pyqtSignal(dict)
    closed = pyqtSignal()

    def __init__(self, fits_viewer):
        super().__init__()
        self.fits_viewer = fits_viewer
        self.wcs = fits_viewer.wcs
        self.header = getattr(fits_viewer, "header", None)

        config_manager = getattr(fits_viewer, "config_manager", None)
        self.config = getattr(config_manager, "config", {}) if config_manager else {}
        self.decimal = self.config.get("decimal", True)
        self.number_decimals = self.config.get("number_decimals", 6)

        self.axis_controls: List[Dict[str, object]] = []
        self._manual_row_heights: List[int] = []

        self.template_path_edit: Optional[QLineEdit] = None
        self.target_system_combo: Optional[QComboBox] = None
        self.current_system_label: Optional[QLabel] = None
        self.interpolation_combo: Optional[QComboBox] = None
        self.run_button: Optional[QPushButton] = None
        self.progress_bar: Optional[QProgressBar] = None
        self._shortcuts: List[QShortcut] = []
        self._current_system = self._determine_current_system()
        self._target_systems = self._build_target_system_list()
        self._regrid_running = False

        self.setWindowTitle(f"Regrid: {self.fits_viewer.filename}")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self._init_ui()

    # ------------------------------------------------------------------
    # UI construction
    def _init_ui(self):
        root_layout = QHBoxLayout()
        root_layout.setContentsMargins(12, 12, 12, 6)
        root_layout.setSpacing(16)
        self.setLayout(root_layout)

        self.tab_widget = QTabWidget()
        self.tab_widget.setDocumentMode(False)
        root_layout.addWidget(self.tab_widget, stretch=4)

        self.manual_tab = QWidget()
        self.template_tab = QWidget()
        self.transform_tab = QWidget()
        self.manual_tab_index = self.tab_widget.addTab(self.manual_tab, "Manual Grid")
        self.template_tab_index = self.tab_widget.addTab(self.template_tab, "Template FITS")
        self.transform_tab_index = self.tab_widget.addTab(self.transform_tab, "Coordinate Transform")

        supported_systems = {"galactic", "fk4", "fk5", "icrs"}
        is_supported = (self._current_system or "").strip().lower() in supported_systems
        if not is_supported:
            self.tab_widget.setTabEnabled(self.transform_tab_index, False)
            self.tab_widget.setTabToolTip(
                self.transform_tab_index,
                "Coordinate transform is only supported for ICRS, FK5, FK4, and Galactic systems.",
            )

        self._build_manual_tab()
        self._build_template_tab()
        self._build_transform_tab()

        side_container = QWidget()
        side_layout = QVBoxLayout()
        side_layout.setSpacing(4)
        side_layout.setContentsMargins(1, 1, 1, 1)
        side_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._build_side_panel(side_layout)
        side_container.setLayout(side_layout)
        root_layout.addWidget(side_container, stretch=1)

        hint = self.sizeHint()
        self.resize(hint)
        self.setMinimumSize(hint)

    def sizeHint(self):
        base_hint = super().sizeHint()
        return QSize(max(base_hint.width(), 520), max(base_hint.height(), 165))

    def _build_manual_tab(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.manual_tab.setLayout(layout)

        self._manual_row_heights.clear()

        manual_group = QGroupBox("Manual Grid Definition")
        manual_layout = QGridLayout()
        manual_layout.setContentsMargins(8, 4, 8, 6)
        manual_layout.setHorizontalSpacing(8)
        manual_layout.setVerticalSpacing(6)
        manual_group.setLayout(manual_layout)

        headers = [("Axis", 0), ("Pixel Step", 1), ("Grid Width", 2), ("Reference (CRVAL)", 3)]
        for text, column in headers:
            label = QLabel(text)
            manual_layout.addWidget(label, 0, column)
            #spacer = QWidget()
            #spacer.setFixedHeight(3)
            #manual_layout.addWidget(spacer, 1, column)

        header_offset = 1

        ctype_list = getattr(self.wcs.wcs, "ctype", [])
        crval_list = getattr(self.wcs.wcs, "crval", [])
        cdelt_list = getattr(self.wcs.wcs, "cdelt", [])
        naxis = getattr(self.wcs.wcs, "naxis", len(ctype_list))

        axis_letters = ["X", "Y", "Z", "U", "V", "W"]

        for index in range(naxis):
            ctype = ctype_list[index] if index < len(ctype_list) else f"AXIS{index + 1}"
            crval = crval_list[index] if index < len(crval_list) else 0.0
            cdelt = cdelt_list[index] if index < len(cdelt_list) else 0.0
            abs_cdelt = abs(cdelt)
            sign = -1.0 if cdelt < 0 else 1.0
            if cdelt == 0:
                sign = 1.0

            axis_label = axis_letters[index] if index < len(axis_letters) else f"A{index + 1}"
            pretty_ctype = ctype.split("-")[0] if ctype else f"AXIS{index + 1}"
            row = index + 1 + header_offset

            axis_name_label = QLabel(f"{axis_label} ({pretty_ctype})")
            axis_name_label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
            manual_layout.addWidget(axis_name_label, row, 0)

            pixel_spin = QDoubleSpinBox()
            pixel_spin.setDecimals(2)
            pixel_spin.setRange(0.0, 1e9)
            pixel_spin.setSingleStep(1.0)
            pixel_spin.setValue(1.0)
            pixel_spin.setAlignment(Qt.AlignmentFlag.AlignLeft)
            digit_width = pixel_spin.fontMetrics().horizontalAdvance("0") * 5
            pixel_spin.setFixedWidth(digit_width + 32)
            manual_layout.addWidget(pixel_spin, row, 1)

            world_edit = QLineEdit()
            world_edit.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
            world_edit.setText(self._format_spacing_value(abs_cdelt, ctype))
            manual_layout.addWidget(world_edit, row, 2)

            anchor_edit = QLineEdit()
            anchor_edit.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
            anchor_edit.setText(self._format_world_value(crval, ctype))
            manual_layout.addWidget(anchor_edit, row, 3)

            control = {
                "axis_index": index,
                "ctype": ctype,
                "pixel_spin": pixel_spin,
                "world_edit": world_edit,
                "anchor_edit": anchor_edit,
                "abs_cdelt": abs_cdelt,
                "sign": sign,
                "anchor_value": crval,
                "spacing_value": abs_cdelt if abs_cdelt > 0 else 0.0,
                "updating": False,
            }
            row_height = max(pixel_spin.sizeHint().height(), world_edit.sizeHint().height()) + 4
            manual_layout.setRowMinimumHeight(row, row_height)
            self._manual_row_heights.append(row_height)
            self.axis_controls.append(control)

            pixel_spin.valueChanged.connect(partial(self._on_pixel_spacing_changed, control))
            world_edit.editingFinished.connect(partial(self._on_world_spacing_edited, control))
            anchor_edit.editingFinished.connect(partial(self._on_anchor_edited, control))

        layout.addWidget(manual_group)

    def _build_template_tab(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.template_tab.setLayout(layout)

        template_group = QGroupBox("Resample to Template FITS")
        template_layout = QHBoxLayout()
        template_layout.setContentsMargins(8, 8, 8, 8)
        template_group.setLayout(template_layout)

        self.template_path_edit = QLineEdit()
        self.template_path_edit.setReadOnly(True)
        template_button = QPushButton("Select Template FITS…")
        template_button.clicked.connect(self._select_template_fits)

        template_layout.addWidget(self.template_path_edit)
        template_layout.addWidget(template_button)

        layout.addWidget(template_group)

    def _build_transform_tab(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.transform_tab.setLayout(layout)

        transform_group = QGroupBox("Reproject Coordinate System")
        transform_layout = QVBoxLayout()
        transform_layout.setContentsMargins(8, 8, 8, 8)
        transform_group.setLayout(transform_layout)

        current_layout = QHBoxLayout()
        current_label_title = QLabel("Current System:")
        current_label_title.setMinimumWidth(120)

        current_system_key = (self._current_system or "").strip().lower()
        system_map = dict(self._target_systems)
        current_system_display = system_map.get(
            current_system_key, (current_system_key or "Unknown").upper()
        )

        self.current_system_label = QLabel(current_system_display)
        current_layout.addWidget(current_label_title)
        current_layout.addWidget(self.current_system_label)
        current_layout.addStretch(1)

        target_layout = QHBoxLayout()
        target_layout.addWidget(QLabel("Target System:"))
        self.target_system_combo = QComboBox()
        for key, text in self._target_systems:
            if key == current_system_key:
                continue
            self.target_system_combo.addItem(text, key)
        if self.target_system_combo.count() == 0:
            self.target_system_combo.setEnabled(False)
            transform_group.setToolTip("No alternative coordinate systems available.")
        target_layout.addWidget(self.target_system_combo)
        target_layout.addStretch(1)

        transform_layout.addLayout(current_layout)
        transform_layout.addLayout(target_layout)

        layout.addWidget(transform_group)

    def _build_side_panel(self, layout: QVBoxLayout):
        interpolation_group = QGroupBox("Interpolation")
        interpolation_layout = QVBoxLayout()
        interpolation_layout.setContentsMargins(10, 6, 10, 6)
        interpolation_layout.setSpacing(4)
        interpolation_group.setLayout(interpolation_layout)

        self.interpolation_combo = QComboBox()
        self.interpolation_combo.addItems(
            ["Bilinear", "Nearest", "Bicubic", "Biquadratic"]
        )

        algorithm_label = QLabel("Algorithm")
        interpolation_layout.addWidget(algorithm_label)
        interpolation_layout.addWidget(self.interpolation_combo)

        first_row_height = self._manual_row_heights[0] if self._manual_row_heights else interpolation_group.sizeHint().height()
        interpolation_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        target_height = max(first_row_height, interpolation_group.sizeHint().height())
        interpolation_group.setMinimumHeight(target_height)
        interpolation_group.setMaximumHeight(target_height)

        layout.addWidget(interpolation_group)

        layout.addStretch(1)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setFixedHeight(0)
        layout.addWidget(self.progress_bar)

        self.run_button = QPushButton("Run Regrid")
        self.run_button.clicked.connect(self._trigger_regrid)
        self.run_button.setDefault(True)
        self.run_button.setAutoDefault(True)
        layout.addWidget(self.run_button)

        for key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(self._trigger_regrid)
            self._shortcuts.append(shortcut)

    # ------------------------------------------------------------------
    # Event handlers
    def _select_template_fits(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Template FITS",
            "",
            "FITS Files (*.fits *.FITS);;All Files (*)",
        )
        if file_path and self.template_path_edit:
            self.template_path_edit.setText(file_path)

    def _trigger_regrid(self):
        params = self.collect_parameters()
        if not params:
            return
        self.regrid_requested.emit(params)

    def _on_pixel_spacing_changed(self, control, value):
        if control["updating"]:
            return

        abs_cdelt = control["abs_cdelt"]
        spacing = abs_cdelt * value
        control["spacing_value"] = spacing

        control["updating"] = True
        try:
            control["world_edit"].setText(self._format_spacing_value(spacing, control["ctype"]))
        finally:
            control["updating"] = False

    def _on_world_spacing_edited(self, control):
        if control["updating"]:
            return

        world_edit: QLineEdit = control["world_edit"]
        text = world_edit.text().strip()
        if not text:
            QMessageBox.warning(self, "Invalid Input", "Grid width must be provided.")
            world_edit.setText(
                self._format_spacing_value(control["spacing_value"], control["ctype"])
            )
            return

        try:
            magnitude = self._parse_spacing_text(text, control["ctype"])
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid Input", str(exc))
            world_edit.setText(
                self._format_spacing_value(control["spacing_value"], control["ctype"])
            )
            return

        abs_cdelt = control["abs_cdelt"]
        if abs_cdelt == 0:
            QMessageBox.warning(
                self,
                "Unsupported Axis",
                "Original grid width is zero; cannot convert world spacing to pixels.",
            )
            world_edit.setText("0")
            return

        pixel_value = magnitude / abs_cdelt
        if pixel_value <= 0:
            QMessageBox.warning(
                self,
                "Invalid Input",
                "Pixel step must stay positive.",
            )
            world_edit.setText(
                self._format_spacing_value(control["spacing_value"], control["ctype"])
            )
            return

        control["spacing_value"] = magnitude

        control["updating"] = True
        try:
            control["pixel_spin"].setValue(pixel_value)
            world_edit.setText(self._format_spacing_value(magnitude, control["ctype"]))
        finally:
            control["updating"] = False

    def _on_anchor_edited(self, control):
        anchor_edit: QLineEdit = control["anchor_edit"]
        text = anchor_edit.text().strip()
        if not text:
            QMessageBox.warning(self, "Invalid Input", "Anchor value must be provided.")
            anchor_edit.setText(
                self._format_world_value(control["anchor_value"], control["ctype"])
            )
            return
        try:
            value = self._parse_world_text(text, control["ctype"])
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid Input", str(exc))
            anchor_edit.setText(
                self._format_world_value(control["anchor_value"], control["ctype"])
            )
            return

        control["anchor_value"] = value
        anchor_edit.setText(self._format_world_value(value, control["ctype"]))

    # ------------------------------------------------------------------
    # Parameter collection
    def collect_parameters(self) -> Optional[Dict]:
        current_index = self.tab_widget.currentIndex()
        if current_index == self.manual_tab_index:
            mode = "manual"
        elif current_index == self.template_tab_index:
            mode = "template_fits"
        else:
            mode = "reproject_system"

        interpolation = (
            self.interpolation_combo.currentText() if self.interpolation_combo else "Bilinear"
        )

        params: Dict = {
            "mode": mode,
            "interpolation": interpolation,
        }

        if mode == "manual":
            anchor_values: List[float] = []
            grid_cdelt: List[float] = []

            for control in self.axis_controls:
                anchor_edit: QLineEdit = control["anchor_edit"]
                world_edit: QLineEdit = control["world_edit"]

                anchor_text = anchor_edit.text().strip()
                spacing_text = world_edit.text().strip()

                if not anchor_text or not spacing_text:
                    QMessageBox.warning(
                        self,
                        "Incomplete Input",
                        "Please fill in grid width and anchor values for every axis.",
                    )
                    return None

                try:
                    anchor_value = self._parse_world_text(anchor_text, control["ctype"])
                except ValueError as exc:
                    QMessageBox.warning(self, "Invalid Value", str(exc))
                    return None

                try:
                    spacing_value = self._parse_spacing_text(spacing_text, control["ctype"])
                except ValueError as exc:
                    QMessageBox.warning(self, "Invalid Value", str(exc))
                    return None

                if spacing_value == 0:
                    QMessageBox.warning(
                        self,
                        "Invalid Value",
                        "Grid width must be non-zero.",
                    )
                    return None

                control["anchor_value"] = anchor_value
                control["spacing_value"] = spacing_value

                anchor_values.append(anchor_value)
                grid_cdelt.append(control["sign"] * spacing_value)

            params["anchor_world"] = anchor_values
            params["grid_cdelt"] = grid_cdelt

        elif mode == "template_fits":
            template_path = self.template_path_edit.text().strip() if self.template_path_edit else ""
            if not template_path:
                QMessageBox.warning(
                    self,
                    "Missing Template",
                    "Please select a template FITS file.",
                )
                return None
            params["template_path"] = template_path

        elif mode == "reproject_system":
            if not self.target_system_combo or self.target_system_combo.count() == 0:
                QMessageBox.warning(
                    self,
                    "No Target",
                    "No alternate coordinate systems are available for reprojection.",
                )
                return None
            params["target_system"] = self.target_system_combo.currentData() or self.target_system_combo.currentText()

        return params

    # ------------------------------------------------------------------
    # Helpers
    def _determine_current_system(self) -> str:
        ctypes = [str(ctype or "").upper() for ctype in getattr(self.wcs.wcs, "ctype", [])]
        header = getattr(self.fits_viewer, "header", None)

        def _header_value(keys: List[str]) -> str:
            if not header:
                return ""
            for key in keys:
                value = header.get(key)
                if value:
                    return str(value).strip().upper()
            return ""

        # CTYPE is a strong indicator
        if any("GLON" in c or "GLAT" in c for c in ctypes):
            return "galactic"
        if any("ELON" in c or "ELAT" in c for c in ctypes):
            return "ecliptic"

        try:
            radesys = (self.wcs.wcs.radesys or "").strip().upper()
        except Exception:
            radesys = ""
        if not radesys:
            radesys = _header_value(["RADESYS", "RADECSYS"])
        
        equinox = _header_value(["EQUINOX"])

        # RADESYS is the most reliable keyword
        if "ECLIPTIC" in radesys:
            return "ecliptic"
        if "GAL" in radesys:
            return "galactic"
        if "ICRS" in radesys:
            return "icrs"
        if "FK5" in radesys:
            return "fk5"
        if "FK4" in radesys:
            return "fk4"

        # If RADESYS is absent, fall back to EQUINOX
        if equinox.startswith("J") or equinox in ("2000", "2000.0"):
            return "fk5"
        if equinox.startswith("B") or equinox in ("1950", "1950.0"):
            return "fk4"

        # If still undetermined, but celestial-like CTYPEs exist, default to ICRS
        if any("RA" in c or "DEC" in c for c in ctypes):
            return "icrs"

        # Final fallback for other celestial data is ICRS, the modern standard.
        # If WCS is not celestial, this tab shouldn't be used, but a default is needed.
        if self.wcs and self.wcs.is_celestial:
            return "icrs"

        if radesys:
            return radesys.lower()

        return ""  # Unknown

    def _build_target_system_list(self) -> List[Tuple[str, str]]:
        systems = [
            ("icrs", "ICRS"),
            ("galactic", "Galactic"),
            ("fk5", "FK5"),
        ]
        return systems

    # ------------------------------------------------------------------
    # Formatting helpers
    def _format_world_value(self, value: float, axis_type: str) -> str:
        if self.decimal:
            return f"{value:.{self.number_decimals}f}"

        if self._is_longitude(axis_type):
            angle = Angle(value, unit=u.deg)
            return angle.to_string(unit=u.hourangle, sep=":", precision=self.number_decimals, pad=True)
        if self._is_latitude(axis_type):
            angle = Angle(value, unit=u.deg)
            return angle.to_string(
                unit=u.deg,
                sep=":",
                precision=self.number_decimals,
                pad=True,
                alwayssign=True,
            )
        return f"{value:.{self.number_decimals}f}"

    def _format_spacing_value(self, value: float, axis_type: str) -> str:
        if value < 0:
            value = abs(value)
        if self.decimal:
            return f"{value:.{self.number_decimals}f}"

        angle = Angle(value, unit=u.deg)
        if self._is_longitude(axis_type):
            return angle.to_string(unit=u.hourangle, sep=":", precision=self.number_decimals, pad=True)
        if self._is_latitude(axis_type):
            return angle.to_string(unit=u.deg, sep=":", precision=self.number_decimals, pad=True)
        return f"{value:.{self.number_decimals}f}"

    def _parse_world_text(self, text: str, axis_type: str) -> float:
        try:
            return float(text)
        except ValueError:
            pass

        try:
            if self._is_longitude(axis_type):
                return Angle(text, unit=u.hourangle).degree
            if self._is_latitude(axis_type):
                return Angle(text, unit=u.deg).degree
            return Angle(text, unit=u.deg).degree
        except Exception as exc:  # pragma: no cover - relies on user input
            raise ValueError(f"Could not parse coordinate: {text}") from exc

    def _parse_spacing_text(self, text: str, axis_type: str) -> float:
        value = self._parse_world_text(text, axis_type)
        return abs(value)

    @staticmethod
    def _is_longitude(axis_type: str) -> bool:
        upper = (axis_type or "").upper()
        return upper.startswith("RA") or "GLON" in upper or "LONG" in upper

    @staticmethod
    def _is_latitude(axis_type: str) -> bool:
        upper = (axis_type or "").upper()
        return any(token in upper for token in ("DEC", "GLAT", "LAT"))

    # ------------------------------------------------------------------
    # Progress API
    def on_regrid_started(self):
        self._regrid_running = True
        if self.run_button:
            self.run_button.setEnabled(False)
        if self.progress_bar:
            self.progress_bar.setFixedHeight(max(8, self.progress_bar.sizeHint().height()))
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setFormat("Starting regrid...")
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(True)

    def on_regrid_finished(self, success: bool = True):
        self._regrid_running = False
        if self.run_button:
            self.run_button.setEnabled(True)
        if self.progress_bar:
            self.progress_bar.setVisible(False)
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setFormat("%p%")
            self.progress_bar.setFixedHeight(0)
            if not success:
                self.progress_bar.setValue(0)

    def update_progress(self, value: int):
        if self.progress_bar:
            if not self.progress_bar.isVisible():
                self.progress_bar.setFixedHeight(max(8, self.progress_bar.sizeHint().height()))
            if self.progress_bar.maximum() == 0:
                self.progress_bar.setRange(0, 100)
            if value < 30:
                self.progress_bar.setFormat("%p% Preparing grid")
            elif value < 90:
                self.progress_bar.setFormat("%p% Processing planes")
            else:
                self.progress_bar.setFormat("%p% Finalizing")
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(value)

    # ------------------------------------------------------------------
    # Workspace persistence (settings only; never auto-opens the panel).
    def export_workspace_state(self) -> dict:
        """Return restorable panel settings.

        Only dataset-independent choices are persisted (interpolation, active
        tab, template path, target system).  The manual-grid per-axis values are
        intentionally omitted: they are derived from the loaded cube's WCS and
        must be re-derived for each dataset, so restoring them across cubes could
        be invalid.
        """
        state: Dict = {"schema": 1}
        try:
            state["tab_index"] = int(self.tab_widget.currentIndex())
        except Exception:
            pass
        if self.interpolation_combo is not None:
            state["interpolation"] = self.interpolation_combo.currentText()
        if self.template_path_edit is not None:
            state["template_path"] = self.template_path_edit.text().strip()
        if self.target_system_combo is not None and self.target_system_combo.count():
            data = self.target_system_combo.currentData()
            state["target_system"] = data if data is not None else self.target_system_combo.currentText()
        return state

    def restore_workspace_state(self, state) -> bool:
        """Apply previously saved settings.

        Every field is guarded, so a workspace captured on a different dataset
        can never leave the panel in an invalid state.
        """
        if not isinstance(state, dict):
            return False

        interpolation = state.get("interpolation")
        if interpolation and self.interpolation_combo is not None:
            idx = self.interpolation_combo.findText(str(interpolation))
            if idx >= 0:
                self.interpolation_combo.setCurrentIndex(idx)

        template_path = state.get("template_path")
        if template_path and self.template_path_edit is not None:
            self.template_path_edit.setText(str(template_path))

        target_system = state.get("target_system")
        if target_system is not None and self.target_system_combo is not None:
            idx = self.target_system_combo.findData(target_system)
            if idx < 0:
                idx = self.target_system_combo.findText(str(target_system))
            if idx >= 0:
                self.target_system_combo.setCurrentIndex(idx)

        tab_index = state.get("tab_index")
        if tab_index is not None:
            try:
                tab_index = int(tab_index)
            except (TypeError, ValueError):
                tab_index = None
            if (
                tab_index is not None
                and 0 <= tab_index < self.tab_widget.count()
                and self.tab_widget.isTabEnabled(tab_index)
            ):
                self.tab_widget.setCurrentIndex(tab_index)
        return True

    # ------------------------------------------------------------------
    def closeEvent(self, event):
        if self._regrid_running:
            choice = confirm_pending_close(
                self,
                "Close Regrid Panel",
                "A regrid job is running in the background.",
                keep_label="Close and Keep Running",
                discard_label=None,
            )
            if choice == "cancel":
                event.ignore()
                return
        self.closed.emit()
        super().closeEvent(event)
