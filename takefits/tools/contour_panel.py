import copy
import math
import os
from typing import List, Optional, Tuple, Set

import numpy as np
from PySide6.QtCore import Qt, QTimer, QSettings
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QDoubleSpinBox,
    QAbstractItemView,
    QSpinBox,
    QTextEdit,
    QSizePolicy,
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from takefits.core.contour_manager import (
    ContourManager,
    ContourParameters,
    serialize_state_to_json,
    deserialize_state_from_json,
)
from takefits.ui.widget_sizing import fit_button_to_text
from takefits.core.usecases_contour import compute_contours, clear_contours
from takefits.core.contour_external import (
    ExternalContourError,
    build_contour_state_from_app_state,
    channel_world_value,
    default_levels,
    describe_source,
    estimate_rms,
    sigma_levels,
    smooth_plane,
)
from takefits.app_paths import app_config_path

_RECENT_EXTERNAL_FITS_KEY = "contours/recent_external_fits"
_RECENT_EXTERNAL_FITS_MAX = 8

# The DS9 helpers are implemented in core.contour_manager during Step 4.
from takefits.core.contour_manager import write_state_to_ds9, read_state_from_ds9


class ContourPanel(QDialog):
    COLOR_OPTIONS = [
        "original",
        "white",
        "black",
        "gray",
        "red",
        "orange",
        "yellow",
        "yellowgreen",
        "green",
        "cyan",
        "blue",
        "magenta",
        "purple",
        "rainbow",
    ]

    def __init__(self, parent=None, default_targets: Optional[List[str]] = None):
        super().__init__(parent)
        self.setWindowTitle("Contours")
        self.setModal(False)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)

        self.manager = ContourManager.instance()
        self.manager.targets_changed.connect(self.refresh_targets)
        self.manager.contour_updated.connect(self._on_contour_updated)

        self._default_target_ids = list(default_targets or [])
        self._default_selection_applied = False
        self._step_auto = False
        self._custom_levels = False
        self._updating_level_display = False
        self._suspend_target_callbacks = False

        self._build_ui()
        self._apply_saved_parameters()
        self.refresh_targets()
        self.setAcceptDrops(True)
        self.adjustSize()
        self.resize(self.minimumSizeHint().width(), self.height())


    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(8)
        #main_layout.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)

        metrics = self.fontMetrics()
        char_width = metrics.horizontalAdvance("0")
        entry_width = max(70, char_width * 5 + 12)
        spin_width = max(60, char_width * 4 + 12)

        # Row 1: Level settings
        level_layout = QHBoxLayout()
        level_layout.setSpacing(6)

        level_layout.addWidget(QLabel("Min", self))
        self.entry_min = QLineEdit(self)
        self.entry_min.setPlaceholderText("auto")
        self.entry_min.setFixedWidth(entry_width)
        level_layout.addWidget(self.entry_min)

        level_layout.addWidget(QLabel("Max", self))
        self.entry_max = QLineEdit(self)
        self.entry_max.setPlaceholderText("auto")
        self.entry_max.setFixedWidth(entry_width)
        level_layout.addWidget(self.entry_max)

        level_layout.addWidget(QLabel("Interval", self))
        self.entry_step = QLineEdit(self)
        self.entry_step.setPlaceholderText("auto")
        self.entry_step.setFixedWidth(entry_width)
        level_layout.addWidget(self.entry_step)

        level_layout.addWidget(QLabel("Steps", self))
        self.spin_levels = QSpinBox(self)
        self.spin_levels.setRange(1, 30)
        self.spin_levels.setValue(5)
        self.spin_levels.setFixedWidth(max(70, char_width * 4 + 12))
        level_layout.addWidget(self.spin_levels)
        level_layout.addStretch()
        main_layout.addLayout(level_layout)

        # Row 2: Appearance settings
        appearance_layout = QHBoxLayout()
        appearance_layout.setSpacing(6)

        appearance_layout.addWidget(QLabel("Smoothness", self))
        self.spin_smooth = QDoubleSpinBox(self)
        self.spin_smooth.setRange(0.0, 20.0)
        self.spin_smooth.setSingleStep(0.5)
        self.spin_smooth.setDecimals(1)
        self.spin_smooth.setValue(0.0)
        self.spin_smooth.setFixedWidth(spin_width)
        appearance_layout.addWidget(self.spin_smooth)

        appearance_layout.addWidget(QLabel("Linewidth", self))
        self.spin_linewidth = QDoubleSpinBox(self)
        self.spin_linewidth.setRange(0.1, 20.0)
        self.spin_linewidth.setSingleStep(0.5)
        self.spin_linewidth.setDecimals(1)
        self.spin_linewidth.setValue(1.0)
        self.spin_linewidth.setFixedWidth(spin_width)
        appearance_layout.addWidget(self.spin_linewidth)

        appearance_layout.addWidget(QLabel("Color", self))
        self.combo_color = QComboBox(self)
        self.combo_color.addItems(self.COLOR_OPTIONS)
        self.combo_color.setEditable(False)
        self.combo_color.setCurrentText("white")
        appearance_layout.addWidget(self.combo_color)
        appearance_layout.addStretch()
        main_layout.addLayout(appearance_layout)

        # Row 3: Target list
        targets_layout = QHBoxLayout()
        targets_layout.setSpacing(8)
        targets_layout.setStretch(0, 1)
        targets_layout.setStretch(1, 1)

        targets_container = QVBoxLayout()
        targets_container.setSpacing(4)
        targets_header_layout = QHBoxLayout()
        targets_header_layout.setSpacing(4)
        targets_header_layout.addWidget(QLabel("Targets", self))
        self.btn_check_all = QPushButton("All", self)
        fit_button_to_text(self.btn_check_all, minimum_width=50)
        self.btn_check_none = QPushButton("None", self)
        fit_button_to_text(self.btn_check_none, minimum_width=60)
        targets_header_layout.addWidget(self.btn_check_all)
        targets_header_layout.addWidget(self.btn_check_none)
        targets_header_layout.addStretch()
        targets_container.addLayout(targets_header_layout)

        self.target_list = QListWidget(self)
        self.target_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.target_list.setFixedHeight(120)
        self.target_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.target_list.itemChanged.connect(self._on_target_item_changed)
        targets_container.addWidget(self.target_list)

        targets_layout.addLayout(targets_container)

        values_container = QVBoxLayout()
        values_container.setSpacing(4)
        values_container.addWidget(QLabel("Values", self))
        self.level_display = QTextEdit(self)
        self.level_display.setReadOnly(False)
        self.level_display.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.level_display.setFixedHeight(120)
        self.level_display.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        values_container.addWidget(self.level_display)

        targets_layout.addLayout(values_container)
        main_layout.addLayout(targets_layout)

        # Row 4: Execute/Clear/Load/Save buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(8)
        self.btn_execute = QPushButton("Apply", self)
        self.btn_clear = QPushButton("Clear", self)
        fit_button_to_text(self.btn_execute, minimum_width=140)
        fit_button_to_text(self.btn_clear, minimum_width=80)
        self.btn_execute.setDefault(True)
        self.btn_execute.setAutoDefault(True)
        button_layout.addWidget(self.btn_execute)
        button_layout.addWidget(self.btn_clear)
        self.btn_load = QPushButton("Load", self)
        self.btn_load_fits = QPushButton("From FITS", self)
        self._from_fits_menu = QMenu(self.btn_load_fits)
        self._from_fits_menu.aboutToShow.connect(self._populate_from_fits_menu)
        self.btn_load_fits.setMenu(self._from_fits_menu)
        self.btn_save = QPushButton("Save", self)
        button_layout.addStretch()
        button_layout.addWidget(self.btn_load)
        button_layout.addWidget(self.btn_load_fits)
        button_layout.addWidget(self.btn_save)
        main_layout.addLayout(button_layout)

        # Signal wiring
        self.btn_execute.clicked.connect(self.execute_contours)
        self.btn_clear.clicked.connect(self.clear_contours)
        self.btn_load.clicked.connect(self.load_contours)
        self.btn_save.clicked.connect(self.save_contours)
        self.target_list.itemDoubleClicked.connect(self._on_target_double_clicked)
        self.btn_check_all.clicked.connect(self._check_all_targets)
        self.btn_check_none.clicked.connect(self._uncheck_all_targets)
        for line_edit in (self.entry_min, self.entry_max, self.entry_step):
            line_edit.returnPressed.connect(self.execute_contours)
        self.entry_step.textEdited.connect(self._on_step_text_edited)
        self.entry_step.editingFinished.connect(self._update_level_display)
        self.spin_levels.valueChanged.connect(self._handle_levels_changed)
        self.entry_min.textEdited.connect(self._on_bounds_text_edited)
        self.entry_max.textEdited.connect(self._on_bounds_text_edited)
        self.entry_min.editingFinished.connect(self._on_bounds_editing_finished)
        self.entry_max.editingFinished.connect(self._on_bounds_editing_finished)
        self.level_display.textChanged.connect(self._on_level_display_text_changed)


    def _apply_saved_parameters(self) -> None:
        params = self.manager.get_parameters()
        self.entry_min.setText("" if params.level_min is None else f"{params.level_min:g}")
        self.entry_max.setText("" if params.level_max is None else f"{params.level_max:g}")
        self.entry_step.setText("" if params.level_step is None else f"{params.level_step:g}")
        self.spin_smooth.setValue(float(params.smoothing))
        self.spin_linewidth.setValue(float(params.linewidth))
        if params.color in self.COLOR_OPTIONS:
            self.combo_color.setCurrentText(params.color)
        self._ensure_default_bounds(force=True)
        self._ensure_step()
        if params.levels is not None:
            self._custom_levels = True
            self._set_level_display_text("\n".join(f"{float(value):g}" for value in params.levels))
        else:
            self._update_level_display()

    def _selected_entries(self) -> List[dict]:
        entries: List[dict] = []
        for idx in range(self.target_list.count()):
            item = self.target_list.item(idx)
            if item.checkState() != Qt.CheckState.Checked:
                continue
            data = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(data, dict):
                entries.append(data)
        return entries

    def _selected_layer_ids(self) -> List[str]:
        return [entry["id"] for entry in self._selected_entries() if entry.get("kind") == "layer"]

    def _selected_overlay_ids(self) -> List[str]:
        return [entry["id"] for entry in self._selected_entries() if entry.get("kind") == "overlay"]

    def _set_all_checks(self, checked: bool) -> None:
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        self._suspend_target_callbacks = True
        try:
            for idx in range(self.target_list.count()):
                item = self.target_list.item(idx)
                item.setCheckState(state)
        finally:
            self._suspend_target_callbacks = False
        self._ensure_default_bounds()
        self._update_level_display()

    def _check_all_targets(self) -> None:
        self._set_all_checks(True)

    def _uncheck_all_targets(self) -> None:
        self._set_all_checks(False)

    def _collect_parameters(self) -> Optional[ContourParameters]:
        custom_levels: Optional[List[float]] = None
        if self._custom_levels:
            try:
                custom_levels = self._parse_level_values(self.level_display.toPlainText())
            except ValueError as exc:
                QMessageBox.warning(self, "Invalid Input", str(exc))
                return None
            if not custom_levels:
                QMessageBox.warning(self, "Invalid Input", "Please enter at least one contour value.")
                return None

        try:
            level_min = self._parse_optional_float(self.entry_min.text())
            level_max = self._parse_optional_float(self.entry_max.text())
            level_step = self._parse_optional_float(self.entry_step.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter numeric values for min/max/interval.")
            return None

        if level_step is not None:
            self._step_auto = False

        if custom_levels is None and level_step is None:
            step_value = self._ensure_step(warn_on_failure=True)
            if step_value is None:
                return None
            level_step = step_value

        params = ContourParameters(
            level_min=level_min,
            level_max=level_max,
            level_step=level_step,
            levels=custom_levels,
            smoothing=float(self.spin_smooth.value()),
            linewidth=float(self.spin_linewidth.value()),
            color=self.combo_color.currentText(),
        )
        return params

    @staticmethod
    def _parse_optional_float(value: str) -> Optional[float]:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            return None
        return float(stripped)

    def _current_min_max(self) -> Tuple[Optional[float], Optional[float]]:
        try:
            min_val = self._parse_optional_float(self.entry_min.text())
        except ValueError:
            min_val = None
        try:
            max_val = self._parse_optional_float(self.entry_max.text())
        except ValueError:
            max_val = None
        return min_val, max_val

    def _determine_bounds(self) -> Optional[Tuple[float, float]]:
        candidate_ids = self._selected_layer_ids()
        if not candidate_ids:
            candidate_ids = list(self._default_target_ids) or list(
                self.manager.registered_layers().keys()
            )
        for layer_id in candidate_ids:
            bounds = self.manager.layer_data_bounds(layer_id)
            if bounds is not None:
                return bounds
        return None

    def _mark_step_manual(self, *_args) -> None:
        self._step_auto = False

    def _mark_generated_levels(self) -> None:
        self._custom_levels = False

    def _on_step_text_edited(self, *_args) -> None:
        self._mark_generated_levels()
        self._mark_step_manual()
        self._update_level_display()

    def _on_bounds_text_edited(self, *_args) -> None:
        self._mark_generated_levels()
        if self._step_auto or not self.entry_step.text().strip():
            if self._ensure_step(fill_bounds=False) is None:
                self._update_level_display()
        else:
            self._update_level_display()

    def _handle_levels_changed(self, _value: int) -> None:
        self._mark_generated_levels()
        self._step_auto = True
        self._ensure_step(fill_bounds=False)

    def _on_bounds_editing_finished(self) -> None:
        if self._step_auto or not self.entry_step.text().strip():
            self._ensure_step(fill_bounds=False)
        else:
            self._update_level_display()

    def _on_target_item_changed(self, item: QListWidgetItem) -> None:
        if self._suspend_target_callbacks:
            return
        if item is None:
            return
        self._ensure_default_bounds()
        if self._step_auto or not self.entry_step.text().strip():
            self._ensure_step(fill_bounds=False)
        else:
            self._update_level_display()

    def _ensure_default_bounds(self, force: bool = False) -> None:
        if not force:
            return
        need_min = force or not self.entry_min.text().strip()
        need_max = force or not self.entry_max.text().strip()
        if not need_min and not need_max:
            return
        bounds = self._determine_bounds()
        if bounds is None:
            return
        min_val, max_val = bounds
        changed = False
        if need_min:
            self.entry_min.setText(f"{min_val:g}")
            changed = True
        if need_max:
            self.entry_max.setText(f"{max_val:g}")
            changed = True
        if changed:
            if force:
                self._step_auto = True
            if self._step_auto or not self.entry_step.text().strip():
                self._ensure_step()
            else:
                self._update_level_display()

    def _ensure_step(
        self,
        warn_on_failure: bool = False,
        fill_bounds: bool = True,
    ) -> Optional[float]:
        min_val, max_val = self._current_min_max()
        if min_val is None or max_val is None:
            bounds = self._determine_bounds()
            if bounds is None:
                if warn_on_failure:
                    QMessageBox.warning(self, "Levels", "Could not determine data range for step calculation.")
                return None
            min_val, max_val = bounds
            if fill_bounds and not self.entry_min.text().strip():
                self.entry_min.setText(f"{min_val:g}")
            if fill_bounds and not self.entry_max.text().strip():
                self.entry_max.setText(f"{max_val:g}")

        if min_val is None or max_val is None:
            if warn_on_failure:
                QMessageBox.warning(self, "Levels", "Please provide valid min and max values.")
            return None

        diff = max_val - min_val
        if diff <= 0:
            if warn_on_failure:
                QMessageBox.warning(self, "Levels", "Max must be greater than min to compute steps.")
            return None

        levels = self.spin_levels.value()
        if levels <= 0:
            levels = 1
        step = diff / levels
        self.entry_step.setText(f"{step:g}")
        self._step_auto = True
        self._update_level_display()
        return step

    def _preview_levels(self) -> List[float]:
        try:
            level_min = self._parse_optional_float(self.entry_min.text())
        except ValueError:
            return []
        try:
            level_max = self._parse_optional_float(self.entry_max.text())
        except ValueError:
            return []
        try:
            step_value = self._parse_optional_float(self.entry_step.text())
        except ValueError:
            return []

        bounds = self._determine_bounds()
        default_min = bounds[0] if bounds is not None else None
        default_max = bounds[1] if bounds is not None else None

        if level_min is None:
            level_min = default_min
        if level_max is None:
            level_max = default_max
        if level_min is None or level_max is None:
            return []

        vmin, vmax = level_min, level_max
        if vmin > vmax:
            vmin, vmax = vmax, vmin

        if not math.isfinite(vmin) or not math.isfinite(vmax):
            return []

        if math.isclose(vmin, vmax):
            return [vmin]

        diff = vmax - vmin
        if diff <= 0:
            return []

        if step_value is None or step_value == 0:
            levels_count = max(self.spin_levels.value(), 1)
            step_value = diff / levels_count if levels_count else None
        if step_value is None:
            return []

        step = abs(step_value)
        if not math.isfinite(step) or step == 0:
            return []

        count = int(math.floor(diff / step)) + 1
        if count < 1:
            count = 1
        if count > 200:
            count = 200
        levels = [vmin + i * step for i in range(count)]
        ordered: List[float] = []
        for val in levels:
            if not ordered or not math.isclose(ordered[-1], val):
                ordered.append(val)
        return ordered

    def _update_level_display(self) -> None:
        if self._custom_levels:
            return
        levels = self._preview_levels()
        if not levels:
            self._set_level_display_text("-")
            return
        max_items = 200
        display_levels = levels[:max_items]
        text = "\n".join(f"{value:g}" for value in display_levels)
        if len(levels) > max_items:
            text += "\n…"
        self._set_level_display_text(text)

    def _set_level_display_text(self, text: str) -> None:
        self._updating_level_display = True
        try:
            self.level_display.setPlainText(text)
        finally:
            self._updating_level_display = False

    def _on_level_display_text_changed(self) -> None:
        if self._updating_level_display:
            return
        self._custom_levels = True

    @staticmethod
    def _parse_level_values(text: str) -> List[float]:
        raw = (text or "").replace(",", " ").split()
        levels: List[float] = []
        for token in raw:
            if token == "-":
                continue
            try:
                value = float(token)
            except ValueError as exc:
                raise ValueError("Please enter contour values as numbers separated by spaces, commas, or new lines.") from exc
            if not math.isfinite(value):
                raise ValueError("Contour values must be finite numbers.")
            levels.append(value)
        if len(levels) > 200:
            raise ValueError("Please enter 200 or fewer contour values.")
        ordered: List[float] = []
        for value in sorted(levels):
            if not ordered or not math.isclose(ordered[-1], value):
                ordered.append(value)
        return ordered


    # Target list management
    def refresh_targets(self) -> None:
        previously_checked: Set[Tuple[str, str]] = set()
        explicit_unchecked_overlays: Set[str] = set()
        for idx in range(self.target_list.count()):
            item = self.target_list.item(idx)
            data = item.data(Qt.ItemDataRole.UserRole)
            if item.checkState() == Qt.CheckState.Checked and isinstance(data, dict):
                kind = data.get("kind")
                entry_id = data.get("id")
                if kind and entry_id:
                    previously_checked.add((kind, entry_id))
            elif item.checkState() == Qt.CheckState.Unchecked and isinstance(data, dict):
                if data.get("kind") == "overlay":
                    entry_id = data.get("id")
                    if entry_id:
                        explicit_unchecked_overlays.add(entry_id)

        if (
            not previously_checked
            and not self._default_selection_applied
            and self._default_target_ids
        ):
            for layer_id in self._default_target_ids:
                previously_checked.add(("layer", layer_id))
            self._default_selection_applied = True

        self.target_list.clear()

        entries = self.manager.entries()

        self._suspend_target_callbacks = True
        try:
            for entry in entries:
                text = entry.label or entry.id
                if entry.kind == "layer":
                    plane = entry.plane.upper() if entry.plane else ""
                    if plane and f"[{plane}]" not in text.upper():
                        text += f" [{plane}]"
                    if entry.active:
                        text += "  • active"
                else:
                    text = f"  ↳ {text}"

                item = QListWidgetItem(text, self.target_list)
                flags = item.flags() | Qt.ItemFlag.ItemIsUserCheckable
                item.setFlags(flags)
                data = {"kind": entry.kind, "id": entry.id}
                if entry.parent_id:
                    data["parent"] = entry.parent_id
                item.setData(Qt.ItemDataRole.UserRole, data)

                is_checked = (entry.kind, entry.id) in previously_checked
                if (
                    not is_checked
                    and entry.kind == "overlay"
                    and entry.parent_id is not None
                    and entry.id not in explicit_unchecked_overlays
                ):
                    is_checked = ("layer", entry.parent_id) in previously_checked
                item.setCheckState(
                    Qt.CheckState.Checked if is_checked else Qt.CheckState.Unchecked
                )
        finally:
            self._suspend_target_callbacks = False

        self._ensure_default_bounds()
        self._update_level_display()

    def _on_contour_updated(self, _layer_id: str) -> None:
        # Refresh labels to account for active state changes.
        self.refresh_targets()


    def execute_contours(self) -> None:
        layer_ids = self._selected_layer_ids()
        overlay_ids = self._selected_overlay_ids()
        if not layer_ids and not overlay_ids:
            QMessageBox.information(self, "Contours", "Please select at least one target.")
            return

        params = self._collect_parameters()
        if params is None:
            return

        compute_contours(layer_ids, params, overlay_ids)
        self.refresh_targets()

    def clear_contours(self) -> None:
        layer_ids = self._selected_layer_ids()
        overlay_ids = self._selected_overlay_ids()
        if not layer_ids and not overlay_ids:
            QMessageBox.information(self, "Contours", "Please select at least one target to clear.")
            return
        clear_contours(layer_ids, overlay_ids)
        self.refresh_targets()

    def save_contours(self) -> None:
        layer_ids = self._selected_layer_ids()
        overlay_ids = self._selected_overlay_ids()
        total = len(layer_ids) + len(overlay_ids)
        if total == 0:
            QMessageBox.information(self, "Save Contours", "Select a target to save contours from.")
            return
        if total > 1:
            QMessageBox.information(self, "Save Contours", "Select only one target when saving contours.")
            return

        exporting_overlay = bool(overlay_ids)
        layers_map = self.manager.registered_layers()
        default_basename: Optional[str] = None
        if exporting_overlay:
            overlay_id = overlay_ids[0]
            if overlay_id.startswith("channel-"):
                QMessageBox.warning(self, "Save Error", "Saving contours from a channel map is not supported.")
                return
            state = self.manager.export_overlay_state(overlay_id)
            if state is None:
                QMessageBox.information(self, "Save Contours", "No contour data available for the selected target.")
                return
            default_label = state.label or "overlay"
            default_basename = self.manager.default_save_basename(overlay_id=overlay_id)
        else:
            layer_id = layer_ids[0]
            if layer_id.startswith("channel-"):
                QMessageBox.warning(self, "Save Error", "Saving contours from a channel map is not supported.")
                return
            state = self.manager.export_layer_state(layer_id)
            if state is None:
                QMessageBox.information(self, "Save Contours", "No contour data available for the selected target.")
                return
            layer = layers_map.get(layer_id)
            default_label = layer.label if layer is not None and layer.label else "contour"
            default_basename = self.manager.default_save_basename(layer_id=layer_id)

        default_dir = os.getcwd()
        if default_basename:
            default_filename = f"{default_basename}.tctr"
        else:
            base_label = default_label.split('[')[0].strip()
            filename_base, _ = os.path.splitext(base_label)
            base_component = filename_base or base_label or "contour"
            default_filename = f"{base_component}.tctr"
        default_path = os.path.join(default_dir, default_filename)

        options = "Contours (*.tctr *.ctr *.con *.txt);;TakeFITS Contour (*.tctr);;DS9 Contour (*.ctr *.con *.txt);;All Files (*)"
        path, selected_filter = QFileDialog.getSaveFileName(self, "Save Contours", default_path, options)
        if not path:
            return

        _, ext = os.path.splitext(path)
        if not ext:
            if selected_filter.startswith("TakeFITS"):
                path = f"{path}.tctr"
            elif selected_filter.startswith("DS9"):
                path = f"{path}.ctr"

        ext_lower = os.path.splitext(path)[1].lower()

        try:
            if selected_filter.startswith("DS9") or ext_lower in {".con", ".ctr", ".txt"}:
                write_state_to_ds9(path, state)
            else:
                # TakeFITS Contour (.tctr) or any other extension uses JSON with color preservation
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write(serialize_state_to_json(state))
        except Exception as exc:
            QMessageBox.critical(self, "Save Contours", f"Failed to save contour data:\n{exc}")
            return

        QMessageBox.information(self, "Save Contours", f"Contours saved to:\n{path}")

    def load_contours(self) -> None:
        layer_ids = self._selected_layer_ids()
        if not layer_ids:
            QMessageBox.information(self, "Load Contours", "Select at least one target to apply contours to.")
            return

        options = "TakeFITS Contour (*.tctr);;DS9 Contour (*.ctr *.con *.txt);;All Files (*)"
        path, selected_filter = QFileDialog.getOpenFileName(self, "Load Contours", os.getcwd(), options)
        if not path:
            return

        try:
            ext_lower = os.path.splitext(path)[1].lower()
            if selected_filter.startswith("DS9") or ext_lower in {".con", ".ctr", ".txt"}:
                state = read_state_from_ds9(path)
            else:
                # TakeFITS Contour (.tctr) or JSON uses JSON deserialization
                with open(path, "r", encoding="utf-8") as handle:
                    state = deserialize_state_from_json(handle.read())
        except Exception as exc:
            QMessageBox.critical(self, "Load Contours", f"Failed to load contour data:\n{exc}")
            return

        # Determine if this is a TakeFITS format with per-segment colors
        is_tctr_format = ext_lower in {".tctr", ".json"}
        has_segment_colors = False
        if is_tctr_format:
            for item in state.items:
                for seg in item.segments:
                    if seg.color is not None:
                        has_segment_colors = True
                        break
                if has_segment_colors:
                    break

        overlay_label = os.path.basename(path) or (state.label or "overlay")
        self._import_state_to_targets(state, layer_ids, overlay_label, has_segment_colors)

    def _import_state_to_targets(
        self,
        state,
        layer_ids: List[str],
        overlay_label: str,
        has_segment_colors: bool,
        apply_panel_style: bool = True,
    ) -> None:
        style_color = self.combo_color.currentText()
        style_linewidth = float(self.spin_linewidth.value())

        for target_id in layer_ids:
            # For each target, create a fresh, deep copy of the source state.
            target_state = copy.deepcopy(state)

            # Customize this copy for the target layer.
            target_state.layer_id = target_id
            target_state.label = overlay_label
            layer = self.manager.registered_layers().get(target_id)
            if layer:
                target_state.plane = layer.plane

            # For .tctr files with per-segment colors, preserve the original colors and linewidth
            if has_segment_colors:
                # Backup per-segment colors to original_color for later restoration
                for item_state in target_state.items:
                    for segment in item_state.segments:
                        if segment.color is not None and segment.original_color is None:
                            segment.original_color = segment.color.copy()
                # Keep original linewidth from file
            else:
                # DS9 or files without per-segment colors: apply UI settings
                if apply_panel_style and target_state.parameters:
                    target_state.parameters.color = style_color
                    target_state.parameters.linewidth = style_linewidth
                # Clear segment colors to use parameters color
                for item_state in target_state.items:
                    for segment in item_state.segments:
                        segment.color = None

            # Import the prepared state.
            self.manager.import_layer_state(target_id, target_state)

        self.refresh_targets()

    # ------------------------------------------------------------------
    # Contours from an external FITS file

    def _recent_external_fits(self) -> List[str]:
        settings = QSettings(app_config_path("takefits.ini"), QSettings.Format.IniFormat)
        paths = settings.value(_RECENT_EXTERNAL_FITS_KEY, [])
        if isinstance(paths, str):
            paths = [paths]
        return [p for p in (paths or []) if isinstance(p, str) and p]

    def _remember_external_fits(self, path: str) -> None:
        path = os.path.abspath(path)
        paths = [p for p in self._recent_external_fits() if p != path]
        paths.insert(0, path)
        settings = QSettings(app_config_path("takefits.ini"), QSettings.Format.IniFormat)
        settings.setValue(_RECENT_EXTERNAL_FITS_KEY, paths[:_RECENT_EXTERNAL_FITS_MAX])
        settings.sync()

    def _populate_from_fits_menu(self) -> None:
        self._from_fits_menu.clear()
        browse_action = self._from_fits_menu.addAction("Browse...")
        browse_action.triggered.connect(lambda: self.load_contours_from_fits())
        recent = [p for p in self._recent_external_fits() if os.path.isfile(p)]
        if recent:
            self._from_fits_menu.addSeparator()
            for path in recent:
                action = self._from_fits_menu.addAction(os.path.basename(path))
                action.setToolTip(path)
                action.triggered.connect(
                    lambda _checked=False, p=path: self.load_contours_from_fits(p)
                )

    def dragEnterEvent(self, event):
        if self._fits_path_from_mime(event.mimeData()):
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dropEvent(self, event):
        path = self._fits_path_from_mime(event.mimeData())
        if path:
            event.acceptProposedAction()
            self.load_contours_from_fits(path)
            return
        super().dropEvent(event)

    @staticmethod
    def _fits_path_from_mime(mime) -> Optional[str]:
        if mime is None or not mime.hasUrls():
            return None
        for url in mime.urls():
            local = url.toLocalFile()
            if local and os.path.splitext(local)[1].lower() in {".fits", ".fit"}:
                return local
        return None

    def load_contours_from_fits(
        self,
        path: Optional[str] = None,
        *,
        replace_overlay_id: Optional[str] = None,
        target_layer_ids: Optional[List[str]] = None,
        initial_settings: Optional[dict] = None,
    ) -> None:
        layer_ids = target_layer_ids if target_layer_ids else self._selected_layer_ids()
        if not layer_ids:
            QMessageBox.information(
                self, "Contours from FITS", "Select at least one target to apply contours to."
            )
            return

        if not path:
            options = "FITS Files (*.fits *.FITS *.fit);;All Files (*)"
            path, _ = QFileDialog.getOpenFileName(
                self, "Contours from FITS", os.getcwd(), options
            )
            if not path:
                return

        try:
            from takefits.core.usecases import load_fits_data

            source_state = load_fits_data(path)
        except Exception as exc:
            QMessageBox.critical(self, "Contours from FITS", f"Failed to load FITS file:\n{exc}")
            return

        dialog = ExternalFitsContourDialog(
            self,
            source_state,
            os.path.basename(path),
            default_color=self.combo_color.currentText(),
            default_linewidth=float(self.spin_linewidth.value()),
            initial_settings=initial_settings,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        levels = dialog.selected_levels()
        if not levels:
            QMessageBox.information(
                self, "Contours from FITS", "No valid contour levels were provided."
            )
            return

        source_meta = {"type": "external_fits", "path": os.path.abspath(path)}
        source_meta.update(dialog.settings())
        try:
            contour_state = build_contour_state_from_app_state(
                source_state,
                levels,
                channel=dialog.selected_channel(),
                color=dialog.selected_color(),
                linewidth=dialog.selected_linewidth(),
                smoothing=dialog.selected_smoothing(),
                label=os.path.basename(path),
                source_meta=source_meta,
            )
        except ExternalContourError as exc:
            QMessageBox.warning(self, "Contours from FITS", str(exc))
            return
        except Exception as exc:
            QMessageBox.critical(self, "Contours from FITS", f"Failed to build contours:\n{exc}")
            return

        if replace_overlay_id:
            self.manager.clear_overlays([replace_overlay_id])
        self._import_state_to_targets(
            contour_state,
            layer_ids,
            os.path.basename(path),
            has_segment_colors=False,
            apply_panel_style=False,
        )
        self._remember_external_fits(path)

    def _on_target_double_clicked(self, item: QListWidgetItem) -> None:
        data = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(data, dict) or data.get("kind") != "overlay":
            return
        overlay_id = data.get("id")
        parent_id = data.get("parent")
        if not overlay_id or not parent_id:
            return
        state = self.manager.export_overlay_state(overlay_id)
        meta = getattr(state, "source_meta", None) if state is not None else None
        if not isinstance(meta, dict) or meta.get("type") != "external_fits":
            return
        path = meta.get("path")
        if not path or not os.path.isfile(str(path)):
            QMessageBox.warning(
                self,
                "Contours from FITS",
                f"The source FITS file is no longer available:\n{path}",
            )
            return
        self.load_contours_from_fits(
            str(path),
            replace_overlay_id=overlay_id,
            target_layer_ids=[parent_id],
            initial_settings=meta,
        )

    def closeEvent(self, event):
        pass
        super().closeEvent(event)


class ExternalFitsContourDialog(QDialog):
    """Configure contour levels from an external FITS file with a live preview."""

    _PREVIEW_MAX_DIM = 512

    def __init__(
        self,
        parent,
        source_state,
        filename: str,
        *,
        default_color: str = "white",
        default_linewidth: float = 1.0,
        initial_settings: Optional[dict] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Contours from FITS")
        self.setModal(True)
        self._state = source_state
        self._fields_edited = False
        self._plane_cache: dict = {}
        initial = initial_settings if isinstance(initial_settings, dict) else {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        # Header: filename + source metadata
        header_layout = QHBoxLayout()
        header_layout.setSpacing(10)
        name_label = QLabel(self)
        name_font = name_label.font()
        name_font.setBold(True)
        name_label.setFont(name_font)
        name_label.setText(
            name_label.fontMetrics().elidedText(filename, Qt.TextElideMode.ElideMiddle, 200)
        )
        name_label.setToolTip(filename)
        header_layout.addWidget(name_label)
        info = describe_source(source_state)
        info_text = "   ".join(
            value for value in (info.get("shape"), info.get("bunit"), info.get("frame")) if value
        )
        if info_text:
            info_label = QLabel(info_text, self)
            info_label.setStyleSheet("color: palette(mid);")
            header_layout.addWidget(info_label)
        header_layout.addStretch()
        layout.addLayout(header_layout)

        # Channel selector (cubes only)
        self.spin_channel = None
        self.slider_channel = None
        self.channel_world_label = None
        n_channels = int(getattr(source_state, "n_channels", 1) or 1)
        if n_channels > 1:
            channel_layout = QHBoxLayout()
            channel_layout.setSpacing(6)
            channel_layout.addWidget(QLabel("Channel", self))
            self.slider_channel = QSlider(Qt.Orientation.Horizontal, self)
            self.slider_channel.setRange(0, n_channels - 1)
            channel_layout.addWidget(self.slider_channel, 1)
            self.spin_channel = QSpinBox(self)
            self.spin_channel.setRange(0, n_channels - 1)
            channel_layout.addWidget(self.spin_channel)
            self.channel_world_label = QLabel("", self)
            # Fixed width so varying text does not reflow the layout.
            self.channel_world_label.setFixedWidth(88)
            self.channel_world_label.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            channel_layout.addWidget(self.channel_world_label)
            layout.addLayout(channel_layout)
            self.slider_channel.valueChanged.connect(self.spin_channel.setValue)
            self.spin_channel.valueChanged.connect(self.slider_channel.setValue)
            self.spin_channel.valueChanged.connect(self._on_channel_changed)

        # Body: preview on top, level controls below (keeps the dialog narrow)
        body_layout = QVBoxLayout()
        body_layout.setSpacing(8)

        preview_container = QVBoxLayout()
        preview_container.setSpacing(4)
        # Trim the canvas height to the displayed sky aspect of the xy plane
        # (fixed per file), so wide images don't leave large blank bands.
        canvas_height = 280
        try:
            shape = getattr(source_state, "shape", None)
            if shape and len(shape) >= 2 and shape[-1] > 0:
                display_ratio = (shape[-2] * self._preview_aspect()) / shape[-1]
                canvas_height = int(round(280 * min(max(display_ratio, 0.3), 1.0)))
        except Exception:
            canvas_height = 280
        self._fig = Figure(figsize=(2.8, canvas_height / 100.0), dpi=100)
        window_color = self.palette().window().color()
        self._fig.patch.set_facecolor(window_color.getRgbF()[:3])
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setFixedSize(280, canvas_height)
        self._ax = self._fig.add_axes([0.005, 0.005, 0.99, 0.99])
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        self._contour_set = None
        preview_container.addWidget(self._canvas, 0, Qt.AlignmentFlag.AlignHCenter)
        self.range_label = QLabel("", self)
        self.range_label.setStyleSheet("color: palette(mid);")
        # Fixed two-line size so per-channel text changes can never reflow
        # the layout or get clipped after the dialog is shown.
        self.range_label.setFixedSize(
            280, 2 * self.range_label.fontMetrics().lineSpacing() + 4
        )
        self.range_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        preview_container.addWidget(self.range_label, 0, Qt.AlignmentFlag.AlignHCenter)
        body_layout.addLayout(preview_container)

        controls_container = QVBoxLayout()
        controls_container.setSpacing(6)

        # Level inputs, unified like the main contour panel: Min/Max/Steps
        # generate levels into an editable Levels field (the source of truth).
        range_grid = QGridLayout()
        range_grid.setContentsMargins(0, 0, 0, 0)
        range_grid.setSpacing(6)
        range_grid.addWidget(QLabel("Min", self), 0, 0)
        self.entry_min = QLineEdit(self)
        self.entry_min.setFixedWidth(76)
        range_grid.addWidget(self.entry_min, 0, 1)
        range_grid.addWidget(QLabel("Max", self), 0, 2)
        self.entry_max = QLineEdit(self)
        self.entry_max.setFixedWidth(76)
        range_grid.addWidget(self.entry_max, 0, 3)
        range_grid.addWidget(QLabel("Steps", self), 0, 4)
        self.spin_steps = QSpinBox(self)
        self.spin_steps.setRange(1, 30)
        self.spin_steps.setValue(5)
        range_grid.addWidget(self.spin_steps, 0, 5)
        range_grid.setColumnStretch(6, 1)
        controls_container.addLayout(range_grid)

        levels_layout = QHBoxLayout()
        levels_layout.setSpacing(6)
        levels_layout.addWidget(QLabel("Levels", self))
        self.entry_levels = QLineEdit(self)
        self.entry_levels.setPlaceholderText("e.g. 0.1, 0.3, 0.5")
        self.entry_levels.setMinimumWidth(160)
        levels_layout.addWidget(self.entry_levels, 1)
        controls_container.addLayout(levels_layout)

        # Style row lives in the controls column to keep the dialog narrow.
        style_layout = QHBoxLayout()
        style_layout.setSpacing(6)
        style_layout.addWidget(QLabel("Color", self))
        self.combo_color = QComboBox(self)
        self.combo_color.addItems(
            [c for c in ContourPanel.COLOR_OPTIONS if c != "original"]
        )
        if default_color and default_color != "original":
            self.combo_color.setCurrentText(default_color)
        style_layout.addWidget(self.combo_color)
        style_layout.addWidget(QLabel("Width", self))
        self.spin_linewidth = QDoubleSpinBox(self)
        self.spin_linewidth.setRange(0.1, 20.0)
        self.spin_linewidth.setSingleStep(0.5)
        self.spin_linewidth.setDecimals(1)
        self.spin_linewidth.setValue(float(default_linewidth))
        style_layout.addWidget(self.spin_linewidth)
        style_layout.addWidget(QLabel("Smooth", self))
        self.spin_smooth = QDoubleSpinBox(self)
        self.spin_smooth.setRange(0.0, 20.0)
        self.spin_smooth.setSingleStep(0.5)
        self.spin_smooth.setDecimals(1)
        self.spin_smooth.setValue(0.0)
        self.spin_smooth.setToolTip("Gaussian smoothing in source pixels before contouring")
        style_layout.addWidget(self.spin_smooth)
        style_layout.addStretch()
        controls_container.addLayout(style_layout)

        body_layout.addLayout(controls_container)
        layout.addLayout(body_layout)

        # Buttons
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(8)
        bottom_layout.addStretch()
        btn_cancel = QPushButton("Cancel", self)
        self.btn_apply = QPushButton("Apply", self)
        self.btn_apply.setDefault(True)
        btn_cancel.clicked.connect(self.reject)
        self.btn_apply.clicked.connect(self.accept)
        bottom_layout.addWidget(btn_cancel)
        bottom_layout.addWidget(self.btn_apply)
        layout.addLayout(bottom_layout)

        # Debounced preview updates
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(250)
        self._preview_timer.timeout.connect(self._update_preview)

        self.entry_min.textEdited.connect(self._on_range_inputs_edited)
        self.entry_max.textEdited.connect(self._on_range_inputs_edited)
        self.spin_steps.valueChanged.connect(self._on_range_inputs_edited)
        self.entry_levels.textEdited.connect(self._on_levels_edited)
        self.combo_color.currentTextChanged.connect(self._schedule_preview)
        self.spin_smooth.valueChanged.connect(self._schedule_preview)

        self._prefill_defaults()
        self._apply_initial_settings(initial)
        self._refresh_stats()
        self._refresh_image()
        self._update_preview()
        # Lock the width so later text updates can never reflow the layout.
        self.setFixedWidth(self.sizeHint().width())

    # ----- data access -----

    def _current_channel(self) -> Optional[int]:
        if self.spin_channel is None:
            return None
        return int(self.spin_channel.value())

    def _plane(self):
        channel = self._current_channel()
        if channel in self._plane_cache:
            return self._plane_cache[channel]
        try:
            if channel is not None:
                self._state.current_z = channel
            plane = self._state.get_slice_2d('xy')
        except Exception:
            plane = None
        if plane is not None:
            plane = np.asarray(plane, dtype=float)
        self._plane_cache = {channel: plane}
        return plane

    def _preview_array(self):
        plane = self._plane()
        if plane is None or plane.ndim != 2 or plane.size == 0:
            self._preview_stride = 1
            return None
        stride = max(1, int(np.ceil(max(plane.shape) / self._PREVIEW_MAX_DIM)))
        self._preview_stride = stride
        return plane[::stride, ::stride]

    # ----- defaults / prefill -----

    def _prefill_defaults(self) -> None:
        plane = self._preview_array()
        if plane is None:
            return
        levels = default_levels(plane, int(self.spin_steps.value()))
        if levels:
            self.entry_min.setText(f"{levels[0]:.4g}")
            self.entry_max.setText(f"{levels[-1]:.4g}")
            self._regenerate_levels_from_range()

    def _apply_initial_settings(self, meta: dict) -> None:
        if not meta:
            return
        channel = meta.get("channel")
        if channel is not None and self.spin_channel is not None:
            try:
                self.spin_channel.setValue(int(channel))
            except Exception:
                pass
        if meta.get("level_min") is not None:
            self.entry_min.setText(f"{float(meta['level_min']):.4g}")
            self._fields_edited = True
        if meta.get("level_max") is not None:
            self.entry_max.setText(f"{float(meta['level_max']):.4g}")
            self._fields_edited = True
        if meta.get("steps") is not None:
            try:
                self.spin_steps.setValue(int(meta["steps"]))
            except Exception:
                pass
        levels_list = meta.get("levels_list")
        if not levels_list and meta.get("rms") is not None:
            # Legacy sigma-mode metadata: expand to explicit levels.
            try:
                levels_list = sigma_levels(
                    float(meta["rms"]),
                    meta.get("factors") or [],
                    bool(meta.get("negative")),
                )
            except Exception:
                levels_list = None
        if isinstance(levels_list, (list, tuple)) and levels_list:
            self.entry_levels.setText(
                ", ".join(f"{float(v):.4g}" for v in levels_list)
            )
            self._fields_edited = True
        color = meta.get("color")
        if isinstance(color, str) and color:
            self.combo_color.setCurrentText(color)
        if meta.get("linewidth") is not None:
            try:
                self.spin_linewidth.setValue(float(meta["linewidth"]))
            except Exception:
                pass
        if meta.get("smoothing") is not None:
            try:
                self.spin_smooth.setValue(float(meta["smoothing"]))
            except Exception:
                pass

    # ----- event handlers -----

    def _regenerate_levels_from_range(self) -> None:
        vmin = self._parse_float(self.entry_min.text())
        vmax = self._parse_float(self.entry_max.text())
        steps = int(self.spin_steps.value())
        if vmin is None or vmax is None or vmax <= vmin:
            return
        values = [float(v) for v in np.linspace(vmin, vmax, steps)]
        self.entry_levels.setText(", ".join(f"{v:.4g}" for v in values))

    def _on_range_inputs_edited(self, *_args) -> None:
        self._fields_edited = True
        self._regenerate_levels_from_range()
        self._schedule_preview()

    def _on_levels_edited(self, *_args) -> None:
        self._fields_edited = True
        self._schedule_preview()

    def _on_channel_changed(self, *_args) -> None:
        self._plane_cache = {}
        self._refresh_stats()
        if not self._fields_edited:
            self._prefill_defaults()
        self._refresh_image()
        self._schedule_preview()

    def _schedule_preview(self, *_args) -> None:
        self._preview_timer.start()

    # ----- preview rendering -----

    def _refresh_stats(self) -> None:
        plane = self._preview_array()
        if self.channel_world_label is not None:
            channel = self._current_channel()
            world = channel_world_value(self._state, channel) if channel is not None else None
            if world is not None:
                value, unit = world
                self.channel_world_label.setText(f"{value:.4g} {unit}".strip())
            else:
                self.channel_world_label.setText("")
        if plane is None:
            self.range_label.setText("")
            return
        finite = plane[np.isfinite(plane)]
        rms = estimate_rms(plane)
        if finite.size:
            text = f"range: {finite.min():.4g} … {finite.max():.4g}"
            text += f"\nrms ≈ {rms:.4g}" if rms is not None else "\n"
            self.range_label.setText(text)
        else:
            self.range_label.setText("range: (no finite values)\n")

    def _preview_aspect(self) -> float:
        """Sky aspect ratio for the xy plane (y over x pixel scale)."""
        try:
            from astropy.wcs.utils import proj_plane_pixel_scales

            scale_x, scale_y = proj_plane_pixel_scales(self._state.wcs.celestial)
            aspect = float(scale_y / scale_x)
            if np.isfinite(aspect) and aspect > 0:
                return aspect
        except Exception:
            pass
        return 1.0

    def _refresh_image(self) -> None:
        self._ax.clear()
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        self._contour_set = None
        plane = self._preview_array()
        if plane is None:
            self._canvas.draw_idle()
            return
        finite = plane[np.isfinite(plane)]
        if finite.size:
            vmin, vmax = np.percentile(finite, [1.0, 99.5])
            if vmin >= vmax:
                vmin, vmax = float(finite.min()), float(finite.max() or 1.0)
        else:
            vmin, vmax = 0.0, 1.0
        self._ax.imshow(
            plane,
            origin='lower',
            cmap='gray',
            vmin=vmin,
            vmax=vmax,
            aspect=self._preview_aspect(),
        )
        self._ax.set_anchor('C')
        self._canvas.draw_idle()

    def _remove_preview_contours(self) -> None:
        if self._contour_set is None:
            return
        try:
            self._contour_set.remove()
        except Exception:
            for collection in getattr(self._contour_set, "collections", []) or []:
                try:
                    collection.remove()
                except Exception:
                    pass
        self._contour_set = None

    def _update_preview(self) -> None:
        levels = self._compute_levels()
        self._remove_preview_contours()
        plane = self._preview_array()
        if plane is not None and levels:
            smoothing = self.selected_smoothing()
            if smoothing > 0:
                # Scale sigma to the downsampled preview grid.
                plane = smooth_plane(
                    plane, smoothing / max(getattr(self, "_preview_stride", 1), 1)
                )
            color = self.selected_color()
            kwargs = {"levels": levels, "linewidths": 1.0}
            if color == "rainbow":
                kwargs["cmap"] = "rainbow"
            else:
                kwargs["colors"] = color
            try:
                self._contour_set = self._ax.contour(plane, **kwargs)
            except Exception:
                self._contour_set = None
        self._canvas.draw_idle()

    # ----- level computation -----

    @staticmethod
    def _parse_float(text: str) -> Optional[float]:
        try:
            value = float(str(text).strip())
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    def _compute_levels(self) -> List[float]:
        # The Levels field is the source of truth; Min/Max/Steps only
        # regenerate its contents (same model as the main contour panel).
        try:
            return ContourPanel._parse_level_values(self.entry_levels.text())
        except Exception:
            return []

    # ----- results -----

    def selected_channel(self) -> Optional[int]:
        return self._current_channel()

    def selected_levels(self) -> List[float]:
        return self._compute_levels()

    def selected_color(self) -> str:
        return self.combo_color.currentText()

    def selected_linewidth(self) -> float:
        return float(self.spin_linewidth.value())

    def selected_smoothing(self) -> float:
        return float(self.spin_smooth.value())

    def settings(self) -> dict:
        return {
            "channel": self._current_channel(),
            "level_min": self._parse_float(self.entry_min.text()),
            "level_max": self._parse_float(self.entry_max.text()),
            "steps": int(self.spin_steps.value()),
            "levels_list": self._compute_levels(),
            "color": self.selected_color(),
            "linewidth": self.selected_linewidth(),
            "smoothing": self.selected_smoothing(),
        }

    def accept(self) -> None:
        if not self._compute_levels():
            QMessageBox.information(
                self, "Contours from FITS", "No valid contour levels were provided."
            )
            return
        super().accept()
