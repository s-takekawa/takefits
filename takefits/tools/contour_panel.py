import copy
import math
import os
from typing import List, Optional, Tuple, Set

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QDoubleSpinBox,
    QAbstractItemView,
    QSpinBox,
    QTextEdit,
    QSizePolicy,
)

from takefits.core.contour_manager import (
    ContourManager,
    ContourParameters,
    serialize_state_to_json,
    deserialize_state_from_json,
)
from takefits.core.usecases_contour import compute_contours, clear_contours

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
        self._suspend_target_callbacks = False

        self._build_ui()
        self._apply_saved_parameters()
        self.refresh_targets()
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
        self.btn_check_all.setFixedWidth(50)
        self.btn_check_none = QPushButton("None", self)
        self.btn_check_none.setFixedWidth(60)
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
        self.level_display.setReadOnly(True)
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
        self.btn_execute.setFixedWidth(140)
        self.btn_clear.setFixedWidth(80)
        self.btn_execute.setDefault(True)
        self.btn_execute.setAutoDefault(True)
        button_layout.addWidget(self.btn_execute)
        button_layout.addWidget(self.btn_clear)
        self.btn_load = QPushButton("Load", self)
        self.btn_save = QPushButton("Save", self)
        button_layout.addStretch()
        button_layout.addWidget(self.btn_load)
        button_layout.addWidget(self.btn_save)
        main_layout.addLayout(button_layout)

        # Signal wiring
        self.btn_execute.clicked.connect(self.execute_contours)
        self.btn_clear.clicked.connect(self.clear_contours)
        self.btn_load.clicked.connect(self.load_contours)
        self.btn_save.clicked.connect(self.save_contours)
        self.btn_check_all.clicked.connect(self._check_all_targets)
        self.btn_check_none.clicked.connect(self._uncheck_all_targets)
        for line_edit in (self.entry_min, self.entry_max, self.entry_step):
            line_edit.returnPressed.connect(self.execute_contours)
        self.entry_step.textEdited.connect(self._mark_step_manual)
        self.entry_step.editingFinished.connect(self._update_level_display)
        self.spin_levels.valueChanged.connect(self._handle_levels_changed)
        self.entry_min.editingFinished.connect(self._on_bounds_editing_finished)
        self.entry_max.editingFinished.connect(self._on_bounds_editing_finished)


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
        try:
            level_min = self._parse_optional_float(self.entry_min.text())
            level_max = self._parse_optional_float(self.entry_max.text())
            level_step = self._parse_optional_float(self.entry_step.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter numeric values for min/max/interval.")
            return None

        if level_step is not None:
            self._step_auto = False

        if level_step is None:
            step_value = self._ensure_step(warn_on_failure=True)
            if step_value is None:
                return None
            level_step = step_value

        params = ContourParameters(
            level_min=level_min,
            level_max=level_max,
            level_step=level_step,
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

    def _handle_levels_changed(self, _value: int) -> None:
        self._step_auto = True
        self._ensure_step()

    def _on_bounds_editing_finished(self) -> None:
        if self._step_auto or not self.entry_step.text().strip():
            self._ensure_step()
        else:
            self._update_level_display()

    def _on_target_item_changed(self, item: QListWidgetItem) -> None:
        if self._suspend_target_callbacks:
            return
        if item is None:
            return
        self._ensure_default_bounds()
        if self._step_auto or not self.entry_step.text().strip():
            self._ensure_step()
        else:
            self._update_level_display()

    def _ensure_default_bounds(self, force: bool = False) -> None:
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

    def _ensure_step(self, warn_on_failure: bool = False) -> Optional[float]:
        min_val, max_val = self._current_min_max()
        if min_val is None or max_val is None:
            bounds = self._determine_bounds()
            if bounds is None:
                if warn_on_failure:
                    QMessageBox.warning(self, "Levels", "Could not determine data range for step calculation.")
                return None
            min_val, max_val = bounds
            if not self.entry_min.text().strip():
                self.entry_min.setText(f"{min_val:g}")
            if not self.entry_max.text().strip():
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
        levels = self._preview_levels()
        if not levels:
            self.level_display.setPlainText("-")
            return
        max_items = 200
        display_levels = levels[:max_items]
        text = "\n".join(f"{value:g}" for value in display_levels)
        if len(levels) > max_items:
            text += "\n…"
        self.level_display.setPlainText(text)


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

        style_color = self.combo_color.currentText()
        style_linewidth = float(self.spin_linewidth.value())
        overlay_label = os.path.basename(path) or (state.label or "overlay")

        for target_id in layer_ids:
            # For each target, create a fresh, deep copy of the state from the file.
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
                if target_state.parameters:
                    target_state.parameters.color = style_color
                    target_state.parameters.linewidth = style_linewidth
                # Clear segment colors to use parameters color
                for item_state in target_state.items:
                    for segment in item_state.segments:
                        segment.color = None

            # Import the prepared state.
            self.manager.import_layer_state(target_id, target_state)

        self.refresh_targets()

    def closeEvent(self, event):
        pass
        super().closeEvent(event)
