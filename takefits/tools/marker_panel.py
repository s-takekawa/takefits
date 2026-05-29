from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable

from PySide6.QtCore import Qt, QTimer, Signal as pyqtSignal
from PySide6.QtGui import QKeyEvent, QMouseEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from astropy import units as u
from astropy.wcs.utils import proj_plane_pixel_scales

from takefits.core.marker import (
    Marker,
    MarkerState,
    MarkerStyle,
    LineMarker,
    SymbolMarker,
    TextMarker,
    marker_states_to_ds9,
    marker_states_to_json,
)
from takefits.core.marker_manager import MarkerManager
from takefits.core.wcs_frames import (
    available_display_frames,
    display_axis_type,
    display_frame_label,
    frame_is_available,
    normalize_display_frame,
    plane_axis_indices,
    plane_inputs_to_native,
    plane_values_for_display,
)



class DeselectableListWidget(QListWidget):
    delete_pressed = pyqtSignal()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        super().mousePressEvent(event)
        if self.itemAt(event.pos()) is None:
            self.clearSelection()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        super().keyPressEvent(event)
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.delete_pressed.emit()



@dataclass
class SymbolControls:
    widget: QWidget
    size_spin: QDoubleSpinBox
    linewidth_spin: QDoubleSpinBox
    edge_color_label: QLabel
    edge_color_combo: QComboBox


@dataclass
class LineControls:
    widget: QWidget
    length_spin: QDoubleSpinBox
    unit_combo: QComboBox
    angle_spin: QDoubleSpinBox
    linewidth_spin: QDoubleSpinBox
    style_combo: QComboBox


@dataclass
class TextControls:
    widget: QWidget
    font_size_spin: QDoubleSpinBox
    rotation_spin: QDoubleSpinBox
    font_combo: QComboBox


class MarkerPanel(QDialog):
    """Dialog for managing draw markers, lines, and annotations across planes."""

    COLOR_OPTIONS = [
        "None",
        "blue",
        "red",
        "green",
        "cyan",
        "magenta",
        "black",
        "white",
        "gray",
        "orange",
        "purple",
        "yellow",
        "yellowgreen",
        "olive",
        "lime",
    ]

    UNIT_OPTIONS = ["pixel", "arcsec", "arcmin", "deg"]

    FONT_OPTIONS = [
        "Arial",
        "DejaVu Sans",
        "STIXGeneral",
        "Helvetica",
        "Times New Roman",
        "Courier New",
        "Monaco",
        "Verdana",
    ]

    TYPE_OPTIONS = [
        ("Circle", {"kind": "symbol", "symbol": "o", "glyph": "●"}),
        ("Square", {"kind": "symbol", "symbol": "s", "glyph": "■"}),
        ("Diamond", {"kind": "symbol", "symbol": "D", "glyph": "◆"}),
        ("Triangle Up", {"kind": "symbol", "symbol": "^", "glyph": "▲"}),
        ("Triangle Down", {"kind": "symbol", "symbol": "v", "glyph": "▼"}),
        ("Star", {"kind": "symbol", "symbol": "*", "glyph": "★"}),
        ("Plus", {"kind": "symbol", "symbol": "+", "glyph": "+"}),
        ("Cross", {"kind": "symbol", "symbol": "x", "glyph": "✕"}),
        ("Line", {"kind": "line", "glyph": "―"}),
        ("Text", {"kind": "text"}),
    ]

    def __init__(self, viewer, marker_manager: MarkerManager):
        super().__init__(viewer)
        self.viewer = viewer
        self.marker_manager = marker_manager
        self._current_plane = getattr(viewer, "plane", "xy")
        default_plane_getter = getattr(viewer, "default_marker_plane", None)
        if callable(default_plane_getter):
            try:
                default_plane = default_plane_getter()
            except Exception:
                default_plane = None
            if default_plane:
                self._current_plane = str(default_plane)
        self._current_marker_id: Optional[str] = None
        self._suspend_updates = False
        self._suspend_selection_sync = False
        self._pending_selection_id: Optional[str] = None
        self._selection_from_list = False
        self.property_rows = {"symbol": [], "line": [], "text": []}
        self.link_all_checkbox: Optional[QCheckBox] = None
        self._last_type_kind: Optional[str] = None
        self._last_text_value: str = "Text"
        self._suppress_label_from_text: bool = False
        self.detail_group: Optional[QGroupBox] = None
        self._detail_group_heights: Dict[str, int] = {}
        self._detail_group_fixed_height: Optional[int] = None

        self.setWindowTitle(self._window_title_for_viewer())
        self.setModal(False)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)

        self._build_ui()
        self._connect_signals()
        self._precompute_detail_group_heights()
        self._on_type_changed(self.type_combo.currentIndex())
        self.placement_toggle.setChecked(True)
        self._configure_placement()
        self._refresh_marker_list()

    def _window_title_for_viewer(self) -> str:
        viewer = getattr(self, "viewer", None)
        class_name = str(getattr(viewer, "__class__", type(viewer)).__name__ or "").lower()
        if class_name == "channelmapwindow":
            return "Markers (Channel map)"
        if class_name == "integresultwindow":
            return "Markers (Integ)"
        if class_name in {"mainwindow", "subwindow"}:
            return "Markers (Main)"
        return "Markers"

    # ------------------------------------------------------------------
    # UI construction
    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        type_row = QHBoxLayout()
        type_row.setSpacing(6)
        type_row.addWidget(QLabel("Type", self))

        self.type_combo = QComboBox(self)
        for label, data in self.TYPE_OPTIONS:
            glyph = data.get("glyph")
            if glyph:
                display = f"{label} {glyph}"
            else:
                display = label
            self.type_combo.addItem(display)
        type_row.addWidget(self.type_combo, stretch=1)

        self.placement_toggle = QCheckBox("Placement mode", self)
        type_row.addWidget(self.placement_toggle)
        if self._viewer_supports_link_all():
            self.link_all_checkbox = QCheckBox("All tiles", self)
            self.link_all_checkbox.setToolTip("Place markers on every channel tile")
            self.link_all_checkbox.setChecked(self._viewer_link_all_enabled())
            type_row.addWidget(self.link_all_checkbox)
        type_row.addStretch()
        main_layout.addLayout(type_row)

        detail_group = QGroupBox("Properties", self)
        self.detail_group = detail_group
        self.properties_form = QFormLayout(detail_group)
        self.properties_form.setContentsMargins(8, 8, 8, 8)
        self.properties_form.setVerticalSpacing(4)

        self.pixel_x_spin = self._make_coordinate_spin(detail_group)
        self.pixel_y_spin = self._make_coordinate_spin(detail_group)
        self.world_x_edit = self._make_world_line_edit(detail_group, "World X")
        self.world_y_edit = self._make_world_line_edit(detail_group, "World Y")
        self.world_frame_combo = QComboBox(detail_group)
        self._populate_world_frame_combo()
        self.pixel_x_spin.setFixedWidth(110)
        self.pixel_y_spin.setFixedWidth(110)
        if self.pixel_x_spin.lineEdit() is not None:
            self.pixel_x_spin.lineEdit().returnPressed.connect(self._on_pixel_entry_return_pressed)
        if self.pixel_y_spin.lineEdit() is not None:
            self.pixel_y_spin.lineEdit().returnPressed.connect(self._on_pixel_entry_return_pressed)
        self.world_x_edit.returnPressed.connect(self._on_world_entry_return_pressed)
        self.world_y_edit.returnPressed.connect(self._on_world_entry_return_pressed)

        x_row = QHBoxLayout()
        x_row.setContentsMargins(0, 0, 0, 0)
        x_row.setSpacing(4)
        x_row.addWidget(self.pixel_x_spin)
        x_row.addWidget(self.world_x_edit)
        x_widget = QWidget(detail_group)
        x_widget.setLayout(x_row)

        y_row = QHBoxLayout()
        y_row.setContentsMargins(0, 0, 0, 0)
        y_row.setSpacing(4)
        y_row.addWidget(self.pixel_y_spin)
        y_row.addWidget(self.world_y_edit)
        y_widget = QWidget(detail_group)
        y_widget.setLayout(y_row)

        self.properties_form.addRow("World Frame", self.world_frame_combo)
        self.properties_form.addRow("X (pix)", x_widget)
        self.properties_form.addRow("Y (pix)", y_widget)

        self.label_field_label = QLabel("Label", self)
        self.label_edit = QLineEdit(self)
        self.label_edit.setPlaceholderText("Label")
        self.properties_form.addRow(self.label_field_label, self.label_edit)

        self.color_combo = QComboBox(self)
        self.color_combo.addItems(self.COLOR_OPTIONS)
        if "white" in self.COLOR_OPTIONS:
            self.color_combo.setCurrentText("white")

        self.opacity_spin = QDoubleSpinBox(self)
        self.opacity_spin.setRange(0.05, 1.0)
        self.opacity_spin.setSingleStep(0.05)
        self.opacity_spin.setValue(1.0)

        color_row = QHBoxLayout()
        color_row.setContentsMargins(0, 0, 0, 0)
        color_row.setSpacing(6)
        color_row.addWidget(self.color_combo)
        opacity_label = QLabel("Opacity", self)
        color_row.addWidget(opacity_label)
        color_row.addWidget(self.opacity_spin)
        color_container = QWidget(self)
        color_container.setLayout(color_row)
        self.properties_form.addRow("Color", color_container)

        # Symbol Controls
        self.symbol_controls = self._build_symbol_controls()
        size_label = QLabel("Size", detail_group)
        size_row_widget = QWidget(detail_group)
        size_row_layout = QHBoxLayout(size_row_widget)
        size_row_layout.setContentsMargins(0, 0, 0, 0)
        size_row_layout.setSpacing(6)
        size_row_layout.addWidget(self.symbol_controls.size_spin)
        size_row_layout.addSpacing(8)
        size_row_layout.addWidget(QLabel("Linewidth", size_row_widget))
        size_row_layout.addWidget(self.symbol_controls.linewidth_spin)
        size_row_layout.addStretch(1)
        self.properties_form.addRow(size_label, size_row_widget)
        self.properties_form.addRow(self.symbol_controls.edge_color_label, self.symbol_controls.edge_color_combo)
        self.property_rows["symbol"].extend([(size_label, size_row_widget), (self.symbol_controls.edge_color_label, self.symbol_controls.edge_color_combo)])

        # Line Controls
        self.line_controls = self._build_line_controls()
        length_label = QLabel("Length", detail_group)
        length_row_widget = QWidget(detail_group)
        length_row_layout = QHBoxLayout(length_row_widget)
        length_row_layout.setContentsMargins(0, 0, 0, 0)
        length_row_layout.setSpacing(6)
        length_row_layout.addWidget(self.line_controls.length_spin)
        length_row_layout.addWidget(self.line_controls.unit_combo)
        length_row_layout.addStretch(1)
        self.properties_form.addRow(length_label, length_row_widget)
        angle_label = QLabel("Angle (deg)", detail_group)
        angle_row_widget = QWidget(detail_group)
        angle_row_layout = QHBoxLayout(angle_row_widget)
        angle_row_layout.setContentsMargins(0, 0, 0, 0)
        angle_row_layout.setSpacing(6)
        angle_row_layout.addWidget(self.line_controls.angle_spin)
        angle_row_layout.addSpacing(8)
        angle_row_layout.addWidget(QLabel("Linewidth", angle_row_widget))
        angle_row_layout.addWidget(self.line_controls.linewidth_spin)
        angle_row_layout.addStretch(1)
        self.properties_form.addRow(angle_label, angle_row_widget)
        style_label = QLabel("Style", detail_group)
        self.properties_form.addRow(style_label, self.line_controls.style_combo)
        self.property_rows["line"].extend([(length_label, length_row_widget), (angle_label, angle_row_widget), (style_label, self.line_controls.style_combo)])

        # Text Controls
        self.text_controls = self._build_text_controls()
        font_size_label = QLabel("Font Size", detail_group)
        self.properties_form.addRow(font_size_label, self.text_controls.font_size_spin)
        rotation_label = QLabel("Angle (deg)", detail_group)
        self.properties_form.addRow(rotation_label, self.text_controls.rotation_spin)
        font_label = QLabel("Font", detail_group)
        self.properties_form.addRow(font_label, self.text_controls.font_combo)
        self.property_rows["text"].extend([(font_size_label, self.text_controls.font_size_spin), (rotation_label, self.text_controls.rotation_spin), (font_label, self.text_controls.font_combo)])

        # Set initial visibility
        self._set_property_rows_visible("symbol", "o")

        main_layout.addWidget(detail_group)

        list_group = QGroupBox("Markers", self)
        list_layout = QVBoxLayout(list_group)
        list_layout.setContentsMargins(8, 8, 8, 8)
        list_layout.setSpacing(6)

        self.marker_list = DeselectableListWidget(self)
        self.marker_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        list_layout.addWidget(self.marker_list)

        list_button_row = QHBoxLayout()
        list_button_row.setSpacing(8)
        self.delete_button = QPushButton("Delete", self)
        self.delete_all_button = QPushButton("Delete All", self)
        self.save_button = QPushButton("Save", self)
        self.load_button = QPushButton("Load", self)
        self.delete_button.setDefault(False)
        self.delete_button.setAutoDefault(False)
        self.delete_all_button.setDefault(False)
        self.delete_all_button.setAutoDefault(False)
        self.save_button.setDefault(False)
        self.save_button.setAutoDefault(False)
        self.load_button.setDefault(False)
        self.load_button.setAutoDefault(False)
        list_button_row.addWidget(self.save_button)
        list_button_row.addWidget(self.load_button)
        list_button_row.addWidget(self.delete_button)
        list_button_row.addWidget(self.delete_all_button)
        list_button_row.addStretch()
        list_layout.addLayout(list_button_row)

        main_layout.addWidget(list_group)

    def _build_symbol_controls(self) -> SymbolControls:
        size_spin = QDoubleSpinBox()
        size_spin.setRange(1.0, 100.0)
        size_spin.setSingleStep(1.0)
        size_spin.setValue(8.0)

        linewidth_spin = QDoubleSpinBox()
        linewidth_spin.setRange(0.0, 10.0)
        linewidth_spin.setSingleStep(0.1)
        linewidth_spin.setValue(0.0)

        edge_color_label = QLabel("Edge Color")
        edge_color_combo = QComboBox()
        edge_color_combo.addItems(self.COLOR_OPTIONS)

        return SymbolControls(
            widget=QWidget(),  # Dummy widget, no longer used for layout
            size_spin=size_spin,
            linewidth_spin=linewidth_spin,
            edge_color_label=edge_color_label,
            edge_color_combo=edge_color_combo,
        )

    def _build_line_controls(self) -> LineControls:
        length_spin = QDoubleSpinBox()
        length_spin.setRange(0.1, 10_000.0)
        length_spin.setSingleStep(1.0)
        length_spin.setValue(20.0)

        unit_combo = QComboBox()
        unit_combo.addItems(self.UNIT_OPTIONS)

        angle_spin = QDoubleSpinBox()
        angle_spin.setRange(-360.0, 360.0)
        angle_spin.setSingleStep(1.0)
        angle_spin.setValue(0.0)

        linewidth_spin = QDoubleSpinBox()
        linewidth_spin.setRange(0.1, 20.0)
        linewidth_spin.setSingleStep(0.2)
        linewidth_spin.setValue(1.0)

        style_combo = QComboBox()
        style_combo.addItem("Solid", "solid")
        style_combo.addItem("Dashed", "dashed")
        style_combo.addItem("Dotted", "dotted")
        style_combo.addItem("Arrow →", "arrow")
        style_combo.addItem("Arrow ↔", "double_arrow")
        style_combo.addItem("Scale", "scale")

        return LineControls(
            widget=QWidget(), # Dummy
            length_spin=length_spin,
            unit_combo=unit_combo,
            angle_spin=angle_spin,
            linewidth_spin=linewidth_spin,
            style_combo=style_combo,
        )

    def _build_text_controls(self) -> TextControls:
        font_size_spin = QDoubleSpinBox()
        font_size_spin.setRange(6.0, 72.0)
        font_size_spin.setSingleStep(1.0)
        font_size_spin.setValue(14.0)

        rotation_spin = QDoubleSpinBox()
        rotation_spin.setRange(-360.0, 360.0)
        rotation_spin.setSingleStep(1.0)
        rotation_spin.setValue(0.0)

        font_combo = QComboBox()
        for family in self.FONT_OPTIONS:
            font_combo.addItem(family)
        font_combo.setCurrentText("Arial")

        return TextControls(
            widget=QWidget(), # Dummy
            font_size_spin=font_size_spin,
            rotation_spin=rotation_spin,
            font_combo=font_combo,
        )

    def _set_property_rows_visible(self, kind: str, symbol: Optional[str] = None) -> None:
        for row_kind in self.property_rows:
            for label, widget in self.property_rows[row_kind]:
                is_visible = row_kind == kind
                label.setVisible(is_visible)
                widget.setVisible(is_visible)
        if kind == "symbol":
            symbol_value = symbol
            if symbol_value is None:
                index = self.type_combo.currentIndex()
                kind_data = self.TYPE_OPTIONS[index][1]
                symbol_value = kind_data.get("symbol")
            is_plus_cross = symbol_value in {"+", "x"}
            self.symbol_controls.edge_color_label.setVisible(not is_plus_cross)
            self.symbol_controls.edge_color_combo.setVisible(not is_plus_cross)
        else:
            self.symbol_controls.edge_color_label.setVisible(False)
            self.symbol_controls.edge_color_combo.setVisible(False)

    def _capture_detail_group_height(self, kind: str) -> None:
        group = self.detail_group
        if group is None:
            return
        height = int(group.sizeHint().height() or 0)
        if height <= 0:
            return
        previous = self._detail_group_heights.get(kind, 0)
        if height > previous:
            self._detail_group_heights[kind] = height
        max_height = max(self._detail_group_heights.values(), default=0)
        if max_height and max_height != self._detail_group_fixed_height:
            self._detail_group_fixed_height = max_height
            group.setMinimumHeight(max_height)
            group.setMaximumHeight(max_height)

    def _precompute_detail_group_heights(self) -> None:
        group = self.detail_group
        if group is None:
            return
        current_index = self.type_combo.currentIndex()
        current_data = self.TYPE_OPTIONS[current_index][1]
        original_kind = current_data.get("kind", "symbol")
        original_symbol = current_data.get("symbol")
        self._suspend_updates = True
        try:
            for _, data in self.TYPE_OPTIONS:
                kind = data.get("kind")
                if kind not in self.property_rows:
                    continue
                self._set_property_rows_visible(kind, data.get("symbol"))
                self._update_label_field_visibility(kind)
                group.adjustSize()
                self._capture_detail_group_height(kind)
        finally:
            self._set_property_rows_visible(original_kind, original_symbol)
            self._update_label_field_visibility(original_kind)
            self._suspend_updates = False
        self._capture_detail_group_height(original_kind)

    def _make_coordinate_spin(self, parent: QWidget) -> QDoubleSpinBox:
        spin = QDoubleSpinBox(parent)
        spin.setDecimals(3)
        spin.setRange(-1_000_000.0, 1_000_000.0)
        spin.setSingleStep(0.5)
        spin.setEnabled(False)
        return spin

    def _make_world_line_edit(self, parent: QWidget, placeholder: str) -> QLineEdit:
        edit = QLineEdit(parent)
        edit.setPlaceholderText(placeholder)
        edit.setEnabled(False)
        return edit

    def _populate_world_frame_combo(self) -> None:
        current = normalize_display_frame(self._get_shared_display_frame())
        self.world_frame_combo.blockSignals(True)
        self.world_frame_combo.clear()
        frames = available_display_frames(getattr(self.viewer, "wcs", None))
        for frame in frames:
            self.world_frame_combo.addItem(display_frame_label(frame), frame)
        idx = self.world_frame_combo.findData(current)
        if idx < 0:
            idx = self.world_frame_combo.findData("native")
        if idx >= 0:
            self.world_frame_combo.setCurrentIndex(idx)
        self.world_frame_combo.setEnabled(self.world_frame_combo.count() > 1)
        self.world_frame_combo.blockSignals(False)

    def _current_world_frame(self) -> str:
        data = self.world_frame_combo.currentData()
        frame = normalize_display_frame(data)
        if not frame_is_available(getattr(self.viewer, "wcs", None), frame):
            return "native"
        return frame

    def _shared_world_vector(self, viewer=None):
        target = viewer or self.viewer
        values = []
        for name in ("_get_shared_world_x", "_get_shared_world_y", "_get_shared_world_z", "_get_shared_world_s"):
            getter = getattr(target, name, None)
            if callable(getter):
                try:
                    values.append(float(getter()))
                except Exception:
                    values.append(None)
            else:
                values.append(None)
        return values

    def _get_shared_display_frame(self) -> str:
        getter = getattr(self.viewer, "_get_shared_display_frame", None)
        if callable(getter):
            try:
                return getter()
            except Exception:
                return "native"
        return "native"

    # ------------------------------------------------------------------
    # Signal wiring
    def _connect_signals(self) -> None:
        self.type_combo.currentIndexChanged.connect(self._on_type_changed)
        self.placement_toggle.toggled.connect(self._on_placement_toggled)
        if self.link_all_checkbox is not None:
            self.link_all_checkbox.toggled.connect(self._on_link_all_toggled)

        self.marker_list.itemSelectionChanged.connect(self._on_marker_list_selection)
        self.marker_list.delete_pressed.connect(self._on_delete_marker)
        self.delete_button.clicked.connect(self._on_delete_marker)
        self.delete_all_button.clicked.connect(self._on_delete_all)
        self.save_button.clicked.connect(self._on_save_json)
        self.load_button.clicked.connect(self._on_load_json)

        self.label_edit.editingFinished.connect(self._on_label_changed)
        self.label_edit.returnPressed.connect(self._on_label_changed)
        self.color_combo.currentTextChanged.connect(lambda _: self._apply_common_style())
        self.opacity_spin.valueChanged.connect(lambda _: self._apply_common_style())

        self.pixel_x_spin.valueChanged.connect(self._on_pixel_spin_changed)
        self.pixel_y_spin.valueChanged.connect(self._on_pixel_spin_changed)
        self.world_x_edit.editingFinished.connect(self._on_world_edit_finished)
        self.world_y_edit.editingFinished.connect(self._on_world_edit_finished)
        self.world_frame_combo.currentIndexChanged.connect(self._on_world_frame_changed)

        self.symbol_controls.size_spin.valueChanged.connect(lambda _: self._apply_symbol_style())
        self.symbol_controls.linewidth_spin.valueChanged.connect(lambda _: self._apply_symbol_style())
        self.symbol_controls.edge_color_combo.currentTextChanged.connect(
            lambda _: self._apply_symbol_style()
        )

        self.line_controls.length_spin.valueChanged.connect(lambda _: self._apply_line_style())
        self.line_controls.unit_combo.currentTextChanged.connect(self._on_line_unit_changed)
        self.line_controls.angle_spin.valueChanged.connect(lambda _: self._apply_line_style())
        self.line_controls.linewidth_spin.valueChanged.connect(lambda _: self._apply_line_style())
        self.line_controls.style_combo.currentIndexChanged.connect(self._on_line_style_changed)

        self.text_controls.font_size_spin.valueChanged.connect(lambda _: self._apply_text_style())
        self.text_controls.rotation_spin.valueChanged.connect(lambda _: self._apply_text_style())
        self.text_controls.font_combo.currentTextChanged.connect(lambda _: self._apply_text_style())

        self.marker_manager.markers_changed.connect(self._on_markers_changed)
        self.marker_manager.selection_changed.connect(self._on_manager_selection_changed)
        self.marker_manager.geometry_changed.connect(self._on_marker_geometry_changed)

    # ------------------------------------------------------------------
    # Marker list and selection helpers
    def _refresh_marker_list(self) -> None:
        self._suspend_selection_sync = True
        try:
            selected_lookup = getattr(self.marker_manager, "selected_marker", None)
            selected_marker = selected_lookup() if callable(selected_lookup) else None
            selected_ids = set(self._selected_marker_ids())
            current_id = None
            if selected_marker is not None:
                current_id = selected_marker.marker_id
            elif self._current_marker_id:
                current_id = self._current_marker_id
            if current_id:
                selected_ids.add(current_id)
            self.marker_list.clear()
            items: List[Tuple[str, Marker]] = []
            for plane, layer in self.marker_manager._layers.items():
                for marker in layer.markers.values():
                    items.append((plane, marker))
            # Stable order: plane order xy -> xz -> zy, then insertion order
            plane_order = {"xy": 0, "xz": 1, "zy": 2}
            items.sort(key=lambda entry: plane_order.get(entry[0], 99))
            for plane, marker in items:
                display_text = self._list_entry_text(marker)
                item = QListWidgetItem(display_text, self.marker_list)
                item.setData(Qt.ItemDataRole.UserRole, marker.marker_id)
                self.marker_list.addItem(item)
                if marker.marker_id in selected_ids:
                    item.setSelected(True)
            if selected_ids:
                primary = self._primary_marker()
                if primary is not None:
                    self._current_marker_id = primary.marker_id
                    self._populate_detail_fields(primary, preserve_selection=True)
                else:
                    self._clear_detail_fields()
            else:
                self._clear_detail_fields()
        finally:
            self._suspend_selection_sync = False

    def _selected_marker_ids(self) -> List[str]:
        return [
            item.data(Qt.ItemDataRole.UserRole)
            for item in self.marker_list.selectedItems()
            if item.data(Qt.ItemDataRole.UserRole)
        ]

    def _selected_markers(self) -> List[Marker]:
        markers: List[Marker] = []
        for marker_id in self._selected_marker_ids():
            marker = self.marker_manager.marker_for_id(marker_id)
            if marker is not None:
                markers.append(marker)
        return markers

    def _primary_marker(self) -> Optional[Marker]:
        markers = self._selected_markers()
        if markers:
            return markers[0]
        if self._current_marker_id is None:
            return None
        return self.marker_manager.marker_for_id(self._current_marker_id)

    def _current_marker(self) -> Optional[Marker]:
        return self._primary_marker()

    def set_selection_by_id(self, marker_ids: List[str]) -> None:
        self._suspend_selection_sync = True
        self.marker_list.clearSelection()
        ids_to_select = set(marker_ids)
        for i in range(self.marker_list.count()):
            item = self.marker_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) in ids_to_select:
                item.setSelected(True)
        self._suspend_selection_sync = False
        self._on_marker_list_selection()

    def _list_entry_text(self, marker: Marker) -> str:
        kind_desc = self._kind_display(marker)
        color_value = marker.style.color
        
        # For Plus/Cross markers (SymbolMarker), the primary color is stored in edgecolor.
        # If color is 'none' or missing, check edgecolor.
        if not color_value or (isinstance(color_value, str) and color_value.lower() == "none"):
            if isinstance(marker, SymbolMarker):
                edge = marker.style.edgecolor
                if edge and str(edge).lower() != "none":
                    color_value = edge
        
        if not color_value:
             color_value = "auto"
        elif isinstance(color_value, str) and color_value.lower() == "none":
             color_value = "none"
        color_text = str(color_value)

        label = (marker.label or "").strip()
        detail = ""
        if isinstance(marker, TextMarker):
            content = (getattr(marker, "text", "") or label).strip()
            if content:
                detail = f' "{content}"'
        elif label:
            detail = f' "{label}"'

        plane = marker.plane.upper()
        viewer = self._viewer_for_marker(marker)
        converter = getattr(viewer, "converter", None) if viewer else None
        wcs = getattr(converter, "wcs", None) if converter is not None else getattr(viewer, "wcs", None)
        axis_indices = self._plane_axis_indices(marker.plane, wcs)
        world_values = self._world_strings_from_marker(
            marker,
            converter=converter,
            axis_indices=axis_indices,
            viewer=viewer,
        )
        if world_values and any(world_values):
            world_text = ", ".join(world_values)
        else:
            world_text = "--"
        return f"{kind_desc}{detail} ({color_text}) [{plane}] {world_text}"

    def _kind_display(self, marker: Marker) -> str:
        if isinstance(marker, SymbolMarker):
            symbol = getattr(marker, "symbol", "o")
            symbol_map = {
                "o": "Circle",
                "s": "Square",
                "D": "Diamond",
                "d": "Diamond",
                "^": "Triangle Up",
                "v": "Triangle Down",
                "*": "Star",
                "+": "Plus",
                "x": "Cross",
            }
            return symbol_map.get(symbol, f"Symbol ({symbol})")
        if isinstance(marker, LineMarker):
            return "Line"
        if isinstance(marker, TextMarker):
            return "Text"
        return marker.kind.capitalize()

    # ------------------------------------------------------------------
    # Selection and detail updates
    def _clear_detail_fields(self) -> None:
        self._suspend_updates = True
        try:
            self._current_marker_id = None
            self._pending_selection_id = None
            self.label_edit.clear()
            if "white" in self.COLOR_OPTIONS:
                self.color_combo.setCurrentText("white")
            else:
                self.color_combo.setCurrentIndex(0)
            self.opacity_spin.setValue(1.0)

            self._set_position_controls_enabled(False)
            self.pixel_x_spin.blockSignals(True)
            self.pixel_y_spin.blockSignals(True)
            self.pixel_x_spin.setValue(0.0)
            self.pixel_y_spin.setValue(0.0)
            self.pixel_x_spin.blockSignals(False)
            self.pixel_y_spin.blockSignals(False)
            self.world_x_edit.blockSignals(True)
            self.world_y_edit.blockSignals(True)
            self.world_x_edit.clear()
            self.world_y_edit.clear()
            self.world_x_edit.blockSignals(False)
            self.world_y_edit.blockSignals(False)
            self.line_controls.style_combo.setCurrentIndex(0)
            current_kind = self.TYPE_OPTIONS[self.type_combo.currentIndex()][1].get("kind", "symbol")
            self._update_label_field_visibility(current_kind)
            if current_kind == "text":
                template_text = self._last_text_value or "Text"
                self.label_edit.setText(template_text)
            fallback_font = next((font for font in self.FONT_OPTIONS if font != "STIXGeneral"), self.FONT_OPTIONS[0])
            self.text_controls.font_combo.setCurrentText(fallback_font)
            self.pixel_x_spin.setEnabled(True)
            self.pixel_y_spin.setEnabled(True)
            self.world_x_edit.setEnabled(True)
            self.world_y_edit.setEnabled(True)
        finally:
            self._suspend_updates = False

    def _populate_detail_fields(self, marker: Marker, *, preserve_selection: bool = False) -> None:
        self._suspend_updates = True
        focus_label_field = False
        fallback_font_option = next((font for font in self.FONT_OPTIONS if font != "STIXGeneral"), self.FONT_OPTIONS[0])
        try:
            self._current_plane = marker.plane
            self._current_marker_id = marker.marker_id
            if hasattr(self.marker_manager, "set_active_plane"):
                current_plane_getter = getattr(self.marker_manager, "active_plane", None)
                active_plane = current_plane_getter() if callable(current_plane_getter) else None
                if active_plane != marker.plane:
                    self._pending_selection_id = marker.marker_id
                    self.marker_manager.set_active_plane(marker.plane)
            selected_markers = self._selected_markers()
            multi_selected = len(selected_markers) > 1
            allow_multi_position = multi_selected and self._viewer_supports_position_edit(selected_markers)
            self._set_position_controls_enabled(not multi_selected or allow_multi_position)
            style = marker.style
            type_index = next(
                (i for i, (label, data) in enumerate(self.TYPE_OPTIONS) if self._type_matches_marker(data, marker)),
                0,
            )
            if self.type_combo.currentIndex() != type_index:
                self.type_combo.setCurrentIndex(type_index)
            kind = self.TYPE_OPTIONS[type_index][1].get("kind", "symbol")
            self._update_label_field_visibility(kind)
            if kind == "text":
                self.label_edit.setText(marker.text)
            else:
                self.label_edit.setText(marker.label)
            color_value = style.color
            if isinstance(color_value, str) and color_value.lower() == "none":
                self.color_combo.setCurrentText("None")
            elif isinstance(color_value, str) and color_value in self.COLOR_OPTIONS:
                self.color_combo.setCurrentText(color_value)
            elif "white" in self.COLOR_OPTIONS:
                self.color_combo.setCurrentText("white")
            else:
                self.color_combo.setCurrentIndex(0)
            self.opacity_spin.setValue(style.opacity)

            if isinstance(marker, SymbolMarker):
                
                is_plus_cross = getattr(marker, "symbol", None) in {"+", "x"}
                self.symbol_controls.edge_color_label.setVisible(not is_plus_cross)
                self.symbol_controls.edge_color_combo.setVisible(not is_plus_cross)

                self.symbol_controls.size_spin.setValue(style.size)
                linewidth_value = style.linewidth or 0.0
                if is_plus_cross:
                    linewidth_value = max(1.0, linewidth_value)
                self.symbol_controls.linewidth_spin.setValue(linewidth_value)

                if is_plus_cross:
                    edge_color = style.edgecolor or style.color
                    if (
                        isinstance(edge_color, str)
                        and edge_color in self.COLOR_OPTIONS
                    ):
                        self.color_combo.setCurrentText(edge_color)
                    else:
                        self.color_combo.setCurrentText("white")
                else:
                    edge_color_value = style.edgecolor
                    if (
                        isinstance(edge_color_value, str)
                        and edge_color_value.lower() == "none"
                    ):
                        self.symbol_controls.edge_color_combo.setCurrentText("None")
                    elif (
                        isinstance(edge_color_value, str)
                        and edge_color_value in self.COLOR_OPTIONS
                    ):
                        self.symbol_controls.edge_color_combo.setCurrentText(
                            edge_color_value
                        )
                    else:
                        self.symbol_controls.edge_color_combo.setCurrentText("None")

            elif isinstance(marker, LineMarker):
                #self.detail_stack.setCurrentIndex(1)
                self._update_line_detail_fields(marker, style=style)

            elif isinstance(marker, TextMarker):
                fallback_font_option = next((font for font in self.FONT_OPTIONS if font != "STIXGeneral"), self.FONT_OPTIONS[0])
                self.text_controls.font_size_spin.setValue(style.size)
                self.text_controls.rotation_spin.setValue(style.rotation)
                text_value = marker.text or ""
                if not text_value:
                    text_value = "Text"
                    marker.update_text(text_value)
                self.label_edit.setText(text_value)
                self._last_text_value = text_value
                if "$" in text_value:
                    self.text_controls.font_combo.setCurrentText("STIXGeneral")
                else:
                    family = style.font_family or fallback_font_option
                    if family not in self.FONT_OPTIONS:
                        self.text_controls.font_combo.addItem(family)
                    self.text_controls.font_combo.setCurrentText(family)
                if not text_value.strip():
                    focus_label_field = True
            self._update_position_fields(marker)
        finally:
            self._suspend_updates = False
        self._ensure_marker_selected(marker, preserve_existing=preserve_selection)
        if focus_label_field:
            def _focus_text_field() -> None:
                self.label_edit.setFocus(Qt.FocusReason.OtherFocusReason)
                self.label_edit.selectAll()
            QTimer.singleShot(0, _focus_text_field)

    def _type_matches_marker(self, data: Dict[str, str], marker: Marker) -> bool:
        if data.get("kind") == "symbol" and isinstance(marker, SymbolMarker):
            symbol = data.get("symbol", "")
            return symbol == getattr(marker, "symbol", None)
        if data.get("kind") == "line" and isinstance(marker, LineMarker):
            return True
        if data.get("kind") == "text" and isinstance(marker, TextMarker):
            return True
        return False

    def _set_position_controls_enabled(self, enabled: bool) -> None:
        for widget in (self.pixel_x_spin, self.pixel_y_spin):
            widget.setEnabled(enabled)
        for widget in (self.world_x_edit, self.world_y_edit):
            widget.setEnabled(enabled)

    def _normalized_color_value(self, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        lowered = value.lower() if isinstance(value, str) else value
        if lowered == "none":
            return "none"
        return value

    def _current_kind(self) -> str:
        return self.TYPE_OPTIONS[self.type_combo.currentIndex()][1].get("kind", "symbol")

    def _style_configuration_for_kind(self, kind: str) -> Dict[str, object]:
        style_dict: Dict[str, object] = {
            "color": self._normalized_color_value(self.color_combo.currentText()),
            "opacity": self.opacity_spin.value(),
        }
        kwargs: Dict[str, object] = {}
        style_dict["edgecolor"] = None

        if kind == "symbol":
            size = self.symbol_controls.size_spin.value()
            style_dict["size"] = size
            symbol = self.TYPE_OPTIONS[self.type_combo.currentIndex()][1].get("symbol", "o")
            linewidth_value = self.symbol_controls.linewidth_spin.value()
            if symbol in {"+", "x"}:
                linewidth_value = max(1.0, linewidth_value)
                style_dict["color"] = "none"
                style_dict["edgecolor"] = self._normalized_color_value(
                    self.color_combo.currentText()
                )
            else:
                style_dict["edgecolor"] = self._normalized_color_value(
                    self.symbol_controls.edge_color_combo.currentText()
                )
            style_dict["linewidth"] = linewidth_value
            style_dict["marker_symbol"] = symbol
            kwargs["symbol"] = symbol
        elif kind == "line":
            size = self.symbol_controls.size_spin.value()
            style_dict["size"] = size
            style_dict["linewidth"] = self.line_controls.linewidth_spin.value()
            raw_length = self.line_controls.length_spin.value()
            unit = self.line_controls.unit_combo.currentText()
            pixels = self._convert_length_to_pixels(raw_length, unit, plane=self._current_plane)
            kwargs.update(
                {
                    "length": pixels,
                    "angle_deg": self.line_controls.angle_spin.value(),
                    "unit": unit,
                    "length_source_value": raw_length,
                    "style_mode": self.line_controls.style_combo.currentData(),
                }
            )
        elif kind == "text":
            style_dict["size"] = self.text_controls.font_size_spin.value()
            style_dict["rotation"] = self.text_controls.rotation_spin.value()
            text_value = self.label_edit.text().strip()
            kwargs["text"] = text_value
            if "$" in text_value:
                style_dict["font_family"] = "STIXGeneral"
            else:
                style_dict["font_family"] = self.text_controls.font_combo.currentText()
        else:
            style_dict["size"] = self.symbol_controls.size_spin.value()

        style = MarkerStyle.from_dict(style_dict)
        kwargs["style"] = style

        label = self.label_edit.text().strip()
        if kind != "text":
            if self._suppress_label_from_text and label == self._last_text_value:
                label = ""
            if label:
                kwargs["label"] = label
        return kwargs

    def _resolve_current_plane(self) -> str:
        current_plane = self._current_plane
        selected_lookup = getattr(self.marker_manager, "selected_marker", None)
        selected_marker = selected_lookup() if callable(selected_lookup) else None
        active_plane_getter = getattr(self.marker_manager, "active_plane", None)
        manager_plane = active_plane_getter() if callable(active_plane_getter) else None
        if selected_marker is not None:
            current_plane = selected_marker.plane
        elif manager_plane:
            current_plane = manager_plane
        if not current_plane:
            current_plane = getattr(self.viewer, "plane", "xy")
        self._current_plane = current_plane
        return current_plane

    def _world_inputs_to_pixels(
        self,
        plane: str,
        world_inputs: Tuple[str, str],
    ) -> Optional[Tuple[float, float]]:
        viewer = self._viewer_for_plane(plane)
        converter = getattr(viewer, "converter", None) if viewer else None
        wcs = getattr(converter, "wcs", None) if converter else None
        axis_indices = self._plane_axis_indices(plane, wcs)
        if converter is None or wcs is None or axis_indices is None:
            return None
        cleaned_inputs = [value.strip() for value in world_inputs]
        if not all(cleaned_inputs):
            return None
        try:
            ref_pixel = getattr(wcs, "wcs", None).crpix if hasattr(wcs, "wcs") else wcs.crpix
            ref_world = wcs.wcs_pix2world([ref_pixel], 0)[0]
        except Exception:
            return None
        target_world = list(ref_world)
        fallback_world = self._shared_world_vector(viewer)
        native_values = plane_inputs_to_native(
            wcs,
            plane,
            cleaned_inputs[0],
            cleaned_inputs[1],
            frame=self._current_world_frame(),
            fallback_native_world=fallback_world,
        )
        if native_values is None:
            return None
        for value, idx in zip(native_values, axis_indices):
            target_world[idx] = value
        try:
            pix_coords = converter.world_to_pix(*target_world)
            x_pix = float(pix_coords[axis_indices[0]])
            y_pix = float(pix_coords[axis_indices[1]])
        except Exception:
            return None
        return (x_pix, y_pix)

    def _create_marker_from_coordinates(self, plane: str, x: float, y: float) -> Optional[Marker]:
        entry = self.TYPE_OPTIONS[self.type_combo.currentIndex()]
        kind = entry[1].get("kind", "symbol")
        kwargs = dict(self._style_configuration_for_kind(kind))
        marker: Optional[Marker] = None
        if kind == "symbol":
            marker = self.marker_manager.create_symbol_marker(plane, (x, y), **kwargs)
        elif kind == "line":
            marker = self.marker_manager.create_line_marker(plane, (x, y), **kwargs)
        elif kind == "text":
            marker = self.marker_manager.create_text_marker(plane, (x, y), **kwargs)
        else:
            marker = self.marker_manager.create_symbol_marker(plane, (x, y), **kwargs)
        if marker is None:
            return None
        if kind == "text" and self.placement_toggle.isChecked():
            self.marker_manager.select_marker(None)
            self._current_marker_id = None
            self._on_deselect_markers()
            self._refresh_marker_list()
            self._focus_label_edit()
        else:
            self.marker_manager.select_marker(marker)
            self._refresh_marker_list()
            self._populate_detail_fields(marker)
        return marker

    def _focus_label_edit(self) -> None:
        def _focus() -> None:
            try:
                if self.isMinimized():
                    self.showNormal()
                self.raise_()
                self.activateWindow()
            except Exception:
                pass
            self.label_edit.setFocus(Qt.FocusReason.ActiveWindowFocusReason)
            self.label_edit.selectAll()
        for delay_ms in (0, 40, 120):
            QTimer.singleShot(delay_ms, _focus)

    def focus_text_entry_for_marker(self, marker: Optional[Marker] = None) -> None:
        """Focus the label editor when the target marker is a text marker."""
        target = marker if marker is not None else self._current_marker()
        if not isinstance(target, TextMarker):
            return
        if not self.isVisible():
            return
        self._focus_label_edit()

    def _update_label_field_visibility(self, kind: str) -> None:
        if kind == "text":
            self.label_field_label.setText("Text")
            self.label_edit.setPlaceholderText("Text")
        else:
            self.label_field_label.setText("Label")
            self.label_edit.setPlaceholderText("Label")
        self.label_field_label.setVisible(True)
        self.label_edit.setVisible(True)

    def _update_position_fields(self, marker: Marker) -> None:
        pixel = list(marker.pixel)
        if len(pixel) < 2:
            pixel.extend([0.0] * (2 - len(pixel)))
        x_pix, y_pix = float(pixel[0]), float(pixel[1])
        self.pixel_x_spin.blockSignals(True)
        self.pixel_y_spin.blockSignals(True)
        self.pixel_x_spin.setValue(x_pix)
        self.pixel_y_spin.setValue(y_pix)
        self.pixel_x_spin.blockSignals(False)
        self.pixel_y_spin.blockSignals(False)

        previous_world_x = self.world_x_edit.text()
        previous_world_y = self.world_y_edit.text()

        viewer = self._viewer_for_plane(marker.plane)
        converter = getattr(viewer, "converter", None) if viewer else None
        wcs = getattr(converter, "wcs", None) if converter else None
        axis_indices = self._plane_axis_indices(marker.plane, wcs)

        world_editable = bool(converter is not None and wcs is not None and axis_indices is not None)
        world_texts: Optional[Tuple[str, str]] = None

        if world_editable:
            try:
                pixel_vector = self._build_pixel_vector(marker.plane, x_pix, y_pix, wcs)
                native_world = wcs.wcs_pix2world([pixel_vector], 0)[0]
                frame = self._current_world_frame()
                fallback_world = self._shared_world_vector(viewer)
                wx_val, wy_val, wx_axis, wy_axis = plane_values_for_display(
                    wcs,
                    marker.plane,
                    native_world[axis_indices[0]],
                    native_world[axis_indices[1]],
                    frame=frame,
                    fallback_native_world=fallback_world,
                )
                if converter is not None and hasattr(converter, "format_world_coordinate"):
                    wx_text = converter.format_world_coordinate(wx_val, wx_axis)
                    wy_text = converter.format_world_coordinate(wy_val, wy_axis)
                else:
                    wx_text = str(wx_val)
                    wy_text = str(wy_val)
                world_texts = (str(wx_text), str(wy_text))
            except Exception:
                world_texts = None

        if world_texts is None:
            world_texts = self._world_strings_from_marker(
                marker,
                converter=converter,
                axis_indices=axis_indices,
                viewer=viewer,
            )

        if not any(world_texts) and (previous_world_x or previous_world_y):
            world_texts = (previous_world_x, previous_world_y)

        self._set_world_fields(world_texts[0], world_texts[1], editable=world_editable)

    def _set_world_fields(self, x_text: str, y_text: str, *, editable: bool) -> None:
        self.world_x_edit.blockSignals(True)
        self.world_y_edit.blockSignals(True)
        self.world_x_edit.setText(x_text)
        self.world_y_edit.setText(y_text)
        self.world_x_edit.blockSignals(False)
        self.world_y_edit.blockSignals(False)
        self.world_x_edit.setEnabled(editable)
        self.world_y_edit.setEnabled(editable)

    def _world_strings_from_marker(
        self,
        marker: Marker,
        *,
        converter=None,
        axis_indices: Optional[Tuple[int, int]] = None,
        viewer=None,
    ) -> Tuple[str, str]:
        world = getattr(marker, "world", None)
        if not world or len(world) < 2:
            return ("", "")
        wcs = getattr(converter, "wcs", None) if converter is not None else None
        frame = self._current_world_frame()
        if wcs is not None and axis_indices:
            try:
                fallback_world = self._shared_world_vector(viewer)
                wx_val, wy_val, wx_axis, wy_axis = plane_values_for_display(
                    wcs,
                    marker.plane,
                    world[0],
                    world[1],
                    frame=frame,
                    fallback_native_world=fallback_world,
                )
                if converter is not None and hasattr(converter, "format_world_coordinate"):
                    return (
                        str(converter.format_world_coordinate(wx_val, wx_axis)),
                        str(converter.format_world_coordinate(wy_val, wy_axis)),
                    )
                return (str(wx_val), str(wy_val))
            except Exception:
                pass
        default_decimals = self._default_world_decimals(converter)
        if (
            converter is not None
            and hasattr(converter, "format_world_coordinate")
            and hasattr(converter, "get_axis_types")
            and axis_indices
        ):
            try:
                axis_types = converter.get_axis_types()
            except Exception:
                axis_types = None
            if axis_types:
                formatted_values: List[str] = []
                for value, axis_idx in zip(world, axis_indices):
                    axis_type = None
                    if axis_idx is not None and 0 <= axis_idx < len(axis_types):
                        try:
                            axis_type = display_axis_type(axis_types[axis_idx], frame)
                            formatted = converter.format_world_coordinate(value, axis_type)
                        except Exception:
                            formatted = None
                        if formatted is not None:
                            formatted_values.append(str(formatted))
                            continue
                    decimals = self._auto_world_decimals_for_axis(
                        converter=converter,
                        wcs=wcs,
                        axis_index=axis_idx,
                        axis_type=axis_type,
                        fallback=default_decimals,
                    )
                    formatted_values.append(self._format_world_numeric_fallback(value, decimals))
                if len(formatted_values) == 2:
                    return (formatted_values[0], formatted_values[1])
        if axis_indices and len(axis_indices) == 2:
            x_decimals = self._auto_world_decimals_for_axis(
                converter=converter,
                wcs=wcs,
                axis_index=axis_indices[0],
                fallback=default_decimals,
            )
            y_decimals = self._auto_world_decimals_for_axis(
                converter=converter,
                wcs=wcs,
                axis_index=axis_indices[1],
                fallback=default_decimals,
            )
        else:
            x_decimals = default_decimals
            y_decimals = default_decimals
        return (
            self._format_world_numeric_fallback(world[0], x_decimals),
            self._format_world_numeric_fallback(world[1], y_decimals),
        )

    @staticmethod
    def _format_world_numeric_fallback(value, decimals: int) -> str:
        try:
            return f"{float(value):.{max(0, int(decimals))}f}"
        except Exception:
            return str(value)

    @staticmethod
    def _default_world_decimals(converter, fallback: int = 6) -> int:
        decimals = fallback
        config = getattr(converter, "config", None) if converter is not None else None
        if config is not None:
            try:
                decimals = int(config.get("number_decimals", decimals))
            except Exception:
                pass
        return max(0, int(decimals))

    @staticmethod
    def _auto_world_decimals_for_axis(
        *,
        converter,
        wcs,
        axis_index: Optional[int],
        axis_type: Optional[str] = None,
        fallback: int = 6,
        max_decimals: int = 10,
    ) -> int:
        try:
            base = max(0, int(fallback))
        except Exception:
            base = 6

        auto_enabled = True
        config = getattr(converter, "config", None) if converter is not None else None
        if isinstance(config, dict):
            auto_enabled = bool(config.get("auto_precision_digits", True))
        if not auto_enabled:
            return base

        if converter is not None and axis_type and hasattr(converter, "_effective_decimals"):
            try:
                return max(0, int(converter._effective_decimals(axis_type)))
            except Exception:
                pass

        if wcs is None or axis_index is None:
            return base
        try:
            step = abs(float(wcs.wcs.cdelt[int(axis_index)]))
        except Exception:
            return base
        if not math.isfinite(step) or step <= 0:
            return base
        try:
            decimals = int(math.ceil(-math.log10(step / 10.0)))
        except Exception:
            return base
        return max(0, min(int(max_decimals), decimals))

    def _update_line_detail_fields(self, marker: LineMarker, *, style: Optional[MarkerStyle] = None) -> None:
        style = style or marker.style
        unit_combo = self.line_controls.unit_combo
        target_unit = marker.unit if marker.unit in self.UNIT_OPTIONS else unit_combo.currentText()
        if marker.unit in self.UNIT_OPTIONS:
            unit_combo.blockSignals(True)
            index = unit_combo.findText(marker.unit, Qt.MatchFlag.MatchFixedString)
            if index >= 0:
                unit_combo.setCurrentIndex(index)
            unit_combo.blockSignals(False)
        else:
            target_unit = unit_combo.currentText()

        display_unit = unit_combo.currentText() or target_unit or "pixel"
        display_length = self._convert_length_from_pixels(marker.length, display_unit, plane=marker.plane)

        length_spin = self.line_controls.length_spin
        length_spin.blockSignals(True)
        length_spin.setValue(display_length)
        length_spin.blockSignals(False)

        angle_spin = self.line_controls.angle_spin
        angle_spin.blockSignals(True)
        angle_spin.setValue(marker.angle_deg)
        angle_spin.blockSignals(False)

        linewidth_spin = self.line_controls.linewidth_spin
        linewidth_spin.blockSignals(True)
        linewidth_spin.setValue(style.linewidth or 1.0)
        linewidth_spin.blockSignals(False)

        style_combo = self.line_controls.style_combo
        style_combo.blockSignals(True)
        found = False
        for idx in range(style_combo.count()):
            if style_combo.itemData(idx) == marker.style_mode:
                style_combo.setCurrentIndex(idx)
                found = True
                break
        if not found and style_combo.count():
            style_combo.setCurrentIndex(0)
        style_combo.blockSignals(False)

    def _update_text_detail_fields(self, marker: TextMarker) -> None:
        rotation_spin = self.text_controls.rotation_spin
        rotation_spin.blockSignals(True)
        rotation_spin.setValue(marker.style.rotation)
        rotation_spin.blockSignals(False)

    def _ensure_marker_selected(
        self,
        marker: Optional[Marker],
        *,
        sync_manager: bool = True,
        preserve_existing: bool = False,
    ) -> None:
        if marker is None:
            return
        if sync_manager:
            self._pending_selection_id = marker.marker_id
        if preserve_existing:
            selected_ids = set(self._selected_marker_ids())
        else:
            selected_ids = set()
        selected_ids.add(marker.marker_id)
        previous_flag = self._suspend_selection_sync
        self._suspend_selection_sync = True
        try:
            for idx in range(self.marker_list.count()):
                item = self.marker_list.item(idx)
                marker_id = item.data(Qt.ItemDataRole.UserRole)
                item.setSelected(marker_id in selected_ids)
        finally:
            self._suspend_selection_sync = previous_flag
        self._current_marker_id = marker.marker_id
        if sync_manager and hasattr(self.marker_manager, "select_marker"):
            try:
                self.marker_manager.select_marker(marker)
            except Exception:
                pass
        if not sync_manager:
            self._pending_selection_id = None

    def _plane_axis_indices(self, plane: Optional[str], wcs) -> Optional[Tuple[int, int]]:
        if wcs is None:
            return None
        fallback = plane_axis_indices(plane, wcs)
        viewer = self._viewer_for_plane(plane)
        if viewer is not None:
            resolver = getattr(viewer, "marker_axis_indices", None)
            if callable(resolver):
                try:
                    indices = resolver(plane)
                except Exception:
                    indices = None
                if indices:
                    try:
                        first, second = indices[:2]
                        return (int(first), int(second))
                    except Exception:
                        pass
        return fallback

    def _schedule_pending_selection_restore(self) -> None:
        pending_id = self._pending_selection_id
        if not pending_id:
            return
        marker = self.marker_manager.marker_for_id(pending_id)
        if marker is None:
            return
        def _restore():
            self._ensure_marker_selected(marker, preserve_existing=True)
        QTimer.singleShot(0, _restore)

    def _viewer_for_marker(self, marker: Marker):
        return self._viewer_for_plane(marker.plane)

    def _viewer_supports_link_all(self) -> bool:
        support_fn = getattr(self.viewer, "supports_marker_link_all", None)
        if callable(support_fn):
            try:
                return bool(support_fn())
            except Exception:
                return False
        return False

    def _viewer_link_all_enabled(self) -> bool:
        getter = getattr(self.viewer, "marker_link_all_enabled", None)
        if callable(getter):
            try:
                return bool(getter())
            except Exception:
                return False
        return False

    def _set_viewer_link_all(self, enabled: bool) -> None:
        setter = getattr(self.viewer, "set_marker_link_all", None)
        if callable(setter):
            try:
                setter(bool(enabled))
            except Exception:
                pass

    def _on_link_all_toggled(self, checked: bool) -> None:
        self._set_viewer_link_all(checked)

    def _viewer_supports_position_edit(self, markers: Iterable[Marker]) -> bool:
        markers = list(markers)
        if not markers:
            return False
        viewers = {self._viewer_for_marker(marker) for marker in markers}
        if None in viewers or len(viewers) != 1:
            return False
        viewer = viewers.pop()
        checker = getattr(viewer, "can_update_marker_positions", None)
        if callable(checker):
            try:
                return bool(checker(markers))
            except Exception:
                return False
        return False

    def _base_plane(self, plane: Optional[str]) -> str:
        plane_value = plane or "xy"
        viewer = None
        try:
            viewer = self._viewer_for_plane(plane_value)
        except Exception:
            viewer = None
        if viewer is not None:
            resolver = getattr(viewer, "marker_plane_base", None)
            if callable(resolver):
                try:
                    base = resolver(plane_value)
                    if base:
                        return str(base).lower()
                except Exception:
                    pass
        plane_key = str(plane_value).lower()
        for candidate in ("xy", "xz", "zy"):
            if candidate in plane_key:
                return candidate
        return plane_key

    def _get_display_slices(self, viewer) -> Optional[List[str]]:
        if viewer is None:
            return None
        displaymap = getattr(viewer, "displaymap", None)
        if displaymap is not None:
            slices = getattr(displaymap, "slices", None)
            if slices:
                return slices
        for attr in ("integ_slice", "projection_slices", "slice"):
            slices = getattr(viewer, attr, None)
            if slices and not callable(slices):
                return slices
        format_pix = getattr(viewer, "format_pix", None)
        if format_pix is not None:
            return getattr(format_pix, "slices", None)
        return None

    def _build_pixel_vector(self, plane: str, x_pix: float, y_pix: float, wcs) -> List[float]:
        naxis = getattr(wcs, "naxis", 0) or 0
        plane_key = self._base_plane(plane)
        x_fixed = float(self.viewer._get_shared_xpix())
        y_fixed = float(self.viewer._get_shared_ypix())
        z_fixed = float(self.viewer._get_shared_zpix())
        stokes_fixed = float(self.viewer._get_shared_spix())
        vector: List[float] = []
        for axis in range(naxis):
            if axis == 0:
                if plane_key == "zy":
                    vector.append(x_fixed)
                else:
                    vector.append(float(x_pix))
            elif axis == 1:
                if plane_key == "xz":
                    vector.append(y_fixed)
                else:
                    vector.append(float(y_pix))
            elif axis == 2:
                if plane_key == "xy":
                    vector.append(z_fixed)
                elif plane_key == "xz":
                    vector.append(float(y_pix))
                elif plane_key == "zy":
                    vector.append(float(x_pix))
                else:
                    vector.append(z_fixed)
            elif axis == 3:
                vector.append(stokes_fixed)
            else:
                vector.append(0.0)
        return vector

    # ------------------------------------------------------------------
    # Signal handlers
    def _on_marker_list_selection(self) -> None:
        if self._suspend_selection_sync:
            return
        markers = self._selected_markers()
        if not markers:
            if self._pending_selection_id:
                self._schedule_pending_selection_restore()
                return
            self._clear_detail_fields()
            self.marker_manager.select_marker(None)
            return
        marker = markers[0]
        self._pending_selection_id = None
        self._current_marker_id = marker.marker_id
        self._populate_detail_fields(marker, preserve_selection=True)
        self._selection_from_list = True
        try:
            self.marker_manager.select_marker(marker)
        finally:
            self._selection_from_list = False

    def _on_markers_changed(self, plane: str) -> None:
        if self._suspend_updates:
            return
        self._refresh_marker_list()
        current = self._current_marker()
        if current is not None:
            self._update_position_fields(current)
            self._ensure_marker_selected(current, preserve_existing=True)
        if self.placement_toggle.isChecked():
            self._configure_placement()

    def _selected_or_current_markers(self, marker_cls=None) -> List[Marker]:
        """Return selected markers, or fallback to current marker when appropriate."""
        if marker_cls is None:
            markers = list(self._selected_markers())
        else:
            markers = [marker for marker in self._selected_markers() if isinstance(marker, marker_cls)]
        if markers:
            return markers
        marker = self._current_marker()
        if marker is None:
            return []
        if marker_cls is not None and not isinstance(marker, marker_cls):
            return []
        return [marker]

    def _finalize_marker_style_update(
        self,
        changed_planes: Iterable[str],
        *,
        focus_marker_list: bool = False,
    ) -> None:
        changed = [str(plane).lower() for plane in changed_planes if plane]
        if not changed:
            return
        for plane in changed:
            self.marker_manager.redraw_plane(plane)
        self._notify_marker_state_changed(changed)
        self._refresh_marker_list()
        if focus_marker_list:
            self.marker_list.setFocus()

    def _update_placement_if_enabled(self) -> None:
        if self.placement_toggle.isChecked():
            self._configure_placement()

    def _notify_marker_state_changed(self, planes: Iterable[str]) -> None:
        """Notify observers that marker state changed without geometry edits."""
        seen: set[str] = set()
        for plane in planes:
            plane_name = str(plane or "").lower()
            if not plane_name or plane_name in seen:
                continue
            seen.add(plane_name)
            try:
                self.marker_manager.markers_changed.emit(plane_name)
            except Exception:
                continue

    def _on_manager_selection_changed(self, marker: Optional[Marker]) -> None:
        if self._suspend_selection_sync:
            return
        self._suspend_selection_sync = True
        try:
            if marker is None:
                if self._pending_selection_id:
                    self._schedule_pending_selection_restore()
                    return
                self.marker_list.clearSelection()
                self._clear_detail_fields()
                return
            self._pending_selection_id = None
            preserve_selection = self._selection_from_list
            self._ensure_marker_selected(
                marker,
                sync_manager=False,
                preserve_existing=preserve_selection,
            )
            self._populate_detail_fields(
                marker,
                preserve_selection=preserve_selection,
            )
        finally:
            self._suspend_selection_sync = False

    def _on_pixel_spin_changed(self, _value: float) -> None:
        if self._suspend_updates:
            return
        markers = self._selected_markers()
        allow_multi = self._viewer_supports_position_edit(markers)
        if not allow_multi:
            marker = self._current_marker()
            if marker is None:
                return
            markers = [marker]
        elif not markers:
            return
        x_val = self.pixel_x_spin.value()
        y_val = self.pixel_y_spin.value()
        primary = markers[0]
        any_updated = False
        self._suspend_updates = True
        try:
            for target in markers:
                existing = list(target.pixel)
                if len(existing) < 2:
                    existing.extend([0.0] * (2 - len(existing)))
                if len(existing) >= 2 and math.isclose(existing[0], x_val, abs_tol=1e-6) and math.isclose(existing[1], y_val, abs_tol=1e-6):
                    continue
                existing[0] = x_val
                existing[1] = y_val
                self.marker_manager.update_marker_pixel(target, tuple(existing))
                any_updated = True
        finally:
            self._suspend_updates = False
        if any_updated:
            self._update_position_fields(primary)
            self._refresh_marker_list()
        else:
            self._update_position_fields(primary)

    def _on_pixel_entry_return_pressed(self) -> None:
        if self._suspend_updates:
            return
        if self._selected_markers() or self._current_marker() is not None:
            return
        if not getattr(self, "marker_manager", None):
            return
        plane = self._resolve_current_plane()
        marker = self._create_marker_from_coordinates(plane, self.pixel_x_spin.value(), self.pixel_y_spin.value())
        if marker is None:
            return
        self.marker_manager.redraw_plane(plane)

    def _on_world_entry_return_pressed(self) -> None:
        if self._suspend_updates:
            return
        if self._selected_markers() or self._current_marker() is not None:
            return
        if not getattr(self, "marker_manager", None):
            return
        world_inputs = (self.world_x_edit.text().strip(), self.world_y_edit.text().strip())
        if not all(world_inputs):
            return
        plane = self._resolve_current_plane()
        pixel_coords = self._world_inputs_to_pixels(plane, world_inputs)
        if pixel_coords is None:
            return
        marker = self._create_marker_from_coordinates(plane, pixel_coords[0], pixel_coords[1])
        if marker is None:
            return
        self.marker_manager.redraw_plane(plane)

    def _on_world_edit_finished(self) -> None:
        if self._suspend_updates:
            return
        markers = self._selected_markers()
        allow_multi = self._viewer_supports_position_edit(markers)
        if not allow_multi:
            marker = self._current_marker()
            if marker is None:
                return
            markers = [marker]
        elif not markers:
            return
        primary = markers[0]
        world_inputs = (self.world_x_edit.text().strip(), self.world_y_edit.text().strip())
        if not all(world_inputs):
            self._update_position_fields(primary)
            return
        pixel_coords = self._world_inputs_to_pixels(primary.plane, world_inputs)
        if pixel_coords is None:
            self._update_position_fields(primary)
            return
        x_pix, y_pix = pixel_coords
        any_updated = False
        self._suspend_updates = True
        try:
            for target in markers:
                existing = list(target.pixel)
                if len(existing) < 2:
                    existing.extend([0.0] * (2 - len(existing)))
                if len(existing) >= 2 and math.isclose(existing[0], x_pix, abs_tol=1e-6) and math.isclose(existing[1], y_pix, abs_tol=1e-6):
                    continue
                existing[0] = x_pix
                existing[1] = y_pix
                self.marker_manager.update_marker_pixel(target, tuple(existing))
                any_updated = True
        finally:
            self._suspend_updates = False
        if any_updated:
            self._update_position_fields(primary)
            self._refresh_marker_list()
        else:
            self._update_position_fields(primary)

    def _on_world_frame_changed(self, _index: int) -> None:
        if self._suspend_updates:
            return
        current = self._current_marker()
        if current is not None:
            self._update_position_fields(current)
        self._refresh_marker_list()

    def _on_marker_geometry_changed(self, marker: Marker) -> None:
        if self._suspend_updates:
            return
        current = self._current_marker()
        if current is not None and marker.marker_id == current.marker_id:
            self._suspend_updates = True
            try:
                self._update_position_fields(marker)
                if isinstance(marker, LineMarker):
                    self._update_line_detail_fields(marker)
                elif isinstance(marker, TextMarker):
                    self._update_text_detail_fields(marker)
            finally:
                self._suspend_updates = False
        self._refresh_marker_list()

    def _on_type_changed(self, index: int) -> None:
        prev_kind = getattr(self, "_last_type_kind", None)
        programmatic = self._suspend_updates
        kind_data = self.TYPE_OPTIONS[index][1]
        kind = kind_data.get("kind")

        self._set_property_rows_visible(kind, kind_data.get("symbol"))

        self._update_label_field_visibility(kind)
        if kind == "text" and not self._selected_markers() and self._current_marker() is None:
            template_text = self._last_text_value or "Text"
            if self.label_edit.text() != template_text:
                self.label_edit.setText(template_text)

        if kind == "symbol":
            symbol = kind_data.get("symbol")
            if symbol in {"+", "x"} and self.symbol_controls.linewidth_spin.value() < 1.0:
                self.symbol_controls.linewidth_spin.setValue(1.0)

        if not programmatic:
            if prev_kind == "text" and kind != "text":
                self._last_text_value = self.label_edit.text().strip()
                self._suppress_label_from_text = True
                self.label_edit.clear()
            elif kind == "text":
                self._suppress_label_from_text = False

        if programmatic:
            self._last_type_kind = kind
            return

        changed_planes: set[str] = set()
        if kind == "symbol":
            markers = self._selected_or_current_markers(SymbolMarker)
            if markers:
                new_symbol = kind_data.get("symbol", "o")
                is_new_plus_cross = new_symbol in {"+", "x"}
                for target in markers:
                    style_dict = target.style.to_dict()
                    was_plus_cross = getattr(target, "symbol", None) in {"+", "x"}

                    if is_new_plus_cross and not was_plus_cross:
                        if style_dict.get("linewidth", 0.0) < 1.0:
                            style_dict["linewidth"] = 1.0
                        current_color = style_dict.get("color", "none")
                        if current_color != "none":
                            style_dict["edgecolor"] = current_color
                        elif style_dict.get("edgecolor", "none") == "none":
                            style_dict["edgecolor"] = "white"
                        style_dict["color"] = "none"
                    elif not is_new_plus_cross and was_plus_cross:
                        current_edge_color = style_dict.get("edgecolor", "none")
                        if current_edge_color != "none":
                            style_dict["color"] = current_edge_color
                            style_dict["edgecolor"] = "none"

                    target.symbol = new_symbol
                    target.update_style(MarkerStyle.from_dict(style_dict))
                    changed_planes.add(target.plane)

        if changed_planes:
            self._finalize_marker_style_update(changed_planes, focus_marker_list=True)

        self._capture_detail_group_height(kind)
        self._update_placement_if_enabled()

        self._last_type_kind = kind

    def _on_placement_toggled(self, enabled: bool) -> None:
        if enabled:
            self._configure_placement()
            if hasattr(self.viewer, "set_marker_mode"):
                self.viewer.set_marker_mode(True)
        else:
            self.marker_manager.cancel_placement()
            if hasattr(self.viewer, "set_marker_mode"):
                self.viewer.set_marker_mode(False)

    def _on_label_changed(self) -> None:
        if self._suspend_updates:
            return
        markers = self._selected_or_current_markers()
        value = self.label_edit.text().strip()
        kind = self._current_kind()

        template_update_only = not markers

        if kind == "text":
            self._last_text_value = value or "Text"
        else:
            self._suppress_label_from_text = False

        if template_update_only:
            self._update_placement_if_enabled()
            return

        if markers:
            changed_planes: set[str] = set()
            for target in markers:
                if isinstance(target, TextMarker):
                    target.update_text(value)
                    changed_planes.add(target.plane)
                else:
                    target.set_label(value)
                    changed_planes.add(target.plane)
            self._finalize_marker_style_update(changed_planes)
            primary = markers[0]
            if isinstance(primary, TextMarker):
                self.label_edit.setText(primary.text)
        if kind == "text":
            if "$" in value:
                self.text_controls.font_combo.setCurrentText("STIXGeneral")
            else:
                if self.text_controls.font_combo.currentText() == "STIXGeneral":
                    fallback_font = next((font for font in self.FONT_OPTIONS if font != "STIXGeneral"), self.FONT_OPTIONS[0])
                    self.text_controls.font_combo.setCurrentText(fallback_font)

        self._update_placement_if_enabled()

    def _apply_common_style(self) -> None:
        if self._suspend_updates:
            return
        markers = self._selected_or_current_markers()
        if not markers:
            self._update_placement_if_enabled()
            return

        color_value = self._normalized_color_value(self.color_combo.currentText())
        style_dict = {"opacity": self.opacity_spin.value()}

        changed_planes: set[str] = set()
        for target in markers:
            is_plus_cross = (
                isinstance(target, SymbolMarker) and getattr(target, "symbol", None) in {"+", "x"}
            )
            target_style = target.style.to_dict()
            target_style.update(style_dict)
            if is_plus_cross:
                target_style["edgecolor"] = color_value
                target_style["color"] = "none"
            else:
                target_style["color"] = color_value
            target.update_style(MarkerStyle.from_dict(target_style))
            changed_planes.add(target.plane)
        self._finalize_marker_style_update(changed_planes)
        self._update_placement_if_enabled()

    def _apply_symbol_style(self) -> None:
        if self._suspend_updates:
            return
        markers = self._selected_or_current_markers(SymbolMarker)
        if not markers:
            self._update_placement_if_enabled()
            return
        changed_planes: set[str] = set()
        for target in markers:
            style_dict = target.style.to_dict()
            style_dict["size"] = self.symbol_controls.size_spin.value()
            linewidth_value = self.symbol_controls.linewidth_spin.value()
            symbol = getattr(target, "symbol", None)

            if symbol in {"+", "x"}:
                linewidth_value = max(1.0, linewidth_value)
                style_dict["edgecolor"] = self._normalized_color_value(
                    self.color_combo.currentText()
                )
                style_dict["color"] = "none"
            else:
                style_dict["edgecolor"] = self._normalized_color_value(
                    self.symbol_controls.edge_color_combo.currentText()
                )

            style_dict["linewidth"] = linewidth_value
            target.update_style(MarkerStyle.from_dict(style_dict))
            changed_planes.add(target.plane)
        self._finalize_marker_style_update(changed_planes)
        self._update_placement_if_enabled()

    def _apply_line_style(self) -> None:
        if self._suspend_updates:
            return
        markers = self._selected_or_current_markers(LineMarker)
        if not markers:
            self._update_placement_if_enabled()
            return
        raw_length = self.line_controls.length_spin.value()
        unit = self.line_controls.unit_combo.currentText()
        linewidth_value = self.line_controls.linewidth_spin.value()
        color_value = self._normalized_color_value(self.color_combo.currentText())
        opacity_value = self.opacity_spin.value()
        style_mode = self.line_controls.style_combo.currentData()
        angle_value = self.line_controls.angle_spin.value()
        changed_planes: set[str] = set()
        for target in markers:
            pixels = self._convert_length_to_pixels(raw_length, unit, plane=target.plane)
            style_dict = target.style.to_dict()
            style_dict["linewidth"] = linewidth_value
            style_dict["color"] = color_value
            style_dict["opacity"] = opacity_value
            target.set_style_mode(style_mode)
            target.update_style(MarkerStyle.from_dict(style_dict))
            target.set_unit(unit)
            target.set_length(pixels, source_value=raw_length)
            target.set_angle(angle_value)
            changed_planes.add(target.plane)
        self._finalize_marker_style_update(changed_planes)
        self._update_placement_if_enabled()

    def _on_line_unit_changed(self, unit: str) -> None:
        if self._suspend_updates:
            return
        markers = self._selected_or_current_markers(LineMarker)
        marker = markers[0] if markers else self._current_marker()
        if isinstance(marker, LineMarker):
            converted = self._convert_length_from_pixels(marker.length, unit, plane=marker.plane)
            self._suspend_updates = True
            try:
                self.line_controls.length_spin.setValue(converted)
            finally:
                self._suspend_updates = False
        self._apply_line_style()

    def _on_line_style_changed(self, _index: int) -> None:
        if self._suspend_updates:
            return
        self._apply_line_style()

    def _apply_text_style(self) -> None:
        if self._suspend_updates:
            return
        markers = self._selected_or_current_markers(TextMarker)
        if not markers:
            self._update_placement_if_enabled()
            return
        size_value = self.text_controls.font_size_spin.value()
        rotation_value = self.text_controls.rotation_spin.value()
        color_value = self._normalized_color_value(self.color_combo.currentText())
        font_value = self.text_controls.font_combo.currentText()
        opacity_value = self.opacity_spin.value()
        changed_planes: set[str] = set()
        for target in markers:
            style_dict = target.style.to_dict()
            style_dict["size"] = size_value
            style_dict["rotation"] = rotation_value
            text_value = target.text or ""
            style_dict["color"] = color_value
            if "$" in text_value:
                style_dict["font_family"] = "STIXGeneral"
            else:
                style_dict["font_family"] = font_value
            style_dict["opacity"] = opacity_value
            target.update_style(MarkerStyle.from_dict(style_dict))
            changed_planes.add(target.plane)
        self._finalize_marker_style_update(changed_planes)
        self._update_placement_if_enabled()

    def _on_delete_marker(self) -> None:
        markers = self._selected_markers()
        if not markers:
            marker = self._current_marker()
            if marker is not None:
                markers = [marker]
        if not markers:
            return
        planes: set[str] = set()
        for marker in markers:
            planes.add(marker.plane)
            self.marker_manager.remove_marker(marker.marker_id, marker.plane)
        self._refresh_marker_list()
        for plane in planes:
            self.marker_manager.redraw_plane(plane)

    def _on_delete_all(self) -> None:
        planes = list(self.marker_manager._layers.keys())
        if not planes:
            self._show_info("No markers to delete.")
            return
        any_removed = False
        for plane in planes:
            layer = self.marker_manager._layers.get(plane)
            if layer is None or not layer.markers:
                continue
            any_removed = True
            self.marker_manager.clear_plane(plane)
            self.marker_manager.redraw_plane(plane)
        if not any_removed:
            self._show_info("No markers to delete.")
            return
        self._current_marker_id = None
        self.marker_manager.select_marker(None)
        self._refresh_marker_list()

    def _on_deselect_markers(self) -> None:
        self._pending_selection_id = None
        self.marker_list.clearSelection()

    # ------------------------------------------------------------------
    # Persistence and export helpers
    def _choose_plane_for_export(self, title: str) -> Optional[str]:
        planes_set = {marker.plane for layer in self.marker_manager._layers.values() for marker in layer.markers.values()}
        def _plane_sort_key(name: str) -> Tuple[int, int, str]:
            base_order = {"xy": 0, "xz": 1, "zy": 2}.get(name, 50)
            suffix_num = 0
            try:
                parts = name.rsplit("_", 1)
                if len(parts) == 2:
                    suffix_num = int(parts[1])
            except Exception:
                suffix_num = 0
            return (base_order, suffix_num, name)
        planes = sorted(planes_set, key=_plane_sort_key)
        if not planes:
            return None
        multi_choice = len(planes) > 1
        options = list(planes)
        if multi_choice:
            options.insert(0, "all")
        if len(options) == 1:
            return options[0]
        plane_labels = [opt.upper() for opt in options]
        choice, ok = QInputDialog.getItem(self, title, "Select plane:", plane_labels, 0, False)
        if not ok or not choice:
            return None
        return options[plane_labels.index(choice)]

    def _on_save_json(self) -> None:
        plane = self._choose_plane_for_export("Save Markers")
        if plane is None:
            return
        if plane.lower() == "all":
            states: Dict[str, MarkerState] = {}
            for layer_plane in list(self.marker_manager._layers.keys()):
                states.update(self._collect_states(layer_plane))
            if not states:
                self._show_info("No markers to save.")
                return
        else:
            states = self._collect_states(plane)
            if not states:
                self._show_info(f"No markers to save for plane '{plane.upper()}'.")
                return
        default_dir = os.path.dirname(getattr(self.viewer, "filename_path", "")) or os.getcwd()
        file_plane = "all" if plane.lower() == "all" else plane
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Markers",
            os.path.join(default_dir, f"markers_{file_plane}.marker"),
            "Marker Files (*.marker *.marker.json *.json);;JSON Files (*.json);;All Files (*)",
        )
        if not filename:
            return
        if not filename.lower().endswith((".marker", ".marker.json", ".json")):
            filename += ".marker"
        try:
            world_frame = None
            if hasattr(self.marker_manager, "world_frame_for_plane"):
                try:
                    target_plane = next(iter(states.values())).plane if states else plane
                    world_frame = self.marker_manager.world_frame_for_plane(target_plane)
                except Exception:
                    world_frame = None
            payload = marker_states_to_json(states, plane=file_plane, world_frame=world_frame)
            with open(filename, "w", encoding="utf-8") as handle:
                handle.write(payload)
            self._show_info(f"Saved {len(states)} marker(s) to {os.path.basename(filename)}.")
        except Exception as exc:
            self._show_error(f"Failed to save markers: {exc}")

    def _on_load_json(self) -> None:
        default_dir = os.path.dirname(getattr(self.viewer, "filename_path", "")) or os.getcwd()
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Markers",
            default_dir,
            "Marker Files (*.marker *.marker.json *.json);;All Files (*)",
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            plane = self.marker_manager.import_from_dict(payload)
            self.marker_manager.redraw_plane(plane)
            self._current_plane = plane
            self._refresh_marker_list()
            # self._show_info(f"Loaded markers for plane '{plane.upper()}'.")
        except Exception as exc:
            self._show_error(f"Failed to load markers: {exc}")

    def _on_export_ds9(self) -> None:
        plane = self._choose_plane_for_export("Export to DS9 Region")
        if plane is None:
            return
        states = self._collect_states(plane)
        if not states:
            self._show_info(f"No markers to export for plane '{plane.upper()}'.")
            return
        default_dir = os.path.dirname(getattr(self.viewer, "filename_path", "")) or os.getcwd()
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Markers to DS9",
            os.path.join(default_dir, f"markers_{plane}.reg"),
            "DS9 Region Files (*.reg);;All Files (*)",
        )
        if not path:
            return
        if not path.lower().endswith(".reg"):
            path += ".reg"
        try:
            content = marker_states_to_ds9(states, plane)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(content)
            self._show_info(f"Exported DS9 region file to {os.path.basename(path)}.")
        except Exception as exc:
            self._show_error(f"Failed to export DS9 region file: {exc}")

    # ------------------------------------------------------------------
    # Placement configuration helpers
    def _configure_placement(self) -> None:
        if not self.placement_toggle.isChecked():
            return
        current_plane = self._resolve_current_plane()
        entry = self.TYPE_OPTIONS[self.type_combo.currentIndex()]
        config = dict(entry[1])
        kind = config.get("kind", "symbol")
        kwargs = dict(self._style_configuration_for_kind(kind))
        if kind == "symbol":
            kwargs.setdefault("symbol", config.get("symbol", "o"))
        self.marker_manager.begin_placement(kind, plane=current_plane, continuous=True, **kwargs)
        if hasattr(self.viewer, "set_marker_mode") and not getattr(self.viewer, "marker_mode_enabled", False):
            self.viewer.set_marker_mode(True)

    # ------------------------------------------------------------------
    # Utility helpers
    def _collect_states(self, plane: str) -> Dict[str, MarkerState]:
        try:
            self.marker_manager.refresh_world_coordinates(plane)
        except Exception:
            pass
        markers = self.marker_manager.markers_for_plane(plane)
        states = {marker.marker_id: marker.to_state() for marker in markers}

        # Attach world endpoints for lines so they can be reprojected with correct angle.
        viewer = self.marker_manager.viewer_for_plane(plane)
        format_pix = getattr(viewer, "format_pix", None) if viewer else None
        wcs = getattr(viewer, "wcs", None) if viewer else None
        base_plane_resolver = getattr(self.marker_manager, "_base_plane_for", None)
        base_plane = base_plane_resolver(plane) if callable(base_plane_resolver) else plane
        if format_pix is not None and wcs is not None:
            world_pair_resolver = getattr(self.marker_manager, "_pixel_to_world_pair", None)
            for marker in markers:
                if not isinstance(marker, LineMarker):
                    continue
                state = states.get(marker.marker_id)
                if state is None:
                    continue
                try:
                    start, end = marker._endpoints()
                    meta = dict(state.metadata)
                    if callable(world_pair_resolver):
                        start_world = world_pair_resolver(plane, start)
                        end_world = world_pair_resolver(plane, end)
                    else:
                        start_world = end_world = None
                    if start_world is None:
                        start_world = format_pix.pix_to_wcs(wcs, start[0], start[1], base_plane)
                    if end_world is None:
                        end_world = format_pix.pix_to_wcs(wcs, end[0], end[1], base_plane)
                    wx0, wy0 = start_world
                    wx1, wy1 = end_world
                    meta["pixel_endpoints"] = [(float(start[0]), float(start[1])), (float(end[0]), float(end[1]))]
                    meta["world_endpoints"] = [(float(wx0), float(wy0)), (float(wx1), float(wy1))]
                    state.metadata = meta
                except Exception:
                    continue
        return states

    def _show_info(self, message: str) -> None:
        QMessageBox.information(self, "Markers", message)

    def _show_error(self, message: str) -> None:
        QMessageBox.warning(self, "Markers", message)

    def _pixel_scale_deg(self, plane: str) -> Optional[float]:
        resolver = getattr(self.marker_manager, "_pixel_scale_deg", None)
        if callable(resolver):
            try:
                scale = resolver(plane)
            except Exception:
                scale = None
            if scale:
                return float(scale)
        viewer = self._viewer_for_plane(plane)
        if viewer is None:
            return None
        wcs = getattr(viewer, "wcs", None)
        if wcs is None:
            return None
        axis_indices = self._plane_axis_indices(plane, wcs)
        if not axis_indices:
            axis_indices = (0, 1)
        try:
            scales = proj_plane_pixel_scales(wcs)
            if scales is not None and len(scales) >= 2:
                values = []
                for idx in axis_indices:
                    if idx < 0 or idx >= len(scales):
                        continue
                    val = scales[idx]
                    try:
                        values.append(abs(val.to_value(u.deg)))
                    except AttributeError:
                        try:
                            values.append(abs(float(val)))
                        except Exception:
                            continue
                if values:
                    avg = sum(values) / len(values)
                    if avg > 0:
                        return avg
        except Exception:
            pass
        try:
            cdelt = getattr(getattr(wcs, "wcs", None), "cdelt", None)
            if cdelt is not None and len(cdelt) > 0:
                values = []
                for idx in axis_indices:
                    if idx < 0 or idx >= len(cdelt):
                        continue
                    scale = abs(cdelt[idx])
                    if scale > 0:
                        values.append(float(scale))
                if values:
                    avg = sum(values) / len(values)
                    if avg > 0:
                        return avg
        except Exception:
            pass
        return None

    def _convert_length_to_pixels(self, value: float, unit: str, *, plane: str) -> float:
        if unit == "pixel":
            return float(value)
        scale_deg = self._pixel_scale_deg(plane)
        if not scale_deg:
            return float(value)
        if unit == "deg":
            return float(value) / scale_deg
        if unit == "arcmin":
            return float(value) / (scale_deg * 60.0)
        if unit == "arcsec":
            return float(value) / (scale_deg * 3600.0)
        return float(value)

    def _convert_length_from_pixels(self, value: float, unit: str, *, plane: str) -> float:
        if unit == "pixel":
            return float(value)
        scale_deg = self._pixel_scale_deg(plane)
        if not scale_deg:
            return float(value)
        if unit == "deg":
            return float(value) * scale_deg
        if unit == "arcmin":
            return float(value) * scale_deg * 60.0
        if unit == "arcsec":
            return float(value) * scale_deg * 3600.0
        return float(value)

    def _viewer_for_plane(self, plane: str):
        viewer_lookup = getattr(self.marker_manager, "viewer_for_plane", None)
        viewer = None
        if callable(viewer_lookup):
            try:
                viewer = viewer_lookup(plane)
            except Exception:
                viewer = None
        if viewer is None and getattr(self.viewer, "plane", None) == plane:
            viewer = self.viewer
        if viewer is None:
            has_plane = getattr(self.viewer, "has_marker_plane", None)
            if callable(has_plane):
                try:
                    if has_plane(plane):
                        viewer = self.viewer
                except Exception:
                    viewer = None
        return viewer or self.viewer

    # ------------------------------------------------------------------
    # Qt overrides
    def closeEvent(self, event) -> None:
        if self.placement_toggle.isChecked():
            self.placement_toggle.setChecked(False)
        super().closeEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._on_deselect_markers()
