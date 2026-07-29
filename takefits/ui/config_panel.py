import copy

from PySide6.QtWidgets import QWidget, QGridLayout, QLineEdit, QPushButton, QLabel, QMessageBox, QTabWidget, QHBoxLayout, QVBoxLayout, QScrollArea, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QCompleter, QRadioButton, QButtonGroup, QGroupBox
from PySide6.QtCore import Qt
from PySide6.QtGui import QFontDatabase
import matplotlib as mpl
from matplotlib import colormaps
import yaml
import os
import math

from takefits.core.config import axes_positions_are_valid
from takefits.ui.widget_sizing import fit_button_to_text



class CircularSpinBox(QSpinBox):
    def __init__(self):
        super().__init__()
        self.setRange(-180, 180)
        self.setSingleStep(1)
        self.setValue(0)
        self.setWrapping(True)
        self.setKeyboardTracking(False)
        self.valueChanged.connect(self._on_value_changed)
        self._last_value = 0

    def _on_value_changed(self, value):
        if abs(value - self._last_value) > 180:
            if value > self._last_value:
                self.setValue(value - 360)
            else:
                self.setValue(value + 360)
        self._last_value = self.value()

class ConfigPanel(QWidget):
    CLICK_LINESTYLE_OPTIONS = (
        ("solid", "-"),
        ("dashed", "--"),
        ("dotted", ":"),
        ("dash-dot", "-."),
    )
    GRID_LINESTYLE_OPTIONS = (
        ("Solid", "solid"),
        ("Dashed", "dashed"),
        ("Dotted", "dotted"),
        ("Dash-dot", "dashdot"),
    )
    GRID_STYLE_KEYS = frozenset(
        {
            "grid_color",
            "grid_alpha",
            "grid_linestyle",
            "grid_linewidth",
            "grid_overlay_color",
            "grid_overlay_linestyle",
            "grid_overlay_label_color",
            "grid_overlay_show_lines",
            "grid_overlay_show_ticklabels",
            "grid_overlay_axislabel_placement",
            "grid_overlay_longitude_axislabel_pad",
            "grid_overlay_latitude_axislabel_pad",
        }
    )
    AUTO_COLORBAR_GEOMETRY_KEYS = frozenset(
        {
            "cbar_pos_x",
            "cbar_pos_y",
            "cbar_width",
            "cbar_height",
        }
    )

    def __init__(self, config_manager, fits_viewer):
        super().__init__()
        self.config_manager = config_manager
        self.fits_viewer = fits_viewer
        self._session_authoritative_updates = {}
        self.initUI()

    @staticmethod
    def _normalize_colorbar_length_mode(mode):
        value = str(mode or "").strip().lower()
        if value == "px":
            return "px"
        return "ratio"

    @staticmethod
    def _normalize_colorbar_placement(value):
        return str(value or "").strip().lower()

    @staticmethod
    def _normalize_click_linestyle(value):
        token = str(value or "").strip().lower()
        mapping = {
            "-": "-",
            "--": "--",
            ":": ":",
            "-.": "-.",
            "solid": "-",
            "dashed": "--",
            "dotted": ":",
            "dash-dot": "-.",
            "dashdot": "-.",
            "dash dot": "-.",
        }
        return mapping.get(token, "-")

    def _set_click_linestyle_combo(self, value):
        token = self._normalize_click_linestyle(value)
        index = self.click_linestyle_input.findData(token)
        if index < 0:
            index = 0
        self.click_linestyle_input.setCurrentIndex(index)

    @staticmethod
    def _normalize_grid_linestyle(value, default="solid"):
        token = str(value or "").strip().lower()
        mapping = {
            "-": "solid",
            "--": "dashed",
            ":": "dotted",
            "-.": "dashdot",
            "solid": "solid",
            "dashed": "dashed",
            "dotted": "dotted",
            "dashdot": "dashdot",
            "dash-dot": "dashdot",
            "dash dot": "dashdot",
        }
        return mapping.get(token, default)

    def _create_grid_linestyle_combo(self, value, default="solid"):
        combo = QComboBox()
        for label, token in self.GRID_LINESTYLE_OPTIONS:
            combo.addItem(label, token)
        self._set_grid_linestyle_combo(combo, value, default)
        return combo

    def _set_grid_linestyle_combo(self, combo, value, default="solid"):
        token = self._normalize_grid_linestyle(value, default)
        index = combo.findData(token)
        combo.setCurrentIndex(index if index >= 0 else 0)

    def _grid_linestyle_value(self, combo, default="solid"):
        return self._normalize_grid_linestyle(
            combo.currentData() or combo.currentText(),
            default,
        )

    def _on_grid_advanced_toggled(self, checked):
        self.grid_advanced_widget.setVisible(bool(checked))
        self.grid_advanced_button.setText(
            "Advanced ▾" if checked else "Advanced ▸"
        )

    def _on_grid_label_color_policy_changed(self, *_args):
        is_custom = self.grid_overlay_label_color_policy_input.currentData() == "custom"
        self.grid_overlay_label_color_input.setEnabled(is_custom)

    def _on_grid_axislabel_gap_mode_changed(self, *_args):
        is_custom = self.grid_overlay_axislabel_gap_mode_input.currentData() == "custom"
        self.grid_overlay_longitude_axislabel_pad_input.setEnabled(is_custom)
        self.grid_overlay_latitude_axislabel_pad_input.setEnabled(is_custom)

    def _load_grid_controls(self, source_config):
        config = source_config if isinstance(source_config, dict) else {}
        self.grid_color_input.setCurrentText(
            str(config.get("grid_color", "white"))
        )
        self._set_grid_linestyle_combo(
            self.grid_linestyle_input,
            config.get("grid_linestyle", "solid"),
            "solid",
        )
        self.grid_overlay_color_input.setCurrentText(
            str(config.get("grid_overlay_color", "#00ff66"))
        )
        self._set_grid_linestyle_combo(
            self.grid_overlay_linestyle_input,
            config.get("grid_overlay_linestyle", "dashed"),
            "dashed",
        )
        placement = str(
            config.get("grid_overlay_axislabel_placement", "inside")
            or "inside"
        ).strip().lower()
        if placement not in {"outside", "inside", "hidden"}:
            placement = "inside"
        placement_index = self.grid_overlay_axislabel_placement_input.findData(
            placement
        )
        self.grid_overlay_axislabel_placement_input.setCurrentIndex(
            placement_index if placement_index >= 0 else 0
        )

        self.grid_linewidth_input.setValue(
            float(config.get("grid_linewidth", 0.5))
        )
        self.grid_alpha_input.setValue(float(config.get("grid_alpha", 0.5)))
        label_color = str(
            config.get("grid_overlay_label_color", "auto") or "auto"
        ).strip()
        normalized_label_color = label_color.lower()
        if normalized_label_color in {"auto", "same"}:
            policy = normalized_label_color
        else:
            policy = "custom"
            self.grid_overlay_label_color_input.setCurrentText(label_color)
        policy_index = self.grid_overlay_label_color_policy_input.findData(
            policy
        )
        self.grid_overlay_label_color_policy_input.setCurrentIndex(
            policy_index if policy_index >= 0 else 0
        )
        if policy != "custom":
            self.grid_overlay_label_color_input.setCurrentText(
                str(config.get("grid_overlay_color", "#00ff66"))
            )
        self.grid_overlay_show_lines_checkbox.setChecked(
            bool(config.get("grid_overlay_show_lines", True))
        )
        self.grid_overlay_show_ticklabels_checkbox.setChecked(
            bool(config.get("grid_overlay_show_ticklabels", True))
        )

        longitude_pad = config.get(
            "grid_overlay_longitude_axislabel_pad",
            None,
        )
        latitude_pad = config.get(
            "grid_overlay_latitude_axislabel_pad",
            None,
        )
        gap_mode = (
            "auto"
            if longitude_pad is None and latitude_pad is None
            else "custom"
        )
        gap_index = self.grid_overlay_axislabel_gap_mode_input.findData(gap_mode)
        self.grid_overlay_axislabel_gap_mode_input.setCurrentIndex(
            gap_index if gap_index >= 0 else 0
        )
        if longitude_pad is not None:
            self.grid_overlay_longitude_axislabel_pad_input.setValue(
                abs(float(longitude_pad))
            )
        if latitude_pad is not None:
            self.grid_overlay_latitude_axislabel_pad_input.setValue(
                abs(float(latitude_pad))
            )
        self._on_grid_label_color_policy_changed()
        self._on_grid_axislabel_gap_mode_changed()

    def _grid_config_updates(self):
        label_policy = (
            self.grid_overlay_label_color_policy_input.currentData() or "auto"
        )
        label_color = (
            self.grid_overlay_label_color_input.currentText()
            if label_policy == "custom"
            else label_policy
        )
        custom_gap = (
            self.grid_overlay_axislabel_gap_mode_input.currentData() == "custom"
        )
        return [
            ("grid_color", self.grid_color_input.currentText()),
            (
                "grid_linestyle",
                self._grid_linestyle_value(
                    self.grid_linestyle_input,
                    "solid",
                ),
            ),
            ("grid_overlay_color", self.grid_overlay_color_input.currentText()),
            (
                "grid_overlay_linestyle",
                self._grid_linestyle_value(
                    self.grid_overlay_linestyle_input,
                    "dashed",
                ),
            ),
            (
                "grid_overlay_axislabel_placement",
                self.grid_overlay_axislabel_placement_input.currentData()
                or "inside",
            ),
            ("grid_linewidth", self.grid_linewidth_input.value()),
            ("grid_alpha", self.grid_alpha_input.value()),
            ("grid_overlay_label_color", label_color),
            (
                "grid_overlay_show_lines",
                self.grid_overlay_show_lines_checkbox.isChecked(),
            ),
            (
                "grid_overlay_show_ticklabels",
                self.grid_overlay_show_ticklabels_checkbox.isChecked(),
            ),
            (
                "grid_overlay_longitude_axislabel_pad",
                self.grid_overlay_longitude_axislabel_pad_input.value()
                if custom_gap
                else None,
            ),
            (
                "grid_overlay_latitude_axislabel_pad",
                self.grid_overlay_latitude_axislabel_pad_input.value()
                if custom_gap
                else None,
            ),
        ]

    def _is_inside_colorbar_placement(self, value):
        placement = self._normalize_colorbar_placement(value)
        return placement in {"inside-right", "inside-left", "inside-top", "inside-bottom"}

    @staticmethod
    def _tick_side_value_from_flags(axis, a_enabled, b_enabled):
        axis_name = str(axis or "").strip().lower()
        if axis_name == "y":
            if a_enabled and b_enabled:
                return "both"
            if a_enabled:
                return "left"
            if b_enabled:
                return "right"
            return "none"
        if a_enabled and b_enabled:
            return "both"
        if a_enabled:
            return "top"
        if b_enabled:
            return "bottom"
        return "none"

    @staticmethod
    def _tick_side_flags_from_value(axis, value):
        axis_name = str(axis or "").strip().lower()
        token = str(value or "").strip().lower()
        if axis_name == "y":
            if token == "left":
                return True, False
            if token == "right":
                return False, True
            if token == "none":
                return False, False
            return True, True
        if token == "top":
            return True, False
        if token == "bottom":
            return False, True
        if token == "none":
            return False, False
        return True, True

    def _load_colorbar_tick_side_comboboxes(self, source_config):
        config = source_config if isinstance(source_config, dict) else {}
        y_value = self._tick_side_value_from_flags(
            "y",
            bool(config.get('colorbar_tick_left', False)),
            bool(config.get('colorbar_tick_right', True)),
        )
        x_value = self._tick_side_value_from_flags(
            "x",
            bool(config.get('colorbar_tick_top', False)),
            bool(config.get('colorbar_tick_bottom', True)),
        )
        self.cbar_tick_y_side_combo.setCurrentText(y_value)
        self.cbar_tick_x_side_combo.setCurrentText(x_value)
        self.cbar_tick_y_label_side_combo.setCurrentText(
            "left" if bool(config.get('colorbar_tick_labelleft', False)) else "right"
        )
        self.cbar_tick_x_label_side_combo.setCurrentText(
            "top" if bool(config.get('colorbar_tick_labeltop', False)) else "bottom"
        )

    def _read_colorbar_tick_side_values(self):
        tick_left, tick_right = self._tick_side_flags_from_value(
            "y", self.cbar_tick_y_side_combo.currentText()
        )
        tick_top, tick_bottom = self._tick_side_flags_from_value(
            "x", self.cbar_tick_x_side_combo.currentText()
        )
        label_left = str(self.cbar_tick_y_label_side_combo.currentText()).strip().lower() == "left"
        label_top = str(self.cbar_tick_x_label_side_combo.currentText()).strip().lower() == "top"
        return {
            "colorbar_tick_left": tick_left,
            "colorbar_tick_right": tick_right,
            "colorbar_tick_top": tick_top,
            "colorbar_tick_bottom": tick_bottom,
            "colorbar_tick_labelleft": label_left,
            "colorbar_tick_labeltop": label_top,
        }

    def _on_colorbar_placement_changed(self, value):
        placement = self._normalize_colorbar_placement(value)
        if not self._is_inside_colorbar_placement(placement):
            return

        # Inside placement layout preset.
        self.cbar_gap_x_px_input.setValue(12.0)
        self.cbar_gap_y_px_input.setValue(12.0)
        self.cbar_length_mode_combo.setCurrentText("ratio")
        self.cbar_length_value_input.setValue(0.4)
        self.cbar_align_combo.setCurrentText("end")

        # Inside placement preset: ticks face inward, high-contrast tick colors.
        self.cbar_tick_direction_combo.setCurrentText("in")
        self.cbar_tick_color_input.setCurrentText("white")
        self.cbar_tick_label_color_input.setCurrentText("white")

        if placement in {"inside-right", "inside-left"}:
            self.cbar_tick_y_side_combo.setCurrentText("both")
        else:
            self.cbar_tick_x_side_combo.setCurrentText("both")

        if placement == "inside-top":
            self.cbar_tick_x_label_side_combo.setCurrentText("bottom")
        elif placement == "inside-bottom":
            self.cbar_tick_x_label_side_combo.setCurrentText("top")
        elif placement == "inside-right":
            self.cbar_tick_y_label_side_combo.setCurrentText("left")
        elif placement == "inside-left":
            self.cbar_tick_y_label_side_combo.setCurrentText("right")

    def _update_colorbar_inside_preset_enabled(self, value=None):
        if value is None:
            value = self.cbar_placement_combo.currentText()
        button = getattr(self, "cbar_inside_preset_button", None)
        if button is not None:
            button.setEnabled(self._is_inside_colorbar_placement(value))

    def initUI(self):
        layout = QVBoxLayout()


        self.tabs = QTabWidget()

        # Add tabs for different configuration categories
        self.tabs.addTab(self.create_general_tab(), "General")
        self.tabs.addTab(self.create_display_tab(), "Display")
        self.tabs.addTab(self.create_colorbar_tab(), "Colorbar")
        self.tabs.addTab(self.create_ticks_tab(), "Ticks")
        self.tabs.addTab(self.create_click_tab(), "Click")

        layout.addWidget(self.tabs)


        button_layout = QHBoxLayout()
        

        self.apply_button = QPushButton('Apply', self)
        self.apply_button.clicked.connect(self.apply_changes)
        fit_button_to_text(self.apply_button, minimum_width=50)
        self.apply_button.setAutoDefault(True)
        self.apply_button.setDefault(True)
        button_layout.addWidget(self.apply_button)

        self.reset_button = QPushButton('Reset', self)
        self.reset_button.clicked.connect(self.reset_to_loaded_config)
        fit_button_to_text(self.reset_button, minimum_width=50)
        button_layout.addWidget(self.reset_button)
        
        
        button_layout.addSpacing(50)  
        self.default_button = QPushButton('Default', self)
        self.default_button.clicked.connect(self.reset_to_default)
        fit_button_to_text(self.default_button, minimum_width=80)
        button_layout.addWidget(self.default_button)

        self.save_button = QPushButton('Save', self)
        self.save_button.clicked.connect(self.save_config)
        fit_button_to_text(self.save_button, minimum_width=80)
        button_layout.addWidget(self.save_button)
        

        layout.addLayout(button_layout)

        self.setLayout(layout)
        self.setWindowTitle('Configuration Settings')
        # Compare future Apply operations with the values the form could
        # actually represent when it opened. This prevents untouched
        # QDoubleSpinBox rounding (and None-to-empty text normalization) from
        # turning a grid-only edit into an unrelated full viewer reload.
        self._form_baseline_values = dict(
            self._collect_preference_updates()
        )

    def create_general_tab(self):
        """Create the general settings tab."""
        page_layout = QVBoxLayout()

        defaults_group = QGroupBox("Defaults")
        defaults_layout = QGridLayout()

        self.colorscale_input = QComboBox()
        self.colorscale_input.setEditable(True)
        colormap_names = [name for name in colormaps.keys() if not name.endswith('_r')]
        self.colorscale_input.addItems(sorted(colormap_names))
        self.colorscale_input.setFixedWidth(120)
        self.colorscale_input.setCurrentText(self.config_manager.config.get('colorscale', 'Rainbow'))
        defaults_layout.addWidget(QLabel('Default Colorscale:'), 0, 0)
        defaults_layout.addWidget(self.colorscale_input, 0, 1)

        self.startup_show_subwindow1_checkbox = QCheckBox("Show SubWindow1 (XZ)")
        self.startup_show_subwindow1_checkbox.setChecked(
            bool(self.config_manager.config.get('startup_show_subwindow1', True))
        )
        defaults_layout.addWidget(self.startup_show_subwindow1_checkbox, 1, 0, 1, 2)

        self.startup_show_subwindow2_checkbox = QCheckBox("Show SubWindow2 (ZY)")
        self.startup_show_subwindow2_checkbox.setChecked(
            bool(self.config_manager.config.get('startup_show_subwindow2', False))
        )
        defaults_layout.addWidget(self.startup_show_subwindow2_checkbox, 2, 0, 1, 2)

        # Range file is managed in Range Control panel; keep this widget for
        # backward compatibility with apply/reset handlers.
        self.range_file_input = QLineEdit(self.config_manager.config.get('range_file', 'takefits.range'))
        defaults_group.setLayout(defaults_layout)

        viewer_data = getattr(getattr(self, "fits_viewer", None), "data", None)
        has_subwindows = bool(getattr(viewer_data, "ndim", 0) > 2)
        if not has_subwindows:
            self.startup_show_subwindow1_checkbox.setEnabled(False)
            self.startup_show_subwindow2_checkbox.setEnabled(False)

        window_group = QGroupBox("Figure Window")
        window_layout = QGridLayout()
        self.figure_pos_x_input = QSpinBox()
        self.figure_pos_x_input.setRange(0, 9999)
        self.figure_pos_x_input.setValue(self.config_manager.config.get('figure_pos_x', 100))
        self.figure_pos_x_input.setFixedWidth(100)
        window_layout.addWidget(QLabel('Figure X Position:'), 0, 0)
        window_layout.addWidget(self.figure_pos_x_input, 0, 1)

        self.figure_pos_y_input = QSpinBox()
        self.figure_pos_y_input.setRange(0, 9999)
        self.figure_pos_y_input.setValue(self.config_manager.config.get('figure_pos_y', 100))
        self.figure_pos_y_input.setFixedWidth(100)
        window_layout.addWidget(QLabel('Figure Y Position:'), 1, 0)
        window_layout.addWidget(self.figure_pos_y_input, 1, 1)

        self.figure_width_input = QSpinBox()
        self.figure_width_input.setRange(579, 9999)
        self.figure_width_input.setValue(self.config_manager.config.get('figure_width', 640))
        self.figure_width_input.setFixedWidth(100)
        window_layout.addWidget(QLabel('Figure Width:'), 2, 0)
        window_layout.addWidget(self.figure_width_input, 2, 1)

        self.figure_height_input = QSpinBox()
        self.figure_height_input.setRange(100, 9999)
        self.figure_height_input.setValue(self.config_manager.config.get('figure_height', 640))
        self.figure_height_input.setFixedWidth(100)
        window_layout.addWidget(QLabel('Figure Height:'), 3, 0)
        window_layout.addWidget(self.figure_height_input, 3, 1)
        window_group.setLayout(window_layout)

        coord_group = QGroupBox("Coordinates")
        coord_layout = QGridLayout()

        coord_format_layout = QHBoxLayout()
        coord_format_layout.setContentsMargins(0, 0, 0, 0)
        coord_format_layout.setSpacing(12)
        self.coord_format_group = QButtonGroup(self)
        self.decimal_radio = QRadioButton('Decimal')
        self.sexagesimal_radio = QRadioButton('Sexagesimal')
        self.coord_format_group.addButton(self.decimal_radio)
        self.coord_format_group.addButton(self.sexagesimal_radio)
        coord_format_layout.addWidget(self.decimal_radio)
        coord_format_layout.addWidget(self.sexagesimal_radio)
        coord_format_layout.addStretch()
        self.set_coordinate_format(self.config_manager.config.get('decimal', True))
        coord_layout.addWidget(QLabel('Coordinate Format:'), 0, 0)
        coord_layout.addLayout(coord_format_layout, 0, 1)

        self.number_decimals_input = QSpinBox()
        self.number_decimals_input.setValue(self.config_manager.config.get('number_decimals', 6))
        self.number_decimals_input.setFixedWidth(80)
        self.auto_precision_checkbox = QCheckBox("Auto")
        self.auto_precision_checkbox.setChecked(
            bool(self.config_manager.config.get('auto_precision_digits', True))
        )
        self.auto_precision_checkbox.toggled.connect(self._on_auto_precision_toggled)
        coord_layout.addWidget(QLabel('Precision digits:'), 1, 0)
        precision_layout = QHBoxLayout()
        precision_layout.setContentsMargins(0, 0, 0, 0)
        precision_layout.setSpacing(8)
        precision_layout.addWidget(self.number_decimals_input)
        precision_layout.addWidget(self.auto_precision_checkbox)
        precision_layout.addStretch()
        coord_layout.addLayout(precision_layout, 1, 1)
        self._on_auto_precision_toggled(self.auto_precision_checkbox.isChecked())

        self.wrap_angle_input = QSpinBox()
        self.wrap_angle_input.setSingleStep(180)
        self.wrap_angle_input.setRange(180, 360)
        self.wrap_angle_input.setValue(self.config_manager.config.get('coord_wrap', 180))
        self.wrap_angle_input.setFixedWidth(120)
        coord_layout.addWidget(QLabel('Wrap Angle:'), 2, 0)
        coord_layout.addWidget(self.wrap_angle_input, 2, 1)
        coord_group.setLayout(coord_layout)

        nav_group = QGroupBox("Navigation")
        nav_layout = QGridLayout()
        self.scrollspeed_input = QDoubleSpinBox()
        self.scrollspeed_input.setValue(self.config_manager.config.get('scrollspeed', 0.1))
        self.scrollspeed_input.setSingleStep(0.1)
        self.scrollspeed_input.setFixedWidth(80)
        self.scrollspeed_input.setAlignment(Qt.AlignmentFlag.AlignRight)
        nav_layout.addWidget(QLabel('Scroll Speed:'), 0, 0)
        nav_layout.addWidget(self.scrollspeed_input, 0, 1)
        self.invert_wheel_direction_checkbox = QCheckBox("Invert Wheel Direction")
        self.invert_wheel_direction_checkbox.setChecked(
            bool(self.config_manager.config.get('invert_wheel_direction', False))
        )
        nav_layout.addWidget(self.invert_wheel_direction_checkbox, 1, 0, 1, 2)
        nav_group.setLayout(nav_layout)

        page_layout.addWidget(defaults_group)
        page_layout.addWidget(window_group)
        page_layout.addWidget(coord_group)
        page_layout.addWidget(nav_group)
        page_layout.addStretch(1)

        general_widget = QWidget()
        general_widget.setLayout(page_layout)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(general_widget)
        return scroll_area

    def create_display_tab(self):
        """Create the display settings tab."""
        page_layout = QVBoxLayout()

        color_group = QGroupBox("Canvas Colors")
        color_layout = QGridLayout()
        self.fig_background_color_input = self.create_color_combobox(self.config_manager.config.get('fig_background_color', '#ececec'))
        self.fig_background_color_input.setFixedWidth(100)
        color_layout.addWidget(QLabel('Figure Background Color:'), 0, 0)
        color_layout.addWidget(self.fig_background_color_input, 0, 1)

        self.ax_background_color_input = self.create_color_combobox(self.config_manager.config.get('ax_background_color', 'white'))
        self.ax_background_color_input.setFixedWidth(100)
        color_layout.addWidget(QLabel('Axes Background Color:'), 1, 0)
        color_layout.addWidget(self.ax_background_color_input, 1, 1)

        self.bad_color_input = self.create_color_combobox(self.config_manager.config.get('bad_color', 'black'))
        self.bad_color_input.setFixedWidth(100)
        color_layout.addWidget(QLabel('Bad (NaN) Color:'), 2, 0)
        color_layout.addWidget(self.bad_color_input, 2, 1)
        color_group.setLayout(color_layout)

        axis_group = QGroupBox("Axes And Labels")
        axis_layout = QGridLayout()
        self.axislabel_fontsize_input = QSpinBox()
        self.axislabel_fontsize_input.setValue(self.config_manager.config.get('axislabel_fontsize', 14))
        self.axislabel_fontsize_input.setRange(1, 100)
        self.axislabel_fontsize_input.setFixedWidth(100)
        axis_layout.addWidget(QLabel('Axis Label Font Size:'), 0, 0)
        axis_layout.addWidget(self.axislabel_fontsize_input, 0, 1)

        self.axislabel_fontfamily_input = self.create_font_combobox(self.config_manager.config.get('axislabel_fontfamily', 'DejaVu Sans'))
        self.axislabel_fontfamily_input.setFixedWidth(100)
        axis_layout.addWidget(QLabel('Axis Label Font Family:'), 1, 0)
        axis_layout.addWidget(self.axislabel_fontfamily_input, 1, 1)

        self.axislabel_color_input = self.create_color_combobox(self.config_manager.config.get('axislabel_color', 'black'))
        self.axislabel_color_input.setFixedWidth(100)
        axis_layout.addWidget(QLabel('Axis Label Color:'), 2, 0)
        axis_layout.addWidget(self.axislabel_color_input, 2, 1)

        self.axis_left_spinbox = QDoubleSpinBox()
        self.axis_left_spinbox.setRange(0.0, 2.0)
        self.axis_left_spinbox.setSingleStep(0.01)
        self.axis_left_spinbox.setValue(self.config_manager.config.get('ax_pos_l', 0.15))
        axis_layout.addWidget(QLabel("Axis Left Position"), 3, 0)
        axis_layout.addWidget(self.axis_left_spinbox, 3, 1)

        self.axis_right_spinbox = QDoubleSpinBox()
        self.axis_right_spinbox.setRange(0.0, 2.0)
        self.axis_right_spinbox.setSingleStep(0.01)
        self.axis_right_spinbox.setValue(self.config_manager.config.get('ax_pos_r', 0.85))
        axis_layout.addWidget(QLabel("Axis Right Position"), 4, 0)
        axis_layout.addWidget(self.axis_right_spinbox, 4, 1)

        self.axis_top_spinbox = QDoubleSpinBox()
        self.axis_top_spinbox.setRange(0.0, 2.0)
        self.axis_top_spinbox.setSingleStep(0.01)
        self.axis_top_spinbox.setValue(self.config_manager.config.get('ax_pos_t', 0.9))
        axis_layout.addWidget(QLabel("Axis Top Position"), 5, 0)
        axis_layout.addWidget(self.axis_top_spinbox, 5, 1)

        self.axis_bottom_spinbox = QDoubleSpinBox()
        self.axis_bottom_spinbox.setRange(0.0, 2.0)
        self.axis_bottom_spinbox.setSingleStep(0.01)
        self.axis_bottom_spinbox.setValue(self.config_manager.config.get('ax_pos_b', 0.12))
        axis_layout.addWidget(QLabel("Axis Bottom Position"), 6, 0)
        axis_layout.addWidget(self.axis_bottom_spinbox, 6, 1)
        axis_group.setLayout(axis_layout)

        grid_group = QGroupBox("Coordinate Grid")
        grid_layout = QGridLayout()
        self.grid_color_input = self.create_color_combobox(
            self.config_manager.config.get('grid_color', 'white')
        )
        self.grid_color_input.setFixedWidth(100)
        grid_layout.addWidget(QLabel("Native Line Color:"), 0, 0)
        grid_layout.addWidget(self.grid_color_input, 0, 1)

        self.grid_linestyle_input = self._create_grid_linestyle_combo(
            self.config_manager.config.get("grid_linestyle", "solid"),
            "solid",
        )
        self.grid_linestyle_input.setFixedWidth(100)
        grid_layout.addWidget(QLabel("Native Line Style:"), 1, 0)
        grid_layout.addWidget(self.grid_linestyle_input, 1, 1)

        self.grid_overlay_color_input = self.create_color_combobox(
            self.config_manager.config.get('grid_overlay_color', '#00ff66')
        )
        self.grid_overlay_color_input.setFixedWidth(100)
        grid_layout.addWidget(QLabel("Overlay Line Color:"), 2, 0)
        grid_layout.addWidget(self.grid_overlay_color_input, 2, 1)

        self.grid_overlay_linestyle_input = self._create_grid_linestyle_combo(
            self.config_manager.config.get(
                "grid_overlay_linestyle",
                "dashed",
            ),
            "dashed",
        )
        self.grid_overlay_linestyle_input.setFixedWidth(100)
        grid_layout.addWidget(QLabel("Overlay Line Style:"), 3, 0)
        grid_layout.addWidget(self.grid_overlay_linestyle_input, 3, 1)

        self.grid_overlay_axislabel_placement_input = QComboBox()
        self.grid_overlay_axislabel_placement_input.addItem("Inside", "inside")
        self.grid_overlay_axislabel_placement_input.addItem("Outside", "outside")
        self.grid_overlay_axislabel_placement_input.addItem("Hidden", "hidden")
        self.grid_overlay_axislabel_placement_input.setFixedWidth(100)
        grid_layout.addWidget(QLabel("Overlay Axis Titles:"), 4, 0)
        grid_layout.addWidget(
            self.grid_overlay_axislabel_placement_input,
            4,
            1,
        )

        self.grid_advanced_button = QPushButton("Advanced ▸")
        self.grid_advanced_button.setCheckable(True)
        self.grid_advanced_button.setChecked(False)
        fit_button_to_text(self.grid_advanced_button, minimum_width=100)
        grid_layout.addWidget(self.grid_advanced_button, 5, 0, 1, 2)

        self.grid_advanced_widget = QWidget()
        advanced_layout = QGridLayout()
        advanced_layout.setContentsMargins(0, 4, 0, 0)

        self.grid_linewidth_input = QDoubleSpinBox()
        self.grid_linewidth_input.setRange(0.0, 20.0)
        self.grid_linewidth_input.setDecimals(2)
        self.grid_linewidth_input.setSingleStep(0.1)
        self.grid_linewidth_input.setFixedWidth(100)
        advanced_layout.addWidget(QLabel("Shared Grid Line Width:"), 0, 0)
        advanced_layout.addWidget(self.grid_linewidth_input, 0, 1)

        self.grid_alpha_input = QDoubleSpinBox()
        self.grid_alpha_input.setRange(0.0, 1.0)
        self.grid_alpha_input.setDecimals(2)
        self.grid_alpha_input.setSingleStep(0.05)
        self.grid_alpha_input.setFixedWidth(100)
        advanced_layout.addWidget(QLabel("Shared Grid Line Opacity:"), 1, 0)
        advanced_layout.addWidget(self.grid_alpha_input, 1, 1)

        self.grid_overlay_label_color_policy_input = QComboBox()
        self.grid_overlay_label_color_policy_input.addItem("Auto contrast", "auto")
        self.grid_overlay_label_color_policy_input.addItem("Same as line", "same")
        self.grid_overlay_label_color_policy_input.addItem("Custom", "custom")
        self.grid_overlay_label_color_policy_input.setFixedWidth(120)
        advanced_layout.addWidget(QLabel("Overlay Label Color:"), 2, 0)
        advanced_layout.addWidget(
            self.grid_overlay_label_color_policy_input,
            2,
            1,
        )

        self.grid_overlay_label_color_input = self.create_color_combobox(
            self.config_manager.config.get(
                "grid_overlay_label_color",
                self.config_manager.config.get(
                    "grid_overlay_color",
                    "#00ff66",
                ),
            )
        )
        self.grid_overlay_label_color_input.setFixedWidth(120)
        advanced_layout.addWidget(QLabel("Custom Label Color:"), 3, 0)
        advanced_layout.addWidget(
            self.grid_overlay_label_color_input,
            3,
            1,
        )

        self.grid_overlay_show_lines_checkbox = QCheckBox(
            "Show overlay grid lines"
        )
        advanced_layout.addWidget(
            self.grid_overlay_show_lines_checkbox,
            4,
            0,
            1,
            2,
        )
        self.grid_overlay_show_ticklabels_checkbox = QCheckBox(
            "Show overlay ticks and numeric labels"
        )
        advanced_layout.addWidget(
            self.grid_overlay_show_ticklabels_checkbox,
            5,
            0,
            1,
            2,
        )

        self.grid_overlay_axislabel_gap_mode_input = QComboBox()
        self.grid_overlay_axislabel_gap_mode_input.addItem("Auto", "auto")
        self.grid_overlay_axislabel_gap_mode_input.addItem("Custom", "custom")
        self.grid_overlay_axislabel_gap_mode_input.setFixedWidth(100)
        advanced_layout.addWidget(QLabel("Axis Title Gap:"), 6, 0)
        advanced_layout.addWidget(
            self.grid_overlay_axislabel_gap_mode_input,
            6,
            1,
        )

        self.grid_overlay_longitude_axislabel_pad_input = QDoubleSpinBox()
        self.grid_overlay_longitude_axislabel_pad_input.setRange(0.0, 30.0)
        self.grid_overlay_longitude_axislabel_pad_input.setSingleStep(0.5)
        self.grid_overlay_longitude_axislabel_pad_input.setValue(2.0)
        self.grid_overlay_longitude_axislabel_pad_input.setFixedWidth(100)
        self.grid_overlay_longitude_axislabel_pad_input.setToolTip(
            "Gap for the horizontal (top) overlay-axis title."
        )
        advanced_layout.addWidget(QLabel("Top Title Gap:"), 7, 0)
        advanced_layout.addWidget(
            self.grid_overlay_longitude_axislabel_pad_input,
            7,
            1,
        )

        self.grid_overlay_latitude_axislabel_pad_input = QDoubleSpinBox()
        self.grid_overlay_latitude_axislabel_pad_input.setRange(0.0, 30.0)
        self.grid_overlay_latitude_axislabel_pad_input.setSingleStep(0.5)
        self.grid_overlay_latitude_axislabel_pad_input.setValue(5.0)
        self.grid_overlay_latitude_axislabel_pad_input.setFixedWidth(100)
        self.grid_overlay_latitude_axislabel_pad_input.setToolTip(
            "Gap for the vertical (right) overlay-axis title."
        )
        advanced_layout.addWidget(QLabel("Right Title Gap:"), 8, 0)
        advanced_layout.addWidget(
            self.grid_overlay_latitude_axislabel_pad_input,
            8,
            1,
        )
        advanced_layout.setColumnStretch(0, 1)
        advanced_layout.setColumnStretch(1, 1)
        self.grid_advanced_widget.setLayout(advanced_layout)
        grid_layout.addWidget(self.grid_advanced_widget, 6, 0, 1, 2)

        self.grid_advanced_button.toggled.connect(
            self._on_grid_advanced_toggled
        )
        self.grid_overlay_label_color_policy_input.currentIndexChanged.connect(
            self._on_grid_label_color_policy_changed
        )
        self.grid_overlay_axislabel_gap_mode_input.currentIndexChanged.connect(
            self._on_grid_axislabel_gap_mode_changed
        )
        self._load_grid_controls(self.config_manager.config)
        self._on_grid_advanced_toggled(False)
        grid_layout.setColumnStretch(0, 1)
        grid_layout.setColumnStretch(1, 1)
        grid_group.setLayout(grid_layout)

        beam_group = QGroupBox("Beam")
        beam_layout = QGridLayout()
        self.beam_facecolor_input = self.create_color_combobox(self.config_manager.config.get('beam_facecolor', 'white'))
        self.beam_facecolor_input.setFixedWidth(100)
        beam_layout.addWidget(QLabel('Beam FaceColor:'), 0, 0)
        beam_layout.addWidget(self.beam_facecolor_input, 0, 1)

        self.beam_edgecolor_input = self.create_color_combobox(self.config_manager.config.get('beam_edgecolor', 'None'))
        self.beam_edgecolor_input.setFixedWidth(100)
        beam_layout.addWidget(QLabel('Beam EdgeColor:'), 1, 0)
        beam_layout.addWidget(self.beam_edgecolor_input, 1, 1)

        self.beam_linewidth_input = QDoubleSpinBox()
        self.beam_linewidth_input.setValue(self.config_manager.config.get('beam_linewidth', 0.0))
        self.beam_linewidth_input.setSingleStep(0.5)
        beam_layout.addWidget(QLabel('Beam Line Width:'), 2, 0)
        beam_layout.addWidget(self.beam_linewidth_input, 2, 1)

        self.beam_pos_x_spinbox = QDoubleSpinBox()
        self.beam_pos_x_spinbox.setRange(0.0, 1.0)
        self.beam_pos_x_spinbox.setSingleStep(0.01)
        self.beam_pos_x_spinbox.setValue(self.config_manager.config.get('beam_pos_x', 0.10))
        beam_layout.addWidget(QLabel("Beam Position X"), 3, 0)
        beam_layout.addWidget(self.beam_pos_x_spinbox, 3, 1)

        self.beam_pos_y_spinbox = QDoubleSpinBox()
        self.beam_pos_y_spinbox.setRange(0.0, 1.0)
        self.beam_pos_y_spinbox.setSingleStep(0.01)
        self.beam_pos_y_spinbox.setValue(self.config_manager.config.get('beam_pos_y', 0.10))
        beam_layout.addWidget(QLabel("Beam Position Y"), 4, 0)
        beam_layout.addWidget(self.beam_pos_y_spinbox, 4, 1)
        beam_group.setLayout(beam_layout)

        page_layout.addWidget(color_group)
        page_layout.addWidget(axis_group)
        page_layout.addWidget(grid_group)
        page_layout.addWidget(beam_group)
        page_layout.addStretch(1)

        display_widget = QWidget()
        display_widget.setLayout(page_layout)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(display_widget)
        return scroll_area
        
        

    def create_ticks_tab(self):
        """Create the ticks settings tab."""
        page_layout = QVBoxLayout()

        line_group = QGroupBox("Tick Lines")
        line_layout = QGridLayout()
        self.tick_direction_input = QComboBox()
        self.tick_direction_input.addItems(['in', 'out'])
        self.tick_direction_input.setCurrentText(self.config_manager.config.get('tick_direction', 'out'))
        self.tick_direction_input.setFixedWidth(100)
        line_layout.addWidget(QLabel('Tick Direction:'), 0, 0)
        line_layout.addWidget(self.tick_direction_input, 0, 1)

        self.default_ticks_position_input = QComboBox()
        self.default_ticks_position_input.addItems(['None', 'b', 't', 'l', 'r', 'bt', 'bl', 'br', 'tl', 'tr', 'lr', 'btlr'])
        self.default_ticks_position_input.setCurrentText(self.config_manager.config.get('default_ticks_position', 'btlr'))
        self.default_ticks_position_input.setFixedWidth(100)
        line_layout.addWidget(QLabel('Tick Position:'), 1, 0)
        line_layout.addWidget(self.default_ticks_position_input, 1, 1)

        self.tick_length_input = QSpinBox()
        self.tick_length_input.setValue(self.config_manager.config.get('tick_length', 4))
        self.tick_length_input.setFixedWidth(100)
        line_layout.addWidget(QLabel('Tick Length:'), 2, 0)
        line_layout.addWidget(self.tick_length_input, 2, 1)

        self.mtick_length_input = QSpinBox()
        self.mtick_length_input.setValue(self.config_manager.config.get('mtick_length', 2))
        self.mtick_length_input.setFixedWidth(100)
        line_layout.addWidget(QLabel('Minor Tick Length:'), 3, 0)
        line_layout.addWidget(self.mtick_length_input, 3, 1)

        self.tick_width_input = QDoubleSpinBox()
        self.tick_width_input.setValue(self.config_manager.config.get('tick_width', 1))
        self.tick_width_input.setFixedWidth(100)
        self.tick_width_input.setSingleStep(0.5)
        line_layout.addWidget(QLabel('Tick Line Width:'), 4, 0)
        line_layout.addWidget(self.tick_width_input, 4, 1)
        line_group.setLayout(line_layout)

        label_style_group = QGroupBox("Tick Label Style")
        label_style_layout = QGridLayout()
        self.tick_labelsize_input = QSpinBox()
        self.tick_labelsize_input.setValue(self.config_manager.config.get('tick_labelsize', 10))
        self.tick_labelsize_input.setFixedWidth(100)
        label_style_layout.addWidget(QLabel('Tick Label Font Size:'), 0, 0)
        label_style_layout.addWidget(self.tick_labelsize_input, 0, 1)

        self.tick_color_input = self.create_color_combobox(self.config_manager.config.get('tick_color', 'black'))
        self.tick_color_input.setFixedWidth(100)
        label_style_layout.addWidget(QLabel('Tick Color:'), 1, 0)
        label_style_layout.addWidget(self.tick_color_input, 1, 1)

        self.tick_labelcolor_input = self.create_color_combobox(self.config_manager.config.get('tick_labelcolor', 'black'))
        self.tick_labelcolor_input.setFixedWidth(100)
        label_style_layout.addWidget(QLabel('Tick Label Color:'), 2, 0)
        label_style_layout.addWidget(self.tick_labelcolor_input, 2, 1)

        self.ticklabel_fontfamily_input = self.create_font_combobox(self.config_manager.config.get('tick_font', 'DejaVu Sans'))
        self.ticklabel_fontfamily_input.setFixedWidth(100)
        label_style_layout.addWidget(QLabel('Tick Label Font Family:'), 3, 0)
        label_style_layout.addWidget(self.ticklabel_fontfamily_input, 3, 1)
        label_style_group.setLayout(label_style_layout)

        label_layout_group = QGroupBox("Tick Label Layout")
        label_layout = QGridLayout()
        self.xticklabel_position_input = QComboBox()
        self.xticklabel_position_input.addItems(['t', 'b'])
        self.xticklabel_position_input.setCurrentText(self.config_manager.config.get('xticklabel_position', 'b'))
        self.xticklabel_position_input.setFixedWidth(100)
        label_layout.addWidget(QLabel('X Tick Label Position:'), 0, 0)
        label_layout.addWidget(self.xticklabel_position_input, 0, 1)

        self.yticklabel_position_input = QComboBox()
        self.yticklabel_position_input.addItems(['l', 'r'])
        self.yticklabel_position_input.setCurrentText(self.config_manager.config.get('yticklabel_position', 'l'))
        self.yticklabel_position_input.setFixedWidth(100)
        label_layout.addWidget(QLabel('Y Tick Label Position:'), 1, 0)
        label_layout.addWidget(self.yticklabel_position_input, 1, 1)

        self.tick_xlabelrotation_input = CircularSpinBox()
        self.tick_xlabelrotation_input.setValue(self.config_manager.config.get('tick_xlabelrotation', 0))
        self.tick_xlabelrotation_input.setFixedWidth(100)
        label_layout.addWidget(QLabel('X Tick Label Rotation:'), 2, 0)
        label_layout.addWidget(self.tick_xlabelrotation_input, 2, 1)

        self.tick_ylabelrotation_input = CircularSpinBox()
        self.tick_ylabelrotation_input.setValue(self.config_manager.config.get('tick_ylabelrotation', 0))
        self.tick_ylabelrotation_input.setFixedWidth(100)
        label_layout.addWidget(QLabel('Y Tick Label Rotation:'), 3, 0)
        label_layout.addWidget(self.tick_ylabelrotation_input, 3, 1)

        self.tick_xpad_input = QSpinBox()
        self.tick_xpad_input.setValue(self.config_manager.config.get('tick_pad_x', 5))
        self.tick_xpad_input.setFixedWidth(100)
        label_layout.addWidget(QLabel('X Tick Label Space:'), 4, 0)
        label_layout.addWidget(self.tick_xpad_input, 4, 1)

        self.tick_ypad_input = QSpinBox()
        self.tick_ypad_input.setValue(self.config_manager.config.get('tick_pad_y', 5))
        self.tick_ypad_input.setFixedWidth(100)
        label_layout.addWidget(QLabel('Y Tick Label Space:'), 5, 0)
        label_layout.addWidget(self.tick_ypad_input, 5, 1)
        label_layout_group.setLayout(label_layout)

        mtick_group = QGroupBox("Minor Tick Frequency")
        mtick_layout = QGridLayout()
        self.x_mtick_freq_input = QSpinBox()
        self.x_mtick_freq_input.setValue(self.config_manager.config.get('x_mtick_freq', 5))
        self.x_mtick_freq_input.setFixedWidth(100)
        self.x_mtick_freq_input.setSingleStep(1)
        self.x_mtick_freq_input.setRange(1, 10)
        mtick_layout.addWidget(QLabel('X Minor Tick Freq.:'), 0, 0)
        mtick_layout.addWidget(self.x_mtick_freq_input, 0, 1)

        self.y_mtick_freq_input = QSpinBox()
        self.y_mtick_freq_input.setValue(self.config_manager.config.get('y_mtick_freq', 5))
        self.y_mtick_freq_input.setFixedWidth(100)
        self.y_mtick_freq_input.setSingleStep(1)
        self.y_mtick_freq_input.setRange(1, 10)
        mtick_layout.addWidget(QLabel('Y Minor Tick Freq.:'), 1, 0)
        mtick_layout.addWidget(self.y_mtick_freq_input, 1, 1)

        self.z_mtick_freq_input = QSpinBox()
        self.z_mtick_freq_input.setValue(self.config_manager.config.get('z_mtick_freq', 5))
        self.z_mtick_freq_input.setFixedWidth(100)
        self.z_mtick_freq_input.setSingleStep(1)
        self.z_mtick_freq_input.setRange(1, 10)
        mtick_layout.addWidget(QLabel('Z Minor Tick Freq.:'), 2, 0)
        mtick_layout.addWidget(self.z_mtick_freq_input, 2, 1)
        mtick_group.setLayout(mtick_layout)

        page_layout.addWidget(line_group)
        page_layout.addWidget(label_style_group)
        page_layout.addWidget(label_layout_group)
        page_layout.addWidget(mtick_group)
        page_layout.addStretch(1)

        ticks_widget = QWidget()
        ticks_widget.setLayout(page_layout)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(ticks_widget)
        return scroll_area

    def create_click_tab(self):
        """Create the click settings tab."""
        page_layout = QVBoxLayout()

        crosshair_group = QGroupBox("Crosshair")
        crosshair_layout = QGridLayout()
        self.click_show_crosshair_checkbox = QCheckBox("Show Crosshair")
        self.click_show_crosshair_checkbox.setChecked(
            bool(self.config_manager.config.get('click_show_crosshair', True))
        )
        crosshair_layout.addWidget(self.click_show_crosshair_checkbox, 0, 0, 1, 2)

        self.click_crosshair_mode_label = QLabel('Crosshair Mode:')
        self.click_crosshair_mode_input = QComboBox()
        self.click_crosshair_mode_input.addItems(['both', 'vertical', 'horizontal'])
        self.click_crosshair_mode_input.setCurrentText(
            str(self.config_manager.config.get('click_crosshair_mode', 'both'))
        )
        self.click_crosshair_mode_input.setFixedWidth(100)
        crosshair_layout.addWidget(self.click_crosshair_mode_label, 1, 0)
        crosshair_layout.addWidget(self.click_crosshair_mode_input, 1, 1)

        self.click_show_center_marker_checkbox = QCheckBox("Show Center Marker")
        self.click_show_center_marker_checkbox.setChecked(
            bool(self.config_manager.config.get('click_show_center_marker', False))
        )
        crosshair_layout.addWidget(self.click_show_center_marker_checkbox, 2, 0, 1, 2)

        self.click_linecolor_label = QLabel('Click Line Color:')
        self.click_linecolor_input = self.create_color_combobox(self.config_manager.config.get('click_linecolor', 'cyan'))
        self.click_linecolor_input.setFixedWidth(100)
        crosshair_layout.addWidget(self.click_linecolor_label, 3, 0)
        crosshair_layout.addWidget(self.click_linecolor_input, 3, 1)

        self.click_linewidth_label = QLabel('Click Line Width:')
        self.click_linewidth_input = QDoubleSpinBox()
        self.click_linewidth_input.setValue(self.config_manager.config.get('click_linewidth', 0.5))
        self.click_linewidth_input.setFixedWidth(100)
        self.click_linewidth_input.setSingleStep(0.25)
        crosshair_layout.addWidget(self.click_linewidth_label, 4, 0)
        crosshair_layout.addWidget(self.click_linewidth_input, 4, 1)

        self.click_linestyle_label = QLabel('Click Line Style:')
        self.click_linestyle_input = QComboBox()
        for label, value in self.CLICK_LINESTYLE_OPTIONS:
            self.click_linestyle_input.addItem(label, value)
        self._set_click_linestyle_combo(self.config_manager.config.get('click_linestyle', '-'))
        self.click_linestyle_input.setFixedWidth(100)
        crosshair_layout.addWidget(self.click_linestyle_label, 5, 0)
        crosshair_layout.addWidget(self.click_linestyle_input, 5, 1)

        self.click_alpha_label = QLabel('Click Line Alpha:')
        self.click_alpha_input = QDoubleSpinBox()
        self.click_alpha_input.setRange(0.0, 1.0)
        self.click_alpha_input.setSingleStep(0.05)
        self.click_alpha_input.setDecimals(2)
        self.click_alpha_input.setValue(float(self.config_manager.config.get('click_alpha', 1.0)))
        self.click_alpha_input.setFixedWidth(100)
        crosshair_layout.addWidget(self.click_alpha_label, 6, 0)
        crosshair_layout.addWidget(self.click_alpha_input, 6, 1)
        crosshair_group.setLayout(crosshair_layout)
        self.click_show_crosshair_checkbox.toggled.connect(self._on_click_show_crosshair_toggled)
        self.click_show_center_marker_checkbox.toggled.connect(
            lambda _checked: self._on_click_show_crosshair_toggled(
                self.click_show_crosshair_checkbox.isChecked()
            )
        )
        self._on_click_show_crosshair_toggled(self.click_show_crosshair_checkbox.isChecked())

        coord_label_group = QGroupBox("Cursor Label")
        coord_label_layout = QGridLayout()
        self.click_label_color_input = self.create_color_combobox(self.config_manager.config.get('click_label_color', 'grey'))
        self.click_label_color_input.setFixedWidth(100)
        coord_label_layout.addWidget(QLabel('Click Label Color:'), 0, 0)
        coord_label_layout.addWidget(self.click_label_color_input, 0, 1)

        self.poslabel_x_input = QDoubleSpinBox()
        self.poslabel_x_input.setValue(self.config_manager.config.get('poslabel_x', 0.99))
        self.poslabel_x_input.setFixedWidth(100)
        self.poslabel_x_input.setRange(0.0, 1.0)
        self.poslabel_x_input.setSingleStep(0.01)
        coord_label_layout.addWidget(QLabel('Click Label X Position:'), 1, 0)
        coord_label_layout.addWidget(self.poslabel_x_input, 1, 1)

        self.poslabel_y_input = QDoubleSpinBox()
        self.poslabel_y_input.setValue(self.config_manager.config.get('poslabel_y', 0.99))
        self.poslabel_y_input.setSingleStep(0.01)
        self.poslabel_y_input.setFixedWidth(100)
        self.poslabel_y_input.setRange(0.0, 1.0)
        coord_label_layout.addWidget(QLabel('Click Label Y Position:'), 2, 0)
        coord_label_layout.addWidget(self.poslabel_y_input, 2, 1)

        self.poslabel_w_input = QSpinBox()
        self.poslabel_w_input.setFixedWidth(100)
        self.poslabel_w_input.setRange(0, 999)
        self.poslabel_w_input.setValue(self.config_manager.config.get('poslabel_w', 250))
        coord_label_layout.addWidget(QLabel('Click Label Width (px):'), 3, 0)
        coord_label_layout.addWidget(self.poslabel_w_input, 3, 1)

        self.poslabel_h_input = QSpinBox()
        self.poslabel_h_input.setFixedWidth(100)
        self.poslabel_h_input.setRange(0, 999)
        self.poslabel_h_input.setValue(self.config_manager.config.get('poslabel_h', 30))
        coord_label_layout.addWidget(QLabel('Click Label Height (px):'), 4, 0)
        coord_label_layout.addWidget(self.poslabel_h_input, 4, 1)
        coord_label_group.setLayout(coord_label_layout)

        channel_label_group = QGroupBox("Channel Label")
        channel_label_layout = QGridLayout()
        self.ch_label_color_input = self.create_color_combobox(self.config_manager.config.get('ch_label_color', 'grey'))
        self.ch_label_color_input.setEditable(True)
        self.ch_label_color_input.setFixedWidth(100)
        channel_label_layout.addWidget(QLabel('Ch. Label Color:'), 0, 0)
        channel_label_layout.addWidget(self.ch_label_color_input, 0, 1)

        self.ch_label_size_input = QSpinBox()
        self.ch_label_size_input.setFixedWidth(100)
        self.ch_label_size_input.setRange(1, 100)
        self.ch_label_size_input.setValue(self.config_manager.config.get('ch_label_size', 10))
        channel_label_layout.addWidget(QLabel('Ch. Label Font Size:'), 1, 0)
        channel_label_layout.addWidget(self.ch_label_size_input, 1, 1)

        self.ch_label_font_input = self.create_font_combobox(self.config_manager.config.get('ch_label_font', 'DejaVu Sans'))
        self.ch_label_font_input.setEditable(True)
        self.ch_label_font_input.setFixedWidth(100)
        channel_label_layout.addWidget(QLabel('Ch. Label Font:'), 2, 0)
        channel_label_layout.addWidget(self.ch_label_font_input, 2, 1)

        self.pos_chlabel_x_input = QDoubleSpinBox()
        self.pos_chlabel_x_input.setFixedWidth(100)
        self.pos_chlabel_x_input.setRange(0.0, 2.0)
        self.pos_chlabel_x_input.setSingleStep(0.01)
        self.pos_chlabel_x_input.setValue(self.config_manager.config.get('pos_chlabel_x', 0.98))
        channel_label_layout.addWidget(QLabel('Ch. Label X Position:'), 3, 0)
        channel_label_layout.addWidget(self.pos_chlabel_x_input, 3, 1)

        self.pos_chlabel_y_input = QDoubleSpinBox()
        self.pos_chlabel_y_input.setFixedWidth(100)
        self.pos_chlabel_y_input.setRange(0.0, 2.0)
        self.pos_chlabel_y_input.setSingleStep(0.01)
        self.pos_chlabel_y_input.setValue(self.config_manager.config.get('pos_chlabel_y', 0.02))
        channel_label_layout.addWidget(QLabel('Ch. Label Y Position:'), 4, 0)
        channel_label_layout.addWidget(self.pos_chlabel_y_input, 4, 1)
        channel_label_group.setLayout(channel_label_layout)
        
        """
        self.pos_chlabel_w_input = QSpinBox()
        self.pos_chlabel_w_input.setFixedWidth(100)
        self.pos_chlabel_w_input.setRange(0, 999)
        self.pos_chlabel_w_input.setValue(self.config_manager.config.get('pos_chlabel_w', 250))
        click_layout.addWidget(QLabel('Ch. Label Width (pixel value):'), 12, 0)
        click_layout.addWidget(self.pos_chlabel_w_input, 12, 1)

        self.pos_chlabel_h_input = QSpinBox()
        self.pos_chlabel_h_input.setFixedWidth(100)
        self.pos_chlabel_h_input.setRange(0, 999)
        self.pos_chlabel_h_input.setValue(self.config_manager.config.get('pos_chlabel_h', 20))
        click_layout.addWidget(QLabel('Ch. Label Height (pixel value):'), 13, 0)
        click_layout.addWidget(self.pos_chlabel_h_input, 13, 1)
        """
    
        page_layout.addWidget(crosshair_group)
        page_layout.addWidget(coord_label_group)
        page_layout.addWidget(channel_label_group)
        page_layout.addStretch(1)

        click_widget = QWidget()
        click_widget.setLayout(page_layout)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(click_widget)
        return scroll_area

    def _on_click_show_crosshair_toggled(self, checked):
        lines_enabled = bool(checked)
        center_marker_enabled = bool(
            getattr(self, "click_show_center_marker_checkbox", None)
            and self.click_show_center_marker_checkbox.isChecked()
        )
        color_enabled = bool(lines_enabled or center_marker_enabled)

        line_widgets = [
            self.click_crosshair_mode_label,
            self.click_crosshair_mode_input,
            self.click_linewidth_label,
            self.click_linewidth_input,
            self.click_linestyle_label,
            self.click_linestyle_input,
        ]
        color_widgets = [
            self.click_linecolor_label,
            self.click_linecolor_input,
            self.click_alpha_label,
            self.click_alpha_input,
        ]
        for widget in line_widgets:
            try:
                widget.setEnabled(lines_enabled)
            except Exception:
                continue
        for widget in color_widgets:
            try:
                widget.setEnabled(color_enabled)
            except Exception:
                continue

    def create_color_combobox(self, current_color):
        """Helper method to create a combo box for color selection."""
        color_combobox = QComboBox()
        colors = list(mpl.colors.CSS4_COLORS.keys())
        colors.append('#323232')
        colors.append('#ececec')
        colors.append('None')
        color_combobox.addItems(colors)
        color_combobox.setEditable(True)
        color_combobox.setCompleter(QCompleter(colors))
        if current_color is not None:
            color_combobox.setCurrentText(str(current_color))
        return color_combobox

    def create_font_combobox(self, current_font):
        """Helper method to create a combo box for font selection."""
        font_combobox = QComboBox()
        fonts = QFontDatabase.families()
        font_combobox.addItems(fonts)
        font_combobox.setEditable(True)
        font_combobox.setCompleter(QCompleter(fonts))
        if current_font:
            font_combobox.setCurrentText(str(current_font))
        return font_combobox
        
        
    def create_colorbar_tab(self):
        """Create the colorbar settings tab."""
        page_layout = QVBoxLayout()

        auto_group = QGroupBox("Auto Layout")
        auto_layout = QGridLayout()
        auto_row = 0

        self.cbar_auto_layout_checkbox = QCheckBox("Auto fit colorbar")
        self.cbar_auto_layout_checkbox.setChecked(bool(self.config_manager.config.get('colorbar_auto_layout', True)))
        auto_layout.addWidget(self.cbar_auto_layout_checkbox, auto_row, 0)

        self.cbar_fit_now_button = QPushButton("Fit now")
        fit_button_to_text(self.cbar_fit_now_button, minimum_width=100)
        auto_layout.addWidget(self.cbar_fit_now_button, auto_row, 1)
        auto_row += 1

        self.cbar_placement_label = QLabel("Placement:")
        self.cbar_placement_combo = QComboBox()
        self.cbar_placement_combo.addItems([
            "right",
            "left",
            "top",
            "bottom",
            "inside-right",
            "inside-left",
            "inside-top",
            "inside-bottom",
        ])
        self.cbar_placement_combo.setCurrentText(self.config_manager.config.get('colorbar_placement', 'right'))
        auto_layout.addWidget(self.cbar_placement_label, auto_row, 0)
        auto_layout.addWidget(self.cbar_placement_combo, auto_row, 1)
        auto_row += 1

        self.cbar_inside_preset_button = QPushButton("Apply inside preset")
        self.cbar_inside_preset_button.setToolTip(
            "Apply TakeFits' compact, high-contrast defaults for an inside colorbar."
        )
        fit_button_to_text(self.cbar_inside_preset_button, minimum_width=120)
        auto_layout.addWidget(
            self.cbar_inside_preset_button,
            auto_row,
            0,
            1,
            2,
        )
        auto_row += 1

        self.cbar_align_label = QLabel("Align:")
        self.cbar_align_combo = QComboBox()
        self.cbar_align_combo.addItems(["center", "start", "end"])
        self.cbar_align_combo.setCurrentText(self.config_manager.config.get('colorbar_align', 'center'))
        auto_layout.addWidget(self.cbar_align_label, auto_row, 0)
        auto_layout.addWidget(self.cbar_align_combo, auto_row, 1)
        auto_row += 1

        self.cbar_gap_x_px_label = QLabel("Gap X (px):")
        self.cbar_gap_x_px_input = QDoubleSpinBox()
        self.cbar_gap_x_px_input.setRange(0.0, 200.0)
        self.cbar_gap_x_px_input.setSingleStep(1.0)
        self.cbar_gap_x_px_input.setFixedWidth(100)
        self.cbar_gap_x_px_input.setValue(float(self.config_manager.config.get('colorbar_gap_x_px', self.config_manager.config.get('colorbar_gap_px', 24.0))))
        auto_layout.addWidget(self.cbar_gap_x_px_label, auto_row, 0)
        auto_layout.addWidget(self.cbar_gap_x_px_input, auto_row, 1)
        auto_row += 1

        self.cbar_gap_y_px_label = QLabel("Gap Y (px):")
        self.cbar_gap_y_px_input = QDoubleSpinBox()
        self.cbar_gap_y_px_input.setRange(0.0, 200.0)
        self.cbar_gap_y_px_input.setSingleStep(1.0)
        self.cbar_gap_y_px_input.setFixedWidth(100)
        self.cbar_gap_y_px_input.setValue(float(self.config_manager.config.get('colorbar_gap_y_px', self.config_manager.config.get('colorbar_gap_px', 24.0))))
        auto_layout.addWidget(self.cbar_gap_y_px_label, auto_row, 0)
        auto_layout.addWidget(self.cbar_gap_y_px_input, auto_row, 1)
        auto_row += 1

        self.cbar_thickness_px_label = QLabel("Thickness (px):")
        self.cbar_thickness_px_input = QDoubleSpinBox()
        self.cbar_thickness_px_input.setRange(1.0, 400.0)
        self.cbar_thickness_px_input.setSingleStep(1.0)
        self.cbar_thickness_px_input.setFixedWidth(100)
        self.cbar_thickness_px_input.setValue(float(self.config_manager.config.get('colorbar_thickness_px', 24.0)))
        auto_layout.addWidget(self.cbar_thickness_px_label, auto_row, 0)
        auto_layout.addWidget(self.cbar_thickness_px_input, auto_row, 1)
        auto_row += 1

        self.cbar_length_mode_label = QLabel("Length Mode:")
        self.cbar_length_mode_combo = QComboBox()
        self.cbar_length_mode_combo.addItems(["ratio", "px"])
        current_mode = self._normalize_colorbar_length_mode(
            self.config_manager.config.get('colorbar_length_mode', 'ratio')
        )
        self.cbar_length_mode_combo.setCurrentText(current_mode)
        auto_layout.addWidget(self.cbar_length_mode_label, auto_row, 0)
        auto_layout.addWidget(self.cbar_length_mode_combo, auto_row, 1)
        auto_row += 1

        self.cbar_length_value_label = QLabel("Length Value:")
        self.cbar_length_value_input = QDoubleSpinBox()
        self.cbar_length_value_input.setRange(0.01, 5000.0)
        self.cbar_length_value_input.setSingleStep(0.05)
        self.cbar_length_value_input.setFixedWidth(100)
        self.cbar_length_value_input.setValue(float(self.config_manager.config.get('colorbar_length_value', 1.0)))
        auto_layout.addWidget(self.cbar_length_value_label, auto_row, 0)
        auto_layout.addWidget(self.cbar_length_value_input, auto_row, 1)
        auto_layout.setColumnStretch(0, 1)
        auto_layout.setColumnStretch(1, 1)
        auto_group.setLayout(auto_layout)

        self.cbar_manual_group = QGroupBox("Manual Layout")
        manual_layout = QGridLayout()
        manual_row = 0
        self.cbar_orientation_label = QLabel("Colorbar Orientation:")
        self.cbar_orientation_combo = QComboBox()
        self.cbar_orientation_combo.addItems(["vertical", "horizontal"])
        self.cbar_orientation_combo.setCurrentText(self.config_manager.config.get('colorbar_orientation', 'vertical'))
        manual_layout.addWidget(self.cbar_orientation_label, manual_row, 0)
        manual_layout.addWidget(self.cbar_orientation_combo, manual_row, 1)
        manual_row += 1

        self.cbar_x_label = QLabel("Colorbar X Position:")
        self.cbar_x_input = QDoubleSpinBox()
        self.cbar_x_input.setRange(0.0, 1.0)
        self.cbar_x_input.setSingleStep(0.01)
        self.cbar_x_input.setFixedWidth(100)
        self.cbar_x_input.setValue(self.config_manager.config.get('cbar_pos_x', 0.9))
        manual_layout.addWidget(self.cbar_x_label, manual_row, 0)
        manual_layout.addWidget(self.cbar_x_input, manual_row, 1)
        manual_row += 1

        self.cbar_y_label = QLabel("Colorbar Y Position:")
        self.cbar_y_input = QDoubleSpinBox()
        self.cbar_y_input.setRange(0.0, 1.0)
        self.cbar_y_input.setSingleStep(0.01)
        self.cbar_y_input.setFixedWidth(100)
        self.cbar_y_input.setValue(self.config_manager.config.get('cbar_pos_y', 0.11))
        manual_layout.addWidget(self.cbar_y_label, manual_row, 0)
        manual_layout.addWidget(self.cbar_y_input, manual_row, 1)
        manual_row += 1

        self.cbar_width_label = QLabel("Colorbar Width:")
        self.cbar_width_input = QDoubleSpinBox()
        self.cbar_width_input.setRange(0.0, 1.0)
        self.cbar_width_input.setSingleStep(0.01)
        self.cbar_width_input.setFixedWidth(100)
        self.cbar_width_input.setValue(self.config_manager.config.get('cbar_width', 0.04))
        manual_layout.addWidget(self.cbar_width_label, manual_row, 0)
        manual_layout.addWidget(self.cbar_width_input, manual_row, 1)
        manual_row += 1

        self.cbar_height_label = QLabel("Colorbar Height:")
        self.cbar_height_input = QDoubleSpinBox()
        self.cbar_height_input.setRange(0.0, 1.0)
        self.cbar_height_input.setSingleStep(0.01)
        self.cbar_height_input.setFixedWidth(100)
        self.cbar_height_input.setValue(self.config_manager.config.get('cbar_height', 0.77))
        manual_layout.addWidget(self.cbar_height_label, manual_row, 0)
        manual_layout.addWidget(self.cbar_height_input, manual_row, 1)
        manual_layout.setColumnStretch(0, 1)
        manual_layout.setColumnStretch(1, 1)
        self.cbar_manual_group.setLayout(manual_layout)

        label_group = QGroupBox("Label")
        label_layout = QGridLayout()
        label_row = 0
        self.cbar_label_text_label = QLabel("Colorbar Label:")
        self.cbar_label_text_input = QLineEdit()
        self.cbar_label_text_input.setFixedWidth(140)
        current_cbar_label = self.config_manager.config.get('colorbar_label', '')
        if current_cbar_label is None:
            current_cbar_label = ''
        self.cbar_label_text_input.setText(str(current_cbar_label))
        label_layout.addWidget(self.cbar_label_text_label, label_row, 0)
        label_layout.addWidget(self.cbar_label_text_input, label_row, 1)
        label_row += 1

        self.cbar_label_fontsize_label = QLabel("Label Font Size:")
        self.cbar_label_fontsize_input = QSpinBox()
        self.cbar_label_fontsize_input.setRange(1, 100)
        self.cbar_label_fontsize_input.setFixedWidth(100)
        self.cbar_label_fontsize_input.setValue(self.config_manager.config.get('colorbar_label_fontsize', 12))
        label_layout.addWidget(self.cbar_label_fontsize_label, label_row, 0)
        label_layout.addWidget(self.cbar_label_fontsize_input, label_row, 1)
        label_row += 1

        self.cbar_label_color_label = QLabel("Label Color:")
        self.cbar_label_color_input = self.create_color_combobox(self.config_manager.config.get('colorbar_label_color', 'black'))
        self.cbar_label_color_input.setFixedWidth(100)
        label_layout.addWidget(self.cbar_label_color_label, label_row, 0)
        label_layout.addWidget(self.cbar_label_color_input, label_row, 1)
        label_layout.setColumnStretch(0, 1)
        label_layout.setColumnStretch(1, 1)
        label_group.setLayout(label_layout)

        tick_group = QGroupBox("Ticks")
        tick_layout = QGridLayout()
        tick_row = 0
        self.cbar_tick_direction_label = QLabel("Tick Direction:")
        self.cbar_tick_direction_combo = QComboBox()
        self.cbar_tick_direction_combo.addItems(["in", "out", "inout"])
        self.cbar_tick_direction_combo.setCurrentText(self.config_manager.config.get('colorbar_tick_direction', 'out'))
        tick_layout.addWidget(self.cbar_tick_direction_label, tick_row, 0)
        tick_layout.addWidget(self.cbar_tick_direction_combo, tick_row, 1)
        tick_row += 1

        self.cbar_tick_length_label = QLabel("Colorbar Tick Length:")
        self.cbar_tick_length_input = QDoubleSpinBox()
        self.cbar_tick_length_input.setRange(0, 20)
        self.cbar_tick_length_input.setSingleStep(0.5)
        self.cbar_tick_length_input.setFixedWidth(100)
        self.cbar_tick_length_input.setValue(self.config_manager.config.get('colorbar_tick_length', 2))
        tick_layout.addWidget(self.cbar_tick_length_label, tick_row, 0)
        tick_layout.addWidget(self.cbar_tick_length_input, tick_row, 1)
        tick_row += 1

        self.cbar_tick_width_label = QLabel("Tick Width:")
        self.cbar_tick_width_input = QDoubleSpinBox()
        self.cbar_tick_width_input.setRange(0, 20)
        self.cbar_tick_width_input.setSingleStep(0.5)
        self.cbar_tick_width_input.setFixedWidth(100)
        self.cbar_tick_width_input.setValue(self.config_manager.config.get('colorbar_tick_width', 1))
        tick_layout.addWidget(self.cbar_tick_width_label, tick_row, 0)
        tick_layout.addWidget(self.cbar_tick_width_input, tick_row, 1)
        tick_row += 1

        self.cbar_mtick_freq_label = QLabel("Minor Tick Frequency:")
        self.cbar_mtick_freq_input = QSpinBox()
        self.cbar_mtick_freq_input.setRange(1, 10)
        self.cbar_mtick_freq_input.setFixedWidth(100)
        self.cbar_mtick_freq_input.setValue(self.config_manager.config.get('colorbar_mtick_freq', 2))
        tick_layout.addWidget(self.cbar_mtick_freq_label, tick_row, 0)
        tick_layout.addWidget(self.cbar_mtick_freq_input, tick_row, 1)
        tick_row += 1

        self.cbar_mtick_length_label = QLabel("Minor Tick Length:")
        self.cbar_mtick_length_input = QDoubleSpinBox()
        self.cbar_mtick_length_input.setRange(0, 20)
        self.cbar_mtick_length_input.setSingleStep(0.5)
        self.cbar_mtick_length_input.setFixedWidth(100)
        self.cbar_mtick_length_input.setValue(self.config_manager.config.get('colorbar_mtick_length', 1))
        tick_layout.addWidget(self.cbar_mtick_length_label, tick_row, 0)
        tick_layout.addWidget(self.cbar_mtick_length_input, tick_row, 1)
        tick_row += 1

        self.cbar_tick_color_label = QLabel("Tick Color:")
        self.cbar_tick_color_input = self.create_color_combobox(self.config_manager.config.get('colorbar_tick_color', 'black'))
        self.cbar_tick_color_input.setFixedWidth(100)
        tick_layout.addWidget(self.cbar_tick_color_label, tick_row, 0)
        tick_layout.addWidget(self.cbar_tick_color_input, tick_row, 1)
        tick_row += 1

        self.cbar_tick_label_color_label = QLabel("Tick Label Color:")
        self.cbar_tick_label_color_input = self.create_color_combobox(self.config_manager.config.get('colorbar_tick_labelcolor', 'black'))
        self.cbar_tick_label_color_input.setFixedWidth(100)
        tick_layout.addWidget(self.cbar_tick_label_color_label, tick_row, 0)
        tick_layout.addWidget(self.cbar_tick_label_color_input, tick_row, 1)
        tick_row += 1

        self.cbar_tick_y_side_label = QLabel("Vertical Tick Sides:")
        self.cbar_tick_y_side_combo = QComboBox()
        self.cbar_tick_y_side_combo.addItems(["both", "left", "right", "none"])
        self.cbar_tick_y_side_combo.setFixedWidth(100)
        tick_layout.addWidget(self.cbar_tick_y_side_label, tick_row, 0)
        tick_layout.addWidget(self.cbar_tick_y_side_combo, tick_row, 1)
        tick_row += 1

        self.cbar_tick_x_side_label = QLabel("Horizontal Tick Sides:")
        self.cbar_tick_x_side_combo = QComboBox()
        self.cbar_tick_x_side_combo.addItems(["both", "top", "bottom", "none"])
        self.cbar_tick_x_side_combo.setFixedWidth(100)
        tick_layout.addWidget(self.cbar_tick_x_side_label, tick_row, 0)
        tick_layout.addWidget(self.cbar_tick_x_side_combo, tick_row, 1)
        tick_row += 1

        self.cbar_tick_y_label_side_label = QLabel("Vertical Tick Label Side:")
        self.cbar_tick_y_label_side_combo = QComboBox()
        self.cbar_tick_y_label_side_combo.addItems(["left", "right"])
        self.cbar_tick_y_label_side_combo.setFixedWidth(100)
        tick_layout.addWidget(self.cbar_tick_y_label_side_label, tick_row, 0)
        tick_layout.addWidget(self.cbar_tick_y_label_side_combo, tick_row, 1)
        tick_row += 1

        self.cbar_tick_x_label_side_label = QLabel("Horizontal Tick Label Side:")
        self.cbar_tick_x_label_side_combo = QComboBox()
        self.cbar_tick_x_label_side_combo.addItems(["top", "bottom"])
        self.cbar_tick_x_label_side_combo.setFixedWidth(100)
        tick_layout.addWidget(self.cbar_tick_x_label_side_label, tick_row, 0)
        tick_layout.addWidget(self.cbar_tick_x_label_side_combo, tick_row, 1)
        tick_row += 1

        self._load_colorbar_tick_side_comboboxes(self.config_manager.config)
        tick_layout.setColumnStretch(0, 1)
        tick_layout.setColumnStretch(1, 1)
        tick_group.setLayout(tick_layout)

        self.cbar_placement_combo.currentTextChanged.connect(
            self._update_colorbar_inside_preset_enabled
        )
        self.cbar_inside_preset_button.clicked.connect(
            lambda: self._on_colorbar_placement_changed(
                self.cbar_placement_combo.currentText()
            )
        )
        self.cbar_fit_now_button.clicked.connect(self._fit_colorbar_now)
        self.cbar_auto_layout_checkbox.toggled.connect(self._on_colorbar_auto_layout_toggled)
        self._on_colorbar_auto_layout_toggled(self.cbar_auto_layout_checkbox.isChecked())
        self._update_colorbar_inside_preset_enabled(
            self.cbar_placement_combo.currentText()
        )

        page_layout.addWidget(auto_group)
        page_layout.addWidget(self.cbar_manual_group)
        page_layout.addWidget(label_group)
        page_layout.addWidget(tick_group)
        page_layout.addStretch(1)

        # Put the layout into a QWidget

        colorbar_settings_widget = QWidget()
        colorbar_settings_widget.setLayout(page_layout)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(colorbar_settings_widget)
        return scroll_area

    def _on_colorbar_auto_layout_toggled(self, checked):
        auto_enabled = bool(checked)
        editable = not auto_enabled
        manual_group = getattr(self, "cbar_manual_group", None)
        if manual_group is not None:
            try:
                manual_group.setEnabled(editable)
            except Exception:
                pass
        widgets = [
            self.cbar_orientation_label,
            self.cbar_orientation_combo,
            self.cbar_x_label,
            self.cbar_x_input,
            self.cbar_y_label,
            self.cbar_y_input,
            self.cbar_width_label,
            self.cbar_width_input,
            self.cbar_height_label,
            self.cbar_height_input,
        ]
        for widget in widgets:
            try:
                widget.setEnabled(editable)
            except Exception:
                continue

    def _fit_colorbar_now(self):
        viewer = getattr(self, "fits_viewer", None)
        if viewer is None:
            return
        fit_now = getattr(viewer, "fit_colorbar_now", None)
        if not callable(fit_now):
            root = getattr(viewer, "main_viewer", None)
            fit_now = getattr(root, "fit_colorbar_now", None) if root is not None else None
        if callable(fit_now):
            try:
                fit_now()
            except Exception:
                pass
        

    def set_coordinate_format(self, use_decimal: bool):
        """Select the coordinate format radio button based on the config flag."""
        if use_decimal is None:
            use_decimal = True
        if use_decimal:
            self.decimal_radio.setChecked(True)
        else:
            self.sexagesimal_radio.setChecked(True)

    def sync_colorbar_geometry(self, pos_x, pos_y, width, height):
        """Mirror live colorbar geometry without making Preferences dirty."""
        values = {
            "cbar_pos_x": float(pos_x),
            "cbar_pos_y": float(pos_y),
            "cbar_width": float(width),
            "cbar_height": float(height),
        }
        widgets = {
            "cbar_pos_x": self.cbar_x_input,
            "cbar_pos_y": self.cbar_y_input,
            "cbar_width": self.cbar_width_input,
            "cbar_height": self.cbar_height_input,
        }
        for key, value in values.items():
            widgets[key].setValue(value)
        represented_values = {
            key: widget.value()
            for key, widget in widgets.items()
        }
        baseline = getattr(self, "_form_baseline_values", None)
        if isinstance(baseline, dict):
            baseline.update(represented_values)
        session_updates = getattr(
            self,
            "_session_authoritative_updates",
            None,
        )
        if isinstance(session_updates, dict):
            if bool(
                self.config_manager.config.get(
                    "colorbar_auto_layout",
                    True,
                )
            ):
                for key in values:
                    session_updates.pop(key, None)
            else:
                for key, value in values.items():
                    if key in session_updates:
                        session_updates[key] = value

    def _on_auto_precision_toggled(self, enabled: bool):
        """When auto precision is enabled, manual precision digits are read-only."""
        self.number_decimals_input.setEnabled(not bool(enabled))

    @staticmethod
    def _values_equivalent(current, new):
        """Check whether two config values should be considered identical."""
        if isinstance(current, bool) or isinstance(new, bool):
            return current == new
        if isinstance(current, (int, float)) and isinstance(new, (int, float)):
            return math.isclose(float(current), float(new), rel_tol=1e-9, abs_tol=1e-9)
        return current == new

    def _update_config_value(self, key, value, changed_keys):
        """Write a config value only when it actually changes."""
        current = self.config_manager.config.get(key)
        if self._values_equivalent(current, value):
            return
        self.config_manager.config[key] = value
        changed_keys.add(key)

    def _preference_roots(self):
        roots = [self.fits_viewer]
        try:
            from takefits.ui.window_registry import WindowRegistry

            roots.extend(WindowRegistry.instance().windows())
        except Exception:
            pass
        deduped = []
        seen = set()
        for root in roots:
            if root is None or id(root) in seen:
                continue
            seen.add(id(root))
            if getattr(root, "config_manager", None) is None:
                continue
            deduped.append(root)
        return deduped

    def _validate_grid_colors(self):
        candidates = [
            ("Native line color", self.grid_color_input.currentText()),
            ("Overlay line color", self.grid_overlay_color_input.currentText()),
        ]
        if (
            self.grid_overlay_label_color_policy_input.currentData()
            == "custom"
        ):
            candidates.append(
                (
                    "Overlay label color",
                    self.grid_overlay_label_color_input.currentText(),
                )
            )
        invalid = [
            label
            for label, color in candidates
            if str(color).strip().lower() == "none"
            or not mpl.colors.is_color_like(color)
        ]
        if not invalid:
            return True
        QMessageBox.critical(
            self,
            "Invalid Grid Color",
            "Please enter a valid Matplotlib color for: "
            + ", ".join(invalid),
        )
        return False

    def _validate_axes_positions(self):
        values = {
            "ax_pos_l": self.axis_left_spinbox.value(),
            "ax_pos_r": self.axis_right_spinbox.value(),
            "ax_pos_t": self.axis_top_spinbox.value(),
            "ax_pos_b": self.axis_bottom_spinbox.value(),
        }
        if axes_positions_are_valid(values):
            return True
        QMessageBox.critical(
            self,
            "Invalid Axes Position",
            "Axis Left must be smaller than Axis Right, and Axis Bottom "
            "must be smaller than Axis Top.",
        )
        return False

    def _propagate_preference_updates(self, updates, roots):
        """Copy the form values to every registered top-level config."""
        changed_keys = set()
        for root in roots:
            manager = getattr(root, "config_manager", None)
            config = getattr(manager, "config", None)
            if not isinstance(config, dict):
                continue
            for key, value in updates:
                current = config.get(key)
                if self._values_equivalent(current, value):
                    continue
                config[key] = value
                changed_keys.add(key)
        return changed_keys

    def _apply_grid_preferences_to_roots(self, roots):
        failed = []
        for root in roots:
            refresh = getattr(
                root,
                "refresh_coordinate_grid_preferences",
                None,
            )
            if callable(refresh):
                try:
                    if refresh() is False:
                        failed.append(root)
                except Exception:
                    failed.append(root)
        if not failed:
            return True
        QMessageBox.critical(
            self,
            "Coordinate Grid Error",
            "The grid style could not be refreshed in one or more open "
            "FITS windows. The settings were kept so Apply can be retried.",
        )
        return False

    def _apply_full_preferences_to_roots(self, roots, updated_keys):
        coordinate_keys = {
            "decimal",
            "auto_precision_digits",
            "number_decimals",
            "coord_wrap",
        }
        geometry_keys = {
            "figure_pos_x",
            "figure_pos_y",
            "figure_width",
            "figure_height",
        }
        coordinate_changed = bool(updated_keys.intersection(coordinate_keys))
        geometry_changed = bool(updated_keys.intersection(geometry_keys))
        decimal = bool(self.config_manager.config.get("decimal", True))
        first_error = None
        for root in roots:
            # reload_viewer() historically reapplies the configured geometry.
            # With application-wide Preferences that would move every open FITS
            # window onto the same rectangle even for a Decimal/Ticks edit.
            # Only an explicit geometry edit may move/resize a live window,
            # and it applies to the FITS root that owns this panel. Other roots
            # retain their current independent layouts.
            apply_geometry = (
                root is self.fits_viewer and geometry_changed
            )
            if coordinate_changed:
                set_decimal = getattr(root, "set_wcs_decimal_mode", None)
                if callable(set_decimal):
                    try:
                        set_decimal(decimal, refresh=False)
                    except TypeError:
                        set_decimal(decimal)
            try:
                root.reload_viewer(apply_geometry=apply_geometry)
            except ValueError as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Icon.Critical)
            msg_box.setText(
                "Error: Invalid config parameter(s) detected.\n"
                f"Details: {first_error}"
            )
            msg_box.setWindowTitle("Parameter Error")
            msg_box.exec()
            return False
        return True

    def keyPressEvent(self, event):
        """Allow pressing Enter/Return anywhere in the panel to trigger Apply."""
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.apply_button.click()
            event.accept()
            return
        super().keyPressEvent(event)

    def _collect_preference_updates(self):
        """Return normalized values currently represented by the form."""
        updates = [
            # General settings
            ('colorscale', self.colorscale_input.currentText()),
            ('decimal', self.decimal_radio.isChecked()),
            ('auto_precision_digits', self.auto_precision_checkbox.isChecked()),
            ('number_decimals', self.number_decimals_input.value()),
            ('coord_wrap', self.wrap_angle_input.value()),
            ('scrollspeed', self.scrollspeed_input.value()),
            ('invert_wheel_direction', self.invert_wheel_direction_checkbox.isChecked()),
            ('range_file', self.range_file_input.text()),
            ('startup_show_subwindow1', self.startup_show_subwindow1_checkbox.isChecked()),
            ('startup_show_subwindow2', self.startup_show_subwindow2_checkbox.isChecked()),

            # Display settings
            ('fig_background_color', self.fig_background_color_input.currentText()),
            ('ax_background_color', self.ax_background_color_input.currentText()),
            ('bad_color', self.bad_color_input.currentText()),
            ('figure_pos_x', int(self.figure_pos_x_input.text())),
            ('figure_pos_y', int(self.figure_pos_y_input.text())),
            ('figure_width', int(self.figure_width_input.text())),
            ('figure_height', int(self.figure_height_input.text())),
            ('axislabel_fontsize', self.axislabel_fontsize_input.value()),
            ('axislabel_fontfamily', self.axislabel_fontfamily_input.currentText()),
            ('axislabel_color', self.axislabel_color_input.currentText()),

            # Axis positions
            ('ax_pos_l', self.axis_left_spinbox.value()),
            ('ax_pos_r', self.axis_right_spinbox.value()),
            ('ax_pos_t', self.axis_top_spinbox.value()),
            ('ax_pos_b', self.axis_bottom_spinbox.value()),

            # Beam
            ('beam_facecolor', self.beam_facecolor_input.currentText()),
            ('beam_edgecolor', self.beam_edgecolor_input.currentText()),
            ('beam_linewidth', self.beam_linewidth_input.value()),
            ('beam_pos_x', self.beam_pos_x_spinbox.value()),
            ('beam_pos_y', self.beam_pos_y_spinbox.value()),

            # Colorbar settings
            ('colorbar_orientation', self.cbar_orientation_combo.currentText()),
            ('colorbar_auto_layout', self.cbar_auto_layout_checkbox.isChecked()),
            ('colorbar_placement', self.cbar_placement_combo.currentText()),
            ('colorbar_align', self.cbar_align_combo.currentText()),
            ('colorbar_gap_px', self.cbar_gap_x_px_input.value()),
            ('colorbar_gap_x_px', self.cbar_gap_x_px_input.value()),
            ('colorbar_gap_y_px', self.cbar_gap_y_px_input.value()),
            ('colorbar_thickness_px', self.cbar_thickness_px_input.value()),
            ('colorbar_length_mode', self.cbar_length_mode_combo.currentText()),
            ('colorbar_length_value', self.cbar_length_value_input.value()),
            ('colorbar_label', self.cbar_label_text_input.text()),
            ('colorbar_label_fontsize', self.cbar_label_fontsize_input.value()),
            ('colorbar_label_color', self.cbar_label_color_input.currentText()),
            ('cbar_pos_x', self.cbar_x_input.value()),
            ('cbar_pos_y', self.cbar_y_input.value()),
            ('cbar_width', self.cbar_width_input.value()),
            ('cbar_height', self.cbar_height_input.value()),
            ('colorbar_tick_direction', self.cbar_tick_direction_combo.currentText()),
            ('colorbar_tick_length', self.cbar_tick_length_input.value()),
            ('colorbar_tick_width', self.cbar_tick_width_input.value()),
            ('colorbar_mtick_freq', self.cbar_mtick_freq_input.value()),
            ('colorbar_mtick_length', self.cbar_mtick_length_input.value()),
            ('colorbar_tick_color', self.cbar_tick_color_input.currentText()),
            ('colorbar_tick_labelcolor', self.cbar_tick_label_color_input.currentText()),

            # Tick settings
            ('tick_labelsize', self.tick_labelsize_input.value()),
            ('tick_color', self.tick_color_input.currentText()),
            ('tick_labelcolor', self.tick_labelcolor_input.currentText()),
            ('tick_direction', self.tick_direction_input.currentText()),
            ('default_ticks_position', self.default_ticks_position_input.currentText()),
            ('xticklabel_position', self.xticklabel_position_input.currentText()),
            ('yticklabel_position', self.yticklabel_position_input.currentText()),
            ('tick_length', self.tick_length_input.value()),
            ('mtick_length', self.mtick_length_input.value()),
            ('tick_width', self.tick_width_input.value()),
            ('tick_xlabelrotation', self.tick_xlabelrotation_input.value()),
            ('tick_ylabelrotation', self.tick_ylabelrotation_input.value()),
            ('tick_pad_x', self.tick_xpad_input.value()),
            ('tick_pad_y', self.tick_ypad_input.value()),
            ('tick_font', self.ticklabel_fontfamily_input.currentText()),
            ('x_mtick_freq', self.x_mtick_freq_input.value()),
            ('y_mtick_freq', self.y_mtick_freq_input.value()),
            ('z_mtick_freq', self.z_mtick_freq_input.value()),

            # Click label settings
            ('click_label_color', self.click_label_color_input.currentText()),
            ('click_linewidth', self.click_linewidth_input.value()),
            ('click_linecolor', self.click_linecolor_input.currentText()),
            (
                'click_linestyle',
                self._normalize_click_linestyle(
                    self.click_linestyle_input.currentData()
                    or self.click_linestyle_input.currentText()
                ),
            ),
            ('click_alpha', self.click_alpha_input.value()),
            ('click_show_crosshair', self.click_show_crosshair_checkbox.isChecked()),
            ('click_crosshair_mode', self.click_crosshair_mode_input.currentText()),
            ('click_show_center_marker', self.click_show_center_marker_checkbox.isChecked()),
            ('poslabel_x', self.poslabel_x_input.value()),
            ('poslabel_y', self.poslabel_y_input.value()),
            ('poslabel_w', self.poslabel_w_input.value()),
            ('poslabel_h', self.poslabel_h_input.value()),
            ('ch_label_color', self.ch_label_color_input.currentText()),
            ('ch_label_size', self.ch_label_size_input.value()),
            ('ch_label_font', self.ch_label_font_input.currentText()),
            ('pos_chlabel_x', self.pos_chlabel_x_input.value()),
            ('pos_chlabel_y', self.pos_chlabel_y_input.value()),
            # ('pos_chlabel_w', self.pos_chlabel_w_input.value()),
            # ('pos_chlabel_h', self.pos_chlabel_h_input.value()),
        ]

        updates.extend(self._grid_config_updates())
        updates.extend(self._read_colorbar_tick_side_values().items())
        return updates

    def apply_changes(self):
        """Apply user-edited preferences to all open FITS viewers."""
        if not self._validate_axes_positions():
            return False
        if not self._validate_grid_colors():
            return False

        all_updates = self._collect_preference_updates()
        baseline = getattr(self, "_form_baseline_values", {})
        updates = [
            (key, value)
            for key, value in all_updates
            if key not in baseline
            or not self._values_equivalent(baseline.get(key), value)
        ]
        updated_keys = set()

        for key, value in updates:
            self._update_config_value(key, value, updated_keys)
            self._session_authoritative_updates[key] = copy.deepcopy(
                self.config_manager.config.get(key, value)
            )

        roots = self._preference_roots()
        # The panel may outlive the moment when another FITS root is opened.
        # Reapply only values explicitly changed during this panel session so
        # late roots catch up without treating per-window auto-layout geometry
        # as a global preference edit.
        source_updates = [
            (key, copy.deepcopy(value))
            for key, value in self._session_authoritative_updates.items()
        ]
        updated_keys.update(
            self._propagate_preference_updates(source_updates, roots)
        )
        if not updated_keys:
            return True

        if updated_keys.issubset(self.GRID_STYLE_KEYS):
            applied = self._apply_grid_preferences_to_roots(roots)
            if applied:
                self._form_baseline_values = dict(all_updates)
            return applied
        applied = self._apply_full_preferences_to_roots(roots, updated_keys)
        if applied:
            self._form_baseline_values = dict(all_updates)
        return applied
        



    def save_config(self):
        """Save the current config to the config.yaml file."""
        if os.path.exists(self.config_manager.config_file):
            reply = QMessageBox.question(self, 'Overwrite Confirmation',
                                        'The config.yaml file already exists. Do you want to overwrite it?',
                                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                        QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No:
                return
        if not self.apply_changes():
            return
        try:
            with open(self.config_manager.config_file, 'w', encoding='utf-8') as f:
                yaml.safe_dump(self.config_manager.config, f, default_flow_style=False, sort_keys=False)
            saved_path = os.path.abspath(self.config_manager.config_file)
            managers = [self.config_manager]
            managers.extend(
                getattr(root, "config_manager", None)
                for root in self._preference_roots()
            )
            seen = set()
            for manager in managers:
                if manager is None or id(manager) in seen:
                    continue
                seen.add(id(manager))
                manager_path = os.path.abspath(
                    getattr(manager, "config_file", "")
                )
                if manager_path != saved_path:
                    continue
                manager.config_bu = copy.deepcopy(
                    self.config_manager.config
                )
            self._session_authoritative_updates.clear()
            QMessageBox.information(self, 'Success', 'Configuration saved successfully.')
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to save configuration: {str(e)}')

    def reset_to_loaded_config(self):
        """Reset the configuration to default values."""
        # General settings
        self.colorscale_input.setCurrentText(self.config_manager.config_bu.get('colorscale'))
        self.set_coordinate_format(self.config_manager.config_bu.get('decimal'))
        self.auto_precision_checkbox.setChecked(
            bool(self.config_manager.config_bu.get('auto_precision_digits', True))
        )
        self.number_decimals_input.setValue(self.config_manager.config_bu.get('number_decimals'))
        self._on_auto_precision_toggled(self.auto_precision_checkbox.isChecked())
        self.wrap_angle_input.setValue(self.config_manager.config_bu.get('coord_wrap'))
        self.scrollspeed_input.setValue(self.config_manager.config_bu.get('scrollspeed'))
        self.invert_wheel_direction_checkbox.setChecked(
            bool(self.config_manager.config_bu.get('invert_wheel_direction', False))
        )
        self.range_file_input.setText(self.config_manager.config_bu.get('range_file'))
        self.startup_show_subwindow1_checkbox.setChecked(
            bool(
                self.config_manager.config_bu.get(
                    'startup_show_subwindow1',
                    self.config_manager.default_config.get('startup_show_subwindow1', True),
                )
            )
        )
        self.startup_show_subwindow2_checkbox.setChecked(
            bool(
                self.config_manager.config_bu.get(
                    'startup_show_subwindow2',
                    self.config_manager.default_config.get('startup_show_subwindow2', False),
                )
            )
        )
        
        # Display settings
        self.fig_background_color_input.setCurrentText(self.config_manager.config_bu.get('fig_background_color'))
        self.ax_background_color_input.setCurrentText(self.config_manager.config_bu.get('ax_background_color'))
        self.bad_color_input.setCurrentText(self.config_manager.config_bu.get('bad_color'))
        self.figure_pos_x_input.setValue(self.config_manager.config_bu.get('figure_pos_x'))
        self.figure_pos_y_input.setValue(self.config_manager.config_bu.get('figure_pos_y'))
        self.figure_width_input.setValue(self.config_manager.config_bu.get('figure_width'))
        self.figure_height_input.setValue(self.config_manager.config_bu.get('figure_height'))
        self.axislabel_fontsize_input.setValue(self.config_manager.config_bu.get('axislabel_fontsize'))
        self.axislabel_fontfamily_input.setCurrentText(self.config_manager.config_bu.get('axislabel_fontfamily'))
        self.axislabel_color_input.setCurrentText(self.config_manager.config_bu.get('axislabel_color'))
        self._load_grid_controls(self.config_manager.config_bu)
        
        # Axis positions
        self.axis_left_spinbox.setValue(self.config_manager.config_bu.get('ax_pos_l'))
        self.axis_right_spinbox.setValue(self.config_manager.config_bu.get('ax_pos_r'))
        self.axis_top_spinbox.setValue(self.config_manager.config_bu.get('ax_pos_t'))
        self.axis_bottom_spinbox.setValue(self.config_manager.config_bu.get('ax_pos_b'))
        
        #Beam
        self.beam_facecolor_input.setCurrentText(self.config_manager.config_bu.get('beam_facecolor'))
        self.beam_edgecolor_input.setCurrentText(self.config_manager.config_bu.get('beam_edgecolor'))
        self.beam_linewidth_input.setValue(self.config_manager.config_bu.get('beam_linewidth'))
        self.beam_pos_x_spinbox.setValue(self.config_manager.config_bu.get('beam_pos_x'))
        self.beam_pos_y_spinbox.setValue(self.config_manager.config_bu.get('beam_pos_y'))
        
        # Colorbar settings
        self.cbar_orientation_combo.setCurrentText(self.config_manager.config_bu.get('colorbar_orientation'))
        self.cbar_x_input.setValue(self.config_manager.config_bu.get('cbar_pos_x'))
        self.cbar_y_input.setValue(self.config_manager.config_bu.get('cbar_pos_y'))
        self.cbar_width_input.setValue(self.config_manager.config_bu.get('cbar_width'))
        self.cbar_height_input.setValue(self.config_manager.config_bu.get('cbar_height'))
        self.cbar_auto_layout_checkbox.setChecked(bool(self.config_manager.config_bu.get('colorbar_auto_layout', True)))
        self.cbar_placement_combo.setCurrentText(self.config_manager.config_bu.get('colorbar_placement', 'right'))
        self.cbar_align_combo.setCurrentText(self.config_manager.config_bu.get('colorbar_align', 'center'))
        self.cbar_gap_x_px_input.setValue(float(self.config_manager.config_bu.get('colorbar_gap_x_px', self.config_manager.config_bu.get('colorbar_gap_px', 24.0))))
        self.cbar_gap_y_px_input.setValue(float(self.config_manager.config_bu.get('colorbar_gap_y_px', self.config_manager.config_bu.get('colorbar_gap_px', 24.0))))
        self.cbar_thickness_px_input.setValue(float(self.config_manager.config_bu.get('colorbar_thickness_px', 24.0)))
        self.cbar_length_mode_combo.setCurrentText(
            self._normalize_colorbar_length_mode(
                self.config_manager.config_bu.get('colorbar_length_mode', 'ratio')
            )
        )
        self.cbar_length_value_input.setValue(float(self.config_manager.config_bu.get('colorbar_length_value', 1.0)))
        
        self.cbar_tick_direction_combo.setCurrentText(self.config_manager.config_bu.get('colorbar_tick_direction'))
        
        self.cbar_tick_length_input.setValue(self.config_manager.config_bu.get('colorbar_tick_length'))
        self.cbar_tick_width_input.setValue(self.config_manager.config_bu.get('colorbar_tick_width'))
        
        self.cbar_mtick_freq_input.setValue(self.config_manager.config_bu.get('colorbar_mtick_freq'))
        self.cbar_mtick_length_input.setValue(self.config_manager.config_bu.get('colorbar_mtick_length'))
        loaded_cbar_label = self.config_manager.config_bu.get('colorbar_label', '')
        if loaded_cbar_label is None:
            loaded_cbar_label = ''
        self.cbar_label_text_input.setText(str(loaded_cbar_label))
        self.cbar_label_fontsize_input.setValue(self.config_manager.config_bu.get('colorbar_label_fontsize', 12))
        self.cbar_label_color_input.setCurrentText(self.config_manager.config_bu.get('colorbar_label_color', 'black'))
        self.cbar_tick_color_input.setCurrentText(self.config_manager.config_bu.get('colorbar_tick_color'))
        self.cbar_tick_label_color_input.setCurrentText(self.config_manager.config_bu.get('colorbar_tick_labelcolor'))
        self._load_colorbar_tick_side_comboboxes(self.config_manager.config_bu)
        self._on_colorbar_auto_layout_toggled(self.cbar_auto_layout_checkbox.isChecked())
        self._update_colorbar_inside_preset_enabled(
            self.cbar_placement_combo.currentText()
        )
        
        # Ticks settings
        self.tick_labelsize_input.setValue(self.config_manager.config_bu.get('tick_labelsize'))
        self.tick_color_input.setCurrentText(self.config_manager.config_bu.get('tick_color'))
        self.tick_labelcolor_input.setCurrentText(self.config_manager.config_bu.get('tick_labelcolor'))
        self.tick_direction_input.setCurrentText(self.config_manager.config_bu.get('tick_direction'))
        self.default_ticks_position_input.setCurrentText(self.config_manager.config_bu.get('default_ticks_position'))
        self.xticklabel_position_input.setCurrentText(self.config_manager.config_bu.get('xticklabel_position'))
        self.yticklabel_position_input.setCurrentText(self.config_manager.config_bu.get('yticklabel_position'))
        self.tick_length_input.setValue(self.config_manager.config_bu.get('tick_length'))
        self.mtick_length_input.setValue(self.config_manager.config_bu.get('mtick_length'))
        self.tick_width_input.setValue(self.config_manager.config_bu.get('tick_width'))
        self.tick_xlabelrotation_input.setValue(self.config_manager.config_bu.get('tick_xlabelrotation'))
        self.tick_ylabelrotation_input.setValue(self.config_manager.config_bu.get('tick_ylabelrotation'))
        self.tick_xpad_input.setValue(self.config_manager.config_bu.get('tick_pad_x'))
        self.tick_ypad_input.setValue(self.config_manager.config_bu.get('tick_pad_y'))
        self.ticklabel_fontfamily_input.setCurrentText(self.config_manager.config_bu.get('tick_font'))
        
        self.x_mtick_freq_input.setValue(self.config_manager.config_bu.get('x_mtick_freq'))
        self.y_mtick_freq_input.setValue(self.config_manager.config_bu.get('y_mtick_freq'))
        self.z_mtick_freq_input.setValue(self.config_manager.config_bu.get('z_mtick_freq'))
        
        # Click settings
        self.click_label_color_input.setCurrentText(self.config_manager.config_bu.get('click_label_color'))
        self.click_linewidth_input.setValue(self.config_manager.config_bu.get('click_linewidth'))
        self.click_linecolor_input.setCurrentText(self.config_manager.config_bu.get('click_linecolor'))
        self._set_click_linestyle_combo(self.config_manager.config_bu.get('click_linestyle', '-'))
        self.click_alpha_input.setValue(float(self.config_manager.config_bu.get('click_alpha', 1.0)))
        self.click_show_crosshair_checkbox.setChecked(
            bool(self.config_manager.config_bu.get('click_show_crosshair', True))
        )
        self.click_crosshair_mode_input.setCurrentText(self.config_manager.config_bu.get('click_crosshair_mode', 'both'))
        self.click_show_center_marker_checkbox.setChecked(
            bool(self.config_manager.config_bu.get('click_show_center_marker', False))
        )
        self._on_click_show_crosshair_toggled(self.click_show_crosshair_checkbox.isChecked())
        self.poslabel_x_input.setValue(self.config_manager.config_bu.get('poslabel_x'))
        self.poslabel_y_input.setValue(self.config_manager.config_bu.get('poslabel_y'))
        self.poslabel_w_input.setValue(self.config_manager.config_bu.get('poslabel_w'))
        self.poslabel_h_input.setValue(self.config_manager.config_bu.get('poslabel_h'))
        
        self.ch_label_color_input.setCurrentText(self.config_manager.config_bu.get('ch_label_color'))
        self.ch_label_size_input.setValue(self.config_manager.config_bu.get('ch_label_size'))
        self.ch_label_font_input.setCurrentText(self.config_manager.config_bu.get('ch_label_font'))
        self.pos_chlabel_x_input.setValue(self.config_manager.config_bu.get('pos_chlabel_x'))
        self.pos_chlabel_y_input.setValue(self.config_manager.config_bu.get('pos_chlabel_y'))
        #self.pos_chlabel_w_input.setValue(self.config_manager.config_bu.get('pos_chlabel_w'))
        #self.pos_chlabel_h_input.setValue(self.config_manager.config_bu.get('pos_chlabel_h'))


    def reset_to_default(self):
        """Reset the configuration to default values."""
        #self.config_manager.reset_to_default()
        # General settings
        self.colorscale_input.setCurrentText(self.config_manager.default_config.get('colorscale'))
        self.set_coordinate_format(self.config_manager.default_config.get('decimal'))
        self.auto_precision_checkbox.setChecked(
            bool(self.config_manager.default_config.get('auto_precision_digits', True))
        )
        self.number_decimals_input.setValue(self.config_manager.default_config.get('number_decimals'))
        self._on_auto_precision_toggled(self.auto_precision_checkbox.isChecked())
        self.wrap_angle_input.setValue(self.config_manager.default_config.get('coord_wrap'))
        self.scrollspeed_input.setValue(self.config_manager.default_config.get('scrollspeed'))
        self.invert_wheel_direction_checkbox.setChecked(
            bool(self.config_manager.default_config.get('invert_wheel_direction', False))
        )
        self.range_file_input.setText(self.config_manager.default_config.get('range_file'))
        self.startup_show_subwindow1_checkbox.setChecked(
            bool(self.config_manager.default_config.get('startup_show_subwindow1', True))
        )
        self.startup_show_subwindow2_checkbox.setChecked(
            bool(self.config_manager.default_config.get('startup_show_subwindow2', False))
        )
        
        # Display settings
        self.fig_background_color_input.setCurrentText(self.config_manager.default_config.get('fig_background_color'))
        self.ax_background_color_input.setCurrentText(self.config_manager.default_config.get('ax_background_color'))
        self.bad_color_input.setCurrentText(self.config_manager.default_config.get('bad_color'))
        self.figure_pos_x_input.setValue(self.config_manager.default_config.get('figure_pos_x'))
        self.figure_pos_y_input.setValue(self.config_manager.default_config.get('figure_pos_y'))
        self.figure_width_input.setValue(self.config_manager.default_config.get('figure_width'))
        self.figure_height_input.setValue(self.config_manager.default_config.get('figure_height'))
        self.axislabel_fontsize_input.setValue(self.config_manager.default_config.get('axislabel_fontsize'))
        self.axislabel_fontfamily_input.setCurrentText(self.config_manager.default_config.get('axislabel_fontfamily'))
        self.axislabel_color_input.setCurrentText(self.config_manager.default_config.get('axislabel_color'))
        self._load_grid_controls(self.config_manager.default_config)
        
        # Axis positions
        self.axis_left_spinbox.setValue(self.config_manager.default_config.get('ax_pos_l'))
        self.axis_right_spinbox.setValue(self.config_manager.default_config.get('ax_pos_r'))
        self.axis_top_spinbox.setValue(self.config_manager.default_config.get('ax_pos_t'))
        self.axis_bottom_spinbox.setValue(self.config_manager.default_config.get('ax_pos_b'))
        
        #Beam
        self.beam_facecolor_input.setCurrentText(self.config_manager.default_config.get('beam_facecolor'))
        self.beam_edgecolor_input.setCurrentText(self.config_manager.default_config.get('beam_edgecolor'))
        self.beam_linewidth_input.setValue(self.config_manager.default_config.get('beam_linewidth'))
        self.beam_pos_x_spinbox.setValue(self.config_manager.default_config.get('beam_pos_x'))
        self.beam_pos_y_spinbox.setValue(self.config_manager.default_config.get('beam_pos_y'))
        
        # Colorbar settings
        self.cbar_orientation_combo.setCurrentText(self.config_manager.default_config.get('colorbar_orientation'))
        self.cbar_x_input.setValue(self.config_manager.default_config.get('cbar_pos_x'))
        self.cbar_y_input.setValue(self.config_manager.default_config.get('cbar_pos_y'))
        self.cbar_width_input.setValue(self.config_manager.default_config.get('cbar_width'))
        self.cbar_height_input.setValue(self.config_manager.default_config.get('cbar_height'))
        self.cbar_auto_layout_checkbox.setChecked(bool(self.config_manager.default_config.get('colorbar_auto_layout', True)))
        self.cbar_placement_combo.setCurrentText(self.config_manager.default_config.get('colorbar_placement', 'right'))
        self.cbar_align_combo.setCurrentText(self.config_manager.default_config.get('colorbar_align', 'center'))
        self.cbar_gap_x_px_input.setValue(float(self.config_manager.default_config.get('colorbar_gap_x_px', self.config_manager.default_config.get('colorbar_gap_px', 24.0))))
        self.cbar_gap_y_px_input.setValue(float(self.config_manager.default_config.get('colorbar_gap_y_px', self.config_manager.default_config.get('colorbar_gap_px', 24.0))))
        self.cbar_thickness_px_input.setValue(float(self.config_manager.default_config.get('colorbar_thickness_px', 24.0)))
        self.cbar_length_mode_combo.setCurrentText(
            self._normalize_colorbar_length_mode(
                self.config_manager.default_config.get('colorbar_length_mode', 'ratio')
            )
        )
        self.cbar_length_value_input.setValue(float(self.config_manager.default_config.get('colorbar_length_value', 1.0)))
        
        self.cbar_tick_direction_combo.setCurrentText(self.config_manager.default_config.get('colorbar_tick_direction'))
        
        self.cbar_tick_length_input.setValue(self.config_manager.default_config.get('colorbar_tick_length'))
        self.cbar_tick_width_input.setValue(self.config_manager.default_config.get('colorbar_tick_width'))
        
        self.cbar_mtick_freq_input.setValue(self.config_manager.default_config.get('colorbar_mtick_freq'))
        self.cbar_mtick_length_input.setValue(self.config_manager.default_config.get('colorbar_mtick_length'))
        default_cbar_label = self.config_manager.default_config.get('colorbar_label', '')
        if default_cbar_label is None:
            default_cbar_label = ''
        self.cbar_label_text_input.setText(str(default_cbar_label))
        self.cbar_label_fontsize_input.setValue(self.config_manager.default_config.get('colorbar_label_fontsize', 12))
        self.cbar_label_color_input.setCurrentText(self.config_manager.default_config.get('colorbar_label_color', 'black'))
        self.cbar_tick_color_input.setCurrentText(self.config_manager.default_config.get('colorbar_tick_color'))
        self.cbar_tick_label_color_input.setCurrentText(self.config_manager.default_config.get('colorbar_tick_labelcolor'))
        self._load_colorbar_tick_side_comboboxes(self.config_manager.default_config)
        self._on_colorbar_auto_layout_toggled(self.cbar_auto_layout_checkbox.isChecked())
        self._update_colorbar_inside_preset_enabled(
            self.cbar_placement_combo.currentText()
        )
        
        
        # Ticks settings
        self.tick_labelsize_input.setValue(self.config_manager.default_config.get('tick_labelsize'))
        self.tick_color_input.setCurrentText(self.config_manager.default_config.get('tick_color'))
        self.tick_labelcolor_input.setCurrentText(self.config_manager.default_config.get('tick_labelcolor'))
        self.tick_direction_input.setCurrentText(self.config_manager.default_config.get('tick_direction'))
        self.default_ticks_position_input.setCurrentText(self.config_manager.default_config.get('default_ticks_position'))
        self.xticklabel_position_input.setCurrentText(self.config_manager.default_config.get('xticklabel_position'))
        self.yticklabel_position_input.setCurrentText(self.config_manager.default_config.get('yticklabel_position'))
        self.tick_length_input.setValue(self.config_manager.default_config.get('tick_length'))
        self.mtick_length_input.setValue(self.config_manager.default_config.get('mtick_length'))
        self.tick_width_input.setValue(self.config_manager.default_config.get('tick_width'))
        self.tick_xlabelrotation_input.setValue(self.config_manager.default_config.get('tick_xlabelrotation'))
        self.tick_ylabelrotation_input.setValue(self.config_manager.default_config.get('tick_ylabelrotation'))
        self.tick_xpad_input.setValue(self.config_manager.default_config.get('tick_pad_x'))
        self.tick_ypad_input.setValue(self.config_manager.default_config.get('tick_pad_y'))
        self.ticklabel_fontfamily_input.setCurrentText(self.config_manager.default_config.get('tick_font'))
        self.x_mtick_freq_input.setValue(self.config_manager.default_config.get('x_mtick_freq'))
        self.y_mtick_freq_input.setValue(self.config_manager.default_config.get('y_mtick_freq'))
        self.z_mtick_freq_input.setValue(self.config_manager.default_config.get('z_mtick_freq'))
        
        # Click settings
        self.click_label_color_input.setCurrentText(self.config_manager.default_config.get('click_label_color'))
        self.click_linewidth_input.setValue(self.config_manager.default_config.get('click_linewidth'))
        self.click_linecolor_input.setCurrentText(self.config_manager.default_config.get('click_linecolor'))
        self._set_click_linestyle_combo(self.config_manager.default_config.get('click_linestyle', '-'))
        self.click_alpha_input.setValue(float(self.config_manager.default_config.get('click_alpha', 1.0)))
        self.click_show_crosshair_checkbox.setChecked(
            bool(self.config_manager.default_config.get('click_show_crosshair', True))
        )
        self.click_crosshair_mode_input.setCurrentText(self.config_manager.default_config.get('click_crosshair_mode', 'both'))
        self.click_show_center_marker_checkbox.setChecked(
            bool(self.config_manager.default_config.get('click_show_center_marker', False))
        )
        self._on_click_show_crosshair_toggled(self.click_show_crosshair_checkbox.isChecked())
        self.poslabel_x_input.setValue(self.config_manager.default_config.get('poslabel_x'))
        self.poslabel_y_input.setValue(self.config_manager.default_config.get('poslabel_y'))
        self.poslabel_w_input.setValue(self.config_manager.default_config.get('poslabel_w'))
        self.poslabel_h_input.setValue(self.config_manager.default_config.get('poslabel_h'))
        
        self.ch_label_color_input.setCurrentText(self.config_manager.default_config.get('ch_label_color'))
        self.ch_label_size_input.setValue(self.config_manager.default_config.get('ch_label_size'))
        self.ch_label_font_input.setCurrentText(self.config_manager.default_config.get('ch_label_font'))
        self.pos_chlabel_x_input.setValue(self.config_manager.default_config.get('pos_chlabel_x'))
        self.pos_chlabel_y_input.setValue(self.config_manager.default_config.get('pos_chlabel_y'))
        #self.pos_chlabel_w_input.setValue(self.config_manager.default_config.get('pos_chlabel_w'))
        #self.pos_chlabel_h_input.setValue(self.config_manager.default_config.get('pos_chlabel_h'))
