"""Baseline subtraction tool panel."""
from types import SimpleNamespace

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QButtonGroup,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from takefits.core.coordinate import CoordinateConverter
from takefits.core.region import CircleRegion, EllipseRegion, RectangleRegion
from takefits.ui.widget_sizing import fit_button_to_text
from takefits.core.wcs_frames import (
    normalize_display_frame,
    plane_values_for_display,
    axis_is_longitude,
    axis_is_latitude,
)
from takefits.core.history_provenance import build_processing_history_lines
from takefits.core.app_state import RegionSpec
from takefits.core.action_session import ActionSession
from takefits.core.usecases import (
    compute_polynomial_baseline_subtraction,
    export_baseline_model_fits,
    get_averaged_spectrum,
)
from takefits.ui.save_fits_dialog import SaveFITS
from takefits.tools.base_panel import BaseToolPanel, confirm_pending_close


class BaselinePanel(BaseToolPanel):
    """Tool panel for polynomial baseline subtraction."""

    def __init__(self, fits_viewer, subwindows=None):
        self.baseline_range_rows = []
        self.baseline_model_data = None
        self._last_subtracted_data = None
        self._previous_data_snapshot = None
        self._active_plot_source_data = None
        self._initialized_on_show = False
        self.spectrum = None
        self.x = 0
        self.y = 0
        self.z = 0
        self.active_region = None
        self.spec_axis = int(getattr(fits_viewer.wcs.wcs, "spec", 2))
        self.velocity_values = self._spectral_world_values_for_viewer(fits_viewer, self.spec_axis)
        self.converter = CoordinateConverter(fits_viewer.wcs, fits_viewer.displaymap.config)

        self._range_patches = []
        self._drag_preview_patch = None
        self._drag_start_world = None
        self._range_drag_state = None
        self._overlay_retry_pending = False
        self._region_signal_connected = False
        self._has_pending_changes = False
        self._line_color_default = "blue"
        self._line_color_subtracted = "#d9480f"

        super().__init__(fits_viewer, subwindows=subwindows)

        manager = getattr(self.fits_viewer, "region_manager", None)
        if manager is not None and hasattr(manager, "selected_region_changed"):
            try:
                manager.selected_region_changed.connect(self.on_region_changed)
                self._region_signal_connected = True
            except Exception:
                self._region_signal_connected = False

    def initUI(self):
        self.setWindowTitle(f"Baseline: {self.fits_viewer.filename}")
        self.resize(900, 300)

        layout = QHBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 8, 8, 8)

        controls_wrap = QWidget(self)
        controls_wrap.setMaximumWidth(430)
        controls_outer = QVBoxLayout(controls_wrap)
        controls_outer.setSpacing(0)
        controls_outer.setContentsMargins(0, 0, 0, 0)

        baseline_group = QGroupBox("Polynomial Baseline Subtraction")
        baseline_layout = QVBoxLayout()
        baseline_layout.setSpacing(3)

        top_row = QHBoxLayout()
        top_row.setSpacing(4)
        top_row.addWidget(QLabel("Order:"))

        self.baseline_order_spinbox = QSpinBox()
        self.baseline_order_spinbox.setRange(0, 9)
        self.baseline_order_spinbox.setValue(1)
        self.baseline_order_spinbox.setFixedWidth(46)
        top_row.addWidget(self.baseline_order_spinbox)

        self.clear_ranges_button = QPushButton("Clear Ranges")
        fit_button_to_text(self.clear_ranges_button, minimum_width=104)
        self.clear_ranges_button.clicked.connect(self._clear_ranges)
        top_row.addWidget(self.clear_ranges_button)

        self._range_icon_button_width = 30
        self.baseline_add_range_button = QPushButton("+")
        self.baseline_add_range_button.setFixedWidth(self._range_icon_button_width)
        self.baseline_add_range_button.clicked.connect(self.add_baseline_range_row)
        top_row.addWidget(self.baseline_add_range_button)
        top_row.addStretch(1)
        baseline_layout.addLayout(top_row)

        self._range_field_width = 91
        probe_to = QLabel("to", self)
        self._range_to_width = max(12, int(probe_to.sizeHint().width()))
        probe_to.deleteLater()

        self.baseline_ranges_layout = QVBoxLayout()
        self.baseline_ranges_layout.setSpacing(0)
        self.baseline_ranges_layout.setContentsMargins(0, 0, 0, 0)
        baseline_layout.addLayout(self.baseline_ranges_layout)

        action_row = QHBoxLayout()
        action_row.setSpacing(3)

        self.apply_baseline_button = QPushButton("Execute")
        self.apply_baseline_button.clicked.connect(self.apply_polynomial_baseline_subtraction)
        action_row.addWidget(self.apply_baseline_button)

        self.reset_button = QPushButton("Reset")
        self.reset_button.setEnabled(False)
        self.reset_button.clicked.connect(self.reset_baseline_result)

        self.save_subtracted_button = QPushButton("Save as FITS")
        self.save_subtracted_button.setEnabled(False)
        self.save_subtracted_button.clicked.connect(self.save_subtracted_fits)
        action_row.addWidget(self.save_subtracted_button)
        baseline_layout.addLayout(action_row)

        aux_row = QHBoxLayout()
        aux_row.setSpacing(3)
        aux_row.addWidget(self.reset_button)

        self.save_baseline_model_button = QPushButton("Save Model")
        self.save_baseline_model_button.setEnabled(False)
        self.save_baseline_model_button.clicked.connect(self.save_baseline_model_fits)
        aux_row.addWidget(self.save_baseline_model_button)
        baseline_layout.addLayout(aux_row)

        baseline_group.setLayout(baseline_layout)
        controls_outer.addWidget(baseline_group)
        controls_outer.addStretch(1)

        plot_wrap = QWidget(self)
        plot_layout = QVBoxLayout(plot_wrap)
        plot_layout.setSpacing(3)
        plot_layout.setContentsMargins(0, 0, 0, 0)

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.canvas.mpl_connect("button_press_event", self._on_plot_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_plot_motion)
        self.canvas.mpl_connect("button_release_event", self._on_plot_release)

        self.ax.axhline(y=0.0, color="gray", linewidth=0.5)
        self.line, = self.ax.step([], [], where="mid", color=self._line_color_default, linewidth=1.2, zorder=5)
        self.model_line, = self.ax.step(
            [],
            [],
            where="mid",
            color="crimson",
            linewidth=1.2,
            linestyle="-",
            zorder=6,
        )
        self.model_line.set_visible(False)
        self.cursor_line = self.ax.axvline(x=0.0, color="cyan", linestyle="-", linewidth=0.75, zorder=7)

        self.ax.set_xlabel(str(getattr(self.fits_viewer.displaymap, "third_axis_label", "Spectral Axis")))
        self.ax.set_ylabel(f"Intensity [{self.fits_viewer.bunit}]")
        self.ax.set_title("Spectrum", loc="left")

        display_row = QHBoxLayout()
        display_row.setSpacing(6)
        display_row.addWidget(QLabel("Display:"))
        self.display_mode_group = QButtonGroup(self)
        self.display_mode_group.setExclusive(True)

        self.show_fit_result_radio = QRadioButton("Show Fit Result")
        self.show_fit_result_radio.setToolTip("Display the original spectrum with the fitted baseline model.")
        self.show_fit_result_radio.setChecked(True)
        self.show_fit_result_radio.setEnabled(False)
        self.show_fit_result_radio.toggled.connect(self._on_display_mode_toggled)
        self.display_mode_group.addButton(self.show_fit_result_radio)
        display_row.addWidget(self.show_fit_result_radio)

        self.show_subtracted_radio = QRadioButton("Show Subtracted")
        self.show_subtracted_radio.setToolTip("Display the baseline-subtracted spectrum.")
        self.show_subtracted_radio.setEnabled(False)
        self.show_subtracted_radio.toggled.connect(self._on_display_mode_toggled)
        self.display_mode_group.addButton(self.show_subtracted_radio)
        display_row.addWidget(self.show_subtracted_radio)
        display_row.addStretch(1)

        plot_layout.addWidget(self.canvas, 1)
        plot_layout.addWidget(self.toolbar, 0)
        plot_layout.addLayout(display_row)

        layout.addWidget(controls_wrap, 0)
        layout.addWidget(plot_wrap, 1)

        self._seed_default_baseline_ranges()
        self._apply_last_baseline_state_from_metadata()

        self._update_controls_enabled()

    def showEvent(self, event):
        super().showEvent(event)
        if self._initialized_on_show:
            return
        self._initialized_on_show = True
        QTimer.singleShot(0, self._initialize_from_viewer_state)

    def _initialize_from_viewer_state(self):
        x = 0
        y = 0
        z = 0
        if bool(getattr(self.fits_viewer, "region_mode_enabled", False)):
            manager = getattr(self.fits_viewer, "region_manager", None)
            if manager is not None:
                self.active_region = getattr(manager, "selected_region", None)
        try:
            getter_x = getattr(self.fits_viewer, "_get_shared_xpix", None)
            getter_y = getattr(self.fits_viewer, "_get_shared_ypix", None)
            getter_z = getattr(self.fits_viewer, "_get_shared_zpix", None)
            if callable(getter_x):
                x = int(getter_x())
            if callable(getter_y):
                y = int(getter_y())
            if callable(getter_z):
                z = int(getter_z())
        except Exception:
            x = 0
            y = 0
            z = 0
        self.update_spectrum(x, y, z)

    def _cube_data_available(self) -> bool:
        data = getattr(self.fits_viewer, "data", None)
        return bool(hasattr(data, "ndim") and int(data.ndim) >= 3)

    def _update_controls_enabled(self):
        enabled = self._cube_data_available()

        for control in (
            self.baseline_order_spinbox,
            self.baseline_add_range_button,
            self.clear_ranges_button,
            self.apply_baseline_button,
        ):
            control.setEnabled(enabled)

        self.reset_button.setEnabled(enabled and self._last_subtracted_data is not None)
        self.save_subtracted_button.setEnabled(enabled and self._last_subtracted_data is not None)
        self.save_baseline_model_button.setEnabled(enabled and self.baseline_model_data is not None)
        has_pre_subtracted = bool(enabled and self._active_plot_source_data is not None)
        self.show_fit_result_radio.setEnabled(has_pre_subtracted)
        self.show_subtracted_radio.setEnabled(has_pre_subtracted)
        if not has_pre_subtracted and self.show_fit_result_radio.isChecked():
            self.show_subtracted_radio.setChecked(True)

        for row in self.baseline_range_rows:
            row["min_edit"].setEnabled(enabled)
            row["max_edit"].setEnabled(enabled)
            row["remove_button"].setEnabled(enabled and len(self.baseline_range_rows) > 1)

        if not enabled:
            self._drag_start_world = None
            self._range_drag_state = None
            self._clear_drag_preview(redraw=False)

    def _on_display_mode_toggled(self, _checked):
        self.update_spectrum(self.x, self.y, self.z)

    def _show_pre_subtracted(self) -> bool:
        return bool(
            self._active_plot_source_data is not None
            and getattr(self, "show_fit_result_radio", None) is not None
            and self.show_fit_result_radio.isChecked()
        )

    @staticmethod
    def _spectral_world_values_for_viewer(fits_viewer, spec_axis):
        data = getattr(fits_viewer, "data", None)
        if data is None or int(getattr(data, "ndim", 0)) < 3:
            return np.asarray([], dtype=float)
        n_channels = int(data.shape[-3])
        try:
            axis_id = int(spec_axis) + 1
            crval = float(fits_viewer.header[f"CRVAL{axis_id}"])
            cdelt = float(fits_viewer.header[f"CDELT{axis_id}"])
            crpix = float(fits_viewer.header[f"CRPIX{axis_id}"])
            return crval + (np.arange(n_channels, dtype=float) - (crpix - 1.0)) * cdelt
        except Exception:
            return np.arange(n_channels, dtype=float)

    def _default_baseline_edge_ranges(self):
        velocity = np.asarray(self.velocity_values, dtype=float).reshape(-1)
        finite = velocity[np.isfinite(velocity)]
        if finite.size < 2:
            return [(-5.0, -2.0), (2.0, 5.0)]
        lo = float(np.nanmin(finite))
        hi = float(np.nanmax(finite))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return [(-5.0, -2.0), (2.0, 5.0)]
        width = 0.2 * (hi - lo)
        if width <= 0:
            width = max(abs(lo), abs(hi), 1.0) * 0.1
        return [(lo, lo + width), (hi - width, hi)]

    def _seed_default_baseline_ranges(self):
        if self.baseline_range_rows:
            return
        for lo, hi in self._default_baseline_edge_ranges():
            self.add_baseline_range_row(f"{float(lo):.6g}", f"{float(hi):.6g}")

    def _apply_last_baseline_state_from_metadata(self):
        app_state = getattr(self.fits_viewer, "app_state", None)
        spectral_meta = getattr(app_state, "spectral_metadata", None)
        if not isinstance(spectral_meta, dict):
            return

        try:
            order = int(spectral_meta.get("baseline_last_order", self.baseline_order_spinbox.value()))
            order = max(int(self.baseline_order_spinbox.minimum()), min(int(self.baseline_order_spinbox.maximum()), order))
            self.baseline_order_spinbox.setValue(order)
        except Exception:
            pass

        world_ranges = spectral_meta.get("baseline_last_world_ranges")
        if not isinstance(world_ranges, list):
            return

        self._remove_all_range_rows()
        for pair in world_ranges:
            if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                continue
            try:
                lo = float(pair[0])
                hi = float(pair[1])
            except Exception:
                continue
            if not (np.isfinite(lo) and np.isfinite(hi)):
                continue
            self.add_baseline_range_row(f"{lo:.6g}", f"{hi:.6g}")

        if not self.baseline_range_rows:
            self._seed_default_baseline_ranges()

    def add_baseline_range_row(self, min_text="", max_text=""):
        row_widget = QWidget(self)
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(2)

        min_edit = QLineEdit(str(min_text), row_widget)
        min_edit.setPlaceholderText("min world")
        min_edit.setFixedWidth(int(getattr(self, "_range_field_width", 92)))
        min_edit.editingFinished.connect(self._on_ranges_edited)
        row_layout.addWidget(min_edit)

        to_label = QLabel("to", row_widget)
        to_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        to_label.setFixedWidth(int(getattr(self, "_range_to_width", 12)))
        row_layout.addWidget(to_label)

        max_edit = QLineEdit(str(max_text), row_widget)
        max_edit.setPlaceholderText("max world")
        max_edit.setFixedWidth(int(getattr(self, "_range_field_width", 92)))
        max_edit.editingFinished.connect(self._on_ranges_edited)
        row_layout.addWidget(max_edit)

        remove_button = QPushButton("-", row_widget)
        remove_button.setFixedWidth(int(getattr(self, "_range_icon_button_width", 30)))
        row_layout.addWidget(remove_button)
        row_layout.addStretch(1)

        row_entry = {
            "widget": row_widget,
            "min_edit": min_edit,
            "max_edit": max_edit,
            "remove_button": remove_button,
        }
        remove_button.clicked.connect(lambda _=False, target=row_entry: self._remove_baseline_range_row(target))
        self.baseline_range_rows.append(row_entry)
        self.baseline_ranges_layout.addWidget(row_widget)
        self._refresh_baseline_row_labels()
        self._update_controls_enabled()
        self._render_range_overlays()

    def _remove_baseline_range_row(self, row_entry):
        if row_entry not in self.baseline_range_rows:
            return
        if len(self.baseline_range_rows) <= 1:
            return
        self._range_drag_state = None
        self.baseline_range_rows.remove(row_entry)
        widget = row_entry.get("widget")
        if widget is not None:
            widget.setParent(None)
            widget.deleteLater()
        self._refresh_baseline_row_labels()
        self._update_controls_enabled()
        self._render_range_overlays()

    def _remove_all_range_rows(self):
        self._range_drag_state = None
        for row in list(self.baseline_range_rows):
            widget = row.get("widget")
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self.baseline_range_rows = []

    def _clear_ranges(self):
        self._remove_all_range_rows()
        self._seed_default_baseline_ranges()
        self._update_controls_enabled()
        self._render_range_overlays()

    def _refresh_baseline_row_labels(self):
        for row in self.baseline_range_rows:
            row["remove_button"].setEnabled(len(self.baseline_range_rows) > 1)

    def _on_ranges_edited(self):
        self._render_range_overlays()

    def _collect_baseline_world_ranges(self, *, strict=True):
        ranges = []
        for idx, row in enumerate(self.baseline_range_rows, start=1):
            text_lo = str(row["min_edit"].text() or "").strip()
            text_hi = str(row["max_edit"].text() or "").strip()
            if not text_lo and not text_hi:
                continue
            if not text_lo or not text_hi:
                if strict:
                    raise ValueError(f"Range {idx} is incomplete.")
                continue
            try:
                value_lo = float(text_lo)
                value_hi = float(text_hi)
            except Exception as exc:
                if strict:
                    raise ValueError(f"Range {idx} has non-numeric values.") from exc
                continue
            if not (np.isfinite(value_lo) and np.isfinite(value_hi)):
                if strict:
                    raise ValueError(f"Range {idx} must be finite.")
                continue
            lo, hi = (value_lo, value_hi) if value_lo <= value_hi else (value_hi, value_lo)
            ranges.append((float(lo), float(hi)))
        if strict and not ranges:
            raise ValueError("At least one baseline world range is required.")
        return ranges

    def _channel_to_world_value(self, channel):
        velocity = np.asarray(self.velocity_values, dtype=float).reshape(-1)
        if velocity.size == 0 or not np.all(np.isfinite(velocity)):
            return None
        channels = np.arange(velocity.size, dtype=float)
        ch = float(np.clip(channel, channels[0], channels[-1]))
        return float(np.interp(ch, channels, velocity))

    def _world_bounds(self):
        velocity = np.asarray(self.velocity_values, dtype=float).reshape(-1)
        finite = velocity[np.isfinite(velocity)]
        if finite.size <= 0:
            return (0.0, 0.0)
        return (float(np.nanmin(finite)), float(np.nanmax(finite)))

    def _minimum_world_width(self):
        lo, hi = self._world_bounds()
        span = abs(float(hi) - float(lo))
        if not np.isfinite(span) or span <= 0:
            return 1e-6
        return max(span * 1e-3, 1e-6)

    def _range_hit_tolerance_world(self):
        try:
            x0, x1 = self.ax.get_xlim()
            span = abs(float(x1) - float(x0))
        except Exception:
            span = 0.0
        if not np.isfinite(span) or span <= 0:
            span = 100.0
        floor = self._minimum_world_width()
        return float(np.clip(span * 0.012, floor, max(floor * 16.0, span)))

    def _row_world_ranges(self):
        rows = []
        for idx, row in enumerate(self.baseline_range_rows):
            text_lo = str(row["min_edit"].text() or "").strip()
            text_hi = str(row["max_edit"].text() or "").strip()
            if not text_lo or not text_hi:
                continue
            try:
                lo_w = float(text_lo)
                hi_w = float(text_hi)
            except Exception:
                continue
            if not (np.isfinite(lo_w) and np.isfinite(hi_w)):
                continue
            lo_w, hi_w = (lo_w, hi_w) if lo_w <= hi_w else (hi_w, lo_w)
            rows.append(
                {
                    "index": int(idx),
                    "row": row,
                    "lo_world": float(lo_w),
                    "hi_world": float(hi_w),
                }
            )
        return rows

    def _set_row_world_range(self, row, lo_world, hi_world):
        lo = float(lo_world)
        hi = float(hi_world)
        if hi < lo:
            lo, hi = hi, lo
        row["min_edit"].setText(f"{lo:.6g}")
        row["max_edit"].setText(f"{hi:.6g}")

    def _row_has_valid_range(self, row):
        text_lo = str(row["min_edit"].text() or "").strip()
        text_hi = str(row["max_edit"].text() or "").strip()
        if not text_lo or not text_hi:
            return False
        try:
            lo = float(text_lo)
            hi = float(text_hi)
        except Exception:
            return False
        return bool(np.isfinite(lo) and np.isfinite(hi))

    def _assign_to_first_pending_range_row(self, lo_world, hi_world):
        for row in self.baseline_range_rows:
            if self._row_has_valid_range(row):
                continue
            self._set_row_world_range(row, lo_world, hi_world)
            return True
        return False

    def _hit_test_existing_range(self, x_world):
        x = float(x_world)
        tol = self._range_hit_tolerance_world()
        candidates = self._row_world_ranges()
        if not candidates:
            return None

        best_edge = None
        best_edge_dist = None
        best_inside = None
        best_inside_dist = None
        for entry in candidates:
            lo = entry["lo_world"]
            hi = entry["hi_world"]
            d_lo = abs(x - lo)
            d_hi = abs(x - hi)
            if d_lo <= tol:
                if best_edge_dist is None or d_lo < best_edge_dist:
                    best_edge = (entry, "left")
                    best_edge_dist = d_lo
            if d_hi <= tol:
                if best_edge_dist is None or d_hi < best_edge_dist:
                    best_edge = (entry, "right")
                    best_edge_dist = d_hi
            if lo <= x <= hi:
                center = 0.5 * (lo + hi)
                d_center = abs(x - center)
                if best_inside_dist is None or d_center < best_inside_dist:
                    best_inside = (entry, "move")
                    best_inside_dist = d_center

        if best_edge is not None:
            return {"entry": best_edge[0], "mode": best_edge[1]}
        if best_inside is not None:
            return {"entry": best_inside[0], "mode": best_inside[1]}
        return None

    def _baseline_reference_pixel(self):
        data = getattr(self.fits_viewer, "data", None)
        if data is None:
            return None
        x_val = float(int(round(self.x)))
        y_val = float(int(round(self.y)))
        z_val = float(int(round(self.z)))
        if int(getattr(data, "ndim", 0)) >= 4:
            app_state = getattr(self.fits_viewer, "app_state", None)
            s_val = float(int(getattr(app_state, "current_s", 0))) if app_state is not None else 0.0
            return [x_val, y_val, z_val, s_val]
        return [x_val, y_val, z_val]

    def _clear_range_overlays(self, *, redraw=False):
        for patch in list(self._range_patches):
            try:
                patch.remove()
            except Exception:
                pass
        self._range_patches = []
        if redraw:
            self.canvas.draw_idle()

    def _clear_drag_preview(self, *, redraw=False):
        patch = self._drag_preview_patch
        self._drag_preview_patch = None
        if patch is not None:
            try:
                patch.remove()
            except Exception:
                pass
        if redraw:
            self.canvas.draw_idle()

    def _render_range_overlays(self):
        if not self._cube_data_available():
            return

        self._overlay_retry_pending = False
        self._clear_range_overlays(redraw=False)
        # Guard against singular transform during early/hidden draw phases.
        try:
            x0, x1 = self.ax.get_xlim()
            y0, y1 = self.ax.get_ylim()
            if not (np.isfinite(x0) and np.isfinite(x1)) or abs(float(x1) - float(x0)) < 1e-12:
                lo_w, hi_w = self._world_bounds()
                if np.isfinite(lo_w) and np.isfinite(hi_w) and abs(float(hi_w) - float(lo_w)) > 0:
                    self.ax.set_xlim(lo_w, hi_w)
                else:
                    self.ax.set_xlim(-0.5, 0.5)
            if not (np.isfinite(y0) and np.isfinite(y1)) or abs(float(y1) - float(y0)) < 1e-12:
                self.ax.set_ylim(-1.0, 1.0)
        except Exception:
            pass

        ranges = self._row_world_ranges()
        active_index = None
        if isinstance(self._range_drag_state, dict):
            try:
                active_index = int(self._range_drag_state.get("index"))
            except Exception:
                active_index = None
        needs_retry = False
        for entry in ranges:
            lo = float(entry["lo_world"])
            hi = float(entry["hi_world"])
            if active_index is not None and int(entry["index"]) == active_index:
                face = "#fab005"
                edge = "#e8590c"
                alpha = 0.25
            else:
                face = "#f59f00"
                edge = "#d9480f"
                alpha = 0.16
            try:
                patch = self.ax.axvspan(
                    lo,
                    hi,
                    facecolor=face,
                    edgecolor=edge,
                    alpha=alpha,
                    hatch="////",
                    linewidth=0.8,
                    zorder=2,
                )
            except Exception:
                needs_retry = True
                continue
            self._range_patches.append(patch)
        if needs_retry and not self._overlay_retry_pending:
            self._overlay_retry_pending = True
            QTimer.singleShot(60, self._retry_render_range_overlays)
        self.canvas.draw_idle()

    def _retry_render_range_overlays(self):
        self._overlay_retry_pending = False
        if not self._cube_data_available():
            return
        self._render_range_overlays()

    def _on_plot_press(self, event):
        if not self._cube_data_available():
            return
        if event.inaxes != self.ax:
            return
        if getattr(self.toolbar, "mode", "") != "":
            return
        if getattr(event, "button", None) != 1:
            return
        x_world = event.xdata
        if x_world is None or not np.isfinite(x_world):
            return

        hit = self._hit_test_existing_range(float(x_world))
        if hit is not None:
            entry = hit["entry"]
            self._range_drag_state = {
                "index": int(entry["index"]),
                "row": entry["row"],
                "mode": str(hit["mode"]),
                "start_world": float(x_world),
                "start_lo_world": float(entry["lo_world"]),
                "start_hi_world": float(entry["hi_world"]),
            }
            self._drag_start_world = None
            self._clear_drag_preview(redraw=False)
            self._render_range_overlays()
            return

        self._range_drag_state = None
        self._drag_start_world = float(x_world)
        self._clear_drag_preview(redraw=False)
        self._drag_preview_patch = self.ax.axvspan(
            self._drag_start_world,
            self._drag_start_world,
            facecolor="#74c0fc",
            edgecolor="#1864ab",
            alpha=0.18,
            hatch="////",
            linewidth=0.8,
            zorder=3,
        )
        self.canvas.draw_idle()

    def _on_plot_motion(self, event):
        if not self._cube_data_available():
            return
        state = self._range_drag_state
        if isinstance(state, dict):
            x_world = event.xdata
            if x_world is None or not np.isfinite(x_world):
                x_world = state["start_world"]
            x_now = float(x_world)

            lo_bound, hi_bound = self._world_bounds()
            min_width = self._minimum_world_width()
            mode = str(state.get("mode", "move"))
            start_lo = float(state["start_lo_world"])
            start_hi = float(state["start_hi_world"])
            start_x = float(state["start_world"])
            dx = x_now - start_x

            if mode == "left":
                new_lo = min(start_hi - min_width, start_lo + dx)
                new_hi = start_hi
            elif mode == "right":
                new_lo = start_lo
                new_hi = max(start_lo + min_width, start_hi + dx)
            else:
                width = max(min_width, start_hi - start_lo)
                new_lo = start_lo + dx
                new_hi = start_hi + dx
                if new_lo < lo_bound:
                    new_hi += lo_bound - new_lo
                    new_lo = lo_bound
                if new_hi > hi_bound:
                    new_lo -= new_hi - hi_bound
                    new_hi = hi_bound
                if new_hi - new_lo < width:
                    new_hi = min(hi_bound, new_lo + width)
                    new_lo = max(lo_bound, new_hi - width)

            new_lo = max(lo_bound, min(new_lo, hi_bound - min_width))
            new_hi = min(hi_bound, max(new_hi, lo_bound + min_width))
            if new_hi <= new_lo:
                new_hi = min(hi_bound, new_lo + min_width)

            row = state.get("row")
            if row is not None:
                self._set_row_world_range(row, float(new_lo), float(new_hi))
                self._render_range_overlays()
            return

        if self._drag_start_world is None:
            if event.inaxes == self.ax and getattr(self.toolbar, "mode", "") == "":
                x_world = event.xdata
                if x_world is not None and np.isfinite(x_world):
                    hit = self._hit_test_existing_range(float(x_world))
                    if hit is not None:
                        mode = str(hit.get("mode", "move"))
                        if mode in {"left", "right"}:
                            self.canvas.setCursor(Qt.CursorShape.SizeHorCursor)
                        else:
                            self.canvas.setCursor(Qt.CursorShape.SizeAllCursor)
                    else:
                        self.canvas.setCursor(Qt.CursorShape.CrossCursor)
                else:
                    self.canvas.setCursor(Qt.CursorShape.ArrowCursor)
            else:
                self.canvas.setCursor(Qt.CursorShape.ArrowCursor)

        if self._drag_start_world is None:
            return
        x_world = event.xdata
        if x_world is None or not np.isfinite(x_world):
            x_world = self._drag_start_world
        lo, hi = (
            (self._drag_start_world, float(x_world))
            if self._drag_start_world <= float(x_world)
            else (float(x_world), self._drag_start_world)
        )
        self._clear_drag_preview(redraw=False)
        self._drag_preview_patch = self.ax.axvspan(
            lo,
            hi,
            facecolor="#74c0fc",
            edgecolor="#1864ab",
            alpha=0.18,
            hatch="////",
            linewidth=0.8,
            zorder=3,
        )
        self.canvas.draw_idle()

    def _on_plot_release(self, event):
        if not self._cube_data_available():
            return
        state = self._range_drag_state
        if isinstance(state, dict):
            self._range_drag_state = None
            self._render_range_overlays()
            self.canvas.setCursor(Qt.CursorShape.ArrowCursor)
            return

        start = self._drag_start_world
        if start is None:
            return
        self._drag_start_world = None

        x_world = event.xdata
        if x_world is None or not np.isfinite(x_world):
            x_world = start
        end = float(x_world)

        self._clear_drag_preview(redraw=False)

        if abs(end - start) < self._minimum_world_width():
            self.canvas.draw_idle()
            self.canvas.setCursor(Qt.CursorShape.ArrowCursor)
            return

        lo_w, hi_w = (start, end) if start <= end else (end, start)
        if not self._assign_to_first_pending_range_row(float(lo_w), float(hi_w)):
            self.add_baseline_range_row(f"{float(lo_w):.6g}", f"{float(hi_w):.6g}")
        self._refresh_baseline_row_labels()
        self._render_range_overlays()
        self.canvas.setCursor(Qt.CursorShape.ArrowCursor)

    def update_position(self, x, y, z):
        self.update_spectrum(x, y, z)

    def refresh_coordinate_display(self):
        self._sync_coordinate_context()
        self.update_spectrum(self.x, self.y, self.z)

    def _sync_coordinate_context(self):
        wcs = getattr(self.fits_viewer, "wcs", None)
        if wcs is not None:
            self.converter.wcs = wcs
        displaymap = getattr(self.fits_viewer, "displaymap", None)
        config = getattr(displaymap, "config", None)
        if isinstance(config, dict):
            self.converter.config = config

    def _cube_from_data(self, data):
        if data is None or int(getattr(data, "ndim", 0)) < 3:
            return None
        if int(data.ndim) == 4:
            app_state = getattr(self.fits_viewer, "app_state", None)
            s_idx = int(getattr(app_state, "current_s", 0)) if app_state is not None else 0
            s_idx = max(0, min(s_idx, int(data.shape[0]) - 1))
            return data[s_idx]
        return data

    def _current_cube(self):
        data = getattr(self.fits_viewer, "data", None)
        return self._cube_from_data(data)

    def _current_plot_cube(self):
        show_pre_subtracted = self._show_pre_subtracted()
        if show_pre_subtracted:
            src_cube = self._cube_from_data(self._active_plot_source_data)
            if src_cube is not None:
                return src_cube
        return self._current_cube()

    def _current_model_cube(self):
        return self._cube_from_data(self.baseline_model_data)

    def _current_display_frame(self):
        getter = getattr(self.fits_viewer, "_get_shared_display_frame", None)
        if callable(getter):
            try:
                return normalize_display_frame(getter())
            except Exception:
                return "native"
        return "native"

    def _shared_native_world_values(self):
        values = []
        for getter_name in (
            "_get_shared_world_x",
            "_get_shared_world_y",
            "_get_shared_world_z",
            "_get_shared_world_s",
        ):
            getter = getattr(self.fits_viewer, getter_name, None)
            if not callable(getter):
                values.append(None)
                continue
            try:
                values.append(float(getter()))
            except Exception:
                values.append(None)
        return values

    def _format_title_world_coordinates(self, world_native):
        self._sync_coordinate_context()
        if world_native is None or len(world_native) < 2:
            raise ValueError("Insufficient world coordinates for title formatting.")

        native_x = float(world_native[0])
        native_y = float(world_native[1])
        axis_types = self.converter.get_axis_types()
        axis_x = axis_types[0] if len(axis_types) > 0 else ""
        axis_y = axis_types[1] if len(axis_types) > 1 else ""

        display_x = native_x
        display_y = native_y
        display_axis_x = axis_x
        display_axis_y = axis_y

        display_frame = self._current_display_frame()
        if display_frame != "native":
            fallback_native_world = list(world_native)
            shared_world = self._shared_native_world_values()
            for idx, value in enumerate(shared_world):
                if idx >= len(fallback_native_world):
                    break
                if value is not None:
                    fallback_native_world[idx] = value
            try:
                transformed = plane_values_for_display(
                    self.fits_viewer.wcs,
                    "xy",
                    native_x,
                    native_y,
                    frame=display_frame,
                    fallback_native_world=fallback_native_world,
                )
                display_x, display_y, axis_x_t, axis_y_t = transformed
                if axis_x_t:
                    display_axis_x = axis_x_t
                if axis_y_t:
                    display_axis_y = axis_y_t
            except Exception:
                pass

        def _apply_longitude_wrap(value, axis_type):
            axis_upper = str(axis_type or "").upper()
            wrapped = float(value)
            if axis_upper.startswith("RA"):
                if wrapped < 0.0:
                    wrapped += 360.0
                elif wrapped > 360.0:
                    wrapped -= 360.0
                return wrapped
            if "GLON" in axis_upper or axis_upper.endswith("LON"):
                wrap_mode = int(self.fits_viewer.displaymap.config.get("coord_wrap", 180))
                if wrap_mode == 180:
                    if wrapped < -180.0:
                        wrapped += 360.0
                    elif wrapped > 180.0:
                        wrapped -= 360.0
                else:
                    if wrapped < 0.0:
                        wrapped += 360.0
                    elif wrapped > 360.0:
                        wrapped -= 360.0
            return wrapped

        display_x = _apply_longitude_wrap(display_x, display_axis_x)
        display_y = _apply_longitude_wrap(display_y, display_axis_y)

        world_x_str = self.converter.format_world_coordinate(display_x, display_axis_x)
        world_y_str = self.converter.format_world_coordinate(display_y, display_axis_y)

        # Append " deg" unit if configured for decimal display and the axis is angular
        is_decimal = self.fits_viewer.displaymap.config.get("decimal", True)
        if is_decimal:
            if axis_is_longitude(display_axis_x) or axis_is_latitude(display_axis_x):
                world_x_str += " deg"
            if axis_is_longitude(display_axis_y) or axis_is_latitude(display_axis_y):
                world_y_str += " deg"

        return world_x_str, world_y_str

    def _region_center_pixel(self, region):
        if region is None or not hasattr(region, "get_state"):
            return None
        try:
            state = region.get_state() or {}
        except Exception:
            return None
        center = state.get("center")
        if isinstance(center, (list, tuple)) and len(center) >= 2:
            try:
                return (float(center[0]), float(center[1]))
            except Exception:
                return None
        return None

    def _ui_region_to_spec(self, region):
        if region is None or not hasattr(region, "get_state"):
            return None
        state = region.get_state()
        if isinstance(region, CircleRegion):
            cx, cy = state["center"]
            radius = state["radius"]
            return RegionSpec(type="circle", center_x=cx, center_y=cy, params={"radius": radius})
        if isinstance(region, RectangleRegion):
            cx, cy = state["center"]
            width, height = state["width"], state["height"]
            angle = state.get("angle", 0.0)
            return RegionSpec(
                type="rectangle",
                center_x=cx,
                center_y=cy,
                params={"width": width, "height": height, "angle": angle},
            )
        if isinstance(region, EllipseRegion):
            cx, cy = state["center"]
            width, height = state["width"], state["height"]
            angle = state.get("angle", 0.0)
            return RegionSpec(
                type="ellipse",
                center_x=cx,
                center_y=cy,
                params={"width": width, "height": height, "angle": angle},
            )
        return None

    def _calculate_average_spectrum(self, region, cube, *, update_velocity=True):
        region_spec = self._ui_region_to_spec(region)
        if region_spec is None:
            return None, "Unsupported region shape"

        state = SimpleNamespace(data=cube, wcs=self.fits_viewer.wcs)
        try:
            velocity, spectrum, _unit = get_averaged_spectrum(state, region_spec)
            if update_velocity and velocity is not None:
                self.velocity_values = np.asarray(velocity, dtype=float).reshape(-1)
        except Exception as exc:
            return None, f"Error calculating spectrum: {exc}"

        region_label = getattr(region, "label_text", "").strip()
        title_part1 = (
            f"Average Spectrum ({region_label})"
            if region_label
            else f"Average Spectrum (Region {getattr(region, 'region_id', '?')})"
        )
        title_part2 = ""
        center_pixel = self._region_center_pixel(region)
        if center_pixel is not None:
            try:
                if self.fits_viewer.data.ndim == 3:
                    world_native = self.fits_viewer.wcs.wcs_pix2world(
                        [[float(center_pixel[0]), float(center_pixel[1]), float(self.z)]],
                        0,
                    )[0]
                else:
                    world_native = self.fits_viewer.wcs.wcs_pix2world(
                        [[float(center_pixel[0]), float(center_pixel[1]), float(self.z), 0.0]],
                        0,
                    )[0]
                world_x_str, world_y_str = self._format_title_world_coordinates(world_native)
                title_part2 = f"around ({world_x_str}, {world_y_str})"
            except Exception:
                title_part2 = ""
        title = f"{title_part1}\n{title_part2}" if title_part2 else title_part1
        return np.asarray(spectrum, dtype=float).reshape(-1), title

    def on_region_changed(self, region):
        self.active_region = region
        self.update_spectrum(self.x, self.y, self.z)

    def _x_values_for_spectrum(self, length):
        if int(length) <= 0:
            return np.asarray([], dtype=float)
        velocity = np.asarray(self.velocity_values, dtype=float).reshape(-1)
        if velocity.size == int(length) and np.all(np.isfinite(velocity)):
            return velocity
        return np.arange(int(length), dtype=float)

    def update_spectrum(self, x, y, z):
        if not self._cube_data_available():
            self.spectrum = None
            self.line.set_data([], [])
            self.model_line.set_data([], [])
            self.model_line.set_visible(False)
            self.cursor_line.set_visible(False)
            self.ax.set_title("Baseline: unavailable for 2D data", loc="left")
            self.canvas.draw_idle()
            self._update_controls_enabled()
            return

        cube = self._current_plot_cube()
        if cube is None or cube.ndim < 3:
            return

        if not bool(getattr(self.fits_viewer, "region_mode_enabled", False)):
            self.active_region = None

        z_max = max(0, int(cube.shape[0]) - 1)
        y_max = max(0, int(cube.shape[1]) - 1)
        x_max = max(0, int(cube.shape[2]) - 1)
        self.x = max(0, min(int(round(float(x))), x_max))
        self.y = max(0, min(int(round(float(y))), y_max))
        self.z = max(0, min(int(round(float(z))), z_max))

        title = "Spectrum"
        if self.active_region is not None:
            spectrum, avg_title = self._calculate_average_spectrum(self.active_region, cube, update_velocity=True)
            if spectrum is not None:
                self.spectrum = spectrum
                title = avg_title
            else:
                self.spectrum = np.asarray(cube[:, self.y, self.x], dtype=float).reshape(-1)
                title = avg_title
        else:
            self.spectrum = np.asarray(cube[:, self.y, self.x], dtype=float).reshape(-1)
            try:
                if self.fits_viewer.data.ndim == 3:
                    world_native = self.fits_viewer.wcs.wcs_pix2world(
                        [[float(self.x), float(self.y), float(self.z)]],
                        0,
                    )[0]
                else:
                    world_native = self.fits_viewer.wcs.wcs_pix2world(
                        [[float(self.x), float(self.y), float(self.z), 0.0]],
                        0,
                    )[0]
                world_x_str, world_y_str = self._format_title_world_coordinates(world_native)
                title = f"Spectrum at ({world_x_str}, {world_y_str})"
            except Exception:
                title = "Spectrum"

        x_data = self._x_values_for_spectrum(self.spectrum.size)
        y_data = np.ma.masked_invalid(self.spectrum)
        self.line.set_data(x_data, y_data)
        show_pre_subtracted = self._show_pre_subtracted()
        if self._active_plot_source_data is not None and not show_pre_subtracted:
            self.line.set_color(self._line_color_subtracted)
        else:
            self.line.set_color(self._line_color_default)

        model_values = None
        if show_pre_subtracted:
            model_cube = self._current_model_cube()
            if model_cube is not None and model_cube.ndim >= 3:
                if self.active_region is not None:
                    model_values, _ = self._calculate_average_spectrum(
                        self.active_region,
                        model_cube,
                        update_velocity=False,
                    )
                if model_values is None:
                    y_idx = max(0, min(self.y, int(model_cube.shape[1]) - 1))
                    x_idx = max(0, min(self.x, int(model_cube.shape[2]) - 1))
                    model_values = np.asarray(model_cube[:, y_idx, x_idx], dtype=float).reshape(-1)

        if model_values is not None:
            model_values = np.asarray(model_values, dtype=float).reshape(-1)
            model_x = self._x_values_for_spectrum(model_values.size)
            self.model_line.set_data(model_x, np.ma.masked_invalid(model_values))
            self.model_line.set_visible(True)
        else:
            self.model_line.set_data([], [])
            self.model_line.set_visible(False)
        self.cursor_line.set_visible(True)
        cursor_x = self._channel_to_world_value(self.z)
        if cursor_x is None:
            if x_data.size > 0:
                cursor_index = int(max(0, min(self.z, int(x_data.size) - 1)))
                cursor_x = float(x_data[cursor_index])
            else:
                cursor_x = float(self.z)
        self.cursor_line.set_xdata([cursor_x, cursor_x])

        if x_data.size > 1:
            self.ax.set_xlim(float(x_data[0]), float(x_data[-1]))
        elif x_data.size == 1:
            center = float(x_data[0])
            width = max(self._minimum_world_width(), 1.0)
            self.ax.set_xlim(center - width, center + width)
        else:
            self.ax.set_xlim(-0.5, 0.5)

        finite_values = self.spectrum[np.isfinite(self.spectrum)]
        if model_values is not None:
            model_finite = model_values[np.isfinite(model_values)]
            if model_finite.size > 0:
                finite_values = np.concatenate([finite_values, model_finite]) if finite_values.size > 0 else model_finite
        if finite_values.size > 0:
            y_min = float(np.nanmin(finite_values))
            y_max = float(np.nanmax(finite_values))
            span = y_max - y_min
            if span <= 0:
                span = max(abs(y_min), abs(y_max), 1.0) * 0.1
            self.ax.set_ylim(y_min - 0.1 * span, y_max + 0.1 * span)
        else:
            self.ax.set_ylim(-1.0, 1.0)

        if show_pre_subtracted:
            title = f"{title}  [fit result]"
        self.ax.set_title(title, loc="left")

        self._render_range_overlays()
        self.canvas.draw_idle()
        self._update_controls_enabled()

    def _slice_for_viewer(self, viewer, data, channel_index):
        if viewer is None or data is None:
            return None
        plane = getattr(viewer, "plane", "xy")
        idx = int(channel_index)
        if data.ndim == 4:
            idx = max(0, min(idx, data.shape[1] - 1))
            if plane == "xy":
                return data[0, idx, :, :]
            if plane == "xz":
                y_idx = max(0, min(getattr(viewer, "y", 0), data.shape[2] - 1))
                return data[0, :, y_idx, :]
            if plane == "zy":
                x_idx = max(0, min(getattr(viewer, "x", 0), data.shape[3] - 1))
                return data[0, :, :, x_idx].T
        elif data.ndim == 3:
            idx = max(0, min(idx, data.shape[0] - 1))
            if plane == "xy":
                return data[idx, :, :]
            if plane == "xz":
                y_idx = max(0, min(getattr(viewer, "y", 0), data.shape[1] - 1))
                return data[:, y_idx, :]
            if plane == "zy":
                x_idx = max(0, min(getattr(viewer, "x", 0), data.shape[2] - 1))
                return data[:, :, x_idx].T
        elif data.ndim == 2:
            return data
        return None

    def _apply_data_to_all_windows(self, new_data):
        main_window = getattr(self.fits_viewer, "main_window", None) or self.fits_viewer
        windows = [main_window] + list(getattr(main_window, "subwindows", []) or [])
        if self.fits_viewer not in windows:
            windows.append(self.fits_viewer)

        seen = set()
        unique_windows = []
        for window in windows:
            if window is None:
                continue
            token = id(window)
            if token in seen:
                continue
            seen.add(token)
            unique_windows.append(window)

        for window in unique_windows:
            window.data = new_data
            if hasattr(window, "update_cube"):
                window.update_cube()

        for window in unique_windows:
            channel_index = 0
            getter = getattr(window, "current_channel_index", None)
            if callable(getter):
                try:
                    channel_index = int(getter())
                except Exception:
                    channel_index = 0
            updater = getattr(window, "update_channel", None)
            if callable(updater) and getattr(new_data, "ndim", 0) >= 3:
                try:
                    updater(window.plane, channel_index)
                    continue
                except Exception:
                    pass
            if hasattr(window, "im"):
                try:
                    data_slice = self._slice_for_viewer(window, new_data, channel_index)
                    if data_slice is not None:
                        window.im.set_data(data_slice)
                        window.canvas.draw_idle()
                except Exception:
                    continue

    def _record_baseline_action(self, params):
        main_window = getattr(self.fits_viewer, "main_window", None) or self.fits_viewer
        recorder = getattr(main_window, "record_action", None)
        if callable(recorder):
            recorder("apply_baseline_subtraction", params=params)

    def _session_records_up_to_cursor(self):
        main_window = getattr(self.fits_viewer, "main_window", None) or self.fits_viewer
        session = getattr(main_window, "action_session", None)
        if session is None:
            return [], None
        try:
            records = list(getattr(session, "history", []) or [])
        except Exception:
            return [], session
        try:
            cursor = int(getattr(session, "cursor", len(records)))
        except Exception:
            cursor = len(records)
        cursor = max(0, min(cursor, len(records)))
        return records[:cursor], session

    @staticmethod
    def _resolve_baseline_action_index(records, preferred_index=None):
        if not isinstance(records, list) or not records:
            return None
        if preferred_index is not None:
            try:
                idx = int(preferred_index)
            except Exception:
                idx = None
            if idx is not None and 0 <= idx < len(records):
                action_name = str(getattr(records[idx], "action", "") or "").strip().lower()
                if action_name == "apply_baseline_subtraction":
                    return idx
        for idx in range(len(records) - 1, -1, -1):
            action_name = str(getattr(records[idx], "action", "") or "").strip().lower()
            if action_name == "apply_baseline_subtraction":
                return idx
        return None

    def _restore_baseline_result_from_history(self, preferred_action_index=None):
        records, source_session = self._session_records_up_to_cursor()
        if source_session is None or not records:
            return False

        action_index = self._resolve_baseline_action_index(records, preferred_index=preferred_action_index)
        if action_index is None:
            return False

        baseline_record = records[action_index]
        params = dict(getattr(baseline_record, "params", {}) or {})
        world_ranges = params.get("world_ranges")
        if not isinstance(world_ranges, list):
            return False
        try:
            order = int(params.get("order", int(self.baseline_order_spinbox.value())))
        except Exception:
            order = int(self.baseline_order_spinbox.value())
        reference_pixel = params.get("reference_pixel")
        if not isinstance(reference_pixel, (list, tuple)):
            reference_pixel = None

        registry = getattr(source_session, "registry", None)
        if registry is None:
            return False
        try:
            replay_session = ActionSession(registry=registry, state=None)
        except Exception:
            return False

        seed_state = getattr(source_session, "_initial_state_seed", None)
        if seed_state is None:
            seed_state = getattr(source_session, "state", None)
        if seed_state is None:
            return False

        try:
            replay_session.set_initial_state_seed(seed_state)
            replay_session.reset_to_initial()
        except Exception:
            return False

        for idx in range(action_index):
            record = records[idx]
            action_name = str(getattr(record, "action", "") or "")
            action_params = dict(getattr(record, "params", {}) or {})
            try:
                replay_session.execute(action_name, **action_params)
            except Exception:
                return False

        replay_state = getattr(replay_session, "state", None)
        if replay_state is None or getattr(replay_state, "data", None) is None:
            return False

        try:
            result = compute_polynomial_baseline_subtraction(
                replay_state,
                world_ranges=world_ranges,
                order=order,
                reference_pixel=reference_pixel,
            )
        except Exception:
            return False

        self._previous_data_snapshot = np.asarray(replay_state.data)
        self._active_plot_source_data = self._previous_data_snapshot
        self.baseline_model_data = np.asarray(result.baseline_model)
        self._last_subtracted_data = np.asarray(result.subtracted_data)
        return True

    def _sync_app_state_data(self, data):
        main_window = getattr(self.fits_viewer, "main_window", None) or self.fits_viewer
        sync_data = getattr(main_window, "sync_app_state_data", None)
        if callable(sync_data):
            try:
                sync_data(data=data, header=self.fits_viewer.header, wcs=self.fits_viewer.wcs)
                return
            except Exception:
                pass
        app_state = getattr(main_window, "app_state", None)
        if app_state is not None:
            app_state.data = data
            app_state.header = self.fits_viewer.header
            app_state.wcs = self.fits_viewer.wcs

    def _refresh_spectrum_panel_if_open(self):
        control_panel = getattr(self.fits_viewer, "control_panel", None)
        panel = getattr(control_panel, "spec_window", None) if control_panel is not None else None
        if panel is None:
            return
        refresh = getattr(panel, "update_spectrum", None)
        if not callable(refresh):
            return
        try:
            panel.last_update_time = 0
        except Exception:
            pass
        try:
            refresh(getattr(panel, "x", 0), getattr(panel, "y", 0), getattr(panel, "z", 0))
        except Exception:
            pass

    def apply_polynomial_baseline_subtraction(self):
        if not self._cube_data_available():
            QMessageBox.information(
                self,
                "Baseline Subtraction",
                "Baseline subtraction is available only for 3D/4D cube data.",
            )
            return

        app_state = getattr(self.fits_viewer, "app_state", None)
        if app_state is None:
            QMessageBox.warning(self, "Baseline Subtraction", "Application state is not initialized.")
            return

        try:
            world_ranges = self._collect_baseline_world_ranges(strict=True)
            order = int(self.baseline_order_spinbox.value())
            reference_pixel = self._baseline_reference_pixel()
            current_data = getattr(self.fits_viewer, "data", None)
            result = compute_polynomial_baseline_subtraction(
                app_state,
                world_ranges=world_ranges,
                order=order,
                reference_pixel=reference_pixel,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Baseline Subtraction Error", str(exc))
            return

        self.baseline_model_data = result.baseline_model
        self._last_subtracted_data = result.subtracted_data
        self._previous_data_snapshot = current_data
        self._active_plot_source_data = current_data
        self.show_fit_result_radio.setChecked(True)
        self._apply_data_to_all_windows(result.subtracted_data)
        self._sync_app_state_data(result.subtracted_data)

        action_params = {
            "order": int(result.order),
            "world_ranges": [[float(lo), float(hi)] for lo, hi in result.world_ranges],
        }
        if reference_pixel is not None:
            action_params["reference_pixel"] = [float(v) for v in reference_pixel]
        self._record_baseline_action(action_params)
        self._has_pending_changes = True

        self.update_spectrum(self.x, self.y, self.z)
        self._refresh_spectrum_panel_if_open()
        self._update_controls_enabled()

        QMessageBox.information(
            self,
            "Baseline Subtraction",
            (
                f"Applied polynomial baseline subtraction (order={result.order}).\n"
                f"Fitted spectra: {result.n_fitted_spectra}/{result.n_total_spectra}"
            ),
        )

    def reset_baseline_result(self, *, silent=False):
        if self._last_subtracted_data is None and self.baseline_model_data is None:
            if not silent:
                QMessageBox.information(self, "Reset", "No baseline result to reset.")
            return False

        reverted = False
        main_window = getattr(self.fits_viewer, "main_window", None) or self.fits_viewer
        session = getattr(main_window, "action_session", None)
        if session is not None:
            try:
                cursor = int(getattr(session, "cursor", 0))
                history = list(getattr(session, "history", []) or [])
                if cursor > 0 and cursor <= len(history):
                    last_action = str(getattr(history[cursor - 1], "action", "") or "")
                    if last_action == "apply_baseline_subtraction":
                        undo = getattr(main_window, "undo_last_action", None)
                        if callable(undo):
                            undo()
                            reverted = True
            except Exception:
                reverted = False

        if not reverted:
            if self._previous_data_snapshot is None:
                if not silent:
                    QMessageBox.warning(self, "Reset", "Could not restore previous data snapshot.")
                return False
            self._apply_data_to_all_windows(self._previous_data_snapshot)
            self._sync_app_state_data(self._previous_data_snapshot)
            reverted = True

        if reverted:
            self.baseline_model_data = None
            self._last_subtracted_data = None
            self._previous_data_snapshot = None
            self._active_plot_source_data = None
            self._has_pending_changes = False
            self.update_spectrum(self.x, self.y, self.z)
            self._refresh_spectrum_panel_if_open()
            self._update_controls_enabled()
            if not silent:
                QMessageBox.information(self, "Reset", "Baseline subtraction result has been reset.")
            return True
        return False

    def save_subtracted_fits(self):
        if self._last_subtracted_data is None:
            QMessageBox.warning(self, "Save", "No baseline-subtracted data available. Apply baseline first.")
            return

        data_to_save = np.asarray(getattr(self.fits_viewer, "data", self._last_subtracted_data))
        header = self.fits_viewer.header.copy()

        finite = np.isfinite(data_to_save)
        if np.any(finite):
            header["DATAMIN"] = float(np.nanmin(data_to_save))
            header["DATAMAX"] = float(np.nanmax(data_to_save))
        else:
            header.pop("DATAMIN", None)
            header.pop("DATAMAX", None)

        for entry in build_processing_history_lines(self.fits_viewer):
            header.add_history(entry)

        saver = SaveFITS(data_to_save, header, self.fits_viewer.filename)
        saver.save(suffix="baseline")

    def save_baseline_model_fits(self):
        if self.baseline_model_data is None:
            QMessageBox.warning(self, "Baseline Model", "No baseline model available. Apply baseline subtraction first.")
            return

        base_filename = getattr(self.fits_viewer, "filename", "takefits.fits")
        default_filename = f"{base_filename.rsplit('.', 1)[0]}.baseline_model.fits"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Baseline Model FITS",
            default_filename,
            "FITS Files (*.fits);;All Files (*)",
        )
        if not path:
            return

        try:
            export_baseline_model_fits(
                state=self.fits_viewer.app_state,
                output_path=path,
                baseline_model=self.baseline_model_data,
                history_entries=build_processing_history_lines(self.fits_viewer),
            )
            QMessageBox.information(self, "Save Successful", f"FITS successfully saved as: {path}")
        except Exception as exc:
            QMessageBox.warning(self, "Save Error", f"Failed to save baseline model FITS:\n{exc}")

    def export_workspace_state(self):
        records, _session = self._session_records_up_to_cursor()
        baseline_action_index = self._resolve_baseline_action_index(records)
        has_result = bool(self._active_plot_source_data is not None and self.baseline_model_data is not None)
        return {
            "schema": 1,
            "order": int(self.baseline_order_spinbox.value()),
            "pixel": {
                "x": int(round(float(self.x))),
                "y": int(round(float(self.y))),
                "z": int(round(float(self.z))),
            },
            "display_mode": "fit_result" if self._show_pre_subtracted() else "subtracted",
            "has_result": has_result,
            "baseline_action_index": int(baseline_action_index) if has_result and baseline_action_index is not None else None,
            "ranges": [
                {
                    "min": str(row["min_edit"].text() or "").strip(),
                    "max": str(row["max_edit"].text() or "").strip(),
                }
                for row in self.baseline_range_rows
            ],
        }

    def restore_workspace_state(self, state):
        if not isinstance(state, dict):
            return False

        self.baseline_model_data = None
        self._last_subtracted_data = None
        self._previous_data_snapshot = None
        self._active_plot_source_data = None
        self._has_pending_changes = False

        restored = False
        try:
            order = int(state.get("order", self.baseline_order_spinbox.value()))
            order = max(int(self.baseline_order_spinbox.minimum()), min(int(self.baseline_order_spinbox.maximum()), order))
            self.baseline_order_spinbox.setValue(order)
            restored = True
        except Exception:
            pass

        ranges = state.get("ranges")
        if isinstance(ranges, list):
            self._remove_all_range_rows()
            for entry in ranges:
                if not isinstance(entry, dict):
                    continue
                self.add_baseline_range_row(
                    str(entry.get("min", "") or "").strip(),
                    str(entry.get("max", "") or "").strip(),
                )
            if not self.baseline_range_rows:
                self._seed_default_baseline_ranges()
            restored = True

        x = self.x
        y = self.y
        z = self.z
        pixel = state.get("pixel")
        if isinstance(pixel, dict):
            try:
                x = int(pixel.get("x", x))
            except Exception:
                pass
            try:
                y = int(pixel.get("y", y))
            except Exception:
                pass
            try:
                z = int(pixel.get("z", z))
            except Exception:
                pass

        action_index = state.get("baseline_action_index")
        has_result = bool(state.get("has_result", action_index is not None))
        restored_baseline_result = bool(
            has_result and self._restore_baseline_result_from_history(preferred_action_index=action_index)
        )
        if restored_baseline_result:
            restored = True
        self._has_pending_changes = restored_baseline_result

        display_mode = str(state.get("display_mode", "") or "").strip().lower()
        use_fit_result = bool(self._active_plot_source_data is not None and display_mode in {"", "fit_result"})
        if use_fit_result:
            self.show_fit_result_radio.setChecked(True)
        else:
            self.show_subtracted_radio.setChecked(True)

        self.update_spectrum(x, y, z)
        self._render_range_overlays()
        self._update_controls_enabled()
        return restored

    def closeEvent(self, event):
        if self._has_pending_changes:
            choice = confirm_pending_close(
                self,
                "Close Baseline Panel",
                "There are unapplied baseline changes.",
            )
            if choice == "cancel":
                event.ignore()
                return
            if choice == "discard":
                if not self.reset_baseline_result(silent=True):
                    event.ignore()
                    return
        if self._region_signal_connected:
            manager = getattr(self.fits_viewer, "region_manager", None)
            if manager is not None and hasattr(manager, "selected_region_changed"):
                try:
                    manager.selected_region_changed.disconnect(self.on_region_changed)
                except Exception:
                    pass
            self._region_signal_connected = False
        try:
            plt.close(self.fig)
        finally:
            super().closeEvent(event)
