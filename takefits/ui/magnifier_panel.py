from __future__ import annotations

import math
import time

import matplotlib.patches as mpatches
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PySide6.QtGui import QCursor, QKeySequence, QShortcut
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QRadioButton,
    QSpinBox,
    QTextEdit,
    QWidget,
)

from takefits.logic.data_tools import sanitize_slice


class MagnifierPanel(QWidget):
    """Floating viewer that shows a live pixel cutout around the cursor."""

    def __init__(self, main_viewer):
        super().__init__()
        self.main_viewer = main_viewer
        self._last_update_time = 0.0
        self._min_update_interval_sec = 1.0 / 30.0
        self._last_request = None
        self._last_source_viewer = None
        self._last_plane = None
        self._last_x = None
        self._last_y = None
        self._last_source_axes = None
        self._pending_source_viewer = None
        self._pending_plane = None
        self._pending_x = None
        self._pending_y = None
        self._pending_source_axes = None
        self._last_lock_toggle_time = 0.0

        self.setWindowTitle("Magnifier")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.resize(280, 360)

        layout = QGridLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setHorizontalSpacing(6)
        layout.setVerticalSpacing(6)

        self.info_label = QLabel("Move cursor over image")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.info_label.setWordWrap(True)
        info_height = self.info_label.fontMetrics().lineSpacing() * 5 + 4
        self.info_label.setMinimumHeight(info_height)
        self.info_label.setMaximumHeight(info_height)
        layout.addWidget(self.info_label, 0, 0, 1, 2)

        top_controls = QHBoxLayout()
        top_controls.setContentsMargins(0, 0, 0, 0)
        top_controls.setSpacing(4)
        top_controls.addWidget(QLabel("Size"))
        self.size_spin = QSpinBox(self)
        self.size_spin.setRange(5, 201)
        self.size_spin.setSingleStep(2)
        self.size_spin.setValue(41)
        self.size_spin.setFixedWidth(58)
        self.size_spin.setToolTip("Pixel width of the cursor-centered cutout.")
        self.size_spin.valueChanged.connect(self._refresh_last_cursor)
        top_controls.addWidget(self.size_spin)
        top_controls.addSpacing(8)
        self.lock_check = QCheckBox("Lock", self)
        self.lock_check.setToolTip("Freeze magnifier updates. Press F while the app is focused to toggle.")
        self.lock_check.toggled.connect(self._on_lock_toggled)
        top_controls.addWidget(self.lock_check)
        top_controls.addSpacing(8)
        self.beam_check = QCheckBox("Beam", self)
        self.beam_check.setToolTip("Overlay the FITS beam size on XY magnifier views.")
        self.beam_check.toggled.connect(self._refresh_beam_overlay)
        top_controls.addWidget(self.beam_check)
        top_controls.addStretch(1)
        layout.addLayout(top_controls, 1, 0, 1, 2)

        marker_controls = QHBoxLayout()
        marker_controls.setContentsMargins(0, 0, 0, 0)
        marker_controls.setSpacing(6)
        marker_controls.addWidget(QLabel("Center:"))
        self.cross_radio = QRadioButton("Cross", self)
        self.box_radio = QRadioButton("Box", self)
        self.none_radio = QRadioButton("None", self)
        self.cross_radio.setToolTip("Draw a short crosshair at the center pixel.")
        self.box_radio.setToolTip("Draw a one-pixel square at the center pixel.")
        self.none_radio.setToolTip("Hide the center marker.")
        self.cross_radio.setChecked(True)
        self.marker_group = QButtonGroup(self)
        for button in (self.cross_radio, self.box_radio, self.none_radio):
            self.marker_group.addButton(button)
            button.toggled.connect(self._refresh_marker_mode)
            marker_controls.addWidget(button)
        marker_controls.addStretch(1)
        layout.addLayout(marker_controls, 2, 0, 1, 2)

        self.lock_shortcut = QShortcut(QKeySequence("F"), self)
        self.lock_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self.lock_shortcut.activated.connect(self._toggle_lock_from_shortcut)

        self.fig = Figure(figsize=(2.6, 2.6))
        self.fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setFixedSize(240, 240)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.im = None
        self.vline, = self.ax.plot([], [], color="cyan", linewidth=0.8, alpha=0.9)
        self.hline, = self.ax.plot([], [], color="cyan", linewidth=0.8, alpha=0.9)
        self.box_line, = self.ax.plot([], [], color="cyan", linewidth=0.8, alpha=0.9)
        self.beam_ellipse = mpatches.Ellipse((0, 0), 0, 0, visible=False, zorder=4)
        self.ax.add_patch(self.beam_ellipse)
        self._sync_beam_control(self.main_viewer)
        layout.addWidget(self.canvas, 3, 0, 1, 2, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.setRowStretch(3, 1)
        layout.setColumnStretch(1, 1)

    def closeEvent(self, event):
        main = getattr(self, "main_viewer", None)
        callback = getattr(main, "on_magnifier_panel_closed", None)
        if callable(callback):
            callback()
        super().closeEvent(event)

    def keyPressEvent(self, event):
        try:
            key = event.key()
        except Exception:
            key = None
        if key == Qt.Key.Key_F:
            self.toggle_lock()
            try:
                event.accept()
            except Exception:
                pass
            return
        super().keyPressEvent(event)

    def is_locked(self) -> bool:
        return bool(self.lock_check.isChecked())

    def toggle_lock(self) -> bool:
        now = time.monotonic()
        if now - self._last_lock_toggle_time < 0.08:
            return self.is_locked()
        self._last_lock_toggle_time = now
        self.lock_check.toggle()
        return self.is_locked()

    def _toggle_lock_from_shortcut(self):
        if self._shortcut_focus_blocks_lock_toggle():
            return
        self.toggle_lock()

    @staticmethod
    def _shortcut_focus_blocks_lock_toggle() -> bool:
        focus = QApplication.focusWidget()
        if focus is None:
            return False
        if not isinstance(
            focus,
            (
                QAbstractSpinBox,
                QComboBox,
                QLineEdit,
                QPlainTextEdit,
                QTextEdit,
            ),
        ):
            return False
        try:
            cursor_widget = QApplication.widgetAt(QCursor.pos())
        except Exception:
            cursor_widget = None
        if cursor_widget is None:
            return False
        return cursor_widget is focus or focus.isAncestorOf(cursor_widget)

    def _on_lock_toggled(self, checked: bool):
        if checked:
            self._clear_pending_cursor()
            return
        if self._pending_source_viewer is not None and self._pending_plane is not None:
            source_viewer = self._pending_source_viewer
            plane = self._pending_plane
            x = self._pending_x
            y = self._pending_y
            source_axes = self._pending_source_axes
            self._clear_pending_cursor()
            self.update_from_cursor(
                source_viewer,
                plane,
                x,
                y,
                force=True,
                source_axes=source_axes,
                respect_lock=False,
            )
            return
        self.refresh_last_cursor(force=True)

    def _store_pending_cursor(self, source_viewer, plane: str, x: float, y: float, source_axes=None):
        self._pending_source_viewer = source_viewer
        self._pending_plane = plane
        self._pending_x = x
        self._pending_y = y
        self._pending_source_axes = source_axes

    def _clear_pending_cursor(self):
        self._pending_source_viewer = None
        self._pending_plane = None
        self._pending_x = None
        self._pending_y = None
        self._pending_source_axes = None

    def _refresh_last_cursor(self):
        self.refresh_last_cursor(force=True, respect_lock=False)

    def refresh_last_cursor(self, *, force: bool = True, respect_lock: bool = True) -> bool:
        if respect_lock and self.is_locked():
            return False
        if self._last_source_viewer is None or self._last_plane is None:
            return False
        return self.update_from_cursor(
            self._last_source_viewer,
            self._last_plane,
            self._last_x,
            self._last_y,
            force=force,
            source_axes=self._last_source_axes,
            respect_lock=respect_lock,
        )

    def _refresh_marker_mode(self, checked: bool):
        if not checked:
            return
        if self.is_locked():
            self._redraw_current_marker()
            return
        self.refresh_last_cursor(force=True)

    def _redraw_current_marker(self) -> bool:
        if self.im is None or self._last_x is None or self._last_y is None:
            return False
        try:
            extent = self.im.get_extent()
        except Exception:
            return False
        self._set_center_marker(int(round(self._last_x)), int(round(self._last_y)), extent)
        self.canvas.draw_idle()
        return True

    def _refresh_beam_overlay(self, *_):
        self._redraw_current_beam()

    def _redraw_current_beam(self) -> bool:
        if self.im is None or self._last_source_viewer is None or self._last_plane is None:
            self._hide_beam_overlay()
            self.canvas.draw_idle()
            return False
        try:
            extent = self.im.get_extent()
        except Exception:
            self._hide_beam_overlay()
            self.canvas.draw_idle()
            return False
        updated = self._update_beam_overlay(self._last_source_viewer, self._last_plane, extent)
        self.canvas.draw_idle()
        return updated

    def update_from_cursor(
        self,
        source_viewer,
        plane: str,
        x,
        y,
        *,
        force: bool = False,
        source_axes=None,
        respect_lock: bool = True,
    ) -> bool:
        if source_viewer is None:
            return False
        if not force and not self.isVisible():
            return False

        try:
            xf = float(x)
            yf = float(y)
        except Exception:
            return False
        if not math.isfinite(xf) or not math.isfinite(yf):
            return False

        now = time.monotonic()
        plane_key = str(plane or getattr(source_viewer, "plane", "xy")).lower()
        if respect_lock and self.is_locked():
            self._store_pending_cursor(source_viewer, plane_key, xf, yf, source_axes)
            return False

        last_request = self._last_request
        source_changed = (
            last_request is None
            or last_request[0] != id(source_viewer)
            or last_request[1] != plane_key
            or last_request[-1] != id(source_axes)
        )
        if not force and not source_changed and now - self._last_update_time < self._min_update_interval_sec:
            return False

        self._last_source_viewer = source_viewer
        self._last_plane = plane_key
        self._last_x = xf
        self._last_y = yf
        self._last_source_axes = source_axes

        source_im = self._source_image(source_viewer, source_axes)
        cutout = self._cursor_cutout(source_viewer, self._last_plane, xf, yf, source_axes=source_axes)
        if cutout is None:
            return False
        data, extent = cutout
        if data.size == 0:
            return False

        request = (
            id(source_viewer),
            self._last_plane,
            int(round(xf)),
            int(round(yf)),
            tuple(data.shape),
            tuple(float(v) for v in extent),
            self._image_signature(source_im),
            id(source_axes),
        )
        if not force and request == self._last_request:
            return False
        self._last_request = request
        self._last_update_time = now

        if self.im is None:
            self.im = self.ax.imshow(
                data,
                origin="lower",
                interpolation="nearest",
                aspect="equal",
                extent=extent,
            )
        else:
            self.im.set_data(data)
            self.im.set_extent(extent)

        if source_im is not None:
            try:
                self.im.set_cmap(source_im.get_cmap())
            except Exception:
                pass
            try:
                self.im.set_norm(source_im.norm)
            except Exception:
                try:
                    self.im.set_clim(*source_im.get_clim())
                except Exception:
                    pass

        self.ax.set_xlim(extent[0], extent[1])
        self.ax.set_ylim(extent[2], extent[3])
        self._set_center_marker(int(round(xf)), int(round(yf)), extent)
        self._sync_beam_control(source_viewer)
        self._update_beam_overlay(source_viewer, self._last_plane, extent)
        self._update_info_label(source_viewer, self._last_plane, xf, yf, source_axes=source_axes)
        self.canvas.draw_idle()
        return True

    @staticmethod
    def _source_image(source_viewer, source_axes=None):
        if source_axes is not None:
            try:
                images = [im for im in source_axes.get_images() if im is not None and im.get_visible()]
                if images:
                    return images[0]
            except Exception:
                pass
        return getattr(source_viewer, "im", None)

    @staticmethod
    def _image_signature(im):
        if im is None:
            return None
        try:
            cmap_name = im.get_cmap().name
        except Exception:
            cmap_name = None
        try:
            clim = tuple(float(v) for v in im.get_clim())
        except Exception:
            clim = None
        try:
            array_id = id(im.get_array())
        except Exception:
            array_id = None
        return (cmap_name, clim, id(getattr(im, "norm", None)), array_id)

    def _set_center_marker(self, x: int, y: int, extent):
        self.vline.set_data([], [])
        self.hline.set_data([], [])
        self.box_line.set_data([], [])

        mode = self._center_marker_mode()
        if mode == "none":
            return
        if mode == "box":
            self.box_line.set_data(
                [x - 0.5, x + 0.5, x + 0.5, x - 0.5, x - 0.5],
                [y - 0.5, y - 0.5, y + 0.5, y + 0.5, y - 0.5],
            )
            return

        try:
            width = abs(float(extent[1]) - float(extent[0]))
            height = abs(float(extent[3]) - float(extent[2]))
        except Exception:
            width = height = float(self._cutout_size())
        half = max(1.0, min(width, height) * 0.12)
        self.vline.set_data([x, x], [y - half, y + half])
        self.hline.set_data([x - half, x + half], [y, y])

    def _center_marker_mode(self) -> str:
        if self.box_radio.isChecked():
            return "box"
        if self.none_radio.isChecked():
            return "none"
        return "cross"

    def _update_beam_overlay(self, source_viewer, plane: str, extent) -> bool:
        if not self.beam_check.isEnabled() or not self.beam_check.isChecked() or str(plane or "").lower() != "xy":
            self._hide_beam_overlay()
            return False

        beam = self._beam_geometry(source_viewer)
        if beam is None:
            self._hide_beam_overlay()
            return False

        width, height, angle = beam
        if width <= 0 or height <= 0:
            self._hide_beam_overlay()
            return False

        try:
            x0, x1, y0, y1 = [float(v) for v in extent]
        except Exception:
            self._hide_beam_overlay()
            return False

        config = self._beam_config(source_viewer)
        try:
            rel_x = float(config.get("beam_pos_x", 0.1))
            rel_y = float(config.get("beam_pos_y", 0.1))
        except Exception:
            rel_x, rel_y = 0.1, 0.1
        rel_x = max(0.0, min(1.0, rel_x))
        rel_y = max(0.0, min(1.0, rel_y))

        center_x = x0 + rel_x * (x1 - x0)
        center_y = y0 + rel_y * (y1 - y0)
        self.beam_ellipse.center = (center_x, center_y)
        self.beam_ellipse.width = width
        self.beam_ellipse.height = height
        self.beam_ellipse.angle = angle
        self.beam_ellipse.set_facecolor(config.get("beam_facecolor", "white"))
        self.beam_ellipse.set_edgecolor(config.get("beam_edgecolor", "None"))
        self.beam_ellipse.set_linewidth(config.get("beam_linewidth", 0))
        self.beam_ellipse.set_visible(True)
        return True

    def _sync_beam_control(self, source_viewer) -> bool:
        enabled = self._beam_geometry(source_viewer) is not None
        if enabled:
            self.beam_check.setEnabled(True)
            self.beam_check.setToolTip("Overlay the FITS beam size on XY magnifier views.")
            return True

        if self.beam_check.isChecked():
            was_blocked = self.beam_check.blockSignals(True)
            self.beam_check.setChecked(False)
            self.beam_check.blockSignals(was_blocked)
        self.beam_check.setEnabled(False)
        self.beam_check.setToolTip("Beam metadata is not available in this FITS header.")
        self._hide_beam_overlay()
        return False

    def _hide_beam_overlay(self):
        self.beam_ellipse.set_visible(False)

    @staticmethod
    def _beam_geometry(source_viewer):
        header = MagnifierPanel._source_header(source_viewer)
        if header is None:
            return None
        cunit1 = str(header.get("CUNIT1", "") or "").strip().lower()
        cunit2 = str(header.get("CUNIT2", "") or "").strip().lower()
        if (cunit1 or cunit2) and (cunit1 != "deg" or cunit2 != "deg"):
            return None
        try:
            bmaj = float(header.get("BMAJ", 0) or 0)
            bmin = float(header.get("BMIN", 0) or 0)
            bpa = float(header.get("BPA", 0) or 0)
        except Exception:
            return None
        if bmaj <= 0 or bmin <= 0:
            return None
        try:
            cdelt1 = abs(float(header["CDELT1"]))
            cdelt2 = abs(float(header["CDELT2"]))
        except Exception:
            try:
                cdelt1 = math.hypot(float(header["CD1_1"]), float(header["CD2_1"]))
                cdelt2 = math.hypot(float(header["CD1_2"]), float(header["CD2_2"]))
            except Exception:
                return None
        if cdelt1 <= 0 or cdelt2 <= 0:
            return None
        return bmaj / cdelt1, bmin / cdelt2, bpa + 90.0

    @staticmethod
    def _source_header(source_viewer):
        header = getattr(source_viewer, "header", None)
        if header is not None:
            return header
        parent = getattr(source_viewer, "fits_viewer", None)
        return getattr(parent, "header", None)

    def _beam_config(self, source_viewer):
        for candidate in (source_viewer, getattr(source_viewer, "fits_viewer", None), self.main_viewer):
            manager = getattr(candidate, "config_manager", None)
            config = getattr(manager, "config", None)
            if isinstance(config, dict):
                return config
        return {}

    def _update_info_label(self, source_viewer, plane: str, x: float, y: float, *, source_axes=None):
        xi = int(round(x))
        yi = int(round(y))
        pixel_text = self._pixel_label(plane, xi, yi)
        coord_lines = self._coord_label_lines(source_viewer, x, y, source_axes=source_axes)
        label_lines = [f"Pixel: {pixel_text}"]
        if coord_lines:
            label_lines.extend(self._world_label_lines(coord_lines[0]))
            label_lines.extend(self._coord_detail_label_lines(coord_lines[1:]))
        text = "\n".join(label_lines)
        self.info_label.setText(text)
        self.info_label.setToolTip(text)

    @staticmethod
    def _pixel_label(plane: str, x: int, y: int) -> str:
        key = str(plane or "").lower()
        if key == "xz":
            return f"x={x}, z={y}"
        if key == "zy":
            return f"z={x}, y={y}"
        return f"x={x}, y={y}"

    @staticmethod
    def _coord_label_lines(source_viewer, x: float, y: float, *, source_axes=None) -> list[str]:
        formatter = getattr(source_axes, "format_coord", None) if source_axes is not None else None
        if not callable(formatter):
            formatter = getattr(source_viewer, "formatter", None)
        if not callable(formatter):
            return []
        try:
            text = formatter(x, y)
        except Exception:
            return []
        lines = [" ".join(str(line).split()) for line in str(text or "").splitlines()]
        return [line for line in lines if line]

    @staticmethod
    def _world_label_lines(world_text: str) -> list[str]:
        text = str(world_text or "").strip()
        if not text:
            return []
        parts = [part.strip() for part in text.split(",") if part.strip()]
        if len(parts) < 2:
            return [f"World: {text}"]

        label_lines = []
        for part in parts:
            axis, sep, value = part.partition("=")
            axis = axis.strip()
            value = value.strip()
            if sep and axis and value:
                label_lines.append(f"World {axis}: {value}")
            else:
                label_lines.append(f"World: {part}")
        return label_lines

    @staticmethod
    def _coord_detail_label_lines(coord_lines: list[str]) -> list[str]:
        label_lines = []
        for line in coord_lines:
            text = str(line or "").strip()
            if not text:
                continue
            if text.startswith("["):
                close_index = text.find("]")
                if close_index > 0:
                    intensity = text[1:close_index].strip()
                    system = text[close_index + 1:].strip()
                    if system:
                        label_lines.append(f"System: {system}")
                    if intensity:
                        label_lines.append(f"Intensity: {intensity}")
                    continue
            label_lines.append(f"System: {text}")
        return label_lines

    def _cutout_size(self) -> int:
        size = int(self.size_spin.value())
        if size % 2 == 0:
            size += 1
        return max(3, size)

    def _cursor_cutout(self, source_viewer, plane: str, x: float, y: float, *, source_axes=None):
        axes_cutout = self._axes_image_cutout(source_viewer, source_axes, x, y)
        if axes_cutout is not None:
            return axes_cutout

        data = getattr(source_viewer, "data", None)
        if data is None:
            return None

        plane = str(plane or "xy").lower()
        try:
            ndim = int(getattr(data, "ndim", 0))
        except Exception:
            return None
        if ndim < 2:
            return None

        xi = int(round(x))
        yi = int(round(y))
        cube = getattr(source_viewer, "cube", None)

        try:
            if ndim == 2 or plane == "xy":
                if ndim == 2:
                    height, width = data.shape[-2], data.shape[-1]
                    spec = self._cutout_spec(xi, yi, width, height)
                    raw = data[spec["src_y0"]:spec["src_y1"], spec["src_x0"]:spec["src_x1"]]
                else:
                    zpix = self._shared_index(source_viewer, "_get_shared_zpix", cube.shape[0])
                    height, width = cube.shape[1], cube.shape[2]
                    spec = self._cutout_spec(xi, yi, width, height)
                    raw = cube[zpix, spec["src_y0"]:spec["src_y1"], spec["src_x0"]:spec["src_x1"]]
            elif plane == "xz" and cube is not None:
                fixed_y = self._shared_index(source_viewer, "_get_shared_ypix", cube.shape[1])
                depth, width = cube.shape[0], cube.shape[2]
                spec = self._cutout_spec(xi, yi, width, depth)
                raw = cube[spec["src_y0"]:spec["src_y1"], fixed_y, spec["src_x0"]:spec["src_x1"]]
            elif plane == "zy" and cube is not None:
                fixed_x = self._shared_index(source_viewer, "_get_shared_xpix", cube.shape[2])
                depth, height = cube.shape[0], cube.shape[1]
                spec = self._cutout_spec(xi, yi, depth, height)
                raw = cube[spec["src_x0"]:spec["src_x1"], spec["src_y0"]:spec["src_y1"], fixed_x].T
            else:
                return None
        except Exception:
            return None

        return self._padded_cutout(raw, spec, source_viewer)

    def _axes_image_cutout(self, source_viewer, source_axes, x: float, y: float):
        if source_axes is None:
            return None
        try:
            images = [im for im in source_axes.get_images() if im is not None and im.get_visible()]
        except Exception:
            images = []
        if not images:
            return None
        try:
            raw = images[0].get_array()
        except Exception:
            return None
        arr = MagnifierPanel._display_array(raw, source_viewer)
        if getattr(arr, "ndim", 0) != 2:
            return None
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        spec = self._cutout_spec(xi, yi, arr.shape[1], arr.shape[0])
        overlap = arr[spec["src_y0"]:spec["src_y1"], spec["src_x0"]:spec["src_x1"]]
        return self._padded_cutout(overlap, spec, source_viewer)

    def _cutout_spec(self, x: int, y: int, width: int, height: int) -> dict:
        return self._cutout_spec_static(x, y, width, height, self._cutout_size())

    @staticmethod
    def _cutout_spec_static(x: int, y: int, width: int, height: int, size: int) -> dict:
        radius = int(size) // 2
        x0 = int(x) - radius
        x1 = int(x) + radius + 1
        y0 = int(y) - radius
        y1 = int(y) + radius + 1
        src_x0 = max(0, x0)
        src_x1 = min(int(width), x1)
        src_y0 = max(0, y0)
        src_y1 = min(int(height), y1)
        return {
            "size": int(size),
            "x0": x0,
            "x1": x1,
            "y0": y0,
            "y1": y1,
            "src_x0": src_x0,
            "src_x1": max(src_x0, src_x1),
            "src_y0": src_y0,
            "src_y1": max(src_y0, src_y1),
            "dst_x0": src_x0 - x0,
            "dst_y0": src_y0 - y0,
            "extent": (x0 - 0.5, x1 - 0.5, y0 - 0.5, y1 - 0.5),
        }

    def _padded_cutout(self, raw, spec: dict, source_viewer):
        return self._padded_cutout_static(raw, spec, source_viewer)

    @staticmethod
    def _padded_cutout_static(raw, spec: dict, source_viewer):
        size = int(spec["size"])
        output = np.full((size, size), np.nan, dtype=float)
        arr = MagnifierPanel._display_array(raw, source_viewer)
        if getattr(arr, "size", 0) > 0:
            rows, cols = arr.shape[-2], arr.shape[-1]
            dy = int(spec["dst_y0"])
            dx = int(spec["dst_x0"])
            output[dy:dy + rows, dx:dx + cols] = arr
        return output, spec["extent"]

    @staticmethod
    def _shared_index(source_viewer, getter_name: str, length: int) -> int:
        getter = getattr(source_viewer, getter_name, None)
        value = 0
        if callable(getter):
            try:
                value = int(round(float(getter())))
            except Exception:
                value = 0
        return max(0, min(int(value), int(length) - 1))

    @staticmethod
    def _display_array(raw, source_viewer):
        if np.ma.isMaskedArray(raw):
            arr = raw.filled(np.nan)
        else:
            arr = np.asarray(raw)
        metadata = getattr(source_viewer, "spectral_metadata", {})
        if isinstance(metadata, dict) and metadata.get("_needs_per_slice_sanitize"):
            arr = sanitize_slice(np.array(arr, copy=True))
        return arr
