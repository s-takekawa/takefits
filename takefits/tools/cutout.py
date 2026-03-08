import math
import re
from datetime import datetime
from itertools import product
from typing import List, Optional, Sequence, Tuple

import numpy as np
from astropy import units as u
from astropy.coordinates import Angle
from astropy.io import fits
from astropy.wcs import WCS
from PySide6.QtWidgets import (
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from takefits.core.coordinate import CoordinateConverter
from takefits.core.region import CircleRegion, CubeRegion, EllipseRegion, RectangleRegion, Region
from takefits.ui.save_fits_dialog import SaveFITS
from takefits.core.usecases import compute_cutout
from takefits.core.app_state import create_app_state
from takefits.core.history_provenance import build_processing_history_lines


__all__ = [
    "CutoutProcessor",
    "CutoutSettingsDialog",
]


def _axis_display_name(ctype: str) -> str:
    base = ctype.split('-')[0].strip()
    return base or ctype


class CutoutProcessor:
    """Utility for extracting data/WCS cutouts from the current FITS viewer."""

    def __init__(
        self,
        data: np.ndarray,
        header,
        wcs: WCS,
        *,
        wcs_to_data_axis: Optional[Sequence[Optional[int]]] = None,
    ) -> None:
        self._data = data
        self._header = header
        self._wcs = wcs
        self._wcs_to_data_axis = self._initialize_wcs_axis_mapping(wcs_to_data_axis)
        self._data_to_wcs_axis = {
            data_axis: wcs_axis
            for wcs_axis, data_axis in enumerate(self._wcs_to_data_axis)
            if data_axis is not None
        }

    def _initialize_wcs_axis_mapping(
        self, mapping: Optional[Sequence[Optional[int]]]
    ) -> List[Optional[int]]:
        if mapping is not None:
            if len(mapping) != self._wcs.naxis:
                raise ValueError("Provided axis mapping length does not match WCS axes.")
            return [int(m) if m is not None else None for m in mapping]

        data_ndim = getattr(self._data, 'ndim', 0)
        default: List[Optional[int]] = []
        for axis in range(self._wcs.naxis):
            data_axis = data_ndim - axis - 1
            if 0 <= data_axis < data_ndim:
                default.append(data_axis)
            else:
                default.append(None)
        return default

    @staticmethod
    def _clamp_indices(start: int, stop: int, axis_len: int) -> Tuple[int, int]:
        start = max(start, 0)
        stop = min(stop, axis_len)
        return start, stop

    def extract(self, pixel_bounds: Sequence[Tuple[int, int]]):
        if self._wcs.naxis != len(pixel_bounds):
            raise ValueError("Number of cut-out bounds does not match WCS axes.")

        data_slices: List[slice] = [slice(None)] * self._data.ndim
        actual_bounds_wcs: List[Optional[Tuple[int, int]]] = [None] * len(pixel_bounds)

        for wcs_axis_index, data_axis_index in enumerate(self._wcs_to_data_axis):
            req_start, req_stop = pixel_bounds[wcs_axis_index]
            start_i = int(math.floor(req_start))
            stop_i = int(math.ceil(req_stop))

            if data_axis_index is None:
                if stop_i <= start_i:
                    stop_i = start_i + 1
                actual_bounds_wcs[wcs_axis_index] = (start_i, stop_i)
                continue

            axis_len = self._data.shape[data_axis_index]
            clip_start, clip_stop = self._clamp_indices(start_i, stop_i, axis_len)
            if clip_stop <= clip_start:
                raise ValueError("Selected region is outside data bounds.")

            data_slices[data_axis_index] = slice(clip_start, clip_stop)
            actual_bounds_wcs[wcs_axis_index] = (clip_start, clip_stop)

        if any(bound is None for bound in actual_bounds_wcs):
            raise ValueError("Failed to compute actual bounds for all axes.")

        sample_dtype = self._data.dtype
        out_dtype = sample_dtype if sample_dtype.kind in ('f', 'c') else np.float32
        cutout_data = self._data[tuple(data_slices)].astype(out_dtype, copy=True)

        new_header = self._header.copy()

        for axis in range(self._wcs.naxis):
            data_axis = self._wcs_to_data_axis[axis]
            bound = actual_bounds_wcs[axis]
            clip_start = bound[0]
            span = max(bound[1] - bound[0], 1)
            crpix_key = f"CRPIX{axis + 1}"
            if crpix_key in new_header:
                new_header[crpix_key] = float(new_header[crpix_key]) - clip_start
            naxis_key = f"NAXIS{axis + 1}"
            if naxis_key in new_header:
                if data_axis is not None and 0 <= data_axis < cutout_data.ndim:
                    new_header[naxis_key] = cutout_data.shape[data_axis]
                else:
                    new_header[naxis_key] = span

        with np.errstate(all='ignore'):
            try:
                new_header['DATAMIN'] = float(np.nanmin(cutout_data))
                new_header['DATAMAX'] = float(np.nanmax(cutout_data))
            except ValueError:
                new_header['DATAMIN'] = np.nan
                new_header['DATAMAX'] = np.nan

        new_wcs = WCS(new_header)
        return cutout_data, new_header, new_wcs, actual_bounds_wcs


class CutoutSettingsDialog(QDialog):
    def __init__(
        self,
        fits_viewer,
        region: Optional[Region] = None,
        parent=None,
        data_override=None,
        header_override=None,
        wcs_override=None,
        dialog_title: Optional[str] = None,
        collapsed_axes: Optional[Sequence[int]] = None,
        wcs_to_data_axis: Optional[Sequence[Optional[int]]] = None,
    ):
        super().__init__(parent or fits_viewer)
        self.setWindowTitle(dialog_title or "Cut Out")

        self.fits_viewer = fits_viewer
        self.region = region

        self.data = data_override if data_override is not None else getattr(fits_viewer, 'integrated_data', None)
        if self.data is None:
            self.data = getattr(fits_viewer, 'data', None)
        self.data_ndim = getattr(self.data, 'ndim', 0) if self.data is not None else 0

        header_source = header_override if header_override is not None else getattr(fits_viewer, 'header', None)
        self.header = header_source.copy() if header_source is not None else None

        wcs_source = wcs_override if wcs_override is not None else getattr(fits_viewer, 'wcs', None)
        self.wcs = wcs_source.deepcopy() if hasattr(wcs_source, 'deepcopy') else wcs_source

        if self.data is None or self.header is None or self.wcs is None:
            QMessageBox.critical(self, "Cut Out Error", "Required data or WCS information is unavailable.")
            self.close()
            return

        if self.wcs.naxis > 3:
            QMessageBox.warning(self, "Not Supported", "Current cut-out implementation supports up to three axes.")
            self.close()
            return

        self._wcs_to_data_axis = self._initialize_axis_mapping(wcs_to_data_axis)
        self._data_to_wcs_axis = {
            data_axis: wcs_axis
            for wcs_axis, data_axis in enumerate(self._wcs_to_data_axis)
            if data_axis is not None
        }

        config_manager = getattr(fits_viewer, 'config_manager', None)
        if config_manager is not None:
            self.config = config_manager.config
        else:
            self.config = getattr(fits_viewer, 'config', {})
        self.converter = CoordinateConverter(self.wcs, self.config)
        self.axis_types = self.converter.get_axis_types()
        self.axis_units = [self._unit_string(i) for i in range(self.wcs.naxis)]
        roles_provider = getattr(fits_viewer, 'get_axis_roles', None)
        if callable(roles_provider):
            try:
                self.axis_roles = list(roles_provider())
            except Exception:
                self.axis_roles = []
        else:
            self.axis_roles = []
        naxis = self.wcs.naxis
        if not self.axis_roles:
            if naxis >= 1:
                self.axis_roles.append('display_x')
            if naxis >= 2:
                self.axis_roles.append('display_y')
        while len(self.axis_roles) < naxis:
            self.axis_roles.append('depth')
        if len(self.axis_roles) > naxis:
            self.axis_roles = self.axis_roles[:naxis]

        self.collapsed_axes = set(idx for idx, role in enumerate(self.axis_roles) if role == 'collapsed')
        self.collapsed_axes.update(collapsed_axes or [])
        for idx, role in enumerate(self.axis_roles):
            if role == 'collapsed':
                self.collapsed_axes.add(idx)
        for idx, data_axis in enumerate(self._wcs_to_data_axis):
            if data_axis is None:
                self.collapsed_axes.add(idx)

        self._updating_fields = False
        self._current_pixel_bounds: List[Tuple[int, int]] = []

        self._build_ui()
        self._populate_initial_values()

    # ------------------------------------------------------------------
    def _unit_string(self, axis_index: int) -> str:
        try:
            unit = self.wcs.wcs.cunit[axis_index]
            if unit is None or unit == u.dimensionless_unscaled:
                return ''
            unit_str = unit.to_string(format='fits')
            return unit_str
        except Exception:
            return ''

    def _initialize_axis_mapping(
        self, mapping: Optional[Sequence[Optional[int]]]
    ) -> List[Optional[int]]:
        if mapping is not None:
            if len(mapping) != self.wcs.naxis:
                raise ValueError("Provided axis mapping length does not match WCS axes.")
            return [int(m) if m is not None else None for m in mapping]

        data_ndim = getattr(self.data, 'ndim', 0)
        inferred: List[Optional[int]] = []
        for axis in range(self.wcs.naxis):
            data_axis = data_ndim - axis - 1
            if 0 <= data_axis < data_ndim:
                inferred.append(data_axis)
            else:
                inferred.append(None)
        return inferred

    def _axis_short_label(self, axis_index: int) -> str:
        axis_names = ['X', 'Y', 'Z']
        if axis_index < len(axis_names):
            return f"{axis_names[axis_index]}:"
        return f"Axis {axis_index + 1}:"

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        self.tabs = QTabWidget(self)
        layout.addWidget(self.tabs)

        self.center_tab = QWidget(self)
        self.range_tab = QWidget(self)
        self.tabs.addTab(self.range_tab, "Min / Max")
        self.tabs.addTab(self.center_tab, "Center + Width")

        self.center_inputs: List[QLineEdit] = []
        self.size_inputs: List[QLineEdit] = []
        center_layout = QGridLayout()
        self.center_tab.setLayout(center_layout)

        for axis, _ in enumerate(self.axis_types[:self.wcs.naxis]):
            label_text = self._axis_short_label(axis)
            center_input = QLineEdit(self)
            size_input = QLineEdit(self)
            center_layout.addWidget(QLabel(label_text, self), axis, 0)
            center_layout.addWidget(center_input, axis, 1)
            center_layout.addWidget(QLabel("Width", self), axis, 2)
            center_layout.addWidget(size_input, axis, 3)
            self.center_inputs.append(center_input)
            self.size_inputs.append(size_input)
            editable = not self._is_collapsed_axis(axis)
            if editable:
                center_input.editingFinished.connect(self._on_user_inputs_finished)
                size_input.editingFinished.connect(self._on_user_inputs_finished)
            else:
                center_input.setEnabled(False)
                size_input.setEnabled(False)
                center_input.setText("0")
                size_input.setText("1")

        self.min_inputs: List[QLineEdit] = []
        self.max_inputs: List[QLineEdit] = []
        range_layout = QGridLayout()
        self.range_tab.setLayout(range_layout)

        for axis, _ in enumerate(self.axis_types[:self.wcs.naxis]):
            label_text = self._axis_short_label(axis)
            min_input = QLineEdit(self)
            max_input = QLineEdit(self)
            range_layout.addWidget(QLabel(label_text, self), axis, 0)
            range_layout.addWidget(min_input, axis, 1)
            range_layout.addWidget(max_input, axis, 2)
            self.min_inputs.append(min_input)
            self.max_inputs.append(max_input)
            editable = not self._is_collapsed_axis(axis)
            if editable:
                min_input.editingFinished.connect(self._on_user_inputs_finished)
                max_input.editingFinished.connect(self._on_user_inputs_finished)
            else:
                min_input.setEnabled(False)
                max_input.setEnabled(False)
                min_input.setText("0")
                max_input.setText("0")

        controls_row = QHBoxLayout()
        self.reset_view_button = QPushButton("Sync Range From View", self)
        self.reset_view_button.clicked.connect(self._on_reset_to_view)
        controls_row.addWidget(self.reset_view_button)

        self.apply_view_button = QPushButton("Sync View From Range", self)
        self.apply_view_button.clicked.connect(self._on_sync_view_to_range)
        controls_row.addWidget(self.apply_view_button)

        controls_row.addStretch(1)
        layout.addLayout(controls_row)

        self.pixel_summary_label = QLabel(self)
        self.pixel_summary_label.setObjectName("cutoutPixelSummary")
        layout.addWidget(self.pixel_summary_label)

        button_row = QHBoxLayout()
        layout.addLayout(button_row)

        self.save_button = QPushButton("Save FITS", self)
        self.save_button.clicked.connect(self._handle_save)
        self.save_button.setDefault(True)
        button_row.addWidget(self.save_button)

        button_row.addStretch(1)

        self.close_button = QPushButton("Close", self)
        self.close_button.clicked.connect(self.close)
        button_row.addWidget(self.close_button)

        self.setMinimumWidth(420)
        self.tabs.setCurrentIndex(0)

    # ------------------------------------------------------------------
    def _populate_initial_values(self) -> None:
        pixel_bounds: Optional[List[Tuple[int, int]]] = None
        if self.region is not None:
            try:
                pixel_bounds = self._pixel_bounds_from_region(self.region)
            except Exception:
                pixel_bounds = None
        if pixel_bounds is None:
            pixel_bounds = self._pixel_bounds_from_view()
        self._apply_integer_pixel_bounds(pixel_bounds)
        self._update_reset_buttons()

    def _normalize_pixel_bounds(self, raw_bounds: Sequence[Tuple[float, float]]):
        normalized: List[Tuple[int, int]] = []
        for lo, hi in raw_bounds:
            lo, hi = sorted((float(lo), float(hi)))
            start = int(math.floor(lo - 0.5)) + 1
            stop = int(math.ceil(hi + 0.5))
            if stop <= start:
                stop = start + 1
            normalized.append((start, stop))
        return normalized

    def _pixel_bounds_from_view(self) -> List[Tuple[int, int]]:
        ax = getattr(self.fits_viewer, 'ax', None)
        if ax is None:
            raise RuntimeError("Unable to determine the current view bounds.")

        x_bounds = tuple(sorted(ax.get_xlim()))
        y_bounds = tuple(sorted(ax.get_ylim()))

        bounds: List[Tuple[float, float]] = []
        for axis, role in enumerate(self.axis_roles):
            if role == 'display_x':
                bounds.append(x_bounds)
            elif role == 'display_y':
                bounds.append(y_bounds)
            elif role == 'collapsed':
                bounds.append((0.0, 0.0))
            else:
                bounds.append(self._get_depth_pixel_bounds(axis))

        return self._normalize_pixel_bounds(bounds)

    def _pixel_bounds_from_region(self, region: Region) -> List[Tuple[int, int]]:
        horizontal_bounds, vertical_bounds, depth_bounds = self._region_plane_bounds(region)

        bounds: List[Tuple[float, float]] = []
        for axis, role in enumerate(self.axis_roles):
            if role == 'display_x':
                bounds.append(horizontal_bounds if horizontal_bounds is not None else (0.0, 0.0))
            elif role == 'display_y':
                bounds.append(vertical_bounds if vertical_bounds is not None else (0.0, 0.0))
            elif role == 'collapsed':
                bounds.append((0.0, 0.0))
            else:
                if depth_bounds is not None:
                    bounds.append(depth_bounds)
                else:
                    bounds.append(self._get_depth_pixel_bounds(axis))

        while len(bounds) < self.wcs.naxis:
            bounds.append((0.0, 0.0))

        pixel_bounds = self._normalize_pixel_bounds(bounds)

        try:
            self._refine_region_display_bounds(region, pixel_bounds)
        except Exception:
            pass

        return pixel_bounds

    def _refine_region_display_bounds(self, region: Region, pixel_bounds: List[Tuple[int, int]]) -> None:
        x_axis = next((idx for idx, role in enumerate(self.axis_roles) if role == 'display_x'), None)
        y_axis = next((idx for idx, role in enumerate(self.axis_roles) if role == 'display_y'), None)
        if x_axis is None or y_axis is None:
            return

        x_start, x_stop = pixel_bounds[x_axis]
        y_start, y_stop = pixel_bounds[y_axis]
        if x_stop <= x_start or y_stop <= y_start:
            return

        x_centers = np.arange(x_start, x_stop, dtype=float)
        y_centers = np.arange(y_start, y_stop, dtype=float)
        if x_centers.size == 0 or y_centers.size == 0:
            return

        grid_x, grid_y = np.meshgrid(x_centers, y_centers, indexing='xy')
        inside = region.contains(grid_x, grid_y)
        if not np.any(inside):
            return

        valid_x = np.where(inside.any(axis=0))[0]
        valid_y = np.where(inside.any(axis=1))[0]
        if valid_x.size == 0 or valid_y.size == 0:
            return

        pixel_bounds[x_axis] = (x_start + int(valid_x[0]), x_start + int(valid_x[-1]) + 1)
        pixel_bounds[y_axis] = (y_start + int(valid_y[0]), y_start + int(valid_y[-1]) + 1)

    @staticmethod
    def _rotated_rectangle_bounds(region: RectangleRegion) -> List[Tuple[float, float]]:
        cx = region.xy[0] + region.width / 2.0
        cy = region.xy[1] + region.height / 2.0
        half_w = region.width / 2.0
        half_h = region.height / 2.0
        angle = math.radians(getattr(region, 'angle', 0.0))
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        corners = []
        for dx in (-half_w, half_w):
            for dy in (-half_h, half_h):
                x = cx + cos_a * dx - sin_a * dy
                y = cy + sin_a * dx + cos_a * dy
                corners.append((x, y))
        xs, ys = zip(*corners)
        return [(min(xs), max(xs)), (min(ys), max(ys))]

    @staticmethod
    def _ellipse_bounds(region: EllipseRegion) -> List[Tuple[float, float]]:
        cx, cy = region.center
        a = region.width / 2.0
        b = region.height / 2.0
        angle = math.radians(getattr(region, 'angle', 0.0))
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        x_radius = math.hypot(a * cos_a, b * sin_a)
        y_radius = math.hypot(a * sin_a, b * cos_a)
        return [(cx - x_radius, cx + x_radius), (cy - y_radius, cy + y_radius)]

    def _integer_bounds_to_world(self, pixel_bounds: Sequence[Tuple[int, int]]):
        if not pixel_bounds:
            return [], [], []

        axis_indices = []
        for start, stop in pixel_bounds:
            if stop <= start:
                raise ValueError("Selected range collapses to zero width.")
            last = stop - 1
            axis_indices.append([start, last])

        pixel_corners = np.array(list(product(*axis_indices)), dtype=float)
        world_array = self.wcs.wcs_pix2world(pixel_corners, 0)

        axis_samples: List[np.ndarray] = []
        axis_world_min: List[float] = []
        axis_world_max: List[float] = []
        self._axis_centers: List[float] = []  # type: ignore[attr-defined]
        self._axis_spans: List[float] = []    # type: ignore[attr-defined]

        for axis, samples in enumerate(world_array.T):
            axis_samples.append(samples)
            lo, hi, center, span = self._compute_axis_bounds(self.axis_types[axis], samples)
            axis_world_min.append(lo)
            axis_world_max.append(hi)
            self._axis_centers.append(center)
            self._axis_spans.append(span)

        return axis_world_min, axis_world_max, axis_samples

    def _is_wrap_axis(self, axis_type: str) -> bool:
        axis_upper = axis_type.upper()
        return 'GLON' in axis_upper or axis_upper.startswith('RA') or axis_upper.endswith('LON')

    def _compute_axis_bounds(self, axis_type: str, samples: Sequence[float]):
        values = np.asarray(samples, dtype=float)
        if values.size == 0:
            return 0.0, 0.0, 0.0, 0.0

        if not self._is_wrap_axis(axis_type):
            lo = float(np.nanmin(values))
            hi = float(np.nanmax(values))
            return lo, hi, (lo + hi) / 2.0, hi - lo

        # Handle circular coordinates (e.g., GLON/RA)
        normalized = np.mod(values, 360.0)
        normalized.sort()

        if normalized.size == 1:
            start = normalized[0]
            span = 0.0
        else:
            diffs = np.diff(np.concatenate([normalized, normalized[:1] + 360.0]))
            max_gap_idx = int(np.argmax(diffs))
            span = 360.0 - float(diffs[max_gap_idx])
            start = float(normalized[(max_gap_idx + 1) % normalized.size])

        end = start + span
        center = start + span / 2.0
        return start, end, center, span

    def _wrap_world_value(self, value: float, axis_type: str) -> float:
        if not self._is_wrap_axis(axis_type):
            return value

        coord_wrap = getattr(self.converter, 'coord_wrap', self.config.get('coord_wrap', 180))
        axis_type_upper = axis_type.upper()
        if 'GLON' in axis_type_upper or axis_type_upper.startswith('RA') or axis_type_upper.endswith('LON'):
            effective_wrap = 360 if axis_type_upper.startswith('RA') else coord_wrap
            if effective_wrap == 180:
                while value > 180:
                    value -= 360
                while value <= -180:
                    value += 360
            elif effective_wrap == 360:
                value = value % 360
        return value

    def _format_length(self, length: float) -> str:
        if length < 0:
            length = abs(length)
        return f"{length:.6f}"

    def _apply_integer_pixel_bounds(self, pixel_bounds: Sequence[Tuple[int, int]]):
        clamped_bounds = self._clamp_bounds_to_data(pixel_bounds)
        previous_world_bounds = getattr(self, '_current_world_bounds', None)
        self._current_pixel_bounds = clamped_bounds
        world_min, world_max, axis_samples = self._integer_bounds_to_world(self._current_pixel_bounds)

        if previous_world_bounds:
            for axis, mapping in enumerate(self._wcs_to_data_axis):
                if mapping is not None:
                    continue
                if axis >= len(previous_world_bounds):
                    continue
                prev_lo, prev_hi = previous_world_bounds[axis]
                world_min[axis] = prev_lo
                world_max[axis] = prev_hi
                if hasattr(self, '_axis_centers') and axis < len(self._axis_centers):
                    self._axis_centers[axis] = (prev_lo + prev_hi) / 2.0
                if hasattr(self, '_axis_spans') and axis < len(self._axis_spans):
                    self._axis_spans[axis] = prev_hi - prev_lo
                if axis < len(axis_samples):
                    axis_samples[axis] = np.array([prev_lo, prev_hi], dtype=float)

        self._axis_world_samples = axis_samples
        self._current_world_bounds = list(zip(world_min, world_max))
        self._apply_world_bounds(world_min, world_max, axis_samples)
        self._update_pixel_summary()
        return world_min, world_max

    def _update_pixel_summary(self):
        if not self._current_pixel_bounds:
            self.pixel_summary_label.clear()
            return
        parts: List[str] = []

        if self.region is not None:
            region_type = self.region.__class__.__name__.replace('Region', '')
            label = getattr(self.region, 'label_text', '')
            header = f"Region: {region_type}"
            if label:
                header += f" ({label})"
            parts.append(header)

            angle = getattr(self.region, 'angle', None)
            if angle is not None:
                parts.append(f"  Angle: {float(angle):.2f} deg")
            else:
                parts.append("  Angle: --")
        else:
            parts.append("Region: View bounds")
            parts.append("  Angle: --")

        for axis, (start, stop) in enumerate(self._current_pixel_bounds):
            axis_label = _axis_display_name(self.axis_types[axis])
            if self._is_collapsed_axis(axis):
                parts.append(f"{axis_label}: collapsed (fixed)")
            else:
                count = stop - start
                parts.append(f"{axis_label}: [{start}, {stop - 1}] px  (count: {count})")

        self.pixel_summary_label.setText("\n".join(parts))

    def _update_reset_buttons(self):
        if hasattr(self, 'reset_view_button'):
            self.reset_view_button.setEnabled(True)

    def _on_reset_to_view(self):
        self.reset_to_view()

    def _on_sync_view_to_range(self):
        if self._updating_fields:
            return
        try:
            pixel_bounds = self._ensure_bounds_from_ui()
        except Exception as exc:
            QMessageBox.warning(self, "Sync View Failed", f"Could not interpret the current cutout range.\n{exc}")
            return

        if not pixel_bounds:
            return

        try:
            self._apply_cutout_bounds_to_view(pixel_bounds)
        except Exception as exc:
            QMessageBox.warning(self, "Sync View Failed", f"Could not update the view.\n{exc}")

    def _ensure_bounds_from_ui(self) -> List[Tuple[int, int]]:
        try:
            world_min, world_max = self._collect_world_bounds_from_ui()
        except Exception:
            if self._current_pixel_bounds and getattr(self, '_current_world_bounds', None):
                return [tuple(bounds) for bounds in self._current_pixel_bounds]
            raise

        pixel_bounds = self._world_to_pixel_bounds(world_min, world_max)
        self._apply_integer_pixel_bounds(pixel_bounds)

        if not self._current_pixel_bounds or not getattr(self, '_current_world_bounds', None):
            raise ValueError("No valid cutout bounds are available.")

        return [tuple(bounds) for bounds in self._current_pixel_bounds]

    def _pixel_bounds_to_world_edges(self, pixel_bounds: Sequence[Tuple[int, int]]):
        default_world = getattr(self, '_current_world_bounds', None)
        if self.wcs is None:
            if default_world is not None:
                return [tuple(map(float, pair)) for pair in default_world]
            return [
                (float(bounds[0]), float(bounds[1]))
                for bounds in pixel_bounds
            ]

        axis_count = min(len(pixel_bounds), self.wcs.naxis)
        centers: List[float] = []
        for axis in range(self.wcs.naxis):
            if axis < len(pixel_bounds):
                start, stop = pixel_bounds[axis]
                span = float(stop - start)
                centers.append(float(start) + span / 2.0 - 0.5)
            else:
                centers.append(0.0)

        world_edges: List[Tuple[float, float]] = []
        for axis in range(axis_count):
            start, stop = pixel_bounds[axis]
            low = float(start) - 0.5
            high = float(stop) - 0.5

            low_vec = centers.copy()
            high_vec = centers.copy()
            low_vec[axis] = low
            high_vec[axis] = high

            try:
                converted = self.wcs.wcs_pix2world([low_vec, high_vec], 0)
                low_world = float(converted[0][axis])
                high_world = float(converted[1][axis])
                if not (math.isfinite(low_world) and math.isfinite(high_world)):
                    raise ValueError("Non-finite world coordinate")
            except Exception:
                if default_world is not None and axis < len(default_world):
                    lo_def, hi_def = default_world[axis]
                    world_edges.append((float(lo_def), float(hi_def)))
                else:
                    world_edges.append((0.0, 0.0))
                continue

            world_edges.append((low_world, high_world))

        if len(pixel_bounds) > axis_count:
            for axis in range(axis_count, len(pixel_bounds)):
                if default_world is not None and axis < len(default_world):
                    lo_def, hi_def = default_world[axis]
                    world_edges.append((float(lo_def), float(hi_def)))
                else:
                    start, stop = pixel_bounds[axis]
                    world_edges.append((float(start), float(stop)))

        return world_edges

    def _apply_cutout_bounds_to_view(
        self,
        pixel_bounds: Sequence[Tuple[int, int]],
    ) -> None:
        viewer = getattr(self, 'fits_viewer', None)
        if viewer is None:
            return

        ax = getattr(viewer, 'ax', None)
        if ax is None:
            return

        world_edges = self._pixel_bounds_to_world_edges(pixel_bounds)

        def format_world(axis_index: int, value: float) -> str:
            axis_type = self.axis_types[axis_index]
            wrapped = self._wrap_world_value(value, axis_type)
            return self.converter.format_world_coordinate(wrapped, axis_type)

        def compute_limits(bounds: Tuple[int, int]) -> Tuple[float, float]:
            start, stop = bounds
            start_f = float(start) - 0.5
            stop_f = float(stop) - 0.5
            if not math.isfinite(start_f) or not math.isfinite(stop_f):
                raise ValueError("Encountered non-finite pixel bounds.")
            if stop_f <= start_f:
                stop_f = start_f + 1.0
            return start_f, stop_f

        axis_count = min(len(pixel_bounds), len(self.axis_roles))
        x_axis = next((idx for idx in range(axis_count) if self.axis_roles[idx] == 'display_x'), None)
        y_axis = next((idx for idx in range(axis_count) if self.axis_roles[idx] == 'display_y'), None)

        collapsed = getattr(self, 'collapsed_axes', set())
        z_axis = next(
            (idx for idx in range(axis_count)
             if idx not in (x_axis, y_axis) and idx not in collapsed),
            None,
        )

        view_changed = False

        if x_axis is not None and 0 <= x_axis < len(pixel_bounds):
            x_limits = compute_limits(pixel_bounds[x_axis])
            current_xlim = ax.get_xlim()
            if current_xlim[0] > current_xlim[1]:
                ax.set_xlim(x_limits[1], x_limits[0])
            else:
                ax.set_xlim(x_limits[0], x_limits[1])
            view_changed = True

        if y_axis is not None and 0 <= y_axis < len(pixel_bounds):
            y_limits = compute_limits(pixel_bounds[y_axis])
            current_ylim = ax.get_ylim()
            if current_ylim[0] > current_ylim[1]:
                ax.set_ylim(y_limits[1], y_limits[0])
            else:
                ax.set_ylim(y_limits[0], y_limits[1])
            view_changed = True

        def set_line_edit_text(line_edit: Optional[QLineEdit], text: str) -> None:
            if line_edit is None:
                return
            block = line_edit.blockSignals(True)
            try:
                line_edit.setText(text)
            finally:
                line_edit.blockSignals(block)

        def apply_viewer_axis_updates(target_viewer, plane: str) -> None:
            if target_viewer is None:
                return
            if plane not in ('xy', 'xz', 'zy'):
                return

            need_x = plane in ('xy', 'xz') and x_axis is not None and x_axis < len(world_edges)
            need_y = plane in ('xy', 'zy') and y_axis is not None and y_axis < len(world_edges)
            need_z = plane in ('xz', 'zy') and z_axis is not None and z_axis < len(world_edges)

            if need_x:
                lo, hi = world_edges[x_axis]
                set_line_edit_text(getattr(target_viewer, 'x_min_input', None), format_world(x_axis, lo))
                set_line_edit_text(getattr(target_viewer, 'x_max_input', None), format_world(x_axis, hi))

            if need_y:
                lo, hi = world_edges[y_axis]
                set_line_edit_text(getattr(target_viewer, 'y_min_input', None), format_world(y_axis, lo))
                set_line_edit_text(getattr(target_viewer, 'y_max_input', None), format_world(y_axis, hi))

            if need_z:
                lo, hi = world_edges[z_axis]
                set_line_edit_text(getattr(target_viewer, 'z_min_input', None), format_world(z_axis, lo))
                set_line_edit_text(getattr(target_viewer, 'z_max_input', None), format_world(z_axis, hi))

        apply_viewer_axis_updates(viewer, getattr(viewer, 'plane', 'xy'))

        subwindows = getattr(viewer, 'subwindows', []) or []
        for sub in subwindows:
            sub_ax = getattr(sub, 'ax', None)
            if sub_ax is None:
                continue

            plane = getattr(sub, 'plane', '')

            if plane == 'xz':
                if x_axis is not None and 0 <= x_axis < len(pixel_bounds):
                    x_limits = compute_limits(pixel_bounds[x_axis])
                    current = sub_ax.get_xlim()
                    if current[0] > current[1]:
                        sub_ax.set_xlim(x_limits[1], x_limits[0])
                    else:
                        sub_ax.set_xlim(x_limits[0], x_limits[1])
                if z_axis is not None and 0 <= z_axis < len(pixel_bounds):
                    z_limits = compute_limits(pixel_bounds[z_axis])
                    current = sub_ax.get_ylim()
                    if current[0] > current[1]:
                        sub_ax.set_ylim(z_limits[1], z_limits[0])
                    else:
                        sub_ax.set_ylim(z_limits[0], z_limits[1])
                if hasattr(sub, 'overlay_ax'):
                    try:
                        sub.overlay_ax.set_position(sub.ax.get_position())
                    except Exception:
                        pass
                if hasattr(sub, 'canvas'):
                    sub.canvas.draw_idle()
                apply_viewer_axis_updates(sub, plane)
            elif plane == 'zy':
                if y_axis is not None and 0 <= y_axis < len(pixel_bounds):
                    y_limits = compute_limits(pixel_bounds[y_axis])
                    current = sub_ax.get_ylim()
                    if current[0] > current[1]:
                        sub_ax.set_ylim(y_limits[1], y_limits[0])
                    else:
                        sub_ax.set_ylim(y_limits[0], y_limits[1])
                if z_axis is not None and 0 <= z_axis < len(pixel_bounds):
                    z_limits = compute_limits(pixel_bounds[z_axis])
                    current = sub_ax.get_xlim()
                    if current[0] > current[1]:
                        sub_ax.set_xlim(z_limits[1], z_limits[0])
                    else:
                        sub_ax.set_xlim(z_limits[0], z_limits[1])
                if hasattr(sub, 'overlay_ax'):
                    try:
                        sub.overlay_ax.set_position(sub.ax.get_position())
                    except Exception:
                        pass
                if hasattr(sub, 'canvas'):
                    sub.canvas.draw_idle()
                apply_viewer_axis_updates(sub, plane)

        if view_changed:
            if hasattr(viewer, 'overlay_ax'):
                try:
                    viewer.overlay_ax.set_position(viewer.ax.get_position())
                except Exception:
                    pass
            if hasattr(viewer, 'canvas'):
                viewer.canvas.draw_idle()
            if hasattr(viewer, 'redraw_main_overlay_and_blit'):
                try:
                    viewer.redraw_main_overlay_and_blit()
                except Exception:
                    pass

            range_panel = getattr(viewer, 'range_panel', None)
            plane = getattr(viewer, 'plane', 'xy')
            if range_panel is not None and plane == 'xy':
                try:
                    range_panel.update_ranges('xy', viewer.ax.get_xlim(), viewer.ax.get_ylim())
                except Exception:
                    pass

    def _on_user_inputs_finished(self):
        if self._updating_fields:
            return
        try:
            world_min, world_max = self._collect_world_bounds_from_ui()
            pixel_bounds = self._world_to_pixel_bounds(world_min, world_max)
            self._apply_integer_pixel_bounds(pixel_bounds)
        except Exception:
            pass

    def _clamp_bounds_to_data(self, pixel_bounds: Sequence[Tuple[int, int]]):
        clamped: List[Tuple[int, int]] = []
        for axis, (start, stop) in enumerate(pixel_bounds):
            if start is None or stop is None:
                clamped.append((0, 1))
                continue

            start_i = int(math.floor(start))
            stop_i = int(math.ceil(stop))

            data_axis = self._wcs_to_data_axis[axis]
            if data_axis is None or self.data is None or data_axis >= self.data.ndim:
                axis_len = 1
            else:
                axis_len = self.data.shape[data_axis]

            if axis_len <= 0:
                clamped.append((0, 1))
                continue

            clamped_start = max(start_i, 0)
            clamped_stop = min(stop_i, axis_len)

            if clamped_start >= axis_len:
                clamped_start = axis_len - 1
                clamped_stop = axis_len
            elif clamped_stop <= clamped_start:
                clamped_stop = clamped_start + 1
                if clamped_stop > axis_len:
                    clamped_stop = axis_len
                    clamped_start = max(0, clamped_stop - 1)

            clamped.append((clamped_start, clamped_stop))

        return clamped

    def _is_collapsed_axis(self, axis_index: int) -> bool:
        if axis_index >= len(self._wcs_to_data_axis):
            return True
        if self._wcs_to_data_axis[axis_index] is None:
            return True
        if axis_index in self.collapsed_axes:
            return True
        if axis_index < len(self.axis_roles):
            role = self.axis_roles[axis_index]
            if role == 'collapsed':
                return True
            if role in ('display_x', 'display_y'):
                return False
        return False

    def _region_plane_bounds(self, region: Region):
        horizontal = vertical = None
        depth = None

        if isinstance(region, CircleRegion):
            cx, cy = region.center
            r = float(region.radius)
            horizontal = (cx - r, cx + r)
            vertical = (cy - r, cy + r)
        elif isinstance(region, (RectangleRegion, CubeRegion)):
            rect_bounds = self._rotated_rectangle_bounds(region)
            horizontal, vertical = rect_bounds[0], rect_bounds[1]
        elif isinstance(region, EllipseRegion):
            ell_bounds = self._ellipse_bounds(region)
            horizontal, vertical = ell_bounds[0], ell_bounds[1]
        else:
            raise ValueError("The selected region shape is not supported.")

        if isinstance(region, CubeRegion):
            z_min = math.floor(region.z_min)
            z_max = math.ceil(region.z_max)
            if z_max < z_min:
                z_min, z_max = z_max, z_min
            depth = (float(z_min), float(z_max))

        return horizontal, vertical, depth

    def _data_axis_length(self, wcs_axis_index: int) -> int:
        if self.data is None:
            return 1
        if wcs_axis_index >= len(self._wcs_to_data_axis):
            return 1
        data_axis = self._wcs_to_data_axis[wcs_axis_index]
        if data_axis is None or data_axis < 0 or data_axis >= self.data_ndim:
            return 1
        try:
            length = int(self.data.shape[data_axis])
            return max(length, 1)
        except Exception:
            return 1

    def _get_depth_pixel_bounds(self, wcs_axis_index: int) -> Tuple[float, float]:
        axis_type = self.axis_types[wcs_axis_index].upper()
        if hasattr(self.fits_viewer, 'current_z_pixel_bounds'):
            if any(key in axis_type for key in ('VRAD', 'VELO', 'FREQ', 'WAVE', 'Z')):
                try:
                    start, stop = self.fits_viewer.current_z_pixel_bounds()
                    if stop <= start:
                        stop = start + 1
                    return (float(start), float(stop - 1))
                except Exception:
                    pass

        length = self._data_axis_length(wcs_axis_index)
        return (0.0, float(length - 1))

    def _apply_world_bounds(self, world_min: Sequence[float], world_max: Sequence[float], axis_samples: Optional[Sequence[Sequence[float]]] = None) -> None:
        formatted_min = []
        formatted_max = []
        formatted_center = []
        formatted_size = []

        self._updating_fields = True
        try:
            for axis, (wmin, wmax) in enumerate(zip(world_min, world_max)):
                axis_type = self.axis_types[axis]
                center_val = getattr(self, '_axis_centers', [None]*len(world_min))[axis] if hasattr(self, '_axis_centers') else (wmin + wmax) / 2.0
                span_val = getattr(self, '_axis_spans', [None]*len(world_min))[axis] if hasattr(self, '_axis_spans') else (wmax - wmin)

                wrapped_min = self._wrap_world_value(wmin, axis_type)
                wrapped_max = self._wrap_world_value(wmax, axis_type)
                wrapped_center = self._wrap_world_value(center_val, axis_type)

                if self._is_wrap_axis(axis_type) and wrapped_max < wrapped_min:
                    pass

                formatted_min.append(self.converter.format_world_coordinate(wrapped_min, axis_type))
                formatted_max.append(self.converter.format_world_coordinate(wrapped_max, axis_type))
                formatted_center.append(self.converter.format_world_coordinate(wrapped_center, axis_type))
                formatted_size.append(self._format_length(span_val))

            for idx, (text_min, text_max) in enumerate(zip(formatted_min, formatted_max)):
                self.min_inputs[idx].setText(text_min)
                self.max_inputs[idx].setText(text_max)

            for idx, (center_text, size_text) in enumerate(zip(formatted_center, formatted_size)):
                self.center_inputs[idx].setText(center_text)
                self.size_inputs[idx].setText(size_text)
        finally:
            self._updating_fields = False

    # ------------------------------------------------------------------
    def _handle_save(self) -> None:
        try:
            world_min, world_max = self._collect_world_bounds_from_ui()
            pixel_bounds = self._world_to_pixel_bounds(world_min, world_max)

            # Create minimal AppState for compute_cutout
            if hasattr(self.fits_viewer, 'app_state'):
                 state = self.fits_viewer.app_state
            else:
                 # Fallback if app_state not attached (should be there in phase 5)
                 state = create_app_state(
                     data=self.data,
                     header=self.header,
                     wcs=self.wcs,
                     filepath=getattr(self.fits_viewer, 'filename', None)
                 )

            # Use usecase to extract cutout
            # compute_cutout expects pixel_bounds in WCS axis order?
            # compute_cutout doc says: "Order follows WCS convention: [x_bounds, y_bounds, z_bounds, ...]"
            # self._world_to_pixel_bounds returns bounds corresponding to WCS axes.
            
            result = compute_cutout(state, pixel_bounds=pixel_bounds)
            
            cutout_data = result.data
            new_header = result.header
            # result.pixel_bounds holds actual bounds used
            actual_pixel_bounds = result.pixel_bounds
            
            original_pixel_bounds = list(actual_pixel_bounds)

            # Apply region masking and trimming (GUI specific behavior)
            cutout_data, actual_pixel_bounds = self._apply_region_mask(cutout_data, actual_pixel_bounds)
            self._adjust_header_for_trim(new_header, original_pixel_bounds, actual_pixel_bounds)

            self._apply_integer_pixel_bounds(actual_pixel_bounds)

            save_header = self._collapse_header_for_save(new_header, cutout_data.shape)
            history_ranges = self._history_ranges_from_header(save_header, cutout_data.shape)
            suffix = self._build_suffix()

            # We can stick to SaveFITS for now to preserve strict saving behavior, 
            # or map to export_cutout_fits. 
            # Given the complex header collapsing logic, SaveFITS with the prepared header is safer 
            # to preserve exact behavior than export_cutout_fits which reconstructs header simply.
            
            saver = SaveFITS(cutout_data, save_header, self.fits_viewer.filename)
            self._append_history(save_header, history_ranges)
            saver.save(suffix=suffix)
            
        except Exception as exc:
            QMessageBox.critical(self, "Cut Out Error", f"Cut out failed:\n{exc}")

    def _apply_region_mask(self, data: np.ndarray, pixel_bounds: Sequence[Tuple[int, int]]):
        if self.region is None:
            return data, list(pixel_bounds)

        x_axis = next((idx for idx, role in enumerate(self.axis_roles) if role == 'display_x'), None)
        y_axis = next((idx for idx, role in enumerate(self.axis_roles) if role == 'display_y'), None)
        if x_axis is None or y_axis is None:
            return data, list(pixel_bounds)

        x_data_axis = self._wcs_to_data_axis[x_axis] if x_axis < len(self._wcs_to_data_axis) else None
        y_data_axis = self._wcs_to_data_axis[y_axis] if y_axis < len(self._wcs_to_data_axis) else None
        if x_data_axis is None or y_data_axis is None:
            return data, list(pixel_bounds)

        x_start, x_stop = pixel_bounds[x_axis]
        y_start, y_stop = pixel_bounds[y_axis]
        x_coords = np.arange(x_start, x_stop, dtype=float)
        y_coords = np.arange(y_start, y_stop, dtype=float)
        if x_coords.size == 0 or y_coords.size == 0:
            return data, list(pixel_bounds)

        grid_x, grid_y = np.meshgrid(x_coords, y_coords, indexing='xy')

        inside_xy = self.region.contains(grid_x, grid_y)
        mask_xy = ~inside_xy

        result = data.astype(np.float32, copy=True)

        expanded_mask = mask_xy.astype(bool)
        while expanded_mask.ndim < result.ndim:
            expanded_mask = np.expand_dims(expanded_mask, axis=-1)

        permutation: List[int] = []
        extra_axis = 2
        for axis in range(result.ndim):
            if axis == y_data_axis:
                permutation.append(0)
            elif axis == x_data_axis:
                permutation.append(1)
            else:
                permutation.append(extra_axis)
                extra_axis += 1

        expanded_mask = np.transpose(expanded_mask, axes=permutation)
        expanded_mask = np.broadcast_to(expanded_mask, result.shape)
        result[expanded_mask] = np.nan

        valid_mask = ~np.isnan(result)

        axes_except_x = tuple(i for i in range(result.ndim) if i != x_data_axis)
        valid_x = np.any(valid_mask, axis=axes_except_x)
        if valid_x.any():
            x_indices = np.where(valid_x)[0]
            x_min_trim = int(x_indices[0])
            x_max_trim = int(x_indices[-1]) + 1
        else:
            x_min_trim = 0
            x_max_trim = result.shape[x_data_axis]

        axes_except_y = tuple(i for i in range(result.ndim) if i != y_data_axis)
        valid_y = np.any(valid_mask, axis=axes_except_y)
        if valid_y.any():
            y_indices = np.where(valid_y)[0]
            y_min_trim = int(y_indices[0])
            y_max_trim = int(y_indices[-1]) + 1
        else:
            y_min_trim = 0
            y_max_trim = result.shape[y_data_axis]

        trim_slices = [slice(None)] * result.ndim
        trim_slices[x_data_axis] = slice(x_min_trim, x_max_trim)
        trim_slices[y_data_axis] = slice(y_min_trim, y_max_trim)
        trimmed = result[tuple(trim_slices)]

        new_bounds = list(pixel_bounds)
        new_bounds[x_axis] = (
            new_bounds[x_axis][0] + x_min_trim,
            new_bounds[x_axis][0] + x_max_trim,
        )
        new_bounds[y_axis] = (
            new_bounds[y_axis][0] + y_min_trim,
            new_bounds[y_axis][0] + y_max_trim,
        )

        return trimmed, new_bounds

    def _build_suffix(self) -> str:
        if self.region and getattr(self.region, 'label_text', ''):
            label = self.region.label_text.strip()
            if label:
                safe = re.sub(r'[^0-9A-Za-z._-]+', '_', label)
                safe = safe.strip('_') or 'region'
                return f"cut_{safe}"
        return "cut"

    def _collapse_header_for_save(self, header, data_shape: Sequence[int]):
        active_axes = [idx for idx, data_axis in enumerate(self._wcs_to_data_axis) if data_axis is not None]
        if len(active_axes) == len(self._wcs_to_data_axis):
            return header

        if not active_axes:
            collapsed = fits.Header()
            for keyword in ('SIMPLE', 'BITPIX', 'EXTEND'):
                if keyword in header:
                    collapsed[keyword] = header[keyword]
            collapsed['NAXIS'] = len(data_shape)
            for axis, size in enumerate(reversed(data_shape), start=1):
                collapsed[f'NAXIS{axis}'] = int(size)
            return collapsed

        axis_order: List[int] = []
        for data_axis in sorted(self._data_to_wcs_axis.keys(), reverse=True):
            wcs_axis = self._data_to_wcs_axis[data_axis]
            if wcs_axis is None:
                continue
            if wcs_axis in active_axes:
                axis_order.append(wcs_axis)

        if not axis_order:
            return header

        axis_lookup = {src_axis + 1: new_axis for new_axis, src_axis in enumerate(axis_order, start=1)}

        collapsed_header = fits.Header()
        for keyword in ('SIMPLE', 'BITPIX', 'EXTEND'):
            if keyword in header:
                collapsed_header[keyword] = header[keyword]

        collapsed_header['NAXIS'] = len(data_shape)
        for axis, size in enumerate(reversed(data_shape), start=1):
            collapsed_header[f'NAXIS{axis}'] = int(size)

        if 'WCSAXES' in header:
            collapsed_header['WCSAXES'] = (len(axis_order), header.comments['WCSAXES'])
        else:
            collapsed_header['WCSAXES'] = len(axis_order)

        scalar_keywords = ('CTYPE', 'CRVAL', 'CRPIX', 'CUNIT', 'CDELT', 'CROTA')
        for new_axis, src_axis in enumerate(axis_order, start=1):
            src_number = src_axis + 1
            for prefix in scalar_keywords:
                key_src = f'{prefix}{src_number}'
                if key_src in header:
                    card = header.cards[key_src]
                    collapsed_header[f'{prefix}{new_axis}'] = (card.value, card.comment)

        matrix_prefixes = ('CD', 'PC')
        for new_i, src_i in enumerate(axis_order, start=1):
            src_i_number = src_i + 1
            for new_j, src_j in enumerate(axis_order, start=1):
                src_j_number = src_j + 1
                for prefix in matrix_prefixes:
                    key_src = f'{prefix}{src_i_number}_{src_j_number}'
                    if key_src in header:
                        card = header.cards[key_src]
                        collapsed_header[f'{prefix}{new_i}_{new_j}'] = (card.value, card.comment)

        for prefix in ('PV', 'PS'):
            for card in header.cards:
                key = card.keyword
                if not key.startswith(prefix):
                    continue
                try:
                    base, index_part = key.split('_', 1)
                    axis_num = int(''.join(filter(str.isdigit, base[len(prefix):])))
                except Exception:
                    continue
                if axis_num not in axis_lookup:
                    continue
                new_key = f'{prefix}{axis_lookup[axis_num]}_{index_part}'
                collapsed_header[new_key] = (card.value, card.comment)

        copied_special = {'HISTORY', 'COMMENT'}
        wcs_prefixes = scalar_keywords + matrix_prefixes + ('PV', 'PS')

        for card in header.cards:
            key = card.keyword
            if key in ('SIMPLE', 'BITPIX', 'EXTEND', 'NAXIS'):
                continue
            if key.startswith('NAXIS'):
                continue
            if any(key.startswith(prefix) for prefix in wcs_prefixes):
                continue
            if key in collapsed_header and key not in copied_special:
                continue
            if key == 'HISTORY':
                collapsed_header.add_history(card.value)
                continue
            if key == 'COMMENT':
                collapsed_header.add_comment(card.value)
                continue
            collapsed_header.append(card, useblanks=False)

        for new_axis, src_axis in enumerate(axis_order, start=1):
            self._apply_axis_unit(collapsed_header, new_axis, src_axis)

        self._convert_velocity_axes_to_kms(collapsed_header, axis_order)
        self._append_velocity_history(collapsed_header)

        return collapsed_header

    def _history_ranges_from_header(self, header, data_shape: Sequence[int]):
        try:
            wcs = WCS(header)
        except Exception:
            return []

        converter = CoordinateConverter(wcs, self.config)
        axis_types = converter.get_axis_types()

        size_per_axis: List[int] = []
        for axis in range(wcs.naxis):
            data_axis = len(data_shape) - axis - 1
            if 0 <= data_axis < len(data_shape):
                size = int(data_shape[data_axis])
            else:
                size = 1
            size_per_axis.append(max(size, 1))

        if not size_per_axis:
            return []

        corners = np.array(list(product(*[(0, size - 1) for size in size_per_axis])), dtype=float)
        if corners.size == 0:
            return []

        world_coords = wcs.wcs_pix2world(corners, 0)
        history_ranges = []
        for axis, axis_type in enumerate(axis_types):
            values = world_coords[:, axis]
            with np.errstate(all='ignore'):
                lo = float(np.nanmin(values))
                hi = float(np.nanmax(values))
            formatted_min = converter.format_world_coordinate(lo, axis_type)
            formatted_max = converter.format_world_coordinate(hi, axis_type)
            history_ranges.append((formatted_min, formatted_max, _axis_display_name(axis_type)))

        return history_ranges

    def _apply_axis_unit(self, header: fits.Header, new_axis: int, src_axis: int) -> None:
        unit_key = f'CUNIT{new_axis}'
        comment = header.comments[unit_key] if unit_key in header else ''

        unit_value = header[unit_key] if unit_key in header else ''
        resolved = self._resolve_unit_string(src_axis, unit_value)

        if resolved is None:
            return

        header[unit_key] = (resolved, comment)

    def _resolve_unit_string(self, src_axis: int, fallback) -> Optional[str]:
        if self._should_force_kms(src_axis):
            return 'km/s'

        unit_from_wcs = None
        try:
            if self.wcs is not None and 0 <= src_axis < self.wcs.wcs.naxis:
                unit_obj = self.wcs.wcs.cunit[src_axis]
                if unit_obj not in (None, u.dimensionless_unscaled):
                    unit_from_wcs = unit_obj.to_string(format='fits')
        except Exception:
            unit_from_wcs = None

        candidates: List[str] = []

        inferred = self._infer_label_unit(src_axis)
        if inferred:
            candidates.append(inferred)

        if unit_from_wcs is not None:
            candidates.append(unit_from_wcs)

        if fallback not in (None, ''):
            if not isinstance(fallback, str):
                fallback = str(fallback)
            candidates.append(str(fallback))

        for candidate in candidates:
            if candidate is None:
                continue
            if not isinstance(candidate, str):
                candidate = str(candidate)
            candidate = candidate.strip()
            if not candidate:
                continue
            normalized = self._normalize_unit(candidate)
            if normalized:
                return normalized

        return ''

    def _normalize_unit(self, unit_str: str) -> str:
        sanitized = unit_str.replace('\u2212', '-').replace('\u2013', '-').replace('\u2014', '-')
        try:
            normalized = u.Unit(sanitized, format='fits').to_string('fits')
            return self._strip_unit_whitespace(normalized)
        except Exception:
            try:
                normalized = u.Unit(sanitized).to_string('fits')
                return self._strip_unit_whitespace(normalized)
            except Exception:
                cleaned = sanitized.strip().replace(' ', '')
                lower = cleaned.lower()
                fallback_map = {
                    'ms-1': 'm/s',
                    'm s-1': 'm/s',
                    'm·s-1': 'm/s',
                    'm*s-1': 'm/s',
                    'kms-1': 'km/s',
                    'km s-1': 'km/s',
                    'km·s-1': 'km/s',
                    'km*s-1': 'km/s',
                    'm*s-1': 'm/s',
                    'kms^-1': 'km/s',
                    'ms^-1': 'm/s',
                    'km/s': 'km/s',
                    'm/s': 'm/s',
                }
                return fallback_map.get(lower, unit_str)

    @staticmethod
    def _strip_unit_whitespace(unit_str: str) -> str:
        sanitized = unit_str.replace('\u2212', '-').replace('\u2013', '-').replace('\u2014', '-')
        compact = sanitized.replace(' ', '')
        replacements = {
            'km/s': 'km/s',
            'm/s': 'm/s',
        }
        lower = compact.lower()
        if lower in replacements:
            return replacements[lower]
        return compact

    def _should_force_kms(self, src_axis: int) -> bool:
        if src_axis < 0 or src_axis >= len(self.axis_types):
            return False

        axis_type = self.axis_types[src_axis]
        if not axis_type:
            return False

        axis_type_upper = axis_type.upper()
        if not any(token in axis_type_upper for token in ('VRAD', 'VELO', 'VOPT')):
            return False

        viewer = self.fits_viewer
        converted = False
        if viewer is not None:
            converted = getattr(viewer, 'velocity_unit_converted', False)
            if not converted:
                converted = getattr(getattr(viewer, '__class__', object), 'velocity_unit_converted', False)

        if converted:
            return True

        header_unit = ''
        try:
            header_unit = self.header.get(f'CUNIT{src_axis + 1}', '') if self.header is not None else ''
        except Exception:
            header_unit = ''

        if isinstance(header_unit, str) and header_unit.strip().lower().replace(' ', '') == 'km/s':
            return True

        label_unit = self._infer_label_unit(src_axis)
        if isinstance(label_unit, str) and label_unit.strip().lower().replace(' ', '') == 'km/s':
            return True

        return False

    def _convert_velocity_axes_to_kms(self, header: fits.Header, axis_order: Sequence[int]) -> None:
        for new_axis, src_axis in enumerate(axis_order, start=1):
            axis_type = ''
            if 0 <= src_axis < len(self.axis_types):
                axis_type = self.axis_types[src_axis]
            axis_type_upper = axis_type.upper() if axis_type else ''
            if not any(token in axis_type_upper for token in ('VRAD', 'VELO', 'VOPT')):
                continue

            unit_key = f'CUNIT{new_axis}'
            cdelt_key = f'CDELT{new_axis}'
            crval_key = f'CRVAL{new_axis}'

            current_unit = header.get(unit_key, '')
            normalized_current = self._normalize_unit(current_unit) if current_unit else ''

            if normalized_current == 'km/s':
                header[unit_key] = 'km/s'
                continue

            conversion_factor = None
            if normalized_current:
                try:
                    conversion_factor = (1 * u.Unit(normalized_current)).to(u.km / u.s).value
                except Exception:
                    conversion_factor = None

            if conversion_factor is None:
                # Assume the axis is already in km/s; just set the unit
                header[unit_key] = 'km/s'
                continue

            if cdelt_key in header:
                try:
                    header[cdelt_key] = float(header[cdelt_key]) * conversion_factor
                except Exception:
                    pass
            if crval_key in header:
                try:
                    header[crval_key] = float(header[crval_key]) * conversion_factor
                except Exception:
                    pass

            header[unit_key] = 'km/s'

    def _infer_label_unit(self, wcs_axis: int) -> Optional[str]:
        if self.fits_viewer is None:
            return None

        try:
            role = self.axis_roles[wcs_axis]
        except Exception:
            role = None

        viewer_ax = getattr(self.fits_viewer, 'ax', None)
        if viewer_ax is None:
            return None

        label_text = ''
        if role == 'display_x':
            label_text = viewer_ax.get_xlabel() or ''
        elif role == 'display_y':
            label_text = viewer_ax.get_ylabel() or ''

        if not label_text and hasattr(self.fits_viewer, 'overlay_ax'):
            overlay_ax = getattr(self.fits_viewer, 'overlay_ax', None)
            if overlay_ax is not None:
                if role == 'display_x':
                    label_text = overlay_ax.get_xlabel() or ''
                elif role == 'display_y':
                    label_text = overlay_ax.get_ylabel() or ''

        if not label_text:
            return None

        match = re.search(r'\[(.*?)\]', label_text)
        if match:
            return match.group(1).strip()

        if 'km/s' in label_text:
            return 'km/s'
        if 'm/s' in label_text:
            return 'm/s'

        return None

    def _append_velocity_history(self, header: fits.Header) -> None:
        if getattr(self, '_velocity_history_added', False):
            return

        converted = False
        try:
            converted = getattr(self.fits_viewer, 'velocity_unit_converted', False)
        except Exception:
            converted = False

        if not converted:
            converted = getattr(getattr(self.fits_viewer, '__class__', object), 'velocity_unit_converted', False)

        if not converted:
            return

        message = 'Velocity axis converted to km/s inside takefits'
        for card in header.cards:
            if card.keyword == 'HISTORY' and card.value == message:
                self._velocity_history_added = True
                return

        header.add_history(message)
        self._velocity_history_added = True

    def _parse_length_value(self, value: str, axis_type: str) -> float:
        value = value.strip()
        if not value:
            raise ValueError("Enter both center and width.")
        try:
            return float(value)
        except ValueError:
            pass
        if any(token in value for token in ('h', 'm', 's', 'd', ':')):
            unit = u.hourangle if axis_type.upper().startswith('RA') else u.deg
            try:
                return Angle(value, unit=unit).degree
            except Exception as exc:
                raise ValueError(f"Could not parse coordinate value: '{value}'") from exc
        raise ValueError(f"Could not convert to float: '{value}'")

    def _collect_world_bounds_from_ui(self):
        axis_count = self.wcs.naxis
        world_min: List[float] = []
        world_max: List[float] = []

        use_center = self.tabs.currentWidget() is self.center_tab
        wrap_period = 360.0

        for axis in range(axis_count):
            axis_type = self.axis_types[axis]
            if use_center:
                center_text = self.center_inputs[axis].text().strip()
                size_text = self.size_inputs[axis].text().strip()
                if not center_text or not size_text:
                    raise ValueError("Enter both center and width.")
                center_val = self._parse_world_value(center_text, axis_type)
                size_val = self._parse_length_value(size_text, axis_type)
                if size_val < 0:
                    raise ValueError("Width must be positive.")
                half = size_val / 2.0
                lo_val = center_val - half
                hi_val = center_val + half
            else:
                min_text = self.min_inputs[axis].text().strip()
                max_text = self.max_inputs[axis].text().strip()
                if not min_text or not max_text:
                    raise ValueError("Enter both minimum and maximum values.")
                lo_val = self._parse_world_value(min_text, axis_type)
                hi_val = self._parse_world_value(max_text, axis_type)

            if lo_val > hi_val:
                if self._is_wrap_axis(axis_type):
                    hi_val += wrap_period
                else:
                    lo_val, hi_val = hi_val, lo_val

            world_min.append(lo_val)
            world_max.append(hi_val)

        return world_min, world_max

    def _parse_world_value(self, value: str, axis_type: str) -> float:
        if any(token in value for token in ('h', 'm', 's', 'd', ':')):
            try:
                if axis_type.upper().startswith('RA'):
                    return Angle(value, unit=u.hourangle).degree
                return Angle(value, unit=u.deg).degree
            except Exception as exc:
                raise ValueError(f"Could not parse coordinate value: '{value}'") from exc
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"Could not convert to float: '{value}'") from exc

    def _world_to_pixel_bounds(self, world_min: Sequence[float], world_max: Sequence[float]):
        combos = product(*[(lo, hi) for lo, hi in zip(world_min, world_max)])
        pixel_values = []
        for combo in combos:
            pix = self.wcs.wcs_world2pix([list(combo)], 0)[0]
            pixel_values.append(pix)
        pix_array = np.array(pixel_values)
        raw_bounds = list(zip(pix_array.min(axis=0).tolist(), pix_array.max(axis=0).tolist()))
        return self._normalize_pixel_bounds(raw_bounds)

    def _update_region_reference(self, region: Optional[Region]):
        self.region = region
        self._populate_initial_values()
        self._update_reset_buttons()

    def reset_region(self, region: Optional[Region]):
        """Re-initialize the dialog based on the provided region."""
        self._update_region_reference(region)

    def reset_to_view(self):
        self._update_region_reference(None)

    def _append_history(self, header, history_ranges: Sequence[Tuple[str, str, str]]) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header.add_history(f"Cutout generated by takefits on {timestamp}")
        source = getattr(self.fits_viewer, 'filename', None)
        if source:
            header.add_history(f"Source file: {source}")
        for lo, hi, axis_label in history_ranges:
            header.add_history(f"{axis_label} axis range: {lo} to {hi}")
        for entry in build_processing_history_lines(self.fits_viewer):
            header.add_history(entry)

    def _adjust_header_for_trim(self, header, original_bounds, trimmed_bounds):
        if header is None or original_bounds is None or trimmed_bounds is None:
            return

        for axis, (orig, new) in enumerate(zip(original_bounds, trimmed_bounds)):
            if orig is None or new is None:
                continue
            orig_start, orig_stop = orig
            new_start, new_stop = new
            delta = new_start - orig_start
            length = max(new_stop - new_start, 1)

            crpix_key = f"CRPIX{axis + 1}"
            if crpix_key in header:
                header[crpix_key] = float(header[crpix_key]) - delta

            naxis_key = f"NAXIS{axis + 1}"
            if naxis_key in header:
                header[naxis_key] = length
