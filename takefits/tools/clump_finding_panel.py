# tools/cloud_identifier_panel.py
"""
Clump Finding Panel for takefits2.
Provides clump/cloud identification using Clumpfind, FellWalker, and Dendrogram/SCIMES algorithms.
"""
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QGridLayout, QLineEdit,
                             QPushButton, QLabel, QHBoxLayout, QMessageBox,
                             QRadioButton, QGroupBox, QFileDialog, QTabWidget,
                             QComboBox, QButtonGroup, QCheckBox, QApplication,
                             QProgressBar, QProgressDialog, QDoubleSpinBox, QSpinBox,
                             QSizePolicy)
from PySide6.QtCore import Qt, QThread, QTimer
import numpy as np
import os
import csv
import random
import base64
import io
import textwrap
import zlib
from datetime import datetime
from scipy.ndimage import find_objects
from scipy import ndimage

from takefits.core.usecases.clump import (
    run_clumpfind,
    run_fellwalker,
    run_dendrogram,
    ClumpResult,
    check_scimes_availability,
    generate_catalog,
    export_clump_mask,
)
from takefits.logic.clump_worker import ClumpWorker
from takefits.logic.data_tools import (
    ensure_operation_memory_budget,
    estimate_materialized_nbytes,
    format_nbytes,
    is_lazy_scaled,
)
from takefits.logic.progress import CancellationToken


# Keep orphaned (cancelled / closed-over) clump jobs alive until their worker
# thread unwinds, so Qt never deletes a still-running QThread.  Each entry is a
# (QThread, ClumpWorker) tuple removed once the thread finishes.
_DETACHED_CLUMP_JOBS: set = set()
_WORKSPACE_MASK_MAX_RAW_BYTES = 256 * 1024 ** 2
from takefits.core.history_provenance import build_processing_history_lines
from takefits.core.contour_manager import ContourManager, ContourParameters, ContourSegment, ContourItemState, ContourState
from takefits.tools.base_panel import (
    clear_action_preview_record,
    confirm_pending_close,
    record_action_preview,
)


class ClumpFindingPanel(QWidget):
    """
    A widget for identifying clouds/clumps in FITS data using various algorithms.
    """

    # Color palette for displaying clumps
    CLUMP_COLORS = [
        '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
        '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4',
        '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000',
        '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9'
    ]

    def __init__(self, fits_viewer, subwindows):
        super().__init__()
        self.fits_viewer = fits_viewer
        self.subwindows = subwindows
        self.data = fits_viewer.data
        self.header = fits_viewer.header
        self.wcs = fits_viewer.wcs
        self.cube = getattr(fits_viewer, "cube", None)
        if self.cube is None:
            try:
                fits_viewer.update_cube()
                self.cube = getattr(fits_viewer, "cube", None)
            except Exception:
                self.cube = None

        self.result_mask = None  # 2D or 3D label mask
        self._base_result_mask = None
        self.analysis_data = None  # 2D image or 3D cube used for analysis
        self.contour_items = []
        self.mask_overlay = None
        self.catalog = []
        self._base_catalog = []
        self._last_clump_result = None
        self._edge_label_flags = {}
        self._edge_excluded_labels = set()

        self._label_view_active = False
        self._baseline_data = None
        self._cloud_overlay_ids = {}
        self._direct_contour_artists = {} # plane -> list of artists (for projected view)
        self._label_color_map = {}
        self.dendro_handler = None
        self._cached_dendro_params = None
        self._contour_update_timer = QTimer(self)
        self._contour_update_timer.setSingleShot(True)
        self._contour_update_timer.timeout.connect(self._update_cloud_contours_for_all_planes)
        self._invalid_edge_distance_cache = None
        self._order_state_cache = {}
        self._quality_refilter_timer = QTimer(self)
        self._quality_refilter_timer.setSingleShot(True)
        self._quality_refilter_timer.timeout.connect(self._apply_pending_quality_refilter)
        self._last_update_params = {}
        self._last_run_metadata = {}
        self._action_record_tag = "panel:clump"
        self._scimes_availability_checked = False
        self._scimes_available = None
        self._scimes_error = ""

        # Background worker state (clump finding runs off the UI thread).
        self._clump_thread = None
        self._clump_worker = None
        self._clump_cancel = None
        self._clump_job = None

        self._input_width = 90
        self._rms_input_width = 120
        self._projected_mode_setting = "all"
        self._id_order_default_2d = "size_area"
        self._id_order_default_3d = "velocity_centroid"
        self._id_order_options_2d = [
            ("detection", "Detection Order"),
            ("peak", "Peak Intensity"),
            ("flux", "Total Flux"),
            ("centroid_yx", "Centroid (Y,X)"),
            ("size_area", "Size (Area)"),
        ]
        self._id_order_options_3d = [
            ("detection", "Detection Order"),
            ("peak", "Peak Intensity"),
            ("flux", "Total Flux"),
            ("velocity_centroid", "Velocity Centroid"),
            ("size_volume", "Size (Volume)"),
        ]
        self.setWindowTitle(f'Clump Finding: {self.fits_viewer.filename}')
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # RMS noise input
        rms_layout = QHBoxLayout()
        rms_layout.addWidget(QLabel("RMS Noise:"))
        self.rms_input = QLineEdit()
        self.rms_input.setPlaceholderText("Enter RMS value manually")
        self.rms_input.setToolTip("RMS noise level. Used to set thresholds in units of sigma.")
        self.rms_input.setMaximumWidth(self._rms_input_width)
        self.rms_input.installEventFilter(self)
        rms_layout.addWidget(self.rms_input)
        rms_layout.addStretch()
        main_layout.addLayout(rms_layout)
        main_layout.addSpacing(4)

        # 3D data is analyzed in 3D; takefits already provides XY/XZ/ZY windows for inspection.
        self.xy_radio = None

        # Algorithm tabs
        self.tabs = QTabWidget()
        self.tabs.addTab(self._create_dendro_tab(), "Dendrogram")
        self.tabs.addTab(self._create_clumpfind_tab(), "Clumpfind")
        self.tabs.addTab(self._create_fellwalker_tab(), "FellWalker")
        main_layout.addWidget(self.tabs)

        # Display mode
        display_group = QGroupBox("Display Mode")
        display_layout = QVBoxLayout(display_group)
        display_row = QHBoxLayout()
        self.display_buttons = QButtonGroup()
        self.contour_radio = QRadioButton("Contours")
        self.mask_radio = QRadioButton("Label Cube")
        self.contour_radio.setChecked(True)
        self.display_buttons.addButton(self.contour_radio)
        self.display_buttons.addButton(self.mask_radio)
        display_row.addWidget(self.contour_radio)
        display_row.addWidget(self.mask_radio)
        # Projection mode for 3D contours (project all structures onto plane vs channel-following)
        self.projection_mode_checkbox = QCheckBox("Projected")
        self.projection_mode_checkbox.setToolTip(
            "If checked, project all structures onto the plane (integrated view, all contours).\\n"
            "If unchecked, follow the current channel (slice view)."
        )
        self.projection_mode_checkbox.setEnabled(False)  # Enabled for 3D data or SCIMES preselect
        self.projection_mode_checkbox.toggled.connect(self._on_projection_mode_toggled)
        display_row.addWidget(self.projection_mode_checkbox)
        display_row.addStretch()
        display_layout.addLayout(display_row)

        id_order_row = QHBoxLayout()
        self.id_order_label = QLabel("ID Order:")
        self.id_order_combo = QComboBox()
        self.id_order_combo.setEnabled(False)
        self.id_order_combo.currentIndexChanged.connect(self._on_id_order_changed)
        id_order_row.addWidget(self.id_order_label)
        id_order_row.addWidget(self.id_order_combo)
        id_order_row.addStretch()
        display_layout.addLayout(id_order_row)
        main_layout.addWidget(display_group)

        quality_group = QGroupBox("Quality Flags")
        quality_layout = QVBoxLayout(quality_group)
        self.edge_exclude_checkbox = QCheckBox("Hide edge-flagged clouds")
        self.edge_exclude_checkbox.setToolTip(
            "Hide labels touching the image edge or valid-data/NaN footprint edge. "
            "Mask and catalog exports follow the current visibility; the original "
            "detection result is retained."
        )
        quality_layout.addWidget(self.edge_exclude_checkbox)
        edge_row = QHBoxLayout()
        self.edge_margin_label = QLabel("Spatial edge margin:")
        edge_row.addWidget(self.edge_margin_label)
        self.edge_margin_spin = self._make_int_spin(0, minimum=0, maximum=10_000)
        self.edge_margin_spin.setToolTip(
            "0 px flags only labels touching the image edge or valid-data footprint. "
            "Increase to hide labels within N pixels of the spatial edge."
        )
        edge_row.addWidget(self.edge_margin_spin)
        self.edge_margin_unit_label = QLabel("px")
        edge_row.addWidget(self.edge_margin_unit_label)
        edge_row.addStretch()
        quality_layout.addLayout(edge_row)
        main_layout.addWidget(quality_group)

        # Action buttons
        # NOTE: A "Cancel" button intentionally has no UI here.  Cooperative
        # cancellation cannot interrupt the uninterruptible heavy calls (notably
        # astrodendro/SCIMES), so exposing it was misleading.  The cancellation
        # machinery is kept in the backend (cancel token, _cancel_clump_job,
        # _detach_running_job, worker cancel support) and is still used to close
        # the panel mid-run; re-add a button wired to _cancel_clump_job to revive.
        button_layout = QHBoxLayout()
        self.run_button = QPushButton("Run")
        self.export_catalog_button = QPushButton("Export Catalog")
        self.export_mask_button = QPushButton("Export Mask")
        self.clear_button = QPushButton("Clear")

        self.run_button.setDefault(True)
        button_layout.addWidget(self.run_button)
        button_layout.addWidget(self.export_catalog_button)
        button_layout.addWidget(self.export_mask_button)
        button_layout.addWidget(self.clear_button)
        button_layout.addStretch()
        main_layout.addLayout(button_layout)

        # Status
        self.status_label = QLabel("Status: Ready")
        self.status_label.setWordWrap(True)
        self.status_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )
        self.count_label = QLabel("Detected: -- clumps")
        main_layout.addWidget(self.status_label)
        main_layout.addWidget(self.count_label)

        # Progress bar (hidden until a job runs; busy mode for indeterminate steps)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # Connect signals
        self.run_button.clicked.connect(self.run_identification)
        self.export_catalog_button.clicked.connect(self.export_catalog)
        self.export_mask_button.clicked.connect(self.export_mask)
        self.clear_button.clicked.connect(self.clear_results)
        self.contour_radio.toggled.connect(self._on_display_mode_changed)
        self.mask_radio.toggled.connect(self._on_display_mode_changed)
        self.edge_margin_spin.valueChanged.connect(self._on_quality_settings_changed)
        self.edge_exclude_checkbox.toggled.connect(self._on_quality_settings_changed)
        self._update_edge_margin_enabled()

        # Initially disable export buttons
        self.export_catalog_button.setEnabled(False)
        self.export_mask_button.setEnabled(False)

        self.setLayout(main_layout)
        self.adjustSize()
        self.move_to_default_position()
        self._attach_slice_listeners()
        self._update_projected_availability()
        self._configure_id_order_options()

    def eventFilter(self, source, event):
        if event.type() == event.Type.KeyPress and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            # Ensure spinbox values are committed
            if isinstance(source, (QSpinBox, QDoubleSpinBox)):
                source.interpretText()
            self.run_button.animateClick()
            return True
        return super().eventFilter(source, event)

    def _make_double_spin(self, value, minimum=0.0, maximum=1.0e6):
        spin = QDoubleSpinBox()
        spin.setDecimals(1)
        spin.setSingleStep(1.0)
        spin.setRange(minimum, maximum)
        spin.setValue(float(value))
        spin.setFixedWidth(self._input_width)
        spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        spin.installEventFilter(self)
        return spin

    def _make_int_spin(self, value, minimum=0, maximum=1_000_000):
        spin = QSpinBox()
        spin.setRange(int(minimum), int(maximum))
        spin.setValue(int(value))
        spin.setFixedWidth(self._input_width)
        spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        spin.installEventFilter(self)
        return spin

    def _label_mode_shows_contours(self):
        return self.mask_radio.isChecked() and self.projection_mode_checkbox.isChecked()

    def _projected_mode(self):
        if not self.projection_mode_checkbox.isChecked():
            return "none"
        mode = getattr(self, "_projected_mode_setting", "all")
        return "max" if mode == "max" else "all"

    def _update_projected_availability(self):
        has_3d_result = self.result_mask is not None and self.result_mask.ndim == 3
        has_3d_data = self.data is not None and self.data.ndim >= 3
        scimes_ready = getattr(self, "use_scimes", None) is not None and self.use_scimes.isChecked()
        should_enable = has_3d_result or (scimes_ready and has_3d_data)
        self.projection_mode_checkbox.setEnabled(bool(should_enable))
        if not should_enable:
            self.projection_mode_checkbox.setChecked(False)

    def _configure_id_order_options(self):
        if not hasattr(self, "id_order_combo"):
            return
        current_key = self.id_order_combo.currentData()
        if self.result_mask is not None:
            is_3d = self.result_mask.ndim == 3
        else:
            is_3d = self.data is not None and self.data.ndim >= 3
        options = self._id_order_options_3d if is_3d else self._id_order_options_2d
        default_key = self._id_order_default_3d if is_3d else self._id_order_default_2d
        if not current_key:
            current_key = default_key

        self.id_order_combo.blockSignals(True)
        self.id_order_combo.clear()
        for key, label in options:
            self.id_order_combo.addItem(label, key)

        selected_index = 0
        if current_key:
            for i in range(self.id_order_combo.count()):
                if self.id_order_combo.itemData(i) == current_key:
                    selected_index = i
                    break
            else:
                for i in range(self.id_order_combo.count()):
                    if self.id_order_combo.itemData(i) == default_key:
                        selected_index = i
                        break
        self.id_order_combo.setCurrentIndex(selected_index)
        self.id_order_combo.blockSignals(False)
        self.id_order_combo.setEnabled(self.result_mask is not None)

    def _on_id_order_changed(self):
        if self.result_mask is None:
            return
        self._apply_id_order(refresh=True)

    def _on_quality_settings_changed(self):
        self._update_edge_margin_enabled()
        if self._base_result_mask is None:
            return
        # Debounce: the margin spin box fires valueChanged on every tick or
        # keystroke, and each refilter walks the full mask (find_objects,
        # relabel, copies) on the GUI thread.
        self._quality_refilter_timer.start(200)

    def _apply_pending_quality_refilter(self):
        if self._base_result_mask is None:
            return
        self._apply_id_order(refresh=True)

    def _update_edge_margin_enabled(self):
        if not hasattr(self, "edge_exclude_checkbox"):
            return
        enabled = bool(self.edge_exclude_checkbox.isChecked())
        for widget_name in (
            "edge_margin_label",
            "edge_margin_spin",
            "edge_margin_unit_label",
        ):
            widget = getattr(self, widget_name, None)
            if widget is not None:
                widget.setEnabled(enabled)

    def _edge_margin_value(self):
        if not hasattr(self, "edge_margin_spin"):
            return 0
        return max(0, int(self.edge_margin_spin.value()))

    @staticmethod
    def _finite_mask(values):
        try:
            finite = np.isfinite(np.asarray(values))
        except TypeError:
            return None
        if np.ma.isMaskedArray(values):
            finite = finite & ~np.ma.getmaskarray(values)
        return np.asarray(finite, dtype=bool)

    @classmethod
    def _spatial_valid_footprint(cls, data, spatial_shape):
        if data is None:
            return None
        try:
            data_array = np.asanyarray(data)
        except Exception:
            return None
        if data_array.ndim < 2 or tuple(data_array.shape[-2:]) != tuple(spatial_shape):
            return None

        if data_array.ndim == 2:
            return cls._finite_mask(data_array)

        spatial_valid = np.zeros(tuple(spatial_shape), dtype=bool)
        for leading_index in np.ndindex(tuple(data_array.shape[:-2])):
            finite_plane = cls._finite_mask(data_array[leading_index])
            if finite_plane is None:
                return None
            spatial_valid |= finite_plane
            if bool(np.all(spatial_valid)):
                break
        return spatial_valid

    @classmethod
    def _spatial_invalid_edge_distance_map(cls, data, spatial_shape):
        spatial_valid = cls._spatial_valid_footprint(data, spatial_shape)
        # The all-valid short-circuit is load-bearing: distance_transform_cdt
        # returns -1 everywhere when the input has no background (zero) pixel.
        if spatial_valid is None or not np.any(spatial_valid) or np.all(spatial_valid):
            return None

        distances = ndimage.distance_transform_cdt(spatial_valid, metric="taxicab")
        return np.maximum(np.asarray(distances, dtype=np.int32) - 1, 0)

    def _cached_invalid_edge_distance_map(self, spatial_shape):
        """Memoized _spatial_invalid_edge_distance_map for the current analysis data.

        The map depends only on analysis_data, which is reassigned in
        run_identification() and on workspace restore; both reset the cache.
        """
        spatial_shape = tuple(int(value) for value in spatial_shape)
        cache = getattr(self, "_invalid_edge_distance_cache", None)
        if cache is not None and cache[0] == spatial_shape:
            return cache[1]
        distance_map = self._spatial_invalid_edge_distance_map(
            getattr(self, "analysis_data", None), spatial_shape
        )
        self._invalid_edge_distance_cache = (spatial_shape, distance_map)
        return distance_map

    def _invalidate_order_caches(self):
        """Drop caches derived from the base mask / analysis data.

        Call after every reassignment of ``_base_result_mask`` or
        ``analysis_data``; the per-order relabel mappings, edge flags and the
        valid-footprint distance map are all functions of those two.
        """
        self._order_state_cache = {}
        self._invalid_edge_distance_cache = None

    @staticmethod
    def _edge_flags_from_distances(distances, invalid_distance=None):
        distance_values = list(distances.values())
        if invalid_distance is not None:
            distance_values.append(int(invalid_distance))
        edge_axes = [
            axis for axis, distance in distances.items()
            if distance == 0
        ]
        if invalid_distance == 0:
            edge_axes.append("valid_data")
        return {
            "spatial_edge_touch": bool(edge_axes),
            "spatial_edge_axes": ",".join(edge_axes),
            "distance_to_spatial_edge_pix": int(min(distance_values)),
        }

    @classmethod
    def _spatial_edge_flags_from_xy(cls, y_idx, x_idx, shape, invalid_edge_distance=None):
        if y_idx.size == 0 or x_idx.size == 0:
            return {
                "spatial_edge_touch": False,
                "spatial_edge_axes": "",
                "distance_to_spatial_edge_pix": -1,
            }

        ny = int(shape[-2])
        nx = int(shape[-1])
        distances = {
            "x_min": int(np.min(x_idx)),
            "x_max": int(nx - 1 - np.max(x_idx)),
            "y_min": int(np.min(y_idx)),
            "y_max": int(ny - 1 - np.max(y_idx)),
        }

        invalid_distance = None
        if invalid_edge_distance is not None:
            try:
                invalid_distance = int(np.min(invalid_edge_distance[y_idx, x_idx]))
            except IndexError:
                invalid_distance = None
        return cls._edge_flags_from_distances(distances, invalid_distance)

    def _compute_spatial_edge_flags(self, mask):
        """Return per-label spatial edge metadata."""
        mask = np.asarray(mask)
        if mask.ndim < 2 or mask.size == 0:
            return {}

        y_axis = mask.ndim - 2
        x_axis = mask.ndim - 1
        ny = int(mask.shape[y_axis])
        nx = int(mask.shape[x_axis])
        # A label is present exactly when its find_objects slice is not None,
        # so the slices double as the label list (no np.unique pass needed).
        obj_slices = find_objects(mask)
        if not obj_slices:
            return {}
        invalid_edge_distance = self._cached_invalid_edge_distance_map((ny, nx))
        leading_axes = tuple(range(mask.ndim - 2))

        flags = {}
        for label_index, obj_slice in enumerate(obj_slices):
            if obj_slice is None:
                continue
            label = label_index + 1

            y_slice = obj_slice[y_axis]
            x_slice = obj_slice[x_axis]
            # find_objects boxes are tight, so the slice bounds equal the exact
            # per-pixel min/max along each spatial axis.
            distances = {
                "x_min": int(x_slice.start),
                "x_max": int(nx - x_slice.stop),
                "y_min": int(y_slice.start),
                "y_max": int(ny - y_slice.stop),
            }
            invalid_distance = None
            if invalid_edge_distance is not None:
                spatial_footprint = mask[obj_slice] == label
                if leading_axes:
                    spatial_footprint = spatial_footprint.any(axis=leading_axes)
                invalid_distance = int(
                    invalid_edge_distance[y_slice, x_slice][spatial_footprint].min()
                )
            flags[label] = self._edge_flags_from_distances(distances, invalid_distance)
        return flags

    @staticmethod
    def _remove_labels_from_mask(mask, labels):
        labels = {int(label) for label in labels if int(label) > 0}
        if not labels:
            return np.array(mask, copy=True)

        max_label = int(np.max(mask)) if np.size(mask) else 0
        if max_label <= 0:
            return np.array(mask, copy=True)

        mapping = np.arange(max_label + 1, dtype=mask.dtype)
        for label in labels:
            if label <= max_label:
                mapping[label] = 0
        return mapping[mask]

    def _apply_quality_filter_to_mask(self, mask, flags_cache=None):
        """Filter ``mask`` by the edge-quality settings.

        ``mask`` must be an array the caller owns (a fresh copy or relabel
        output): when nothing is excluded it is returned as-is, and it becomes
        ``self.result_mask``, which must not alias ``self._base_result_mask``.

        ``flags_cache`` is the per-order-key dict owned by _apply_id_order;
        the edge flags depend only on the mask geometry (not the margin), so
        margin changes reuse them instead of re-running find_objects.
        """
        cached_flags = None if flags_cache is None else flags_cache.get("flags")
        if cached_flags is None:
            cached_flags = self._compute_spatial_edge_flags(mask)
            if flags_cache is not None:
                flags_cache["flags"] = cached_flags
        self._edge_label_flags = cached_flags
        if self.edge_exclude_checkbox.isChecked():
            margin = self._edge_margin_value()
            edge_labels = {
                label for label, flags in self._edge_label_flags.items()
                if int(flags.get("distance_to_spatial_edge_pix", -1)) <= margin
                and int(flags.get("distance_to_spatial_edge_pix", -1)) >= 0
            }
        else:
            edge_labels = set()

        if edge_labels:
            self._edge_excluded_labels = set(edge_labels)
            return self._remove_labels_from_mask(mask, edge_labels)

        self._edge_excluded_labels = set()
        return mask

    def _catalog_row_with_quality_flags(self, row):
        row_out = dict(row)
        try:
            label = int(row_out.get("id", 0))
        except (TypeError, ValueError):
            label = 0
        flags = self._edge_label_flags.get(label, {})
        row_out["spatial_edge_touch"] = bool(flags.get("spatial_edge_touch", False))
        row_out["spatial_edge_axes"] = str(flags.get("spatial_edge_axes", ""))
        row_out["distance_to_spatial_edge_pix"] = int(
            flags.get("distance_to_spatial_edge_pix", -1)
        )
        return row_out

    def _catalog_with_quality_flags(self, catalog):
        if not catalog:
            return []
        rows = []
        for row in catalog:
            try:
                label = int(row.get("id", 0))
            except (AttributeError, TypeError, ValueError):
                label = 0
            if label in self._edge_excluded_labels:
                continue
            rows.append(self._catalog_row_with_quality_flags(row))
        return rows

    def _spatial_edge_flags_from_indices(self, indices, shape, invalid_edge_distance=None):
        if len(indices) < 2 or len(shape) < 2:
            return {
                "spatial_edge_touch": False,
                "spatial_edge_axes": "",
                "distance_to_spatial_edge_pix": -1,
            }

        y_idx = np.asarray(indices[-2])
        x_idx = np.asarray(indices[-1])
        if y_idx.size == 0 or x_idx.size == 0:
            return {
                "spatial_edge_touch": False,
                "spatial_edge_axes": "",
                "distance_to_spatial_edge_pix": -1,
            }

        return self._spatial_edge_flags_from_xy(
            y_idx.astype(int, copy=False),
            x_idx.astype(int, copy=False),
            shape,
            invalid_edge_distance=invalid_edge_distance,
        )

    def _native_catalog_with_quality_flags(self, cat):
        if cat is None or self.dendro_handler is None or "_idx" not in cat.colnames:
            return cat

        out = cat.copy()
        data_shape = getattr(self.analysis_data, "shape", None)
        if data_shape is None and self._base_result_mask is not None:
            data_shape = self._base_result_mask.shape
        if data_shape is None:
            return out
        invalid_edge_distance = None
        if len(out) > 0:
            invalid_edge_distance = self._cached_invalid_edge_distance_map(
                data_shape[-2:]
            )

        edge_touch = []
        edge_axes = []
        distances = []
        for row in out:
            flags = None
            try:
                struct = self.dendro_handler.d[int(row["_idx"])]
                flags = self._spatial_edge_flags_from_indices(
                    struct.indices(),
                    data_shape,
                    invalid_edge_distance=invalid_edge_distance,
                )
            except Exception:
                pass
            if flags is None:
                flags = {
                    "spatial_edge_touch": False,
                    "spatial_edge_axes": "",
                    "distance_to_spatial_edge_pix": -1,
                }
            edge_touch.append(bool(flags["spatial_edge_touch"]))
            edge_axes.append(str(flags["spatial_edge_axes"]))
            distances.append(int(flags["distance_to_spatial_edge_pix"]))

        for name, values in (
            ("spatial_edge_touch", edge_touch),
            ("spatial_edge_axes", edge_axes),
            ("distance_to_spatial_edge_pix", distances),
        ):
            out[name] = values

        if self.edge_exclude_checkbox.isChecked() and len(out) > 0:
            distances = np.asarray(out["distance_to_spatial_edge_pix"], dtype=int)
            valid_distances = distances >= 0
            keep = np.logical_not(valid_distances & (distances <= self._edge_margin_value()))
            out = out[keep]
        return out

    def _update_result_count_label(self):
        if self._base_result_mask is None or self.result_mask is None:
            self.count_label.setText("Detected: -- clumps")
            return

        if self._edge_label_flags:
            # _apply_quality_filter_to_mask rebuilds the flags from the full
            # pre-exclusion mask (one entry per label, identical count to the
            # base mask since relabeling is a bijection), so the totals come
            # for free instead of two np.unique passes over the cube.
            total_count = len(self._edge_label_flags)
            visible_count = total_count - len(self._edge_excluded_labels)
        else:
            total_labels = np.unique(self._base_result_mask)
            total_count = int(np.count_nonzero(total_labels > 0))
            visible_labels = np.unique(self.result_mask)
            visible_count = int(np.count_nonzero(visible_labels > 0))
        edge_count = sum(
            1 for flags in self._edge_label_flags.values()
            if flags.get("spatial_edge_touch")
        )
        hidden_count = len(self._edge_excluded_labels)

        if hidden_count:
            self.count_label.setText(
                f"Detected: {visible_count}/{total_count} clumps "
                f"({hidden_count} edge-hidden)"
            )
        elif edge_count:
            self.count_label.setText(
                f"Detected: {total_count} clumps ({edge_count} edge-flagged)"
            )
        else:
            self.count_label.setText(f"Detected: {total_count} clumps")

    def _collect_label_stats(self, mask, data):
        labels = np.unique(mask)
        labels = labels[labels > 0]
        if labels.size == 0:
            return None

        data_clean = np.nan_to_num(data, nan=0.0)
        sums = ndimage.sum(data_clean, labels=mask, index=labels)
        peaks = ndimage.maximum(data_clean, labels=mask, index=labels)
        ones = np.ones_like(mask, dtype=np.float32)
        sizes = ndimage.sum(ones, labels=mask, index=labels)
        weighted_com = ndimage.center_of_mass(data_clean, labels=mask, index=labels)
        unweighted_com = ndimage.center_of_mass(ones, labels=mask, index=labels)

        centroids = []
        for i, (wcom, gcom) in enumerate(zip(weighted_com, unweighted_com)):
            if not np.isfinite(sums[i]) or sums[i] <= 0:
                centroids.append(gcom)
            else:
                centroids.append(wcom)

        return {
            "labels": np.asarray(labels, dtype=int),
            "peak": np.asarray(peaks, dtype=float),
            "flux": np.asarray(sums, dtype=float),
            "size": np.asarray(sizes, dtype=float),
            "centroid": centroids,
        }

    def _order_mapping_for_key(self, mask, order_key, data):
        """Return the old-label -> new-label mapping array for ``order_key``.

        Returns None when the mask has no labels. The mapping depends only on
        the mask and data, so callers can cache it and reapply it with
        ``mapping[mask]`` without recomputing the label statistics.
        """
        stats = self._collect_label_stats(mask, data)
        if stats is None:
            return None

        labels = stats["labels"]
        if order_key == "detection":
            sorted_labels = labels
        elif order_key == "peak":
            sort_idx = np.lexsort((labels, -stats["peak"]))
            sorted_labels = labels[sort_idx]
        elif order_key == "flux":
            sort_idx = np.lexsort((labels, -stats["flux"]))
            sorted_labels = labels[sort_idx]
        elif order_key in ("size_volume", "size_area"):
            sort_idx = np.lexsort((labels, -stats["size"]))
            sorted_labels = labels[sort_idx]
        elif order_key == "velocity_centroid":
            v = np.array([c[0] for c in stats["centroid"]], dtype=float)
            y = np.array([c[1] for c in stats["centroid"]], dtype=float)
            x = np.array([c[2] for c in stats["centroid"]], dtype=float)
            sort_idx = np.lexsort((labels, x, y, v))
            sorted_labels = labels[sort_idx]
        elif order_key == "centroid_yx":
            y = np.array([c[0] for c in stats["centroid"]], dtype=float)
            x = np.array([c[1] for c in stats["centroid"]], dtype=float)
            sort_idx = np.lexsort((labels, x, y))
            sorted_labels = labels[sort_idx]
        else:
            sorted_labels = labels

        max_label = int(labels.max())
        mapping = np.zeros(max_label + 1, dtype=mask.dtype)
        for new_id, old_id in enumerate(sorted_labels, start=1):
            mapping[int(old_id)] = int(new_id)
        return mapping

    def _apply_id_order(self, refresh=True):
        if self._base_result_mask is None:
            return
        order_key = self.id_order_combo.currentData() or "detection"
        data = self.analysis_data if self.analysis_data is not None else self.data
        # Relabel mappings and edge flags depend only on the base mask and
        # order key, not on the margin/visibility settings, so margin changes
        # reuse them from _order_state_cache (reset by _invalidate_order_caches).
        order_cache = self._order_state_cache.setdefault(order_key, {})
        if order_key == "detection":
            ordered_mask = np.array(self._base_result_mask, copy=True)
            # Do NOT clear catalog here if we are just reverting to detection order
            # The catalog from the usecase IS in detection order
        else:
            if "mapping" not in order_cache:
                order_cache["mapping"] = self._order_mapping_for_key(
                    self._base_result_mask, order_key, data
                )
            mapping = order_cache["mapping"]
            if mapping is None:
                ordered_mask = np.array(self._base_result_mask, copy=True)
            else:
                ordered_mask = mapping[self._base_result_mask]
            # If we reordered, the original catalog (which is detection ordered) is mismatched ID-wise
            self.catalog = []
        self.result_mask = self._apply_quality_filter_to_mask(
            ordered_mask, flags_cache=order_cache
        )
        if order_key == "detection" and not self._edge_excluded_labels:
            source_catalog = self._base_catalog if self._base_catalog else self.catalog
            self.catalog = self._catalog_with_quality_flags(source_catalog)
        else:
            self.catalog = []
        self._update_result_count_label()
        self._label_color_map = {}
        if refresh:
            self._clear_cloud_overlays()
            if self.mask_radio.isChecked():
                label_data = self._make_label_view_data()
                if label_data is not None:
                    self._apply_data_to_all_windows(label_data)
            else:
                self._schedule_contour_update()

    def _create_clumpfind_tab(self):
        """Create the Clumpfind algorithm parameter tab."""
        tab = QWidget()
        layout = QGridLayout(tab)

        layout.addWidget(QLabel("Min Threshold (sigma):"), 0, 0)
        self.cf_min_threshold = self._make_double_spin(3.0)
        self.cf_min_threshold.setToolTip("Minimum threshold in units of RMS noise")
        layout.addWidget(self.cf_min_threshold, 0, 1)

        layout.addWidget(QLabel("Step Size (sigma):"), 1, 0)
        self.cf_step = self._make_double_spin(2.0)
        self.cf_step.setToolTip("Contour level step size in units of RMS noise")
        layout.addWidget(self.cf_step, 1, 1)

        layout.addWidget(QLabel("Min Pixels:"), 2, 0)
        self.cf_min_pixels = self._make_int_spin(10)
        self.cf_min_pixels.setToolTip("Minimum number of pixels for a valid clump")
        layout.addWidget(self.cf_min_pixels, 2, 1)

        layout.setHorizontalSpacing(8)
        layout.setRowStretch(3, 1)
        return tab

    def _create_fellwalker_tab(self):
        """Create the FellWalker algorithm parameter tab."""
        tab = QWidget()
        layout = QGridLayout(tab)

        layout.addWidget(QLabel("Min Threshold (sigma):"), 0, 0)
        self.fw_min_threshold = self._make_double_spin(3.0)
        self.fw_min_threshold.setToolTip("Minimum threshold in units of RMS noise")
        layout.addWidget(self.fw_min_threshold, 0, 1)

        layout.addWidget(QLabel("Min Dip (sigma):"), 1, 0)
        self.fw_min_dip = self._make_double_spin(2.0)
        self.fw_min_dip.setToolTip("Prominence threshold: peaks must differ by this amount")
        layout.addWidget(self.fw_min_dip, 1, 1)

        layout.addWidget(QLabel("Min Pixels:"), 2, 0)
        self.fw_min_pixels = self._make_int_spin(10)
        self.fw_min_pixels.setToolTip("Minimum number of pixels for a valid clump")
        layout.addWidget(self.fw_min_pixels, 2, 1)

        layout.setHorizontalSpacing(8)
        layout.setRowStretch(3, 1)
        return tab

    def _create_dendro_tab(self):
        """Create the Dendrogram/SCIMES algorithm parameter tab."""
        tab = QWidget()
        layout = QGridLayout(tab)

        layout.addWidget(QLabel("Min Value (sigma):"), 0, 0)
        self.dend_min_value = self._make_double_spin(3.0)
        self.dend_min_value.setToolTip("Minimum intensity to consider (in sigma)")
        layout.addWidget(self.dend_min_value, 0, 1)

        layout.addWidget(QLabel("Min Delta (sigma):"), 1, 0)
        self.dend_min_delta = self._make_double_spin(2.0)
        self.dend_min_delta.setToolTip("Minimum height difference for structures")
        layout.addWidget(self.dend_min_delta, 1, 1)

        layout.addWidget(QLabel("Min Pixels:"), 2, 0)
        self.dend_min_npix = self._make_int_spin(10)
        self.dend_min_npix.setToolTip("Minimum number of pixels for a structure")
        layout.addWidget(self.dend_min_npix, 2, 1)
        
        # Output Mode Selection
        layout.addWidget(QLabel("Output Mode:"), 3, 0)
        self.output_mode_combo = QComboBox()
        self.output_mode_combo.addItems(["Leaves Only", "Roots Only", "All (Flattened)"])
        self.output_mode_combo.setToolTip("Select which structures to visualize.\n'All' flattens the hierarchy (Parent shells around children).")
        layout.addWidget(self.output_mode_combo, 3, 1)

        # SCIMES options
        self.use_scimes = QCheckBox("Use SCIMES clustering")
        self.use_scimes.setToolTip(
            "Apply SCIMES spectral clustering to group dendrogram leaves. "
            "Availability is checked when enabled."
        )

        layout.addWidget(self.use_scimes, 4, 0, 1, 2)

        self.scimes_isol = QCheckBox("Include isolated leaves")
        self.scimes_isol.setChecked(True)
        self.scimes_isol.setEnabled(False) # Default disabled until SCIMES checked
        
        # Keep the SCIMES dependency check out of panel construction; it can
        # import heavy optional packages on first use.
        self.use_scimes.toggled.connect(self._on_scimes_toggled)
        layout.addWidget(self.scimes_isol, 5, 0, 1, 2)

        # SCIMES criteria (3D use-case) - includes User K
        crit_group = QGroupBox("SCIMES Options")
        crit_layout = QGridLayout(crit_group)
        
        # Criteria checkboxes
        crit_layout.addWidget(QLabel("Similarity Criteria:"), 0, 0)
        self.scimes_lum_check = QCheckBox("Luminosity")
        self.scimes_vol_check = QCheckBox("Volume")
        self.scimes_lum_check.setChecked(True)
        self.scimes_vol_check.setChecked(True)
        crit_layout.addWidget(self.scimes_lum_check, 0, 1)
        crit_layout.addWidget(self.scimes_vol_check, 0, 2)
        crit_layout.setColumnStretch(2, 1)
        
        # User K (Target Clusters) - inside the group
        self.user_k_label = QLabel("Target Clusters (user_k):")
        self.user_k_input = QSpinBox()
        self.user_k_input.setRange(0, 1_000_000)
        self.user_k_input.setSpecialValueText("Auto")
        self.user_k_input.setValue(0)
        self.user_k_input.setToolTip("Approximate target number of clusters. Use Auto for detection.")
        self.user_k_input.setFixedWidth(self._input_width)
        self.user_k_input.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.user_k_input.installEventFilter(self)
        crit_layout.addWidget(self.user_k_label, 1, 0)
        crit_layout.addWidget(self.user_k_input, 1, 1, alignment=Qt.AlignmentFlag.AlignLeft)

        self.scimes_options_group = crit_group
        self.scimes_options_group.setEnabled(False)
        layout.addWidget(self.scimes_options_group, 6, 0, 1, 2)

        layout.setHorizontalSpacing(8)
        layout.setRowStretch(7, 1)
        return tab

    def _ensure_scimes_available(self, show_message=False):
        if not self._scimes_availability_checked:
            self._scimes_availability_checked = True
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                available, error = check_scimes_availability()
            finally:
                QApplication.restoreOverrideCursor()
            self._scimes_available = bool(available)
            self._scimes_error = str(error or "")

        if self._scimes_available:
            self.use_scimes.setToolTip("Apply SCIMES spectral clustering to group dendrogram leaves")
            return True

        message = "SCIMES module is not installed or available."
        if self._scimes_error:
            message = f"{message}\nError: {self._scimes_error}"
        self.use_scimes.setToolTip(message)
        if show_message:
            QMessageBox.warning(self, "SCIMES Unavailable", message)
        return False

    def _apply_scimes_ui_state(self, checked):
        checked = bool(checked)
        self.scimes_isol.setEnabled(checked)
        self.output_mode_combo.setDisabled(checked)
        self.scimes_options_group.setEnabled(checked)
        self._update_projected_availability()

    def _on_scimes_toggled(self, checked):
        if checked and not self._ensure_scimes_available(show_message=True):
            self.use_scimes.blockSignals(True)
            self.use_scimes.setChecked(False)
            self.use_scimes.blockSignals(False)
            checked = False
        self._apply_scimes_ui_state(checked)

    def move_to_default_position(self):
        """Position the panel relative to the main viewer."""
        try:
            if hasattr(self.fits_viewer, 'control_panel') and self.fits_viewer.control_panel:
                cp_geom = self.fits_viewer.control_panel.geometry()
                self.move(cp_geom.x() + cp_geom.width(), cp_geom.y())
            else:
                mainwindow_geometry = self.fits_viewer.geometry()
                self.move(mainwindow_geometry.x() + mainwindow_geometry.width() + 10,
                          mainwindow_geometry.y())
        except Exception:
            pass

    def get_rms(self):
        """Get RMS noise value, either from input or auto-estimate."""
        rms_text = self.rms_input.text().strip()
        if rms_text:
            try:
                rms = float(rms_text)
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", "RMS noise must be a number.")
                return None
            if not np.isfinite(rms) or rms <= 0:
                QMessageBox.warning(self, "Invalid Input", "RMS noise must be a positive number.")
                return None
            return rms

        QMessageBox.warning(self, "Invalid Input", "RMS noise is required.")
        return None

    def get_app_state(self):
        """Helper to get AppState from main window."""
        # Try to find app_state on fits_viewer (bridged) or its main_window
        if hasattr(self.fits_viewer, 'app_state'):
             return self.fits_viewer.app_state
        if hasattr(self.fits_viewer, 'main_window') and hasattr(self.fits_viewer.main_window, 'app_state'):
            return self.fits_viewer.main_window.app_state
        return None

    def get_analysis_data(self):
        """Return 2D image or 3D cube for analysis."""
        if self.data.ndim == 2:
            return self.data if is_lazy_scaled(self.data) else self.data.copy()

        cube = self.cube if self.cube is not None else self.data
        # A scaled memmap cannot satisfy NumPy 2's ``copy=False`` contract.
        # Keep it lazy here; the worker usecase performs the memory preflight
        # and materializes it off the GUI thread.
        return cube if is_lazy_scaled(cube) else np.asarray(cube)

    def run_identification(self):
        """Start the selected clump-finding algorithm on a worker thread."""
        if self._clump_thread is not None and self._clump_thread.isRunning():
            return  # already running; ignore a re-entrant Run click

        rms = self.get_rms()
        if rms is None:
            self.status_label.setText("Status: Error - invalid RMS")
            return

        # Snapshot the analysis data (2D image or 3D cube) for display/relabel.
        self.analysis_data = self.get_analysis_data()
        self._invalidate_order_caches()
        if self.analysis_data is None:
            self.status_label.setText("Status: Error - no data")
            return

        state = self.get_app_state()
        if state is None:
            QMessageBox.critical(self, "Error", "AppState not available")
            self.status_label.setText("Status: Error - no AppState")
            return

        # Validate + collect parameters on the UI thread (may pop dialogs).
        job = self._collect_clump_job(rms)
        if job is None:
            return

        self._start_clump_job(state, job)

    def _start_clump_job(self, state, job):
        """Spin up the worker thread for a collected clump-finding job."""
        self._clump_job = job
        self._clump_cancel = CancellationToken()
        self._clump_thread = QThread(self)
        self._clump_worker = ClumpWorker(
            state,
            job["algorithm"],
            job["params"],
            cancel_token=self._clump_cancel,
        )
        self._clump_worker.moveToThread(self._clump_thread)

        self._clump_worker.progress.connect(self._on_clump_progress)
        self._clump_worker.status.connect(self._on_clump_status)
        self._clump_worker.finished.connect(self._on_clump_finished)
        self._clump_worker.error.connect(self._on_clump_error)
        self._clump_worker.cancelled.connect(self._on_clump_cancelled)

        self._clump_thread.started.connect(self._clump_worker.run)
        self._clump_worker.finished.connect(self._clump_thread.quit)
        self._clump_worker.error.connect(self._clump_thread.quit)
        self._clump_worker.cancelled.connect(self._clump_thread.quit)
        self._clump_thread.finished.connect(self._cleanup_clump_thread)

        self._set_running_ui(True)
        self._clump_thread.start()

    # ------------------------------------------------------------------
    # Worker signal handlers
    def _on_clump_progress(self, value):
        if not self.progress_bar.isVisible():
            self.progress_bar.setVisible(True)
        if value < 0:
            # Indeterminate/busy phase (e.g. astrodendro compute).
            self.progress_bar.setRange(0, 0)
        else:
            if self.progress_bar.maximum() == 0:
                self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(int(value))

    def _on_clump_status(self, text):
        self.status_label.setText(f"Status: {text}")
        self.status_label.setToolTip("")

    @staticmethod
    def _dialog_error_text(message: str) -> str:
        return textwrap.fill(str(message), width=96, break_long_words=True)

    def _on_clump_finished(self, result):
        job = self._clump_job or {}
        self.result_mask = result.mask
        self._base_catalog = list(result.catalog or [])
        self.catalog = list(self._base_catalog)
        self._last_clump_result = result
        self._last_run_metadata = {
            "Algorithm": job.get("algorithm_label", "Clump"),
            "Parameters": result.parameters,
        }
        action_name = job.get("action_name")
        if action_name:
            self._record_run_action(action_name, job.get("action_payload", {}))

        self._set_running_ui(False)
        self._apply_clump_result()

    def _on_clump_error(self, message):
        self._set_running_ui(False)
        self.status_label.setText("Status: Error - see details")
        self.status_label.setToolTip(str(message))
        QMessageBox.critical(
            self,
            "Error",
            f"Algorithm failed:\n{self._dialog_error_text(message)}",
        )

    def _on_clump_cancelled(self):
        self._set_running_ui(False)
        self.status_label.setText("Status: Cancelled")
        self.count_label.setText("Detected: -- clumps")

    def _cancel_clump_job(self):
        """Cancel the running job and return control to the user immediately.

        Cooperative algorithms (Clumpfind/FellWalker) stop at their next
        checkpoint; an uninterruptible one (astrodendro) keeps running detached
        in the background with its result discarded.  Either way the panel is
        usable again at once.
        """
        if self._clump_thread is None:
            return
        if self._clump_cancel is not None:
            self._clump_cancel.cancel()
        self._detach_running_job()
        self._set_running_ui(False)
        self.status_label.setText("Status: Cancelled")
        self.count_label.setText("Detected: -- clumps")

    def _detach_running_job(self):
        """Orphan the running worker/thread so this panel can move on now.

        The caller has already flipped the cancel token.  We sever the worker's
        links to this panel (so late signals never touch a closed panel), keep
        the thread alive in a module-level registry (so Qt does not delete a
        live QThread), and let it self-clean once it finishes.
        """
        thread = self._clump_thread
        worker = self._clump_worker
        if thread is None or worker is None:
            return

        for signal, slot in (
            (worker.progress, self._on_clump_progress),
            (worker.status, self._on_clump_status),
            (worker.finished, self._on_clump_finished),
            (worker.error, self._on_clump_error),
            (worker.cancelled, self._on_clump_cancelled),
        ):
            try:
                signal.disconnect(slot)
            except (TypeError, RuntimeError):
                pass
        try:
            thread.finished.disconnect(self._cleanup_clump_thread)
        except (TypeError, RuntimeError):
            pass

        # The worker stays wired to thread.quit, so the thread exits when the
        # worker returns.  Detach from the panel's object tree to survive close.
        thread.setParent(None)
        entry = (thread, worker)
        _DETACHED_CLUMP_JOBS.add(entry)

        def _finalize():
            if entry not in _DETACHED_CLUMP_JOBS:
                return
            _DETACHED_CLUMP_JOBS.discard(entry)
            worker.deleteLater()
            thread.deleteLater()

        thread.finished.connect(_finalize)
        if thread.isFinished():
            _finalize()

        self._clump_thread = None
        self._clump_worker = None
        self._clump_cancel = None
        self._clump_job = None

    def _cleanup_clump_thread(self):
        worker = self._clump_worker
        thread = self._clump_thread
        self._clump_worker = None
        self._clump_thread = None
        self._clump_cancel = None
        if worker is not None:
            worker.deleteLater()
        if thread is not None:
            thread.deleteLater()

    # ------------------------------------------------------------------
    # Running-state UI
    def _set_running_ui(self, running):
        self.run_button.setEnabled(not running)
        self.clear_button.setEnabled(not running)
        # Disable the tab *bar* (not individual tabs): disabling the currently
        # selected tab makes QTabWidget jump to the next enabled tab, which left
        # the selection on FellWalker after a Dendrogram run.  Disabling the bar
        # blocks switching while preserving the current page.
        self.tabs.tabBar().setEnabled(not running)
        if running:
            self.export_catalog_button.setEnabled(False)
            self.export_mask_button.setEnabled(False)
            self.status_label.setText("Status: Running...")
            self.progress_bar.setRange(0, 0)  # busy until first concrete update
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(True)
        else:
            self.progress_bar.setVisible(False)
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)

    def _apply_clump_result(self):
        """Update displays from self.result_mask after a successful run."""
        try:
            if self.result_mask is None:
                return
            # max() is a single pass; the exact count comes from the edge flags
            # via _update_result_count_label inside _apply_id_order below.
            has_clumps = self.result_mask.size > 0 and int(np.max(self.result_mask)) > 0
            if has_clumps:
                self.status_label.setText("Status: Complete")
                self.export_catalog_button.setEnabled(True)
                self.export_mask_button.setEnabled(True)
                self._base_result_mask = np.array(self.result_mask, copy=True)
                self._invalidate_order_caches()
                self._configure_id_order_options()
                self._apply_id_order(refresh=False)
                self._update_projected_availability()
                if self.contour_radio.isChecked():
                    self.display_as_contours()
                else:
                    self.display_as_label_cube()
            else:
                self.count_label.setText("Detected: 0 clumps")
                self.status_label.setText("Status: No clumps detected. Try adjusting parameters.")
                self._base_result_mask = None
                self._invalidate_order_caches()
                self.catalog = []
                self._base_catalog = []
                self._edge_label_flags = {}
                self._edge_excluded_labels = set()
                self.id_order_combo.setEnabled(False)
        except Exception as e:
            self.status_label.setText(f"Status: Error - {str(e)}")
            QMessageBox.critical(self, "Error", f"Display failed:\n{str(e)}")

    def _collect_clump_job(self, rms):
        """Validate inputs and build a worker job for the active algorithm tab.

        Runs on the UI thread (may show dialogs). Returns a job dict consumed by
        :meth:`_start_clump_job`, or ``None`` when validation fails / is aborted.
        """
        current_tab = self.tabs.currentIndex()

        if current_tab == 1:
            # Clumpfind
            params = {
                "rms": float(rms),
                "min_threshold_sigma": float(self.cf_min_threshold.value()),
                "step_sigma": float(self.cf_step.value()),
                "min_pixels": int(self.cf_min_pixels.value()),
            }
            return {
                "algorithm": "clumpfind",
                "algorithm_label": "ClumpFind",
                "action_name": "run_clumpfind",
                "action_payload": dict(params),
                "params": params,
            }

        if current_tab == 2:
            # FellWalker
            params = {
                "rms": float(rms),
                "min_threshold_sigma": float(self.fw_min_threshold.value()),
                "min_dip_sigma": float(self.fw_min_dip.value()),
                "min_pixels": int(self.fw_min_pixels.value()),
            }
            return {
                "algorithm": "fellwalker",
                "algorithm_label": "FellWalker",
                "action_name": "run_fellwalker",
                "action_payload": dict(params),
                "params": params,
            }

        # current_tab == 0 -> Dendrogram / SCIMES
        min_value_sigma = float(self.dend_min_value.value())
        min_delta_sigma = float(self.dend_min_delta.value())
        min_npix = int(self.dend_min_npix.value())

        mode_text = self.output_mode_combo.currentText()
        output_mode = 'leaves'
        if 'Roots' in mode_text:
            output_mode = 'roots'
        elif 'All' in mode_text:
            output_mode = 'all'

        use_scimes = self.use_scimes.isChecked()
        scimes_criteria = []
        scimes_user_k = 0

        if use_scimes:
            if not self._ensure_scimes_available(show_message=True):
                # _ensure_scimes_available already surfaced the reason.
                return None
            output_mode = 'clusters'
            if self.scimes_lum_check.isChecked():
                scimes_criteria.append("luminosity")
            if self.scimes_vol_check.isChecked():
                is_3d = self.analysis_data is not None and self.analysis_data.ndim >= 3
                scimes_criteria.append("volume" if is_3d else "area_exact")
            if not scimes_criteria:
                QMessageBox.warning(self, "SCIMES", "Select at least one criterion.")
                use_scimes = False
                output_mode = 'leaves'
            scimes_user_k = int(self.user_k_input.value())

        # The usecase layer re-runs every time; drop any stale cached handler.
        self.dendro_handler = None

        params = {
            "rms": float(rms),
            "min_value_sigma": min_value_sigma,
            "min_delta_sigma": min_delta_sigma,
            "min_npix": min_npix,
            "output_mode": output_mode,
            "use_scimes": bool(use_scimes),
            "scimes_criteria": list(scimes_criteria),
            "scimes_user_k": int(scimes_user_k),
            "scimes_save_isol": bool(self.scimes_isol.isChecked()),
        }
        payload = {
            "rms": float(rms),
            "min_value_sigma": min_value_sigma,
            "min_delta_sigma": min_delta_sigma,
            "min_npix": min_npix,
            "output_mode": output_mode,
            "use_scimes": bool(use_scimes),
        }
        if use_scimes:
            payload["scimes_criteria"] = list(scimes_criteria)
            payload["scimes_user_k"] = int(scimes_user_k)
            payload["scimes_save_isol"] = bool(self.scimes_isol.isChecked())

        return {
            "algorithm": "dendrogram",
            "algorithm_label": "Dendrogram" + (" (SCIMES)" if use_scimes else ""),
            "action_name": "run_dendrogram",
            "action_payload": payload,
            "params": params,
        }

    def _record_run_action(self, action_name: str, payload: dict) -> None:
        record_action_preview(
            self.fits_viewer,
            action_name,
            payload,
            replace_tag=self._action_record_tag,
        )

    # _generate_catalog removed - using usecase results


    def display_as_contours(self):
        """Display results as contours on XY/XZ/ZY planes."""
        self.restore_original_data_view()
        self._clear_cloud_overlays()
        self._update_cloud_contours_for_all_planes()

    def clear_display(self):
        """Clear contours and mask overlay from the viewer."""
        self._clear_cloud_overlays()
        self.restore_original_data_view()

    def clear_results(self):
        """Clear all results and overlays."""
        self.clear_display()
        self.result_mask = None
        self._base_result_mask = None
        self._invalidate_order_caches()
        self.catalog = []
        self._base_catalog = []
        self._last_clump_result = None
        self._label_color_map = {}
        self._edge_label_flags = {}
        self._edge_excluded_labels = set()
        self.count_label.setText("Detected: -- clumps")
        self.status_label.setText("Status: Ready")
        self.export_catalog_button.setEnabled(False)
        self.export_mask_button.setEnabled(False)
        self.projection_mode_checkbox.setChecked(False)
        self._update_projected_availability()
        self._configure_id_order_options()
        clear_action_preview_record(self.fits_viewer, self._action_record_tag)

    # ------------------------------------------------------------------
    # Display helpers (label cube / contours)
    def _on_display_mode_changed(self):
        if self.result_mask is None:
            return
        if self.contour_radio.isChecked():
            self.display_as_contours()
        elif self.mask_radio.isChecked():
            self.display_as_label_cube()

    def _get_projected_mask_for_plane(self, mask3d: np.ndarray, plane: str) -> np.ndarray:
        """Project all labels onto the 2D plane using max projection."""
        if plane == 'xy':
            # Project along Z axis (axis 0)
            return np.max(mask3d, axis=0)
        elif plane == 'xz':
            # Project along Y axis (axis 1)
            return np.max(mask3d, axis=1)
        elif plane == 'zy':
            # Project along X axis (axis 2), then transpose
            return np.max(mask3d, axis=2).T
        return np.max(mask3d, axis=0)

    def _apply_data_to_all_windows(self, new_data):
        self.fits_viewer.data = new_data
        self.fits_viewer.update_cube()
        for window in self.subwindows:
            if window:
                window.data = new_data
                window.update_cube()
        self._update_all_displays()

    def _apply_label_data_by_plane(self, plane_data):
        for window in self._plane_windows():
            if not window:
                continue
            plane = getattr(window, "plane", None) or "xy"
            data = plane_data.get(plane)
            if data is None:
                continue
            window.data = data
            window.update_cube()
        self._update_all_displays()

    def _update_all_displays(self):
        all_windows = [self.fits_viewer] + list(self.subwindows or [])
        for window in all_windows:
            if not window:
                continue
            current_channel = window.current_channel_index()
            data_slice = None
            if window.data.ndim == 4:
                if window.plane == 'xy':
                    data_slice = window.data[0, current_channel, :, :]
                elif window.plane == 'xz':
                    data_slice = window.data[0, :, current_channel, :]
                elif window.plane == 'zy':
                    data_slice = window.data[0, :, :, current_channel].T
            elif window.data.ndim == 3:
                if window.plane == 'xy':
                    data_slice = window.data[current_channel, :, :]
                elif window.plane == 'xz':
                    data_slice = window.data[:, current_channel, :]
                elif window.plane == 'zy':
                    data_slice = window.data[:, :, current_channel].T
            elif window.data.ndim == 2:
                data_slice = window.data
            if data_slice is None:
                continue
            try:
                window.im.set_data(data_slice)
                window.canvas.draw_idle()
            except Exception:
                pass

    def _make_label_view_data(self):
        if self.result_mask is None:
            return None
        base = self.fits_viewer.data
        mask = np.asanyarray(self.result_mask)
        output_shape = base.shape if base.ndim == 4 and mask.ndim == 3 else mask.shape
        output_count = int(np.prod(output_shape, dtype=np.int64))
        mask_count = int(mask.size)
        ensure_operation_memory_budget(
            output_count * np.dtype(np.float32).itemsize
            + mask_count * np.dtype(np.bool_).itemsize,
            operation_name="Clump label display",
            guidance=(
                "Close unused result windows or make a smaller cutout before "
                "displaying the label cube."
            ),
        )

        out = np.full(output_shape, np.nan, dtype=np.float32)
        target = out[0] if base.ndim == 4 and mask.ndim == 3 else out
        nonzero = mask != 0
        np.copyto(target, mask, where=nonzero, casting="unsafe")
        return out

    def display_as_label_cube(self):
        if not self._label_mode_shows_contours():
            self._clear_cloud_overlays()
        if self.result_mask is None:
            return
        if self._baseline_data is None:
            self._baseline_data = self.fits_viewer.data
        label_data = self._make_label_view_data()
        if label_data is None:
            return
        self._apply_data_to_all_windows(label_data)
        try:
            self.fits_viewer.setWindowTitle(f"[CLOUD LABELS] {self.fits_viewer.original_window_title}")
        except Exception:
            pass
        self._label_view_active = True
        if self._label_mode_shows_contours():
            self._update_cloud_contours_for_all_planes()

    def restore_original_data_view(self):
        if not self._label_view_active:
            return
        if self._baseline_data is None:
            return
        self._apply_data_to_all_windows(self._baseline_data)
        try:
            self.fits_viewer.setWindowTitle(self.fits_viewer.original_window_title)
        except Exception:
            pass
        self._label_view_active = False

    # ------------------------------------------------------------------
    # Contour overlays (XY/XZ/ZY)
    def _plane_windows(self):
        windows = [self.fits_viewer]
        for sub in self.subwindows or []:
            if sub:
                windows.append(sub)
        return windows

    @staticmethod
    def _hex_to_rgba(color_hex: str, alpha: float = 1.0):
        r = int(color_hex[1:3], 16) / 255.0
        g = int(color_hex[3:5], 16) / 255.0
        b = int(color_hex[5:7], 16) / 255.0
        return (r, g, b, float(alpha))

    def _mask_slice_for_plane(self, mask3d: np.ndarray, plane: str, channel_index: int) -> np.ndarray:
        if plane == 'xy':
            return mask3d[channel_index, :, :]
        if plane == 'xz':
            return mask3d[:, channel_index, :]
        if plane == 'zy':
            return mask3d[:, :, channel_index].T
        return mask3d[channel_index, :, :]

    def _generate_random_color(self):
        """Generate a random vibrant color."""
        # HSV to RGB conversion simplified or just random bright RGB
        # Generating in RGB directly for simplicity, bias towards bright colors
        r = random.randint(50, 255)
        g = random.randint(50, 255)
        b = random.randint(50, 255)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _get_color_for_label(self, label_int):
        """Get or generate color for a label."""
        if label_int not in self._label_color_map:
            # Use preset palette for first N labels, then random
            if label_int <= len(self.CLUMP_COLORS):
                color = self.CLUMP_COLORS[label_int - 1]
            else:
                color = self._generate_random_color()
            self._label_color_map[label_int] = color
        return self._label_color_map[label_int]

    def _build_overlay_state(self, viewer, slice2d: np.ndarray, overlay_id: str) -> ContourState:
        try:
            from skimage.measure import find_contours
        except Exception:
            return None
        
        # Optimization: Use find_objects to get bounding box for each label
        # slice2d must be int type for find_objects
        slice_int = slice2d.astype(int)
        
        # find_objects returns a list of slices. Index i corresponds to label i+1.
        # labels in slice might not be contiguous, so we safely iterate
        obj_slices = find_objects(slice_int)
        
        # Only process labels present in this slice (np.unique is decently fast)
        labels_in_slice = np.unique(slice_int)
        labels_in_slice = labels_in_slice[labels_in_slice > 0]
        
        if labels_in_slice.size == 0:
            return None

        segments = []
        for label in labels_in_slice.tolist():
            label_int = int(label)
            
            # Get bounding box slice for this label
            # find_objects indices are 0-based for label 1.
            if label_int - 1 < len(obj_slices):
                sl = obj_slices[label_int - 1]
            else:
                sl = None
                
            if sl is None:
                continue
                
            # Extract sub-image (bounding box)
            # This dramatically speeds up find_contours for small clumps in large fields
            sub_img = slice2d[sl]
            
            # Create binary mask within bounding box
            binary = (sub_img == label_int).astype(np.uint8)
            
            # PAD the binary mask to ensure contours capture edges touching the bounding box
            # Otherwise find_contours treats edges as open or cuts them off.
            binary_padded = np.pad(binary, pad_width=1, mode='constant', constant_values=0)
            
            try:
                contours = find_contours(binary_padded, 0.5)
            except Exception:
                continue
                
            color_hex = self._get_color_for_label(label_int)
            rgba = self._hex_to_rgba(color_hex, alpha=0.95)
            
            # Offset for bounding box
            y_offset = sl[0].start
            x_offset = sl[1].start
            
            for poly in contours:
                if poly is None or len(poly) < 2:
                    continue
                coords = np.asarray(poly, dtype=float)
                # poly returns (row, col) -> (y, x).
                # We need to add offsets.
                # Subtract 1 because of the padding we added at the start of binary_padded
                coords[:, 0] += (y_offset - 1.0)
                coords[:, 1] += (x_offset - 1.0)
                
                # pixels expect (x, y) for display usually, but here...
                # Current code: pixels = np.column_stack([coords[:, 1], coords[:, 0]])
                # This implies coords[:, 1] is X, coords[:, 0] is Y.
                pixels = np.column_stack([coords[:, 1], coords[:, 0]])
                segments.append(ContourSegment(level=float(label_int), pixels=pixels, color=np.asarray(rgba, dtype=float)))

        if not segments:
            return None

        item_label = viewer._default_contour_label()
        params = ContourParameters(color="white", linewidth=1.5, smoothing=0.0)
        state = ContourState(
            layer_id=getattr(viewer, "_contour_layer_id", "") or "",
            plane=getattr(viewer, "plane", None),
            label="Clump Finding",
            parameters=params,
            levels=[0.5],
            items=[ContourItemState(item_label=item_label, segments=segments)],
            world_frame=None,
            overlay_id=overlay_id,
        )
        return state

    def _build_projected_overlay_state_all(self, viewer, mask3d: np.ndarray, plane: str, overlay_id: str, obj_slices=None) -> ContourState:
        try:
            from skimage.measure import find_contours
        except Exception:
            return None

        mask_int = np.asarray(mask3d)
        if mask_int.dtype.kind not in "iu":
            mask_int = mask_int.astype(np.int32)
        if obj_slices is None:
            obj_slices = find_objects(mask_int)
        if not obj_slices:
            return None

        segments = []
        for label_idx, sl in enumerate(obj_slices, start=1):
            if sl is None:
                continue
            label_sub = (mask_int[sl] == label_idx)
            if not np.any(label_sub):
                continue

            if plane == 'xy':
                proj = np.max(label_sub, axis=0)
                y_offset = sl[1].start
                x_offset = sl[2].start
            elif plane == 'xz':
                proj = np.max(label_sub, axis=1)
                y_offset = sl[0].start
                x_offset = sl[2].start
            elif plane == 'zy':
                proj = np.max(label_sub, axis=2).T
                y_offset = sl[1].start
                x_offset = sl[0].start
            else:
                proj = np.max(label_sub, axis=0)
                y_offset = sl[1].start
                x_offset = sl[2].start

            if not np.any(proj):
                continue

            binary = proj.astype(np.uint8)
            binary_padded = np.pad(binary, pad_width=1, mode='constant', constant_values=0)
            try:
                contours = find_contours(binary_padded, 0.5)
            except Exception:
                continue

            color_hex = self._get_color_for_label(label_idx)
            rgba = self._hex_to_rgba(color_hex, alpha=0.95)

            for poly in contours:
                if poly is None or len(poly) < 2:
                    continue
                coords = np.asarray(poly, dtype=float)
                coords[:, 0] += (y_offset - 1.0)
                coords[:, 1] += (x_offset - 1.0)
                pixels = np.column_stack([coords[:, 1], coords[:, 0]])
                segments.append(ContourSegment(level=float(label_idx), pixels=pixels, color=np.asarray(rgba, dtype=float)))

        if not segments:
            return None

        item_label = viewer._default_contour_label()
        params = ContourParameters(color="white", linewidth=1.5, smoothing=0.0)
        state = ContourState(
            layer_id=getattr(viewer, "_contour_layer_id", "") or "",
            plane=getattr(viewer, "plane", None),
            label="Clump Finding",
            parameters=params,
            levels=[0.5],
            items=[ContourItemState(item_label=item_label, segments=segments)],
            world_frame=None,
            overlay_id=overlay_id,
        )
        return state

    def _clear_cloud_overlays(self):
        manager = ContourManager.instance()
        overlay_ids = list(self._cloud_overlay_ids.values())
        if overlay_ids:
            manager.clear_overlays(overlay_ids)
        self._cloud_overlay_ids = {}
        
        # Clear directly drawn contour artists
        for plane, artists in list(self._direct_contour_artists.items()):
            if artists:
                self._cleanup_expired_artists(artists)
        self._direct_contour_artists = {}

        self._last_update_params = {}

    def _schedule_contour_update(self):
        if self.result_mask is None:
            return
        if not self.contour_radio.isChecked() and not self._label_mode_shows_contours():
            return
        if self._label_view_active and not self._label_mode_shows_contours():
            return
        self._update_cloud_contours_for_all_planes()

    def _on_projection_mode_toggled(self):
        if self.result_mask is None:
            return
        if self.mask_radio.isChecked():
            self.display_as_label_cube()
            return
        self._schedule_contour_update()

    def _update_cloud_contours_for_all_planes(self):
        if self.result_mask is None:
            return
        if not self.contour_radio.isChecked() and not self._label_mode_shows_contours():
            return
        if self._label_view_active and not self._label_mode_shows_contours():
            return

        mask = self.result_mask
        if mask.ndim != 3:
            # 2D: draw only on the main XY plane.
            self._clear_cloud_overlays()
            viewer = self.fits_viewer
            layer_id = getattr(viewer, "_contour_layer_id", None)
            if not layer_id:
                try:
                    viewer._register_contour_layer()
                except Exception:
                    return
                layer_id = getattr(viewer, "_contour_layer_id", None)
            if not layer_id:
                return
            overlay_id = f"{layer_id}::overlay::cloudid"
            state = self._build_overlay_state(viewer, np.asarray(mask), overlay_id)
            if state is None:
                return
            manager = ContourManager.instance()
            manager.clear_overlays([overlay_id])
            manager.import_overlay_state(layer_id, state)
            self._cloud_overlay_ids[viewer.plane] = overlay_id
            return

        # 3D Case
        # DO NOT wipe everything immediately with self._clear_cloud_overlays()
        # or else we lose the point of checking the cache.
        
        manager = ContourManager.instance()
        is_projected = self.projection_mode_checkbox.isChecked()
        projection_mode = self._projected_mode()
        # The 3D label bounding boxes are identical for all three plane
        # viewers; compute them once for the projected-"all" builds below.
        projected_all_slices = None

        for viewer in self._plane_windows():
            plane = getattr(viewer, "plane", None)
            if plane not in ("xy", "xz", "zy"):
                continue

            try:
                ch = int(viewer.current_channel_index())
            except Exception:
                ch = 0

            # caching check
            # For projected mode, we ignore channel changes (ch is effectively irrelevant for the mask content)
            cache_key = (plane, is_projected, projection_mode, -1 if is_projected else ch)
            if self._last_update_params.get(plane) == cache_key:
                # Already showing the correct contours for this state
                continue

            layer_id = getattr(viewer, "_contour_layer_id", None)
            if not layer_id:
                try:
                    viewer._register_contour_layer()
                except Exception:
                    continue
                layer_id = getattr(viewer, "_contour_layer_id", None)
            if not layer_id:
                continue
            overlay_id = f"{layer_id}::overlay::cloudid"

            try:
                # Use projection mode or channel-following based on checkbox
                if is_projected:
                    # Revert to simple max projection using ContourManager
                    # Cleanup direct artists if any
                    if plane in self._direct_contour_artists and self._direct_contour_artists[plane]:
                         self._cleanup_expired_artists(self._direct_contour_artists[plane])
                         self._direct_contour_artists[plane] = []
                    if projection_mode == "all":
                        slice2d = None
                    else:
                        slice2d = self._get_projected_mask_for_plane(mask, plane)
                else:
                    # Normal slice mode: use ContourManager (stable)
                    # Clear direct artists if any
                    if plane in self._direct_contour_artists and self._direct_contour_artists[plane]:
                        self._cleanup_expired_artists(self._direct_contour_artists[plane])
                        self._direct_contour_artists[plane] = []

                    slice2d = self._mask_slice_for_plane(mask, plane, ch)
            except Exception:
                continue
            
            if is_projected and projection_mode == "all":
                if projected_all_slices is None:
                    projected_all_slices = find_objects(np.asarray(mask))
                state = self._build_projected_overlay_state_all(
                    viewer,
                    np.asarray(mask),
                    plane,
                    overlay_id,
                    obj_slices=projected_all_slices,
                )
            else:
                state = self._build_overlay_state(viewer, np.asarray(slice2d), overlay_id)
            
            # If we are updating this plane, we should clear the old overlay for THIS plane
            # if we have one tracked.
            old_overlay_id = self._cloud_overlay_ids.get(plane)
            if old_overlay_id:
                 manager.clear_overlays([old_overlay_id])
                 # Del from tracker, will re-add if we have new state
                 del self._cloud_overlay_ids[plane]

            if state is None:
                # Even if None, we track that we tried to update for this state
                self._last_update_params[plane] = cache_key
                continue

            manager.import_overlay_state(layer_id, state)
            self._cloud_overlay_ids[plane] = overlay_id
            self._last_update_params[plane] = cache_key


    def _attach_slice_listeners(self):
        for viewer in self._plane_windows():
            slider = getattr(viewer, "slider", None)
            if slider is None:
                continue
            try:
                slider.valueChanged.connect(self._schedule_contour_update)
            except Exception:
                pass

    def _cleanup_expired_artists(self, artists):
        """Cleanup old artists."""
        from takefits.core.contour_manager import contour_set_artists

        for cs in artists:
            for artist in contour_set_artists(cs):
                try:
                    artist.remove()
                except Exception:
                    pass
        # Trigger redraw to clear artifacts
        for viewer in self._plane_windows():
            if viewer.canvas:
                viewer.canvas.draw_idle()

    def export_catalog(self):
        """Export clump properties to CSV."""
        # Generate catalog on demand if not present (was deferred for speed)
        if not self.catalog and self.result_mask is not None:
             self.status_label.setText("Status: Generating catalog...")
             QApplication.processEvents()
             
             state = self.get_app_state()
             if state:
                 self.catalog = self._catalog_with_quality_flags(
                     generate_catalog(state, self.result_mask)
                 )
                 self.status_label.setText("Status: Catalog generated.")
             
        if not self.catalog:
            QMessageBox.information(self, "Export", "No catalog data to export.")
            return

        # Generate default filename based on mode
        # User requested to remove 'test_' from filename if present
        raw_base = os.path.basename(self.fits_viewer.filename)
        base_name = os.path.splitext(raw_base)[0].replace("test_", "")
        suffix = "_clumps"
        
        # Check mode for suffix
        current_tab = self.tabs.currentIndex()
        is_dendro = (current_tab == 0)
        filter_mask = None # Boolean mask for catalog rows
        
        if is_dendro and self.dendro_handler is not None:
            if self.use_scimes.isChecked():
                suffix = "_scimes_clusters"
            else:
                mode_text = self.output_mode_combo.currentText()
                if 'Roots' in mode_text:
                    suffix = "_dendro_roots"
                elif 'All' in mode_text:
                    suffix = "_dendro_full"
                else: # Leaves
                    suffix = "_dendro_leaves"

        default_path = os.path.join(os.getcwd(), f"{base_name}{suffix}.csv")

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Catalog", default_path,
            "CSV Files (*.csv);;All Files (*)"
        )

        if not path:
            return

        try:
            # Check if we are in Dendrogram mode (Tab 2) and if handler is available
            if is_dendro and self.dendro_handler is not None:
                # Use native Astrodendro catalog for research quality (full flux integration)
                cat = self.dendro_handler.get_native_catalog()
                if cat is not None:
                    # Filter catalog based on mode
                    if self.use_scimes.isChecked():
                        # Filter to include only structures in self.dendro_handler.clusters
                        # We map via _idx column
                        if '_idx' in cat.colnames:
                            # Create a set of indices for fast lookup
                            cluster_indices = set(s.idx for s in self.dendro_handler.clusters)
                            # Create boolean mask
                            filter_mask = [row['_idx'] in cluster_indices for row in cat]
                            cat = cat[filter_mask]
                    else:
                        mode_text = self.output_mode_combo.currentText()
                        if 'structure_type' in cat.colnames:
                            # Use structure_type logic
                            # 'trunk' tag: is_trunk=True (no parent) and not leaf (so trunk)
                            # 'isolated' tag: is_leaf=True and is_trunk=True (no parent)
                            # 'leaf' tag: is_leaf=True and has parent
                            # 'branch' tag: not leaf and has parent
                            
                            if 'Roots' in mode_text:
                                # Roots Only should show Trunks and Isolated (roots that are also leaves)
                                mask = (cat['structure_type'] == 'trunk') | (cat['structure_type'] == 'isolated')
                                cat = cat[mask]
                            elif 'Leaves' in mode_text:
                                # Leaves Only should show Leaves and Isolated
                                mask = (cat['structure_type'] == 'leaf') | (cat['structure_type'] == 'isolated')
                                cat = cat[mask]
                            # 'All' means no filter
                        else:
                            # Fallback if column missing (shouldn't happen with new logic)
                            pass

                    cat = self._native_catalog_with_quality_flags(cat)
                    # Astropy Table write
                    # format='ascii.csv' handles header and delimiters
                    cat.write(path, format='ascii.csv', overwrite=True)
                    QMessageBox.information(self, "Export", 
                        f"Native Astrodendro Catalog (Research Quality) exported to:\n{path}\n\n"
                        f"Mode: {suffix[1:]}\n"
                        f"Rows: {len(cat)}")
                    return
            
            # Fallback / Classical Methods (ClumpFind/FellWalker)
            # Get all keys from the first catalog entry
            if self.catalog:
                fieldnames = list(self.catalog[0].keys())

                with open(path, 'w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(self.catalog)
                
                QMessageBox.information(self, "Export", f"Catalog exported to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export catalog:\n{e}")

    def export_mask(self):
        """Export mask as FITS file."""
        if self.result_mask is None:
            QMessageBox.information(self, "Export", "No mask data to export.")
            return

        # User requested to remove 'test_' from filename if present
        raw_base = os.path.basename(self.fits_viewer.filename)
        base_name = os.path.splitext(raw_base)[0].replace("test_", "")
        default_path = os.path.join(os.getcwd(), f"{base_name}_clump_mask.fits")

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Mask", default_path,
            "FITS Files (*.fits);;All Files (*)"
        )

        if not path:
            return

        try:
            state = self.get_app_state()
            if state is None:
                raise ValueError("AppState not available")

            previous_result = self._last_clump_result
            parameters = dict(
                previous_result.parameters
                if previous_result is not None
                else self._last_run_metadata.get("Parameters") or {}
            )
            parameters.update({
                "spatial_edge_margin_pix": int(self._edge_margin_value()),
                "exclude_spatial_edge_labels": bool(self.edge_exclude_checkbox.isChecked()),
                "excluded_spatial_edge_labels": int(len(self._edge_excluded_labels)),
            })
            algorithm_label = str(self._last_run_metadata.get("Algorithm") or "unknown")
            algorithm_map = {
                "ClumpFind": "clumpfind",
                "FellWalker": "fellwalker",
                "Dendrogram": "dendrogram",
                "SCIMES": "scimes",
            }
            algorithm = (
                previous_result.algorithm
                if previous_result is not None
                else algorithm_map.get(algorithm_label, algorithm_label.lower())
            )
            mask = np.asarray(self.result_mask, dtype=np.int32)
            labels = np.unique(mask)
            n_clumps = int(np.count_nonzero(labels > 0))
            result = ClumpResult(
                mask=mask,
                n_clumps=n_clumps,
                catalog=list(self.catalog or []),
                algorithm=algorithm,
                parameters=parameters,
            )

            export_clump_mask(
                state,
                result,
                path,
                history_entries=build_processing_history_lines(self.fits_viewer),
            )
            QMessageBox.information(self, "Export", f"Mask exported to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export mask:\n{e}")

    @staticmethod
    def _serialize_workspace_array_payload(array):
        if array is None:
            return None
        projected_bytes = estimate_materialized_nbytes(array)
        shape = tuple(int(dim) for dim in getattr(array, "shape", ()) or ())
        dtype = str(getattr(array, "dtype", "unknown"))
        if (
            projected_bytes is not None
            and int(projected_bytes) > _WORKSPACE_MASK_MAX_RAW_BYTES
        ):
            return {
                "encoding": "omitted-large-array",
                "reason": (
                    "Clump mask was omitted from workspace state because its "
                    "raw array exceeds the safe serialization limit."
                ),
                "shape": list(shape),
                "dtype": dtype,
                "nbytes": int(projected_bytes),
                "limit_bytes": int(_WORKSPACE_MASK_MAX_RAW_BYTES),
            }
        try:
            if projected_bytes is not None:
                # np.save + BytesIO.getvalue + zlib + base64 can briefly retain
                # roughly five raw-array equivalents for incompressible labels.
                ensure_operation_memory_budget(
                    int(projected_bytes) * 5,
                    operation_name="Clump workspace serialization",
                    guidance=(
                        "Save/export the clump mask as FITS instead of embedding "
                        "it in the workspace."
                    ),
                )
            arr = np.asarray(array)
            buffer = io.BytesIO()
            np.save(buffer, arr, allow_pickle=False)
            compressed = zlib.compress(buffer.getvalue(), level=3)
            return {
                "encoding": "npy+zlib+base64",
                "payload": base64.b64encode(compressed).decode("ascii"),
            }
        except MemoryError as exc:
            return {
                "encoding": "omitted-large-array",
                "reason": str(exc),
                "shape": list(shape),
                "dtype": dtype,
                "nbytes": (
                    None if projected_bytes is None else int(projected_bytes)
                ),
            }
        except Exception:
            return None

    @staticmethod
    def _deserialize_workspace_array_payload(payload):
        if not isinstance(payload, dict):
            return None
        if str(payload.get("encoding") or "").strip().lower() != "npy+zlib+base64":
            return None
        encoded = payload.get("payload")
        if not isinstance(encoded, str) or not encoded:
            return None
        try:
            compressed = base64.b64decode(encoded.encode("ascii"))
            raw = zlib.decompress(compressed)
            buffer = io.BytesIO(raw)
            return np.load(buffer, allow_pickle=False)
        except Exception:
            return None

    def export_workspace_state(self):
        state = {
            "schema": 1,
            "tab_index": int(self.tabs.currentIndex()),
            "rms_text": str(self.rms_input.text() or "").strip(),
            "clumpfind": {
                "min_threshold_sigma": float(self.cf_min_threshold.value()),
                "step_sigma": float(self.cf_step.value()),
                "min_pixels": int(self.cf_min_pixels.value()),
            },
            "fellwalker": {
                "min_threshold_sigma": float(self.fw_min_threshold.value()),
                "min_dip_sigma": float(self.fw_min_dip.value()),
                "min_pixels": int(self.fw_min_pixels.value()),
            },
            "dendrogram": {
                "min_value_sigma": float(self.dend_min_value.value()),
                "min_delta_sigma": float(self.dend_min_delta.value()),
                "min_npix": int(self.dend_min_npix.value()),
                "output_mode_index": int(self.output_mode_combo.currentIndex()),
                "use_scimes": bool(self.use_scimes.isChecked()),
                "include_isolated": bool(self.scimes_isol.isChecked()),
                "criterion_luminosity": bool(self.scimes_lum_check.isChecked()),
                "criterion_volume": bool(self.scimes_vol_check.isChecked()),
                "scimes_user_k": int(self.user_k_input.value()),
            },
            "display": {
                "mode": "mask" if self.mask_radio.isChecked() else "contour",
                "projected": bool(self.projection_mode_checkbox.isChecked()),
                "projected_mode": str(getattr(self, "_projected_mode_setting", "all")),
                "id_order_key": self.id_order_combo.currentData(),
                "label_view_active": bool(self._label_view_active),
            },
            "quality_flags": {
                "spatial_edge_margin_pix": int(self._edge_margin_value()),
                "exclude_spatial_edge_labels": bool(self.edge_exclude_checkbox.isChecked()),
            },
            "last_run_metadata": dict(self._last_run_metadata or {}),
        }

        source_mask = self._base_result_mask if self._base_result_mask is not None else self.result_mask
        mask_payload = self._serialize_workspace_array_payload(source_mask)
        if (
            isinstance(mask_payload, dict)
            and mask_payload.get("encoding") == "npy+zlib+base64"
        ):
            state["result_mask"] = mask_payload
            unique = np.unique(source_mask)
            state["detected_count"] = int(np.count_nonzero(unique > 0))
        elif isinstance(mask_payload, dict):
            state["result_mask_omitted"] = mask_payload
        return state

    def restore_workspace_state(self, state):
        if not isinstance(state, dict):
            return False

        self.data = self.fits_viewer.data
        self.header = self.fits_viewer.header
        self.wcs = self.fits_viewer.wcs
        self.cube = getattr(self.fits_viewer, "cube", None)

        def _set_spin_value(widget, value, *, integer=False):
            try:
                if value is None:
                    return
                if integer:
                    widget.setValue(int(value))
                else:
                    widget.setValue(float(value))
            except Exception:
                pass

        tab_index = state.get("tab_index")
        try:
            if tab_index is not None:
                index = int(tab_index)
                if 0 <= index < self.tabs.count():
                    self.tabs.setCurrentIndex(index)
        except Exception:
            pass

        rms_text = state.get("rms_text")
        if isinstance(rms_text, str):
            self.rms_input.setText(rms_text)

        clumpfind = state.get("clumpfind")
        if isinstance(clumpfind, dict):
            _set_spin_value(self.cf_min_threshold, clumpfind.get("min_threshold_sigma"))
            _set_spin_value(self.cf_step, clumpfind.get("step_sigma"))
            _set_spin_value(self.cf_min_pixels, clumpfind.get("min_pixels"), integer=True)

        fellwalker = state.get("fellwalker")
        if isinstance(fellwalker, dict):
            _set_spin_value(self.fw_min_threshold, fellwalker.get("min_threshold_sigma"))
            _set_spin_value(self.fw_min_dip, fellwalker.get("min_dip_sigma"))
            _set_spin_value(self.fw_min_pixels, fellwalker.get("min_pixels"), integer=True)

        dendrogram = state.get("dendrogram")
        if isinstance(dendrogram, dict):
            _set_spin_value(self.dend_min_value, dendrogram.get("min_value_sigma"))
            _set_spin_value(self.dend_min_delta, dendrogram.get("min_delta_sigma"))
            _set_spin_value(self.dend_min_npix, dendrogram.get("min_npix"), integer=True)
            try:
                output_mode = int(dendrogram.get("output_mode_index", 0))
                output_mode = max(0, min(self.output_mode_combo.count() - 1, output_mode))
                self.output_mode_combo.setCurrentIndex(output_mode)
            except Exception:
                pass
            try:
                use_scimes = bool(dendrogram.get("use_scimes", self.use_scimes.isChecked()))
                self.use_scimes.blockSignals(True)
                self.use_scimes.setChecked(use_scimes)
                self.use_scimes.blockSignals(False)
                self._apply_scimes_ui_state(use_scimes)
            except Exception:
                pass
            try:
                self.scimes_isol.setChecked(bool(dendrogram.get("include_isolated", self.scimes_isol.isChecked())))
            except Exception:
                pass
            try:
                self.scimes_lum_check.setChecked(bool(dendrogram.get("criterion_luminosity", self.scimes_lum_check.isChecked())))
            except Exception:
                pass
            try:
                self.scimes_vol_check.setChecked(bool(dendrogram.get("criterion_volume", self.scimes_vol_check.isChecked())))
            except Exception:
                pass
            _set_spin_value(self.user_k_input, dendrogram.get("scimes_user_k"), integer=True)

        quality_flags = state.get("quality_flags")
        if isinstance(quality_flags, dict):
            _set_spin_value(
                self.edge_margin_spin,
                quality_flags.get("spatial_edge_margin_pix"),
                integer=True,
            )
            try:
                self.edge_exclude_checkbox.setChecked(
                    bool(quality_flags.get(
                        "exclude_spatial_edge_labels",
                        self.edge_exclude_checkbox.isChecked(),
                    ))
                )
            except Exception:
                pass

        self._last_run_metadata = dict(state.get("last_run_metadata") or {})
        self._clear_cloud_overlays()
        self._baseline_data = None
        self._label_view_active = False
        self.result_mask = None
        self._base_result_mask = None
        self._invalidate_order_caches()
        self.catalog = []
        self._base_catalog = []
        self._label_color_map = {}
        self._edge_label_flags = {}
        self._edge_excluded_labels = set()

        restored_mask = self._deserialize_workspace_array_payload(state.get("result_mask"))
        if restored_mask is None:
            self.count_label.setText("Detected: -- clumps")
            self.status_label.setText("Status: Ready")
            self.export_catalog_button.setEnabled(False)
            self.export_mask_button.setEnabled(False)
            self._configure_id_order_options()
            self._update_projected_availability()
            return False

        self.result_mask = np.asarray(restored_mask)
        self._base_result_mask = np.array(self.result_mask, copy=True)
        self.analysis_data = self.get_analysis_data()
        self._invalidate_order_caches()
        self._configure_id_order_options()

        display = state.get("display")
        id_order_key = None
        display_mode = "contour"
        projected = False
        projected_mode = "all"
        if isinstance(display, dict):
            id_order_key = display.get("id_order_key")
            display_mode = str(display.get("mode") or "contour").strip().lower()
            projected = bool(display.get("projected", False))
            projected_mode = str(display.get("projected_mode") or "all").strip().lower()
        if projected_mode not in ("all", "max"):
            projected_mode = "all"
        self._projected_mode_setting = projected_mode

        if id_order_key is not None:
            for idx in range(self.id_order_combo.count()):
                if self.id_order_combo.itemData(idx) == id_order_key:
                    self.id_order_combo.setCurrentIndex(idx)
                    break
        # Restoring the quality widgets above may have queued a debounced
        # refilter; the explicit apply below supersedes it.
        self._quality_refilter_timer.stop()
        self._apply_id_order(refresh=False)

        self._update_result_count_label()
        self.status_label.setText("Status: Complete")
        self.export_catalog_button.setEnabled(True)
        self.export_mask_button.setEnabled(True)
        self._update_projected_availability()

        self.projection_mode_checkbox.blockSignals(True)
        self.projection_mode_checkbox.setChecked(projected)
        self.projection_mode_checkbox.blockSignals(False)
        self.contour_radio.blockSignals(True)
        self.mask_radio.blockSignals(True)
        self.contour_radio.setChecked(display_mode != "mask")
        self.mask_radio.setChecked(display_mode == "mask")
        self.contour_radio.blockSignals(False)
        self.mask_radio.blockSignals(False)

        if display_mode == "mask":
            self.display_as_label_cube()
        else:
            self.display_as_contours()
        return True

    def resync_after_workspace_restore(self):
        if self.result_mask is None:
            return
        if self.mask_radio.isChecked():
            self.display_as_label_cube()
            return
        self._schedule_contour_update()

    def closeEvent(self, event):
        """Handle panel close event."""
        # A clump-finding job is still running: cancel + detach it so the window
        # closes immediately.  The orphaned worker keeps a cancelled token and
        # self-cleans once its thread unwinds (it is no longer parented here).
        if self._clump_thread is not None and self._clump_thread.isRunning():
            if self._clump_cancel is not None:
                self._clump_cancel.cancel()
            self._detach_running_job()

        has_pending = (
            self.result_mask is not None
            or self._label_view_active
            or bool(self._cloud_overlay_ids)
        )
        if has_pending:
            choice = confirm_pending_close(
                self,
                "Close Clump Finding Panel",
                "There are unapplied clump display/results changes.",
            )
            if choice == "cancel":
                event.ignore()
                return
            if choice == "discard":
                self.clear_results()
        super().closeEvent(event)
