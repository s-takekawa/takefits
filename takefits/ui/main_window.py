from takefits.ui.viewer import FITSViewer
from takefits.ui.menu_bar import MenuBar, mirror_menu_bar_to_window
from takefits.ui.subwindow import SubWindow, SubWindow_control

from takefits.ui.control_panel import ControlPanel
from takefits.ui.range_control import RangeControlPanel
from takefits.tools.regrid_panel import RegridPanel
from takefits.logic.regridder import Regridder
from takefits.ui.save_fits_dialog import SaveFITS
from takefits.core.app_state import create_app_state, MarkerSpec, RegionSpec
from takefits.core.action_session import ActionSession
from takefits.core.actions import ActionRegistry, register_default_actions
from takefits.core.annotation_serialization import (
    build_marker_payload_from_specs,
    build_region_payload_from_specs,
    snapshot_marker_specs,
    snapshot_region_specs,
)
from takefits.core.history_provenance import build_processing_history_lines_with_action
from takefits.core.workspace_restore import (
    build_workspace_restore_diagnostics,
    build_workspace_restore_status_line,
    compute_range_restore_mode,
    invalidate_workspace_restore_blit_cache,
    normalize_wcs_axis_type,
    normalize_wcs_unit,
    resolve_workspace_main_colorbar_layouts,
)
from PySide6.QtWidgets import (
    QApplication,
    QMessageBox,
    QFileDialog,
    QInputDialog,
    QWidget,
    QLineEdit,
    QComboBox,
    QCheckBox,
    QRadioButton,
    QSlider,
    QSpinBox,
    QDoubleSpinBox,
    QTabWidget,
    QTextEdit,
    QPlainTextEdit,
    QAbstractSpinBox,
    QToolTip,
)
from PySide6.QtCore import QThread, QTimer, QSignalBlocker, QEvent, Qt
from PySide6.QtGui import QCursor
import base64
import io
import os
import json
import math
import zlib
from datetime import datetime, timezone
from pathlib import Path
from functools import partial
from takefits.core.viewer_coordinator import ViewerCoordinator
from takefits.core.wcs_frames import frame_is_available, normalize_display_frame, preferred_display_frame


class MainWindow(FITSViewer):
    _WORKSPACE_COLORBAR_CONFIG_KEYS = (
        "colorbar_orientation",
        "colorbar_auto_layout",
        "colorbar_placement",
        "colorbar_align",
        "colorbar_gap_px",
        "colorbar_gap_x_px",
        "colorbar_gap_y_px",
        "colorbar_thickness_px",
        "colorbar_length_mode",
        "colorbar_length_value",
        "cbar_pos_x",
        "cbar_pos_y",
        "cbar_width",
        "cbar_height",
        "colorbar_tick_color",
        "colorbar_tick_length",
        "colorbar_mtick_length",
        "colorbar_tick_width",
        "colorbar_tick_direction",
        "colorbar_mtick_freq",
        "colorbar_tick_left",
        "colorbar_tick_right",
        "colorbar_tick_top",
        "colorbar_tick_bottom",
        "colorbar_tick_labelcolor",
        "colorbar_tick_labelleft",
        "colorbar_tick_labeltop",
        "colorbar_label",
        "colorbar_label_fontsize",
        "colorbar_label_color",
    )

    def __init__(self, plane, windowtitle, data, header, wcs, filename, spectral_metadata=None):
        super().__init__(data, header, wcs, filename, spectral_metadata)
        self.data = data
        self.header = header
        self.wcs = wcs
        self._browse_first_startup = bool(self.is_large_data_mode())
        
        self.plane = plane
        self.integ_result_windows = []
        self.initUI(plane)
        self.setWindowTitle(windowtitle)
        self.original_window_title = windowtitle
        
        self.filename = os.path.basename(filename)

        self.menu_bar = MenuBar.get_instance(self) #Make menubar
    
        self.region_mode_enabled = False
        self.region_shape = None

        # Connect the new actions to the handler
        self.menu_bar.circle_action.triggered.connect(lambda: self.set_region_shape("circle"))
        self.menu_bar.rectangle_action.triggered.connect(lambda: self.set_region_shape("rectangle"))
        self.menu_bar.ellipse_action.triggered.connect(lambda: self.set_region_shape("ellipse"))
        self.menu_bar.cube_action.triggered.connect(lambda: self.set_region_shape('cube')) 

        # Initialize the control panel
        self.subwindows = []
        self.subwindow1 = None
        self.subwindow2 = None
        startup_show_subwindow1 = bool(
            self.config_manager.config.get("startup_show_subwindow1", True)
        ) and not self._browse_first_startup
        startup_show_subwindow2 = bool(
            self.config_manager.config.get("startup_show_subwindow2", False)
        ) and not self._browse_first_startup
        if self.data.ndim > 2:
            SubWindow_control.update_subwindow(self.subwindow1, self.subwindow2)
            self.menu_bar.enable_plane_menu(True)
            self.menu_bar.sub1_action.setChecked(False)
            self.menu_bar.sub2_action.setChecked(False)
            if startup_show_subwindow1:
                self.ensure_subwindow1()
                self.subwindow1.setVisible(True)
                self.menu_bar.sub1_action.setChecked(True)
                self.subwindow1.raise_()
        else:
            self.menu_bar.enable_plane_menu(False)
        
        self.menu_bar.main_action.setChecked(True)
        self.control_panel = None
        self.range_panel = None
        if self._browse_first_startup:
            self.menu_bar.control_panel_action.setChecked(False)
            self.menu_bar.range_panel_action.setChecked(False)
            # Populate initial range values even in browse-first mode so the
            # viewer range inputs are not blank.  Pass None so that
            # update_ranges uses world_extent() which computes world
            # coordinates from the full pixel extent.
            self.update_ranges('xy', None, None)
        else:
            self.ensure_control_panel()
            self.ensure_range_panel()
            self.show_range_panel()
            # RangeControlPanel initialisation already seeds all range inputs.
            # Keep a light sync only, avoiding extra full-range redraw work at startup.
            self.range_panel._sync_inputs('xy')
            if self.data.ndim > 2:
                self.range_panel._sync_inputs('xz')
        
        #Color Scale connect
        self.color_button.clicked.connect(lambda: self._open_control_panel_tool("open_color_settings"))
        self.smooth_button.clicked.connect(lambda: self._open_control_panel_tool("open_smooth_settings"))
        if self.data.ndim > 2:
            #Integ connect
            self.integ_button.clicked.connect(lambda: self._open_control_panel_tool("open_integ_settings"))
            self.spec_button.clicked.connect(lambda: self._open_control_panel_tool("open_spec_window"))
        
        FITSViewer.main_window = self

        # Create ViewerCoordinator and register all viewers
        self.coordinator = ViewerCoordinator(self)
        self.coordinator.register_viewer('xy', self)
        if self.data.ndim > 2 and self.subwindow1 is not None:
            self.coordinator.register_viewer('xz', self.subwindow1)
            # Provide coordinator reference to subwindows
            self.subwindow1.coordinator = self.coordinator
        self.set_wcs_display_frame(preferred_display_frame(getattr(self, "wcs", None)), refresh=False)

        for subwindow in self.subwindows:
            self._connect_subwindow_controls(subwindow)

        if self.data.ndim > 2 and startup_show_subwindow2:
            subwindow2 = self.ensure_subwindow2()
            if subwindow2 is not None:
                subwindow2.setVisible(True)
                subwindow2.raise_()
                subwindow2.activateWindow()
                self.menu_bar.sub2_action.setChecked(True)

        self.raise_()
        self.activateWindow()

        # Create AppState for usecase layer (Phase 5 bridge)
        self.app_state = create_app_state(
            data=data,
            header=header,
            wcs=wcs,
            filepath=filename,
            spectral_metadata=spectral_metadata or {},
        )
        self.report_large_data_mode()
        # Keep app_state in sync with viewer position changes
        self.position_updated.connect(lambda x, y, z: self.sync_app_state())

        # ActionSession: record/replay friendly (foundation for Undo/Redo + CLI parity)
        registry = ActionRegistry()
        register_default_actions(registry)
        self.action_session = ActionSession(
            registry=registry,
            state=self.app_state,
            defer_initial_state_seed=self.is_large_data_mode(),
        )
        self._suspend_action_recording = False
        self._last_regions_fingerprint = None
        self._last_markers_fingerprint = None
        self._workspace_save_path = None
        self._workspace_window_order_clock = 0
        self._workspace_window_order = {}
        self._regions_commit_timer = QTimer(self)
        self._regions_commit_timer.setSingleShot(True)
        self._regions_commit_timer.timeout.connect(self._commit_regions_to_session)
        self._markers_commit_timer = QTimer(self)
        self._markers_commit_timer.setSingleShot(True)
        self._markers_commit_timer.timeout.connect(self._commit_markers_to_session)
        try:
            self.region_manager.selected_region_changed.connect(lambda *_: self._schedule_regions_commit())
        except Exception:
            pass
        try:
            if hasattr(self, "marker_manager") and self.marker_manager is not None:
                self.marker_manager.markers_changed.connect(lambda *_: self._schedule_markers_commit())
        except Exception:
            pass

        self._regrid_panel = None
        self._regrid_thread = None
        self._regrid_worker = None
        self._is_app_closing = False
        self._shared_view_history = []
        self._shared_view_history_index = -1
        self._suspend_view_history_recording = False
        self._view_history_batch_depth = 0
        self._record_shared_view_history(reason="init", force=True)
        self._refresh_undo_redo_actions()
        self._refresh_view_navigation_actions()
        app = QApplication.instance()
        if app is not None:
            try:
                app.focusWindowChanged.connect(lambda *_: self._refresh_view_navigation_actions())
            except Exception:
                pass
            try:
                app.installEventFilter(self)
            except Exception:
                pass
        self._seed_workspace_window_order()

    def ensure_subwindow1(self):
        """Create the XZ subwindow on first demand."""
        if self.data.ndim <= 2:
            return None
        if getattr(self, "subwindow1", None) is not None:
            return self.subwindow1

        filename = getattr(self, "filename_path", getattr(self, "filename", ""))
        self.subwindow1 = SubWindow('xz', "SubWindow1: %s" % filename, self)
        if self.subwindow1 not in self.subwindows:
            self.subwindows.append(self.subwindow1)
        SubWindow_control.update_subwindow(self.subwindow1, self.subwindow2)

        if getattr(self, "coordinator", None) is not None:
            self.coordinator.register_viewer('xz', self.subwindow1)
            self.subwindow1.coordinator = self.coordinator

        self._connect_subwindow_controls(self.subwindow1)

        x_limits = tuple(self.ax.get_xlim())
        z_limits = tuple(getattr(self, "original_zlim", (0.0, 0.0)))
        self.subwindow1.ax.set_xlim(*x_limits)
        self.subwindow1.ax.set_ylim(*z_limits)
        self.subwindow1.overlay_ax.set_position(self.subwindow1.ax.get_position())
        self._sync_color_to_subwindow(self.subwindow1)
        self.subwindow1.canvas.draw_idle()
        self.update_ranges('xz', x_limits, z_limits)

        self.subwindow1.original_xval = getattr(self, "original_xval", None)
        self.subwindow1.original_yval = getattr(self, "original_yval", None)
        self.subwindow1.original_zval = getattr(self, "original_zval", None)

        toolbar = getattr(self.subwindow1, "toolbar", None)
        sync_mode = getattr(toolbar, "sync_navigation_mode_from_linked", None)
        if callable(sync_mode):
            sync_mode()

        if getattr(self, "range_panel", None) is not None:
            self.range_panel.subwindows = self.subwindows
            self.range_panel._sync_inputs('xz')
        self._refresh_view_navigation_actions()

        return self.subwindow1

    def ensure_subwindow2(self):
        """Create the ZY subwindow on first demand to reduce startup cost."""
        if self.data.ndim <= 2:
            return None
        if getattr(self, "subwindow2", None) is not None:
            return self.subwindow2

        filename = getattr(self, "filename_path", getattr(self, "filename", ""))
        self.subwindow2 = SubWindow('zy', "SubWindow2: %s" % filename, self)
        if self.subwindow2 not in self.subwindows:
            self.subwindows.append(self.subwindow2)
        SubWindow_control.update_subwindow(self.subwindow1, self.subwindow2)

        # SubWindow2 can be created after unsaved config Apply.
        # Mirror the live navigation config so wheel behavior matches Main/Sub1.
        live_config = getattr(getattr(self, "config_manager", None), "config", {})
        if isinstance(live_config, dict):
            self.subwindow2.scrollspeed = float(live_config.get('scrollspeed', getattr(self, "scrollspeed", 0.1)))
            self.subwindow2.invert_wheel_direction = bool(
                live_config.get('invert_wheel_direction', getattr(self, "invert_wheel_direction", False))
            )
            sub_cfg_mgr = getattr(self.subwindow2, "config_manager", None)
            if sub_cfg_mgr is not None and isinstance(getattr(sub_cfg_mgr, "config", None), dict):
                sub_cfg_mgr.config['scrollspeed'] = self.subwindow2.scrollspeed
                sub_cfg_mgr.config['invert_wheel_direction'] = self.subwindow2.invert_wheel_direction

        if getattr(self, "coordinator", None) is not None:
            self.coordinator.register_viewer('zy', self.subwindow2)
            self.subwindow2.coordinator = self.coordinator

        self._connect_subwindow_controls(self.subwindow2)

        # Mirror current Main(XY) + XZ ranges into the newly created ZY view.
        z_limits = tuple(self.subwindow1.ax.get_ylim()) if self.subwindow1 is not None else tuple(getattr(self, "original_zlim", (0.0, 0.0)))
        y_limits = tuple(self.ax.get_ylim())
        self.subwindow2.ax.set_xlim(*z_limits)
        self.subwindow2.ax.set_ylim(*y_limits)
        self.subwindow2.overlay_ax.set_position(self.subwindow2.ax.get_position())
        self._sync_color_to_subwindow(self.subwindow2)
        self.subwindow2.canvas.draw_idle()
        self.update_ranges('zy', z_limits, y_limits)

        # Seed conversion anchors for newly created ZY range edits.
        self.subwindow2.original_xval = getattr(self, "original_xval", None)
        self.subwindow2.original_yval = getattr(self, "original_yval", None)
        self.subwindow2.original_zval = getattr(self, "original_zval", None)

        toolbar = getattr(self.subwindow2, "toolbar", None)
        sync_mode = getattr(toolbar, "sync_navigation_mode_from_linked", None)
        if callable(sync_mode):
            sync_mode()

        # If cursor lines are already visible on XY/XZ, mirror that state to ZY on first creation.
        try:
            xy_vline = self._get_plane_vline('xy')
            xy_hline = self._get_plane_hline('xy')
            xz_vline = self._get_plane_vline('xz')
            xz_hline = self._get_plane_hline('xz')
            show_cursor = any(
                line is not None and line.get_visible()
                for line in (xy_vline, xy_hline, xz_vline, xz_hline)
            )
            if show_cursor:
                zpix = float(self._get_shared_zpix())
                ypix = float(self._get_shared_ypix())
                zy_vline = self._get_plane_vline('zy')
                zy_hline = self._get_plane_hline('zy')
                if zy_vline is not None:
                    zy_vline.set_xdata([zpix])
                if zy_hline is not None:
                    zy_hline.set_ydata([ypix])
                self._update_plane_cursor('zy', x=zpix, y=ypix)
                self._set_crosshair_point_for_plane('zy', x=zpix, y=ypix)
                self._set_crosshair_visibility_for_plane('zy', True)
                zy_chlabel = self._get_plane_chlabel('zy')
                if zy_chlabel is not None:
                    zy_chlabel.set_visible(True)
                self.subwindow2.canvas.draw_idle()
        except Exception:
            pass

        if getattr(self, "range_panel", None) is not None:
            self.range_panel.subwindows = self.subwindows
            self.range_panel._sync_inputs('zy')
        self._refresh_view_navigation_actions()

        return self.subwindow2

    def _sync_color_to_subwindow(self, subwindow):
        """Propagate the current main viewer's colormap and clim to a new subwindow."""
        main_im = getattr(self, "im", None)
        sub_im = getattr(subwindow, "im", None)
        if main_im is None or sub_im is None:
            return
        try:
            sub_im.set_cmap(main_im.get_cmap())
            sub_im.set_norm(main_im.norm)
            sub_im.set_clim(*main_im.get_clim())
        except Exception:
            pass
        # Keep displaymap.colorscale in sync so future redraws use the right cmap.
        main_cs = getattr(getattr(self, "displaymap", None), "colorscale", None)
        if main_cs and getattr(subwindow, "displaymap", None) is not None:
            subwindow.displaymap.colorscale = main_cs

    def _connect_subwindow_controls(self, subwindow):
        if subwindow is None:
            return
        if bool(getattr(subwindow, "_control_tool_buttons_connected", False)):
            return
        subwindow.color_button.clicked.connect(lambda: self._open_control_panel_tool("open_color_settings"))
        subwindow.integ_button.clicked.connect(lambda: self._open_control_panel_tool("open_integ_settings"))
        subwindow.smooth_button.clicked.connect(lambda: self._open_control_panel_tool("open_smooth_settings"))
        subwindow.spec_button.clicked.connect(lambda: self._open_control_panel_tool("open_spec_window"))
        subwindow._control_tool_buttons_connected = True

    def _set_panel_toggle_checked(self, action_attr: str, visible: bool):
        menu_bar = getattr(self, "menu_bar", None)
        if menu_bar is None:
            return
        action = getattr(menu_bar, str(action_attr), None)
        if action is None:
            return
        blocker = QSignalBlocker(action)
        action.setChecked(bool(visible))

    def ensure_control_panel(self, *, visible: bool = True):
        """Create the tools panel on first demand."""
        if getattr(self, "control_panel", None) is None:
            self.control_panel = ControlPanel(self, self.subwindows, visible=visible)
            for subwindow in self.subwindows:
                self._connect_subwindow_controls(subwindow)
        return self.control_panel

    def ensure_range_panel(self):
        """Create the range panel on first demand."""
        if getattr(self, "range_panel", None) is None:
            self.range_panel = RangeControlPanel(self, self.subwindows)
            self.range_panel.hide()
        return self.range_panel

    def _open_control_panel_tool(self, method_name: str):
        panel = self.ensure_control_panel(visible=False)
        opener = getattr(panel, str(method_name), None)
        if callable(opener):
            return opener()
        return None

    # ------------------------------------------------------------------
    # Regions/Markers -> AppState action bridge (record/replay friendly)
    def _schedule_regions_commit(self, delay_ms: int = 250) -> None:
        if self._suspend_action_recording:
            return
        self._regions_commit_timer.start(int(delay_ms))

    def _schedule_markers_commit(self, delay_ms: int = 250) -> None:
        if self._suspend_action_recording:
            return
        self._markers_commit_timer.start(int(delay_ms))

    def _region_specs_snapshot(self):
        return snapshot_region_specs(getattr(self, "region_manager", None), default_plane="xy")

    def _marker_specs_snapshot(self):
        return snapshot_marker_specs(getattr(self, "marker_manager", None))

    def _commit_regions_to_session(self) -> None:
        if self._suspend_action_recording:
            return
        specs = self._region_specs_snapshot()
        fingerprint = json.dumps(specs, sort_keys=True, separators=(",", ":"))
        if fingerprint == self._last_regions_fingerprint:
            return
        self._last_regions_fingerprint = fingerprint
        try:
            self.action_session.execute("set_regions", regions=specs)
            self._refresh_undo_redo_actions()
        except Exception:
            # Fallback: keep AppState updated even if ActionSession is unavailable.
            try:
                self.app_state.regions = [RegionSpec.from_dict(entry) for entry in specs]
            except Exception:
                pass

    def _commit_markers_to_session(self) -> None:
        if self._suspend_action_recording:
            return
        specs = self._marker_specs_snapshot()
        fingerprint = json.dumps(specs, sort_keys=True, separators=(",", ":"))
        if fingerprint == self._last_markers_fingerprint:
            return
        self._last_markers_fingerprint = fingerprint
        try:
            self.action_session.execute("set_markers", markers=specs)
            self._refresh_undo_redo_actions()
        except Exception:
            try:
                self.app_state.markers = [MarkerSpec.from_dict(entry) for entry in specs]
            except Exception:
                pass

    def set_region_shape(self, shape):
        """
        Sets the shape for the region selection mode.
        If the selected shape's action is already checked, it disables the region mode.
        """
        action = None
        if shape == "circle":
            action = self.menu_bar.circle_action
        elif shape == "rectangle":
            action = self.menu_bar.rectangle_action
        elif shape == "ellipse":
            action = self.menu_bar.ellipse_action
        elif shape == "cube":
            action = self.menu_bar.cube_action

        # If the current shape is re-selected, toggle region mode off.
        if self.region_shape == shape and self.region_mode_enabled:
            self.region_mode_enabled = False
            self.region_shape = None
            action.setChecked(False)  # Uncheck the menu item
            self.setWindowTitle(self.original_window_title)
        # Otherwise, enable region mode with the selected shape.
        elif action and action.isChecked():
            self.region_mode_enabled = True
            self.region_shape = shape
            self.setWindowTitle(f"[REGION MODE: {shape.upper()}] {self.original_window_title}")
            self.region_manager.set_region_mode(shape)
            if hasattr(self, 'set_marker_mode'):
                self.set_marker_mode(False)
            self._reset_navigation_mode()
        # This case handles unchecking via the action group
        else:
            self.region_mode_enabled = False
            self.region_shape = None
            self.setWindowTitle(self.original_window_title)

        if not self.region_mode_enabled:
            for region_action in (self.menu_bar.circle_action,
                                   self.menu_bar.rectangle_action,
                                   self.menu_bar.ellipse_action):
                region_action.setChecked(False)


        # Propagate the state to subwindows
        for subwindow in self.subwindows:
            subwindow.region_mode_enabled = self.region_mode_enabled
            subwindow.region_shape = self.region_shape 

        # Notify other windows like IntegResultWindow
        self.integ_result_windows = [ref for ref in self.integ_result_windows if ref() is not None]
        for window_ref in self.integ_result_windows:
            window = window_ref()
            if window:
                window.set_region_mode(self.region_mode_enabled)


    def disable_region_mode(self):
        super().disable_region_mode()
        if hasattr(self, 'menu_bar'):
            for action in (self.menu_bar.circle_action,
                           self.menu_bar.rectangle_action,
                           self.menu_bar.ellipse_action,
                           self.menu_bar.cube_action):
                if action is not None:
                    action.setChecked(False)

    def sync_app_state(self):
        """Sync AppState with current viewer state (usecase layer bridge)."""
        if not hasattr(self, 'app_state') or self.app_state is None:
            return
        self.sync_app_state_data()
        coord = self.get_coordinator()
        if coord is not None:
            self.app_state.cursor.xpix = coord.xpix
            self.app_state.cursor.ypix = coord.ypix
            self.app_state.cursor.zpix = coord.zpix
            self.app_state.current_z = coord.zpix
            if hasattr(coord.coord_state, 'spix'):
                self.app_state.cursor.spix = coord.coord_state.spix
                self.app_state.current_s = coord.coord_state.spix

    def sync_app_state_data(self, data=None, header=None, wcs=None):
        """Sync AppState data/header/WCS with the latest viewer state."""
        if not hasattr(self, 'app_state') or self.app_state is None:
            return
        self.app_state.data = self.data if data is None else data
        self.app_state.header = self.header if header is None else header
        self.app_state.wcs = self.wcs if wcs is None else wcs

    def record_action(self, name, params=None, replace_tag=None):
        """Record an action in ActionSession history without re-executing it."""
        if self._suspend_action_recording:
            return
        if not hasattr(self, "action_session") or self.action_session is None:
            return
        self.sync_app_state_data()
        payload = params or {}
        try:
            self.action_session.record(name, params=payload, replace_tag=replace_tag)
            self._refresh_undo_redo_actions()
        except Exception:
            # Keep GUI workflow resilient even if recording fails.
            pass

    def clear_recorded_action(self, replace_tag):
        """Remove a previously recorded preview action."""
        if self._suspend_action_recording:
            return
        if not replace_tag:
            return
        if not hasattr(self, "action_session") or self.action_session is None:
            return
        try:
            self.action_session.remove_record_by_tag(replace_tag)
            self._refresh_undo_redo_actions()
        except Exception:
            pass

    def _refresh_undo_redo_actions(self):
        menu_bar = getattr(self, "menu_bar", None)
        if menu_bar is None:
            return
        owner = self._active_analysis_owner()
        session = getattr(owner, "action_session", None)
        if session is None:
            menu_bar.set_undo_redo_enabled(False, False)
            return
        undo_label = None
        redo_label = None
        try:
            if session.can_undo() and session.cursor > 0:
                undo_label = self._format_action_label(session.history[session.cursor - 1].action)
            if session.can_redo() and session.cursor < len(session.history):
                redo_label = self._format_action_label(session.history[session.cursor].action)
        except Exception:
            undo_label = None
            redo_label = None
        try:
            menu_bar.set_undo_redo_enabled(
                session.can_undo(),
                session.can_redo(),
                undo_label=undo_label,
                redo_label=redo_label,
            )
        except Exception:
            pass

    def _active_analysis_owner(self):
        app = QApplication.instance()
        active_window = app.activeWindow() if app is not None else None
        if active_window is None or active_window is self:
            return self
        session = getattr(active_window, "action_session", None)
        undo_fn = getattr(active_window, "undo_last_action", None)
        redo_fn = getattr(active_window, "redo_last_action", None)
        if session is not None and callable(undo_fn) and callable(redo_fn):
            return active_window
        return self

    def _format_action_label(self, action_name: str) -> str:
        raw = str(action_name or "").strip()
        if not raw:
            return ""
        pretty = raw.replace("_", " ").strip()
        if not pretty:
            return raw
        return pretty[:1].upper() + pretty[1:]

    def _capture_shared_cursor_snapshot(self):
        snapshot = {}
        def _finite_float(value):
            try:
                parsed = float(value)
            except Exception:
                return None
            return parsed if math.isfinite(parsed) else None

        for key, getter in (
            ("xpix", getattr(self, "_get_shared_xpix", None)),
            ("ypix", getattr(self, "_get_shared_ypix", None)),
            ("zpix", getattr(self, "_get_shared_zpix", None)),
        ):
            if not callable(getter):
                continue
            try:
                snapshot[key] = int(round(float(getter())))
            except Exception:
                continue
        # Preserve sub-pixel XY crosshair position so workspace load can restore
        # the exact drawn cursor line, not only the rounded pixel index.
        state = getattr(self, "state", None)
        cursor_x = getattr(state, "cursor_x", None) if state is not None else None
        cursor_y = getattr(state, "cursor_y", None) if state is not None else None
        if cursor_x is None:
            try:
                xdata = self.vline.get_xdata() if getattr(self, "vline", None) is not None else None
                if xdata is not None and len(xdata):
                    cursor_x = float(xdata[0])
            except Exception:
                cursor_x = None
        if cursor_y is None:
            try:
                ydata = self.hline.get_ydata() if getattr(self, "hline", None) is not None else None
                if ydata is not None and len(ydata):
                    cursor_y = float(ydata[0])
            except Exception:
                cursor_y = None
        if cursor_x is not None:
            try:
                snapshot["cursor_x"] = float(cursor_x)
            except Exception:
                pass
        if cursor_y is not None:
            try:
                snapshot["cursor_y"] = float(cursor_y)
            except Exception:
                pass
        try:
            vline_visible = bool(getattr(self, "vline").get_visible())
            hline_visible = bool(getattr(self, "hline").get_visible())
            snapshot["cursor_visible"] = bool(vline_visible or hline_visible)
        except Exception:
            pass

        world_getters = (
            ("world_x", getattr(self, "_get_shared_world_x", None)),
            ("world_y", getattr(self, "_get_shared_world_y", None)),
            ("world_z", getattr(self, "_get_shared_world_z", None)),
            ("world_s", getattr(self, "_get_shared_world_s", None)),
        )
        for key, getter in world_getters:
            if not callable(getter):
                continue
            value = _finite_float(getter())
            if value is None:
                continue
            snapshot[key] = value

        # Persist native (unwrapped/unformatted) WCS values for robust reload.
        wcs = getattr(self, "wcs", None)
        if wcs is not None and cursor_x is not None and cursor_y is not None:
            try:
                naxis = max(int(getattr(wcs, "naxis", 0) or 0), 2)
            except Exception:
                naxis = 2
            pixel = [0.0] * naxis
            pixel[0] = float(cursor_x)
            pixel[1] = float(cursor_y)
            if naxis > 2:
                z_seed = snapshot.get("zpix", 0)
                try:
                    pixel[2] = float(z_seed)
                except Exception:
                    pixel[2] = 0.0
            try:
                native_world = wcs.wcs_pix2world([pixel], 0)[0]
            except Exception:
                native_world = None
            if native_world is not None:
                native_keys = ("world_native_x", "world_native_y", "world_native_z", "world_native_s")
                for idx, key in enumerate(native_keys):
                    if idx >= len(native_world):
                        break
                    value = _finite_float(native_world[idx])
                    if value is not None:
                        snapshot[key] = value

        if "world_x" not in snapshot or "world_y" not in snapshot:
            if wcs is not None and "xpix" in snapshot and "ypix" in snapshot:
                try:
                    naxis = max(int(getattr(wcs, "naxis", 0) or 0), 2)
                except Exception:
                    naxis = 2
                pixel = [0.0] * naxis
                pixel[0] = float(snapshot.get("cursor_x", snapshot["xpix"]))
                pixel[1] = float(snapshot.get("cursor_y", snapshot["ypix"]))
                if naxis > 2:
                    pixel[2] = float(snapshot.get("zpix", 0))
                try:
                    world = wcs.wcs_pix2world([pixel], 0)[0]
                except Exception:
                    world = None
                if world is not None:
                    wx = _finite_float(world[0] if len(world) > 0 else None)
                    wy = _finite_float(world[1] if len(world) > 1 else None)
                    wz = _finite_float(world[2] if len(world) > 2 else None)
                    ws = _finite_float(world[3] if len(world) > 3 else None)
                    if wx is not None and "world_x" not in snapshot:
                        snapshot["world_x"] = wx
                    if wy is not None and "world_y" not in snapshot:
                        snapshot["world_y"] = wy
                    if wz is not None and "world_z" not in snapshot:
                        snapshot["world_z"] = wz
                    if ws is not None and "world_s" not in snapshot:
                        snapshot["world_s"] = ws
        return snapshot

    def _build_workspace_cursor_overlay_snapshot(self, preferred_cursor=None):
        def _safe_float(value, fallback=None):
            try:
                parsed = float(value)
            except Exception:
                return fallback
            if not math.isfinite(parsed):
                return fallback
            return parsed

        def _pick_float(*values, fallback=None):
            for value in values:
                parsed = _safe_float(value, None)
                if parsed is not None:
                    return parsed
            return fallback

        current = self._capture_shared_cursor_snapshot()
        if not isinstance(current, dict):
            current = {}
        preferred = preferred_cursor if isinstance(preferred_cursor, dict) else {}
        viewer_state = getattr(self, "state", None)

        shared_x = _pick_float(current.get("xpix"), fallback=0.0)
        shared_y = _pick_float(current.get("ypix"), fallback=0.0)
        shared_z = _pick_float(current.get("zpix"), fallback=0.0)

        cursor_x = _pick_float(
            preferred.get("cursor_x"),
            current.get("cursor_x"),
            getattr(viewer_state, "cursor_x", None),
            shared_x,
            fallback=shared_x,
        )
        cursor_y = _pick_float(
            preferred.get("cursor_y"),
            current.get("cursor_y"),
            getattr(viewer_state, "cursor_y", None),
            shared_y,
            fallback=shared_y,
        )

        if "cursor_visible" in preferred:
            visible = bool(preferred.get("cursor_visible"))
        elif "cursor_visible" in current:
            visible = bool(current.get("cursor_visible"))
        else:
            visible = False

        snapshot = {
            "xy": {
                "visible": bool(visible),
                "x": float(cursor_x),
                "y": float(cursor_y),
            }
        }
        if getattr(self, "data", None) is not None and getattr(self.data, "ndim", 0) > 2:
            snapshot["xz"] = {
                "visible": bool(visible),
                "x": float(shared_x),
                "y": float(shared_z),
            }
            snapshot["zy"] = {
                "visible": bool(visible),
                "x": float(shared_z),
                "y": float(shared_y),
            }
        return snapshot

    def _refresh_cursor_overlay_after_workspace_restore(self, preferred_cursor=None, *, defer_retry: bool = True):
        restore_crosshair = getattr(self, "_restore_crosshair_state", None)
        if not callable(restore_crosshair):
            return
        snapshot = self._build_workspace_cursor_overlay_snapshot(preferred_cursor=preferred_cursor)
        if not isinstance(snapshot, dict) or not snapshot:
            return
        def _apply_cursor_overlay():
            if bool(getattr(self, "_is_app_closing", False)):
                return
            try:
                restore_crosshair(snapshot)
            except Exception:
                pass

        _apply_cursor_overlay()
        if not bool(defer_retry):
            return
        try:
            # Some panels queue draw_idle() during workspace restore.
            # Re-apply once the event loop drains so animated cursor artists remain visible.
            for delay_ms in (0, 40):
                QTimer.singleShot(delay_ms, _apply_cursor_overlay)
        except Exception:
            pass

    def _set_workspace_colorbar_restore_in_progress(self, active: bool):
        try:
            self._workspace_colorbar_restore_in_progress = bool(active)
        except Exception:
            pass

    def _begin_workspace_restore_canvas_redraw_batch(self):
        self._workspace_restore_canvas_redraw_batch_active = True
        self._workspace_restore_pending_canvas_redraws = {}

    def _queue_workspace_restore_canvas_redraw(self, canvas) -> bool:
        if canvas is None:
            return False
        if not bool(getattr(self, "_workspace_restore_canvas_redraw_batch_active", False)):
            return False
        pending = getattr(self, "_workspace_restore_pending_canvas_redraws", None)
        if not isinstance(pending, dict):
            pending = {}
            self._workspace_restore_pending_canvas_redraws = pending
        pending[id(canvas)] = canvas
        return True

    def _flush_workspace_restore_canvas_redraw_batch(self):
        pending = getattr(self, "_workspace_restore_pending_canvas_redraws", None)
        canvases = list(pending.values()) if isinstance(pending, dict) else []
        self._workspace_restore_pending_canvas_redraws = {}
        self._workspace_restore_canvas_redraw_batch_active = False
        for canvas in canvases:
            draw = getattr(canvas, "draw", None)
            if callable(draw):
                try:
                    draw()
                    continue
                except Exception:
                    pass
            draw_idle = getattr(canvas, "draw_idle", None)
            if callable(draw_idle):
                try:
                    draw_idle()
                except Exception:
                    pass

    def _request_canvas_redraw(self, canvas, *, immediate: bool = False) -> bool:
        if canvas is None:
            return False
        if self._queue_workspace_restore_canvas_redraw(canvas):
            return True
        draw_name = "draw" if immediate else "draw_idle"
        draw = getattr(canvas, draw_name, None)
        if callable(draw):
            try:
                draw()
                return True
            except Exception:
                pass
        if not immediate:
            draw = getattr(canvas, "draw", None)
            if callable(draw):
                try:
                    draw()
                    return True
                except Exception:
                    pass
        return False

    def _refresh_colorbar_layout_after_workspace_restore(self, colorbar_state=None):
        if not isinstance(colorbar_state, dict) or not colorbar_state:
            return
        if bool(getattr(self, "_is_app_closing", False)):
            return

        target_snapshot = self._desired_workspace_colorbar_layout_snapshot(colorbar_state)

        try:
            QApplication.processEvents()
        except Exception:
            pass

        try:
            self._restore_workspace_colorbar_state(colorbar_state, apply_to_open_windows=True)
        except Exception:
            pass

        try:
            QApplication.processEvents()
        except Exception:
            pass

        current_snapshot = self._current_workspace_colorbar_layout_snapshot()
        if self._workspace_colorbar_layouts_match(current_snapshot, target_snapshot):
            return

        try:
            self._restore_workspace_colorbar_state(colorbar_state, apply_to_open_windows=True)
        except Exception:
            pass

    def _cursor_world_to_pixel_snapshot(self, cursor_snapshot, wcs):
        if not isinstance(cursor_snapshot, dict) or wcs is None:
            return None

        try:
            naxis = max(int(getattr(wcs, "naxis", 0) or 0), 2)
        except Exception:
            naxis = 2

        def _parse(value):
            try:
                parsed = float(value)
            except Exception:
                return None
            return parsed if math.isfinite(parsed) else None

        def _axis_is_longitude(axis_index: int) -> bool:
            try:
                ctype_list = list(getattr(getattr(wcs, "wcs", None), "ctype", []) or [])
                token = str(ctype_list[axis_index] if axis_index < len(ctype_list) else "").upper()
            except Exception:
                token = ""
            return token.startswith("RA") or ("GLON" in token) or token.endswith("LON")

        world_values = [None, None, None, None]
        native_keys = ("world_native_x", "world_native_y", "world_native_z", "world_native_s")
        display_keys = ("world_x", "world_y", "world_z", "world_s")
        native_used = [False, False, False, False]
        for idx in range(len(world_values)):
            native_val = _parse(cursor_snapshot.get(native_keys[idx]))
            if native_val is not None:
                world_values[idx] = native_val
                native_used[idx] = True
                continue
            display_val = _parse(cursor_snapshot.get(display_keys[idx]))
            if display_val is not None:
                world_values[idx] = display_val

        if world_values[0] is None or world_values[1] is None:
            return None

        fallback_getters = (
            getattr(self, "_get_shared_world_x", None),
            getattr(self, "_get_shared_world_y", None),
            getattr(self, "_get_shared_world_z", None),
            getattr(self, "_get_shared_world_s", None),
        )
        for idx in range(min(naxis, len(world_values))):
            if world_values[idx] is not None:
                continue
            getter = fallback_getters[idx] if idx < len(fallback_getters) else None
            if callable(getter):
                world_values[idx] = _parse(getter())
            if world_values[idx] is None:
                world_values[idx] = 0.0

        world_vector = [0.0] * naxis
        for idx in range(naxis):
            if idx < len(world_values) and world_values[idx] is not None:
                world_vector[idx] = float(world_values[idx])

        candidates = [list(world_vector)]
        if naxis >= 1 and (not native_used[0]) and _axis_is_longitude(0):
            base = world_vector[0]
            additions = []
            if base < 0.0:
                additions.append(base + 360.0)
            if base > 180.0:
                additions.append(base - 360.0)
            for value in additions:
                candidate = list(world_vector)
                candidate[0] = value
                candidates.append(candidate)
        if naxis >= 2 and (not native_used[1]) and _axis_is_longitude(1):
            base_candidates = list(candidates)
            base = world_vector[1]
            additions = []
            if base < 0.0:
                additions.append(base + 360.0)
            if base > 180.0:
                additions.append(base - 360.0)
            for base_candidate in base_candidates:
                for value in additions:
                    candidate = list(base_candidate)
                    candidate[1] = value
                    candidates.append(candidate)

        ref_x = _parse(cursor_snapshot.get("cursor_x"))
        if ref_x is None:
            ref_x = _parse(cursor_snapshot.get("xpix"))
        ref_y = _parse(cursor_snapshot.get("cursor_y"))
        if ref_y is None:
            ref_y = _parse(cursor_snapshot.get("ypix"))

        best_pixel = None
        best_score = None
        for candidate in candidates:
            try:
                pixel = wcs.wcs_world2pix([candidate], 0)[0]
            except Exception:
                continue
            if len(pixel) < 2:
                continue
            px = _parse(pixel[0])
            py = _parse(pixel[1])
            if px is None or py is None:
                continue
            if ref_x is None or ref_y is None:
                best_pixel = pixel
                break
            score = abs(px - ref_x) + abs(py - ref_y)
            if best_score is None or score < best_score:
                best_score = score
                best_pixel = pixel

        if best_pixel is None:
            return None

        xpix = _parse(best_pixel[0])
        ypix = _parse(best_pixel[1])
        if xpix is None or ypix is None:
            return None

        snapshot = {
            "xpix": xpix,
            "ypix": ypix,
            "cursor_x": xpix,
            "cursor_y": ypix,
        }
        if len(best_pixel) > 2:
            zpix = _parse(best_pixel[2])
            if zpix is not None:
                snapshot["zpix"] = zpix
        return snapshot

    def _view_history_toolbar_state(self, viewer):
        if viewer is self and hasattr(self, "_shared_view_history"):
            return self._shared_view_history_state()
        toolbar = getattr(viewer, "toolbar", None)
        if toolbar is None:
            return (False, False)
        stack = getattr(toolbar, "_nav_stack", None)
        elements = getattr(stack, "_elements", None) if stack is not None else None
        pos = getattr(stack, "_pos", -1) if stack is not None else -1
        if not isinstance(elements, list) or len(elements) == 0:
            return (False, False)
        total = len(elements)
        can_back = pos > 0
        can_forward = pos >= 0 and pos < (total - 1)
        return (can_back, can_forward)

    def _active_view_navigation_owner(self):
        app = QApplication.instance()

        def _is_view_owner(candidate):
            if candidate is None:
                return False
            back_fn = getattr(candidate, "view_back", None)
            forward_fn = getattr(candidate, "view_forward", None)
            if not (callable(back_fn) and callable(forward_fn)):
                return False
            if candidate is self:
                return True
            state_getter = getattr(candidate, "_view_history_state", None)
            shared_getter = getattr(candidate, "_shared_view_history_state", None)
            return callable(state_getter) or callable(shared_getter)

        def _resolve_from_widget(widget):
            if widget is None:
                return None
            seen = set()
            current = widget
            while current is not None and id(current) not in seen:
                seen.add(id(current))
                if _is_view_owner(current):
                    return current
                for attr in ("fits_viewer", "main_viewer", "viewer", "parent"):
                    candidate = getattr(current, attr, None)
                    if _is_view_owner(candidate):
                        return candidate
                parent_getter = getattr(current, "parentWidget", None)
                parent = parent_getter() if callable(parent_getter) else None
                if parent is not None:
                    current = parent
                    continue
                window_getter = getattr(current, "window", None)
                window = window_getter() if callable(window_getter) else None
                if window is not None and window is not current:
                    current = window
                    continue
                break
            return None

        candidates = []
        if app is not None:
            active_window = None
            active_window_getter = getattr(app, "activeWindow", None)
            if callable(active_window_getter):
                try:
                    active_window = active_window_getter()
                except Exception:
                    active_window = None

            for getter_name in ("activeWindow", "activeModalWidget", "activePopupWidget"):
                getter = getattr(app, getter_name, None)
                if callable(getter):
                    try:
                        candidates.append(getter())
                    except Exception:
                        continue

            focus_getter = getattr(app, "focusWidget", None)
            if callable(focus_getter):
                try:
                    focus_widget = focus_getter()
                except Exception:
                    focus_widget = None
                if focus_widget is not None:
                    if active_window is None:
                        candidates.append(focus_widget)
                    else:
                        window_getter = getattr(focus_widget, "window", None)
                        focus_window = window_getter() if callable(window_getter) else None
                        if focus_window is active_window:
                            candidates.append(focus_widget)

            top_level_getter = getattr(app, "topLevelWidgets", None)
            if callable(top_level_getter):
                try:
                    for widget in list(top_level_getter() or []):
                        try:
                            if widget is not None and widget.isActiveWindow():
                                candidates.append(widget)
                        except Exception:
                            continue
                except Exception:
                    pass

        for candidate in candidates:
            owner = _resolve_from_widget(candidate)
            if owner is not None:
                return owner

        for window in self._live_integration_windows():
            try:
                if window is not None and window.isActiveWindow():
                    return window
            except Exception:
                continue
        for window in list(getattr(self, "channel_map_windows", []) or []):
            try:
                if window is not None and window.isActiveWindow():
                    return window
            except Exception:
                continue

        return self

    def _view_navigation_cursor_for_owner(self, owner):
        if owner is None:
            return None
        try:
            if owner is self:
                return ("shared", int(getattr(self, "_shared_view_history_index", -1)))
            if hasattr(owner, "_view_history_index"):
                return ("local", int(getattr(owner, "_view_history_index", -1)))
            if hasattr(owner, "_shared_view_history_index"):
                return ("shared", int(getattr(owner, "_shared_view_history_index", -1)))
            toolbar = getattr(owner, "toolbar", None)
            stack = getattr(toolbar, "_nav_stack", None) if toolbar is not None else None
            if stack is not None and hasattr(stack, "_pos"):
                return ("toolbar", int(getattr(stack, "_pos", -1)))
        except Exception:
            return None
        return None

    def _should_bypass_view_shortcuts(self):
        app = QApplication.instance()
        if app is None:
            return False
        focus = app.focusWidget()
        if focus is None:
            return False
        editable_types = (QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox)
        if not isinstance(focus, editable_types):
            return False
        readonly = False
        if hasattr(focus, "isReadOnly"):
            try:
                readonly = bool(focus.isReadOnly())
            except Exception:
                readonly = False
        return not readonly

    def _workspace_tool_panel_attrs(self):
        return (
            "color_settings_panel",
            "scaling_panel",
            "unit_conversion_panel",
            "integ_settings_panel",
            "chmap_settings_panel",
            "smooth_settings_panel",
            "baseline_panel",
            "spec_window",
            "pvd_panel",
            "mask_settings_panel",
            "contour_panel",
            "arithmetic_panel",
            "clump_finding_panel",
        )

    def _collect_workspace_windows_by_token(self):
        windows = {}

        def _add(token, window):
            if not token or window is None:
                return
            windows[str(token)] = window

        _add("main_window", self)
        _add("control_panel", getattr(self, "control_panel", None))
        _add("range_panel", getattr(self, "range_panel", None))
        _add("marker_panel", getattr(self, "marker_panel", None))
        _add("regrid_panel", getattr(self, "_regrid_panel", None))

        if self.data.ndim > 2:
            _add("subwindow:xz", getattr(self, "subwindow1", None))
            _add("subwindow:zy", getattr(self, "subwindow2", None))

        control_panel = getattr(self, "control_panel", None)
        if control_panel is not None:
            for attr in self._workspace_tool_panel_attrs():
                _add(f"tool:{attr}", getattr(control_panel, attr, None))

        integration_key_counts = {}
        for idx, window in enumerate(self._live_integration_windows()):
            raw_key = str(self._integration_window_color_key(window) or f"idx:{idx}")
            occurrence = int(integration_key_counts.get(raw_key, 0))
            integration_key_counts[raw_key] = occurrence + 1
            key = f"{raw_key}::{occurrence}"
            _add(f"integration:{key}", window)
            _add(f"integration_marker:{key}", getattr(window, "marker_panel", None))

        channel_key_counts = {}
        for idx, window in enumerate(list(getattr(self, "channel_map_windows", []) or [])):
            if window is None:
                continue
            raw_key = str(self._channel_window_color_key(window) or f"idx:{idx}")
            occurrence = int(channel_key_counts.get(raw_key, 0))
            channel_key_counts[raw_key] = occurrence + 1
            key = f"{raw_key}::{occurrence}"
            _add(f"channel:{key}", window)
            _add(f"channel_marker:{key}", getattr(window, "marker_panel", None))

        return windows

    @staticmethod
    def _pop_workspace_entry_by_key_or_next(entries, desired_key):
        if not isinstance(entries, list) or not entries:
            return None
        key = str(desired_key or "").strip()
        if key:
            for idx, entry in enumerate(entries):
                if not isinstance(entry, dict):
                    continue
                if str(entry.get("key") or "").strip() == key:
                    return entries.pop(idx)
        return entries.pop(0)

    def _capture_marker_panel_state_for_window(self, window):
        panel = getattr(window, "marker_panel", None) if window is not None else None
        visible = False
        if panel is not None:
            try:
                visible = bool(panel.isVisible())
            except Exception:
                visible = False
        state = {"visible": visible}
        geometry = self._capture_window_geometry(panel)
        if geometry is not None:
            state["geometry"] = geometry
        return state

    def _restore_marker_panel_state_for_window(self, window, marker_panel_state):
        if window is None or not isinstance(marker_panel_state, dict):
            return False
        should_show = bool(marker_panel_state.get("visible", False))
        panel = getattr(window, "marker_panel", None)

        if should_show and panel is None:
            opener = getattr(window, "open_marker_panel", None)
            if callable(opener):
                try:
                    opener()
                except Exception:
                    panel = None
                else:
                    panel = getattr(window, "marker_panel", None)
        elif should_show and panel is not None:
            try:
                panel.show()
            except Exception:
                pass

        if panel is None:
            return False

        geometry = marker_panel_state.get("geometry")
        if isinstance(geometry, dict):
            self._restore_window_geometry(panel, geometry)

        if should_show:
            try:
                panel.raise_()
                panel.activateWindow()
            except Exception:
                pass
        else:
            try:
                panel.hide()
            except Exception:
                pass
        return True

    def _token_for_workspace_window(self, widget):
        if widget is None:
            return None
        window = widget
        if isinstance(widget, QWidget):
            try:
                window = widget if widget.isWindow() else widget.window()
            except Exception:
                window = widget
        for token, candidate in self._collect_workspace_windows_by_token().items():
            if candidate is window:
                return token
        return None

    def _touch_workspace_window_order(self, widget_or_token):
        token = None
        if isinstance(widget_or_token, str):
            token = widget_or_token
        else:
            token = self._token_for_workspace_window(widget_or_token)
        if not token:
            return
        self._workspace_window_order_clock = int(getattr(self, "_workspace_window_order_clock", 0)) + 1
        self._workspace_window_order[token] = self._workspace_window_order_clock

    def _seed_workspace_window_order(self):
        for token, window in self._collect_workspace_windows_by_token().items():
            if window is None:
                continue
            try:
                if not window.isVisible():
                    continue
            except Exception:
                continue
            self._touch_workspace_window_order(token)

    def _sync_takefits_menu_proxy(self, widget_or_window) -> bool:
        target = widget_or_window
        if isinstance(target, QWidget):
            try:
                target = target if target.isWindow() else target.window()
            except Exception:
                pass
        if target is None or target is self:
            return False
        if not isinstance(target, QWidget):
            return False
        menu_getter = getattr(target, "menuBar", None)
        if not callable(menu_getter):
            return False

        return bool(mirror_menu_bar_to_window(self, target))

    def _capture_workspace_window_z_order(self):
        windows = self._collect_workspace_windows_by_token()
        if not windows:
            return []

        visible_tokens = []
        for token, window in windows.items():
            if window is None:
                continue
            try:
                if window.isVisible():
                    visible_tokens.append(token)
            except Exception:
                continue
        if not visible_tokens:
            return []

        ranked = []
        for idx, token in enumerate(visible_tokens):
            rank = int(self._workspace_window_order.get(token, 0))
            ranked.append((rank, idx, token))
        ranked.sort(key=lambda item: (item[0], item[1]))
        ordered = [token for _rank, _idx, token in ranked]

        app = QApplication.instance()
        active_window = app.activeWindow() if app is not None else None
        active_token = self._token_for_workspace_window(active_window)
        if active_token in ordered:
            ordered = [token for token in ordered if token != active_token] + [active_token]

        return ordered

    def _restore_workspace_window_z_order(self, z_order_state):
        if not isinstance(z_order_state, list):
            return
        windows = self._collect_workspace_windows_by_token()
        ordered_tokens = []
        for entry in z_order_state:
            token = str(entry or "").strip()
            if not token:
                continue
            if token not in windows:
                legacy = f"{token}::0"
                if legacy in windows:
                    token = legacy
            if token not in windows:
                continue
            ordered_tokens.append(token)
        if not ordered_tokens:
            return

        top_window = None
        for token in ordered_tokens:
            window = windows.get(token)
            if window is None:
                continue
            try:
                if not window.isVisible():
                    continue
            except Exception:
                continue
            try:
                window.raise_()
                top_window = window
                self._touch_workspace_window_order(token)
            except Exception:
                continue
        if top_window is not None:
            try:
                top_window.activateWindow()
            except Exception:
                pass

    @staticmethod
    def _direct_widget_children(widget):
        if widget is None:
            return []
        return [child for child in widget.children() if isinstance(child, QWidget)]

    def _widget_path_from_root(self, root, widget):
        if root is None or widget is None:
            return None
        path = []
        current = widget
        while current is not None and current is not root:
            parent = current.parentWidget()
            if parent is None:
                return None
            siblings = self._direct_widget_children(parent)
            try:
                idx = siblings.index(current)
            except ValueError:
                return None
            path.append(int(idx))
            current = parent
        if current is not root:
            return None
        path.reverse()
        return path

    def _widget_from_root_path(self, root, path):
        if root is None or not isinstance(path, list):
            return None
        current = root
        for raw_idx in path:
            try:
                idx = int(raw_idx)
            except Exception:
                return None
            children = self._direct_widget_children(current)
            if idx < 0 or idx >= len(children):
                return None
            current = children[idx]
        return current

    def _capture_widget_state_entry(self, root, widget):
        if root is None or widget is None:
            return None

        kind = None
        payload = {}
        if isinstance(widget, QLineEdit):
            if isinstance(widget.parentWidget(), QAbstractSpinBox):
                return None
            kind = "line_edit"
            payload["text"] = str(widget.text() or "")
        elif isinstance(widget, QComboBox):
            kind = "combo_box"
            payload["index"] = int(widget.currentIndex())
            payload["text"] = str(widget.currentText() or "")
            payload["editable"] = bool(widget.isEditable())
        elif isinstance(widget, QSpinBox):
            kind = "spin_box"
            payload["value"] = int(widget.value())
        elif isinstance(widget, QDoubleSpinBox):
            kind = "double_spin_box"
            payload["value"] = float(widget.value())
        elif isinstance(widget, QCheckBox):
            kind = "check_box"
            payload["checked"] = bool(widget.isChecked())
        elif isinstance(widget, QRadioButton):
            kind = "radio_button"
            payload["checked"] = bool(widget.isChecked())
        elif isinstance(widget, QSlider):
            kind = "slider"
            payload["value"] = int(widget.value())
        elif isinstance(widget, QTabWidget):
            kind = "tab_widget"
            payload["index"] = int(widget.currentIndex())
        elif isinstance(widget, QTextEdit):
            kind = "text_edit"
            payload["text"] = str(widget.toPlainText() or "")
        elif isinstance(widget, QPlainTextEdit):
            kind = "plain_text_edit"
            payload["text"] = str(widget.toPlainText() or "")

        if not kind:
            return None

        object_name = str(widget.objectName() or "").strip()
        path = self._widget_path_from_root(root, widget)
        if not object_name and path is None:
            return None

        entry = {
            "kind": kind,
            "class": str(widget.metaObject().className() or ""),
            "path": path,
            "state": payload,
        }
        if object_name:
            entry["object_name"] = object_name
        return entry

    def _capture_window_ui_state(self, window):
        if window is None:
            return []
        entries = []
        try:
            widgets = window.findChildren(QWidget)
        except Exception:
            return entries
        for widget in widgets:
            entry = self._capture_widget_state_entry(window, widget)
            if entry is not None:
                entries.append(entry)
        return entries

    def _capture_workspace_ui_state(self):
        payload = {}
        for token, window in self._collect_workspace_windows_by_token().items():
            entries = self._capture_window_ui_state(window)
            if entries:
                payload[token] = entries
        return payload

    def _resolve_widget_from_snapshot(self, root, entry):
        if root is None or not isinstance(entry, dict):
            return None

        expected_class = str(entry.get("class") or "").strip()
        object_name = str(entry.get("object_name") or "").strip()
        if object_name:
            try:
                candidates = root.findChildren(QWidget, object_name)
            except Exception:
                candidates = []
            if expected_class:
                candidates = [
                    candidate
                    for candidate in candidates
                    if str(candidate.metaObject().className() or "") == expected_class
                ]
            if len(candidates) == 1:
                return candidates[0]

        widget = self._widget_from_root_path(root, entry.get("path"))
        if widget is None:
            return None
        if expected_class and str(widget.metaObject().className() or "") != expected_class:
            return None
        return widget

    def _apply_widget_state_entry(self, widget, entry):
        if widget is None or not isinstance(entry, dict):
            return False
        state = entry.get("state")
        if not isinstance(state, dict):
            return False

        kind = str(entry.get("kind") or "").strip()
        try:
            if kind == "line_edit" and isinstance(widget, QLineEdit):
                widget.setText(str(state.get("text") or ""))
                return True
            if kind == "combo_box" and isinstance(widget, QComboBox):
                index = state.get("index")
                text = str(state.get("text") or "")
                try:
                    index_int = int(index)
                except Exception:
                    index_int = -1
                if 0 <= index_int < widget.count():
                    widget.setCurrentIndex(index_int)
                elif text:
                    match = widget.findText(text)
                    if match >= 0:
                        widget.setCurrentIndex(match)
                    elif widget.isEditable():
                        widget.setEditText(text)
                return True
            if kind == "spin_box" and isinstance(widget, QSpinBox):
                widget.setValue(int(state.get("value")))
                return True
            if kind == "double_spin_box" and isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(state.get("value")))
                return True
            if kind == "check_box" and isinstance(widget, QCheckBox):
                widget.setChecked(bool(state.get("checked", False)))
                return True
            if kind == "radio_button" and isinstance(widget, QRadioButton):
                widget.setChecked(bool(state.get("checked", False)))
                return True
            if kind == "slider" and isinstance(widget, QSlider):
                widget.setValue(int(state.get("value")))
                return True
            if kind == "tab_widget" and isinstance(widget, QTabWidget):
                idx = int(state.get("index", 0))
                if 0 <= idx < widget.count():
                    widget.setCurrentIndex(idx)
                    return True
                return False
            if kind == "text_edit" and isinstance(widget, QTextEdit):
                widget.setPlainText(str(state.get("text") or ""))
                return True
            if kind == "plain_text_edit" and isinstance(widget, QPlainTextEdit):
                widget.setPlainText(str(state.get("text") or ""))
                return True
        except Exception:
            return False
        return False

    def _restore_window_ui_state(self, window, entries):
        if window is None or not isinstance(entries, list):
            return 0
        restored = 0
        for entry in entries:
            widget = self._resolve_widget_from_snapshot(window, entry)
            if widget is None:
                continue
            if self._apply_widget_state_entry(widget, entry):
                restored += 1
        return restored

    def _restore_workspace_ui_state(self, ui_state):
        restored = 0
        if not isinstance(ui_state, dict):
            return restored
        windows = self._collect_workspace_windows_by_token()
        for raw_token, entries in ui_state.items():
            token = str(raw_token or "").strip()
            if not token:
                continue
            if token not in windows:
                legacy = f"{token}::0"
                if legacy in windows:
                    token = legacy
            window = windows.get(token)
            if window is None:
                continue
            restored += self._restore_window_ui_state(window, entries)
        return restored

    def eventFilter(self, _obj, event):
        if event is None:
            return super().eventFilter(_obj, event)
        event_type = event.type()
        if event_type in (QEvent.Type.WindowActivate, QEvent.Type.FocusIn, QEvent.Type.Show):
            token = self._token_for_workspace_window(_obj)
            if token:
                self._touch_workspace_window_order(token)
                if event_type in (QEvent.Type.WindowActivate, QEvent.Type.Show):
                    self._sync_takefits_menu_proxy(_obj)
        if event_type != QEvent.Type.KeyPress:
            return super().eventFilter(_obj, event)

        try:
            key = int(event.key())
            mods = event.modifiers()
        except Exception:
            return super().eventFilter(_obj, event)

        has_ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)
        has_meta = bool(mods & Qt.KeyboardModifier.MetaModifier)
        has_shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)
        has_alt = bool(mods & Qt.KeyboardModifier.AltModifier)
        if not (has_ctrl or has_meta) or has_alt:
            return super().eventFilter(_obj, event)

        back_combo = (key == Qt.Key.Key_Z) and (not has_shift)
        forward_combo = (key == Qt.Key.Key_Y) or ((key == Qt.Key.Key_Z) and has_shift)
        if not (back_combo or forward_combo):
            return super().eventFilter(_obj, event)
        if self._should_bypass_view_shortcuts():
            return super().eventFilter(_obj, event)

        try:
            handled = self.view_back() if back_combo else self.view_forward()
        except Exception:
            handled = False
        return True if handled else super().eventFilter(_obj, event)

    def _view_navigation_state_for_owner(self, owner):
        if owner is None:
            return (False, False)
        for getter_name in ("_view_history_state", "_shared_view_history_state"):
            getter = getattr(owner, getter_name, None)
            if not callable(getter):
                continue
            try:
                can_back, can_forward = getter()
                return (bool(can_back), bool(can_forward))
            except Exception:
                continue
        if owner is self:
            return self._shared_view_history_state()
        return self._view_history_toolbar_state(owner)

    def _refresh_view_navigation_actions(self):
        menu_bar = getattr(self, "menu_bar", None)
        owner = self._active_view_navigation_owner()
        can_back, can_forward = self._view_navigation_state_for_owner(owner)
        if menu_bar is not None and hasattr(menu_bar, "set_view_navigation_enabled"):
            try:
                menu_bar.set_view_navigation_enabled(can_back, can_forward)
            except Exception:
                pass
        shared_can_back, shared_can_forward = self._shared_view_history_state()
        self._sync_toolbar_view_navigation_state(shared_can_back, shared_can_forward)

    def _flush_pending_annotation_commits(self):
        if getattr(self, "_regions_commit_timer", None) is not None and self._regions_commit_timer.isActive():
            self._regions_commit_timer.stop()
            self._commit_regions_to_session()
        if getattr(self, "_markers_commit_timer", None) is not None and self._markers_commit_timer.isActive():
            self._markers_commit_timer.stop()
            self._commit_markers_to_session()

    def _utc_timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _session_records_up_to_cursor(self):
        session = getattr(self, "action_session", None)
        if session is None:
            return [], 0, 0
        history = list(getattr(session, "history", []) or [])
        total = len(history)
        try:
            cursor = int(getattr(session, "cursor", total))
        except Exception:
            cursor = total
        cursor = max(0, min(cursor, total))
        return history[:cursor], cursor, total

    def _serialize_action_records(self, records):
        payload = []
        for record in list(records or []):
            if record is None:
                continue
            if hasattr(record, "to_dict"):
                try:
                    payload.append(record.to_dict())
                    continue
                except Exception:
                    pass
            payload.append(
                {
                    "action": str(getattr(record, "action", "") or ""),
                    "params": dict(getattr(record, "params", {}) or {}),
                    "timestamp": str(getattr(record, "timestamp", "") or self._utc_timestamp()),
                    "tag": getattr(record, "tag", None),
                }
            )
        return payload

    def _write_json_payload(self, path: str, payload: dict) -> str:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return str(output_path)

    def _build_wcs_signature(self, wcs_obj=None) -> dict:
        wcs = wcs_obj if wcs_obj is not None else getattr(self, "wcs", None)
        signature = {"naxis": 0, "axes": [], "celestial_family": ""}
        if wcs is None:
            return signature
        try:
            naxis = int(getattr(wcs, "naxis", 0) or 0)
        except Exception:
            naxis = 0
        signature["naxis"] = max(0, naxis)
        wcs_core = getattr(wcs, "wcs", None)
        ctype_list = list(getattr(wcs_core, "ctype", []) or [])
        cunit_list = list(getattr(wcs_core, "cunit", []) or [])
        axis_types = []
        for idx in range(signature["naxis"]):
            raw_ctype = str(ctype_list[idx]) if idx < len(ctype_list) else ""
            axis_type = normalize_wcs_axis_type(raw_ctype)
            raw_unit = ""
            if idx < len(cunit_list):
                try:
                    raw_unit = str(cunit_list[idx] or "")
                except Exception:
                    raw_unit = ""
            unit = normalize_wcs_unit(raw_unit)
            axis_types.append(axis_type)
            signature["axes"].append(
                {
                    "ctype": raw_ctype,
                    "axis_type": axis_type,
                    "unit": unit,
                }
            )
        if "RA" in axis_types and "DEC" in axis_types:
            signature["celestial_family"] = "equatorial"
        elif "GLON" in axis_types and "GLAT" in axis_types:
            signature["celestial_family"] = "galactic"
        return signature

    def _dataset_descriptor(self) -> dict:
        filepath = str(getattr(self, "filename_path", "") or "")
        shape = []
        try:
            shape = [int(v) for v in getattr(self.data, "shape", [])]
        except Exception:
            shape = []
        return {
            "filepath": os.path.abspath(filepath) if filepath else "",
            "filename": os.path.basename(filepath) if filepath else str(getattr(self, "filename", "") or ""),
            "data_shape": shape,
            "wcs_naxis": int(getattr(self.wcs, "naxis", 0) or 0),
            "wcs_signature": self._build_wcs_signature(getattr(self, "wcs", None)),
        }

    def _viewer_for_plane(self, plane: str):
        base = str(plane or "").lower()
        if base == "xy":
            return self
        for viewer in list(getattr(self, "subwindows", []) or []):
            if str(getattr(viewer, "plane", "")).lower() == base:
                return viewer
        return None

    def _iter_view_history_viewers(self):
        return [self] + [v for v in list(getattr(self, "subwindows", []) or []) if v is not None]

    def _begin_view_history_batch(self):
        depth = int(getattr(self, "_view_history_batch_depth", 0))
        self._view_history_batch_depth = depth + 1

    def _end_view_history_batch(self):
        depth = int(getattr(self, "_view_history_batch_depth", 0))
        self._view_history_batch_depth = max(0, depth - 1)

    def _shared_view_history_state(self):
        history = list(getattr(self, "_shared_view_history", []) or [])
        index = int(getattr(self, "_shared_view_history_index", -1))
        total = len(history)
        can_back = total > 0 and index > 0
        can_forward = total > 0 and 0 <= index < (total - 1)
        return can_back, can_forward

    def _capture_shared_view_limits_snapshot(self):
        snapshot = {}
        for plane in ("xy", "xz", "zy"):
            viewer = self._viewer_for_plane(plane)
            if viewer is None or not hasattr(viewer, "ax"):
                continue
            try:
                xlim = viewer.ax.get_xlim()
                ylim = viewer.ax.get_ylim()
                snapshot[plane] = {
                    "xlim": [float(xlim[0]), float(xlim[1])],
                    "ylim": [float(ylim[0]), float(ylim[1])],
                }
            except Exception:
                continue
        snapshot = self._fill_missing_plane_limits(snapshot)
        return snapshot if snapshot else None

    def _normalize_shared_color_history_settings(self, settings=None, fallback=None):
        normalized = self._normalize_color_panel_settings(settings, fallback=fallback)

        for key in ("min_val", "max_val"):
            value = normalized.get(key)
            if value is None:
                continue
            try:
                number = float(value)
                normalized[key] = number if math.isfinite(number) else None
            except Exception:
                normalized[key] = None

        try:
            gamma_value = float(normalized.get("gamma_value", 1.0) or 1.0)
        except Exception:
            gamma_value = 1.0
        if not math.isfinite(gamma_value) or gamma_value <= 0:
            gamma_value = 1.0
        normalized["gamma_value"] = gamma_value
        normalized["log_scale"] = bool(normalized.get("log_scale", False))
        normalized["invert"] = bool(normalized.get("invert", False))

        pattern = str(normalized.get("color_pattern") or "").strip()
        if pattern.endswith("_r"):
            pattern = pattern[:-2]
            normalized["invert"] = True
        if not pattern:
            display_pattern = str(getattr(getattr(self, "displaymap", None), "colorscale", "") or "").strip()
            if display_pattern.endswith("_r"):
                display_pattern = display_pattern[:-2]
                normalized["invert"] = True
            if display_pattern:
                pattern = display_pattern
        normalized["color_pattern"] = pattern or None
        return normalized

    def _capture_shared_color_history_snapshot(self):
        fallback = {}
        try:
            global_settings = self._capture_global_color_settings()
            if isinstance(global_settings, dict):
                candidate = global_settings.get("main")
                if isinstance(candidate, dict):
                    fallback = dict(candidate)
        except Exception:
            fallback = {}

        panel = getattr(getattr(self, "control_panel", None), "color_settings_panel", None)
        live = self._extract_live_color_panel_settings(panel)
        normalized = self._normalize_shared_color_history_settings(live, fallback=fallback)
        normalized = self._derive_panel_settings_from_image(getattr(self, "im", None), fallback=normalized)
        normalized = self._normalize_shared_color_history_settings(normalized, fallback=fallback)
        return normalized

    def _shared_color_signature(self, snapshot):
        if not isinstance(snapshot, dict):
            return tuple()
        normalized = self._normalize_shared_color_history_settings(snapshot)

        def _round_or_none(value, digits):
            try:
                number = float(value)
            except Exception:
                return None
            if not math.isfinite(number):
                return None
            return round(number, digits)

        return (
            str(normalized.get("color_pattern") or ""),
            bool(normalized.get("invert", False)),
            bool(normalized.get("log_scale", False)),
            _round_or_none(normalized.get("gamma_value"), 6),
            _round_or_none(normalized.get("min_val"), 9),
            _round_or_none(normalized.get("max_val"), 9),
        )

    def _shared_view_entry_signature(self, limit_snapshot, color_snapshot):
        return {
            "limits": self._shared_view_limits_signature(limit_snapshot),
            "color": self._shared_color_signature(color_snapshot),
        }

    def _normalize_plane_limits(self, limits):
        if not isinstance(limits, dict):
            return None
        xlim = limits.get("xlim")
        ylim = limits.get("ylim")
        if not (
            isinstance(xlim, (list, tuple))
            and len(xlim) == 2
            and isinstance(ylim, (list, tuple))
            and len(ylim) == 2
        ):
            return None
        try:
            x0, x1 = float(xlim[0]), float(xlim[1])
            y0, y1 = float(ylim[0]), float(ylim[1])
        except Exception:
            return None
        return {"xlim": [x0, x1], "ylim": [y0, y1]}

    def _fill_missing_plane_limits(self, snapshot):
        if not isinstance(snapshot, dict):
            return {}

        normalized = {}
        for plane in ("xy", "xz", "zy"):
            limits = self._normalize_plane_limits(snapshot.get(plane))
            if limits is not None:
                normalized[plane] = limits

        if getattr(self, "data", None) is None or getattr(self.data, "ndim", 0) <= 2:
            return normalized

        changed = True
        while changed:
            changed = False
            xy = normalized.get("xy")
            xz = normalized.get("xz")
            zy = normalized.get("zy")

            # XY(x,y), XZ(x,z), ZY(z,y)
            if zy is None and xy is not None and xz is not None:
                normalized["zy"] = {
                    "xlim": [float(xz["ylim"][0]), float(xz["ylim"][1])],
                    "ylim": [float(xy["ylim"][0]), float(xy["ylim"][1])],
                }
                changed = True
                continue

            if xz is None and xy is not None and zy is not None:
                normalized["xz"] = {
                    "xlim": [float(xy["xlim"][0]), float(xy["xlim"][1])],
                    "ylim": [float(zy["xlim"][0]), float(zy["xlim"][1])],
                }
                changed = True
                continue

            if xy is None and xz is not None and zy is not None:
                normalized["xy"] = {
                    "xlim": [float(xz["xlim"][0]), float(xz["xlim"][1])],
                    "ylim": [float(zy["ylim"][0]), float(zy["ylim"][1])],
                }
                changed = True

        return normalized

    def _shared_view_limits_signature(self, snapshot):
        signature = []
        if not isinstance(snapshot, dict):
            return tuple(signature)
        for plane in ("xy", "xz", "zy"):
            limits = snapshot.get(plane)
            if not isinstance(limits, dict):
                continue
            xlim = limits.get("xlim")
            ylim = limits.get("ylim")
            if not (
                isinstance(xlim, (list, tuple))
                and len(xlim) == 2
                and isinstance(ylim, (list, tuple))
                and len(ylim) == 2
            ):
                continue
            try:
                x0, x1 = float(xlim[0]), float(xlim[1])
                y0, y1 = float(ylim[0]), float(ylim[1])
            except Exception:
                continue
            signature.append(
                (
                    plane,
                    round(x0, 9),
                    round(x1, 9),
                    round(y0, 9),
                    round(y1, 9),
                )
            )
        return tuple(signature)

    def _record_shared_view_history(self, reason: str = "", *, force: bool = False):
        if bool(getattr(self, "_suspend_view_history_recording", False)):
            return False
        if int(getattr(self, "_view_history_batch_depth", 0)) > 0 and not force:
            return False

        snapshot = self._capture_shared_view_limits_snapshot()
        if snapshot is None:
            return False
        snapshot = self._fill_missing_plane_limits(snapshot)
        color_snapshot = self._capture_shared_color_history_snapshot()
        signature = self._shared_view_entry_signature(snapshot, color_snapshot)
        history = list(getattr(self, "_shared_view_history", []) or [])
        index = int(getattr(self, "_shared_view_history_index", -1))

        if not force and history and 0 <= index < len(history):
            current_signature = history[index].get("signature")
            if current_signature == signature:
                self._refresh_view_navigation_actions()
                return False

        if 0 <= index < len(history) - 1:
            history = history[: index + 1]

        history.append(
            {
                "limits": snapshot,
                "color": color_snapshot,
                "signature": signature,
                "reason": str(reason or ""),
            }
        )

        max_entries = 200
        if len(history) > max_entries:
            history = history[-max_entries:]

        self._shared_view_history = history
        self._shared_view_history_index = len(history) - 1
        self._refresh_view_navigation_actions()
        return True

    def _reset_shared_view_history_to_current(self, reason: str = "") -> bool:
        snapshot = self._capture_shared_view_limits_snapshot()
        if snapshot is None:
            return False
        snapshot = self._fill_missing_plane_limits(snapshot)
        color_snapshot = self._capture_shared_color_history_snapshot()
        signature = self._shared_view_entry_signature(snapshot, color_snapshot)
        self._shared_view_history = [
            {
                "limits": snapshot,
                "color": color_snapshot,
                "signature": signature,
                "reason": str(reason or ""),
            }
        ]
        self._shared_view_history_index = 0
        self._refresh_view_navigation_actions()
        return True

    def _apply_shared_view_history_entry(self, entry) -> bool:
        if not isinstance(entry, dict):
            return False
        snapshot = entry.get("limits")
        color_snapshot = entry.get("color")
        if isinstance(snapshot, dict):
            snapshot = self._fill_missing_plane_limits(snapshot)
        else:
            snapshot = {}
        if not snapshot and not isinstance(color_snapshot, dict):
            return False

        applied = False
        color_applied = False
        self._suspend_view_history_recording = True
        self._begin_view_history_batch()
        try:
            if snapshot:
                for plane in ("xy", "xz", "zy"):
                    limits = snapshot.get(plane)
                    if not isinstance(limits, dict):
                        continue
                    viewer = self._viewer_for_plane(plane)
                    if viewer is None or not hasattr(viewer, "ax"):
                        continue
                    xlim = limits.get("xlim")
                    ylim = limits.get("ylim")
                    if not (
                        isinstance(xlim, (list, tuple))
                        and len(xlim) == 2
                        and isinstance(ylim, (list, tuple))
                        and len(ylim) == 2
                    ):
                        continue
                    try:
                        x0, x1 = float(xlim[0]), float(xlim[1])
                        y0, y1 = float(ylim[0]), float(ylim[1])
                    except Exception:
                        continue

                    try:
                        viewer.ax.set_xlim(x0, x1)
                        viewer.ax.set_ylim(y0, y1)
                        if hasattr(viewer, "overlay_ax") and viewer.overlay_ax is not None:
                            viewer.overlay_ax.set_position(viewer.ax.get_position())
                        suspend_regions = getattr(viewer, "_suspend_regions_for_full_draw", None)
                        if callable(suspend_regions):
                            suspend_regions()
                        if hasattr(viewer, "canvas") and viewer.canvas is not None:
                            self._request_canvas_redraw(viewer.canvas)
                        self.update_ranges(plane, (x0, x1), (y0, y1))
                        applied = True
                    except Exception:
                        continue

                if applied:
                    if self.data.ndim > 2:
                        self._sync_range_panel_inputs("xy", "xz", "zy")
                    else:
                        self._sync_range_panel_inputs("xy")

            if isinstance(color_snapshot, dict):
                color_applied = bool(self._apply_shared_color_history_snapshot(color_snapshot))
        finally:
            self._end_view_history_batch()
            self._suspend_view_history_recording = False
        return applied or color_applied

    def _sync_main_color_panel_widgets(self, panel, settings):
        if panel is None or not isinstance(settings, dict):
            return

        merged = dict(getattr(panel, "current_settings", {}) or {})
        merged.update(settings)
        merged = self._normalize_shared_color_history_settings(merged)
        try:
            from takefits.tools.color_scale import ColorMode, ColorSettingsPanel

            ColorSettingsPanel.settings[ColorMode.MAIN] = dict(merged)
            panel.current_settings = ColorSettingsPanel.settings[ColorMode.MAIN]
        except Exception:
            panel.current_settings = dict(merged)

        blockers = []
        for widget in (
            getattr(panel, "colorscale_combo", None),
            getattr(panel, "invert_checkbox", None),
            getattr(panel, "log_checkbox", None),
            getattr(panel, "gamma_spinbox", None),
            getattr(panel, "intensity_min", None),
            getattr(panel, "intensity_max", None),
        ):
            if widget is None:
                continue
            try:
                blockers.append(QSignalBlocker(widget))
            except Exception:
                continue

        min_val = merged.get("min_val")
        max_val = merged.get("max_val")
        gamma = float(merged.get("gamma_value", 1.0) or 1.0)
        pattern = str(merged.get("color_pattern") or "").strip()
        invert = bool(merged.get("invert", False))
        log_scale = bool(merged.get("log_scale", False))

        try:
            if getattr(panel, "intensity_min", None) is not None:
                panel.intensity_min.setText("" if min_val is None else f"{float(min_val):.3g}")
        except Exception:
            pass
        try:
            if getattr(panel, "intensity_max", None) is not None:
                panel.intensity_max.setText("" if max_val is None else f"{float(max_val):.3g}")
        except Exception:
            pass
        try:
            if getattr(panel, "gamma_spinbox", None) is not None:
                panel.gamma_spinbox.setValue(gamma)
        except Exception:
            pass
        try:
            if getattr(panel, "invert_checkbox", None) is not None:
                panel.invert_checkbox.setChecked(invert)
        except Exception:
            pass
        try:
            if getattr(panel, "log_checkbox", None) is not None:
                panel.log_checkbox.setChecked(log_scale)
        except Exception:
            pass
        try:
            if pattern and getattr(panel, "colorscale_combo", None) is not None:
                panel.colorscale_combo.setCurrentText(pattern)
        except Exception:
            pass

        panel.color_pattern = pattern or getattr(panel, "color_pattern", None)

        try:
            if getattr(panel, "auto_button", None) is not None:
                panel.auto_button.setEnabled(not log_scale)
            if getattr(panel, "min_max_button", None) is not None:
                panel.min_max_button.setEnabled(not log_scale)
        except Exception:
            pass

        try:
            if (
                min_val is not None
                and max_val is not None
                and getattr(panel, "min_line", None) is not None
                and getattr(panel, "max_line", None) is not None
            ):
                panel.update_histogram_lines(float(min_val), float(max_val))
        except Exception:
            pass
        try:
            if getattr(panel, "canvas", None) is not None:
                self._request_canvas_redraw(panel.canvas)
        except Exception:
            pass

    def _apply_shared_color_history_snapshot(self, settings):
        if not isinstance(settings, dict):
            return False

        normalized = self._normalize_shared_color_history_settings(settings)
        if not normalized:
            return False

        pattern = str(normalized.get("color_pattern") or "").strip()
        if not pattern:
            return False

        try:
            from takefits.tools.color_scale import ColorMode, ColorSettingsPanel
        except Exception:
            return False

        ColorSettingsPanel.settings[ColorMode.MAIN] = dict(normalized)
        self._color_panel_hint_main = dict(normalized)

        display_pattern = pattern
        if bool(normalized.get("invert")) and not display_pattern.endswith("_r"):
            display_pattern = f"{display_pattern}_r"

        try:
            if hasattr(self, "displaymap") and self.displaymap is not None:
                self.displaymap.colorscale = display_pattern
        except Exception:
            pass
        for viewer in list(getattr(self, "subwindows", []) or []):
            if viewer is None:
                continue
            try:
                if hasattr(viewer, "displaymap") and viewer.displaymap is not None:
                    viewer.displaymap.colorscale = display_pattern
            except Exception:
                continue

        self._apply_color_state_to_mode("main", normalized)

        panel = getattr(getattr(self, "control_panel", None), "color_settings_panel", None)
        if panel is not None and getattr(panel, "mode", None) == ColorMode.MAIN:
            self._sync_main_color_panel_widgets(panel, normalized)
        return True

    def _sync_toolbar_view_navigation_state(self, can_back: bool, can_forward: bool):
        for viewer in self._iter_view_history_viewers():
            toolbar = getattr(viewer, "toolbar", None)
            setter = getattr(toolbar, "set_external_history_state", None)
            if callable(setter):
                try:
                    setter(can_back, can_forward)
                except Exception:
                    continue

    def get_wcs_display_frame(self) -> str:
        preferred = preferred_display_frame(getattr(self, "wcs", None))
        frame = self._get_shared_display_frame()
        frame = normalize_display_frame(frame)
        if preferred != "native" and frame == "native":
            return preferred
        if not frame_is_available(getattr(self, "wcs", None), frame):
            return preferred
        return frame

    def get_wcs_decimal_mode(self) -> bool:
        try:
            return bool(getattr(self, "decimal", True))
        except Exception:
            return True

    def set_wcs_decimal_mode(self, use_decimal: bool, *, refresh: bool = True):
        decimal = bool(use_decimal)
        for viewer in [self] + list(getattr(self, "subwindows", []) or []):
            if viewer is None:
                continue
            try:
                viewer.decimal = decimal
            except Exception:
                pass
            try:
                format_pix = getattr(viewer, "format_pix", None)
                if format_pix is not None:
                    format_pix.decimal = decimal
            except Exception:
                pass
            try:
                config_mgr = getattr(viewer, "config_manager", None)
                if config_mgr is not None and hasattr(config_mgr, "config"):
                    config_mgr.config["decimal"] = decimal
            except Exception:
                pass
        for window in self._live_integration_windows():
            try:
                window.decimal = decimal
            except Exception:
                pass
            try:
                format_pix = getattr(window, "format_pix", None)
                if format_pix is not None:
                    format_pix.decimal = decimal
            except Exception:
                pass
        for window in list(getattr(self, "channel_map_windows", []) or []):
            if window is None:
                continue
            try:
                window.decimal = decimal
            except Exception:
                pass
            try:
                format_pix = getattr(window, "format_pix", None)
                if format_pix is not None:
                    format_pix.decimal = decimal
            except Exception:
                pass
        menu_bar = getattr(self, "menu_bar", None)
        if menu_bar is not None and hasattr(menu_bar, "set_wcs_decimal_checked"):
            try:
                menu_bar.set_wcs_decimal_checked(decimal)
            except Exception:
                pass
        if refresh:
            try:
                self.refresh_coordinate_format()
            except Exception:
                pass
            for viewer in list(getattr(self, "subwindows", []) or []):
                if viewer is None:
                    continue
                try:
                    viewer.refresh_coordinate_format()
                except Exception:
                    pass
            self._refresh_wcs_display_strings()

    def _shared_world_vector(self):
        values = []
        for getter in (
            getattr(self, "_get_shared_world_x", None),
            getattr(self, "_get_shared_world_y", None),
            getattr(self, "_get_shared_world_z", None),
            getattr(self, "_get_shared_world_s", None),
        ):
            if not callable(getter):
                values.append(None)
                continue
            try:
                values.append(float(getter()))
            except Exception:
                values.append(None)
        return values

    def _plane_cursor_pixel(self, plane: str):
        key = str(plane or "").lower()
        if key == "xy":
            return (float(self._get_shared_xpix()), float(self._get_shared_ypix()))
        if key == "xz":
            return (float(self._get_shared_xpix()), float(self._get_shared_zpix()))
        if key == "zy":
            return (float(self._get_shared_zpix()), float(self._get_shared_ypix()))
        return None

    @staticmethod
    def _capture_range_texts(target, field_names):
        values = {}
        if target is None:
            return values
        for field_name in field_names:
            widget = getattr(target, field_name, None)
            if widget is None or not hasattr(widget, "text"):
                continue
            try:
                values[field_name] = widget.text()
            except Exception:
                continue
        return values

    @staticmethod
    def _restore_range_texts(target, values):
        if target is None or not isinstance(values, dict):
            return
        for field_name, text in values.items():
            widget = getattr(target, field_name, None)
            if widget is None or not hasattr(widget, "setText"):
                continue
            try:
                widget.setText(str(text))
            except Exception:
                continue

    def _snapshot_wcs_range_inputs(self):
        snapshot = {}
        snapshot["xy"] = self._capture_range_texts(
            self,
            ("x_min_input", "x_max_input", "y_min_input", "y_max_input"),
        )

        xz_viewer = self._viewer_for_plane("xz")
        snapshot["xz"] = self._capture_range_texts(
            xz_viewer,
            ("x_min_input", "x_max_input", "z_min_input", "z_max_input"),
        )

        zy_viewer = self._viewer_for_plane("zy")
        snapshot["zy"] = self._capture_range_texts(
            zy_viewer,
            ("z_min_input", "z_max_input", "y_min_input", "y_max_input"),
        )

        snapshot["range_panel"] = self._capture_range_texts(
            getattr(self, "range_panel", None),
            ("x_min_input", "x_max_input", "y_min_input", "y_max_input", "z_min_input", "z_max_input"),
        )
        return snapshot

    def _restore_wcs_range_inputs(self, snapshot):
        if not isinstance(snapshot, dict):
            return
        self._restore_range_texts(self, snapshot.get("xy"))
        self._restore_range_texts(self._viewer_for_plane("xz"), snapshot.get("xz"))
        self._restore_range_texts(self._viewer_for_plane("zy"), snapshot.get("zy"))
        self._restore_range_texts(getattr(self, "range_panel", None), snapshot.get("range_panel"))

    def _refresh_wcs_display_strings(self):
        for plane in ("xy", "xz", "zy"):
            viewer = self._viewer_for_plane(plane)
            if viewer is None:
                continue
            refresh_chval = getattr(viewer, "refresh_channel_value_display", None)
            if callable(refresh_chval):
                try:
                    refresh_chval()
                except Exception:
                    pass

        plabel_xy = self._get_plane_plabel("xy")
        plabel_xz = self._get_plane_plabel("xz")
        plabel_zy = self._get_plane_plabel("zy")
        xy_text = self._formatted_plane_cursor_text("xy")
        xz_text = self._formatted_plane_cursor_text("xz")
        zy_text = self._formatted_plane_cursor_text("zy")
        intensity_text = self._shared_intensity_text()
        if plabel_xy:
            coord_text = xy_text if xy_text is not None else self._format_cursor_pair_text("xy", self._get_shared_world_x_str(), self._get_shared_world_y_str())
            plabel_xy.setText(self._compose_click_label_text(coord_text, intensity_text))
        if plabel_xz:
            coord_text = xz_text if xz_text is not None else self._format_cursor_pair_text("xz", self._get_shared_world_x_str(), self._get_shared_world_z_str())
            plabel_xz.setText(self._compose_click_label_text(coord_text, intensity_text))
        if plabel_zy:
            coord_text = zy_text if zy_text is not None else self._format_cursor_pair_text("zy", self._get_shared_world_z_str(), self._get_shared_world_y_str())
            plabel_zy.setText(self._compose_click_label_text(coord_text, intensity_text))

        for plane in ("xy", "xz", "zy"):
            viewer = self._viewer_for_plane(plane)
            canvas = getattr(viewer, "canvas", None) if viewer is not None else None
            if canvas is not None:
                try:
                    self._request_canvas_redraw(canvas)
                except Exception:
                    pass

        for window in self._live_integration_windows():
            refresh_display = getattr(window, "refresh_coordinate_display", None)
            if callable(refresh_display):
                try:
                    refresh_display()
                except Exception:
                    pass
            canvas = getattr(window, "canvas", None)
            if canvas is not None:
                try:
                    self._request_canvas_redraw(canvas)
                except Exception:
                    pass

        for window in list(getattr(self, "channel_map_windows", []) or []):
            if window is None:
                continue
            refresh_display = getattr(window, "refresh_coordinate_display", None)
            if callable(refresh_display):
                try:
                    refresh_display()
                except Exception:
                    pass
            canvas = getattr(window, "canvas", None)
            if canvas is not None:
                try:
                    self._request_canvas_redraw(canvas)
                except Exception:
                    pass

        control_panel = getattr(self, "control_panel", None)
        spectrum_panel = getattr(control_panel, "spec_window", None) if control_panel is not None else None
        refresh_spectrum = getattr(spectrum_panel, "refresh_coordinate_display", None)
        if callable(refresh_spectrum):
            try:
                refresh_spectrum()
            except Exception:
                pass
        baseline_panel = getattr(control_panel, "baseline_panel", None) if control_panel is not None else None
        refresh_baseline = getattr(baseline_panel, "refresh_coordinate_display", None)
        if callable(refresh_baseline):
            try:
                refresh_baseline()
            except Exception:
                pass

    def set_wcs_display_frame(self, frame: str, *, refresh: bool = True):
        preferred = preferred_display_frame(getattr(self, "wcs", None))
        normalized = normalize_display_frame(frame)
        if preferred != "native" and normalized == "native":
            normalized = preferred
        if not frame_is_available(getattr(self, "wcs", None), normalized):
            normalized = preferred
        current = normalize_display_frame(self._get_shared_display_frame())
        self._set_shared_display_frame(normalized)
        menu_bar = getattr(self, "menu_bar", None)
        if menu_bar is not None and hasattr(menu_bar, "set_wcs_frame_checked"):
            try:
                menu_bar.set_wcs_frame_checked(normalized)
            except Exception:
                pass
        if refresh and current != normalized:
            range_snapshot = self._snapshot_wcs_range_inputs()
            self._refresh_wcs_display_strings()
            self._restore_wcs_range_inputs(range_snapshot)

    def _capture_view_limits(self) -> dict:
        payload = {}
        for plane in ("xy", "xz", "zy"):
            viewer = self._viewer_for_plane(plane)
            if viewer is None or not hasattr(viewer, "ax"):
                continue
            try:
                xlim = viewer.ax.get_xlim()
                ylim = viewer.ax.get_ylim()
                payload[plane] = {
                    "xlim": [float(xlim[0]), float(xlim[1])],
                    "ylim": [float(ylim[0]), float(ylim[1])],
                    "visible": bool(viewer.isVisible()),
                }
            except Exception:
                continue
        return payload

    def _capture_window_geometry(self, window):
        if window is None:
            return None
        try:
            geo = window.geometry()
            return {
                "x": int(geo.x()),
                "y": int(geo.y()),
                "w": int(geo.width()),
                "h": int(geo.height()),
                "maximized": bool(window.isMaximized()),
                "fullscreen": bool(window.isFullScreen()),
            }
        except Exception:
            return None

    def _restore_window_geometry(self, window, geometry) -> bool:
        if window is None or not isinstance(geometry, dict):
            return False
        try:
            was_visible = bool(window.isVisible())
        except Exception:
            was_visible = True
        try:
            x = int(geometry.get("x"))
            y = int(geometry.get("y"))
            w = int(geometry.get("w"))
            h = int(geometry.get("h"))
        except Exception:
            x = y = w = h = None
        try:
            if all(v is not None for v in (x, y, w, h)) and w > 0 and h > 0:
                window.setGeometry(x, y, w, h)
            if was_visible:
                if bool(geometry.get("fullscreen", False)):
                    window.showFullScreen()
                elif bool(geometry.get("maximized", False)):
                    window.showMaximized()
                elif window.isMaximized() or window.isFullScreen():
                    window.showNormal()
            return True
        except Exception:
            return False

    def _collect_window_axes(self, window):
        axes = []
        if window is None:
            return axes
        primary = getattr(window, "ax", None)
        if primary is not None:
            axes.append(primary)
        for candidate in list(getattr(window, "axes", []) or []):
            if candidate is None:
                continue
            if candidate in axes:
                continue
            axes.append(candidate)
        return axes

    def _capture_window_range_state(self, window):
        if window is None:
            return None

        def _read_axis_inputs(axis_name: str):
            attr_pairs = (
                (f"{axis_name}_min_int_input", f"{axis_name}_max_int_input"),
                (f"{axis_name}_min_ch_input", f"{axis_name}_max_ch_input"),
            )
            for min_attr, max_attr in attr_pairs:
                min_widget = getattr(window, min_attr, None)
                max_widget = getattr(window, max_attr, None)
                if min_widget is None or max_widget is None:
                    continue
                min_text = str(min_widget.text() or "").strip()
                max_text = str(max_widget.text() or "").strip()
                if not min_text or not max_text:
                    continue
                return {"min": min_text, "max": max_text}
            return None

        payload = {}

        axis_inputs = {}
        for axis_name in ("x", "y", "z"):
            entry = _read_axis_inputs(axis_name)
            if entry is not None:
                axis_inputs[axis_name] = entry
        if axis_inputs:
            payload["inputs"] = axis_inputs

        limits = None
        axes = self._collect_window_axes(window)
        if axes:
            try:
                xlim = axes[0].get_xlim()
                ylim = axes[0].get_ylim()
                limits = {
                    "xlim": [float(xlim[0]), float(xlim[1])],
                    "ylim": [float(ylim[0]), float(ylim[1])],
                }
            except Exception:
                limits = None
        if limits is not None:
            payload["axis_limits"] = limits

        cursor_payload = None
        click_vline = getattr(window, "click_v_line", None)
        click_hline = getattr(window, "click_h_line", None)
        if click_vline is not None and click_hline is not None:
            cursor_payload = {}
            try:
                cursor_payload["visible"] = bool(click_vline.get_visible() or click_hline.get_visible())
            except Exception:
                pass
            try:
                xdata = click_vline.get_xdata()
                if xdata is not None and len(xdata):
                    cursor_payload["x"] = float(xdata[0])
            except Exception:
                pass
            try:
                ydata = click_hline.get_ydata()
                if ydata is not None and len(ydata):
                    cursor_payload["y"] = float(ydata[0])
            except Exception:
                pass
            label = getattr(window, "label", None)
            if label is not None:
                try:
                    cursor_payload["label_visible"] = bool(label.isVisible())
                except Exception:
                    pass
                try:
                    text = str(label.text() or "")
                    if text:
                        cursor_payload["label_text"] = text
                except Exception:
                    pass
            if cursor_payload:
                payload["cursor_line"] = cursor_payload

        return payload if payload else None

    def _restore_window_range_state(self, window, range_state, *, allow_axis_limits: bool = True) -> bool:
        if window is None or not isinstance(range_state, dict):
            return False

        applied = False
        inputs = range_state.get("inputs")
        input_applied = False
        if isinstance(inputs, dict):
            restored_axes = set()
            for axis_name in ("x", "y", "z"):
                entry = inputs.get(axis_name)
                if not isinstance(entry, dict):
                    continue
                min_text = str(entry.get("min", "")).strip()
                max_text = str(entry.get("max", "")).strip()
                if not min_text or not max_text:
                    continue
                for min_attr, max_attr in (
                    (f"{axis_name}_min_int_input", f"{axis_name}_max_int_input"),
                    (f"{axis_name}_min_ch_input", f"{axis_name}_max_ch_input"),
                ):
                    min_widget = getattr(window, min_attr, None)
                    max_widget = getattr(window, max_attr, None)
                    if min_widget is None or max_widget is None:
                        continue
                    min_widget.setText(min_text)
                    max_widget.setText(max_text)
                    restored_axes.add(axis_name)
            if restored_axes:
                for axis_name, method_name in (("x", "set_x_range"), ("y", "set_y_range"), ("z", "set_z_range")):
                    if axis_name not in restored_axes:
                        continue
                    method = getattr(window, method_name, None)
                    if not callable(method):
                        continue
                    previous_suppress = bool(getattr(window, "_suppress_range_warning", False))
                    setattr(window, "_suppress_range_warning", True)
                    try:
                        method()
                        applied = True
                        input_applied = True
                    except Exception:
                        continue
                    finally:
                        setattr(window, "_suppress_range_warning", previous_suppress)

        # Pixel limits are dataset-dependent. Keep them as a fallback only when
        # world-input based restoration was not applicable.
        if allow_axis_limits and not input_applied:
            limits = range_state.get("axis_limits")
            if isinstance(limits, dict):
                xlim = limits.get("xlim")
                ylim = limits.get("ylim")
                if (
                    isinstance(xlim, (list, tuple))
                    and len(xlim) == 2
                    and isinstance(ylim, (list, tuple))
                    and len(ylim) == 2
                ):
                    try:
                        x0, x1 = float(xlim[0]), float(xlim[1])
                        y0, y1 = float(ylim[0]), float(ylim[1])
                        for axis in self._collect_window_axes(window):
                            axis.set_xlim(x0, x1)
                            axis.set_ylim(y0, y1)
                        invalidate_workspace_restore_blit_cache(window)
                        canvas = getattr(window, "canvas", None)
                        if canvas is not None:
                            self._request_canvas_redraw(canvas)
                        applied = True
                    except Exception:
                        pass

        cursor_state = range_state.get("cursor_line")
        if isinstance(cursor_state, dict):
            click_vline = getattr(window, "click_v_line", None)
            click_hline = getattr(window, "click_h_line", None)
            cursor_applied = False
            if click_vline is not None and click_hline is not None:
                try:
                    if "x" in cursor_state:
                        xpos = float(cursor_state.get("x"))
                        click_vline.set_data([xpos, xpos], [0, 1])
                        cursor_applied = True
                except Exception:
                    pass
                try:
                    if "y" in cursor_state:
                        ypos = float(cursor_state.get("y"))
                        click_hline.set_data([0, 1], [ypos, ypos])
                        cursor_applied = True
                except Exception:
                    pass
                try:
                    if "visible" in cursor_state:
                        visible = bool(cursor_state.get("visible", False))
                        click_vline.set_visible(visible)
                        click_hline.set_visible(visible)
                        cursor_applied = True
                except Exception:
                    pass

            label = getattr(window, "label", None)
            if label is not None:
                try:
                    if "label_text" in cursor_state:
                        label.setText(str(cursor_state.get("label_text") or ""))
                        cursor_applied = True
                except Exception:
                    pass
                try:
                    if "label_visible" in cursor_state:
                        label.setVisible(bool(cursor_state.get("label_visible", False)))
                        cursor_applied = True
                except Exception:
                    pass

            if cursor_applied:
                redraw_overlay = getattr(window, "redraw_main_overlay_and_blit", None)
                if callable(redraw_overlay):
                    try:
                        redraw_overlay()
                    except Exception:
                        pass
                else:
                    canvas = getattr(window, "canvas", None)
                    if canvas is not None:
                        try:
                            self._request_canvas_redraw(canvas)
                        except Exception:
                            pass
                applied = True

        return applied

    def _capture_workspace_geometry_state(self) -> dict:
        state = {}
        state["main_window"] = self._capture_window_geometry(self)
        state["control_panel"] = self._capture_window_geometry(getattr(self, "control_panel", None))
        state["range_panel"] = self._capture_window_geometry(getattr(self, "range_panel", None))
        state["marker_panel"] = self._capture_window_geometry(getattr(self, "marker_panel", None))
        state["regrid_panel"] = self._capture_window_geometry(getattr(self, "_regrid_panel", None))

        if self.data.ndim > 2:
            state["subwindows"] = {
                "xz": self._capture_window_geometry(getattr(self, "subwindow1", None)),
                "zy": self._capture_window_geometry(getattr(self, "subwindow2", None)),
            }

        control_panel = getattr(self, "control_panel", None)
        tool_geometries = {}
        if control_panel is not None:
            for attr in (
                "color_settings_panel",
                "scaling_panel",
                "unit_conversion_panel",
                "integ_settings_panel",
                "chmap_settings_panel",
                "smooth_settings_panel",
                "baseline_panel",
                "spec_window",
                "pvd_panel",
                "mask_settings_panel",
                "contour_panel",
                "arithmetic_panel",
                "clump_finding_panel",
            ):
                geometry = self._capture_window_geometry(getattr(control_panel, attr, None))
                if geometry is not None:
                    tool_geometries[attr] = geometry
        state["tool_panels"] = tool_geometries

        integration_windows = []
        for window in self._live_integration_windows():
            geometry = self._capture_window_geometry(window)
            if geometry is None:
                continue
            integration_windows.append(
                {
                    "key": self._integration_window_color_key(window),
                    "geometry": geometry,
                    "range_state": self._capture_window_range_state(window),
                    "marker_panel": self._capture_marker_panel_state_for_window(window),
                }
            )
        state["integration_windows"] = integration_windows

        channel_windows = []
        for window in list(getattr(self, "channel_map_windows", []) or []):
            if window is None:
                continue
            geometry = self._capture_window_geometry(window)
            if geometry is None:
                continue
            channel_windows.append(
                {
                    "key": self._channel_window_color_key(window),
                    "geometry": geometry,
                    "range_state": self._capture_window_range_state(window),
                    "marker_panel": self._capture_marker_panel_state_for_window(window),
                }
            )
        state["channel_windows"] = channel_windows
        return state

    def _restore_workspace_geometry_state(self, geometry_state, *, allow_window_axis_limits: bool = True):
        if not isinstance(geometry_state, dict):
            return

        self._restore_window_geometry(self, geometry_state.get("main_window"))
        self._restore_window_geometry(getattr(self, "control_panel", None), geometry_state.get("control_panel"))
        self._restore_window_geometry(getattr(self, "range_panel", None), geometry_state.get("range_panel"))
        self._restore_window_geometry(getattr(self, "marker_panel", None), geometry_state.get("marker_panel"))
        self._restore_window_geometry(getattr(self, "_regrid_panel", None), geometry_state.get("regrid_panel"))

        sub_state = geometry_state.get("subwindows")
        if isinstance(sub_state, dict):
            self._restore_window_geometry(getattr(self, "subwindow1", None), sub_state.get("xz"))
            self._restore_window_geometry(getattr(self, "subwindow2", None), sub_state.get("zy"))

        control_panel = getattr(self, "control_panel", None)
        tool_state = geometry_state.get("tool_panels")
        if isinstance(tool_state, dict) and control_panel is not None:
            for attr, geometry in tool_state.items():
                self._restore_window_geometry(getattr(control_panel, attr, None), geometry)

        integ_entries = geometry_state.get("integration_windows")
        if isinstance(integ_entries, list):
            remaining = [entry for entry in integ_entries if isinstance(entry, dict)]
            for window in self._live_integration_windows():
                entry = self._pop_workspace_entry_by_key_or_next(
                    remaining,
                    self._integration_window_color_key(window),
                )
                if isinstance(entry, dict):
                    self._restore_window_geometry(window, entry.get("geometry"))
                    self._restore_window_range_state(
                        window,
                        entry.get("range_state"),
                        allow_axis_limits=allow_window_axis_limits,
                    )
                    self._restore_marker_panel_state_for_window(window, entry.get("marker_panel"))

        channel_entries = geometry_state.get("channel_windows")
        if isinstance(channel_entries, list):
            remaining = [entry for entry in channel_entries if isinstance(entry, dict)]
            for window in list(getattr(self, "channel_map_windows", []) or []):
                if window is None:
                    continue
                entry = self._pop_workspace_entry_by_key_or_next(
                    remaining,
                    self._channel_window_color_key(window),
                )
                if isinstance(entry, dict):
                    self._restore_window_geometry(window, entry.get("geometry"))
                    self._restore_window_range_state(
                        window,
                        entry.get("range_state"),
                        allow_axis_limits=allow_window_axis_limits,
                    )
                    self._restore_marker_panel_state_for_window(window, entry.get("marker_panel"))

    def _capture_world_ranges(self) -> dict:
        ranges = {}
        panel = getattr(self, "range_panel", None)
        if panel is None:
            return ranges

        mapping = {
            "x": ("x_min_input", "x_max_input"),
            "y": ("y_min_input", "y_max_input"),
        }
        if self.data.ndim > 2:
            mapping["z"] = ("z_min_input", "z_max_input")

        for axis, (min_attr, max_attr) in mapping.items():
            min_widget = getattr(panel, min_attr, None)
            max_widget = getattr(panel, max_attr, None)
            if min_widget is None or max_widget is None:
                continue
            min_text = str(min_widget.text() or "").strip()
            max_text = str(max_widget.text() or "").strip()
            if not min_text and not max_text:
                continue
            ranges[axis] = {"min": min_text, "max": max_text}

        return ranges

    def _restore_workspace_world_ranges(self, world_ranges) -> bool:
        if not isinstance(world_ranges, dict):
            return False
        panel = getattr(self, "range_panel", None)
        if panel is None:
            return False

        axis_targets = {}

        def _valid_pair(axis_key: str) -> bool:
            entry = world_ranges.get(axis_key)
            if not isinstance(entry, dict):
                return False
            min_text = str(entry.get("min", "")).strip()
            max_text = str(entry.get("max", "")).strip()
            if not min_text or not max_text:
                return False
            min_widget = getattr(panel, f"{axis_key}_min_input", None)
            max_widget = getattr(panel, f"{axis_key}_max_input", None)
            if min_widget is None or max_widget is None:
                return False
            axis_targets[axis_key] = (min_text, max_text)
            return True

        def _apply_axis_inputs(*axis_keys: str):
            for key in axis_keys:
                values = axis_targets.get(key)
                if not values:
                    continue
                min_widget = getattr(panel, f"{key}_min_input", None)
                max_widget = getattr(panel, f"{key}_max_input", None)
                if min_widget is None or max_widget is None:
                    continue
                min_widget.setText(values[0])
                max_widget.setText(values[1])

        applied = False
        x_ok = _valid_pair("x")
        y_ok = _valid_pair("y")
        z_ok = _valid_pair("z") if self.data.ndim > 2 else False

        _apply_axis_inputs("x", "y", "z")

        if z_ok:
            z_entry = axis_targets.get("z")
            z_anchor = str(z_entry[0]).strip() if z_entry else ""
            if z_anchor:
                try:
                    panel.original_zval = z_anchor
                except Exception:
                    pass
                try:
                    self.original_zval = z_anchor
                except Exception:
                    pass
                for sw in list(getattr(self, "subwindows", []) or []):
                    try:
                        sw.original_zval = z_anchor
                    except Exception:
                        continue

        # Apply axes independently so one conversion failure does not cancel the rest.
        if x_ok:
            try:
                _apply_axis_inputs("x", "y")
                previous_suppress = bool(getattr(panel, "_suppress_range_warning", False))
                setattr(panel, "_suppress_range_warning", True)
                try:
                    panel.set_x_range()
                finally:
                    setattr(panel, "_suppress_range_warning", previous_suppress)
                applied = True
            except Exception:
                pass
        if y_ok:
            try:
                _apply_axis_inputs("x", "y")
                previous_suppress = bool(getattr(panel, "_suppress_range_warning", False))
                setattr(panel, "_suppress_range_warning", True)
                try:
                    panel.set_y_range()
                finally:
                    setattr(panel, "_suppress_range_warning", previous_suppress)
                applied = True
            except Exception:
                pass
        if z_ok:
            try:
                _apply_axis_inputs("x", "y", "z")
                previous_suppress = bool(getattr(panel, "_suppress_range_warning", False))
                setattr(panel, "_suppress_range_warning", True)
                try:
                    panel.set_z_range()
                finally:
                    setattr(panel, "_suppress_range_warning", previous_suppress)
                applied = True
            except Exception:
                pass

        try:
            if x_ok or y_ok:
                panel._sync_inputs("xy")
            if z_ok:
                panel._sync_inputs("xz")
                panel._sync_inputs("zy")
        except Exception:
            pass
        return applied

    def _capture_pv_workspace_state(self):
        control_panel = getattr(self, "control_panel", None)
        panel = getattr(control_panel, "pvd_panel", None) if control_panel is not None else None
        if panel is None:
            return None
        exporter = getattr(panel, "export_workspace_state", None)
        if not callable(exporter):
            return None
        try:
            payload = exporter()
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    def _capture_spectrum_workspace_state(self):
        control_panel = getattr(self, "control_panel", None)
        panel = getattr(control_panel, "spec_window", None) if control_panel is not None else None
        if panel is None:
            return None
        exporter = getattr(panel, "export_workspace_state", None)
        if not callable(exporter):
            return None
        try:
            payload = exporter()
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    def _capture_baseline_workspace_state(self):
        control_panel = getattr(self, "control_panel", None)
        panel = getattr(control_panel, "baseline_panel", None) if control_panel is not None else None
        if panel is None:
            return None
        exporter = getattr(panel, "export_workspace_state", None)
        if not callable(exporter):
            return None
        try:
            payload = exporter()
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    def _capture_clump_workspace_state(self):
        control_panel = getattr(self, "control_panel", None)
        panel = getattr(control_panel, "clump_finding_panel", None) if control_panel is not None else None
        if panel is None:
            return None
        exporter = getattr(panel, "export_workspace_state", None)
        if not callable(exporter):
            return None
        try:
            payload = exporter()
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    def _restore_pv_workspace_state(self, pv_state) -> bool:
        if not isinstance(pv_state, dict):
            return False
        control_panel = getattr(self, "control_panel", None)
        if control_panel is None:
            return False
        panel = getattr(control_panel, "pvd_panel", None)
        if panel is None:
            opener = getattr(control_panel, "open_pvd_settings", None)
            if callable(opener):
                try:
                    opener()
                except Exception:
                    panel = None
                else:
                    panel = getattr(control_panel, "pvd_panel", None)
        if panel is None:
            return False
        restore = getattr(panel, "restore_workspace_state", None)
        if not callable(restore):
            return False
        try:
            return bool(restore(pv_state))
        except Exception:
            return False

    def _restore_spectrum_workspace_state(self, spectrum_state) -> bool:
        if not isinstance(spectrum_state, dict):
            return False
        control_panel = getattr(self, "control_panel", None)
        if control_panel is None:
            return False
        panel = getattr(control_panel, "spec_window", None)
        if panel is None:
            opener = getattr(control_panel, "open_spec_window", None)
            if callable(opener):
                try:
                    opener()
                except Exception:
                    panel = None
                else:
                    panel = getattr(control_panel, "spec_window", None)
        if panel is None:
            return False
        restore = getattr(panel, "restore_workspace_state", None)
        if not callable(restore):
            return False
        try:
            return bool(restore(spectrum_state))
        except Exception:
            return False

    def _restore_baseline_workspace_state(self, baseline_state) -> bool:
        if not isinstance(baseline_state, dict):
            return False
        control_panel = getattr(self, "control_panel", None)
        if control_panel is None:
            return False
        panel = getattr(control_panel, "baseline_panel", None)
        if panel is None:
            opener = getattr(control_panel, "open_baseline_panel", None)
            if callable(opener):
                try:
                    opener()
                except Exception:
                    panel = None
                else:
                    panel = getattr(control_panel, "baseline_panel", None)
        if panel is None:
            return False
        restore = getattr(panel, "restore_workspace_state", None)
        if not callable(restore):
            return False
        try:
            return bool(restore(baseline_state))
        except Exception:
            return False

    def _restore_clump_workspace_state(
        self,
        clump_state,
        *,
        ensure_panel: bool = False,
        keep_visible: bool = True,
    ) -> bool:
        if not isinstance(clump_state, dict):
            return False
        control_panel = getattr(self, "control_panel", None)
        if control_panel is None:
            return False
        panel = getattr(control_panel, "clump_finding_panel", None)
        opened_for_restore = False
        if panel is None:
            if not ensure_panel:
                return False
            opener = getattr(control_panel, "open_clump_finding_panel", None)
            if callable(opener):
                try:
                    opener()
                except Exception:
                    panel = None
                else:
                    panel = getattr(control_panel, "clump_finding_panel", None)
                    opened_for_restore = panel is not None
        if panel is None:
            return False
        restore = getattr(panel, "restore_workspace_state", None)
        if not callable(restore):
            return False
        try:
            restored = bool(restore(clump_state))
            if opened_for_restore and not bool(keep_visible):
                try:
                    panel.hide()
                except Exception:
                    pass
            return restored
        except Exception:
            return False

    def _capture_panel_visibility_state(self) -> dict:
        panel_state = {
            "control_panel_visible": bool(getattr(self, "control_panel", None) and self.control_panel.isVisible()),
            "range_panel_visible": bool(getattr(self, "range_panel", None) and self.range_panel.isVisible()),
            "marker_panel_visible": bool(getattr(self, "marker_panel", None) and self.marker_panel.isVisible()),
            "regrid_panel_visible": bool(getattr(self, "_regrid_panel", None) and self._regrid_panel.isVisible()),
        }

        if self.data.ndim > 2:
            panel_state["subwindows"] = {
                "xz": bool(getattr(self, "subwindow1", None) and self.subwindow1.isVisible()),
                "zy": bool(getattr(self, "subwindow2", None) and self.subwindow2.isVisible()),
            }

        cp = getattr(self, "control_panel", None)
        tool_panels = {}
        for attr in (
            "color_settings_panel",
            "scaling_panel",
            "unit_conversion_panel",
            "integ_settings_panel",
            "chmap_settings_panel",
            "smooth_settings_panel",
            "baseline_panel",
            "spec_window",
            "pvd_panel",
            "mask_settings_panel",
            "contour_panel",
            "arithmetic_panel",
            "clump_finding_panel",
        ):
            visible = False
            panel = getattr(cp, attr, None) if cp is not None else None
            if panel is not None:
                try:
                    visible = bool(panel.isVisible())
                except Exception:
                    visible = False
            tool_panels[attr] = visible
        panel_state["tool_panels"] = tool_panels
        return panel_state

    def _capture_global_color_settings(self) -> dict:
        try:
            from takefits.tools.color_scale import ColorMode, ColorSettingsPanel
        except Exception:
            return {}

        mode_map = {
            "main": ColorMode.MAIN,
            "integration": ColorMode.INTEG,
            "pv": ColorMode.PV,
            "channel": ColorMode.CHANNEL,
        }
        state = {}
        for name, mode in mode_map.items():
            raw = dict(ColorSettingsPanel.settings.get(mode, {}) or {})
            state[name] = {
                "min_val": raw.get("min_val"),
                "max_val": raw.get("max_val"),
                "log_scale": bool(raw.get("log_scale", False)),
                "gamma_value": float(raw.get("gamma_value", 1.0) or 1.0),
                "invert": bool(raw.get("invert", False)),
                "color_pattern": raw.get("color_pattern"),
            }
        return state

    def _normalize_color_panel_settings(self, settings=None, fallback=None) -> dict:
        normalized = {
            "min_val": None,
            "max_val": None,
            "log_scale": False,
            "gamma_value": 1.0,
            "invert": False,
            "color_pattern": None,
        }
        for source in (fallback, settings):
            if not isinstance(source, dict):
                continue
            if "min_val" in source:
                normalized["min_val"] = source.get("min_val")
            if "max_val" in source:
                normalized["max_val"] = source.get("max_val")
            if "log_scale" in source:
                normalized["log_scale"] = bool(source.get("log_scale", False))
            if "invert" in source:
                normalized["invert"] = bool(source.get("invert", False))
            if "color_pattern" in source:
                value = source.get("color_pattern")
                normalized["color_pattern"] = str(value) if value is not None else None
            if "gamma_value" in source:
                try:
                    normalized["gamma_value"] = float(source.get("gamma_value", 1.0) or 1.0)
                except Exception:
                    pass
        pattern = str(normalized.get("color_pattern") or "")
        if pattern.endswith("_r"):
            normalized["color_pattern"] = pattern[:-2]
            normalized["invert"] = True
        return normalized

    def _extract_live_color_panel_settings(self, panel):
        if panel is None:
            return None
        settings = {}
        current = getattr(panel, "current_settings", None)
        if isinstance(current, dict):
            settings.update(current)
        try:
            combo = getattr(panel, "colorscale_combo", None)
            if combo is not None:
                text = str(combo.currentText() or "").strip()
                if text:
                    settings["color_pattern"] = text
        except Exception:
            pass
        try:
            invert = getattr(panel, "invert_checkbox", None)
            if invert is not None:
                settings["invert"] = bool(invert.isChecked())
        except Exception:
            pass
        try:
            logbox = getattr(panel, "log_checkbox", None)
            if logbox is not None:
                settings["log_scale"] = bool(logbox.isChecked())
        except Exception:
            pass
        try:
            gamma_spin = getattr(panel, "gamma_spinbox", None)
            if gamma_spin is not None:
                settings["gamma_value"] = float(gamma_spin.value())
        except Exception:
            pass
        return settings if settings else None

    def _derive_panel_settings_from_image(self, image, fallback=None) -> dict:
        settings = self._normalize_color_panel_settings(fallback=fallback)
        if image is None:
            return settings
        try:
            clim = image.get_clim()
            if clim is not None:
                settings["min_val"] = float(clim[0])
                settings["max_val"] = float(clim[1])
        except Exception:
            pass
        try:
            import matplotlib as mpl

            settings["log_scale"] = isinstance(getattr(image, "norm", None), mpl.colors.LogNorm)
        except Exception:
            pass
        try:
            cmap_name = str(getattr(image.get_cmap(), "name", "") or "")
            if cmap_name.endswith("_r"):
                base = cmap_name[:-2]
                if base and base != "from_list":
                    settings["color_pattern"] = base
                    settings["invert"] = True
            elif cmap_name and cmap_name != "from_list":
                settings["color_pattern"] = cmap_name
                settings["invert"] = False
        except Exception:
            pass
        return settings

    def _capture_colormap_state(self, cmap) -> dict:
        if cmap is None:
            return {}
        state = {"name": str(getattr(cmap, "name", "") or "")}
        try:
            import numpy as np
            import matplotlib.pyplot as plt

            name = str(state.get("name") or "")
            needs_samples = (not name) or (name == "from_list")
            if not needs_samples:
                try:
                    plt.get_cmap(name)
                except Exception:
                    needs_samples = True
            if needs_samples:
                sample_count = max(2, min(256, int(getattr(cmap, "N", 256) or 256)))
                samples = cmap(np.linspace(0.0, 1.0, sample_count))
                state["rgba"] = np.asarray(samples, dtype=float).tolist()
        except Exception:
            pass
        return state

    def _capture_image_color_state(self, image) -> dict:
        if image is None:
            return {}
        payload = {}
        try:
            payload["cmap"] = self._capture_colormap_state(image.get_cmap())
        except Exception:
            pass

        try:
            clim = image.get_clim()
            if clim is not None:
                payload["clim"] = [float(clim[0]), float(clim[1])]
        except Exception:
            pass

        try:
            import matplotlib as mpl

            norm = getattr(image, "norm", None)
            if isinstance(norm, mpl.colors.LogNorm):
                payload["norm"] = {
                    "type": "log",
                    "vmin": float(norm.vmin) if norm.vmin is not None else None,
                    "vmax": float(norm.vmax) if norm.vmax is not None else None,
                }
            elif isinstance(norm, mpl.colors.Normalize):
                payload["norm"] = {
                    "type": "linear",
                    "vmin": float(norm.vmin) if norm.vmin is not None else None,
                    "vmax": float(norm.vmax) if norm.vmax is not None else None,
                }
        except Exception:
            pass
        return payload

    def _live_integration_windows(self):
        windows = []
        for window_ref in list(getattr(self, "integ_result_windows", []) or []):
            window = window_ref() if callable(window_ref) else window_ref
            if window is None:
                continue
            windows.append(window)
        return windows

    @staticmethod
    def _capture_colorbar_bounds(cax):
        if cax is None:
            return None
        try:
            bounds = [float(v) for v in cax.get_position().bounds]
        except Exception:
            return None
        if len(bounds) != 4 or not all(math.isfinite(v) for v in bounds):
            return None
        return bounds

    @staticmethod
    def _capture_colorbar_orientation(colorbar):
        orientation = str(getattr(colorbar, "orientation", "") or "").strip().lower()
        if orientation in {"vertical", "horizontal"}:
            return orientation
        return None

    def _capture_colorbar_state_for_target(self, cax, colorbar, owner=None) -> dict:
        state = {}
        bounds = self._capture_colorbar_bounds(cax)
        if bounds is not None:
            state["bounds"] = bounds
        orientation = self._capture_colorbar_orientation(colorbar)
        if orientation is not None:
            state["orientation"] = orientation
        auto_layout = None
        check_auto = getattr(owner, "_is_colorbar_auto_layout_enabled", None)
        if callable(check_auto):
            try:
                auto_layout = bool(check_auto())
            except Exception:
                auto_layout = None
        if auto_layout is None:
            owner_config = getattr(owner, "config", None)
            if isinstance(owner_config, dict) and "colorbar_auto_layout" in owner_config:
                try:
                    auto_layout = bool(owner_config.get("colorbar_auto_layout"))
                except Exception:
                    auto_layout = None
        if auto_layout is not None:
            state["auto_layout"] = bool(auto_layout)
        return state

    def _capture_workspace_colorbar_state(self) -> dict:
        config = getattr(getattr(self, "config_manager", None), "config", None)
        payload = {
            "schema": 2,
            "global": {},
            "main": {},
            "main_viewers": {},
            "integration_windows": [],
            "channel_windows": [],
        }

        if isinstance(config, dict):
            for key in self._WORKSPACE_COLORBAR_CONFIG_KEYS:
                if key not in config:
                    continue
                value = config.get(key)
                if isinstance(value, (str, bool)) or value is None:
                    payload["global"][key] = value
                    continue
                if isinstance(value, (int, float)):
                    if isinstance(value, float) and not math.isfinite(value):
                        continue
                    payload["global"][key] = value
                    continue
                payload["global"][key] = str(value)

        for plane in ("xy", "xz", "zy"):
            state = self.get_viewer_state(plane)
            viewer = self._viewer_for_plane(plane)
            colorbar_state = self._capture_colorbar_state_for_target(
                getattr(state, "cax", None) if state is not None else None,
                getattr(state, "colorbar", None) if state is not None else None,
                owner=viewer if viewer is not None else self,
            )
            if colorbar_state:
                payload["main_viewers"][plane] = colorbar_state

        if isinstance(payload["main_viewers"].get("xy"), dict):
            payload["main"] = dict(payload["main_viewers"]["xy"])
        elif payload["main_viewers"]:
            payload["main"] = dict(next(iter(payload["main_viewers"].values())))

        for window in self._live_integration_windows():
            colorbar_state = self._capture_colorbar_state_for_target(
                getattr(window, "cax", None),
                getattr(window, "colorbar", None),
                owner=window,
            )
            if not colorbar_state:
                continue
            payload["integration_windows"].append(
                {
                    "key": self._integration_window_color_key(window),
                    "state": colorbar_state,
                }
            )

        for window in list(getattr(self, "channel_map_windows", []) or []):
            if window is None:
                continue
            colorbar_state = self._capture_colorbar_state_for_target(
                getattr(window, "cax", None),
                getattr(window, "colorbar", None),
                owner=window,
            )
            if not colorbar_state:
                continue
            payload["channel_windows"].append(
                {
                    "key": self._channel_window_color_key(window),
                    "state": colorbar_state,
                }
            )

        if not (
            payload["global"]
            or payload["main"]
            or payload["main_viewers"]
            or payload["integration_windows"]
            or payload["channel_windows"]
        ):
            return {}
        return payload

    @staticmethod
    def _coerce_workspace_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        text = str(value or "").strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
        return bool(value)

    @staticmethod
    def _coerce_workspace_int(value):
        try:
            return int(round(float(value)))
        except Exception:
            return None

    @staticmethod
    def _coerce_workspace_float(value):
        try:
            parsed = float(value)
        except Exception:
            return None
        if not math.isfinite(parsed):
            return None
        return parsed

    @staticmethod
    def _workspace_colorbar_global_state(colorbar_state) -> dict:
        if not isinstance(colorbar_state, dict):
            return {}
        global_state = colorbar_state.get("global")
        if isinstance(global_state, dict):
            return global_state
        return colorbar_state

    @staticmethod
    def _workspace_colorbar_entries(colorbar_state, key: str):
        if not isinstance(colorbar_state, dict):
            return []
        entries = colorbar_state.get(key)
        if not isinstance(entries, list):
            return []
        return [entry for entry in entries if isinstance(entry, dict)]

    def _normalize_workspace_colorbar_layout(self, layout_state) -> dict:
        if not isinstance(layout_state, dict):
            return {}
        normalized = {}

        bounds = layout_state.get("bounds")
        if isinstance(bounds, (list, tuple)) and len(bounds) == 4:
            parsed = [self._coerce_workspace_float(v) for v in bounds]
            if all(v is not None for v in parsed):
                normalized["bounds"] = [float(parsed[0]), float(parsed[1]), float(parsed[2]), float(parsed[3])]
        else:
            px = self._coerce_workspace_float(layout_state.get("cbar_pos_x"))
            py = self._coerce_workspace_float(layout_state.get("cbar_pos_y"))
            pw = self._coerce_workspace_float(layout_state.get("cbar_width"))
            ph = self._coerce_workspace_float(layout_state.get("cbar_height"))
            if None not in (px, py, pw, ph):
                normalized["bounds"] = [float(px), float(py), float(pw), float(ph)]

        orientation = str(layout_state.get("orientation", "") or "").strip().lower()
        if orientation in {"vertical", "horizontal"}:
            normalized["orientation"] = orientation
        if "auto_layout" in layout_state:
            normalized["auto_layout"] = self._coerce_workspace_bool(layout_state.get("auto_layout"))
        elif "colorbar_auto_layout" in layout_state:
            normalized["auto_layout"] = self._coerce_workspace_bool(layout_state.get("colorbar_auto_layout"))
        return normalized

    def _apply_workspace_colorbar_bounds_to_viewer_state(self, state, bounds) -> bool:
        if state is None or getattr(state, "cax", None) is None:
            return False
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
            return False
        try:
            pos_x, pos_y, width, height = [float(v) for v in bounds]
            if hasattr(state.cax, "set_axes_locator"):
                state.cax.set_axes_locator(None)
                setattr(state, "_colorbar_axes_locator", None)
            state.cax.set_position([pos_x, pos_y, width, height])
            state.cax.set_gid("colorbar")
            self._set_colorbar_zorder_for_state(state)
            canvas = getattr(state, "canvas", None)
            if canvas is not None:
                self._request_canvas_redraw(canvas)
            return True
        except Exception:
            return False

    def _apply_workspace_colorbar_state_to_main_viewer(self, config: dict, layout_state=None, per_viewer_layouts=None) -> bool:
        if not isinstance(config, dict):
            return False
        applied = False
        layout = self._normalize_workspace_colorbar_layout(layout_state)
        normalized_per_viewer = {}
        if isinstance(per_viewer_layouts, dict):
            for plane in ("xy", "xz", "zy"):
                plane_layout = self._normalize_workspace_colorbar_layout(per_viewer_layouts.get(plane))
                if plane_layout:
                    normalized_per_viewer[plane] = plane_layout
        if not layout and normalized_per_viewer:
            layout = dict(normalized_per_viewer.get("xy") or next(iter(normalized_per_viewer.values())))
        orientation = str(layout.get("orientation", config.get("colorbar_orientation", "vertical")) or "vertical").strip().lower()
        if orientation not in {"vertical", "horizontal"}:
            orientation = "vertical"
        config["colorbar_orientation"] = orientation
        auto_layout = layout.get("auto_layout")
        if auto_layout is None:
            auto_layout = bool(config.get("colorbar_auto_layout", True))
        else:
            auto_layout = bool(auto_layout)
            config["colorbar_auto_layout"] = auto_layout

        rebuild = getattr(self, "_rebuild_colorbars", None)
        if callable(rebuild):
            try:
                rebuild()
                applied = True
            except Exception:
                pass

        if auto_layout:
            fit_now = getattr(self, "fit_colorbar_now", None)
            if callable(fit_now):
                try:
                    fit_now()
                    applied = True
                except Exception:
                    pass
        else:
            fallback_bounds = layout.get("bounds")
            primary_bounds = None
            for plane in ("xy", "xz", "zy"):
                state = self.get_viewer_state(plane)
                if state is None:
                    continue
                plane_layout = normalized_per_viewer.get(plane, layout)
                bounds = plane_layout.get("bounds", fallback_bounds)
                if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
                    continue
                if self._apply_workspace_colorbar_bounds_to_viewer_state(state, bounds):
                    if primary_bounds is None and plane == "xy":
                        primary_bounds = [float(v) for v in bounds]
                    applied = True
            if primary_bounds is None and isinstance(fallback_bounds, (list, tuple)) and len(fallback_bounds) == 4:
                try:
                    primary_bounds = [float(v) for v in fallback_bounds]
                except Exception:
                    primary_bounds = None
            if primary_bounds is not None:
                try:
                    self._update_colorbar_geometry_config(*primary_bounds)
                except Exception:
                    pass

        try:
            from takefits.tools.color_scale import ColorSettingsPanel

            for plane in ("xy", "xz", "zy"):
                state = self.get_viewer_state(plane)
                if state is None:
                    continue
                cax = getattr(state, "cax", None)
                colorbar = getattr(state, "colorbar", None)
                if cax is None or colorbar is None:
                    continue
                ColorSettingsPanel.apply_colorbar_settings(cax=cax, colorbar=colorbar, config=config)
                canvas = getattr(state, "canvas", None)
                if canvas is not None:
                    try:
                        self._request_canvas_redraw(canvas)
                    except Exception:
                        pass
                applied = True
        except Exception:
            pass
        return applied

    def _apply_workspace_colorbar_state_to_aux_window(self, window, config: dict, layout_state=None) -> bool:
        if window is None or not isinstance(config, dict):
            return False
        applied = False
        layout = self._normalize_workspace_colorbar_layout(layout_state)
        target_orientation = str(layout.get("orientation", config.get("colorbar_orientation", "vertical")) or "vertical").strip().lower()
        if target_orientation not in {"vertical", "horizontal"}:
            target_orientation = "vertical"
        auto_layout = layout.get("auto_layout")
        if auto_layout is None:
            auto_layout = bool(config.get("colorbar_auto_layout", True))
        else:
            auto_layout = bool(auto_layout)
        try:
            setattr(window, "_colorbar_auto_layout_override", bool(auto_layout))
        except Exception:
            pass

        current_orientation = str(getattr(getattr(window, "colorbar", None), "orientation", "") or "").strip().lower()
        if current_orientation not in {"vertical", "horizontal"}:
            current_orientation = ""
        orientation_changed = current_orientation != target_orientation

        if orientation_changed:
            set_orientation = getattr(window, "_set_colorbar_orientation_config", None)
            if callable(set_orientation):
                try:
                    set_orientation(target_orientation)
                except Exception:
                    pass
            rebuild = getattr(window, "_rebuild_colorbar", None)
            if callable(rebuild):
                try:
                    rebuild(target_orientation)
                    applied = True
                except Exception:
                    pass

        bounds = layout.get("bounds")
        if isinstance(bounds, (list, tuple)) and len(bounds) == 4:
            set_geometry = getattr(window, "_set_colorbar_geometry", None)
            if callable(set_geometry):
                try:
                    pos_x, pos_y, width, height = [float(v) for v in bounds]
                    set_geometry(pos_x, pos_y, width, height)
                    applied = True
                except Exception:
                    pass
            cax = getattr(window, "cax", None)
            if cax is not None:
                try:
                    if hasattr(cax, "set_axes_locator"):
                        cax.set_axes_locator(None)
                        setattr(window, "_colorbar_axes_locator", None)
                    cax.set_position([float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3])])
                    cax.set_gid("colorbar")
                    applied = True
                except Exception:
                    pass
        elif auto_layout:
            fit_now = getattr(window, "fit_colorbar_now", None)
            if callable(fit_now):
                try:
                    fit_now()
                    applied = True
                except Exception:
                    pass

        try:
            from takefits.tools.color_scale import ColorSettingsPanel

            cax = getattr(window, "cax", None)
            colorbar = getattr(window, "colorbar", None)
            if cax is not None and colorbar is not None:
                cfg_getter = getattr(window, "_get_colorbar_config", None)
                window_config = cfg_getter() if callable(cfg_getter) else getattr(window, "config", None)
                if not isinstance(window_config, dict):
                    window_config = config
                ColorSettingsPanel.apply_colorbar_settings(cax=cax, colorbar=colorbar, config=window_config)
                applied = True
        except Exception:
            pass

        canvas = getattr(window, "canvas", None)
        if applied and canvas is not None:
            try:
                self._request_canvas_redraw(canvas)
            except Exception:
                pass
        return applied

    def _workspace_colorbar_layout_equivalent(self, current_layout, desired_layout) -> bool:
        desired = self._normalize_workspace_colorbar_layout(desired_layout)
        if not desired:
            return True
        current = self._normalize_workspace_colorbar_layout(current_layout)
        if not current:
            return False

        desired_orientation = desired.get("orientation")
        if desired_orientation and current.get("orientation") != desired_orientation:
            return False

        if "auto_layout" in desired:
            if bool(current.get("auto_layout")) != bool(desired.get("auto_layout")):
                return False

        desired_bounds = desired.get("bounds")
        if isinstance(desired_bounds, (list, tuple)) and len(desired_bounds) == 4:
            current_bounds = current.get("bounds")
            if not isinstance(current_bounds, (list, tuple)) or len(current_bounds) != 4:
                return False
            for current_value, desired_value in zip(current_bounds, desired_bounds):
                try:
                    if not math.isclose(
                        float(current_value),
                        float(desired_value),
                        rel_tol=1e-6,
                        abs_tol=1e-6,
                    ):
                        return False
                except Exception:
                    return False
        return True

    def _current_workspace_colorbar_layout_snapshot(self) -> dict:
        snapshot = {
            "main_viewers": {},
            "integration_windows": {},
            "channel_windows": {},
        }

        for plane in ("xy", "xz", "zy"):
            state = self.get_viewer_state(plane)
            viewer = self._viewer_for_plane(plane)
            layout = self._capture_colorbar_state_for_target(
                getattr(state, "cax", None) if state is not None else None,
                getattr(state, "colorbar", None) if state is not None else None,
                owner=viewer if viewer is not None else self,
            )
            normalized = self._normalize_workspace_colorbar_layout(layout)
            if normalized:
                snapshot["main_viewers"][plane] = normalized

        for window in self._live_integration_windows():
            key = self._integration_window_color_key(window)
            layout = self._capture_colorbar_state_for_target(
                getattr(window, "cax", None),
                getattr(window, "colorbar", None),
                owner=window,
            )
            normalized = self._normalize_workspace_colorbar_layout(layout)
            if normalized:
                snapshot["integration_windows"][key] = normalized

        for window in list(getattr(self, "channel_map_windows", []) or []):
            if window is None:
                continue
            key = self._channel_window_color_key(window)
            layout = self._capture_colorbar_state_for_target(
                getattr(window, "cax", None),
                getattr(window, "colorbar", None),
                owner=window,
            )
            normalized = self._normalize_workspace_colorbar_layout(layout)
            if normalized:
                snapshot["channel_windows"][key] = normalized

        return snapshot

    def _desired_workspace_colorbar_layout_snapshot(self, colorbar_state) -> dict:
        state = colorbar_state if isinstance(colorbar_state, dict) else {}
        snapshot = {
            "main_viewers": {},
            "integration_windows": {},
            "channel_windows": {},
        }

        resolved_main_layouts = resolve_workspace_main_colorbar_layouts(state)
        per_plane = resolved_main_layouts.get("per_plane") if isinstance(resolved_main_layouts, dict) else {}
        if isinstance(per_plane, dict):
            for plane in ("xy", "xz", "zy"):
                normalized = self._normalize_workspace_colorbar_layout(per_plane.get(plane))
                if normalized:
                    snapshot["main_viewers"][plane] = normalized

        global_state = self._workspace_colorbar_global_state(state)

        integ_entries = self._workspace_colorbar_entries(state, "integration_windows")
        integ_fallback = global_state if not integ_entries else None
        remaining_integ_entries = list(integ_entries)
        for window in self._live_integration_windows():
            entry = self._pop_workspace_entry_by_key_or_next(
                remaining_integ_entries,
                self._integration_window_color_key(window),
            )
            layout = None
            if isinstance(entry, dict):
                layout = entry.get("state") if isinstance(entry.get("state"), dict) else entry
            if not isinstance(layout, dict):
                layout = integ_fallback
            normalized = self._normalize_workspace_colorbar_layout(layout)
            if normalized:
                snapshot["integration_windows"][self._integration_window_color_key(window)] = normalized

        channel_entries = self._workspace_colorbar_entries(state, "channel_windows")
        channel_fallback = global_state if not channel_entries else None
        remaining_channel_entries = list(channel_entries)
        for window in list(getattr(self, "channel_map_windows", []) or []):
            if window is None:
                continue
            entry = self._pop_workspace_entry_by_key_or_next(
                remaining_channel_entries,
                self._channel_window_color_key(window),
            )
            layout = None
            if isinstance(entry, dict):
                layout = entry.get("state") if isinstance(entry.get("state"), dict) else entry
            if not isinstance(layout, dict):
                layout = channel_fallback
            normalized = self._normalize_workspace_colorbar_layout(layout)
            if normalized:
                snapshot["channel_windows"][self._channel_window_color_key(window)] = normalized

        return snapshot

    def _workspace_colorbar_layouts_match(self, current_snapshot, desired_snapshot) -> bool:
        if not isinstance(desired_snapshot, dict):
            return True
        current = current_snapshot if isinstance(current_snapshot, dict) else {}

        for bucket in ("main_viewers", "integration_windows", "channel_windows"):
            desired_bucket = desired_snapshot.get(bucket)
            if not isinstance(desired_bucket, dict):
                continue
            current_bucket = current.get(bucket)
            if not isinstance(current_bucket, dict):
                current_bucket = {}
            for key, desired_layout in desired_bucket.items():
                if not self._workspace_colorbar_layout_equivalent(
                    current_bucket.get(key),
                    desired_layout,
                ):
                    return False
        return True

    def _apply_workspace_colorbar_state_to_open_windows(self, colorbar_state=None) -> bool:
        config = getattr(getattr(self, "config_manager", None), "config", None)
        if not isinstance(config, dict):
            return False

        state = colorbar_state if isinstance(colorbar_state, dict) else {}
        global_state = self._workspace_colorbar_global_state(state)
        main_layout = state.get("main") if isinstance(state.get("main"), dict) else None
        if not isinstance(main_layout, dict):
            main_viewers = state.get("main_viewers")
            if isinstance(main_viewers, dict):
                xy_state = main_viewers.get("xy")
                if isinstance(xy_state, dict):
                    main_layout = xy_state
                else:
                    for candidate in main_viewers.values():
                        if isinstance(candidate, dict):
                            main_layout = candidate
                            break
        if not isinstance(main_layout, dict):
            main_layout = global_state

        resolved_main_layouts = resolve_workspace_main_colorbar_layouts(state)
        per_plane_layouts = resolved_main_layouts.get("per_plane")
        resolved_primary = resolved_main_layouts.get("primary")
        if isinstance(resolved_primary, dict) and resolved_primary:
            main_layout = resolved_primary

        applied = self._apply_workspace_colorbar_state_to_main_viewer(
            config,
            layout_state=main_layout,
            per_viewer_layouts=per_plane_layouts,
        )

        integ_entries = self._workspace_colorbar_entries(state, "integration_windows")
        integ_fallback = global_state if not integ_entries else None
        remaining_integ_entries = list(integ_entries)
        for window in self._live_integration_windows():
            entry = self._pop_workspace_entry_by_key_or_next(
                remaining_integ_entries,
                self._integration_window_color_key(window),
            )
            layout = None
            if isinstance(entry, dict):
                layout = entry.get("state") if isinstance(entry.get("state"), dict) else entry
            if not isinstance(layout, dict):
                layout = integ_fallback
            applied = self._apply_workspace_colorbar_state_to_aux_window(window, config, layout_state=layout) or applied

        channel_entries = self._workspace_colorbar_entries(state, "channel_windows")
        channel_fallback = global_state if not channel_entries else None
        remaining_channel_entries = list(channel_entries)
        for window in list(getattr(self, "channel_map_windows", []) or []):
            if window is None:
                continue
            entry = self._pop_workspace_entry_by_key_or_next(
                remaining_channel_entries,
                self._channel_window_color_key(window),
            )
            layout = None
            if isinstance(entry, dict):
                layout = entry.get("state") if isinstance(entry.get("state"), dict) else entry
            if not isinstance(layout, dict):
                layout = channel_fallback
            applied = self._apply_workspace_colorbar_state_to_aux_window(window, config, layout_state=layout) or applied
        return applied

    def _restore_workspace_colorbar_state(self, colorbar_state, *, apply_to_open_windows: bool = False) -> bool:
        config = getattr(getattr(self, "config_manager", None), "config", None)
        if not isinstance(config, dict):
            return False

        def _equivalent(current, new_value) -> bool:
            if isinstance(current, bool) or isinstance(new_value, bool):
                return bool(current) == bool(new_value)
            if isinstance(current, (int, float)) and isinstance(new_value, (int, float)):
                return math.isclose(float(current), float(new_value), rel_tol=1e-9, abs_tol=1e-9)
            return current == new_value

        source = self._workspace_colorbar_global_state(colorbar_state)
        bool_keys = {
            "colorbar_auto_layout",
            "colorbar_tick_left",
            "colorbar_tick_right",
            "colorbar_tick_top",
            "colorbar_tick_bottom",
            "colorbar_tick_labelleft",
            "colorbar_tick_labeltop",
        }
        int_keys = {"colorbar_mtick_freq", "colorbar_label_fontsize"}
        float_keys = {
            "colorbar_gap_px",
            "colorbar_gap_x_px",
            "colorbar_gap_y_px",
            "colorbar_thickness_px",
            "colorbar_length_value",
            "cbar_pos_x",
            "cbar_pos_y",
            "cbar_width",
            "cbar_height",
            "colorbar_tick_length",
            "colorbar_mtick_length",
            "colorbar_tick_width",
        }
        updated = False
        for key in self._WORKSPACE_COLORBAR_CONFIG_KEYS:
            if key not in source:
                continue
            raw = source.get(key)
            if key in bool_keys:
                value = self._coerce_workspace_bool(raw)
            elif key in int_keys:
                value = self._coerce_workspace_int(raw)
                if value is None:
                    continue
            elif key in float_keys:
                value = self._coerce_workspace_float(raw)
                if value is None:
                    continue
            elif raw is None:
                value = None
            else:
                value = str(raw)
            if _equivalent(config.get(key), value):
                continue
            config[key] = value
            updated = True

        if apply_to_open_windows:
            has_window_payload = bool(
                isinstance(colorbar_state, dict)
                and (
                    isinstance(colorbar_state.get("main"), dict)
                    or isinstance(colorbar_state.get("main_viewers"), dict)
                    or isinstance(colorbar_state.get("integration_windows"), list)
                    or isinstance(colorbar_state.get("channel_windows"), list)
                    or bool(source)
                )
            )
            if not has_window_payload and not updated:
                return False
            return self._apply_workspace_colorbar_state_to_open_windows(colorbar_state=colorbar_state) or updated
        return updated

    def _capture_color_settings_state(self) -> dict:
        global_state = self._capture_global_color_settings()
        main_panel = getattr(getattr(self, "control_panel", None), "color_settings_panel", None)
        main_live = self._extract_live_color_panel_settings(main_panel)
        main_fallback = self._normalize_color_panel_settings(
            main_live,
            fallback=getattr(self, "_color_panel_hint_main", None),
        )
        main_image = getattr(self, "im", None)
        main_panel_settings = self._derive_panel_settings_from_image(
            main_image,
            fallback=self._normalize_color_panel_settings(main_fallback, fallback=global_state.get("main")),
        )
        global_state["main"] = dict(main_panel_settings)
        self._color_panel_hint_main = dict(main_panel_settings)
        pv_panel = getattr(getattr(self, "control_panel", None), "pvd_panel", None)
        pv_image = getattr(pv_panel, "im", None) if pv_panel is not None else None
        if pv_image is not None:
            pv_live = self._extract_live_color_panel_settings(getattr(pv_panel, "color_settings_panel", None))
            pv_hint = getattr(pv_panel, "_color_panel_hint", None)
            pv_fallback = self._normalize_color_panel_settings(
                pv_live,
                fallback=pv_hint,
            )
            pv_fallback = self._normalize_color_panel_settings(pv_fallback, fallback=global_state.get("pv"))
            pv_panel_settings = self._derive_panel_settings_from_image(pv_image, fallback=pv_fallback)
            global_state["pv"] = dict(pv_panel_settings)
            try:
                setattr(pv_panel, "_color_panel_hint", dict(pv_panel_settings))
            except Exception:
                pass
        payload = {
            "schema": 2,
            "global": global_state,
            "main": dict(global_state.get("main", {}) or {}),
            "integration": dict(global_state.get("integration", {}) or {}),
            "pv": dict(global_state.get("pv", {}) or {}),
            "channel": dict(global_state.get("channel", {}) or {}),
            "main_panel_settings": dict(main_panel_settings),
            "main_viewers": {},
            "integration_windows": [],
            "channel_windows": [],
        }

        for plane in ("xy", "xz", "zy"):
            viewer = self._viewer_for_plane(plane)
            image = getattr(viewer, "im", None) if viewer is not None else None
            if image is None:
                continue
            payload["main_viewers"][plane] = self._capture_image_color_state(image)

        for window in self._live_integration_windows():
            image = getattr(window, "im", None)
            if image is None:
                continue
            panel_live = self._extract_live_color_panel_settings(getattr(window, "color_settings_panel", None))
            panel_hint = getattr(window, "_color_panel_hint", None)
            fallback_hint = self._normalize_color_panel_settings(
                panel_live,
                fallback=panel_hint,
            )
            fallback_hint = self._normalize_color_panel_settings(fallback_hint, fallback=global_state.get("integration"))
            panel_settings = self._derive_panel_settings_from_image(image, fallback=fallback_hint)
            payload["integration_windows"].append(
                {
                    "key": self._integration_window_color_key(window),
                    "title": str(getattr(window, "windowTitle", lambda: "")() or ""),
                    "image": self._capture_image_color_state(image),
                    "panel_settings": panel_settings,
                }
            )

        for window in list(getattr(self, "channel_map_windows", []) or []):
            if window is None:
                continue
            images = list(getattr(window, "im_list", []) or [])
            if not images:
                continue
            panel_live = self._extract_live_color_panel_settings(getattr(window, "color_settings_panel", None))
            panel_hint = getattr(window, "_color_panel_hint", None)
            fallback_hint = self._normalize_color_panel_settings(
                panel_live,
                fallback=panel_hint,
            )
            fallback_hint = self._normalize_color_panel_settings(fallback_hint, fallback=global_state.get("channel"))
            panel_settings = self._derive_panel_settings_from_image(images[0], fallback=fallback_hint)
            payload["channel_windows"].append(
                {
                    "key": self._channel_window_color_key(window),
                    "title": str(getattr(window, "windowTitle", lambda: "")() or ""),
                    "image": self._capture_image_color_state(images[0]),
                    "panel_settings": panel_settings,
                }
            )

        return payload

    def _cmap_from_state(self, cmap_state):
        if not isinstance(cmap_state, dict):
            return None
        try:
            import numpy as np
            import matplotlib as mpl
            import matplotlib.pyplot as plt
        except Exception:
            return None

        cmap = None
        rgba = cmap_state.get("rgba")
        if isinstance(rgba, list) and rgba:
            try:
                rgba_arr = np.asarray(rgba, dtype=float)
                cmap = mpl.colors.ListedColormap(
                    rgba_arr,
                    name=str(cmap_state.get("name") or "workspace_colormap"),
                )
            except Exception:
                cmap = None
        if cmap is None:
            name = str(cmap_state.get("name") or "")
            if name:
                try:
                    cmap = plt.get_cmap(name)
                    if hasattr(cmap, "copy"):
                        cmap = cmap.copy()
                except Exception:
                    cmap = None
        if cmap is None:
            return None

        bad_color = None
        try:
            bad_color = self.config_manager.config.get("bad_color", "black")
        except Exception:
            bad_color = "black"
        try:
            cmap.set_bad(color=bad_color)
        except Exception:
            pass
        return cmap

    def _apply_image_color_state(self, image, image_state):
        if image is None or not isinstance(image_state, dict):
            return
        try:
            import matplotlib as mpl
        except Exception:
            return

        cmap = self._cmap_from_state(image_state.get("cmap"))
        if cmap is not None:
            try:
                image.set_cmap(cmap)
            except Exception:
                pass

        norm_state = image_state.get("norm")
        if isinstance(norm_state, dict):
            norm_type = str(norm_state.get("type") or "").lower()
            try:
                vmin = norm_state.get("vmin")
                vmax = norm_state.get("vmax")
                if vmin is not None:
                    vmin = float(vmin)
                if vmax is not None:
                    vmax = float(vmax)
                if norm_type == "log" and vmin is not None and vmax is not None and vmin > 0 and vmax > 0:
                    image.set_norm(mpl.colors.LogNorm(vmin=vmin, vmax=vmax))
                elif norm_type in {"linear", "normalize"}:
                    image.set_norm(mpl.colors.Normalize(vmin=vmin, vmax=vmax))
            except Exception:
                pass

        clim = image_state.get("clim")
        if isinstance(clim, (list, tuple)) and len(clim) == 2:
            try:
                image.set_clim(float(clim[0]), float(clim[1]))
            except Exception:
                pass

    def _entry_image_state(self, entry):
        if not isinstance(entry, dict):
            return {}
        if isinstance(entry.get("image"), dict):
            return entry["image"]
        return entry

    def _integration_window_color_key(self, window) -> str:
        meta = getattr(window, "history_metadata", None)
        if isinstance(meta, dict) and meta:
            try:
                return f"meta:{json.dumps(meta, sort_keys=True, separators=(',', ':'))}"
            except Exception:
                pass
        try:
            title = str(getattr(window, "windowTitle", lambda: "")() or "")
        except Exception:
            title = ""
        return f"title:{title}"

    def _channel_window_color_key(self, window) -> str:
        parts = {
            "title": str(getattr(window, "windowTitle", lambda: "")() or ""),
            "plane": getattr(window, "plane", None),
            "plane_num": getattr(window, "plane_num", None),
            "tiles_x": getattr(window, "tiles_x", None),
            "tiles_y": getattr(window, "tiles_y", None),
            "dir_num": getattr(window, "dir_num", None),
            "chlabel_num": getattr(window, "chlabel_num", None),
        }
        try:
            return f"meta:{json.dumps(parts, sort_keys=True, separators=(',', ':'))}"
        except Exception:
            return f"title:{parts.get('title') or ''}"

    def _restore_per_panel_color_state(self, color_state):
        if not isinstance(color_state, dict):
            return False
        applied = False
        global_state = color_state.get("global")
        if not isinstance(global_state, dict):
            global_state = {}

        main_states = color_state.get("main_viewers")
        if isinstance(main_states, dict):
            for plane in ("xy", "xz", "zy"):
                image_state = main_states.get(plane)
                if not isinstance(image_state, dict):
                    continue
                viewer = self._viewer_for_plane(plane)
                image = getattr(viewer, "im", None) if viewer is not None else None
                if image is None:
                    continue
                self._apply_image_color_state(image, image_state)
                try:
                    if getattr(viewer, "canvas", None) is not None:
                        self._request_canvas_redraw(viewer.canvas)
                except Exception:
                    pass
                applied = True
        try:
            from takefits.tools.color_scale import ColorMode, ColorSettingsPanel

            main_settings = color_state.get("main_panel_settings")
            if not isinstance(main_settings, dict):
                main_settings = self._derive_panel_settings_from_image(
                    getattr(self, "im", None),
                    fallback=global_state.get("main"),
                )
            main_settings = self._normalize_color_panel_settings(
                main_settings,
                fallback=global_state.get("main"),
            )
            self._color_panel_hint_main = dict(main_settings)
            ColorSettingsPanel.settings[ColorMode.MAIN] = dict(main_settings)
            pattern = str(main_settings.get("color_pattern") or "")
            if pattern:
                if bool(main_settings.get("invert")) and not pattern.endswith("_r"):
                    pattern = f"{pattern}_r"
                if hasattr(self, "displaymap") and self.displaymap is not None:
                    self.displaymap.colorscale = pattern
                for viewer in list(getattr(self, "subwindows", []) or []):
                    if hasattr(viewer, "displaymap") and viewer.displaymap is not None:
                        viewer.displaymap.colorscale = pattern
        except Exception:
            pass

        integ_entries = color_state.get("integration_windows")
        if isinstance(integ_entries, list):
            remaining_entries = [entry for entry in integ_entries if isinstance(entry, dict)]
            for window in self._live_integration_windows():
                entry = self._pop_workspace_entry_by_key_or_next(
                    remaining_entries,
                    self._integration_window_color_key(window),
                )
                image = getattr(window, "im", None)
                if image is None:
                    continue
                image_state = self._entry_image_state(entry)
                if not isinstance(image_state, dict):
                    continue
                self._apply_image_color_state(image, image_state)
                panel_settings = self._normalize_color_panel_settings(
                    entry.get("panel_settings") if isinstance(entry, dict) else None,
                    fallback=global_state.get("integration"),
                )
                panel_settings = self._derive_panel_settings_from_image(image, fallback=panel_settings)
                try:
                    setattr(window, "_color_panel_hint", panel_settings)
                except Exception:
                    pass
                pattern = str(panel_settings.get("color_pattern") or "")
                if pattern:
                    if bool(panel_settings.get("invert")) and not pattern.endswith("_r"):
                        pattern = f"{pattern}_r"
                    try:
                        window.color_pattern = pattern
                    except Exception:
                        pass
                try:
                    if getattr(window, "canvas", None) is not None:
                        self._request_canvas_redraw(window.canvas)
                except Exception:
                    pass
                applied = True

        channel_entries = color_state.get("channel_windows")
        if isinstance(channel_entries, list):
            channel_windows = list(getattr(self, "channel_map_windows", []) or [])
            remaining_entries = [entry for entry in channel_entries if isinstance(entry, dict)]
            for window in channel_windows:
                if window is None:
                    continue
                entry = self._pop_workspace_entry_by_key_or_next(
                    remaining_entries,
                    self._channel_window_color_key(window),
                )
                images = list(getattr(window, "im_list", []) or [])
                if not images:
                    continue
                image_state = self._entry_image_state(entry)
                if not isinstance(image_state, dict):
                    continue
                for image in images:
                    self._apply_image_color_state(image, image_state)
                panel_settings = self._normalize_color_panel_settings(
                    entry.get("panel_settings") if isinstance(entry, dict) else None,
                    fallback=global_state.get("channel"),
                )
                panel_settings = self._derive_panel_settings_from_image(images[0], fallback=panel_settings)
                try:
                    setattr(window, "_color_panel_hint", panel_settings)
                except Exception:
                    pass
                pattern = str(panel_settings.get("color_pattern") or "")
                if pattern:
                    if bool(panel_settings.get("invert")) and not pattern.endswith("_r"):
                        pattern = f"{pattern}_r"
                    try:
                        window.color_pattern = pattern
                    except Exception:
                        pass
                try:
                    if getattr(window, "canvas", None) is not None:
                        self._request_canvas_redraw(window.canvas)
                except Exception:
                    pass
                applied = True

        return applied

    def _seed_main_color_panel_settings_from_current_image(self):
        try:
            from takefits.tools.color_scale import ColorMode, ColorSettingsPanel
        except Exception:
            return None
        hint = getattr(self, "_color_panel_hint_main", None)
        fallback = dict(ColorSettingsPanel.settings.get(ColorMode.MAIN, {}) or {})
        settings = self._normalize_color_panel_settings(hint, fallback=fallback)
        settings = self._derive_panel_settings_from_image(getattr(self, "im", None), fallback=settings)
        self._color_panel_hint_main = dict(settings)
        ColorSettingsPanel.settings[ColorMode.MAIN] = dict(settings)
        return settings

    def _update_main_color_panel_hint(self):
        try:
            from takefits.tools.color_scale import ColorMode, ColorSettingsPanel
        except Exception:
            return
        self._color_panel_hint_main = dict(ColorSettingsPanel.settings.get(ColorMode.MAIN, {}) or {})

    def _collect_color_targets_for_mode(self, mode_name: str):
        targets = []
        mode = str(mode_name or "").lower()
        if mode == "main":
            viewers = [self] + list(getattr(self, "subwindows", []) or [])
            for viewer in viewers:
                if viewer is None:
                    continue
                im = getattr(viewer, "im", None)
                if im is None:
                    continue
                targets.append((viewer, [im]))
            return targets

        if mode == "integration":
            for window_ref in list(getattr(self, "integ_result_windows", []) or []):
                window = window_ref() if callable(window_ref) else window_ref
                if window is None:
                    continue
                im = getattr(window, "im", None)
                if im is None:
                    continue
                targets.append((window, [im]))
            return targets

        if mode == "pv":
            panel = getattr(getattr(self, "control_panel", None), "pvd_panel", None)
            if panel is None:
                return targets
            im = getattr(panel, "im", None)
            if im is None:
                return targets
            targets.append((panel, [im]))
            return targets

        if mode == "channel":
            for window in list(getattr(self, "channel_map_windows", []) or []):
                if window is None:
                    continue
                images = list(getattr(window, "im_list", []) or [])
                if not images:
                    continue
                targets.append((window, images))
            return targets

        return targets

    def _build_gamma_cmap(self, pattern: str, gamma: float):
        import numpy as np
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        cmap = plt.get_cmap(pattern)
        gamma_value = max(1e-6, float(gamma))
        rgba = cmap(np.linspace(0.0, 1.0, cmap.N) ** gamma_value)
        return mpl.colors.ListedColormap(rgba)

    def _apply_color_state_to_mode(self, mode_name: str, settings: dict):
        if not isinstance(settings, dict):
            return
        targets = self._collect_color_targets_for_mode(mode_name)
        if not targets:
            return

        pattern = settings.get("color_pattern")
        if not pattern:
            return
        pattern = str(pattern)
        if bool(settings.get("invert")) and not pattern.endswith("_r"):
            pattern = f"{pattern}_r"
        gamma = float(settings.get("gamma_value", 1.0) or 1.0)

        try:
            cmap = self._build_gamma_cmap(pattern, gamma)
        except Exception:
            return

        bad_color = None
        try:
            bad_color = self.config_manager.config.get("bad_color", "black")
        except Exception:
            bad_color = "black"
        try:
            cmap.set_bad(color=bad_color)
        except Exception:
            pass

        min_val = settings.get("min_val")
        max_val = settings.get("max_val")
        use_clim = False
        try:
            min_float = float(min_val)
            max_float = float(max_val)
            use_clim = True
        except Exception:
            min_float = None
            max_float = None

        log_scale = bool(settings.get("log_scale", False))

        for window, images in targets:
            for image in images:
                try:
                    image.set_cmap(cmap)
                except Exception:
                    continue

                if not use_clim:
                    continue

                try:
                    if log_scale and min_float is not None and max_float is not None and min_float > 0 and max_float > 0:
                        import matplotlib as mpl

                        image.set_norm(mpl.colors.LogNorm(vmin=min_float, vmax=max_float))
                    elif min_float is not None and max_float is not None:
                        import matplotlib as mpl

                        image.set_norm(mpl.colors.Normalize(vmin=min_float, vmax=max_float))
                    image.set_clim(min_float, max_float)
                except Exception:
                    continue

            try:
                if hasattr(window, "canvas") and window.canvas is not None:
                    self._request_canvas_redraw(window.canvas)
            except Exception:
                continue

    def _restore_color_settings_state(self, color_state):
        if not isinstance(color_state, dict):
            return
        try:
            from takefits.tools.color_scale import ColorMode, ColorSettingsPanel
        except Exception:
            return

        mode_map = {
            "main": ColorMode.MAIN,
            "integration": ColorMode.INTEG,
            "pv": ColorMode.PV,
            "channel": ColorMode.CHANNEL,
        }
        global_state = color_state.get("global")
        if not isinstance(global_state, dict):
            global_state = {
                "main": color_state.get("main"),
                "integration": color_state.get("integration"),
                "pv": color_state.get("pv"),
                "channel": color_state.get("channel"),
            }
        for name, mode in mode_map.items():
            incoming = global_state.get(name)
            if not isinstance(incoming, dict):
                continue
            current = dict(ColorSettingsPanel.settings.get(mode, {}) or {})
            current.update(incoming)
            ColorSettingsPanel.settings[mode] = current

        # Apply global settings first (for backward compatibility), then
        # override with per-panel image state when available.
        for mode_name in ("main", "integration", "pv", "channel"):
            settings = global_state.get(mode_name)
            if isinstance(settings, dict):
                self._apply_color_state_to_mode(mode_name, settings)

        self._restore_per_panel_color_state(color_state)

    def _serialize_contour_state_payload(self, state):
        if state is None:
            return None
        try:
            from takefits.core.contour_manager import serialize_state_to_json

            return json.loads(serialize_state_to_json(state))
        except Exception:
            return None

    def _deserialize_contour_state_payload(self, payload):
        if not isinstance(payload, dict):
            return None
        try:
            from takefits.core.contour_manager import deserialize_state_from_json

            return deserialize_state_from_json(json.dumps(payload))
        except Exception:
            return None

    def _serialize_contour_parameters_payload(self, params):
        if params is None:
            return None
        payload = {}
        for key in ("level_min", "level_max", "level_step"):
            value = getattr(params, key, None)
            if value is None:
                payload[key] = None
            else:
                try:
                    payload[key] = float(value)
                except Exception:
                    payload[key] = None
        try:
            payload["smoothing"] = float(getattr(params, "smoothing", 0.0))
        except Exception:
            payload["smoothing"] = 0.0
        try:
            payload["linewidth"] = float(getattr(params, "linewidth", 1.0))
        except Exception:
            payload["linewidth"] = 1.0
        payload["color"] = str(getattr(params, "color", "white") or "white")
        return payload

    def _deserialize_contour_parameters_payload(self, payload):
        if not isinstance(payload, dict):
            return None
        try:
            from takefits.core.contour_manager import ContourParameters

            def _opt_float(name):
                value = payload.get(name)
                if value is None:
                    return None
                try:
                    return float(value)
                except Exception:
                    return None

            smoothing = payload.get("smoothing", 0.0)
            linewidth = payload.get("linewidth", 1.0)
            try:
                smoothing = float(smoothing)
            except Exception:
                smoothing = 0.0
            try:
                linewidth = float(linewidth)
            except Exception:
                linewidth = 1.0
            color = str(payload.get("color", "white") or "white")
            return ContourParameters(
                level_min=_opt_float("level_min"),
                level_max=_opt_float("level_max"),
                level_step=_opt_float("level_step"),
                smoothing=smoothing,
                linewidth=linewidth,
                color=color,
            )
        except Exception:
            return None

    def _contour_parameters_from_generated_payload(self, generated_payload):
        if not isinstance(generated_payload, dict):
            return None
        params_payload = generated_payload.get("parameters")
        return self._deserialize_contour_parameters_payload(params_payload)

    def _capture_contour_bundle_for_layer(self, layer_id):
        if not layer_id:
            return None
        try:
            from takefits.core.contour_manager import ContourManager

            manager = ContourManager.instance()
        except Exception:
            return None
        layer = getattr(manager, "_layers", {}).get(layer_id)
        if layer is None:
            return None

        payload = {}
        params_payload = self._serialize_contour_parameters_payload(getattr(layer, "_last_parameters", None))
        if isinstance(params_payload, dict):
            payload["generated_parameters"] = params_payload

        generated_state = manager.export_layer_state(layer_id)
        generated_payload = self._serialize_contour_state_payload(generated_state)
        if isinstance(generated_payload, dict):
            payload["generated"] = generated_payload

        overlays = []
        try:
            overlay_states = list(layer.overlay_states())
        except Exception:
            overlay_states = []
        for overlay_state in overlay_states:
            overlay_payload = self._serialize_contour_state_payload(overlay_state)
            if isinstance(overlay_payload, dict):
                overlays.append(overlay_payload)
        if overlays:
            payload["overlays"] = overlays

        return payload if payload else None

    def _restore_contour_bundle_for_layer(self, layer_id, payload):
        if not layer_id or not isinstance(payload, dict):
            return False
        normalized_payload = payload
        # Backward-compatible: allow a direct contour-state payload.
        if str(payload.get("format") or "").lower() == "takefits.contour":
            normalized_payload = {"generated": payload}
        try:
            from takefits.core.contour_manager import ContourManager

            manager = ContourManager.instance()
        except Exception:
            return False
        layer = getattr(manager, "_layers", {}).get(layer_id)
        if layer is None:
            return False

        try:
            layer.clear()
        except Exception:
            pass

        restored = False
        generated_payload = normalized_payload.get("generated")
        generated_params = self._deserialize_contour_parameters_payload(
            normalized_payload.get("generated_parameters")
        )
        if generated_params is None:
            generated_params = self._contour_parameters_from_generated_payload(generated_payload)

        if generated_params is not None:
            try:
                manager.update_parameters(generated_params)
                results = manager.apply_to_layers([layer_id])
                if results.get(layer_id) is not None:
                    restored = True
                else:
                    refreshed_layer = getattr(manager, "_layers", {}).get(layer_id)
                    if refreshed_layer is not None and refreshed_layer.is_active():
                        restored = True
            except Exception:
                restored = False

        if not restored:
            generated_state = self._deserialize_contour_state_payload(generated_payload)
            if generated_state is not None:
                try:
                    manager.import_layer_state(layer_id, generated_state)
                    manager.contour_updated.emit(layer_id)
                    restored = True
                except Exception:
                    pass

        overlays = normalized_payload.get("overlays")
        if isinstance(overlays, list):
            for overlay_payload in overlays:
                overlay_state = self._deserialize_contour_state_payload(overlay_payload)
                if overlay_state is None:
                    continue
                try:
                    manager.import_overlay_state(layer_id, overlay_state)
                    restored = True
                except Exception:
                    continue

        return restored

    def _refresh_viewer_after_contour_restore(self, viewer):
        if viewer is None:
            return
        layer_id = getattr(viewer, "_contour_layer_id", None)
        refresh = getattr(viewer, "refresh_display_after_contour_update", None)
        if callable(refresh) and layer_id:
            try:
                refresh(layer_id)
                return
            except Exception:
                pass
        canvas = getattr(viewer, "canvas", None)
        if canvas is not None:
            try:
                canvas.draw_idle()
            except Exception:
                pass

    def _flush_annotation_commits_for_window(self, window):
        flusher = getattr(window, "_flush_pending_annotation_commits", None)
        if callable(flusher):
            try:
                flusher()
            except Exception:
                pass

    def _decode_marker_specs(self, payload):
        if not isinstance(payload, list):
            return None
        specs = []
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            try:
                specs.append(MarkerSpec.from_dict(entry))
            except Exception:
                continue
        return specs

    def _decode_region_specs(self, payload):
        if not isinstance(payload, list):
            return None
        specs = []
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            try:
                specs.append(RegionSpec.from_dict(entry))
            except Exception:
                continue
        return specs

    def _serialize_workspace_array_payload(self, array):
        try:
            import numpy as np

            arr = np.asarray(array)
            buffer = io.BytesIO()
            np.save(buffer, arr, allow_pickle=False)
            compressed = zlib.compress(buffer.getvalue(), level=3)
            return {
                "encoding": "npy+zlib+base64",
                "payload": base64.b64encode(compressed).decode("ascii"),
            }
        except Exception:
            return None

    def _deserialize_workspace_array_payload(self, payload):
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
            import numpy as np

            return np.load(buffer, allow_pickle=False)
        except Exception:
            return None

    def _capture_workspace_integration_data_state(self):
        payload = []
        for window in self._live_integration_windows():
            array_payload = self._serialize_workspace_array_payload(getattr(window, "integrated_data", None))
            if array_payload is None:
                continue
            entry = {
                "key": self._integration_window_color_key(window),
                "data": array_payload,
            }
            try:
                entry["title"] = str(getattr(window, "windowTitle", lambda: "")() or "")
            except Exception:
                entry["title"] = ""
            payload.append(entry)
        return payload

    def _apply_restored_integration_data_to_window(self, window, restored_array) -> bool:
        try:
            import numpy as np

            integrated = np.asarray(restored_array)
            window.integrated_data = integrated

            app_state = getattr(window, "app_state", None)
            if app_state is not None:
                app_state.data = integrated

            session = getattr(window, "action_session", None)
            if session is not None and getattr(session, "state", None) is not None:
                session.state.data = integrated
                try:
                    session.set_initial_state_seed()
                except Exception:
                    pass

            plane = str(getattr(window, "plane", "xy") or "xy").lower()
            display_data = integrated.T if plane == "zy" else integrated

            image = getattr(window, "im", None)
            if image is not None:
                image.set_data(display_data)
                try:
                    image.autoscale()
                except Exception:
                    pass

            canvas = getattr(window, "canvas", None)
            if canvas is not None:
                try:
                    canvas.draw_idle()
                except Exception:
                    pass
            return True
        except Exception:
            return False

    def _restore_workspace_integration_data_state(self, integration_state):
        if not isinstance(integration_state, list):
            return 0
        windows = self._live_integration_windows()
        if not windows:
            return 0

        ordered_entries = []
        for entry in integration_state:
            if not isinstance(entry, dict):
                continue
            ordered_entries.append(entry)

        if not ordered_entries:
            return 0

        restored = 0
        remaining_entries = list(ordered_entries)
        for window in windows:
            entry = None
            window_key = str(self._integration_window_color_key(window) or "")
            if window_key and remaining_entries:
                for idx, candidate in enumerate(remaining_entries):
                    if str(candidate.get("key") or "") == window_key:
                        entry = remaining_entries.pop(idx)
                        break
            if entry is None and remaining_entries:
                entry = remaining_entries.pop(0)
            if not isinstance(entry, dict):
                continue
            restored_array = self._deserialize_workspace_array_payload(entry.get("data"))
            if restored_array is None:
                continue
            if self._apply_restored_integration_data_to_window(window, restored_array):
                restored += 1
        return restored

    def _capture_workspace_annotation_state(self) -> dict:
        payload = {
            "schema": 1,
            "main_viewers": {},
            "integration_windows": [],
            "channel_windows": [],
        }

        for plane in ("xy", "xz", "zy"):
            viewer = self._viewer_for_plane(plane)
            if viewer is None:
                continue
            contour_bundle = self._capture_contour_bundle_for_layer(getattr(viewer, "_contour_layer_id", None))
            if contour_bundle:
                payload["main_viewers"][plane] = contour_bundle

        for window in self._live_integration_windows():
            self._flush_annotation_commits_for_window(window)
            entry = {"key": self._integration_window_color_key(window)}
            try:
                entry["title"] = str(getattr(window, "windowTitle", lambda: "")() or "")
            except Exception:
                entry["title"] = ""

            marker_snapshot = getattr(window, "_marker_specs_snapshot", None)
            if callable(marker_snapshot):
                try:
                    entry["marker_specs"] = list(marker_snapshot() or [])
                except Exception:
                    entry["marker_specs"] = []

            region_snapshot = getattr(window, "_region_specs_snapshot", None)
            if callable(region_snapshot):
                try:
                    entry["region_specs"] = list(region_snapshot() or [])
                except Exception:
                    entry["region_specs"] = []

            contour_bundle = self._capture_contour_bundle_for_layer(getattr(window, "_contour_layer_id", None))
            if contour_bundle:
                entry["contour"] = contour_bundle

            payload["integration_windows"].append(entry)

        for window in list(getattr(self, "channel_map_windows", []) or []):
            if window is None:
                continue
            self._flush_annotation_commits_for_window(window)
            entry = {"key": self._channel_window_color_key(window)}
            try:
                entry["title"] = str(getattr(window, "windowTitle", lambda: "")() or "")
            except Exception:
                entry["title"] = ""

            marker_snapshot = getattr(window, "_marker_specs_snapshot", None)
            if callable(marker_snapshot):
                try:
                    entry["marker_specs"] = list(marker_snapshot() or [])
                except Exception:
                    entry["marker_specs"] = []

            contour_bundle = self._capture_contour_bundle_for_layer(getattr(window, "_contour_layer_id", None))
            if contour_bundle:
                entry["contour"] = contour_bundle

            payload["channel_windows"].append(entry)

        return payload

    def _restore_workspace_annotation_state(self, annotation_state):
        restored = {"main_contours": 0, "integration": 0, "channel": 0}
        if not isinstance(annotation_state, dict):
            return restored

        main_viewers = annotation_state.get("main_viewers")
        if isinstance(main_viewers, dict):
            for plane in ("xy", "xz", "zy"):
                contour_payload = main_viewers.get(plane)
                if not isinstance(contour_payload, dict):
                    continue
                viewer = self._viewer_for_plane(plane)
                if viewer is None:
                    continue
                if self._restore_contour_bundle_for_layer(
                    getattr(viewer, "_contour_layer_id", None),
                    contour_payload,
                ):
                    restored["main_contours"] += 1
                    self._refresh_viewer_after_contour_restore(viewer)

        integration_entries = annotation_state.get("integration_windows")
        if isinstance(integration_entries, list):
            remaining_entries = [entry for entry in integration_entries if isinstance(entry, dict)]
            for window in self._live_integration_windows():
                entry = self._pop_workspace_entry_by_key_or_next(
                    remaining_entries,
                    self._integration_window_color_key(window),
                )
                if not isinstance(entry, dict):
                    continue

                window_restored = False
                marker_specs = self._decode_marker_specs(entry.get("marker_specs")) if "marker_specs" in entry else None
                region_specs = self._decode_region_specs(entry.get("region_specs")) if "region_specs" in entry else None
                state_changed = False

                session = getattr(window, "action_session", None)
                state = getattr(session, "state", None)
                if state is not None:
                    if marker_specs is not None:
                        try:
                            state.markers = marker_specs
                            state_changed = True
                        except Exception:
                            pass
                    if region_specs is not None:
                        try:
                            state.regions = region_specs
                            state_changed = True
                        except Exception:
                            pass

                if state_changed:
                    apply_state = getattr(window, "_apply_action_session_state_to_viewer", None)
                    if callable(apply_state):
                        try:
                            apply_state()
                            window_restored = True
                        except Exception:
                            pass

                contour_payload = entry.get("contour")
                if isinstance(contour_payload, dict):
                    if self._restore_contour_bundle_for_layer(
                        getattr(window, "_contour_layer_id", None),
                        contour_payload,
                    ):
                        window_restored = True

                if window_restored:
                    restored["integration"] += 1
                    self._refresh_viewer_after_contour_restore(window)

        channel_entries = annotation_state.get("channel_windows")
        if isinstance(channel_entries, list):
            remaining_entries = [entry for entry in channel_entries if isinstance(entry, dict)]
            for window in list(getattr(self, "channel_map_windows", []) or []):
                if window is None:
                    continue
                entry = self._pop_workspace_entry_by_key_or_next(
                    remaining_entries,
                    self._channel_window_color_key(window),
                )
                if not isinstance(entry, dict):
                    continue

                window_restored = False
                marker_specs = self._decode_marker_specs(entry.get("marker_specs")) if "marker_specs" in entry else None
                state_changed = False

                session = getattr(window, "action_session", None)
                state = getattr(session, "state", None)
                if state is not None and marker_specs is not None:
                    try:
                        state.markers = marker_specs
                        state_changed = True
                    except Exception:
                        pass

                if state_changed:
                    apply_state = getattr(window, "_apply_action_session_state_to_viewer", None)
                    if callable(apply_state):
                        try:
                            apply_state()
                            window_restored = True
                        except Exception:
                            pass

                contour_payload = entry.get("contour")
                if isinstance(contour_payload, dict):
                    if self._restore_contour_bundle_for_layer(
                        getattr(window, "_contour_layer_id", None),
                        contour_payload,
                    ):
                        window_restored = True

                if window_restored:
                    restored["channel"] += 1
                    self._refresh_viewer_after_contour_restore(window)

        return restored

    def _build_workspace_payload(self) -> dict:
        records, _cursor, _total = self._session_records_up_to_cursor()
        return {
            "type": "takefits.workspace",
            "version": 1,
            "saved_at": self._utc_timestamp(),
            "history": self._serialize_action_records(records),
            "cursor": len(records),
            "source": self._dataset_descriptor(),
            "workspace_state": {
                "shared_cursor": self._capture_shared_cursor_snapshot(),
                "view_limits": self._capture_view_limits(),
                "world_ranges": self._capture_world_ranges(),
                "wcs_display_frame": self.get_wcs_display_frame(),
                "wcs_decimal": self.get_wcs_decimal_mode(),
                "pv_state": self._capture_pv_workspace_state(),
                "spectrum_state": self._capture_spectrum_workspace_state(),
                "baseline_state": self._capture_baseline_workspace_state(),
                "clump_state": self._capture_clump_workspace_state(),
                "ui_state": self._capture_workspace_ui_state(),
                "window_z_order": self._capture_workspace_window_z_order(),
                "integration_data_state": self._capture_workspace_integration_data_state(),
                "geometry_state": self._capture_workspace_geometry_state(),
                "panel_state": self._capture_panel_visibility_state(),
                "color_settings": self._capture_color_settings_state(),
                "colorbar_state": self._capture_workspace_colorbar_state(),
                "annotation_state": self._capture_workspace_annotation_state(),
            },
        }

    def _log_workspace_restore_diagnostics(self, diagnostics: dict):
        if not isinstance(diagnostics, dict):
            return
        debug_flag = str(os.environ.get("TAKEFITS_DEBUG_WORKSPACE", "")).strip().lower()
        if debug_flag not in {"1", "true", "yes", "on"}:
            return
        source_path = str(diagnostics.get("source_filepath") or "")
        current_path = str(diagnostics.get("current_filepath") or "")
        source_name = os.path.basename(source_path) if source_path else "<unknown>"
        current_name = os.path.basename(current_path) if current_path else "<unknown>"
        print(
            "[takefits] Workspace load diagnostics: "
            f"same_dataset={bool(diagnostics.get('same_dataset'))} ({diagnostics.get('dataset_reason')}), "
            f"wcs_compatible={bool(diagnostics.get('wcs_compatible'))} ({diagnostics.get('wcs_reason')}), "
            f"has_wcs_signature={bool(diagnostics.get('has_wcs_signature'))}, "
            f"saved={source_name}, current={current_name}"
        )

    def _restore_workspace_panel_visibility(self, panel_state):
        if not isinstance(panel_state, dict):
            return

        if panel_state.get("control_panel_visible", True):
            self.show_control_panel()
        else:
            self.hide_control_panel()
        if hasattr(self, "menu_bar") and hasattr(self.menu_bar, "control_panel_action"):
            self.menu_bar.control_panel_action.setChecked(bool(panel_state.get("control_panel_visible", True)))
        if panel_state.get("range_panel_visible", True):
            self.show_range_panel()
        else:
            self.hide_range_panel()
        if hasattr(self, "menu_bar") and hasattr(self.menu_bar, "range_panel_action"):
            self.menu_bar.range_panel_action.setChecked(bool(panel_state.get("range_panel_visible", True)))

        sub_state = panel_state.get("subwindows")
        if isinstance(sub_state, dict) and self.data.ndim > 2:
            if "xz" in sub_state:
                xz_visible = bool(sub_state.get("xz"))
                if xz_visible and getattr(self, "subwindow1", None) is None:
                    self.ensure_subwindow1()
                if getattr(self, "subwindow1", None) is not None:
                    self.subwindow1.setVisible(xz_visible)
                if hasattr(self, "menu_bar") and hasattr(self.menu_bar, "sub1_action"):
                    self.menu_bar.sub1_action.setChecked(xz_visible)
            if "zy" in sub_state:
                zy_visible = bool(sub_state.get("zy"))
                if zy_visible and getattr(self, "subwindow2", None) is None:
                    self.ensure_subwindow2()
                if getattr(self, "subwindow2", None) is not None:
                    self.subwindow2.setVisible(zy_visible)
                if hasattr(self, "menu_bar") and hasattr(self.menu_bar, "sub2_action"):
                    self.menu_bar.sub2_action.setChecked(zy_visible)

        cp = getattr(self, "control_panel", None)
        tool_state = panel_state.get("tool_panels")
        if isinstance(tool_state, dict) and cp is not None:
            openers = {
                "color_settings_panel": "open_color_settings",
                "scaling_panel": "open_scaling_panel",
                "unit_conversion_panel": "open_unit_conversion_panel",
                "integ_settings_panel": "open_integ_settings",
                "chmap_settings_panel": "open_chmap_settings",
                "smooth_settings_panel": "open_smooth_settings",
                "baseline_panel": "open_baseline_panel",
                "spec_window": "open_spec_window",
                "pvd_panel": "open_pvd_settings",
                "mask_settings_panel": "open_mask_settings",
                "contour_panel": "open_contour_panel",
                "arithmetic_panel": "open_arithmetic_panel",
                "clump_finding_panel": "open_clump_finding_panel",
            }
            for attr, opener_name in openers.items():
                if not bool(tool_state.get(attr)):
                    continue
                opener = getattr(cp, opener_name, None)
                if callable(opener):
                    try:
                        opener()
                    except Exception:
                        continue

        if bool(panel_state.get("marker_panel_visible")):
            open_marker = getattr(self, "open_marker_panel", None)
            if callable(open_marker):
                try:
                    open_marker()
                except Exception:
                    pass
        if bool(panel_state.get("regrid_panel_visible")):
            open_regrid = getattr(self, "open_regrid_panel", None)
            if callable(open_regrid):
                try:
                    open_regrid()
                except Exception:
                    pass

    def _resync_viewer_slices_from_shared_state(self):
        if getattr(self, "data", None) is None or getattr(self.data, "ndim", 0) < 3:
            return
        try:
            max_z = max(0, self.data.shape[-3] - 1)
            max_y = max(0, self.data.shape[-2] - 1)
            max_x = max(0, self.data.shape[-1] - 1)
            zpix = max(0, min(int(round(float(self._get_shared_zpix()))), max_z))
            ypix = max(0, min(int(round(float(self._get_shared_ypix()))), max_y))
            xpix = max(0, min(int(round(float(self._get_shared_xpix()))), max_x))
        except Exception:
            return

        try:
            self.update_channel("xy", zpix)
        except Exception:
            pass

        subwindow1 = getattr(self, "subwindow1", None)
        if subwindow1 is not None and not subwindow1.isHidden():
            try:
                subwindow1.update_channel("xz", ypix)
            except Exception:
                pass

        subwindow2 = getattr(self, "subwindow2", None)
        if subwindow2 is not None and not subwindow2.isHidden():
            try:
                subwindow2.update_channel("zy", xpix)
            except Exception:
                pass

        sync_controls = getattr(self, "_sync_channel_controls", None)
        if callable(sync_controls):
            try:
                sync_controls(self, zpix)
            except Exception:
                pass
            if subwindow1 is not None and not subwindow1.isHidden():
                try:
                    sync_controls(subwindow1, ypix)
                except Exception:
                    pass
            if subwindow2 is not None and not subwindow2.isHidden():
                try:
                    sync_controls(subwindow2, xpix)
                except Exception:
                    pass

    def _sync_hpbw_overlay_with_current_header(self, window) -> bool:
        if window is None:
            return False
        hpbw = getattr(window, "hpbw", None)
        if hpbw is None:
            return False

        header = getattr(window, "header", None)
        if header is None:
            fits_viewer = getattr(window, "fits_viewer", None)
            header = getattr(fits_viewer, "header", None) if fits_viewer is not None else None
        if header is None:
            return False

        try:
            hpbw.header = header
        except Exception:
            pass

        refresh = getattr(hpbw, "refresh_geometry_from_header", None)
        if callable(refresh):
            try:
                refresh()
            except Exception:
                pass

        redraw_overlay = getattr(window, "redraw_main_overlay_and_blit", None)
        if callable(redraw_overlay):
            try:
                redraw_overlay()
                return True
            except Exception:
                pass

        canvas = getattr(window, "canvas", None)
        if canvas is not None:
            try:
                self._request_canvas_redraw(canvas)
                return True
            except Exception:
                pass
        return False

    def _resync_analysis_windows_after_restore(self):
        for window in self._live_integration_windows():
            self._sync_hpbw_overlay_with_current_header(window)
            hook = getattr(window, "resync_after_workspace_restore", None)
            if callable(hook):
                try:
                    hook()
                    continue
                except Exception:
                    pass
            canvas = getattr(window, "canvas", None)
            if canvas is not None:
                try:
                    self._request_canvas_redraw(canvas)
                except Exception:
                    pass

        control_panel = getattr(self, "control_panel", None)
        if control_panel is not None:
            for attr in self._workspace_tool_panel_attrs():
                if attr == "color_settings_panel":
                    continue
                panel = getattr(control_panel, attr, None)
                if panel is None:
                    continue
                hook = getattr(panel, "resync_after_workspace_restore", None)
                if not callable(hook):
                    continue
                try:
                    hook()
                except Exception:
                    pass

        for window in list(getattr(self, "channel_map_windows", []) or []):
            if window is None:
                continue
            self._sync_hpbw_overlay_with_current_header(window)
            hook = getattr(window, "resync_after_workspace_restore", None)
            if callable(hook):
                try:
                    hook()
                    continue
                except Exception:
                    pass
            canvas = getattr(window, "canvas", None)
            if canvas is not None:
                try:
                    self._request_canvas_redraw(canvas)
                except Exception:
                    pass

    def _restore_workspace_view_limits(self, view_limits):
        if not isinstance(view_limits, dict):
            return False
        applied = False
        for plane in ("xy", "xz", "zy"):
            limits = view_limits.get(plane)
            if not isinstance(limits, dict):
                continue
            xlim = limits.get("xlim")
            ylim = limits.get("ylim")
            if not (
                isinstance(xlim, (list, tuple)) and len(xlim) == 2
                and isinstance(ylim, (list, tuple)) and len(ylim) == 2
            ):
                continue
            viewer = self._viewer_for_plane(plane)
            if viewer is None or not hasattr(viewer, "ax"):
                continue
            try:
                x0, x1 = float(xlim[0]), float(xlim[1])
                y0, y1 = float(ylim[0]), float(ylim[1])
            except Exception:
                continue
            try:
                viewer.ax.set_xlim(x0, x1)
                viewer.ax.set_ylim(y0, y1)
                if hasattr(viewer, "overlay_ax") and viewer.overlay_ax is not None:
                    viewer.overlay_ax.set_position(viewer.ax.get_position())
                invalidate_workspace_restore_blit_cache(viewer)
                if hasattr(viewer, "canvas") and viewer.canvas is not None:
                    self._request_canvas_redraw(viewer.canvas)
                self.update_ranges(plane, (x0, x1), (y0, y1))
                applied = True
            except Exception:
                continue
        return applied

    def save_recipe_dialog(self):
        if not hasattr(self, "action_session") or self.action_session is None:
            return
        self._flush_pending_annotation_commits()
        default_dir = os.path.dirname(getattr(self, "filename_path", "")) or os.getcwd()
        default_name = os.path.splitext(os.path.basename(getattr(self, "filename", "session") or "session"))[0]
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Recipe",
            os.path.join(default_dir, f"{default_name}.recipe.json"),
            "Recipe Files (*.recipe.json *.json);;All Files (*)",
        )
        if not path:
            return
        if not path.lower().endswith((".recipe.json", ".json")):
            path += ".recipe.json"
        records, cursor, total = self._session_records_up_to_cursor()
        payload = {
            "type": "takefits.recipe",
            "version": 1,
            "saved_at": self._utc_timestamp(),
            "history": self._serialize_action_records(records),
            "cursor": len(records),
            "source": self._dataset_descriptor(),
        }
        try:
            outfile = self._write_json_payload(path, payload)
            dropped = max(0, total - cursor)
            suffix = f"\nExcluded {dropped} redo action(s)." if dropped else ""
            QMessageBox.information(
                self,
                "Recipe Saved",
                f"Saved recipe with {len(records)} action(s).\n{outfile}{suffix}",
            )
        except Exception as exc:
            QMessageBox.warning(self, "Recipe", f"Failed to save recipe: {exc}")
        finally:
            self._refresh_undo_redo_actions()

    def save_workspace_dialog(self):
        if not hasattr(self, "action_session") or self.action_session is None:
            return
        default_dir = os.path.dirname(getattr(self, "filename_path", "")) or os.getcwd()
        default_name = os.path.splitext(os.path.basename(getattr(self, "filename", "session") or "session"))[0]
        current_target = str(getattr(self, "_workspace_save_path", "") or "").strip()
        if current_target:
            default_target = current_target
        else:
            default_target = os.path.join(default_dir, f"{default_name}.workspace.json")
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Workspace",
            default_target,
            "Workspace Files (*.workspace.json *.json);;All Files (*)",
        )
        if not path:
            return False
        return self.save_workspace_to_path(path, show_result_dialog=True)

    def save_workspace(self):
        target = str(getattr(self, "_workspace_save_path", "") or "").strip()
        if target:
            return self.save_workspace_to_path(target, show_result_dialog=False)
        return self.save_workspace_dialog()

    def _show_workspace_status_message(self, message: str, *, timeout_ms: int = 0):
        text = str(message or "").strip()
        if not text:
            return
        duration = int(timeout_ms or 0)
        if duration <= 0:
            duration = 2000
        try:
            QToolTip.showText(
                QCursor.pos(),
                text,
                self,
                self.rect(),
                duration,
            )
        except Exception:
            return

    def set_workspace_save_path(self, path: str | None):
        text = str(path or "").strip()
        if not text:
            self._workspace_save_path = None
            return
        try:
            self._workspace_save_path = os.path.abspath(text)
        except Exception:
            self._workspace_save_path = text

    def save_workspace_to_path(self, path: str, *, show_result_dialog: bool = True) -> bool:
        if not hasattr(self, "action_session") or self.action_session is None:
            return False
        target = str(path or "").strip()
        if not target:
            return False
        if not target.lower().endswith((".workspace.json", ".json")):
            target += ".workspace.json"

        self._flush_pending_annotation_commits()
        payload = self._build_workspace_payload()
        self._show_workspace_status_message(
            f"Saving workspace... {os.path.basename(target)}",
            timeout_ms=1500,
        )
        try:
            QApplication.processEvents()
        except Exception:
            pass
        busy_cursor_set = False
        try:
            QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
            busy_cursor_set = True
        except Exception:
            busy_cursor_set = False
        try:
            outfile = self._write_json_payload(target, payload)
            self.set_workspace_save_path(outfile)
            action_count = len(payload.get("history", []))
            self._show_workspace_status_message(
                f"Workspace saved: {os.path.basename(outfile)}",
                timeout_ms=3000,
            )
            if show_result_dialog:
                QMessageBox.information(
                    self,
                    "Workspace Saved",
                    f"Saved workspace with {action_count} action(s).\n{outfile}",
                )
            return True
        except Exception as exc:
            self._show_workspace_status_message("Workspace save failed.", timeout_ms=5000)
            QMessageBox.warning(self, "Workspace", f"Failed to save workspace: {exc}")
            return False
        finally:
            if busy_cursor_set:
                try:
                    QApplication.restoreOverrideCursor()
                except Exception:
                    pass
            self._refresh_undo_redo_actions()

    def load_recipe_dialog(self):
        if not hasattr(self, "action_session") or self.action_session is None:
            return
        default_dir = os.path.dirname(getattr(self, "filename_path", "")) or os.getcwd()
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Recipe",
            default_dir,
            "Recipe Files (*.recipe.json *.json);;All Files (*)",
        )
        if not path:
            return
        answer = QMessageBox.question(
            self,
            "Load Recipe",
            "Current action history will be replaced by the loaded recipe.\nContinue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._flush_pending_annotation_commits()
        try:
            self.action_session.load_history(path, replay=True, replace=True)
            self._apply_action_session_state_to_viewers()
            restored = self._restore_analysis_windows_from_history()
            self._resync_analysis_windows_after_restore()
            QMessageBox.information(
                self,
                "Recipe Loaded",
                (
                    f"Loaded recipe with {len(self.action_session.history)} action(s).\n"
                    f"Restored windows: Integration={restored.get('integration', 0)}, "
                    f"Channel Map={restored.get('channel_map', 0)}"
                ),
            )
        except Exception as exc:
            QMessageBox.warning(self, "Recipe", f"Failed to load recipe: {exc}")
        finally:
            self._refresh_undo_redo_actions()

    def load_workspace_dialog(self):
        if not hasattr(self, "action_session") or self.action_session is None:
            return
        default_dir = os.path.dirname(getattr(self, "filename_path", "")) or os.getcwd()
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Workspace",
            default_dir,
            "Workspace Files (*.workspace.json *.json);;All Files (*)",
        )
        if not path:
            return
        self.load_workspace_from_path(path, confirm_replace=True, show_result_dialog=True)

    def load_workspace_from_path(
        self,
        path: str,
        *,
        confirm_replace: bool = True,
        show_result_dialog: bool = True,
    ) -> bool:
        if not hasattr(self, "action_session") or self.action_session is None:
            return False
        if not path:
            return False

        if confirm_replace:
            answer = QMessageBox.question(
                self,
                "Load Workspace",
                "Current action history will be replaced by the loaded workspace.\nContinue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return False

        self._flush_pending_annotation_commits()
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:
            QMessageBox.warning(self, "Workspace", f"Failed to read workspace: {exc}")
            return False
        if not isinstance(payload, dict) or not isinstance(payload.get("history"), list):
            QMessageBox.warning(self, "Workspace", "Invalid workspace format.")
            return False

        try:
            current_shape = [int(v) for v in getattr(self.data, "shape", [])]
        except Exception:
            current_shape = []
        diagnostics = build_workspace_restore_diagnostics(
            payload,
            current_path=str(getattr(self, "filename_path", "") or ""),
            current_shape=current_shape,
            current_signature=self._build_wcs_signature(getattr(self, "wcs", None)),
            data_ndim=int(getattr(getattr(self, "data", None), "ndim", 0) or 0),
        )
        same_dataset = bool(diagnostics.get("same_dataset", False))
        wcs_compatible = bool(diagnostics.get("wcs_compatible", False))
        has_wcs_signature = bool(diagnostics.get("has_wcs_signature", False))
        self._log_workspace_restore_diagnostics(diagnostics)
        workspace_state = payload.get("workspace_state") if isinstance(payload.get("workspace_state"), dict) else {}
        colorbar_state = workspace_state.get("colorbar_state")
        saved_display_frame = workspace_state.get("wcs_display_frame")
        saved_decimal = workspace_state.get("wcs_decimal")
        if isinstance(saved_decimal, bool):
            self.set_wcs_decimal_mode(saved_decimal, refresh=False)
        if isinstance(saved_display_frame, str):
            self.set_wcs_display_frame(saved_display_frame, refresh=False)
        self._restore_workspace_colorbar_state(colorbar_state, apply_to_open_windows=False)
        preferred_cursor = workspace_state.get("shared_cursor")
        if not isinstance(preferred_cursor, dict):
            preferred_cursor = None
        elif not same_dataset:
            world_keys = ("world_x", "world_y", "world_z", "world_s")
            if not any(preferred_cursor.get(key) not in (None, "") for key in world_keys):
                preferred_cursor = None
            else:
                preferred_cursor = dict(preferred_cursor)
                for key in ("xpix", "ypix", "zpix", "cursor_x", "cursor_y"):
                    preferred_cursor.pop(key, None)

        self._set_workspace_colorbar_restore_in_progress(True)
        self._begin_workspace_restore_canvas_redraw_batch()
        try:
            self.action_session.load_history(path, replay=True, replace=True)
            self._apply_action_session_state_to_viewers(preferred_cursor=preferred_cursor)
            restored = self._restore_analysis_windows_from_history()
            restored_integration_data = 0
            if same_dataset:
                restored_integration_data = self._restore_workspace_integration_data_state(
                    workspace_state.get("integration_data_state")
                )
            self._restore_color_settings_state(workspace_state.get("color_settings"))
            self._restore_workspace_panel_visibility(workspace_state.get("panel_state"))
            self._resync_viewer_slices_from_shared_state()
            restored_pv = self._restore_pv_workspace_state(workspace_state.get("pv_state"))
            restored_spectrum = self._restore_spectrum_workspace_state(workspace_state.get("spectrum_state"))
            restored_baseline = self._restore_baseline_workspace_state(workspace_state.get("baseline_state"))
            clump_panel_visible = True
            panel_state = workspace_state.get("panel_state")
            if isinstance(panel_state, dict):
                tool_state = panel_state.get("tool_panels")
                if isinstance(tool_state, dict):
                    clump_panel_visible = bool(tool_state.get("clump_finding_panel", False))
            restored_clump = self._restore_clump_workspace_state(
                workspace_state.get("clump_state"),
                ensure_panel=True,
                keep_visible=clump_panel_visible,
            )
            self._restore_workspace_geometry_state(
                workspace_state.get("geometry_state"),
                allow_window_axis_limits=same_dataset,
            )
            restored_ui_widgets = self._restore_workspace_ui_state(workspace_state.get("ui_state"))
            restored_view = False
            restored_world = False
            restore_mode = compute_range_restore_mode(
                same_dataset=same_dataset,
                wcs_compatible=wcs_compatible,
                has_wcs_signature=has_wcs_signature,
            )
            if restore_mode == "view_then_world":
                restored_view = self._restore_workspace_view_limits(workspace_state.get("view_limits"))
                if not restored_view:
                    restored_world = self._restore_workspace_world_ranges(workspace_state.get("world_ranges"))
            elif restore_mode == "world_only":
                restored_world = self._restore_workspace_world_ranges(workspace_state.get("world_ranges"))
            restored_annotations = self._restore_workspace_annotation_state(workspace_state.get("annotation_state"))
            self._resync_analysis_windows_after_restore()
            self._restore_workspace_window_z_order(workspace_state.get("window_z_order"))
            self._refresh_wcs_display_strings()
            self._refresh_colorbar_layout_after_workspace_restore(colorbar_state)
            self._flush_workspace_restore_canvas_redraw_batch()
            self._refresh_cursor_overlay_after_workspace_restore(
                preferred_cursor=preferred_cursor,
                defer_retry=False,
            )
            self._reset_shared_view_history_to_current(reason="workspace_restore")
            try:
                QApplication.processEvents()
            except Exception:
                pass

            status_line = build_workspace_restore_status_line(
                same_dataset=same_dataset,
                wcs_compatible=wcs_compatible,
                has_wcs_signature=has_wcs_signature,
                restored_view=restored_view,
                restored_world=restored_world,
            )
            if show_result_dialog:
                QMessageBox.information(
                    self,
                    "Workspace Loaded",
                    (
                        f"Loaded workspace with {len(self.action_session.history)} action(s).\n"
                        f"{status_line}\n"
                        f"Restored windows: Integration={restored.get('integration', 0)}, "
                        f"IntegrationData={restored_integration_data}, "
                        f"Channel Map={restored.get('channel_map', 0)}, "
                        f"Annotations(main/integration/channel)="
                        f"{restored_annotations.get('main_contours', 0)}/"
                        f"{restored_annotations.get('integration', 0)}/"
                        f"{restored_annotations.get('channel', 0)}, "
                        f"PV={'yes' if restored_pv else 'no'}, "
                        f"Spectrum={'yes' if restored_spectrum else 'no'}, "
                        f"Baseline={'yes' if restored_baseline else 'no'}, "
                        f"Clump={'yes' if restored_clump else 'no'}, "
                        f"UIWidgets={restored_ui_widgets}"
                    ),
                )
            self.set_workspace_save_path(path)
            return True
        except Exception as exc:
            self._flush_workspace_restore_canvas_redraw_batch()
            QMessageBox.warning(self, "Workspace", f"Failed to load workspace: {exc}")
            return False
        finally:
            self._flush_workspace_restore_canvas_redraw_batch()
            self._set_workspace_colorbar_restore_in_progress(False)
            self._refresh_undo_redo_actions()
            self._refresh_view_navigation_actions()

    def _ensure_integration_panel_for_restore(self):
        control_panel = getattr(self, "control_panel", None)
        if control_panel is None:
            return None
        panel = getattr(control_panel, "integ_settings_panel", None)
        if panel is not None:
            return panel
        try:
            from takefits.tools.integration import IntegSettingsPanel
            panel = IntegSettingsPanel(self, self.subwindows)
            panel.hide()
            try:
                panel.destroyed.connect(control_panel.on_integ_settings_closed)
            except Exception:
                pass
            control_panel.integ_settings_panel = panel
            return panel
        except Exception:
            return None

    def _ensure_channel_map_panel_for_restore(self):
        control_panel = getattr(self, "control_panel", None)
        if control_panel is None:
            return None
        panel = getattr(control_panel, "chmap_settings_panel", None)
        if panel is not None:
            return panel
        try:
            from takefits.tools.channel_map import ChannelMapSettingPanel
            panel = ChannelMapSettingPanel(self, self.subwindows)
            panel.hide()
            try:
                panel.destroyed.connect(control_panel.on_chmap_settings_closed)
            except Exception:
                pass
            control_panel.chmap_settings_panel = panel
            return panel
        except Exception:
            return None

    def _close_restorable_result_windows(self):
        for window_ref in list(getattr(self, "integ_result_windows", []) or []):
            window = window_ref() if callable(window_ref) else window_ref
            if window is None:
                continue
            try:
                window.close()
            except Exception:
                continue
        for window in list(getattr(self, "channel_map_windows", []) or []):
            if window is None:
                continue
            try:
                window.close()
            except Exception:
                continue

    def _build_window_restore_replay_session(self):
        source_session = getattr(self, "action_session", None)
        if source_session is None:
            return None
        registry = getattr(source_session, "registry", None)
        if registry is None:
            return None
        try:
            replay_session = ActionSession(registry=registry, state=None)
        except Exception:
            return None

        seed_state = getattr(source_session, "_initial_state_seed", None)
        if seed_state is None:
            seed_state = getattr(source_session, "state", None)
        if seed_state is None:
            return None
        try:
            replay_session.set_initial_state_seed(seed_state)
            replay_session.reset_to_initial()
        except Exception:
            return None
        return replay_session

    def _restore_analysis_windows_from_history(self):
        session = getattr(self, "action_session", None)
        if session is None:
            return {"integration": 0, "channel_map": 0}

        try:
            records = list(session.history[: session.cursor])
        except Exception:
            records = list(getattr(session, "history", []) or [])

        restore_action_indexes = []
        has_integration_actions = False
        has_channel_map_actions = False
        for idx, record in enumerate(records):
            action_name = str(getattr(record, "action", "") or "").strip().lower()
            if action_name == "compute_moment":
                has_integration_actions = True
                restore_action_indexes.append(idx)
            elif action_name == "compute_channel_map":
                has_channel_map_actions = True
                restore_action_indexes.append(idx)

        if not restore_action_indexes:
            return {"integration": 0, "channel_map": 0}

        self._close_restorable_result_windows()

        restored_integration = 0
        restored_channel_map = 0
        integration_panel = self._ensure_integration_panel_for_restore() if has_integration_actions else None
        channel_map_panel = self._ensure_channel_map_panel_for_restore() if has_channel_map_actions else None

        replay_session = self._build_window_restore_replay_session()
        last_restore_index = int(restore_action_indexes[-1])

        for idx, record in enumerate(records):
            if idx > last_restore_index:
                break
            action_name = str(getattr(record, "action", "") or "").strip().lower()
            params = dict(getattr(record, "params", {}) or {})

            if action_name == "compute_moment":
                if integration_panel is not None and hasattr(integration_panel, "restore_window_from_action_params"):
                    state_override = getattr(replay_session, "state", None) if replay_session is not None else None
                    try:
                        if integration_panel.restore_window_from_action_params(
                            params,
                            app_state_override=state_override,
                        ):
                            restored_integration += 1
                    except Exception:
                        pass
                continue

            if action_name == "compute_channel_map":
                if channel_map_panel is not None and hasattr(channel_map_panel, "restore_window_from_action_params"):
                    state_override = getattr(replay_session, "state", None) if replay_session is not None else None
                    try:
                        if channel_map_panel.restore_window_from_action_params(
                            params,
                            app_state_override=state_override,
                        ):
                            restored_channel_map += 1
                    except Exception:
                        pass
                continue

            if replay_session is None:
                continue
            try:
                replay_session.execute(action_name, **params)
            except Exception:
                continue

        return {"integration": restored_integration, "channel_map": restored_channel_map}

    def undo_last_action(self):
        owner = self._active_analysis_owner()
        if owner is not self:
            try:
                owner.undo_last_action()
            finally:
                self._refresh_undo_redo_actions()
                self._refresh_view_navigation_actions()
            return
        session = getattr(self, "action_session", None)
        if session is None:
            return
        preserve_cursor = self._capture_shared_cursor_snapshot()
        self._flush_pending_annotation_commits()
        if not session.can_undo():
            self._refresh_undo_redo_actions()
            return
        try:
            session.undo()
            self._apply_action_session_state_to_viewers(preferred_cursor=preserve_cursor)
        except Exception as exc:
            QMessageBox.warning(self, "Undo", f"Failed to undo action: {exc}")
        finally:
            self._refresh_undo_redo_actions()
            self._refresh_view_navigation_actions()

    def redo_last_action(self):
        owner = self._active_analysis_owner()
        if owner is not self:
            try:
                owner.redo_last_action()
            finally:
                self._refresh_undo_redo_actions()
                self._refresh_view_navigation_actions()
            return
        session = getattr(self, "action_session", None)
        if session is None:
            return
        preserve_cursor = self._capture_shared_cursor_snapshot()
        self._flush_pending_annotation_commits()
        if not session.can_redo():
            self._refresh_undo_redo_actions()
            return
        try:
            session.redo()
            self._apply_action_session_state_to_viewers(preferred_cursor=preserve_cursor)
        except Exception as exc:
            QMessageBox.warning(self, "Redo", f"Failed to redo action: {exc}")
        finally:
            self._refresh_undo_redo_actions()
            self._refresh_view_navigation_actions()

    def view_back(self):
        owner = self._active_view_navigation_owner()
        if owner is not self:
            callback = getattr(owner, "view_back", None)
            if callable(callback):
                cursor_before = self._view_navigation_cursor_for_owner(owner)
                try:
                    result = callback()
                finally:
                    self._refresh_view_navigation_actions()
                if isinstance(result, bool):
                    return result
                cursor_after = self._view_navigation_cursor_for_owner(owner)
                return cursor_after != cursor_before
            self._refresh_view_navigation_actions()
            return False
        can_back, _ = self._shared_view_history_state()
        if not can_back:
            self._refresh_view_navigation_actions()
            return False
        history = list(getattr(self, "_shared_view_history", []) or [])
        index = int(getattr(self, "_shared_view_history_index", -1))
        if not history or index <= 0:
            self._refresh_view_navigation_actions()
            return False
        previous_index = index
        index -= 1
        self._shared_view_history_index = index
        applied = False
        try:
            applied = self._apply_shared_view_history_entry(history[index])
        except Exception:
            applied = False
        if not applied:
            self._shared_view_history_index = previous_index
        self._refresh_view_navigation_actions()
        return bool(applied)

    def view_forward(self):
        owner = self._active_view_navigation_owner()
        if owner is not self:
            callback = getattr(owner, "view_forward", None)
            if callable(callback):
                cursor_before = self._view_navigation_cursor_for_owner(owner)
                try:
                    result = callback()
                finally:
                    self._refresh_view_navigation_actions()
                if isinstance(result, bool):
                    return result
                cursor_after = self._view_navigation_cursor_for_owner(owner)
                return cursor_after != cursor_before
            self._refresh_view_navigation_actions()
            return False
        _, can_forward = self._shared_view_history_state()
        if not can_forward:
            self._refresh_view_navigation_actions()
            return False
        history = list(getattr(self, "_shared_view_history", []) or [])
        index = int(getattr(self, "_shared_view_history_index", -1))
        if not history or index < 0 or index >= (len(history) - 1):
            self._refresh_view_navigation_actions()
            return False
        previous_index = index
        index += 1
        self._shared_view_history_index = index
        applied = False
        try:
            applied = self._apply_shared_view_history_entry(history[index])
        except Exception:
            applied = False
        if not applied:
            self._shared_view_history_index = previous_index
        self._refresh_view_navigation_actions()
        return bool(applied)

    def _build_region_payload_from_state(self):
        state = getattr(self.action_session, "state", None)
        region_specs = list(getattr(state, "regions", []) or []) if state is not None else []
        return build_region_payload_from_specs(region_specs, default_plane="xy")

    def _build_marker_payload_from_state(self):
        state = getattr(self.action_session, "state", None)
        marker_specs = list(getattr(state, "markers", []) or []) if state is not None else []
        return build_marker_payload_from_specs(marker_specs)

    def _apply_action_session_state_to_viewers(self, preferred_cursor=None):
        state = getattr(self.action_session, "state", None)
        if state is None or getattr(state, "data", None) is None:
            return
        self._suspend_action_recording = True
        try:
            self.app_state = state
            self.data = state.data
            self.header = state.header
            self.wcs = state.wcs
            FITSViewer.data = state.data
            FITSViewer.header = state.header
            FITSViewer.wcs = state.wcs

            viewers = [self] + list(getattr(self, "subwindows", []))
            for viewer in viewers:
                if viewer is None:
                    continue
                viewer.data = state.data
                viewer.header = state.header
                viewer.wcs = state.wcs
                if hasattr(viewer, "converter") and viewer.converter is not None:
                    viewer.converter.wcs = state.wcs
                if hasattr(viewer, "format_pix") and viewer.format_pix is not None:
                    viewer.format_pix.wcs = state.wcs
                if hasattr(viewer, "update_cube"):
                    viewer.update_cube()
                self._sync_hpbw_overlay_with_current_header(viewer)

            def _safe_int(value, default=0):
                try:
                    return int(round(float(value)))
                except Exception:
                    return int(default)

            preferred = dict(preferred_cursor) if isinstance(preferred_cursor, dict) else {}
            world_mapped = self._cursor_world_to_pixel_snapshot(preferred, getattr(state, "wcs", None))
            if isinstance(world_mapped, dict):
                for key, value in world_mapped.items():
                    current = preferred.get(key)
                    if current is None:
                        preferred[key] = value
                        continue
                    if isinstance(current, str) and not current.strip():
                        preferred[key] = value
                        continue
                    try:
                        current_num = float(current)
                    except Exception:
                        continue
                    if not math.isfinite(current_num):
                        preferred[key] = value
            cursor_state = getattr(state, "cursor", None)
            raw_x = preferred.get("xpix", getattr(cursor_state, "xpix", 0))
            raw_y = preferred.get("ypix", getattr(cursor_state, "ypix", 0))
            raw_z = preferred.get(
                "zpix",
                getattr(state, "current_z", getattr(cursor_state, "zpix", 0)),
            )
            raw_cursor_x = preferred.get("cursor_x", raw_x)
            raw_cursor_y = preferred.get("cursor_y", raw_y)

            xpix = None
            ypix = None
            if state.data.ndim >= 3:
                max_z = max(0, state.data.shape[-3] - 1)
                target_z = max(0, min(_safe_int(raw_z, 0), max_z))
                try:
                    state.current_z = target_z
                except Exception:
                    pass
                if cursor_state is not None:
                    try:
                        cursor_state.zpix = target_z
                    except Exception:
                        pass
                if state.data.ndim >= 2:
                    max_x = max(0, state.data.shape[-1] - 1)
                    max_y = max(0, state.data.shape[-2] - 1)
                    xpix = max(0, min(_safe_int(raw_x, 0), max_x))
                    ypix = max(0, min(_safe_int(raw_y, 0), max_y))
                    if cursor_state is not None:
                        try:
                            cursor_state.xpix = xpix
                            cursor_state.ypix = ypix
                        except Exception:
                            pass
                    self._update_shared_pix(xpix, ypix, target_z)
                sync_controls = getattr(self, "_sync_channel_controls", None)
                if callable(sync_controls):
                    sync_controls(self, target_z)
                self.update_channel("xy", target_z)
            else:
                self.im.set_data(state.data)
                self.canvas.draw_idle()

            if state.data.ndim >= 2:
                if xpix is None or ypix is None:
                    max_x = max(0, state.data.shape[-1] - 1)
                    max_y = max(0, state.data.shape[-2] - 1)
                    xpix = max(0, min(_safe_int(raw_x, 0), max_x))
                    ypix = max(0, min(_safe_int(raw_y, 0), max_y))
                    if cursor_state is not None:
                        try:
                            cursor_state.xpix = xpix
                            cursor_state.ypix = ypix
                        except Exception:
                            pass
                self._set_shared_xpix(xpix)
                self._set_shared_ypix(ypix)
                try:
                    cursor_x = float(raw_cursor_x)
                except Exception:
                    cursor_x = float(xpix)
                try:
                    cursor_y = float(raw_cursor_y)
                except Exception:
                    cursor_y = float(ypix)
                cursor_x = max(-0.5, min(cursor_x, float(state.data.shape[-1] - 0.5)))
                cursor_y = max(-0.5, min(cursor_y, float(state.data.shape[-2] - 0.5)))
                self.update_clicked_pix(
                    cursor_x,
                    cursor_y,
                    update_slices=True,
                    fast_blit=False,
                    force_slice_refresh=True,
                )
                effective_xpix = int(round(self._get_shared_xpix()))
                effective_ypix = int(round(self._get_shared_ypix()))
                cursor_visible = preferred.get("cursor_visible")
                if cursor_visible is False:
                    for plane_name in ("xy", "xz", "zy"):
                        vline = self._get_plane_vline(plane_name)
                        hline = self._get_plane_hline(plane_name)
                        cpoint = self._get_plane_cpoint(plane_name)
                        if vline is not None:
                            vline.set_visible(False)
                        if hline is not None:
                            hline.set_visible(False)
                        if cpoint is not None:
                            cpoint.set_visible(False)
                    try:
                        self.redraw_main_overlay_and_blit()
                    except Exception:
                        pass
                sync_controls = getattr(self, "_sync_channel_controls", None)
                if callable(sync_controls):
                    for viewer in list(getattr(self, "subwindows", [])):
                        if viewer is None:
                            continue
                        plane_name = str(getattr(viewer, "plane", "")).lower()
                        if plane_name == "xz":
                            sync_controls(viewer, effective_ypix)
                        elif plane_name == "zy":
                            sync_controls(viewer, effective_xpix)

            if hasattr(self, "region_manager") and self.region_manager is not None:
                self.region_manager.delete_all_regions()
                region_payload = self._build_region_payload_from_state()
                if region_payload.get("regions"):
                    try:
                        self.region_manager.import_regions_from_dict(region_payload, clear_existing=True)
                    except Exception:
                        pass

            if hasattr(self, "marker_manager") and self.marker_manager is not None:
                marker_planes_to_redraw = set()
                marker_layers = list((getattr(self.marker_manager, "_layers", None) or {}).keys())
                for plane in marker_layers:
                    plane_name = str(plane or "").lower()
                    if plane_name:
                        marker_planes_to_redraw.add(plane_name)
                    try:
                        self.marker_manager.clear_plane(plane)
                    except Exception:
                        continue
                marker_payload = self._build_marker_payload_from_state()
                marker_entries = list(marker_payload.get("markers") or [])
                for marker_entry in marker_entries:
                    if not isinstance(marker_entry, dict):
                        continue
                    plane_name = str(marker_entry.get("plane") or "").lower()
                    if plane_name:
                        marker_planes_to_redraw.add(plane_name)
                if marker_entries:
                    try:
                        imported_plane = self.marker_manager.import_from_dict(marker_payload)
                        imported_plane_name = str(imported_plane or "").lower()
                        if imported_plane_name:
                            marker_planes_to_redraw.add(imported_plane_name)
                    except Exception:
                        pass
                if marker_planes_to_redraw:
                    base_planes = set()
                    base_resolver = getattr(self, "marker_plane_base", None)
                    for plane_name in marker_planes_to_redraw:
                        if callable(base_resolver):
                            try:
                                base = str(base_resolver(plane_name) or "").lower()
                            except Exception:
                                base = ""
                        else:
                            base = str(plane_name or "").lower()
                        if base in ("xy", "xz", "zy"):
                            base_planes.add(base)
                    refresh_bg = getattr(self, "_refresh_overlay_background", None)
                    if callable(refresh_bg):
                        for base in base_planes:
                            try:
                                refresh_bg(base)
                            except Exception:
                                continue
                    try:
                        self.marker_manager.redraw_planes(marker_planes_to_redraw)
                    except Exception:
                        for plane_name in marker_planes_to_redraw:
                            try:
                                self.marker_manager.redraw_plane(plane_name)
                            except Exception:
                                continue

            self._last_regions_fingerprint = json.dumps(
                self._region_specs_snapshot(), sort_keys=True, separators=(",", ":")
            )
            self._last_markers_fingerprint = json.dumps(
                self._marker_specs_snapshot(), sort_keys=True, separators=(",", ":")
            )
        finally:
            self._suspend_action_recording = False
            self._refresh_undo_redo_actions()

    def show_main_window(self):
        self.show()  # Main window should always be visible

    def hide_main_window(self):
        self.hide()  # Hide the main window

    def show_control_panel(self):
        panel = self.ensure_control_panel()
        if not panel.isVisible():
            panel.show()
        self._set_panel_toggle_checked("control_panel_action", True)
    
    def show_range_panel(self):
        panel = self.ensure_range_panel()
        if not panel.isVisible():
            panel.show()
        self._set_panel_toggle_checked("range_panel_action", True)

    def hide_control_panel(self):
        panel = getattr(self, "control_panel", None)
        if panel is not None:
            panel.hide()
        self._set_panel_toggle_checked("control_panel_action", False)
    
    def hide_range_panel(self):
        panel = getattr(self, "range_panel", None)
        if panel is not None:
            panel.hide()
        self._set_panel_toggle_checked("range_panel_action", False)

    def on_control_panel_closed(self):
        # When control panel is closed, uncheck the menu action
        self.menu_bar.control_panel_action.setChecked(False)


    def clear_all_regions_globally(self):
        """
        Clears all regions from the main viewer and any open integration result windows.
        """
        if hasattr(self, 'region_manager'):
            self.region_manager.delete_all_regions()

        if hasattr(self, 'integ_result_windows'):
            live_windows = []
            for window_ref in self.integ_result_windows:
                window = window_ref()
                if window is not None:
                    if hasattr(window, 'region_manager'):
                        window.region_manager.delete_all_regions()
                    live_windows.append(window_ref)

            self.integ_result_windows = live_windows

    def closeEvent(self, event):
        self._is_app_closing = True
        app = QApplication.instance()
        if app is not None:
            try:
                app.setProperty("takefits_app_closing", True)
            except Exception:
                pass
            try:
                app.removeEventFilter(self)
            except Exception:
                pass
        print("\n\nProgram exited.")
        super().closeEvent(event)
        if app is not None:
            app.quit()

    # ------------------------------------------------------------------
    # Region save/load
    def save_regions_dialog(self):
        if not hasattr(self, "region_manager"):
            return
        # Collect candidates (manager, plane, label) that have regions
        candidates = []
        # Main viewer
        main_regions = getattr(self.region_manager, "regions", [])
        if main_regions:
            planes = {getattr(r, "plane", None) or getattr(self, "plane", "xy") for r in main_regions}
            for plane_name in planes:
                candidates.append((self.region_manager, plane_name.lower(), "Main"))
        # Integ windows
        if hasattr(self, "integ_result_windows"):
            for window_ref in list(self.integ_result_windows):
                window = window_ref()
                if window is None:
                    continue
                mgr = getattr(window, "region_manager", None)
                regs = getattr(mgr, "regions", []) if mgr is not None else []
                if not regs:
                    continue
                planes = {getattr(r, "plane", None) or getattr(window, "plane", "xy") for r in regs}
                label_attr = getattr(window, "original_window_title", None)
                label = None
                if label_attr:
                    label = label_attr
                else:
                    label_attr = getattr(window, "windowTitle", None)
                    if callable(label_attr):
                        try:
                            label = label_attr()
                        except Exception:
                            label = None
                    else:
                        label = label_attr
                if isinstance(label, str) and label.startswith("[REGION MODE"):
                    closing = label.find("]")
                    if closing != -1:
                        label = label[closing + 1 :].strip()
                if not label:
                    label = f"Integ ({getattr(window, 'plane', 'xy').upper()})"
                for plane_name in planes:
                    candidates.append((mgr, plane_name.lower(), label))

        if not candidates:
            QMessageBox.information(self, "Regions", "No regions to save.")
            return

        # Deduplicate manager/plane combos
        unique = []
        seen = set()
        for mgr, plane_name, label in candidates:
            key = (id(mgr), plane_name)
            if key in seen:
                continue
            seen.add(key)
            unique.append((mgr, plane_name, label))

        if len(unique) == 1:
            target_manager, target_plane, _label = unique[0]
        else:
            items = []
            for _, pl, lbl in unique:
                short_lbl = str(lbl)
                if len(short_lbl) > 24:
                    short_lbl = short_lbl[:21] + "..."
                items.append(f"{short_lbl} - {pl.upper()}")
            choice, ok = QInputDialog.getItem(self, "Save Regions", "Select panel/plane to save:", items, 0, False)
            if not ok or not choice:
                return
            idx = items.index(choice)
            target_manager, target_plane, _label = unique[idx]
        default_dir = os.path.dirname(getattr(self, "filename_path", "")) or os.getcwd()
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Regions",
            os.path.join(default_dir, f"regions_{target_plane}.region"),
            "Region Files (*.region.json *.region *.json);;All Files (*)",
        )
        if not path:
            return
        if not path.lower().endswith((".region.json", ".region", ".json")):
            path += ".region"
        try:
            payload = target_manager.export_regions_to_dict(target_plane)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            QMessageBox.information(self, "Regions", f"Saved {len(payload.get('regions', []))} region(s).")
        except Exception as exc:
            QMessageBox.warning(self, "Regions", f"Failed to save regions: {exc}")

    def load_regions_dialog(self):
        if not hasattr(self, "region_manager"):
            return
        default_dir = os.path.dirname(getattr(self, "filename_path", "")) or os.getcwd()
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Regions",
            default_dir,
            "Region Files (*.region.json *.region *.json);;All Files (*)",
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            # Load into main viewer
            self.region_manager.import_regions_from_dict(payload, clear_existing=True)
            # If region mode was off, enable it with circle mode for immediate interaction
            if not getattr(self, "region_mode_enabled", False):
                try:
                    try:
                        self.menu_bar.circle_action.setChecked(True)
                    except Exception:
                        pass
                    self.set_region_shape("circle")
                except Exception:
                    pass
            # Also load into any open integration result windows on the XY plane
            if hasattr(self, "integ_result_windows"):
                for window_ref in list(self.integ_result_windows):
                    window = window_ref()
                    if window is None:
                        continue
                    try:
                        if getattr(window, "plane", "").lower() != "xy":
                            continue
                        if hasattr(window, "region_manager"):
                            window.region_manager.import_regions_from_dict(payload, clear_existing=True)
                    except Exception:
                        continue
            QMessageBox.information(self, "Regions", "Regions loaded.")
        except Exception as exc:
            QMessageBox.warning(self, "Regions", f"Failed to load regions: {exc}")

    def open_regrid_panel(self):
        if self._regrid_panel is None:
            self._regrid_panel = RegridPanel(fits_viewer=self)
            self._regrid_panel.regrid_requested.connect(self._start_regrid_job)
            self._regrid_panel.closed.connect(self._clear_regrid_panel)
        self._regrid_panel.show()
        self._regrid_panel.raise_()
        self._regrid_panel.activateWindow()

    def _clear_regrid_panel(self):
        self._regrid_panel = None

    def _start_regrid_job(self, params):
        if self._regrid_thread is not None and self._regrid_thread.isRunning():
            QMessageBox.information(
                self,
                "Regrid In Progress",
                "A regrid operation is already running. Please wait for it to finish.",
            )
            return

        self._last_regrid_params = dict(params)
        self._regrid_thread = QThread(self)
        header_copy = self.header.copy() if hasattr(self, "header") and self.header is not None else None
        self._regrid_worker = Regridder(
            self.data,
            self.wcs,
            header_copy,
            filename=self.filename,
            state=getattr(self, "app_state", None),
        )
        self._regrid_worker.moveToThread(self._regrid_thread)

        self._regrid_worker.progress.connect(self._handle_regrid_progress)
        self._regrid_worker.finished.connect(self._handle_regrid_finished)
        self._regrid_worker.error.connect(self._handle_regrid_error)

        self._regrid_thread.started.connect(
            partial(self._regrid_worker.perform_regrid, params)
        )
        self._regrid_worker.finished.connect(self._regrid_thread.quit)
        self._regrid_worker.error.connect(self._regrid_thread.quit)
        self._regrid_thread.finished.connect(self._cleanup_regrid_thread)
        self._regrid_thread.start()

        if self._regrid_panel:
            self._regrid_panel.on_regrid_started()

    def _handle_regrid_progress(self, value):
        if self._regrid_panel:
            self._regrid_panel.update_progress(value)

    def _handle_regrid_finished(self, data, header):
        if self._regrid_panel:
            self._regrid_panel.on_regrid_finished(True)
        save_header = header.copy() if hasattr(header, "copy") else header
        if save_header is not None and hasattr(save_header, "add_history"):
            history_entries = build_processing_history_lines_with_action(
                self,
                "compute_regrid",
                {"params": dict(getattr(self, "_last_regrid_params", {}) or {})},
            )
            for entry in history_entries:
                save_header.add_history(entry)

        saver = SaveFITS(data, save_header, self.filename, original_header=self.header)
        suffix = self._resolve_regrid_suffix()
        saver.save(suffix=suffix)

    def _handle_regrid_error(self, message):
        self.clear_recorded_action("panel:regrid")
        if self._regrid_panel:
            self._regrid_panel.on_regrid_finished(False)
            QMessageBox.critical(self._regrid_panel, "Regrid Failed", message)
        else:
            QMessageBox.critical(self, "Regrid Failed", message)

    def _cleanup_regrid_thread(self):
        if self._regrid_worker:
            self._regrid_worker.deleteLater()
            self._regrid_worker = None
        thread = self._regrid_thread
        self._regrid_thread = None
        if thread:
            thread.deleteLater()

    def _resolve_regrid_suffix(self) -> str:
        params = getattr(self, "_last_regrid_params", {}) or {}
        mode = params.get("mode", "").lower()
        if mode == "reproject_system":
            target = (params.get("target_system") or "").strip().lower()
            mapping = {
                "galactic": "lb",
                "icrs": "icrs",
                "fk5": "fk5",
                 "fk4": "fk4",
                "ecliptic": "ecliptic",
            }
            if target in mapping:
                return mapping[target]
        elif mode == "template_fits":
            return "reproject"
        return "reg"
