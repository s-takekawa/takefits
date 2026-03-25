from astropy.wcs import WCS
import numpy as np
import os
import time
import math
from collections import OrderedDict
from typing import Dict, List, Optional
from PySide6.QtCore import Qt, QTimer, Signal as pyqtSignal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QGridLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from takefits.ui.navigation_toolbar import MyNavigationToolbar
from takefits.core.coordinate import CoordinateConverter
from takefits.core.config import ConfigManager
from takefits.core.colorbar_layout import compute_colorbar_geometry, orientation_for_placement
from takefits.core.click_label_layout import compute_click_label_geometry
from takefits.core.viewer_state import ViewerState
from takefits.core.contour_manager import ContourManager, ContourItem
from takefits.core.plotting.display_map import DisplayMap
from takefits.core.coordinate import Format_pix_to_wcs
from matplotlib.figure import Figure
from matplotlib import colormaps
from takefits.logic.add_hpbw import AddHPBW
from takefits.tools.color_scale import ColorSettingsPanel
from astropy import units as u
from takefits.core.region_manager import RegionManager
from takefits.core.marker_manager import MarkerManager
from takefits.core.wcs_frames import (
    axis_is_latitude,
    axis_is_longitude,
    axis_type_for_index,
    display_axis_type,
    display_frame_label,
    native_celestial_frame,
    normalize_display_frame,
)
from takefits.ui.viewer_coord_mixin import ViewerCoordinatorMixin
from takefits.ui.viewer_blit_mixin import ViewerBlitMixin
from takefits.logic.data_tools import (
    DEFAULT_LARGE_DATA_DISPLAY_MAX_DIM,
    build_large_data_profile,
    downsample_2d_for_display,
)

class FITSViewer(QMainWindow, ViewerCoordinatorMixin, ViewerBlitMixin):
    position_updated = pyqtSignal(float, float, float)
    coordinate_updated = pyqtSignal(float, float, float)
    data = None
    header = None
    wcs = None
    instance_initialized = False
    wcs_check_initialized = False
    velocity_unit_converted = False
    main_window = None  # Class-level reference to the main window (set by MainWindow)

    @classmethod
    def get_viewer_by_plane(cls, plane):
        """Return the FITSViewer instance for the given plane ('xy', 'xz', or 'zy')."""
        main = cls.main_window
        if main is None:
            return None
        if plane == 'xy':
            return main
        for subwindow in getattr(main, 'subwindows', []):
            if getattr(subwindow, 'plane', None) == plane:
                return subwindow
        return None

    def __init__(self, data, header, wcs=None, filename="", spectral_metadata=None):
        super().__init__()
        self.data = data
        self.header = header
        self.wcs = wcs
        self.spectral_metadata = spectral_metadata if isinstance(spectral_metadata, dict) else {}
        self.spectral_metadata.setdefault("is_cartesian_interpretation", False)
        self.region_manager = RegionManager(self)
        try:
            self.region_manager.selected_region_changed.connect(self._on_region_selection_changed)
        except Exception:
            pass
        marker_manager = getattr(self, "marker_manager", None)
        self.marker_manager = marker_manager if marker_manager is not None else MarkerManager(self)
        self.marker_mode_enabled = False
        self.marker_panel = None
        self.cutout_dialog = None
        self.config_manager = None
        self._pending_region_restore = []
        self._contour_layer_id = None
        self._contour_title_connected = False
        try:
            from takefits.ui.subwindow import SubWindow_control
            self.SubWindow = SubWindow_control()
        except ImportError:
            self.SubWindow = None
        if not FITSViewer.instance_initialized:
            if data is not None and header is not None:
                FITSViewer.data = data
                FITSViewer.header = header
                FITSViewer.instance_initialized = True
                try: 
                    FITSViewer.wcs = WCS(header) if wcs is None else wcs
                except Exception as e:
                    print(f"WCS initialization error: {e}")
                    print("\033[1;33mWarning: Interpret as a simple Cartesian coordinate system.\033[0m")
                    self.spectral_metadata["is_cartesian_interpretation"] = True
                    for axis in ['CTYPE1', 'CTYPE2']:
                        axis_value = header[axis].split('-')[0]
                        coord_keys = ['VELO', 'VRAD', 'VOPT', 'GLON', 'GLAT', 'RA', 'DEC']
                        if axis_value in coord_keys:
                            replacement = axis_value
                        else:
                            replacement = 'X' if axis == 'CTYPE1' else 'Y'
                        print(f'{axis}: {header[axis]} ==> {replacement}')
                        header[axis] = replacement
                    FITSViewer.wcs = WCS(header)
                self.data = data
                self.header = header
                self.wcs = FITSViewer.wcs if wcs is None else wcs
        else:
            self.data = FITSViewer.data
            self.header = FITSViewer.header
            self.wcs = FITSViewer.wcs if FITSViewer.wcs is not None else WCS(self.header)
        
        #velocity unit conversion [Note: Subject to change in the future.]    
        spectral_meta = self.spectral_metadata
        axis_index = spectral_meta.get('axis_index')
        if axis_index is None and self.wcs.wcs.naxis >= 3:
            axis_index = 3
        if axis_index and spectral_meta.get('axis_index') is None:
            spectral_meta['axis_index'] = axis_index
        if axis_index and axis_index <= self.wcs.wcs.naxis:
            wcs_axis_idx = axis_index - 1
            try:
                unit_wcs = self.wcs.wcs.cunit[wcs_axis_idx].to_string().replace(' ', '').lower()
            except Exception:
                unit_wcs = ''
            unit_header = str(self.header.get(f'CUNIT{axis_index}', '')).replace(' ', '').lower()
            already_adjusted = spectral_meta.get('velocity_unit_adjusted', False)

            if not already_adjusted:
                if unit_header == 'km/s' and unit_wcs != 'km/s':
                    try:
                        wcs_unit = u.Unit(unit_wcs) if unit_wcs else None
                    except Exception:
                        wcs_unit = None
                    spectral_meta['current_axis_unit'] = 'km/s'
                    spectral_meta['current_axis_type'] = 'velocity'
                    spectral_meta['current_axis_ctype'] = self.header.get(f'CTYPE{axis_index}', spectral_meta.get('current_axis_ctype'))
                elif (unit_header in ('m/s', '') and abs(self.wcs.wcs.cdelt[wcs_axis_idx]) > 100.0):
                    self.wcs.wcs.cdelt[wcs_axis_idx] = (self.wcs.wcs.cdelt[wcs_axis_idx] * u.m / u.s).to(u.km / u.s).value
                    self.wcs.wcs.crval[wcs_axis_idx] = (self.wcs.wcs.crval[wcs_axis_idx] * u.m / u.s).to(u.km / u.s).value
                    self.header[f'CUNIT{axis_index}'] = 'km/s'
                    self.header[f'CDELT{axis_index}'] = self.wcs.wcs.cdelt[wcs_axis_idx]
                    self.header[f'CRVAL{axis_index}']  = self.wcs.wcs.crval[wcs_axis_idx]
                    spectral_meta['velocity_unit_adjusted'] = True
                    spectral_meta['velocity_unit_original'] = 'm/s'
                    spectral_meta['velocity_unit_target'] = 'km/s'
                    spectral_meta['current_axis_unit'] = 'km/s'
                    spectral_meta['current_axis_type'] = 'velocity'
                    spectral_meta['current_axis_ctype'] = self.header.get(f'CTYPE{axis_index}', spectral_meta.get('current_axis_ctype'))

                    if not FITSViewer.wcs_check_initialized:
                        FITSViewer.velocity_unit_converted = True
                        FITSViewer.wcs_check_initialized = True
            else:
                if spectral_meta.get('current_axis_unit') is None:
                    current_unit = self.header.get(f'CUNIT{axis_index}', '')
                    spectral_meta['current_axis_unit'] = current_unit.strip() if isinstance(current_unit, str) else None
                if spectral_meta.get('current_axis_type') in (None, 'unknown'):
                    spectral_meta['current_axis_type'] = 'velocity'
                spectral_meta['current_axis_ctype'] = self.header.get(f'CTYPE{axis_index}', spectral_meta.get('current_axis_ctype'))
        
        elif self.wcs.wcs.naxis == 2:
            for i in range(2):
                wcs_unit = self.wcs.wcs.cunit[i].to_string().replace(' ', '').lower()
                if wcs_unit == 'm/s':
                    self.wcs.wcs.cdelt[i] = (self.wcs.wcs.cdelt[i] * u.m / u.s).to(u.km / u.s).value
                    self.wcs.wcs.crval[i] = (self.wcs.wcs.crval[i] * u.m / u.s).to(u.km / u.s).value
        
        

        self.original_data = np.array(data, copy=False)
        self.integ_result_windows = []
        self.channel_map_windows = []
        self._blank_planes = {}
        self._large_data_slice_cache = OrderedDict()
        self._large_data_slice_cache_limit = 12
        self._large_data_prefetch_request = None
        self.world_x = self.world_y = self.world_z = None
    
        self.colorbar = None
        self.cax = None
        if self.data.ndim > 2:
            if self.data.ndim == 3:
                self.cube = self.data
            if self.data.ndim == 4:
                self.cube = self.data[0]
            self._blank_planes = {}
        
        self.filename_path = filename
        self.filename = os.path.basename(self.filename_path)
        windowtitle = f"Mainwindow: {self.filename}"
        
        self.config_manager = ConfigManager()
        config = self.config_manager.config
        perf_enabled = config.get('perf_enabled', False)
        env_perf = os.environ.get("TAKEFITS_PERF", "")
        if env_perf:
            try:
                perf_enabled = bool(int(env_perf))
            except Exception:
                perf_enabled = env_perf.strip().lower() in ("1", "true", "yes", "on")
        self._perf_enabled = bool(perf_enabled)
        perf_threshold = config.get('perf_threshold_ms', 4.0)
        env_threshold = os.environ.get("TAKEFITS_PERF_MS", "")
        if env_threshold:
            try:
                perf_threshold = float(env_threshold)
            except Exception:
                pass
        self._perf_threshold_ms = float(perf_threshold)
        drag_interval_ms = config.get('drag_slice_interval_ms', 10.0)
        env_drag_interval = os.environ.get("TAKEFITS_DRAG_SLICE_MS", "")
        if env_drag_interval:
            try:
                drag_interval_ms = float(env_drag_interval)
            except Exception:
                pass
        self._drag_slice_interval_sec = max(0.0, float(drag_interval_ms) / 1000.0)
        self._large_data_prefetch_timer = QTimer(self)
        self._large_data_prefetch_timer.setSingleShot(True)
        self._large_data_prefetch_timer.timeout.connect(self._run_large_data_prefetch)
        
        self.setWindowTitle(windowtitle)
        self.figure_pos_x = config.get('figure_pos_x', 100)
        self.figure_pos_y = config.get('figure_pos_y', 100)
        self.figure_width = config.get('figure_width', 640)
        self.figure_height = config.get('figure_height', 640)
        self.setGeometry(self.figure_pos_x, self.figure_pos_y, self.figure_width, self.figure_height)


        self.click_label_color = config.get('click_label_color', 'grey')
        self.click_linewidth = config.get('click_linewidth', 0.25)
        self.click_linecolor = config.get('click_linecolor', 'cyan')
        self.click_linestyle = str(config.get('click_linestyle', '-'))
        self.click_alpha = float(config.get('click_alpha', 1.0))
        self.click_show_crosshair = bool(config.get('click_show_crosshair', True))
        self.click_crosshair_mode = str(config.get('click_crosshair_mode', 'both'))
        self.click_show_center_marker = bool(config.get('click_show_center_marker', False))
        self.decimal = config.get('decimal', True)
        self.auto_precision_digits = bool(config.get('auto_precision_digits', True))
        self.number_decimals = config.get('number_decimals', 6)
        self.coord_wrap = config.get('coord_wrap', 180)
        self.scrollspeed = config.get('scrollspeed', 0.1)
        self.invert_wheel_direction = bool(config.get('invert_wheel_direction', False))
        
        self.poslabel_x = config.get('poslabel_x', 0.99)
        self.poslabel_y = config.get('poslabel_y', 0.99)
        self.poslabel_w = config.get('poslabel_w', 250)
        self.poslabel_h = config.get('poslabel_h', 30)
        
        self.beam_facecolor = config.get('beam_facecolor', 'white')
        self.beam_edgecolor = config.get('beam_edgecolor', 'None')
        self.beam_linewidth = config.get('beam_linewidth', 0)
        self.beam_pos_x = config.get('beam_pos_x', 0.1)
        self.beam_pos_y = config.get('beam_pos_y', 0.1)
        
        
        self.pos_chlabel_x = config.get('pos_chlabel_x', 0.98)
        self.pos_chlabel_y = config.get('pos_chlabel_y', 0.02)
        #self.pos_chlabel_w = config.get('pos_chlabel_w', 250)
        #self.pos_chlabel_h = config.get('pos_chlabel_h', 20)
        self.ch_label_color = config.get('ch_label_color', 'grey')
        self.ch_label_font = config.get('ch_label_font', 'Arial')
        self.ch_label_size = config.get('ch_label_size', 12)
        
        self.range_file = config.get('range_file', 'takefits.range')


        
        self.fig = Figure()
        self.k = 0
        self.scroll_accumulation = 0
        self.xlabel = self.ylabel = self.zlabel = None

        self.converter = CoordinateConverter(self.wcs, config)
        self.original_xval = None
        self.original_yval = None
        self.original_zval = None
        self._full_world_limits = {}
        
        if 'BUNIT' in self.header: self.bunit = self.header['BUNIT']
        else: self.bunit = ''
        
        
    def open_cutout_dialog(self, region=None, use_view_bounds=False):
        from takefits.tools.cutout import CutoutSettingsDialog

        dialog = self.cutout_dialog
        if dialog is None:
            collapsed_axes = [idx for idx, role in enumerate(self.get_axis_roles()) if role == 'collapsed']
            dialog = CutoutSettingsDialog(self, region if region is not None else None, self, collapsed_axes=collapsed_axes)
            dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
            dialog.destroyed.connect(lambda *_: setattr(self, 'cutout_dialog', None))
            self.cutout_dialog = dialog

        if region is not None:
            dialog.reset_region(region)
        elif use_view_bounds or getattr(dialog, 'region', None) is None:
            dialog.reset_to_view()

        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def get_axis_roles(self):
        if self.wcs is None:
            return []
        naxis = self.wcs.naxis
        roles = ['depth'] * naxis
        if naxis >= 1:
            roles[0] = 'display_x'
        if naxis >= 2:
            roles[1] = 'display_y'
        for idx in range(2, naxis):
            if roles[idx] not in ('display_x', 'display_y'):
                roles[idx] = 'depth'
        return roles

    def _on_region_selection_changed(self, region):
        if self.cutout_dialog is None:
            return
        if region is not None:
            self.cutout_dialog.reset_region(region)
        else:
            self.cutout_dialog.reset_to_view()

    def current_z_pixel_bounds(self):
        if getattr(self, 'data', None) is None or self.data.ndim < 3:
            return (0, 1)

        try:
            subwindows = getattr(self, 'subwindows', [])
            if subwindows:
                for sub in subwindows:
                    if getattr(sub, 'plane', '') == 'xz' and hasattr(sub, 'ax'):
                        zlim = sub.ax.get_ylim()
                        z_start, z_stop = sorted((float(zlim[0]), float(zlim[1])))
                        start = int(math.floor(z_start))
                        stop = int(math.ceil(z_stop))
                        if stop <= start:
                            stop = start + 1
                        start = max(start, 0)
                        stop = min(stop, self.data.shape[0])
                        if stop <= start:
                            return (0, self.data.shape[0])
                        return (start, stop)
        except Exception:
            pass

        return (0, self.data.shape[0])

    def current_channel_index(self):
        if self.plane == 'xy':
            return self._get_shared_zpix()
        elif self.plane == 'xz':
            return self._get_shared_ypix()
        elif self.plane == 'zy':
            return self._get_shared_xpix()
        else:
            return 0

    def _blank_plane(self, plane):
        """
        Lazily allocate blank (NaN-filled) arrays for out-of-range channel requests.
        """
        if plane in self._blank_planes:
            return self._blank_planes[plane]

        if not hasattr(self, 'cube'):
            return np.array([])

        if plane == 'xy':
            template = self.cube[0]
        elif plane == 'xz':
            template = self.cube[:, 0, :]
        elif plane == 'zy':
            template = self.cube[:, :, 0].T
        else:
            template = self.cube[0]

        dtype = template.dtype if np.issubdtype(template.dtype, np.floating) else np.float32
        blank = np.full(template.shape, np.nan, dtype=dtype)
        self._blank_planes[plane] = blank
        return blank

    def formatter(self, x, y):
        xstr, ystr = self.format_pix.convert(self.plane, x, y)
        xkey, ykey = self._plane_coord_keys(self.plane)
        xstr = self._format_value_with_axis_unit(self.plane, xkey, xstr)
        ystr = self._format_value_with_axis_unit(self.plane, ykey, ystr)
        frame = " ".join(str(self._cursor_coordinate_frame_label(self.plane)).split())
        if self.plane == 'xy':
            line1 = f"x={xstr}, y={ystr}"
        elif self.plane == 'xz':
            line1 = f"x={xstr}, z={ystr}"
        else:
            line1 = f"z={xstr}, y={ystr}"
        intensity = self._toolbar_intensity_text(x, y)
        if intensity:
            line2 = f"{' '.join(str(intensity).split())} {frame}"
        else:
            line2 = f"{frame}"
        return f"{line1}\n{line2}"

    def _format_intensity_with_unit(self, value) -> str:
        try:
            text = self._format_significant_digits(float(value), 4)
        except Exception:
            text = " ".join(str(value).split()).strip()
        unit = str(getattr(self, "bunit", "") or "").replace("\n", " ").strip()
        if unit:
            return f"{text} {unit}"
        return text

    def _toolbar_intensity_text(self, x, y) -> Optional[str]:
        try:
            xi = int(round(float(x)))
            yi = int(round(float(y)))
        except Exception:
            return None

        value = None
        try:
            if self.data is None:
                return None
            if self.data.ndim == 2:
                if yi < 0 or xi < 0 or yi >= self.data.shape[0] or xi >= self.data.shape[1]:
                    return None
                value = self.data[yi, xi]
            else:
                cube = getattr(self, "cube", None)
                if cube is None:
                    return None
                if self.plane == "xy":
                    k = int(self._get_shared_zpix())
                    j = yi
                    i = xi
                elif self.plane == "xz":
                    k = yi
                    j = int(self._get_shared_ypix())
                    i = xi
                else:
                    k = xi
                    j = yi
                    i = int(self._get_shared_xpix())
                if (
                    k < 0 or j < 0 or i < 0
                    or k >= cube.shape[0]
                    or j >= cube.shape[1]
                    or i >= cube.shape[2]
                ):
                    return None
                value = cube[k, j, i]
        except Exception:
            return None

        try:
            text = self._format_intensity_with_unit(value)
        except Exception:
            return None
        return f"[{text}]"

    def _perf_start(self, label: str):
        if not getattr(self, '_perf_enabled', False):
            return None
        return (label, time.perf_counter())

    def _perf_end(self, token):
        if not token:
            return
        label, start = token
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if elapsed_ms >= getattr(self, '_perf_threshold_ms', 0.0):
            print(f"[PERF] {label}: {elapsed_ms:.2f} ms")

    def _contours_active(self) -> bool:
        if not self._contour_layer_id:
            return False
        try:
            manager = ContourManager.instance()
            layer = manager._layers.get(self._contour_layer_id)
            return bool(layer and layer.is_active())
        except Exception:
            return False

    def _update_slice_image(self, viewer, *, fast_blit: bool) -> bool:
        if viewer is None:
            return False
        if fast_blit:
            if viewer._contours_active():
                # Avoid duplicate heavy redraw: refresh_display is usually
                # triggered by contour_updated signal when refresh succeeds.
                if not viewer._refresh_contours():
                    viewer.refresh_display_after_contour_update(viewer._contour_layer_id)
                return True
            viewer._fast_blit_image_and_overlay(
                include_ticks=False,
                include_colorbar=False,
                quick_overlay_background=True,
            )
            return True
        if viewer._contours_active():
            if not viewer._refresh_contours():
                viewer.refresh_display_after_contour_update(viewer._contour_layer_id)
                return True
            return False
        return False

    def on_motion_redirect(self, event):
        if event.inaxes == self.displaymap.overlay_ax:
            event.inaxes = self.ax

    @staticmethod
    def _is_colorbar_axes(axes) -> bool:
        try:
            return axes is not None and axes.get_gid() == "colorbar"
        except Exception:
            return False

    @staticmethod
    def _clamp_float(value: float, low: float, high: float) -> float:
        return max(low, min(high, float(value)))

    def _event_to_figure_coords(self, event):
        fig = getattr(self, "fig", None)
        if fig is None:
            return None, None
        ex = getattr(event, "x", None)
        ey = getattr(event, "y", None)
        if ex is None or ey is None:
            return None, None
        fw = float(getattr(fig.bbox, "width", 0.0) or 0.0)
        fh = float(getattr(fig.bbox, "height", 0.0) or 0.0)
        if fw <= 0.0 or fh <= 0.0:
            return None, None
        return float(ex) / fw, float(ey) / fh

    def _set_colorbar_geometry(self, pos_x: float, pos_y: float, width: float, height: float):
        min_size = 0.01
        width = self._clamp_float(width, min_size, 1.0)
        height = self._clamp_float(height, min_size, 1.0)
        pos_x = self._clamp_float(pos_x, 0.0, max(0.0, 1.0 - width))
        pos_y = self._clamp_float(pos_y, 0.0, max(0.0, 1.0 - height))

        for plane in ("xy", "xz", "zy"):
            state = self.get_viewer_state(plane)
            if state is None:
                continue
            self._apply_colorbar_geometry_to_state(state, pos_x, pos_y, width, height)

        self._update_colorbar_geometry_config(pos_x, pos_y, width, height)

    def _apply_colorbar_geometry_to_state(
        self,
        state,
        pos_x: float,
        pos_y: float,
        width: float,
        height: float,
        *,
        redraw: bool = True,
    ):
        if state is None or state.cax is None:
            return
        try:
            state.cax.set_position([pos_x, pos_y, width, height])
            state.cax.set_gid("colorbar")
            self._set_colorbar_zorder_for_state(state)
        except Exception:
            return
        if redraw and state.canvas is not None:
            try:
                self._request_canvas_redraw(state.canvas)
            except Exception:
                pass

    def _request_canvas_redraw(self, canvas=None, *, immediate: bool = False) -> bool:
        target = canvas if canvas is not None else getattr(self, "canvas", None)
        if target is None:
            return False

        main = self._get_main_viewer()
        dispatcher = getattr(main, "_request_canvas_redraw", None)
        if callable(dispatcher) and main is not self:
            try:
                return bool(dispatcher(target, immediate=immediate))
            except Exception:
                pass

        draw_name = "draw" if immediate else "draw_idle"
        draw = getattr(target, draw_name, None)
        if callable(draw):
            try:
                draw()
                return True
            except Exception:
                pass
        if not immediate:
            draw = getattr(target, "draw", None)
            if callable(draw):
                try:
                    draw()
                    return True
                except Exception:
                    pass
        return False

    def _set_colorbar_zorder_for_state(self, state, zorder: float = 300.0):
        if state is None:
            return
        cax = getattr(state, "cax", None)
        if cax is None:
            return
        try:
            cax.set_zorder(float(zorder))
        except Exception:
            pass
        colorbar = getattr(state, "colorbar", None)
        cbar_ax = getattr(colorbar, "ax", None) if colorbar is not None else None
        if cbar_ax is not None:
            try:
                cbar_ax.set_zorder(float(zorder))
            except Exception:
                pass

    @staticmethod
    def _bbox_overlaps(a, b) -> bool:
        if a is None or b is None:
            return False
        try:
            return bool(a.overlaps(b))
        except Exception:
            pass
        try:
            ax0, ay0, ax1, ay1 = float(a.x0), float(a.y0), float(a.x1), float(a.y1)
            bx0, by0, bx1, by1 = float(b.x0), float(b.y0), float(b.x1), float(b.y1)
        except Exception:
            return False
        return (ax0 < bx1) and (ax1 > bx0) and (ay0 < by1) and (ay1 > by0)

    def _colorbar_needs_foreground_blit(self, state) -> bool:
        if state is None:
            return False
        cax = getattr(state, "cax", None)
        if cax is None:
            return False
        cbar_bbox = getattr(cax, "bbox", None)
        ax_bbox = getattr(getattr(state, "ax", None), "bbox", None)
        overlay_bbox = getattr(getattr(state, "overlay_ax", None), "bbox", None)
        return self._bbox_overlaps(cbar_bbox, ax_bbox) or self._bbox_overlaps(cbar_bbox, overlay_bbox)

    def _blit_colorbar_foreground_for_state(self, state, force: bool = False) -> bool:
        if state is None:
            return False
        canvas = getattr(state, "canvas", None)
        fig = getattr(state, "fig", None)
        cax = getattr(state, "cax", None)
        if canvas is None or fig is None or cax is None:
            return False

        self._set_colorbar_zorder_for_state(state)
        if not force and not self._colorbar_needs_foreground_blit(state):
            return False

        try:
            fig.draw_artist(cax)
            canvas.blit(cax.bbox)
            return True
        except Exception:
            if force:
                try:
                    canvas.draw_idle()
                except Exception:
                    pass
            return False

    def _update_colorbar_geometry_config(self, pos_x: float, pos_y: float, width: float, height: float):
        main = self._get_main_viewer()
        for plane in ("xy", "xz", "zy"):
            state = self.get_viewer_state(plane)
            if state is None:
                continue
            viewer = getattr(state, "viewer", None)
            config_mgr = getattr(viewer, "config_manager", None) if viewer is not None else None
            if config_mgr is not None and isinstance(getattr(config_mgr, "config", None), dict):
                config_mgr.config["cbar_pos_x"] = pos_x
                config_mgr.config["cbar_pos_y"] = pos_y
                config_mgr.config["cbar_width"] = width
                config_mgr.config["cbar_height"] = height

        main_config_mgr = getattr(main, "config_manager", None) if main is not None else None
        if main_config_mgr is not None and isinstance(getattr(main_config_mgr, "config", None), dict):
            main_config_mgr.config["cbar_pos_x"] = pos_x
            main_config_mgr.config["cbar_pos_y"] = pos_y
            main_config_mgr.config["cbar_width"] = width
            main_config_mgr.config["cbar_height"] = height

        menu_bar = getattr(main, "menu_bar", None) if main is not None else None
        panel = getattr(menu_bar, "config_panel", None) if menu_bar is not None else None
        if panel is not None and panel.isVisible():
            try:
                panel.cbar_x_input.setValue(pos_x)
                panel.cbar_y_input.setValue(pos_y)
                panel.cbar_width_input.setValue(width)
                panel.cbar_height_input.setValue(height)
            except Exception:
                pass

    def _is_colorbar_auto_layout_enabled(self) -> bool:
        main = self._get_main_viewer()
        if bool(getattr(main, "_workspace_colorbar_restore_in_progress", False)):
            return False
        config_mgr = getattr(main, "config_manager", None) if main is not None else None
        config = getattr(config_mgr, "config", None)
        if not isinstance(config, dict):
            return False
        return bool(config.get("colorbar_auto_layout", True))

    def _is_crosshair_enabled(self) -> bool:
        main = self._get_main_viewer()
        config_mgr = getattr(main, "config_manager", None) if main is not None else None
        config = getattr(config_mgr, "config", None)
        if not isinstance(config, dict):
            return True
        return bool(config.get("click_show_crosshair", True))

    def _is_center_marker_enabled(self) -> bool:
        main = self._get_main_viewer()
        config_mgr = getattr(main, "config_manager", None) if main is not None else None
        config = getattr(config_mgr, "config", None)
        if not isinstance(config, dict):
            return False
        return bool(config.get("click_show_center_marker", False))

    def _is_cursor_overlay_enabled(self) -> bool:
        return bool(self._is_crosshair_enabled() or self._is_center_marker_enabled())

    @staticmethod
    def _normalize_crosshair_mode(mode: str) -> str:
        value = str(mode or "").strip().lower()
        if value in {"vertical", "horizontal"}:
            return value
        return "both"

    def _crosshair_component_visibility(self, base_visible: bool):
        if not base_visible:
            return False, False, False
        main = self._get_main_viewer()
        config_mgr = getattr(main, "config_manager", None) if main is not None else None
        config = getattr(config_mgr, "config", None)
        if not isinstance(config, dict):
            crosshair_enabled = True
            mode = "both"
            show_point = False
        else:
            crosshair_enabled = bool(config.get("click_show_crosshair", True))
            mode = self._normalize_crosshair_mode(config.get("click_crosshair_mode", "both"))
            show_point = bool(config.get("click_show_center_marker", False))
        if not crosshair_enabled:
            return False, False, show_point
        if mode == "vertical":
            return True, False, show_point
        if mode == "horizontal":
            return False, True, show_point
        return True, True, show_point

    def _set_crosshair_point_for_plane(self, plane: str, x=None, y=None):
        cpoint = self._get_plane_cpoint(plane)
        if cpoint is None:
            return
        state = self.get_viewer_state(plane)
        if state is not None:
            if x is None:
                x = getattr(state, "cursor_x", None)
            if y is None:
                y = getattr(state, "cursor_y", None)
        if x is None or y is None:
            return
        try:
            cpoint.set_data([float(x)], [float(y)])
        except Exception:
            pass

    def _set_crosshair_visibility_for_plane(self, plane: str, base_visible: bool):
        vline = self._get_plane_vline(plane)
        hline = self._get_plane_hline(plane)
        cpoint = self._get_plane_cpoint(plane)
        show_v, show_h, show_point = self._crosshair_component_visibility(bool(base_visible))
        if vline is not None:
            try:
                vline.set_visible(show_v)
            except Exception:
                pass
        if hline is not None:
            try:
                hline.set_visible(show_h)
            except Exception:
                pass
        if cpoint is not None:
            if show_point:
                self._set_crosshair_point_for_plane(plane)
            try:
                cpoint.set_visible(show_point)
            except Exception:
                pass

    @staticmethod
    def _first_line_value(values):
        if values is None:
            return None
        try:
            if len(values) == 0:
                return None
            return float(values[0])
        except Exception:
            return None

    def _snapshot_crosshair_state(self):
        snapshot = {}
        for plane in ("xy", "xz", "zy"):
            state = self.get_viewer_state(plane)
            if state is None:
                continue
            hline = getattr(state, "hline", None)
            vline = getattr(state, "vline", None)
            cpoint = getattr(state, "cpoint", None)
            if hline is None or vline is None:
                continue
            try:
                h_visible = bool(hline.get_visible())
            except Exception:
                h_visible = False
            try:
                v_visible = bool(vline.get_visible())
            except Exception:
                v_visible = False
            try:
                p_visible = bool(cpoint is not None and cpoint.get_visible())
            except Exception:
                p_visible = False
            base_visible = bool(h_visible or v_visible or p_visible)
            snapshot[plane] = {
                "visible": base_visible,
                "x": self._first_line_value(getattr(vline, "get_xdata", lambda: None)()),
                "y": self._first_line_value(getattr(hline, "get_ydata", lambda: None)()),
            }
        return snapshot

    def _restore_crosshair_state(self, snapshot):
        if not isinstance(snapshot, dict) or not snapshot:
            return
        show_cursor_overlay = self._is_cursor_overlay_enabled()
        for plane, payload in snapshot.items():
            state = self.get_viewer_state(plane)
            if state is None:
                continue
            xval = payload.get("x")
            yval = payload.get("y")
            if xval is not None and yval is not None:
                self._update_plane_cursor(plane, x=xval, y=yval)
                vline = getattr(state, "vline", None)
                hline = getattr(state, "hline", None)
                if vline is not None:
                    try:
                        vline.set_xdata([xval])
                    except Exception:
                        pass
                if hline is not None:
                    try:
                        hline.set_ydata([yval])
                    except Exception:
                        pass
                self._set_crosshair_point_for_plane(plane, x=xval, y=yval)
            self._set_crosshair_visibility_for_plane(
                plane,
                bool(payload.get("visible", False)) and show_cursor_overlay,
            )

        main = self._get_main_viewer()
        if main is None:
            main = self
        try:
            main.redraw_main_overlay_and_blit(lightweight=True)
        except Exception:
            pass
        for plane in ("xz", "zy"):
            try:
                main.redraw_overlay_for_plane(plane, lightweight=True)
            except Exception:
                continue

    def _set_colorbar_orientation_config(self, orientation: str):
        if str(orientation).lower() not in ("horizontal", "vertical"):
            return
        orientation = str(orientation).lower()

        for plane in ("xy", "xz", "zy"):
            state = self.get_viewer_state(plane)
            if state is None:
                continue
            viewer = getattr(state, "viewer", None)
            config_mgr = getattr(viewer, "config_manager", None) if viewer is not None else None
            if config_mgr is not None and isinstance(getattr(config_mgr, "config", None), dict):
                config_mgr.config["colorbar_orientation"] = orientation

        main = self._get_main_viewer()
        main_config_mgr = getattr(main, "config_manager", None) if main is not None else None
        if main_config_mgr is not None and isinstance(getattr(main_config_mgr, "config", None), dict):
            main_config_mgr.config["colorbar_orientation"] = orientation

        menu_bar = getattr(main, "menu_bar", None) if main is not None else None
        panel = getattr(menu_bar, "config_panel", None) if menu_bar is not None else None
        if panel is not None and hasattr(panel, "cbar_orientation_combo"):
            try:
                combo = panel.cbar_orientation_combo
                blocked = combo.blockSignals(True)
                combo.setCurrentText(orientation)
                combo.blockSignals(blocked)
            except Exception:
                pass

    def _apply_colorbar_auto_layout(self, force: bool = False, *, redraw: bool = True) -> bool:
        main = self._get_main_viewer()
        if main is not None and main is not self:
            apply_layout = getattr(main, "_apply_colorbar_auto_layout", None)
            if callable(apply_layout):
                try:
                    return bool(apply_layout(force=force))
                except Exception:
                    return False
            return False

        if not force and not self._is_colorbar_auto_layout_enabled():
            return False
        if bool(getattr(self, "_applying_colorbar_auto_layout", False)):
            return False

        config = getattr(getattr(self, "config_manager", None), "config", {}) or {}
        placement = str(config.get("colorbar_placement", "right") or "right")
        align = str(config.get("colorbar_align", "center") or "center")
        gap_px = config.get("colorbar_gap_px", 24.0)
        gap_x_px = config.get("colorbar_gap_x_px", gap_px)
        gap_y_px = config.get("colorbar_gap_y_px", gap_px)
        thickness_px = config.get("colorbar_thickness_px", 24.0)
        length_mode = str(config.get("colorbar_length_mode", "ratio") or "ratio")
        length_value = config.get("colorbar_length_value", 1.0)
        target_orientation = orientation_for_placement(
            placement,
            fallback=config.get("colorbar_orientation", "vertical"),
        )
        def _collect_targets():
            targets = {}
            for plane in ("xy", "xz", "zy"):
                state = self.get_viewer_state(plane)
                if state is None or state.ax is None or state.cax is None:
                    continue
                try:
                    ax_bounds = tuple(float(v) for v in state.ax.get_position().bounds)
                    cbar_bounds = tuple(float(v) for v in state.cax.get_position().bounds)
                except Exception:
                    continue
                if len(ax_bounds) != 4 or len(cbar_bounds) != 4:
                    continue
                if ax_bounds[2] <= 0.0 or ax_bounds[3] <= 0.0:
                    continue
                fig = getattr(state, "fig", None)
                fig_w_px = float(getattr(getattr(fig, "bbox", None), "width", 0.0) or 0.0)
                fig_h_px = float(getattr(getattr(fig, "bbox", None), "height", 0.0) or 0.0)
                if fig_w_px <= 0.0:
                    fig_w_px = 1.0
                if fig_h_px <= 0.0:
                    fig_h_px = 1.0

                target_x, target_y, target_w, target_h, _ = compute_colorbar_geometry(
                    ax_bounds,
                    fig_w_px,
                    fig_h_px,
                    placement=placement,
                    align=align,
                    gap_px=gap_px,
                    gap_x_px=gap_x_px,
                    gap_y_px=gap_y_px,
                    thickness_px=thickness_px,
                    length_mode=length_mode,
                    length_value=length_value,
                )

                targets[plane] = {
                    "state": state,
                    "current": cbar_bounds,
                    "target": (target_x, target_y, target_w, target_h),
                }
            return targets

        targets = _collect_targets()
        if not targets:
            return False

        current_orientation = str(config.get("colorbar_orientation", "vertical")).lower()
        orientation_changed = target_orientation != current_orientation

        eps = 1e-6
        geometry_changed = bool(force)
        if not geometry_changed:
            for payload in targets.values():
                current_bounds = payload["current"]
                target_bounds = payload["target"]
                if any(abs(current - target) > eps for current, target in zip(current_bounds, target_bounds)):
                    geometry_changed = True
                    break
        if not orientation_changed and not geometry_changed:
            return False

        self._applying_colorbar_auto_layout = True
        try:
            if orientation_changed:
                self._set_colorbar_orientation_config(target_orientation)
                self._rebuild_colorbars()
                targets = _collect_targets()
                if not targets:
                    return False

            if geometry_changed or orientation_changed:
                for payload in targets.values():
                    state = payload["state"]
                    target_x, target_y, target_w, target_h = payload["target"]
                    self._apply_colorbar_geometry_to_state(
                        state,
                        target_x,
                        target_y,
                        target_w,
                        target_h,
                        redraw=redraw,
                    )

                anchor = targets.get("xy")
                if anchor is None:
                    anchor = next(iter(targets.values()))
                anchor_x, anchor_y, anchor_w, anchor_h = anchor["target"]
                self._update_colorbar_geometry_config(anchor_x, anchor_y, anchor_w, anchor_h)
        finally:
            self._applying_colorbar_auto_layout = False
        return True

    def fit_colorbar_now(self) -> bool:
        return bool(self._apply_colorbar_auto_layout(force=True))

    def _schedule_colorbar_auto_layout(self, force: bool = False):
        if not force and bool(getattr(self, "_suspend_colorbar_auto_layout", False)):
            return
        if not self._is_colorbar_auto_layout_enabled():
            return
        if bool(getattr(self, "_colorbar_sync_redraw_in_progress", False)):
            return
        from_draw_event = bool(getattr(self, "_colorbar_layout_from_draw_event", False))
        changed = bool(self._apply_colorbar_auto_layout(force=force, redraw=not from_draw_event))
        if changed and from_draw_event:
            canvas = getattr(self, "canvas", None)
            if canvas is not None:
                self._colorbar_sync_redraw_in_progress = True
                try:
                    self._request_canvas_redraw(canvas, immediate=True)
                finally:
                    self._colorbar_sync_redraw_in_progress = False

    def _colorbar_layout_anchor_signature(self):
        state = getattr(self, "state", None)
        ax = getattr(state, "ax", None) if state is not None else getattr(self, "ax", None)
        fig = getattr(state, "fig", None) if state is not None else getattr(self, "fig", None)
        if ax is None or fig is None:
            return None
        try:
            ax_bounds = tuple(round(float(v), 8) for v in ax.get_position().bounds)
            fig_w = round(float(getattr(getattr(fig, "bbox", None), "width", 0.0) or 0.0), 3)
            fig_h = round(float(getattr(getattr(fig, "bbox", None), "height", 0.0) or 0.0), 3)
        except Exception:
            return None
        return ax_bounds + (fig_w, fig_h)

    def _schedule_colorbar_auto_layout_if_anchor_changed(self, force: bool = False):
        if not force and bool(getattr(self, "_suspend_colorbar_auto_layout", False)):
            return
        if force:
            if not self._is_colorbar_auto_layout_enabled():
                return
            self._colorbar_auto_anchor_sig = None
            self._schedule_colorbar_auto_layout(force=True)
            return
        if not self._is_colorbar_auto_layout_enabled():
            return
        sig = self._colorbar_layout_anchor_signature()
        if sig is None:
            return
        if sig == getattr(self, "_colorbar_auto_anchor_sig", None):
            return
        self._colorbar_auto_anchor_sig = sig
        self._schedule_colorbar_auto_layout(force=False)

    def _handle_colorbar_double_click(self, event) -> bool:
        if not getattr(event, "dblclick", False):
            return False
        if not self._is_colorbar_axes(getattr(event, "inaxes", None)):
            return False
        
        # Toggle auto-layout configuration
        main = self._get_main_viewer()
        config_mgr = getattr(main, "config_manager", None)
        if config_mgr is not None and isinstance(getattr(config_mgr, "config", None), dict):
            current_state = bool(config_mgr.config.get("colorbar_auto_layout", True))
            new_state = not current_state
            config_mgr.config["colorbar_auto_layout"] = new_state
            
            state_str = "ON" if new_state else "OFF"
            print(f"Colorbar Auto-Layout: {state_str}")
            
            if new_state:
                # If turning ON, snap to the correct position immediately
                self._schedule_colorbar_auto_layout(force=True)
                
        return True

    def _begin_colorbar_drag(self, event) -> bool:
        if getattr(self, "toolbar", None) is not None and self.toolbar.mode in ("zoom rect", "pan/zoom"):
            return False
        if self._is_colorbar_auto_layout_enabled():
            return False
        if getattr(event, "dblclick", False):
            return False
        if getattr(event, "button", None) not in (1, 3):
            return False
        cax = getattr(event, "inaxes", None)
        if not self._is_colorbar_axes(cax):
            return False
        xfig, yfig = self._event_to_figure_coords(event)
        if xfig is None or yfig is None:
            return True
        x0, y0, w0, h0 = [float(v) for v in cax.get_position().bounds]
        key_text = str(getattr(event, "key", "") or "").lower()
        mode = "resize" if (event.button == 3 or "shift" in key_text) else "move"
        self._colorbar_drag_state = {
            "mode": mode,
            "start_x": xfig,
            "start_y": yfig,
            "orig_x": x0,
            "orig_y": y0,
            "orig_w": w0,
            "orig_h": h0,
        }
        return True

    def _drag_colorbar(self, event) -> bool:
        drag_state = getattr(self, "_colorbar_drag_state", None)
        if not drag_state:
            return False
        xfig, yfig = self._event_to_figure_coords(event)
        if xfig is None or yfig is None:
            return True

        dx = xfig - drag_state["start_x"]
        dy = yfig - drag_state["start_y"]
        if drag_state["mode"] == "move":
            pos_x = drag_state["orig_x"] + dx
            pos_y = drag_state["orig_y"] + dy
            width = drag_state["orig_w"]
            height = drag_state["orig_h"]
        else:
            pos_x = drag_state["orig_x"]
            pos_y = drag_state["orig_y"]
            width = drag_state["orig_w"] + dx
            height = drag_state["orig_h"] + dy

        self._set_colorbar_geometry(pos_x, pos_y, width, height)
        return True

    def _end_colorbar_drag(self, event) -> bool:
        if getattr(self, "_colorbar_drag_state", None) is None:
            return False
        self._drag_colorbar(event)
        self._colorbar_drag_state = None
        return True
        
    def _pv_arrow_active(self) -> bool:
        panel = getattr(self, 'control_panel', None)
        if panel is None:
            main = getattr(FITSViewer, 'main_window', None)
            if main is not None:
                panel = getattr(main, 'control_panel', None)
        if panel is None:
            return False
        pvd = getattr(panel, 'pvd_panel', None)
        if pvd is None:
            return False
        try:
            return bool(pvd.isVisible())
        except Exception:
            return True

    def _sync_overlay_axes_to_main(self):
        ax = getattr(self, "ax", None)
        overlay_ax = getattr(self, "overlay_ax", None)
        if ax is None or overlay_ax is None:
            return False

        try:
            overlay_ax.set_position(ax.get_position())
        except Exception:
            pass

        synced = False
        try:
            ax_xlim = tuple(float(v) for v in ax.get_xlim())
            overlay_xlim = tuple(float(v) for v in overlay_ax.get_xlim())
            if overlay_xlim != ax_xlim:
                overlay_ax.set_xlim(*ax_xlim)
                synced = True
        except Exception:
            pass

        try:
            ax_ylim = tuple(float(v) for v in ax.get_ylim())
            overlay_ylim = tuple(float(v) for v in overlay_ax.get_ylim())
            if overlay_ylim != ax_ylim:
                overlay_ax.set_ylim(*ax_ylim)
                synced = True
        except Exception:
            pass

        return synced

    def update_overlay_position(self, event):
        canvas = getattr(event, 'canvas', None)
        if canvas is not None and canvas is not self.canvas:
            return
        if not getattr(self, '_overlay_updates_enabled', True):
            return
        if getattr(self, '_updating_overlay', False):
            return
        self._updating_overlay = True
        try:
            self._sync_overlay_axes_to_main()
            self._position_click_label()
            self._colorbar_layout_from_draw_event = True
            try:
                self._schedule_colorbar_auto_layout_if_anchor_changed(force=False)
            finally:
                self._colorbar_layout_from_draw_event = False
            self._invalidate_image_background()

            hidden_regions = []
            hidden_markers = []
            region_manager = getattr(self, 'region_manager', None)
            if region_manager is not None:
                hidden_regions = region_manager.prepare_for_background_capture()
            marker_manager = getattr(self, 'marker_manager', None)
            if marker_manager is not None:
                hidden_markers = marker_manager.prepare_for_background_capture(self.plane)

            vline_visible = self.vline.get_visible()
            hline_visible = self.hline.get_visible()
            cpoint_visible = bool(getattr(self, "cpoint", None) is not None and self.cpoint.get_visible())
            self.vline.set_visible(False)
            self.hline.set_visible(False)
            if getattr(self, "cpoint", None) is not None:
                self.cpoint.set_visible(False)
            background = self.canvas.copy_from_bbox(self.overlay_ax.bbox)
            self._background = background
            self._background_initialized = True
            if hasattr(self, 'state') and self.state is not None:
                 self.state.update_background(background)
            
            self.vline.set_visible(vline_visible)
            self.hline.set_visible(hline_visible)
            cursor_x = getattr(self.state, "cursor_x", None) if hasattr(self, "state") else None
            cursor_y = getattr(self.state, "cursor_y", None) if hasattr(self, "state") else None
            if self.plane == 'xy':
                xval = cursor_x if cursor_x is not None else self._get_shared_xpix()
                yval = cursor_y if cursor_y is not None else self._get_shared_ypix()
                self.vline.set_xdata([xval])
                self.hline.set_ydata([yval])
            elif self.plane == 'xz':
                xval = cursor_x if cursor_x is not None else self._get_shared_xpix()
                yval = cursor_y if cursor_y is not None else self._get_shared_zpix()
                self.vline.set_xdata([xval])
                self.hline.set_ydata([yval])
            elif self.plane == 'zy':
                xval = cursor_x if cursor_x is not None else self._get_shared_zpix()
                yval = cursor_y if cursor_y is not None else self._get_shared_ypix()
                self.vline.set_xdata([xval])
                self.hline.set_ydata([yval])
            self._set_crosshair_point_for_plane(self.plane, x=xval, y=yval)
            if getattr(self, "cpoint", None) is not None:
                self.cpoint.set_visible(cpoint_visible)
            self.overlay_ax.draw_artist(self.vline)
            self.overlay_ax.draw_artist(self.hline)
            if getattr(self, "cpoint", None) is not None and self.cpoint.get_visible():
                self.overlay_ax.draw_artist(self.cpoint)

            pending_hidden = getattr(self, '_pending_region_restore', [])
            restore_batches = []
            if hidden_regions:
                restore_batches.append(hidden_regions)
            if pending_hidden:
                restore_batches.append(pending_hidden)
                self._pending_region_restore = []

            if region_manager is not None:
                for batch in restore_batches:
                    region_manager.restore_after_background_capture(batch)
            if marker_manager is not None:
                if hidden_markers:
                    marker_manager.restore_after_background_capture(hidden_markers)
                marker_manager.draw_markers_for_blit()
                if self.plane == 'xy':
                    marker_manager.redraw_planes(['xz', 'zy'])

            if restore_batches and getattr(self, 'plane', None) == 'xy':
                self.redraw_main_overlay_and_blit()
        finally:
            QTimer.singleShot(0, lambda: setattr(self, '_updating_overlay', False))


    def initUI(self, plane):
        self.plane = plane
        self._pending_drag_coords = None
        self._last_drag_update_key = None
        self._colorbar_drag_state = None
        self._colorbar_layout_from_draw_event = False
        self._colorbar_sync_redraw_in_progress = False
        self._colorbar_auto_anchor_sig = None
        # Initialize per-plane ViewerState
        self.state = ViewerState(plane)
        self.state.set_viewer(self)

        self.displaymap = DisplayMap(
            self.data,
            self.header,
            self.wcs,
            self.config_manager.config,
            viewer_state=self.state,
            large_data_mode=self.is_large_data_mode(),
        )
        self.im, self.ax = self.displaymap.display(self.fig, plane)
        self.ax.format_coord = self.formatter
        self.overlay_ax = self.displaymap.overlay_ax
        # draw_event/motion handlers are attached after FigureCanvas is created
        
        # Update ViewerState with matplotlib objects
        self.state.update_im(self.im)
        self.state.update_ax(self.ax)
        self.state.update_overlay_ax(self.overlay_ax)
        self.state.update_fig(self.fig)
        # Colorbar and cax are set by displaymap.display() via viewer_state


        self.format_pix = Format_pix_to_wcs(
            self.wcs,
            self.displaymap.slices,
            self.ax,
            self.plane,
            self.decimal,
            self.number_decimals,
            self.coord_wrap,
            fits_viewer=self,
            auto_precision_digits=self.auto_precision_digits,
        )
        self.format_pix.convert(self.plane, 0, 0)
        

        self.canvas = FigureCanvas(self.fig)
        self.canvas.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas.updateGeometry()
        self.state.update_canvas(self.canvas)
        self._overlay_updates_enabled = True
        self.canvas.mpl_connect("draw_event", self.update_overlay_position)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion_redirect)

        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.canvas.mpl_connect('key_release_event', self.on_key_release)
        self.last_update_time = 0 
        self.canvas.mpl_connect('motion_notify_event', self.cursor_position)

       
        self.toolbar = MyNavigationToolbar(self.canvas, self, self.plane, self.ax, default_image_name = self.filename)
        #self.canvas.mpl_connect('motion_notify_event', self.update_toolbar_message)

        # Add matplotlib widget to a window

        central_widget = QWidget()
        layout = QGridLayout(central_widget)
        layout.setHorizontalSpacing(3)
        layout.setVerticalSpacing(3)
        layout.setContentsMargins(12, 0, 12, 12)
        # Keep the right-side action buttons (Spec/Smooth/Integ/Colorscale) anchored to the right.
        layout.setColumnStretch(10, 1)
        
        # Channel slider and related controls
        has_depth = self.wcs.naxis > 2 and 'NAXIS3' in self.header
        if has_depth:
            self.canvas.mpl_connect('scroll_event', self.scroll_slider_mpl)
            self.slider = QSlider(Qt.Orientation.Horizontal)
            self.slider.setTickPosition(QSlider.TickPosition.TicksAbove)
            #self.slider = QScrollBar(Qt.Orientation.Horizontal)

            self.current_value_label = QLabel("1") 
            self.current_value_label.setFixedWidth(30)
        
            self.chval_box = QLineEdit()
            if self.plane == 'xy':
                naxis = self.header['NAXIS3']
            elif self.plane == 'xz':
                naxis = self.header['NAXIS2']
            elif self.plane == 'zy':
                naxis = self.header['NAXIS1']            
            rmin, rmax = 0, naxis-1
            self.slider.setTickInterval(int(naxis/10.))
            self.slider.setObjectName("slider")
            self.slider.setSingleStep(1)
            self.slider.setRange(rmin,rmax)
            self.slider.valueChanged.connect(self.scroll_slider)
        
            self.chval_box.setObjectName("chval_vox")
            self.chval_box.setMaximumWidth(80)
            if self.plane in ("xz", "zy"):
                self.chval_box.setToolTip("Slice value is interpreted in the cube axis frame.")
            init_z = self.format_pix.convert_chpix_to_world(self.plane, 0, 0, 0)
            z_str = self.format_pix.convert_chval_to_world_str(self.plane, init_z)
            self.chval_box.setText(z_str)
            self.chval_box.setCursorPosition(0)
            self.b_button = QPushButton('◀︎',self)
            self.n_button = QPushButton('▶︎',self)
            self.b_button.setMaximumWidth(25)
            self.n_button.setMaximumWidth(25)
            self.n_button.clicked.connect(self.forward_ch)
            self.b_button.clicked.connect(self.backward_ch)
            self.n_button.setAutoRepeat(True)
            self.b_button.setAutoRepeat(True)
            layout.addWidget(self.current_value_label, 0, 19, 1, 1, alignment=Qt.AlignmentFlag.AlignLeft) 
            
            layout.addWidget(self.b_button, 0, 4, 1, 1)
            layout.addWidget(self.slider, 0, 5, 1, 13)
            layout.addWidget(self.n_button, 0, 18, 1, 1)
            layout.addWidget(self.chval_box, 0, 20, 1, 1, alignment=Qt.AlignmentFlag.AlignCenter)
            
            self.integ_button = QPushButton('Integ', self)
            self.spec_button = QPushButton('Spec', self)

            self.slider.sliderPressed.connect(self.clicked_slider)
            self.chval_box.returnPressed.connect(self.get_chval)
            self.state.update_slider(self.slider)
            self.state.update_chval_box(self.chval_box)
        else:
            # No depth axis (2D): still place Spec/Integ buttons but disabled
            self.integ_button = QPushButton('Integ', self)
            self.spec_button = QPushButton('Spec', self)
            self.integ_button.setEnabled(False)
            self.spec_button.setEnabled(False)

        if self.plane == 'xy' or self.plane == 'xz':
            # X-axis range input fields (MainWindow & SubWindow1 horizontal)
            self.xr_label = QLabel('X:')
            self.xr_label.setFixedWidth(11)
            self.x_min_input = QLineEdit(self)
            self.x_min_input.setPlaceholderText("X min value")
            
            self.x_min_input.setFixedWidth(90)
            self.x_min_input.returnPressed.connect(self.set_x_range)
            self.x_max_input = QLineEdit(self)
            self.x_max_input.setPlaceholderText("X max value")
            self.x_max_input.setFixedWidth(90)
            self.x_max_input.returnPressed.connect(self.set_x_range)
            self.x_button = QPushButton('Set X', self)
            self.x_button.clicked.connect(self.set_x_range)
            self.x_button.setFixedWidth(45)
            
            self.state.update_xrange_input(self.x_min_input, self.x_max_input)

            layout.addWidget(self.xr_label, 0, 0)
            layout.addWidget(self.x_min_input, 0, 1)
            layout.addWidget(self.x_max_input, 0, 2)
            layout.addWidget(self.x_button, 0, 3)

        if self.plane == 'xy' or self.plane == 'zy':
            # Y-axis range input fields (MainWindow & SubWindow1 horizontal)
            self.yr_label = QLabel('Y:')
            self.yr_label.setFixedWidth(11)
            self.y_min_input = QLineEdit(self)
            self.y_min_input.setPlaceholderText("Y min value")
            
            self.y_min_input.setFixedWidth(90)
            self.y_min_input.returnPressed.connect(self.set_y_range)
            self.y_max_input = QLineEdit(self)
            self.y_max_input.setPlaceholderText("Y max value")
            self.y_max_input.setFixedWidth(90)
            self.y_max_input.returnPressed.connect(self.set_y_range)
            self.y_button = QPushButton('Set Y', self)
            self.y_button.clicked.connect(self.set_y_range)
            self.y_button.setFixedWidth(45)
    
            self.state.update_yrange_input(self.y_min_input, self.y_max_input)

            layout.addWidget(self.yr_label, 1, 0)
            layout.addWidget(self.y_min_input, 1, 1)
            layout.addWidget(self.y_max_input, 1, 2)
            layout.addWidget(self.y_button, 1, 3)

        if self.plane == 'xz' or self.plane == 'zy':
            # Z-axis range input fields (MainWindow & SubWindow1 horizontal)
            self.zr_label = QLabel('Z:')
            self.zr_label.setFixedWidth(11)
            self.z_min_input = QLineEdit(self)
            self.z_min_input.setPlaceholderText("Z min value")
            
            self.z_min_input.setFixedWidth(90)
            self.z_min_input.returnPressed.connect(self.set_z_range)
            self.z_max_input = QLineEdit(self)
            self.z_max_input.setPlaceholderText("Z max value")
            self.z_max_input.setFixedWidth(90)
            self.z_max_input.returnPressed.connect(self.set_z_range)
            self.z_button = QPushButton('Set Z', self)
            self.z_button.clicked.connect(self.set_z_range)
            self.z_button.setFixedWidth(45)

            self.state.update_zrange_input(self.z_min_input, self.z_max_input)

            if self.plane == 'xz':
                layout.addWidget(self.zr_label, 1, 0)
                layout.addWidget(self.z_min_input, 1, 1)
                layout.addWidget(self.z_max_input, 1, 2)
                layout.addWidget(self.z_button, 1, 3)
            elif self.plane == 'zy':
                layout.addWidget(self.zr_label, 0, 0)
                layout.addWidget(self.z_min_input, 0, 1)
                layout.addWidget(self.z_max_input, 0, 2)
                layout.addWidget(self.z_button, 0, 3)

        self.full_button = QPushButton('Full', self)
        self.full_button.clicked.connect(self.reset_all_ranges)
        #self.full_button.setFixedWidth(45)

        self.color_button = QPushButton('Colorscale', self)
        layout.addWidget(self.color_button,  1, 20, 1, 1, alignment=Qt.AlignmentFlag.AlignRight)

        self.smooth_button = QPushButton('Smooth', self)
        layout.addWidget(self.smooth_button, 1, 14, 1, 3, alignment=Qt.AlignmentFlag.AlignRight)

        layout.addWidget(self.integ_button, 1, 17, 1, 3, alignment=Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.spec_button, 1, 11, 1, 3, alignment=Qt.AlignmentFlag.AlignRight)

        layout.addWidget(self.full_button,  1, 4, 1, 3)
        layout.addWidget(self.canvas, 2, 0, 1, 21)
        layout.addWidget(self.toolbar, 3, 0, 1, 21)
        self.setCentralWidget(central_widget)
        
        
        self.layout = layout
        
        self.label = QLabel(self.canvas)
        self.label.setStyleSheet("QLabel { color : %s; }" % self.click_label_color)
        self.label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.state.update_poslabel(self.label)
        
        
        # Create HPBW ellipse only on the XY plane.
        if self.plane == 'xy':
            self.hpbw = AddHPBW(self.state.overlay_ax, self.header, self.config_manager.config)
        else:
            self.hpbw = None
        self.ch_label = self.overlay_ax.text(self.pos_chlabel_x, self.pos_chlabel_y, "", 
                            transform=self.ax.transAxes, verticalalignment='bottom', horizontalalignment='right', 
                            fontsize=self.ch_label_size, fontfamily=self.ch_label_font, color=self.ch_label_color)
        self.state.update_chlabel(self.ch_label)
        self.state.update_hpbw(self.hpbw)
        
        
        self._position_click_label()

        # Capture baseline pixel limits for later full-range resets.
        self.original_xlim = self.ax.get_xlim()
        self.original_ylim = self.ax.get_ylim()
        if self.data.ndim > 2:
            depth_len = self.data.shape[self.data.ndim - 3]
            self.original_zlim = (-0.5, depth_len - 0.5)
        else:
            self.original_zlim = (0.0, 0.0)

        
        # Create crosshair lines for all planes
        self.hline = self.overlay_ax.axhline(
            y=0,
            visible=False,
            lw=self.click_linewidth,
            c=self.click_linecolor,
            ls=self.click_linestyle,
            alpha=self.click_alpha,
            animated=True,
        )
        self.vline = self.overlay_ax.axvline(
            x=0,
            visible=False,
            lw=self.click_linewidth,
            c=self.click_linecolor,
            ls=self.click_linestyle,
            alpha=self.click_alpha,
            animated=True,
        )
        self.cpoint, = self.overlay_ax.plot(
            [],
            [],
            marker='o',
            linestyle='None',
            markersize=4.0,
            markerfacecolor=self.click_linecolor,
            markeredgecolor=self.click_linecolor,
            alpha=self.click_alpha,
            visible=False,
            animated=True,
        )
        self._background = None

        self.state.update_lines(self.hline, self.vline, self.cpoint)
        self.state.update_background(self._background)
        self._background_initialized = False
        
    def _large_data_profile(self) -> Dict[str, object]:
        metadata = self.spectral_metadata if isinstance(self.spectral_metadata, dict) else {}
        config = getattr(getattr(self, "config_manager", None), "config", None)
        profile = build_large_data_profile(self.data, header=self.header, config=config)
        metadata["large_data_profile"] = profile
        metadata["large_data_mode"] = bool(profile.get("enabled"))
        self.spectral_metadata = metadata
        return profile

    def is_large_data_mode(self) -> bool:
        return bool(self._large_data_profile().get("enabled"))

    def _large_data_display_max_dimension(self) -> int:
        config = getattr(getattr(self, "config_manager", None), "config", {}) or {}
        try:
            return max(256, int(config.get("large_data_display_max_dim", DEFAULT_LARGE_DATA_DISPLAY_MAX_DIM)))
        except Exception:
            return DEFAULT_LARGE_DATA_DISPLAY_MAX_DIM

    def _large_data_cache_key(self, plane: str, cache_index: Optional[int], shape) -> Optional[tuple]:
        if cache_index is None:
            return None
        try:
            normalized_shape = tuple(int(v) for v in shape)
            return (str(plane or ""), int(cache_index), normalized_shape)
        except Exception:
            return None

    def _get_large_data_display_slice(self, plane: str, plane_data, *, cache_index: Optional[int] = None):
        if not self.is_large_data_mode():
            return plane_data

        cache_key = self._large_data_cache_key(plane, cache_index, getattr(plane_data, "shape", None))
        if cache_key is not None and cache_key in self._large_data_slice_cache:
            cached = self._large_data_slice_cache.pop(cache_key)
            self._large_data_slice_cache[cache_key] = cached
            return cached

        display_data = downsample_2d_for_display(
            plane_data,
            max_dimension=self._large_data_display_max_dimension(),
        )
        if cache_key is not None:
            self._large_data_slice_cache[cache_key] = display_data
            while len(self._large_data_slice_cache) > self._large_data_slice_cache_limit:
                self._large_data_slice_cache.popitem(last=False)
        return display_data

    def _set_plane_image_data(self, plane: str, plane_data, *, cache_index: Optional[int] = None) -> bool:
        im = self._get_plane_im(plane)
        if im is None or plane_data is None:
            return False
        if not self.is_large_data_mode():
            im.set_data(plane_data)
            return True

        display_data = self._get_large_data_display_slice(plane, plane_data, cache_index=cache_index)
        im.set_data(display_data)
        try:
            height, width = plane_data.shape[-2], plane_data.shape[-1]
            im.set_extent((-0.5, width - 0.5, -0.5, height - 0.5))
            im.set_interpolation("nearest")
        except Exception:
            pass
        return True

    def _plane_length_for_index(self, plane: str) -> int:
        cube = getattr(self, "cube", None)
        if cube is None:
            data = getattr(self, "data", None)
            if plane == "xy" and data is not None and getattr(data, "ndim", 0) == 2:
                return 1
            return 0
        if plane == "xy":
            return int(cube.shape[0])
        if plane == "xz":
            return int(cube.shape[1])
        if plane == "zy":
            return int(cube.shape[2])
        return 0

    def _plane_slice_for_index(self, plane: str, index: int):
        if plane == "xy":
            if getattr(self.data, "ndim", 0) == 2:
                return self.data
            return self.cube[index]
        if plane == "xz":
            return self.cube[:, index, :]
        if plane == "zy":
            return self.cube[:, :, index].T
        raise KeyError(plane)

    def _schedule_large_data_prefetch(self, plane: str, cache_index: Optional[int]) -> None:
        if not self.is_large_data_mode() or cache_index is None:
            return
        if plane not in ("xy", "xz", "zy"):
            return
        self._large_data_prefetch_request = (str(plane), int(cache_index))
        self._large_data_prefetch_timer.start(0)

    def _run_large_data_prefetch(self) -> None:
        request = self._large_data_prefetch_request
        self._large_data_prefetch_request = None
        if not request or not self.is_large_data_mode():
            return

        plane, index = request
        plane_length = self._plane_length_for_index(plane)
        if plane_length <= 1:
            return

        for neighbor in (index - 1, index + 1):
            if neighbor < 0 or neighbor >= plane_length:
                continue
            try:
                neighbor_slice = self._plane_slice_for_index(plane, neighbor)
            except Exception:
                continue
            cache_key = self._large_data_cache_key(plane, neighbor, getattr(neighbor_slice, "shape", None))
            if cache_key is not None and cache_key in self._large_data_slice_cache:
                continue
            self._get_large_data_display_slice(plane, neighbor_slice, cache_index=neighbor)

    def _large_data_notice_text(self, profile: Dict[str, object]) -> str:
        estimated_size = str(profile.get("estimated_size_text") or "unknown size")
        threshold = str(profile.get("threshold_text") or "unknown threshold")
        return f"Large Data Mode is active. Estimated size: {estimated_size} (threshold: {threshold})."

    def report_large_data_mode(self) -> None:
        profile = self._large_data_profile()
        enabled = bool(profile.get("enabled"))
        if not enabled:
            return

        metadata = self.spectral_metadata if isinstance(self.spectral_metadata, dict) else {}
        if bool(metadata.get("_large_data_notice_reported", False)):
            return
        metadata["_large_data_notice_reported"] = True
        self.spectral_metadata = metadata

        print(f"\033[1;33m\033[1mWarning: {self._large_data_notice_text(profile)}\033[0m")

    
    def reload_viewer(self):
        """Reload the viewer based on the updated configuration settings for all windows."""
        crosshair_snapshot = self._snapshot_crosshair_state()
        config = self.config_manager.config
        colorscale = config.get('colorscale')
        bad_color = config.get('bad_color')
        self.format_pix.decimal = config.get('decimal')
        self.auto_precision_digits = bool(config.get('auto_precision_digits', True))
        self.format_pix.auto_precision_digits = bool(config.get('auto_precision_digits', True))
        self.format_pix.number_decimals = config.get('number_decimals')
        self.format_pix.coord_wrap = config.get('coord_wrap')
        self.scrollspeed = config.get('scrollspeed', 0.1)
        self.invert_wheel_direction = bool(config.get('invert_wheel_direction', False))
        self.poslabel_x = config.get('poslabel_x', 0.99)
        self.poslabel_y = config.get('poslabel_y', 0.99)
        self.poslabel_w = config.get('poslabel_w', 250)
        self.poslabel_h = config.get('poslabel_h', 30)
        self.click_label_color = config.get('click_label_color')
        self.click_linewidth = config.get('click_linewidth', 0.25)
        self.click_linecolor = config.get('click_linecolor', 'cyan')
        self.click_linestyle = str(config.get('click_linestyle', '-'))
        self.click_alpha = float(config.get('click_alpha', 1.0))
        self.click_show_crosshair = bool(config.get('click_show_crosshair', True))
        self.click_crosshair_mode = str(config.get('click_crosshair_mode', 'both'))
        self.click_show_center_marker = bool(config.get('click_show_center_marker', False))
        self.label.setStyleSheet("QLabel { color : %s; }" % self.click_label_color)
        self._position_click_label()
        
        self.pos_chlabel_x = config.get('pos_chlabel_x')
        self.pos_chlabel_y = config.get('pos_chlabel_y')
        
        self.ch_label_color = config.get('ch_label_color')
        self.ch_label_size = config.get('ch_label_size')
        self.ch_label_font = config.get('ch_label_font')

        if hasattr(self, 'displaymap') and self.displaymap is not None:
            if colorscale:
                self.displaymap.colorscale = colorscale
            if bad_color:
                self.displaymap.bad_color = bad_color
        
        self.beam_facecolor = config.get('beam_facecolor')
        self.beam_edgecolor = config.get('beam_edgecolor')
        self.beam_linewidth = config.get('beam_linewidth')
        self.beam_pos_x = config.get('beam_pos_x', 0.1)
        self.beam_pos_y = config.get('beam_pos_y', 0.1)
        
        
        self.resize(config.get('figure_width'), config.get('figure_height'))
        self.fig.subplots_adjust( left = config.get('ax_pos_l'), 
                                right = config.get('ax_pos_r'),
                                bottom = config.get('ax_pos_b'), 
                                top = config.get('ax_pos_t'))
        
        if self.subwindows:
            for subwindow in self.subwindows:
                subwindow.fig.subplots_adjust( left = config.get('ax_pos_l'), 
                                right = config.get('ax_pos_r'),
                                bottom = config.get('ax_pos_b'), 
                                top = config.get('ax_pos_t'))
                subwindow.resize(config.get('figure_width'), config.get('figure_height'))
                subwindow.format_pix.decimal = config.get('decimal')
                subwindow.auto_precision_digits = bool(config.get('auto_precision_digits', True))
                subwindow.format_pix.auto_precision_digits = bool(config.get('auto_precision_digits', True))
                subwindow.format_pix.number_decimals = config.get('number_decimals')
                subwindow.format_pix.coord_wrap = config.get('coord_wrap')
                subwindow.scrollspeed = config.get('scrollspeed', 0.1)
                subwindow.invert_wheel_direction = bool(config.get('invert_wheel_direction', False))
                subwindow.poslabel_x = config.get('poslabel_x', 0.99)
                subwindow.poslabel_y = config.get('poslabel_y', 0.99)
                subwindow.poslabel_w = config.get('poslabel_w', 250)
                subwindow.poslabel_h = config.get('poslabel_h', 30)
                subwindow.label.setStyleSheet("QLabel { color : %s; }" % self.click_label_color)
                
                subwindow.pos_chlabel_x = config.get('pos_chlabel_x')
                subwindow.pos_chlabel_y = config.get('pos_chlabel_y')
                #subwindow.pos_chlabel_w = config.get('pos_chlabel_w')
                #subwindow.pos_chlabel_h = config.get('pos_chlabel_h')
                subwindow.ch_label_color = config.get('ch_label_color')
                subwindow.ch_label_size = config.get('ch_label_size')
                subwindow.ch_label_font = config.get('ch_label_font')
                #subwindow.label2.setStyleSheet("QLabel { color : %s; }" % self.ch_label_color)
                #subwindow.label2.setFont(font)
                if hasattr(subwindow, 'displaymap') and subwindow.displaymap is not None:
                    if colorscale:
                        subwindow.displaymap.colorscale = colorscale
                    if bad_color:
                        subwindow.displaymap.bad_color = bad_color
                
        self._apply_config_to_plane()
        self._restore_crosshair_state(crosshair_snapshot)

    def _rebuild_colorbars(self):
        config = getattr(getattr(self, "config_manager", None), "config", None)
        if not isinstance(config, dict):
            return

        orientation = str(config.get("colorbar_orientation", "vertical")).lower()
        if orientation not in ("horizontal", "vertical"):
            orientation = "vertical"

        for plane in ("xy", "xz", "zy"):
            state = self.get_viewer_state(plane)
            if state is None or state.fig is None or state.im is None:
                continue

            old_cax = getattr(state, "cax", None)
            colorbar = getattr(state, "colorbar", None)
            if colorbar is not None:
                try:
                    colorbar.remove()
                except Exception:
                    pass
            elif old_cax is not None:
                try:
                    old_cax.remove()
                except Exception:
                    pass

            try:
                state.cax = state.fig.add_axes([
                    config.get("cbar_pos_x", 0.9),
                    config.get("cbar_pos_y", 0.11),
                    config.get("cbar_width", 0.04),
                    config.get("cbar_height", 0.77),
                ])
                state.cax.set_gid("colorbar")
                self._set_colorbar_zorder_for_state(state)
                state.colorbar = state.fig.colorbar(state.im, cax=state.cax, orientation=orientation)
                ColorSettingsPanel.apply_colorbar_settings(state.cax, state.colorbar, config)
                self._set_colorbar_zorder_for_state(state)
            except Exception:
                continue

    def _rebuild_colorbars_if_needed(self):
        for plane in ("xy", "xz", "zy"):
            state = self.get_viewer_state(plane)
            if state is None:
                continue
            if getattr(state, "colorbar", None) is not None:
                self._rebuild_colorbars()
                return

    def _apply_config_to_plane(self):
        """Apply configuration settings to all planes using ViewerState."""
        config = self.config_manager.config

        # Get states for all planes
        xy_state = self.get_viewer_state('xy')
        if xy_state is None and getattr(self, 'plane', None) == 'xy':
            xy_state = self.state

        if self.xlabel is None and xy_state and xy_state.ax_coord:
            self.xlabel = xy_state.ax_coord[0].get_axislabel()
            self.ylabel = xy_state.ax_coord[1].get_axislabel()
            if self.data.ndim > 2:
                xz_state = self.get_viewer_state('xz')
                if xz_state and xz_state.ax_coord:
                    self.zlabel = xz_state.ax_coord[1].get_axislabel()

        xtick_label_position = config.get('xticklabel_position')
        ytick_label_position = config.get('yticklabel_position')

        # Apply config to xy plane
        if xy_state:
            self._apply_config_to_state(xy_state, config, xtick_label_position, ytick_label_position, 'xy')

        # Apply config to xz and zy planes for 3D data
        if self.data.ndim > 2:
            xz_state = self.get_viewer_state('xz')
            zy_state = self.get_viewer_state('zy')

            if xz_state:
                self._apply_config_to_state(xz_state, config, xtick_label_position, ytick_label_position, 'xz')
            if zy_state:
                self._apply_config_to_state(zy_state, config, xtick_label_position, ytick_label_position, 'zy')

        self._rebuild_colorbars_if_needed()

        # Re-apply the decimal or dms format to the tick labels for all planes
        is_decimal = config.get('decimal', True)
        ctype_list = list(getattr(getattr(self.wcs, "wcs", None), "ctype", []) or [])

        for plane in ['xy', 'xz', 'zy']:
            state = self.get_viewer_state(plane)
            if state is None or state.ax is None:
                continue

            try:
                ax = state.ax
                for i in range(ax.wcs.naxis):
                    coord = ax.coords[i]
                    axis_type = str(self.wcs.world_axis_physical_types[i] or "").lower()
                    ctype = str(ctype_list[i] or "").upper() if i < len(ctype_list) else ""
                    ctype_head = ctype.split('-')[0]
                    is_ra = ctype_head.startswith("RA") or axis_type.endswith(".ra")
                    is_dec = ctype_head.startswith("DEC") or axis_type.endswith(".dec")
                    is_gal_or_offset = any(token in ctype_head for token in ("GLON", "GLAT", "OFFSET"))
                    is_angular = (
                        axis_type.endswith(".lon")
                        or axis_type.endswith(".lat")
                        or is_ra
                        or is_dec
                        or is_gal_or_offset
                    )
                    if not is_angular:
                        continue
                    if is_ra:
                        if is_decimal:
                            coord.set_format_unit('deg', decimal=True)
                        else:
                            coord.set_format_unit('hour', decimal=False)
                    else:
                        coord.set_format_unit('deg', decimal=is_decimal)

                if state.canvas:
                    state.canvas.draw_idle()

            except Exception as e:
                # In case of any error, print it and continue
                print(f"Could not update tick format for plane {plane}: {e}")

        self._register_contour_layer()
        if self._is_colorbar_auto_layout_enabled():
            self._schedule_colorbar_auto_layout_if_anchor_changed(force=True)

    def _apply_config_to_state(self, state, config, xtick_label_position, ytick_label_position, plane):
        """Apply configuration settings to a single ViewerState."""
        if state is None:
            return

        if state.fig:
            state.fig.set_facecolor(config.get('fig_background_color'))
        if state.ax:
            state.ax.set_facecolor(config.get('ax_background_color'))
        if state.im:
            colorscale = config.get('colorscale')
            if colorscale:
                try:
                    cmap = colormaps.get_cmap(colorscale)
                except Exception:
                    cmap = None
                if cmap is not None:
                    try:
                        state.im.set_cmap(cmap)
                    except Exception:
                        pass
            try:
                state.im.cmap.set_bad(config.get('bad_color'))
            except Exception:
                pass
            # Invalidate cached backgrounds so the new colormap renders correctly.
            state.image_background = None
            state._background = None
            if state.colorbar is not None:
                try:
                    state.colorbar.update_normal(state.im)
                except Exception:
                    pass
                try:
                    ColorSettingsPanel.apply_colorbar_settings(
                        state.cax,
                        state.colorbar,
                        config,
                    )
                except Exception:
                    pass
            self._set_colorbar_zorder_for_state(state)

        # Apply axis coordinate settings
        if state.ax_coord:
            if plane == 'xy':
                state.ax_coord[0].set_axislabel(self.xlabel, fontsize=config.get('axislabel_fontsize'),
                                                 fontfamily=config.get('axislabel_fontfamily'),
                                                 color=config.get('axislabel_color'))
                state.ax_coord[1].set_axislabel(self.ylabel, fontsize=config.get('axislabel_fontsize'),
                                                 fontfamily=config.get('axislabel_fontfamily'),
                                                 color=config.get('axislabel_color'))
            elif plane == 'xz':
                state.ax_coord[0].set_axislabel(self.xlabel, fontsize=config.get('axislabel_fontsize'),
                                                 fontfamily=config.get('axislabel_fontfamily'),
                                                 color=config.get('axislabel_color'))
                state.ax_coord[1].set_axislabel(self.zlabel, fontsize=config.get('axislabel_fontsize'),
                                                 fontfamily=config.get('axislabel_fontfamily'),
                                                 color=config.get('axislabel_color'))
            elif plane == 'zy':
                state.ax_coord[0].set_axislabel(self.zlabel, fontsize=config.get('axislabel_fontsize'),
                                                 fontfamily=config.get('axislabel_fontfamily'),
                                                 color=config.get('axislabel_color'))
                state.ax_coord[1].set_axislabel(self.ylabel, fontsize=config.get('axislabel_fontsize'),
                                                 fontfamily=config.get('axislabel_fontfamily'),
                                                 color=config.get('axislabel_color'))

            state.ax_coord[0].set_axislabel_position(xtick_label_position)
            state.ax_coord[1].set_axislabel_position(ytick_label_position)

            state.ax_coord[0].set_ticklabel(rotation=config.get('tick_xlabelrotation'), pad=config.get('tick_pad_x'))
            state.ax_coord[0].set_ticklabel_position(xtick_label_position)
            state.ax_coord[0].set_ticks_position(config.get('default_ticks_position'))
            state.ax_coord[1].set_ticklabel(rotation=config.get('tick_ylabelrotation'), pad=config.get('tick_pad_y'))
            state.ax_coord[1].set_ticklabel_position(ytick_label_position)
            state.ax_coord[1].set_ticks_position(config.get('default_ticks_position'))

            if plane == 'xy':
                state.ax_coord[0].set_minor_frequency(config.get('x_mtick_freq', 5))
                state.ax_coord[1].set_minor_frequency(config.get('y_mtick_freq', 5))
            elif plane == 'xz':
                state.ax_coord[0].set_minor_frequency(config.get('x_mtick_freq', 5))
                state.ax_coord[1].set_minor_frequency(config.get('z_mtick_freq', 5))
            elif plane == 'zy':
                state.ax_coord[0].set_minor_frequency(config.get('z_mtick_freq', 5))
                state.ax_coord[1].set_minor_frequency(config.get('y_mtick_freq', 5))

        if state.ax:
            state.ax.tick_params(axis='both', which='major', direction=config.get('tick_direction'),
                                  length=config.get('tick_length'), color=config.get('tick_color'),
                                  width=config.get('tick_width'), labelsize=config.get('tick_labelsize'),
                                  labelcolor=config.get('tick_labelcolor'))
            for spine in state.ax.spines.values():
                spine.set_linewidth(config.get('tick_width'))
                spine.set_color(config.get('tick_color'))
            state.ax.tick_params(which='minor', length=config.get('mtick_length'))

        # Apply crosshair settings
        show_crosshair = bool(config.get('click_show_crosshair', True))
        show_center_marker = bool(config.get('click_show_center_marker', False))
        show_cursor_overlay = bool(show_crosshair or show_center_marker)
        if state.hline:
            state.hline.set_color(config.get('click_linecolor'))
            state.hline.set_linewidth(config.get('click_linewidth'))
            state.hline.set_linestyle(str(config.get('click_linestyle', '-')))
            state.hline.set_alpha(float(config.get('click_alpha', 1.0)))
        if state.vline:
            state.vline.set_color(config.get('click_linecolor'))
            state.vline.set_linewidth(config.get('click_linewidth'))
            state.vline.set_linestyle(str(config.get('click_linestyle', '-')))
            state.vline.set_alpha(float(config.get('click_alpha', 1.0)))
        cpoint = getattr(state, "cpoint", None)
        if cpoint is not None:
            cpoint.set_marker('o')
            cpoint.set_markersize(4.0)
            cpoint.set_markerfacecolor(config.get('click_linecolor'))
            cpoint.set_markeredgecolor(config.get('click_linecolor'))
            cpoint.set_alpha(float(config.get('click_alpha', 1.0)))
        if not show_cursor_overlay:
            self._set_crosshair_visibility_for_plane(plane, False)
        else:
            self._set_crosshair_visibility_for_plane(
                plane,
                bool(
                    (state.hline and state.hline.get_visible()) or
                    (state.vline and state.vline.get_visible()) or
                    (cpoint is not None and cpoint.get_visible())
                ),
            )

        # Apply channel label settings
        if state.chlabel:
            state.chlabel.set_position((self.pos_chlabel_x, self.pos_chlabel_y))
            state.chlabel.set_fontsize(self.ch_label_size)
            state.chlabel.set_fontfamily(self.ch_label_font)
            state.chlabel.set_color(self.ch_label_color)

        # Draw artists
        if state.overlay_ax:
            if state.hline:
                state.overlay_ax.draw_artist(state.hline)
            if state.vline:
                state.overlay_ax.draw_artist(state.vline)
            if cpoint is not None and cpoint.get_visible():
                state.overlay_ax.draw_artist(cpoint)
            if state.chlabel:
                state.overlay_ax.draw_artist(state.chlabel)

    def _cursor_coordinate_frame_label(self, plane: Optional[str] = None) -> str:
        if bool(self.spectral_metadata.get("is_cartesian_interpretation")):
            return "PV (Cartesian)"
        key = str(plane or self.plane or "").lower()
        frame_name = "native"
        if key == "xy":
            current_frame = normalize_display_frame(self._get_shared_display_frame())
            if current_frame != "native":
                frame_name = current_frame
        native_frame = normalize_display_frame(native_celestial_frame(getattr(self, "wcs", None)))
        if frame_name == "native" and native_frame != "native":
            frame_name = native_frame
        label = display_frame_label(frame_name)
        if label == "Native (WCS)":
            return "WCS"
        return label

    @staticmethod
    def _normalize_label_text(text) -> str:
        return " ".join(str(text).split()).strip()

    @staticmethod
    def _plane_coord_keys(plane: Optional[str]) -> tuple[str, str]:
        key = str(plane or "").lower()
        if key == "xz":
            return ("x", "z")
        if key == "zy":
            return ("z", "y")
        return ("x", "y")

    @staticmethod
    def _plane_axis_indices(plane: Optional[str]) -> tuple[int, int]:
        mapping = {
            "xy": (0, 1),
            "xz": (0, 2),
            "zy": (2, 1),
        }
        return mapping.get(str(plane or "").lower(), (0, 1))

    @staticmethod
    def _axis_index_for_plane_key(plane: Optional[str], axis_key: str) -> Optional[int]:
        key = str(plane or "").lower()
        axis = str(axis_key or "").lower()
        mapping = {
            "xy": {"x": 0, "y": 1},
            "xz": {"x": 0, "z": 2},
            "zy": {"z": 2, "y": 1},
        }
        return mapping.get(key, {"x": 0, "y": 1}).get(axis)

    def _axis_type_for_display_plane(self, plane: Optional[str], axis_slot: str) -> str:
        wcs = getattr(self, "wcs", None)
        if wcs is None:
            return ""
        axis_index = self._axis_index_for_plane_key(plane, axis_slot)
        if axis_index is None:
            return ""
        try:
            base_axis_type = axis_type_for_index(wcs, axis_index)
        except Exception:
            base_axis_type = ""
        if str(plane or "").lower() == "xy":
            frame = normalize_display_frame(self._get_shared_display_frame())
            if frame != "native":
                try:
                    return str(display_axis_type(base_axis_type, frame) or "")
                except Exception:
                    return str(base_axis_type or "")
        return str(base_axis_type or "")

    def _axis_unit_label_for_value(self, plane: Optional[str], axis_slot: str, value_text: str) -> str:
        cache = getattr(self, '_axis_unit_cache', None)
        if cache is None:
            cache = {}
            self._axis_unit_cache = cache
        cache_key = (str(plane or "").lower(), str(axis_slot or "").lower(), bool(getattr(self, "decimal", True)))
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
        axis_type = self._axis_type_for_display_plane(plane, axis_slot)
        if axis_is_longitude(axis_type) or axis_is_latitude(axis_type):
            if bool(getattr(self, "decimal", True)):
                result = "deg"
            else:
                result = ""
        else:
            result = self._axis_unit_from_axis_label(plane, axis_slot)
        cache[cache_key] = result
        return result

    def _invalidate_axis_unit_cache(self):
        self._axis_unit_cache = {}

    def _axis_unit_from_axis_label(self, plane: Optional[str], axis_slot: str) -> str:
        axis_key = str(axis_slot or "").lower()
        viewer = self._viewer_for_plane_local(plane)
        ax = getattr(viewer, "ax", None) if viewer is not None else None
        label_text = ""
        if ax is not None:
            xkey, ykey = self._plane_coord_keys(plane)
            try:
                if axis_key == xkey:
                    label_text = str(ax.get_xlabel() or "")
                elif axis_key == ykey:
                    label_text = str(ax.get_ylabel() or "")
            except Exception:
                label_text = ""
        if not label_text:
            fallback = {
                "x": str(getattr(self, "xlabel", "") or ""),
                "y": str(getattr(self, "ylabel", "") or ""),
                "z": str(getattr(self, "zlabel", "") or ""),
            }
            label_text = fallback.get(axis_key, "")
        return self._extract_unit_from_axis_label(label_text)

    @staticmethod
    def _extract_unit_from_axis_label(axis_label: str) -> str:
        text = str(axis_label or "")
        left = text.rfind("[")
        right = text.rfind("]")
        if left == -1 or right == -1 or right <= left + 1:
            return ""
        return text[left + 1:right].strip().replace(" ", "")

    def _format_value_with_axis_unit(self, plane: Optional[str], axis_slot: str, value_text: str) -> str:
        text = self._normalize_label_text(value_text)
        unit = self._axis_unit_label_for_value(plane, axis_slot, text)
        if unit:
            return f"{text} {unit}"
        return text

    def _format_cursor_pair_text(
        self,
        plane: Optional[str],
        x_text: str,
        y_text: str,
        *,
        include_frame: bool = False,
    ) -> str:
        xkey, ykey = self._plane_coord_keys(plane)
        x_with_unit = self._format_value_with_axis_unit(plane, xkey, x_text)
        y_with_unit = self._format_value_with_axis_unit(plane, ykey, y_text)
        coord_text = f"{x_with_unit}, {y_with_unit}"
        if include_frame:
            return f"{coord_text} [{self._cursor_coordinate_frame_label(plane)}]"
        return coord_text

    def _compose_click_label_text(self, coord_text: str, intensity_text: Optional[str] = None) -> str:
        coord_line = self._normalize_label_text(coord_text)
        intensity_line = self._normalize_label_text(intensity_text) if intensity_text else ""
        if intensity_line:
            return f"{coord_line}\n[{intensity_line}]"
        return coord_line

    def _shared_intensity_text(self) -> Optional[str]:
        try:
            xpix = int(round(float(self._get_shared_xpix())))
            ypix = int(round(float(self._get_shared_ypix())))
        except Exception:
            return None

        try:
            if self.data is None:
                return None
            if self.data.ndim == 2:
                if (
                    xpix < 0
                    or ypix < 0
                    or ypix >= self.data.shape[0]
                    or xpix >= self.data.shape[1]
                ):
                    return None
                intensity = self.data[ypix, xpix]
            else:
                cube = getattr(self, "cube", None)
                if cube is None:
                    return None
                try:
                    zpix = int(round(float(self._get_shared_zpix())))
                except Exception:
                    return None
                if (
                    xpix < 0
                    or ypix < 0
                    or zpix < 0
                    or zpix >= cube.shape[0]
                    or ypix >= cube.shape[1]
                    or xpix >= cube.shape[2]
                ):
                    return None
                intensity = cube[zpix, ypix, xpix]
        except Exception:
            return None

        try:
            return self._format_intensity_with_unit(intensity)
        except Exception:
            return None

    def _plane_cursor_pixel_tuple(self, plane: str):
        key = str(plane or "").lower()
        if key == "xy":
            return (float(self._get_shared_xpix()), float(self._get_shared_ypix()))
        if key == "xz":
            return (float(self._get_shared_xpix()), float(self._get_shared_zpix()))
        if key == "zy":
            return (float(self._get_shared_zpix()), float(self._get_shared_ypix()))
        return None

    def _viewer_for_plane_local(self, plane: str):
        key = str(plane or "").lower()
        if key == str(getattr(self, "plane", "")).lower():
            return self
        main = self._get_main_viewer()
        getter = getattr(main, "_viewer_for_plane", None)
        if callable(getter):
            try:
                viewer = getter(key)
                if viewer is not None:
                    return viewer
            except Exception:
                pass
        state = self.get_viewer_state(key)
        if state is not None:
            viewer = getattr(state, "viewer", None)
            if viewer is not None:
                return viewer
        return None

    def _formatted_plane_cursor_text(self, plane: str, *, include_frame: bool = False) -> Optional[str]:
        viewer = self._viewer_for_plane_local(plane)
        if viewer is None:
            return None
        format_pix = getattr(viewer, "format_pix", None)
        if format_pix is None:
            return None
        pixel = self._plane_cursor_pixel_tuple(plane)
        if pixel is None:
            return None

        saved_world = (
            self._get_shared_world_x_str(),
            self._get_shared_world_y_str(),
            self._get_shared_world_z_str(),
            self._get_shared_world_s_str(),
        )
        try:
            xstr, ystr = format_pix.convert(plane, pixel[0], pixel[1])
        except Exception:
            return None
        finally:
            self._update_shared_world_xyz_str(*saved_world)
        return self._format_cursor_pair_text(plane, xstr, ystr, include_frame=include_frame)

    def _format_channel_label_text(self, value_text: str, plane: Optional[str] = None) -> str:
        return str(value_text)

    def refresh_channel_value_display(self):
        if self.data is None or self.data.ndim < 3:
            return
        slider = getattr(self, "slider", None)
        if slider is None:
            return
        try:
            k = int(slider.value())
        except Exception:
            return

        native_value = None
        if self.plane == "xy":
            native_value = self.format_pix.convert_chpix_to_world(
                self.plane,
                self._get_shared_xpix(),
                self._get_shared_ypix(),
                k,
            )
        elif self.plane == "xz":
            native_value = self.format_pix.convert_chpix_to_world(
                self.plane,
                self._get_shared_xpix(),
                k,
                self._get_shared_zpix(),
            )
        elif self.plane == "zy":
            native_value = self.format_pix.convert_chpix_to_world(
                self.plane,
                k,
                self._get_shared_ypix(),
                self._get_shared_xpix(),
            )

        if native_value is None:
            return

        value_text = self.format_pix.convert_chval_to_world_str(self.plane, native_value)
        chval_box = self._get_plane_chval_box(self.plane)
        if chval_box is not None:
            chval_box.setText(value_text)
            chval_box.setCursorPosition(0)
        chlabel = self._get_plane_chlabel(self.plane)
        if chlabel is not None:
            chlabel.set_text(self._format_channel_label_text(value_text, self.plane))

    def forward_ch(self):
        self.slider.setValue(self.slider.value()+1)
        
    def backward_ch(self):
        self.slider.setValue(self.slider.value()-1)
    
    def get_chval(self):
        chval = self.chval_box.text()
        if self.plane == 'xy':
            x = self._get_shared_world_x()
            y = self._get_shared_world_y()
            # If world coordinates are not yet set (e.g. at startup), calculate from shared pixels
            if x is None or y is None:
                x, y = self.format_pix.pix_to_wcs(self.wcs, self._get_shared_xpix(), self._get_shared_ypix(), self.plane)

            try:
                z = float(chval)

                if self.data.ndim == 3:
                    zpix = int(round(float(self.converter.world_to_pix(x, y, z)[2])))
                elif self.data.ndim == 4:
                    zpix = int(round(float(self.converter.world_to_pix(x, y, z, 0)[2])))

                chmin, chmax = 0, self.header["NAXIS3"]-1
                if chmin > zpix or chmax < zpix:
                    if chmin > zpix: zpix = chmin
                    if chmax < zpix: zpix = chmax
                    print("\n\nValueError: value out of range")
                self.slider.setValue(zpix)
            except:
                QMessageBox.warning(self, 'Invalid Input', 'Invalid value provided.')
                return

        elif self.plane == 'xz':
            x = self._get_shared_world_x()
            z = self._get_shared_world_z()
            # Fallback if world coords are None
            if x is None or z is None:
                x, z = self.format_pix.pix_to_wcs(self.wcs, self._get_shared_xpix(), self._get_shared_zpix(), self.plane)

            try:
                y = str(chval).strip()
                if not y:
                    raise ValueError("empty input")
                if self.data.ndim == 3:
                    ypix = int(round(float(self.converter.world_to_pix(x, y, z)[1])))
                elif self.data.ndim == 4:
                    ypix = int(round(float(self.converter.world_to_pix(x, y, z, 0)[1])))
                chmin, chmax = 0, self.header["NAXIS2"]-1
                if chmin > ypix or chmax < ypix:
                    if chmin > ypix: ypix = chmin
                    if chmax < ypix: ypix = chmax
                    print("\n\nValueError: value out of range")
                self.slider.setValue(ypix)
            except:
                QMessageBox.warning(self, 'Invalid Input', 'Invalid value provided.')
                return

        elif self.plane == 'zy':
            z = self._get_shared_world_z()
            y = self._get_shared_world_y()
            # Fallback if world coords are None
            if z is None or y is None:
                z, y = self.format_pix.pix_to_wcs(self.wcs, self._get_shared_zpix(), self._get_shared_ypix(), self.plane)

            try:
                x = str(chval).strip()
                if not x:
                    raise ValueError("empty input")
                if self.data.ndim == 3:
                    xpix = int(round(float(self.converter.world_to_pix(x, y, z)[0])))
                elif self.data.ndim == 4:
                    xpix = int(round(float(self.converter.world_to_pix(x, y, z, 0)[0])))
                chmin, chmax = 0, self.header["NAXIS1"]-1
                if chmin > xpix or chmax < xpix:
                    if chmin > xpix: xpix = chmin
                    if chmax < xpix: xpix = chmax
                    print("\n\nValueError: value out of range")
                self.slider.setValue(xpix)
            except:
                QMessageBox.warning(self, 'Invalid Input', 'Invalid value provided.')
                return
        
            
    def clicked_slider(self):
        # Update clicked state via coordinator
        self._set_clicked(self.plane, True)
        self.state.clicked = True

    
    def scroll_slider_mpl(self, event):
        # Use local slider reference
        if not hasattr(self, 'slider') or self.slider is None:
            return

        wheel_direction = -1 if getattr(self, 'invert_wheel_direction', False) else 1

        if self.plane == 'xy':
            k = self.slider.value()
            k_max = self.header['NAXIS3']-1
            self.scroll_accumulation += event.step * self.scrollspeed * wheel_direction
            if abs(self.scroll_accumulation) >= 1:
                step = int(self.scroll_accumulation)
                k = k - step
                k = max(0, min(k, k_max))
                self._set_clicked('xy', False)
                self.state.clicked = False
                self.slider.setValue(k)
                self.scroll_accumulation -= step

        elif self.plane == 'xz':
            k = self.slider.value()
            k_max = self.header['NAXIS2']-1
            self.scroll_accumulation += event.step * self.scrollspeed * wheel_direction
            if abs(self.scroll_accumulation) >= 1:
                step = int(self.scroll_accumulation)
                k = k - step
                k = max(0, min(k, k_max))
                self._set_clicked('xz', False)
                self.state.clicked = False
                self.slider.setValue(k)
                self.scroll_accumulation -= step

        elif self.plane == 'zy':
            k = self.slider.value()
            k_max = self.header['NAXIS1']-1
            self.scroll_accumulation += event.step * self.scrollspeed * wheel_direction
            if abs(self.scroll_accumulation) >= 1:
                step = int(self.scroll_accumulation)
                k = k - step
                k = max(0, min(k, k_max))
                self._set_clicked('zy', False)
                self.state.clicked = False
                self.slider.setValue(k)
                self.scroll_accumulation -= step
    
    def scroll_slider(self, event):
        k = 0
        if self.plane == 'xy':
            self._set_clicked('xy', False)
            self.state.clicked = False
            k = self.slider.value()
            self._set_shared_zpix(k)
        elif self.plane == 'xz':
            self._set_clicked('xz', False)
            self.state.clicked = False
            k = self.slider.value()
            self._set_shared_ypix(k)
        elif self.plane == 'zy':
            self._set_clicked('zy', False)
            self.state.clicked = False
            k = self.slider.value()
            self._set_shared_xpix(k)
        
        # Trigger UI update (Image, Label, Crosshairs)
        self.update_channel(self.plane, k)
        self.current_value_label.setText(str(k+1))  # current slider (ch) value

        # Emit signal for decoupled tools
        self.position_updated.emit(self._get_shared_xpix(), self._get_shared_ypix(), self._get_shared_zpix()) 

            
    def on_click(self, event):
        if self._handle_colorbar_double_click(event):
            return
        if self._begin_colorbar_drag(event):
            return
        if event.dblclick:
            if self.toolbar.mode == 'zoom rect':
                self.toolbar.zoom()
                return
            elif self.toolbar.mode == 'pan/zoom':
                self.toolbar.pan()
                return

        if self.toolbar.mode in ('zoom rect', 'pan/zoom'):
            return

        if getattr(self, 'marker_mode_enabled', False) and self.toolbar.mode == '':
            overlay_ax = getattr(self.displaymap, 'overlay_ax', None)
            if event.inaxes is overlay_ax:
                if hasattr(self, 'marker_manager') and self.marker_manager is not None:
                    self.canvas.setFocus()
                    self.marker_manager.set_active_plane(self.plane)
                    self.marker_manager.handle_press(event)
                    self.redraw_main_overlay_and_blit()
                return

        if hasattr(self, 'region_mode_enabled') and self.region_mode_enabled and self.toolbar.mode == '':
            if self.plane == 'xy' and event.inaxes is self.displaymap.overlay_ax:
                # Give focus to the canvas when clicked in region mode.
                # This ensures it can receive key press events for deletion.
                self.canvas.setFocus()
                self.region_manager.handle_press(event)
                self.redraw_main_overlay_and_blit()
                return

        if event.button == 1 and event.inaxes is self.displaymap.overlay_ax:
            if self.plane == 'xy' and self._pv_arrow_active():
                return
            self.region_manager.draw_regions_for_blit()

            x, y = self.ax.transData.inverted().transform((event.x, event.y))
            xstr, ystr = self.format_pix.convert(self.plane, x, y)
            if self.data.ndim > 2:
                if self.plane == 'xy': 
                    self._set_clicked('xy', True)
                    # Note: Do not reset zpix from slider here. Preserve current slice.
                    zpix = self._get_shared_zpix()
                    world_z = self.format_pix.convert_chpix_to_world(self.plane, x, y, zpix)
                    world_z_str = self.format_pix.convert_chval_to_world_str(self.plane, world_z)
                    i, j, k = int(round(x)), int(round(y)), zpix
                    self._update_shared_world_xyz(self._get_shared_world_x(), self._get_shared_world_y(), world_z)
                    self._update_shared_world_xyz_str(xstr, ystr, world_z_str)
                    try: intensity = self.cube[k,j,i]
                    except: return
                    
                    
                elif self.plane == 'xz':
                    self._set_clicked('xz', True)
                    # Note: Do not reset ypix from slider here. Preserve current slice.
                    ypix = self._get_shared_ypix()
                    world_y = self.format_pix.convert_chpix_to_world(self.plane, x, ypix, y)
                    world_y_str = self.format_pix.convert_chval_to_world_str(self.plane, world_y)
                    i, j, k = int(round(x)), ypix, int(round(y))
                    self._update_shared_world_xyz(self._get_shared_world_x(), world_y, self._get_shared_world_z())
                    self._update_shared_world_xyz_str(xstr, world_y_str, ystr)
                    try: intensity = self.cube[k,j,i]
                    except: return
                    
                elif self.plane == 'zy':
                    self._set_clicked('zy', True)
                    # Note: Do not reset xpix from slider here. Preserve current slice.
                    xpix = self._get_shared_xpix()
                    world_x = self.format_pix.convert_chpix_to_world(self.plane, xpix, y, x)
                    world_x_str = self.format_pix.convert_chval_to_world_str(self.plane, world_x)                
                    i, j, k = xpix, int(round(y)), int(round(x))
                    self._update_shared_world_xyz(world_x, self._get_shared_world_y(), self._get_shared_world_z())
                    self._update_shared_world_xyz_str(world_x_str, ystr, xstr)
                    try: intensity = self.cube[k,j,i]
                    except: return
                print('\r Clicked at (%s, %s, %s)              \n Intensity = %s %s            \033[1A'  % (self._get_shared_world_x_str(), self._get_shared_world_y_str(), self._get_shared_world_z_str(), self._format_significant_digits(intensity, 4), self.bunit), end = '')
                
            elif self.data.ndim == 2: 
                i, j = int(round(x)), int(round(y))
                try: intensity =  self.data[j,i]
                except: return
                print('\r Clicked at (%s, %s)             \n%s %s          \033[1A'  % (self._get_shared_world_x_str(), self._get_shared_world_y_str(), self._format_significant_digits(intensity, 4), self.bunit), end = '')
                
            intensity_text = self._format_intensity_with_unit(intensity)
            self.update_clicked_pix(x, y)
            coord_text = self._format_cursor_pair_text(self.plane, xstr, ystr)
            self.label.setText(self._compose_click_label_text(coord_text, intensity_text))
            self._position_click_label()

        elif event.inaxes != self.ax:
            if event.dblclick:
                # Hide all position labels
                for plane in ['xy', 'xz', 'zy']:
                    plabel = self._get_plane_plabel(plane)
                    if plabel:
                        plabel.hide()
                    chlabel = self._get_plane_chlabel(plane)
                    if chlabel:
                        chlabel.set_visible(False)
                    hline = self._get_plane_hline(plane)
                    if hline:
                        hline.set_visible(False)
                    vline = self._get_plane_vline(plane)
                    if vline:
                        vline.set_visible(False)
                    cpoint = self._get_plane_cpoint(plane)
                    if cpoint:
                        cpoint.set_visible(False)
                    canvas = self._get_plane_canvas(plane)
                    if canvas:
                        canvas.draw_idle()


    def on_release(self, event):
        """Handles mouse release events, delegating to the RegionManager if in region mode."""
        if self._end_colorbar_drag(event):
            return
        # This is primarily for finalizing region drawing.
        if hasattr(self, 'region_mode_enabled') and self.region_mode_enabled:
            if self.plane == 'xy':
                self.region_manager.handle_release(event)
        if getattr(self, 'marker_mode_enabled', False) and hasattr(self, 'marker_manager') and self.marker_manager is not None:
            self.marker_manager.handle_release(event)
        self._last_drag_update_key = None
        pending = getattr(self, '_pending_drag_coords', None)
        if pending is not None:
            try:
                x, y = pending
            except Exception:
                x = y = None
            self._pending_drag_coords = None
            if x is not None and y is not None:
                self.update_clicked_pix(x, y, update_slices=True)
                # Restore the intensity line that update_clicked_pix overwrites.
                try:
                    xstr, ystr = self.format_pix.convert(self.plane, x, y)
                    if self.data.ndim == 2:
                        i, j = int(round(x)), int(round(y))
                        intensity = self.data[j, i]
                    else:
                        if self.plane == 'xy':
                            i, j, k = int(round(x)), int(round(y)), self._get_shared_zpix()
                        elif self.plane == 'xz':
                            i, j, k = int(round(x)), self._get_shared_ypix(), int(round(y))
                        else:
                            i, j, k = self._get_shared_xpix(), int(round(y)), int(round(x))
                        intensity = self.cube[k, j, i]
                    intensity_text = self._format_intensity_with_unit(intensity)
                    coord_text = self._format_cursor_pair_text(self.plane, xstr, ystr)
                    self.label.setText(self._compose_click_label_text(coord_text, intensity_text))
                    if not self.label.isVisible():
                        self.label.setVisible(True)
                except Exception:
                    pass

    def _position_click_label(self):
        label = getattr(self, "label", None)
        canvas = getattr(self, "canvas", None)
        if label is None or canvas is None:
            return
        x, y, width, height = compute_click_label_geometry(
            canvas.width(),
            canvas.height(),
            pos_x=self.poslabel_x,
            pos_y=self.poslabel_y,
            requested_width=self.poslabel_w,
            requested_height=self.poslabel_h,
        )
        if width <= 0 or height <= 0:
            return
        label.setGeometry(x, y, width, height)

    @staticmethod
    def _is_left_drag_motion(event) -> bool:
        if getattr(event, 'button', None) == 1:
            return True
        buttons = getattr(event, 'buttons', None)
        if buttons is None:
            return False
        try:
            return 1 in buttons
        except Exception:
            try:
                return int(buttons) == 1
            except Exception:
                return False

    def _apply_drag_update(self, x: float, y: float):
        # Keep cursor overlays synced with pointer movement.
        show_cursor_overlay = self._is_cursor_overlay_enabled()
        cursor_redrawn = False
        bg = self._get_plane_background(self.plane)
        if bg is None or not getattr(self, '_background_initialized', True):
            bg = self._refresh_overlay_background(self.plane)
        if bg is not None:
            self._update_plane_cursor(self.plane, x=x, y=y)
            self.vline.set_xdata([x])
            self.hline.set_ydata([y])
            self._set_crosshair_point_for_plane(self.plane, x=x, y=y)
            self._set_crosshair_visibility_for_plane(self.plane, show_cursor_overlay)
            chlabel = self._get_plane_chlabel(self.plane)
            if chlabel:
                chlabel.set_visible(True)
            if self.plane == 'xy':
                self.redraw_main_overlay_and_blit(lightweight=True)
            else:
                self.redraw_overlay_for_plane(self.plane, lightweight=True)
            cursor_redrawn = True

        if self.data.ndim == 2:
            i, j, k = int(round(x)), int(round(y)), 0
            drag_key = (i, j)
        elif self.plane == 'xy':
            i, j, k = int(round(x)), int(round(y)), int(self._get_shared_zpix())
            drag_key = (i, j, k)
        elif self.plane == 'xz':
            i, j, k = int(round(x)), int(self._get_shared_ypix()), int(round(y))
            drag_key = (i, j, k)
        else:
            i, j, k = int(self._get_shared_xpix()), int(round(y)), int(round(x))
            drag_key = (i, j, k)
        same_drag_key = drag_key == getattr(self, '_last_drag_update_key', None)

        if self.plane == 'xy':
            xstr, ystr = self.format_pix.convert(self.plane, x, y)
            self._update_shared_world_xyz_str(xstr, ystr, self._get_shared_world_z_str())
        elif self.plane == 'xz':
            xstr, ystr = self.format_pix.convert(self.plane, x, y)
            self._update_shared_world_xyz_str(xstr, self._get_shared_world_y_str(), ystr)
        else:
            xstr, ystr = self.format_pix.convert(self.plane, x, y)
            self._update_shared_world_xyz_str(self._get_shared_world_x_str(), ystr, xstr)

        if self.data.ndim == 2:
            try:
                intensity = self.data[j, i]
            except Exception:
                return
        else:
            try:
                intensity = self.cube[k, j, i]
            except Exception:
                return
        self._last_drag_update_key = drag_key

        self.update_clicked_pix(
            x,
            y,
            # Keep crosshair motion continuous; skip expensive slice swaps when index is unchanged.
            update_slices=not same_drag_key,
            # Keep drag smooth: use fast blit during drag, full update on release.
            fast_blit=True,
            cursor_already_redrawn=cursor_redrawn,
        )
        intensity_text = self._format_intensity_with_unit(intensity)
        coord_text = self._format_cursor_pair_text(self.plane, xstr, ystr)
        self.label.setText(self._compose_click_label_text(coord_text, intensity_text))

    def cursor_position(self, event):
        if self._drag_colorbar(event):
            return
        marker_mgr = getattr(self, 'marker_manager', None)
        if getattr(self, 'marker_mode_enabled', False):
            if marker_mgr is not None:
                if marker_mgr.is_dragging():
                    marker_mgr.handle_motion(event)
                else:
                    marker_mgr.handle_hover(event)
            return

        if marker_mgr is not None and not getattr(self, 'region_mode_enabled', False):
            marker_mgr.handle_hover(event)

        # If region mode is active and drawing, delegate to the region manager.
        if hasattr(self, 'region_mode_enabled') and self.region_mode_enabled:
            if self.plane == 'xy':
                if (self.region_manager.is_drawing or self.region_manager.is_dragging or
                        self.region_manager.is_resizing or self.region_manager.is_rotating):
                    self.region_manager.handle_motion(event)
                else:
                    self.region_manager.update_hover_cursor(event)
                return

        current_time = time.time()
        drag_heavy_interval = max(0.01, float(getattr(self, '_drag_slice_interval_sec', 0.0)))
        throttled = (current_time - self.last_update_time) < drag_heavy_interval
        if not throttled:
            self.last_update_time = current_time

        if self.toolbar.mode == 'zoom rect' or self.toolbar.mode =='pan/zoom':
            return
        if event.inaxes is not self.ax:
            return

        xdata = getattr(event, 'xdata', None)
        ydata = getattr(event, 'ydata', None)
        if xdata is not None and ydata is not None:
            x, y = float(xdata), float(ydata)
        else:
            ex = getattr(event, 'x', None)
            ey = getattr(event, 'y', None)
            if ex is None or ey is None:
                return
            try:
                x, y = self.ax.transData.inverted().transform((ex, ey))
            except Exception:
                return

        if self._is_left_drag_motion(event):
            if self.plane == 'xy' and self._pv_arrow_active():
                return
            # Keep latest drag position for an accurate final release update.
            self._pending_drag_coords = (x, y)
            if throttled:
                return
            self._apply_drag_update(x, y)

            
    def update_clicked_pix(
        self,
        x,
        y,
        update_slices: bool = True,
        fast_blit: bool = False,
        force_slice_refresh: bool = False,
        cursor_already_redrawn: bool = False,
    ):
        """
        Update crosshairs, slice images, and labels when a pixel is clicked/hovered.
        Simplified to match the distribution version's working pattern.
        """
        perf_token = self._perf_start(f"{getattr(self, 'plane', '?')} update_clicked_pix fast={fast_blit} slices={update_slices}")
        if getattr(self, 'marker_mode_enabled', False):
            self._perf_end(perf_token)
            return
        if not fast_blit:
            bg = self._get_plane_background(self.plane)
            if bg is None or not getattr(self, '_background_initialized', True):
                self._rebuild_overlay_background(self.plane)
            # Fallback: if background is still missing, force a one-time full draw.
            if self._get_plane_background(self.plane) is None and getattr(self, 'canvas', None) is not None:
                try:
                    self.canvas.draw()
                    if self.state is not None:
                        fresh_bg = self.state.copy_overlay_background()
                        if fresh_bg is not None:
                            self._background = fresh_bg
                            self.state.update_background(fresh_bg)
                            self._background_initialized = True
                except Exception:
                    pass

        shared_updated = False
        prev_xpix = int(self._get_shared_xpix())
        prev_ypix = int(self._get_shared_ypix())
        prev_zpix = int(self._get_shared_zpix())
        self.xpix = self._get_shared_xpix()
        self.ypix = self._get_shared_ypix()
        self.zpix = self._get_shared_zpix()
        self.clicked_coords = (x, y)
        i, j = int(round(x)), int(round(y))
        show_cursor_overlay = self._is_cursor_overlay_enabled()
        if not cursor_already_redrawn:
            self._update_plane_cursor(self.plane, x=x, y=y)

        # Access subwindow visibility
        subwindow1 = getattr(self.SubWindow, 'subwindow1', None)
        subwindow2 = getattr(self.SubWindow, 'subwindow2', None)
        sub1_visible = subwindow1 is not None and not subwindow1.isHidden()
        sub2_visible = subwindow2 is not None and not subwindow2.isHidden()
        defer_orthogonal_refresh = self.is_large_data_mode() and bool(fast_blit)

        if self.plane == 'xy':
            self.xpix = x
            self.ypix = y
            x_changed = i != prev_xpix
            y_changed = j != prev_ypix

            # Update xy plane crosshairs early for immediate feedback
            self.vline.set_xdata([x])
            self.hline.set_ydata([y])
            self._set_crosshair_point_for_plane('xy', x=x, y=y)
            self._set_crosshair_visibility_for_plane('xy', show_cursor_overlay)

            xy_chlabel = self._get_plane_chlabel('xy')
            if xy_chlabel:
                xy_chlabel.set_text(self._format_channel_label_text(self._get_shared_world_z_str(), "xy"))
                xy_chlabel.set_visible(True)

            if not cursor_already_redrawn:
                self.redraw_main_overlay_and_blit()

            if self.wcs.naxis > 2 and 'NAXIS3' in self.header and not defer_orthogonal_refresh:
                # Get crosshair lines for other planes
                xz_vline = self._get_plane_vline('xz')
                xz_hline = self._get_plane_hline('xz')
                zy_vline = self._get_plane_vline('zy')
                zy_hline = self._get_plane_hline('zy')

                # Update crosshair positions
                if xz_vline:
                    xz_vline.set_xdata([x])
                    self._update_plane_cursor('xz', x=x)
                if zy_hline:
                    zy_hline.set_ydata([y])
                    self._update_plane_cursor('zy', y=y)

                # Make crosshairs visible
                self._set_crosshair_visibility_for_plane('xz', show_cursor_overlay)
                self._set_crosshair_visibility_for_plane('zy', show_cursor_overlay)

                # Update XZ plane (subwindow1)
                if sub1_visible:
                    xz_refreshed = False
                    xz_requires_refresh = y_changed or force_slice_refresh
                    xz_chlabel = self._get_plane_chlabel('xz')
                    if xz_chlabel:
                        xz_chlabel.set_visible(True)

                    xz_im = self._get_plane_im('xz')
                    xz_slider = self._get_plane_slider('xz')
                    xz_canvas = self._get_plane_canvas('xz')
                    xz_overlay = self._get_plane_overlay_ax('xz')

                    if update_slices:
                        if 0 <= j < self.cube.shape[1]:
                            if xz_requires_refresh:
                                if xz_im:
                                    self._set_plane_image_data('xz', self.cube[:, j, :], cache_index=j)
                                if self._get_clicked('xy') and xz_slider:
                                    self._sync_channel_controls(subwindow1, j)
                                self._set_shared_ypix(j)
                        else:
                            if xz_requires_refresh and xz_im:
                                self._set_plane_image_data('xz', self._blank_plane('xz'))
                    if update_slices and subwindow1 and (xz_requires_refresh or self._get_plane_background('xz') is None):
                        xz_refreshed = self._update_slice_image(subwindow1, fast_blit=fast_blit)
                    
                    xz_bg = self._refresh_overlay_background('xz') if (update_slices and not fast_blit and not xz_refreshed) else self._get_plane_background('xz')
                    # Blit xz plane overlay (crosshairs, label)
                    if xz_canvas and xz_overlay:
                        # Only restore background if we didn't just fast-blit the image
                        if not (fast_blit and (xz_requires_refresh or self._get_plane_background('xz') is None)):
                            if xz_bg:
                                xz_canvas.restore_region(xz_bg)
                        xz_cpoint = self._get_plane_cpoint('xz')
                        if xz_vline:
                            xz_overlay.draw_artist(xz_vline)
                        if xz_hline:
                            xz_overlay.draw_artist(xz_hline)
                        if xz_cpoint is not None and xz_cpoint.get_visible():
                            xz_overlay.draw_artist(xz_cpoint)
                        if xz_chlabel:
                            xz_overlay.draw_artist(xz_chlabel)
                        marker_mgr = getattr(self, 'marker_manager', None)
                        if marker_mgr is not None:
                            marker_mgr.draw_markers_for_blit('xz')
                        xz_canvas.blit(xz_overlay.bbox)
                else:
                    if update_slices and (y_changed or force_slice_refresh) and 0 <= j < self.cube.shape[1]:
                        self._set_shared_ypix(j)

                # Update ZY plane (subwindow2)
                if sub2_visible:
                    zy_refreshed = False
                    zy_requires_refresh = x_changed or force_slice_refresh
                    zy_chlabel = self._get_plane_chlabel('zy')
                    if zy_chlabel:
                        zy_chlabel.set_visible(True)

                    zy_im = self._get_plane_im('zy')
                    zy_slider = self._get_plane_slider('zy')
                    zy_canvas = self._get_plane_canvas('zy')
                    zy_overlay = self._get_plane_overlay_ax('zy')

                    if update_slices:
                        if 0 <= i < self.cube.shape[2]:
                            if zy_requires_refresh:
                                if zy_im:
                                    self._set_plane_image_data('zy', self.cube[:, :, i].T, cache_index=i)
                                if self._get_clicked('xy') and zy_slider:
                                    self._sync_channel_controls(subwindow2, i)
                                self._set_shared_xpix(i)
                        else:
                            if zy_requires_refresh and zy_im:
                                self._set_plane_image_data('zy', self._blank_plane('zy'))
                    if update_slices and subwindow2 and (zy_requires_refresh or self._get_plane_background('zy') is None):
                        zy_refreshed = self._update_slice_image(subwindow2, fast_blit=fast_blit)
                    
                    zy_bg = self._refresh_overlay_background('zy') if (update_slices and not fast_blit and not zy_refreshed) else self._get_plane_background('zy')
                    # Blit zy plane overlay (crosshairs, label)
                    if zy_canvas and zy_overlay:
                        # Only restore background if we didn't just fast-blit the image
                        if not (fast_blit and (zy_requires_refresh or self._get_plane_background('zy') is None)):
                            if zy_bg:
                                zy_canvas.restore_region(zy_bg)
                        zy_cpoint = self._get_plane_cpoint('zy')
                        if zy_hline:
                            zy_overlay.draw_artist(zy_hline)
                        if zy_vline:
                            zy_overlay.draw_artist(zy_vline)
                        if zy_cpoint is not None and zy_cpoint.get_visible():
                            zy_overlay.draw_artist(zy_cpoint)
                        if zy_chlabel:
                            zy_overlay.draw_artist(zy_chlabel)
                        marker_mgr = getattr(self, 'marker_manager', None)
                        if marker_mgr is not None:
                            marker_mgr.draw_markers_for_blit('zy')
                        zy_canvas.blit(zy_overlay.bbox)


        elif self.plane == 'xz' and self.header['NAXIS3'] > 1:
            self.xpix = x
            self.zpix = y
            x_changed = i != prev_xpix
            z_changed = j != prev_zpix
            # Update shared coords early to avoid draw-event lag during cross-plane updates.
            if update_slices:
                self._update_shared_pix(i, int(self.ypix), j)
                shared_updated = True

            # Update xz plane crosshairs early for immediate feedback
            self.vline.set_xdata([x])
            self.hline.set_ydata([y])
            self._set_crosshair_point_for_plane('xz', x=x, y=y)
            self._set_crosshair_visibility_for_plane('xz', show_cursor_overlay)

            xz_chlabel = self._get_plane_chlabel('xz')
            if xz_chlabel:
                xz_chlabel.set_visible(True)

            if not cursor_already_redrawn:
                xz_bg = self._get_plane_background('xz')
                if xz_bg is None and update_slices and not fast_blit:
                    xz_bg = self._refresh_overlay_background('xz')
                if self.canvas and xz_bg and self.overlay_ax:
                    self.canvas.restore_region(xz_bg)
                    self.overlay_ax.draw_artist(self.vline)
                    self.overlay_ax.draw_artist(self.hline)
                    if getattr(self, "cpoint", None) is not None and self.cpoint.get_visible():
                        self.overlay_ax.draw_artist(self.cpoint)
                    if xz_chlabel:
                        self.overlay_ax.draw_artist(xz_chlabel)
                    marker_mgr = getattr(self, 'marker_manager', None)
                    if marker_mgr is not None:
                        marker_mgr.draw_markers_for_blit('xz')
                    self.canvas.blit(self.overlay_ax.bbox)
                    self._blit_colorbar_foreground_for_state(self.state, force=False)

            if self.wcs.naxis > 2 and not defer_orthogonal_refresh:
                # Get elements for other planes
                xy_vline = self._get_plane_vline('xy')
                xy_hline = self._get_plane_hline('xy')
                zy_vline = self._get_plane_vline('zy')
                zy_hline = self._get_plane_hline('zy')
                xy_chlabel = self._get_plane_chlabel('xy')
                zy_chlabel = self._get_plane_chlabel('zy')

                # Update crosshair positions
                if xy_vline:
                    xy_vline.set_xdata([x])
                    self._update_plane_cursor('xy', x=x)
                if zy_vline:
                    zy_vline.set_xdata([y])
                    self._update_plane_cursor('zy', x=y)

                # Make crosshairs visible
                self._set_crosshair_visibility_for_plane('zy', show_cursor_overlay)
                self._set_crosshair_visibility_for_plane('xy', show_cursor_overlay)
                if xy_chlabel:
                    xy_chlabel.set_visible(True)
                if zy_chlabel:
                    zy_chlabel.set_visible(True)

                # Update XY plane (main window) with new Z slice
                xy_im = self._get_plane_im('xy')
                xy_slider = self._get_plane_slider('xy')

                if update_slices:
                    if 0 <= j < self.cube.shape[0]:
                        if z_changed:
                            if xy_im:
                                self._set_plane_image_data('xy', self.cube[j], cache_index=j)
                            if self._get_clicked('xz') and xy_slider:
                                self._sync_channel_controls(FITSViewer.main_window, j)
                            self._set_shared_zpix(j)
                    else:
                        if z_changed and xy_im:
                            self._set_plane_image_data('xy', self._blank_plane('xy'))

                # Update ZY plane (subwindow2) with new X slice
                if sub2_visible:
                    zy_refreshed = False
                    zy_im = self._get_plane_im('zy')
                    zy_slider = self._get_plane_slider('zy')
                    zy_canvas = self._get_plane_canvas('zy')
                    zy_overlay = self._get_plane_overlay_ax('zy')

                    if update_slices:
                        if 0 <= i < self.cube.shape[2]:
                            if x_changed:
                                if zy_im:
                                    self._set_plane_image_data('zy', self.cube[:, :, i].T, cache_index=i)
                            if self._get_clicked('xz') and zy_slider:
                                self._sync_channel_controls(subwindow2, i)
                            self._set_shared_xpix(i)
                        else:
                            if x_changed and zy_im:
                                self._set_plane_image_data('zy', self._blank_plane('zy'))
                        if update_slices and subwindow2 and (x_changed or self._get_plane_background('zy') is None):
                            zy_refreshed = self._update_slice_image(subwindow2, fast_blit=fast_blit)
                    
                    zy_bg = self._refresh_overlay_background('zy') if (update_slices and not fast_blit and not zy_refreshed) else self._get_plane_background('zy')
                    # Blit zy plane overlay (crosshairs, label)
                    if zy_canvas and zy_overlay:
                        # Only restore background if we didn't just fast-blit the image
                        if not (fast_blit and (x_changed or self._get_plane_background('zy') is None)):
                            if zy_bg:
                                zy_canvas.restore_region(zy_bg)
                        zy_cpoint = self._get_plane_cpoint('zy')
                        if zy_vline:
                            zy_overlay.draw_artist(zy_vline)
                        if zy_hline:
                            zy_overlay.draw_artist(zy_hline)
                        if zy_cpoint is not None and zy_cpoint.get_visible():
                            zy_overlay.draw_artist(zy_cpoint)
                        if zy_chlabel:
                            zy_overlay.draw_artist(zy_chlabel)
                        marker_mgr = getattr(self, 'marker_manager', None)
                        if marker_mgr is not None:
                            marker_mgr.draw_markers_for_blit('zy')
                        zy_canvas.blit(zy_overlay.bbox)

                # Redraw main window overlay
                main = FITSViewer.main_window
                if main:
                    if update_slices and z_changed:
                        main_updated = self._update_slice_image(main, fast_blit=fast_blit)
                        if not main_updated and not fast_blit and not main._contours_active():
                            main.refresh_display_after_contour_update(main._contour_layer_id)
                        main.redraw_main_overlay_and_blit()
                    elif fast_blit and update_slices:
                        self._update_slice_image(main, fast_blit=True)
                        main.redraw_main_overlay_and_blit()
                    else:
                        main.redraw_main_overlay_and_blit()
                    if not fast_blit and update_slices and z_changed:
                        if hasattr(main, 'control_panel') and main.control_panel and main.control_panel.pvd_panel:
                            main.control_panel.pvd_panel.update_cursor(int(self._get_shared_zpix()))


        elif self.plane == 'zy' and self.header['NAXIS3'] > 1:
            self.zpix = x
            self.ypix = y
            y_changed = j != prev_ypix
            z_changed = i != prev_zpix
            # Update shared coords early to avoid draw-event lag during cross-plane updates.
            if update_slices:
                self._update_shared_pix(int(self.xpix), int(round(y)), int(round(x)))
                shared_updated = True

            # Update zy plane crosshairs early for immediate feedback
            self.vline.set_xdata([x])
            self.hline.set_ydata([y])
            self._set_crosshair_point_for_plane('zy', x=x, y=y)
            self._set_crosshair_visibility_for_plane('zy', show_cursor_overlay)

            zy_chlabel = self._get_plane_chlabel('zy')
            if zy_chlabel:
                zy_chlabel.set_visible(True)

            if not cursor_already_redrawn:
                zy_bg = self._get_plane_background('zy')
                if zy_bg is None and update_slices and not fast_blit:
                    zy_bg = self._refresh_overlay_background('zy')
                if self.canvas and zy_bg and self.overlay_ax:
                    self.canvas.restore_region(zy_bg)
                    self.overlay_ax.draw_artist(self.vline)
                    self.overlay_ax.draw_artist(self.hline)
                    if getattr(self, "cpoint", None) is not None and self.cpoint.get_visible():
                        self.overlay_ax.draw_artist(self.cpoint)
                    if zy_chlabel:
                        self.overlay_ax.draw_artist(zy_chlabel)
                    marker_mgr = getattr(self, 'marker_manager', None)
                    if marker_mgr is not None:
                        marker_mgr.draw_markers_for_blit('zy')
                    self.canvas.blit(self.overlay_ax.bbox)
                    self._blit_colorbar_foreground_for_state(self.state, force=False)

            if self.wcs.naxis > 2 and not defer_orthogonal_refresh:
                # Get elements for other planes
                xy_vline = self._get_plane_vline('xy')
                xy_hline = self._get_plane_hline('xy')
                xz_vline = self._get_plane_vline('xz')
                xz_hline = self._get_plane_hline('xz')
                xy_chlabel = self._get_plane_chlabel('xy')
                xz_chlabel = self._get_plane_chlabel('xz')

                # Update crosshair positions
                if xz_hline:
                    xz_hline.set_ydata([x])
                    self._update_plane_cursor('xz', y=x)
                if xy_hline:
                    xy_hline.set_ydata([y])
                    self._update_plane_cursor('xy', y=y)

                # Make crosshairs visible
                self._set_crosshair_visibility_for_plane('xz', show_cursor_overlay)
                self._set_crosshair_visibility_for_plane('xy', show_cursor_overlay)
                if xz_chlabel:
                    xz_chlabel.set_visible(True)
                if xy_chlabel:
                    xy_chlabel.set_visible(True)

                # Update XY plane (main window) with new Z slice
                xy_im = self._get_plane_im('xy')
                xy_slider = self._get_plane_slider('xy')

                if update_slices:
                    if 0 <= i < self.cube.shape[0]:
                        if z_changed:
                            if xy_im:
                                self._set_plane_image_data('xy', self.cube[i], cache_index=i)
                            if self._get_clicked('zy') and xy_slider:
                                self._sync_channel_controls(FITSViewer.main_window, i)
                            self._set_shared_zpix(i)
                    else:
                        if z_changed and xy_im:
                            self._set_plane_image_data('xy', self._blank_plane('xy'))

                # Update XZ plane (subwindow1) with new Y slice
                if sub1_visible:
                    xz_refreshed = False
                    xz_im = self._get_plane_im('xz')
                    xz_slider = self._get_plane_slider('xz')
                    xz_canvas = self._get_plane_canvas('xz')
                    xz_overlay = self._get_plane_overlay_ax('xz')

                    if update_slices:
                        if 0 <= j < self.cube.shape[1]:
                            if y_changed:
                                if xz_im:
                                    self._set_plane_image_data('xz', self.cube[:, j, :], cache_index=j)
                                if self._get_clicked('zy') and xz_slider:
                                    self._sync_channel_controls(subwindow1, j)
                                self._set_shared_ypix(j)
                        else:
                            if y_changed and xz_im:
                                self._set_plane_image_data('xz', self._blank_plane('xz'))
                        if update_slices and subwindow1 and (y_changed or self._get_plane_background('xz') is None):
                            xz_refreshed = self._update_slice_image(subwindow1, fast_blit=fast_blit)
                    # Blit xz plane (correct order: restore -> draw image -> draw overlays -> blit)
                    xz_bg = self._refresh_overlay_background('xz') if (update_slices and not fast_blit and not xz_refreshed) else self._get_plane_background('xz')
                    xz_ax = self._get_plane_ax('xz')
                    if xz_canvas and xz_overlay:
                        skip_restore = fast_blit and (y_changed or self._get_plane_background('xz') is None)
                        if not skip_restore and xz_bg:
                            xz_canvas.restore_region(xz_bg)

                        # During fast drag updates, _update_slice_image() has already rendered
                        # image/contours/axis for xz when needed; only draw overlays here.
                        if not fast_blit and xz_ax and xz_im:
                            xz_ax.draw_artist(xz_im)

                            # Draw contour artists after image
                            layer_id = getattr(subwindow1, '_contour_layer_id', None)
                            if layer_id:
                                try:
                                    manager = ContourManager.instance()
                                    layer = manager._layers.get(layer_id)
                                    if layer:
                                        for artist in layer.get_generated_artists():
                                            if artist and getattr(artist, 'axes', None) is not None:
                                                artist.axes.draw_artist(artist)
                                        for artist in layer.get_overlay_artists():
                                            if artist and getattr(artist, 'axes', None) is not None:
                                                artist.axes.draw_artist(artist)
                                except Exception:
                                    pass
                            self._draw_axis_foreground(self.get_viewer_state('xz'))

                        if xz_hline:
                            xz_overlay.draw_artist(xz_hline)
                        if xz_vline:
                            xz_overlay.draw_artist(xz_vline)
                        xz_cpoint = self._get_plane_cpoint('xz')
                        if xz_cpoint is not None and xz_cpoint.get_visible():
                            xz_overlay.draw_artist(xz_cpoint)
                        if xz_chlabel:
                            xz_overlay.draw_artist(xz_chlabel)
                        marker_mgr = getattr(self, 'marker_manager', None)
                        if marker_mgr is not None:
                            marker_mgr.draw_markers_for_blit('xz')
                        xz_canvas.blit(xz_overlay.bbox)

                # Redraw main window overlay
                main = FITSViewer.main_window
                if main:
                    if update_slices and z_changed:
                        main_updated = self._update_slice_image(main, fast_blit=fast_blit)
                        if not main_updated and not fast_blit and not main._contours_active():
                            main.refresh_display_after_contour_update(main._contour_layer_id)
                        main.redraw_main_overlay_and_blit()
                    elif fast_blit and update_slices:
                        self._update_slice_image(main, fast_blit=True)
                        main.redraw_main_overlay_and_blit()
                    else:
                        main.redraw_main_overlay_and_blit()
                    if not fast_blit and update_slices and z_changed:
                        if hasattr(main, 'control_panel') and main.control_panel and main.control_panel.pvd_panel:
                            main.control_panel.pvd_panel.update_cursor(int(self._get_shared_zpix()))


        # Update shared pixel coordinates
        if update_slices:
            shared_x = int(round(float(self.xpix)))
            shared_y = int(round(float(self.ypix)))
            shared_z = int(round(float(self.zpix)))
            if not shared_updated:
                self._update_shared_pix(shared_x, shared_y, shared_z)
            self.position_updated.emit(shared_x, shared_y, shared_z)

        # Update position labels for all planes
        plabel_xy = self._get_plane_plabel('xy')
        plabel_xz = self._get_plane_plabel('xz')
        plabel_zy = self._get_plane_plabel('zy')
        xy_text = self._formatted_plane_cursor_text('xy')
        xz_text = self._formatted_plane_cursor_text('xz')
        zy_text = self._formatted_plane_cursor_text('zy')
        intensity_text = self._shared_intensity_text()

        if plabel_xy:
            coord_text = xy_text if xy_text is not None else self._format_cursor_pair_text('xy', self._get_shared_world_x_str(), self._get_shared_world_y_str())
            plabel_xy.setText(self._compose_click_label_text(coord_text, intensity_text))
            if not plabel_xy.isVisible():
                plabel_xy.setVisible(True)
        if plabel_xz:
            coord_text = xz_text if xz_text is not None else self._format_cursor_pair_text('xz', self._get_shared_world_x_str(), self._get_shared_world_z_str())
            plabel_xz.setText(self._compose_click_label_text(coord_text, intensity_text))
            if not plabel_xz.isVisible():
                plabel_xz.setVisible(True)
        if plabel_zy:
            coord_text = zy_text if zy_text is not None else self._format_cursor_pair_text('zy', self._get_shared_world_z_str(), self._get_shared_world_y_str())
            plabel_zy.setText(self._compose_click_label_text(coord_text, intensity_text))
            if not plabel_zy.isVisible():
                plabel_zy.setVisible(True)
        self._perf_end(perf_token)


    def update_channel(self, plane, k):
        if self.data is not None and self.data.ndim < 3:
            return
        show_cursor_overlay = self._is_cursor_overlay_enabled()
            
        # Get plane elements using helpers
        im_xy = self._get_plane_im('xy')
        im_xz = self._get_plane_im('xz')
        im_zy = self._get_plane_im('zy')
        marker_mgr = getattr(self, 'marker_manager', None)
        
        if plane == 'xy':
            k = max(0, min(k, self.data.shape[0 if self.data.ndim == 3 else 1] - 1))
            xpix = max(0, min(self._get_shared_xpix(), self.data.shape[-1] - 1))
            ypix = max(0, min(self._get_shared_ypix(), self.data.shape[-2] - 1))
            z = self.format_pix.convert_chpix_to_world(self.plane, xpix, ypix, k)
            z_str = self.format_pix.convert_chval_to_world_str(self.plane, z)
            
            xy_chlabel = self._get_plane_chlabel('xy')
            xy_chval_box = self._get_plane_chval_box('xy')
            if xy_chval_box:
                if xy_chlabel:
                    xy_chlabel.set_text(self._format_channel_label_text(z_str, "xy"))
                xy_chval_box.setText("%s" % z_str)
                xy_chval_box.setCursorPosition(0)
                if xy_chlabel and not xy_chlabel.get_visible():
                    xy_chlabel.set_visible(True)
            
            self._update_shared_pix(xpix, ypix, k)
            self._update_shared_world_xyz(self._get_shared_world_x(), self._get_shared_world_y(), z)
            self._update_shared_world_xyz_str(self._get_shared_world_x_str(), self._get_shared_world_y_str(), z_str)
            
            if self.data.ndim == 3:
                self._set_plane_image_data('xy', self.data[k], cache_index=k)
            elif self.data.ndim == 4:
                self._set_plane_image_data('xy', self.data[0, k], cache_index=k)
            # Image changed -> invalidate overlay background so blit won't restore stale pixels.
            self._invalidate_plane_background('xy')

            # Draw image and label (blit is handled by refresh_display_after_contour_update)
            # Redundant draw calls removed.

            if hasattr(self, 'control_panel') and self.control_panel.pvd_panel:
                if k >= 0:
                    self.control_panel.pvd_panel.update_cursor(k)

            self._refresh_after_channel_image_update('xy')
            self._schedule_large_data_prefetch('xy', k)

            xz_hline = self._get_plane_hline('xz')
            zy_vline = self._get_plane_vline('zy')
            if not self._get_clicked('zy') and xz_hline:
                xz_hline.set_ydata([k])
                self._update_plane_cursor('xz', y=k)
                self._set_crosshair_visibility_for_plane('xz', show_cursor_overlay)
            if not self._get_clicked('xz') and zy_vline:
                zy_vline.set_xdata([k])
                self._update_plane_cursor('zy', x=k)
                self._set_crosshair_visibility_for_plane('zy', show_cursor_overlay)

            if not self._get_clicked('xy'):
                xy_plabel = self._get_plane_plabel('xy')
                if self.SubWindow.subwindow1.isHidden() == False:
                    if xy_plabel and xy_plabel.isVisible():
                        xz_canvas = self._get_plane_canvas('xz')
                        xz_overlay = self._get_plane_overlay_ax('xz')
                        xz_bg = self._get_plane_background('xz')
                        xz_chlabel = self._get_plane_chlabel('xz')
                        xz_vline = self._get_plane_vline('xz')
                        if xz_canvas and xz_bg:
                            xz_canvas.restore_region(xz_bg)
                        if xz_overlay:
                            if xz_hline:
                                xz_overlay.draw_artist(xz_hline)
                            if xz_vline:
                                xz_overlay.draw_artist(xz_vline)
                            xz_cpoint = self._get_plane_cpoint('xz')
                            if xz_cpoint is not None and xz_cpoint.get_visible():
                                xz_overlay.draw_artist(xz_cpoint)
                            if xz_chlabel:
                                xz_overlay.draw_artist(xz_chlabel)
                        if marker_mgr is not None:
                            marker_mgr.draw_markers_for_blit('xz')
                        if xz_canvas and xz_overlay:
                            xz_canvas.blit(xz_overlay.bbox)

                subwindow2 = getattr(self.SubWindow, 'subwindow2', None)
                if subwindow2 is not None and subwindow2.isHidden() == False:
                    zy_plabel = self._get_plane_plabel('zy')
                    if zy_plabel and zy_plabel.isVisible():
                        zy_canvas = self._get_plane_canvas('zy')
                        zy_overlay = self._get_plane_overlay_ax('zy')
                        zy_bg = self._get_plane_background('zy')
                        zy_chlabel = self._get_plane_chlabel('zy')
                        zy_hline = self._get_plane_hline('zy')
                        if zy_canvas and zy_bg:
                            zy_canvas.restore_region(zy_bg)
                        if zy_overlay:
                            if zy_hline:
                                zy_overlay.draw_artist(zy_hline)
                            if zy_vline:
                                zy_overlay.draw_artist(zy_vline)
                            zy_cpoint = self._get_plane_cpoint('zy')
                            if zy_cpoint is not None and zy_cpoint.get_visible():
                                zy_overlay.draw_artist(zy_cpoint)
                            if zy_chlabel:
                                zy_overlay.draw_artist(zy_chlabel)
                        if marker_mgr is not None:
                            marker_mgr.draw_markers_for_blit('zy')
                        if zy_canvas and zy_overlay:
                            zy_canvas.blit(zy_overlay.bbox)

        elif plane == 'xz':
            k = max(0, min(k, self.data.shape[-2]))
            xpix = max(0, min(self._get_shared_xpix(), self.data.shape[-1] - 1))
            zpix = max(0, min(self._get_shared_zpix(), self.data.shape[0 if self.data.ndim == 3 else 1] - 1))
            z = self.format_pix.convert_chpix_to_world(self.plane, xpix, k, zpix)
            z_str = self.format_pix.convert_chval_to_world_str(self.plane, z)
            
            xz_chlabel = self._get_plane_chlabel('xz')
            xz_chval_box = self._get_plane_chval_box('xz')
            if xz_chval_box:
                if xz_chlabel:
                    xz_chlabel.set_text(self._format_channel_label_text(z_str, "xz"))
                xz_chval_box.setText("%s" % z_str)
                xz_chval_box.setCursorPosition(0)
                if xz_chlabel and not xz_chlabel.get_visible():
                    xz_chlabel.set_visible(True)
            
            self._update_shared_pix(xpix, k, zpix)
            self._update_shared_world_xyz(self._get_shared_world_x(), z, self._get_shared_world_z())
            self._update_shared_world_xyz_str(self._get_shared_world_x_str(), z_str, self._get_shared_world_z_str())
            
            if self.data.ndim == 3:
                self._set_plane_image_data('xz', self.data[:, k, :], cache_index=k)
            elif self.data.ndim == 4:
                self._set_plane_image_data('xz', self.data[0, :, k, :], cache_index=k)
            # Image changed -> invalidate overlay background so blit won't restore stale pixels.
            self._invalidate_plane_background('xz')

            # Draw image and label (blit is handled by refresh_display_after_contour_update)
            # Redundant draw calls removed.

            self._refresh_after_channel_image_update('xz')
            self._schedule_large_data_prefetch('xz', k)

            xy_hline = self._get_plane_hline('xy')
            zy_hline = self._get_plane_hline('zy')
            if not self._get_clicked('zy') and xy_hline:
                xy_hline.set_ydata([k])
                self._update_plane_cursor('xy', y=k)
                self._set_crosshair_visibility_for_plane('xy', show_cursor_overlay)
            if not self._get_clicked('xy') and zy_hline:
                zy_hline.set_ydata([k])
                self._update_plane_cursor('zy', y=k)
                self._set_crosshair_visibility_for_plane('zy', show_cursor_overlay)

            if not self._get_clicked('xz'):
                xy_plabel = self._get_plane_plabel('xy')
                if xy_plabel and xy_plabel.isVisible():
                    FITSViewer.main_window.redraw_main_overlay_and_blit()

                    subwindow2 = getattr(self.SubWindow, 'subwindow2', None)
                    if subwindow2 is not None and subwindow2.isHidden() == False:
                        zy_canvas = self._get_plane_canvas('zy')
                        zy_overlay = self._get_plane_overlay_ax('zy')
                        zy_bg = self._get_plane_background('zy')
                        if zy_bg is None:
                             zy_bg = self._refresh_overlay_background('zy')
                        
                        zy_chlabel = self._get_plane_chlabel('zy')
                        zy_vline = self._get_plane_vline('zy')
                        if zy_canvas and zy_bg:
                            zy_canvas.restore_region(zy_bg)
                        if zy_overlay:
                            if zy_hline:
                                zy_overlay.draw_artist(zy_hline)
                            if zy_vline:
                                zy_overlay.draw_artist(zy_vline)
                            zy_cpoint = self._get_plane_cpoint('zy')
                            if zy_cpoint is not None and zy_cpoint.get_visible():
                                zy_overlay.draw_artist(zy_cpoint)
                            if zy_chlabel:
                                zy_overlay.draw_artist(zy_chlabel)
                        if marker_mgr is not None:
                            marker_mgr.draw_markers_for_blit('zy')
                        if zy_canvas and zy_overlay:
                            zy_canvas.blit(zy_overlay.bbox)

        elif plane == 'zy':
            k = max(0, min(k, self.data.shape[-1] - 1))
            ypix = max(0, min(self._get_shared_ypix(), self.data.shape[-2] - 1))
            xpix = max(0, min(self._get_shared_xpix(), self.data.shape[0 if self.data.ndim == 3 else 1] - 1))
            z = self.format_pix.convert_chpix_to_world(self.plane, k, ypix, xpix)
            z_str = self.format_pix.convert_chval_to_world_str(self.plane, z)
            
            zy_chlabel = self._get_plane_chlabel('zy')
            zy_chval_box = self._get_plane_chval_box('zy')
            if zy_chval_box:
                if zy_chlabel:
                    zy_chlabel.set_text(self._format_channel_label_text(z_str, "zy"))
                zy_chval_box.setText("%s" % z_str)
                zy_chval_box.setCursorPosition(0)
                if zy_chlabel and not zy_chlabel.get_visible():
                    zy_chlabel.set_visible(True)
            
            self._update_shared_pix(k, ypix, self._get_shared_zpix())
            self._update_shared_world_xyz(z, self._get_shared_world_y(), self._get_shared_world_z())
            self._update_shared_world_xyz_str(z_str, self._get_shared_world_y_str(), self._get_shared_world_z_str())
            
            if self.data.ndim == 3:
                self._set_plane_image_data('zy', self.data[:, :, k].T, cache_index=k)
            elif self.data.ndim == 4:
                self._set_plane_image_data('zy', self.data[0, :, :, k].T, cache_index=k)
            # Image changed -> invalidate overlay background so blit won't restore stale pixels.
            self._invalidate_plane_background('zy')

            # Draw image and label (blit is handled by refresh_display_after_contour_update)
            # Redundant draw calls removed.

            self._refresh_after_channel_image_update('zy')
            self._schedule_large_data_prefetch('zy', k)

            xy_vline = self._get_plane_vline('xy')
            xz_vline = self._get_plane_vline('xz')
            if not self._get_clicked('xz') and xy_vline:
                xy_vline.set_xdata([k])
                self._update_plane_cursor('xy', x=k)
                self._set_crosshair_visibility_for_plane('xy', show_cursor_overlay)
            if not self._get_clicked('xy') and xz_vline:
                xz_vline.set_xdata([k])
                self._update_plane_cursor('xz', x=k)
                self._set_crosshair_visibility_for_plane('xz', show_cursor_overlay)

            if not self._get_clicked('zy'):
                xy_plabel = self._get_plane_plabel('xy')
                if xy_plabel and xy_plabel.isVisible():
                    FITSViewer.main_window.redraw_main_overlay_and_blit()





                    if self.SubWindow.subwindow1.isHidden() == False:
                        xz_canvas = self._get_plane_canvas('xz')
                        xz_overlay = self._get_plane_overlay_ax('xz')
                        xz_bg = self._get_plane_background('xz')
                        if xz_bg is None:
                            xz_bg = self._refresh_overlay_background('xz')
                        
                        xz_chlabel = self._get_plane_chlabel('xz')
                        xz_vline = self._get_plane_vline('xz')
                        xz_hline = self._get_plane_hline('xz')
                        if xz_canvas and xz_bg:
                            xz_canvas.restore_region(xz_bg)
                        if xz_overlay:
                            if xz_hline:
                                xz_overlay.draw_artist(xz_hline)
                            if xz_vline:
                                xz_overlay.draw_artist(xz_vline)
                            xz_cpoint = self._get_plane_cpoint('xz')
                            if xz_cpoint is not None and xz_cpoint.get_visible():
                                xz_overlay.draw_artist(xz_cpoint)
                            if xz_chlabel:
                                xz_overlay.draw_artist(xz_chlabel)
                        if marker_mgr is not None:
                            marker_mgr.draw_markers_for_blit('xz')
                        if xz_canvas and xz_overlay:
                            xz_canvas.blit(xz_overlay.bbox)

        if hasattr(self, 'control_panel') and self.control_panel.pvd_panel:
            if k >= 0:
                self.control_panel.pvd_panel.update_cursor(k)
        


    def _default_contour_label(self) -> str:
        plane = getattr(self, "plane", "")
        plane_tag = plane.upper() if plane else ""
        try:
            title = self.windowTitle()
        except Exception:
            title = ""
        title = title or ""
        if plane_tag:
            if title:
                return f"{title} [{plane_tag}]"
            return plane_tag
        return title or self.__class__.__name__

    def _contour_items_provider(self):
        if not hasattr(self, "ax") or self.ax is None:
            return []
        if not hasattr(self, "im") or self.im is None:
            return []
        arr = self.im.get_array()
        if arr is None:
            return []
        data = arr
        if np.ma.isMaskedArray(data):
            data = data.filled(np.nan)
        data = np.asarray(data)
        label = self._default_contour_label()
        metadata = {}
        try:
            clim = self.im.get_clim()
        except Exception:
            clim = None
        if clim is not None:
            metadata["clim"] = tuple(clim)
        return [ContourItem(ax=self.ax, data=data, label=label, metadata=metadata)]

    def _register_contour_layer(self):
        if self._contour_layer_id is not None:
            return
        manager = ContourManager.instance()
        layer_id = f"{self.__class__.__name__.lower()}-{id(self)}"
        label = self._default_contour_label()
        try:
            manager.register_layer(
                layer_id=layer_id,
                label=label,
                plane=getattr(self, "plane", None),
                provider=self._contour_items_provider,
                owner=self,
            )
        except ValueError:
            return
        self._contour_layer_id = layer_id

        # Removed manual manager.contour_updated.disconnect(self.refresh_display_after_contour_update)
        try:
            manager.contour_updated.connect(self.refresh_display_after_contour_update)
        except Exception as e:
            print(f"Failed to connect contour_updated signal: {e}")

        if self._contour_layer_id and not self._contour_title_connected:
            try:
                self.windowTitleChanged.connect(self._handle_title_change_for_contours)
                self._contour_title_connected = True
            except Exception:
                pass

    def _handle_title_change_for_contours(self, _title: str) -> None:
        if not self._contour_layer_id:
            return
        manager = ContourManager.instance()
        manager.rename_layer(self._contour_layer_id, self._default_contour_label())

    def _refresh_contours(self) -> bool:
        """Refresh contours. Returns True if contours were actually refreshed."""
        perf_token = self._perf_start(f"{getattr(self, 'plane', '?')} _refresh_contours")
        if not self._contour_layer_id or not self._contours_active():
            self._perf_end(perf_token)
            return False
        manager = ContourManager.instance()
        layer = manager._layers.get(self._contour_layer_id)
        if layer is None or layer._last_parameters is None:
            self._perf_end(perf_token)
            return False
        manager.refresh_layer(self._contour_layer_id)
        self._perf_end(perf_token)
        return True

    def _refresh_after_channel_image_update(self, plane: str) -> None:
        """
        Refresh contour/background state after set_data() in update_channel().

        If contour refresh already rebuilt the overlay background (via signal),
        skip the extra manual redraw.
        """
        self._refresh_contours()
        if self._get_plane_background(plane) is None:
            self.refresh_display_after_contour_update(self._contour_layer_id)

    def _unregister_contour_layer(self):
        if not self._contour_layer_id:
            return
        ContourManager.instance().unregister_layer(self._contour_layer_id)
        self._contour_layer_id = None


    def resizeEvent(self, event):
        self._position_click_label()
        if self.toolbar._subplot_dialog is not None:
            self.toolbar._subplot_dialog.width_spin.setValue(int(self.window().width()))
            self.toolbar._subplot_dialog.height_spin.setValue(int(self.window().height()))
            
        super().resizeEvent(event)
        if self._is_colorbar_auto_layout_enabled():
            self._schedule_colorbar_auto_layout_if_anchor_changed(force=False)


    def redraw_main_overlay_and_blit(self, *, lightweight: bool = False):
        """
        Redraws overlay artists (crosshairs, PVD, regions, HPBW)
        on the main XY canvas and blits the result.
        Contours are assumed to be part of the background captured in update_channel.
        """
        perf_token = self._perf_start(f"{getattr(self, 'plane', '?')} redraw_main_overlay_and_blit")
        # Get xy plane state
        xy_state = self.get_viewer_state('xy')
        if xy_state is None:
            xy_state = self.state if getattr(self, 'plane', None) == 'xy' else None
        if xy_state is None or xy_state.canvas is None or xy_state.overlay_ax is None:
            self._perf_end(perf_token)
            return
        self._sync_overlay_axes_to_main()
        updates_enabled = None
        if xy_state._background is None:
            if hasattr(xy_state.canvas, "updatesEnabled") and hasattr(xy_state.canvas, "setUpdatesEnabled"):
                try:
                    updates_enabled = bool(xy_state.canvas.updatesEnabled())
                    xy_state.canvas.setUpdatesEnabled(False)
                except Exception:
                    updates_enabled = None
        try:
            # If background is None, rebuild it from a clean frame.
            if xy_state._background is None:
                rebuilt = None
                refresh_bg = getattr(self, "_refresh_overlay_background", None)
                if callable(refresh_bg):
                    try:
                        rebuilt = refresh_bg('xy')
                    except Exception:
                        rebuilt = None
                if rebuilt is None:
                    try:
                        xy_state.canvas.draw()
                    except Exception:
                        pass
                    rebuilt = xy_state.copy_overlay_background()
                xy_state._background = rebuilt
                if xy_state._background is None:
                    # Fallback to full redraw if we can't capture background
                    xy_state.canvas.draw_idle()
                    return

            xy_state.canvas.restore_region(xy_state._background)

            # Draw artists (on overlay_ax)
            xy_state.overlay_ax.draw_artist(xy_state.vline)
            xy_state.overlay_ax.draw_artist(xy_state.hline)
            xy_cpoint = getattr(xy_state, "cpoint", None)
            if xy_cpoint is not None and xy_cpoint.get_visible():
                xy_state.overlay_ax.draw_artist(xy_cpoint)

            if xy_state.chlabel:
                xy_state.overlay_ax.draw_artist(xy_state.chlabel)
            if hasattr(self, 'control_panel'):
                if self.control_panel.pvd_panel is not None and self.control_panel.pvd_panel.arrow_artist is not None:
                    xy_state.overlay_ax.draw_artist(self.control_panel.pvd_panel.arrow_artist)
                    for indicator in self.control_panel.pvd_panel.width_indicators:
                        xy_state.overlay_ax.draw_artist(indicator)
                    if self.control_panel.pvd_panel.pos_indicator_on_arrow is not None:
                        xy_state.overlay_ax.draw_artist(self.control_panel.pvd_panel.pos_indicator_on_arrow)

            if hasattr(self, 'region_manager'):
                self.region_manager.draw_regions_for_blit()

            # Keep HPBW visible even in lightweight drag redraws.
            if xy_state.hpbw:
                xy_state.hpbw.update_position()

            if hasattr(self, 'marker_manager') and self.marker_manager is not None:
                self.marker_manager.draw_markers_for_blit('xy')

            xy_state.canvas.blit(xy_state.overlay_ax.bbox)
            self._blit_colorbar_foreground_for_state(xy_state, force=False)
        finally:
            if updates_enabled is not None:
                try:
                    xy_state.canvas.setUpdatesEnabled(bool(updates_enabled))
                    if updates_enabled:
                        xy_state.canvas.update()
                except Exception:
                    pass
            self._perf_end(perf_token)


    def redraw_overlay_for_plane(self, plane=None, *, lightweight: bool = False):
        """
        Restore overlay background and re-blit crosshair / markers for the specified plane.
        Falls back to the viewer's current plane if not provided.
        """
        plane = plane or getattr(self, 'plane', None)
        if plane not in ('xy', 'xz', 'zy'):
            return

        marker_manager = getattr(self, 'marker_manager', None)
        if plane == 'xy':
            self.redraw_main_overlay_and_blit(lightweight=lightweight)
            return

        # Get state for the target plane
        state = self.get_viewer_state(plane)
        if state is None:
            return

        canvas = state.canvas
        overlay_ax = state.overlay_ax
        background = state._background
        if canvas is None or overlay_ax is None:
            if canvas is not None:
                canvas.draw_idle()
            return

        if background is None:
            rebuilt = None
            refresh_bg = getattr(self, "_refresh_overlay_background", None)
            if callable(refresh_bg):
                try:
                    rebuilt = refresh_bg(plane)
                except Exception:
                    rebuilt = None
            background = rebuilt if rebuilt is not None else state._background
            if background is None:
                canvas.draw_idle()
                return

        try:
            canvas.restore_region(background)
        except Exception:
            rebuilt = None
            refresh_bg = getattr(self, "_refresh_overlay_background", None)
            if callable(refresh_bg):
                try:
                    rebuilt = refresh_bg(plane)
                except Exception:
                    rebuilt = None
            background = rebuilt if rebuilt is not None else state._background
            if background is None:
                canvas.draw_idle()
                return
            try:
                canvas.restore_region(background)
            except Exception:
                canvas.draw_idle()
                return

        for artist in (state.hline, state.vline, getattr(state, "cpoint", None), state.chlabel):
            if artist is None:
                continue
            try:
                overlay_ax.draw_artist(artist)
            except Exception:
                continue

        if marker_manager is not None:
            try:
                marker_manager.draw_markers_for_blit(plane)
            except Exception:
                pass

        try:
            canvas.blit(overlay_ax.bbox)
        except Exception:
            canvas.draw_idle()
            return
        self._blit_colorbar_foreground_for_state(state, force=False)


    def open_marker_panel(self):
        if self.marker_panel is None or not self.marker_panel.isVisible():
            from takefits.tools.marker_panel import MarkerPanel
            self.marker_panel = MarkerPanel(self, self.marker_manager)
            try:
                self.marker_panel.destroyed.connect(lambda: setattr(self, 'marker_panel', None))
            except Exception:
                pass
            self.marker_panel.setProperty("_marker_panel_positioned", False)
        self.marker_panel.show()
        if not bool(self.marker_panel.property("_marker_panel_positioned")):
            self._position_marker_panel(self.marker_panel)
            self.marker_panel.setProperty("_marker_panel_positioned", True)
        self.marker_panel.raise_()
        self.marker_panel.activateWindow()

    # Marker plane helpers -------------------------------------------------
    def has_marker_plane(self, plane: str) -> bool:
        base = (plane or "").lower()
        state = self.get_viewer_state(base)
        if state is not None:
            return state.overlay_ax is not None
        return False

    def marker_axes_for_plane(self, plane: str):
        base = (plane or "").lower()
        state = self.get_viewer_state(base)
        if state is not None:
            return state.overlay_ax
        return None

    def marker_plane_base(self, plane: str) -> str:
        if not plane:
            return getattr(self, "plane", "xy")
        plane_lower = plane.lower()
        if "xz" in plane_lower:
            return "xz"
        if "zy" in plane_lower:
            return "zy"
        if "xy" in plane_lower:
            return "xy"
        return plane_lower

    def remap_loaded_marker_state(self, state, *, source_plane: Optional[str] = None, world_frame: Optional[str] = None):
        """
        Remap incoming marker states when loading saved markers.

        - Channel-map planes (channel_<base>_N) collapse to this viewer's base plane.
        - Otherwise, keep planes this viewer can display; fall back to base plane if supported.
        """
        plane_name = (state.plane or "").lower()
        base = self.marker_plane_base(plane_name)

        if plane_name.startswith("channel_"):
            return [base] if self.has_marker_plane(base) else []

        targets = []
        if self.has_marker_plane(plane_name):
            targets.append(plane_name)
        elif self.has_marker_plane(base):
            targets.append(base)
        return targets

    def _position_marker_panel(self, panel):
        if panel is None:
            return
        try:
            panel.adjustSize()
        except Exception:
            pass
        anchor = FITSViewer.main_window
        if anchor is None:
            anchor = self
        anchor_frame = anchor.frameGeometry()
        top_right = anchor_frame.topRight()
        size_hint = panel.sizeHint()
        width = size_hint.width() if size_hint.width() > 0 else panel.width()
        height = size_hint.height() if size_hint.height() > 0 else panel.height()
        margin = 20
        target_x = top_right.x() + margin
        target_y = top_right.y()

        screen = QGuiApplication.screenAt(top_right)
        if screen is None:
            screen = QGuiApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            if width > 0:
                target_x = min(target_x, available.right() - width)
            if height > 0:
                target_y = max(available.top(), min(target_y, available.bottom() - height))
            target_x = max(target_x, available.left())

        panel.move(target_x, target_y)


    def disable_region_mode(self):
        if not getattr(self, 'region_mode_enabled', False):
            return
        self.region_mode_enabled = False
        region_manager = getattr(self, 'region_manager', None)
        if region_manager is not None:
            try:
                region_manager.deselect_all(skip_redraw=True)
            except Exception:
                pass
        if hasattr(self, 'region_shape'):
            self.region_shape = None
        title = getattr(self, 'original_window_title', None)
        if title:
            try:
                self.setWindowTitle(title)
            except Exception:
                pass
        if getattr(self, 'plane', None) == 'xy':
            try:
                self.redraw_main_overlay_and_blit()
            except Exception:
                pass
        else:
            canvas = getattr(self, 'canvas', None)
            if canvas is not None:
                canvas.draw_idle()


    def _reset_navigation_mode(self):
        toolbar = getattr(self, 'toolbar', None)
        if toolbar is None:
            return
        try:
            toolbar.mode = ''
        except Exception:
            pass
        actions = getattr(toolbar, '_actions', None)
        if isinstance(actions, dict):
            for key in ('zoom', 'pan'):
                action = actions.get(key)
                if action is not None:
                    action.setChecked(False)
        try:
            toolbar._clear_cursor_override()
        except Exception:
            pass


    def set_marker_mode(self, enabled: bool = True):
        enabled = bool(enabled)
        previous = getattr(self, 'marker_mode_enabled', False)
        self.marker_mode_enabled = enabled
        marker_manager = getattr(self, 'marker_manager', None)
        plane = getattr(self, 'plane', 'xy')
        if not enabled:
            if marker_manager is not None:
                marker_manager.cancel_placement()
            panel = getattr(self, 'marker_panel', None)
            if panel is not None and getattr(panel, 'placement_toggle', None) is not None:
                if panel.placement_toggle.isChecked():
                    panel.placement_toggle.blockSignals(True)
                    panel.placement_toggle.setChecked(False)
                    panel.placement_toggle.blockSignals(False)
                    try:
                        panel._on_placement_toggled(False)
                    except Exception:
                        pass
        else:
            if marker_manager is not None:
                marker_manager.set_active_plane(plane)
            if getattr(self, 'region_mode_enabled', False):
                self.disable_region_mode()
            self._reset_navigation_mode()
        if enabled and self.toolbar.mode in ('zoom rect', 'pan/zoom'):
            self._reset_navigation_mode()
        if previous != enabled:
            subwindows = getattr(self, 'subwindows', None)
            if subwindows:
                for subwindow in subwindows:
                    if subwindow is not None and subwindow is not self:
                        try:
                            subwindow.set_marker_mode(enabled)
                        except Exception:
                            pass



    def on_key_press(self, event):
        """
        Handles key press events coming directly from the Matplotlib canvas.
        """
        region_manager = getattr(self, 'region_manager', None)
        if region_manager is not None:
            region_manager.handle_key_press(event)
        marker_manager = getattr(self, 'marker_manager', None)
        if marker_manager is not None:
            marker_manager.handle_key_press(event)
        if event.key == 'backspace' or event.key == 'delete':
            if hasattr(self, 'region_mode_enabled') and self.region_mode_enabled:
                self.region_manager.delete_selected_region()
            elif getattr(self, 'marker_mode_enabled', False) and marker_manager is not None:
                marker = marker_manager.selected_marker()
                if marker is not None:
                    plane = marker.plane
                    marker_manager.remove_marker(marker.marker_id, plane)
                    marker_manager.redraw_plane(plane)

    def on_key_release(self, event):
        region_manager = getattr(self, 'region_manager', None)
        if region_manager is not None:
            region_manager.handle_key_release(event)
        marker_manager = getattr(self, 'marker_manager', None)
        if marker_manager is not None:
            marker_manager.handle_key_release(event)

    def closeEvent(self, event):
        self._unregister_contour_layer()
        super().closeEvent(event)

    def set_overlay_updates_enabled(self, enabled: bool):
        self._overlay_updates_enabled = bool(enabled)

    def update_cube(self):
        if self.data.ndim == 2:
            # For 2D data, self.cube is the same as self.data
            self.cube = self.data
        elif self.data.ndim == 3:
            self.cube = self.data
        elif self.data.ndim == 4:
            self.cube = self.data[0]
        main_window = getattr(self, 'main_window', None)
        if main_window is None and hasattr(self, 'app_state'):
            main_window = self
        if main_window is not None and hasattr(main_window, 'sync_app_state_data'):
            try:
                main_window.sync_app_state_data()
            except Exception:
                pass

    def _get_plane_viewer(self, plane):
        """
        Return the FITSViewer instance responsible for the given plane.
        """
        current_plane = getattr(self, 'plane', None)
        if plane == current_plane or (plane == 'xy' and current_plane is None):
            return self

        # Check subwindows if available (Main window context).
        subwindows = getattr(self, 'subwindows', [])
        if plane == 'xz' and subwindows:
            return subwindows[0] if len(subwindows) >= 1 else None
        if plane == 'zy' and subwindows:
            return subwindows[1] if len(subwindows) >= 2 else None

        # If we are in a subwindow, delegate to the parent main window.
        parent = getattr(self, 'parent', None)
        if parent is not None and hasattr(parent, '_get_plane_viewer'):
            return parent._get_plane_viewer(plane)

        return None

    def _range_limit_strings(self, plane, xlim, ylim):
        """
        Return formatted world limit strings for a plane while keeping shared cursor
        state untouched. XY limits are always computed in native WCS frame.
        """
        plane_viewer = self._get_plane_viewer(plane)
        if plane_viewer is None:
            return None

        fmt = getattr(plane_viewer, 'format_pix', None)
        if fmt is None:
            return None

        x_ref = (xlim[0] + xlim[1]) / 2.0
        y_ref = (ylim[0] + ylim[1]) / 2.0

        saved_world = (
            self._get_shared_world_x(),
            self._get_shared_world_y(),
            self._get_shared_world_z(),
            self._get_shared_world_s(),
        )
        saved_world_str = (
            self._get_shared_world_x_str(),
            self._get_shared_world_y_str(),
            self._get_shared_world_z_str(),
            self._get_shared_world_s_str(),
        )
        saved_frame = None
        frame_switched = False
        if str(plane or "").lower() == "xy":
            try:
                saved_frame = self._get_shared_display_frame()
                if normalize_display_frame(saved_frame) != "native":
                    self._set_shared_display_frame("native")
                    frame_switched = True
            except Exception:
                saved_frame = None
                frame_switched = False

        try:
            primary_min, _ = fmt.convert(plane_viewer.plane, xlim[0], y_ref)
            primary_max, _ = fmt.convert(plane_viewer.plane, xlim[1], y_ref)
            _, secondary_min = fmt.convert(plane_viewer.plane, x_ref, ylim[0])
            _, secondary_max = fmt.convert(plane_viewer.plane, x_ref, ylim[1])
            return (
                str(primary_min),
                str(primary_max),
                str(secondary_min),
                str(secondary_max),
            )
        except Exception:
            return None
        finally:
            try:
                self._update_shared_world_xyz(*saved_world)
            except Exception:
                pass
            try:
                self._update_shared_world_xyz_str(*saved_world_str)
            except Exception:
                pass
            if frame_switched and saved_frame is not None:
                try:
                    self._set_shared_display_frame(saved_frame)
                except Exception:
                    pass

    def world_extent(self, plane, cache=True):
        """
        Return the full data extent for the chosen plane in world coordinates.
        """
        if cache and plane in self._full_world_limits:
            return self._full_world_limits[plane].copy()

        plane_viewer = self._get_plane_viewer(plane)
        if plane_viewer is None:
            return {}

        # Determine full pixel ranges for each axis.
        data = getattr(plane_viewer, 'data', None)
        if data is None:
            return {}

        ndim = getattr(data, 'ndim', 0)

        def axis_limits(axis_index, length):
            return (-0.5, length - 0.5)

        axis_labels = {
            'xy': ('x', 'y'),
            'xz': ('x', 'z'),
            'zy': ('z', 'y'),
        }
        primary_label, secondary_label = axis_labels.get(plane, ('x', 'y'))

        # Determine pixel limits based on data shape.
        limits = {}
        if hasattr(plane_viewer, 'original_xlim'):
            limits['x'] = plane_viewer.original_xlim
        else:
            limits['x'] = axis_limits(0, data.shape[-1])

        if hasattr(plane_viewer, 'original_ylim'):
            limits['y'] = plane_viewer.original_ylim
        else:
            limits['y'] = axis_limits(1, data.shape[-2])

        if ndim >= 3:
            if hasattr(plane_viewer, 'original_zlim'):
                limits['z'] = plane_viewer.original_zlim
            else:
                limits['z'] = axis_limits(2, data.shape[-3])
        else:
            limits['z'] = (0.0, 0.0)

        if plane == 'xy':
            xlim = limits['x']
            ylim = limits['y']
        elif plane == 'xz':
            xlim = limits['x']
            ylim = limits['z']
        elif plane == 'zy':
            xlim = limits['z']
            ylim = limits['y']
        else:
            return {}

        limits_text = self._range_limit_strings(plane, xlim, ylim)
        if limits_text is None:
            return {}
        primary_min, primary_max, secondary_min, secondary_max = limits_text

        extent = {
            f"{primary_label}_min": primary_min,
            f"{primary_label}_max": primary_max,
            f"{secondary_label}_min": secondary_min,
            f"{secondary_label}_max": secondary_max,
        }
        if cache:
            self._full_world_limits[plane] = extent.copy()
        return extent.copy()

    def compute_world_limits(self, plane, xlim, ylim):
        """
        Convert pixel axis limits to world-coordinate strings for the specified plane.
        Returns a dictionary keyed by '<axis>_min'/'<axis>_max'.
        """
        if xlim is None or ylim is None:
            return {}

        plane_viewer = self._get_plane_viewer(plane)
        if plane_viewer is None:
            return {}

        axis_labels = {
            'xy': ('x', 'y'),
            'xz': ('x', 'z'),
            'zy': ('z', 'y'),
        }
        primary_axis, secondary_axis = axis_labels.get(plane, ('x', 'y'))
        limits_text = self._range_limit_strings(plane, xlim, ylim)
        if limits_text is None:
            return {}
        primary_min, primary_max, secondary_min, secondary_max = limits_text

        result = {
            f"{primary_axis}_min": primary_min,
            f"{primary_axis}_max": primary_max,
            f"{secondary_axis}_min": secondary_min,
            f"{secondary_axis}_max": secondary_max,
        }
        return result

    def _suspend_regions_for_full_draw(self):
        """Hide regions before triggering a full canvas draw so backgrounds stay clean."""
        if getattr(self, 'plane', None) != 'xy':
            return
        region_manager = getattr(self, 'region_manager', None)
        if region_manager is None:
            return
        hidden = region_manager.prepare_for_background_capture()
        if hidden:
            self._pending_region_restore.extend(hidden)


    def _get_range_input(self, plane, input_type):
        """Get range input value from the specified plane's state."""
        state = self.get_viewer_state(plane)
        if state is None:
            return ""
        input_field = getattr(state, f'{input_type}_input', None)
        return input_field.text() if input_field else ""

    def _set_range_input(self, plane, input_type, value):
        """Set range input value for the specified plane's state."""
        state = self.get_viewer_state(plane)
        if state is None:
            return
        input_field = getattr(state, f'{input_type}_input', None)
        if input_field:
            input_field.setText(str(value))

    def _get_main_viewer(self):
        """Return the owning main viewer for this window."""
        main = getattr(self, 'main_viewer', None)
        if main is not None:
            return main
        class_main = getattr(FITSViewer, 'main_window', None)
        if class_main is not None:
            return class_main
        return self

    def _resolve_world_anchor(self, axis: str) -> str:
        """Return a safe world-value anchor string for WCS conversions."""
        main = self._get_main_viewer()
        if axis == 'x':
            candidates = [
                getattr(self, 'original_xval', None),
                self._get_range_input('xy', 'xmin'),
                self._get_range_input('xz', 'xmin'),
                getattr(main, 'original_xval', None),
            ]
        elif axis == 'y':
            candidates = [
                getattr(self, 'original_yval', None),
                self._get_range_input('xy', 'ymin'),
                self._get_range_input('zy', 'ymin'),
                getattr(main, 'original_yval', None),
            ]
        else:  # 'z'
            candidates = [
                getattr(self, 'original_zval', None),
                self._get_range_input('xz', 'zmin'),
                self._get_range_input('zy', 'zmin'),
                getattr(main, 'original_zval', None),
            ]

        for candidate in candidates:
            if candidate is None:
                continue
            text = str(candidate).strip()
            if text and text.lower() != 'none':
                return text
        return "0"

    def _sync_range_panel_inputs(self, *planes: str):
        """Synchronize range-panel inputs on the main viewer, if present."""
        root = self._get_main_viewer()
        range_panel = getattr(root, 'range_panel', None)
        if range_panel is None:
            return
        for plane in planes:
            try:
                range_panel._sync_inputs(plane)
            except Exception:
                continue

    def set_x_range(self):
        """Set the X range for both the MainWindow and SubWindow1."""
        try:
            main_viewer = self._get_main_viewer()
            if self.plane == 'xy':
                x_min = self._get_range_input('xy', 'xmin')
                x_max = self._get_range_input('xy', 'xmax')
                y_min = self._get_range_input('xy', 'ymin')
                y_max = self._get_range_input('xy', 'ymax')
            elif self.plane == 'zy':
                x_min = self._get_range_input('xy', 'xmin')
                x_max = self._get_range_input('xy', 'xmax')
                y_min = self._get_range_input('zy', 'ymin')
                y_max = self._get_range_input('zy', 'ymax')

            elif self.plane == 'xz':
                x_min = self._get_range_input('xz', 'xmin')
                x_max = self._get_range_input('xz', 'xmax')
                y_min = self._get_range_input('zy', 'ymin')
                y_max = self._get_range_input('zy', 'ymax')

            z_anchor = self._resolve_world_anchor('z')
            if self.original_zval is None:
                self.original_zval = z_anchor
                
            if self.data.ndim == 3:
                xp_min = float(self.converter.world_to_pix(x_min, y_min, z_anchor)[0])
                xp_max = float(self.converter.world_to_pix(x_max, y_max, z_anchor)[0])
            elif self.data.ndim == 4:
                xp_min = float(self.converter.world_to_pix(x_min, y_min, z_anchor, 0)[0])
                xp_max = float(self.converter.world_to_pix(x_max, y_max, z_anchor, 0)[0])
            elif self.data.ndim == 2:
                xp_min = float(self.converter.world_to_pix(x_min, y_min)[0])
                xp_max = float(self.converter.world_to_pix(x_max, y_max)[0])
                
            if xp_min > xp_max: xp_min, xp_max = xp_max, xp_min
            
            self.ax.set_xlim(xp_min, xp_max)
            self.overlay_ax.set_position(self.ax.get_position())
            #self.hpbw.update_position()
            self._suspend_regions_for_full_draw()
            self.canvas.draw_idle()

            # Apply X range to SubWindow1 (if exists)
            if self.plane == 'xy' and len(self.subwindows) > 0 and self.subwindows[0]:                
                self.subwindows[0].ax.set_xlim(xp_min, xp_max)
                self.subwindows[0].overlay_ax.set_position(self.subwindows[0].ax.get_position())
                self.subwindows[0]._suspend_regions_for_full_draw()
                self.subwindows[0].canvas.draw_idle()
                self.subwindows[0].x_min_input.setText(x_min)
                self.subwindows[0].x_max_input.setText(x_max)


                
            elif self.plane == 'xz':
                main_viewer.ax.set_xlim(xp_min, xp_max)
                main_viewer.overlay_ax.set_position(main_viewer.ax.get_position())
                main_viewer._suspend_regions_for_full_draw()
                main_viewer.canvas.draw_idle()
                main_viewer.x_min_input.setText(x_min)
                main_viewer.x_max_input.setText(x_max)
                
            


        except (ValueError, TypeError):
            if not getattr(self, "_suppress_range_warning", False):
                QMessageBox.warning(self, 'Invalid Input', 'Please enter valid numeric values for the X range.')
            return

        self._sync_range_panel_inputs('xy')
        record_history = getattr(main_viewer, "_record_shared_view_history", None)
        if callable(record_history):
            try:
                record_history(reason=f"set_x:{self.plane}")
            except Exception:
                pass

    def set_y_range(self):
        """Set the Y range for both the MainWindow and SubWindow2."""
        try:
            main_viewer = self._get_main_viewer()
            if self.plane == 'xy':
                x_min = self._get_range_input('xy', 'xmin')
                x_max = self._get_range_input('xy', 'xmax')
                y_min = self._get_range_input('xy', 'ymin')
                y_max = self._get_range_input('xy', 'ymax')
            elif self.plane == 'zy':
                x_min = self._get_range_input('xy', 'xmin')
                x_max = self._get_range_input('xy', 'xmax')
                y_min = self._get_range_input('zy', 'ymin')
                y_max = self._get_range_input('zy', 'ymax')

            elif self.plane == 'xz':
                x_min = self._get_range_input('xz', 'xmin')
                x_max = self._get_range_input('xz', 'xmax')
                y_min = self._get_range_input('zy', 'ymin')
                y_max = self._get_range_input('zy', 'ymax')

            z_anchor = self._resolve_world_anchor('z')
            if self.original_zval is None:
                self.original_zval = z_anchor
                
            if self.data.ndim == 3:
                yp_min = float(self.converter.world_to_pix(x_min, y_min, z_anchor)[1])
                yp_max = float(self.converter.world_to_pix(x_max, y_max, z_anchor)[1])
            elif self.data.ndim == 4:
                yp_min = float(self.converter.world_to_pix(x_min, y_min, z_anchor, 0)[1])
                yp_max = float(self.converter.world_to_pix(x_max, y_max, z_anchor, 0)[1])
            elif self.data.ndim == 2:
                yp_min = float(self.converter.world_to_pix(x_min, y_min)[1])
                yp_max = float(self.converter.world_to_pix(x_max, y_max)[1])
                
            if yp_min > yp_max: yp_min, yp_max = yp_max, yp_min
            
            self.ax.set_ylim(yp_min, yp_max)
            self.overlay_ax.set_position(self.ax.get_position())
            #self.hpbw.update_position()
            self._suspend_regions_for_full_draw()
            self.canvas.draw_idle()

            # Apply Y range to SubWindow1 (if exists)

            if self.plane == 'xy' and len(self.subwindows) > 1 and self.subwindows[1]:
                self.subwindows[1].ax.set_ylim(yp_min, yp_max)
                self.subwindows[1].overlay_ax.set_position(self.subwindows[1].ax.get_position())
                self.subwindows[1]._suspend_regions_for_full_draw()
                self.subwindows[1].canvas.draw_idle()
                self.subwindows[1].y_min_input.setText(y_min)
                self.subwindows[1].y_max_input.setText(y_max)
            elif self.plane == 'zy' and len(main_viewer.subwindows) > 0 and main_viewer.subwindows[0]:
                main_viewer.ax.set_ylim(yp_min, yp_max)
                main_viewer.overlay_ax.set_position(main_viewer.ax.get_position())
                main_viewer._suspend_regions_for_full_draw()
                main_viewer.canvas.draw_idle()
                main_viewer.y_min_input.setText(y_min)
                main_viewer.y_max_input.setText(y_max)

        except (ValueError, TypeError):
            if not getattr(self, "_suppress_range_warning", False):
                QMessageBox.warning(self, 'Invalid Input', 'Please enter valid numeric values for the Y range.')
            return

        self._sync_range_panel_inputs('xy')
        record_history = getattr(main_viewer, "_record_shared_view_history", None)
        if callable(record_history):
            try:
                record_history(reason=f"set_y:{self.plane}")
            except Exception:
                pass

    def set_z_range(self):
        """Set the Z range for both SubWindow1 (vertical in XZ plane) and SubWindow2 (horizontal in ZY plane)."""
        try:
            main_viewer = self._get_main_viewer()
            if self.plane == 'xy' or self.plane == 'xz':
                z_min = self._get_range_input('xz', 'zmin')
                z_max = self._get_range_input('xz', 'zmax')
            elif self.plane == 'zy':
                z_min = self._get_range_input('zy', 'zmin')
                z_max = self._get_range_input('zy', 'zmax')

            x_anchor = self._resolve_world_anchor('x')
            y_anchor = self._resolve_world_anchor('y')
            if self.original_xval is None:
                self.original_xval = x_anchor
            if self.original_yval is None:
                self.original_yval = y_anchor

            if self.data.ndim == 3:
                zp_min = float(self.converter.world_to_pix(x_anchor, y_anchor, z_min)[2])
                zp_max = float(self.converter.world_to_pix(x_anchor, y_anchor, z_max)[2])
            elif self.data.ndim == 4:
                zp_min = float(self.converter.world_to_pix(x_anchor, y_anchor, z_min, 0)[2])
                zp_max = float(self.converter.world_to_pix(x_anchor, y_anchor, z_max, 0)[2])
            if zp_min > zp_max: zp_min, zp_max = zp_max, zp_min

            # Apply Z range to SubWindow1 (if exists)
            if len(main_viewer.subwindows) > 0 and main_viewer.subwindows[0]:  # SubWindow1 (XZ plane vertical axis)
                main_viewer.subwindows[0].ax.set_ylim(zp_min, zp_max)
                main_viewer.subwindows[0].overlay_ax.set_position(main_viewer.subwindows[0].ax.get_position())
                main_viewer.subwindows[0]._suspend_regions_for_full_draw()
                main_viewer.subwindows[0].canvas.draw_idle()
                main_viewer.subwindows[0].z_min_input.setText(z_min)
                main_viewer.subwindows[0].z_max_input.setText(z_max)

            # Apply Z range to SubWindow2 (if exists)
            if len(main_viewer.subwindows) > 1 and main_viewer.subwindows[1]:  # SubWindow2 (ZY plane horizontal axis)
                main_viewer.subwindows[1].ax.set_xlim(zp_min, zp_max)
                main_viewer.subwindows[1].overlay_ax.set_position(main_viewer.subwindows[1].ax.get_position())
                main_viewer.subwindows[1]._suspend_regions_for_full_draw()
                main_viewer.subwindows[1].canvas.draw_idle()
                main_viewer.subwindows[1].z_min_input.setText(z_min)
                main_viewer.subwindows[1].z_max_input.setText(z_max)

        except (ValueError, TypeError):
            if not getattr(self, "_suppress_range_warning", False):
                QMessageBox.warning(self, 'Invalid Input', 'Please enter valid numeric values for the Z range.')
            return

        self._sync_range_panel_inputs('xz', 'zy')
        record_history = getattr(main_viewer, "_record_shared_view_history", None)
        if callable(record_history):
            try:
                record_history(reason=f"set_z:{self.plane}")
            except Exception:
                pass

    def update_ranges(self, plane, xlim, ylim):
        if xlim is None or ylim is None:
            world_limits = self.world_extent(plane, cache=True)
            if plane == "xy":
                new_xlim = getattr(self, 'original_xlim', (-0.5, self.data.shape[-1] - 0.5))
                new_ylim = getattr(self, 'original_ylim', (-0.5, self.data.shape[-2] - 0.5))
            elif plane == "xz":
                new_xlim = getattr(self, 'original_xlim', (-0.5, self.data.shape[-1] - 0.5))
                new_ylim = getattr(self, 'original_zlim', (-0.5, self.data.shape[-3] - 0.5)) if self.data.ndim > 2 else (0.0, 0.0)
            elif plane == "zy":
                new_xlim = getattr(self, 'original_zlim', (-0.5, self.data.shape[-3] - 0.5)) if self.data.ndim > 2 else (0.0, 0.0)
                new_ylim = getattr(self, 'original_ylim', (-0.5, self.data.shape[-2] - 0.5))
            else:
                return
        else:
            new_xlim = xlim
            new_ylim = ylim
            world_limits = self.compute_world_limits(plane, new_xlim, new_ylim)

        def limit(key, fallback):
            if world_limits and key in world_limits:
                return world_limits[key]
            return str(fallback)

        def is_full_range():
            if plane == "xy":
                return (tuple(new_xlim) == tuple(getattr(self, 'original_xlim', new_xlim)) and
                        tuple(new_ylim) == tuple(getattr(self, 'original_ylim', new_ylim)))
            if plane == "xz" and self.data.ndim > 2:
                return (tuple(new_xlim) == tuple(getattr(self, 'original_xlim', new_xlim)) and
                        tuple(new_ylim) == tuple(getattr(self, 'original_zlim', new_ylim)))
            if plane == "zy" and self.data.ndim > 2:
                return (tuple(new_xlim) == tuple(getattr(self, 'original_zlim', new_xlim)) and
                        tuple(new_ylim) == tuple(getattr(self, 'original_ylim', new_ylim)))
            return False

        if is_full_range():
            world_limits = self.world_extent(plane)

        is_main = FITSViewer.main_window is None or self is FITSViewer.main_window

        if plane == "xy" and is_main:
            xmin_val = limit('x_min', new_xlim[0])
            xmax_val = limit('x_max', new_xlim[1])
            ymin_val = limit('y_min', new_ylim[0])
            ymax_val = limit('y_max', new_ylim[1])

            self._set_range_input('xy', 'xmin', xmin_val)
            self._set_range_input('xy', 'xmax', xmax_val)
            self._set_range_input('xy', 'ymin', ymin_val)
            self._set_range_input('xy', 'ymax', ymax_val)

            if self.data.ndim > 2:
                self._set_range_input('xz', 'xmin', xmin_val)
                self._set_range_input('xz', 'xmax', xmax_val)
                self._set_range_input('zy', 'ymin', ymin_val)
                self._set_range_input('zy', 'ymax', ymax_val)
            self._sync_range_panel_inputs('xy')

        elif plane == "xz" and self.data.ndim > 2 and is_main:
            xmin_val = limit('x_min', new_xlim[0])
            xmax_val = limit('x_max', new_xlim[1])
            zmin_val = limit('z_min', new_ylim[0])
            zmax_val = limit('z_max', new_ylim[1])

            self._set_range_input('xz', 'xmin', xmin_val)
            self._set_range_input('xz', 'xmax', xmax_val)
            self._set_range_input('xz', 'zmin', zmin_val)
            self._set_range_input('xz', 'zmax', zmax_val)

            self._set_range_input('xy', 'xmin', xmin_val)
            self._set_range_input('xy', 'xmax', xmax_val)
            self._set_range_input('zy', 'zmin', zmin_val)
            self._set_range_input('zy', 'zmax', zmax_val)
            self._sync_range_panel_inputs('xz')

        elif plane == "zy" and self.data.ndim > 2 and is_main:
            zmin_val = limit('z_min', new_xlim[0])
            zmax_val = limit('z_max', new_xlim[1])
            ymin_val = limit('y_min', new_ylim[0])
            ymax_val = limit('y_max', new_ylim[1])

            self._set_range_input('zy', 'zmin', zmin_val)
            self._set_range_input('zy', 'zmax', zmax_val)
            self._set_range_input('zy', 'ymin', ymin_val)
            self._set_range_input('zy', 'ymax', ymax_val)

            self._set_range_input('xy', 'ymin', ymin_val)
            self._set_range_input('xy', 'ymax', ymax_val)
            self._set_range_input('xz', 'zmin', zmin_val)
            self._set_range_input('xz', 'zmax', zmax_val)
            self._sync_range_panel_inputs('zy')


    def reset_all_ranges(self):
        main_viewer = self._get_main_viewer()
        if main_viewer is not self and hasattr(main_viewer, 'reset_all_ranges'):
            main_viewer.reset_all_ranges()
            return
        # Restore pixel extents for the main view (xy)
        self.ax.set_xlim(*self.original_xlim)
        self.ax.set_ylim(*self.original_ylim)
        self.overlay_ax.set_position(self.ax.get_position())
        self.canvas.draw_idle()

        # Restore subwindow extents if they exist (xz and zy)
        if self.data.ndim > 2 and getattr(self, 'subwindows', None):
            if len(self.subwindows) > 0 and self.subwindows[0]:
                sub = self.subwindows[0] # xz plane
                sub.ax.set_xlim(*self.original_xlim)
                sub.ax.set_ylim(*self.original_zlim)
                sub.overlay_ax.set_position(sub.ax.get_position())
                sub.canvas.draw_idle()
            if len(self.subwindows) > 1 and self.subwindows[1]:
                sub = self.subwindows[1] # zy plane
                sub.ax.set_xlim(*self.original_zlim)
                sub.ax.set_ylim(*self.original_ylim)
                sub.overlay_ax.set_position(sub.ax.get_position())
                sub.canvas.draw_idle()

        # Update world-range displays (QLineEdit boxes) for all planes
        self.update_ranges('xy', self.original_xlim, self.original_ylim)
        if self.data.ndim > 2:
            self.update_ranges('xz', self.original_xlim, self.original_zlim)
            if len(self.subwindows) > 1 and self.subwindows[1]:
                self.update_ranges('zy', self.original_zlim, self.original_ylim)
        
        # Ensure Range Control panel mirrors the refreshed ranges
        if self.data.ndim > 2:
            self._sync_range_panel_inputs('xy', 'xz', 'zy')
        else:
            self._sync_range_panel_inputs('xy')
        record_history = getattr(self, "_record_shared_view_history", None)
        if callable(record_history):
            try:
                record_history(reason="full_reset")
            except Exception:
                pass

    
    def reset_ranges(self, plane):
        self.original_xlim = (-0.5, self.data.shape[self.data.ndim-1]-0.5)
        self.original_ylim = (-0.5, self.data.shape[self.data.ndim-2]-0.5)
        if self.data.ndim > 2:
            self.original_zlim = (-0.5, self.data.shape[self.data.ndim-3]-0.5)
        if plane == 'xy': 
            self.update_ranges(plane, self.original_xlim, self.original_ylim)
        elif plane == 'xz': 
            self.update_ranges(plane, self.original_xlim, self.original_zlim)
        elif plane == 'zy': 
            if len(getattr(self, 'subwindows', []) or []) > 1 and self.subwindows[1]:
                self.update_ranges(plane, self.original_zlim, self.original_ylim)


    def refresh_coordinate_format(self):
        self.displaymap.update_axes_format()
        # Clear cached full-range strings so next access reflects new formatting.
        if hasattr(self, '_full_world_limits'):
            self._full_world_limits.clear()
        self.canvas.draw_idle()
    
    def _format_significant_digits(self, value, digits=4):
        """
        Format a number with appropriate precision:
        - For very large/small numbers (|value| >= 10^6 or |value| <= 10^-6): use scientific notation
        - For numbers >= 1: show up to 4 decimal places (remove trailing zeros)
        - For numbers < 1: show 4 significant digits
        """
        import math

        if value == 0:
            return "0"
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Inf" if value > 0 else "-Inf"
        
        magnitude = math.floor(math.log10(abs(value)))
        
        # Use scientific notation for very large or very small numbers
        if magnitude >= 6 or magnitude <= -6:
            return f"{value:.{digits-1}e}"
        
        # For numbers >= 1, use up to 4 decimal places but remove trailing zeros
        if magnitude >= 0:
            formatted = f"{value:.4f}"
            # Remove trailing zeros and decimal point if not needed
            return formatted.rstrip('0').rstrip('.')
        
        # For numbers < 1, use 4 significant digits
        decimal_places = digits - 1 - magnitude
        decimal_places = max(0, decimal_places)
        
        return f"{value:.{decimal_places}f}"

    def refresh_display_after_contour_update(self, layer_id=None):
        perf_token = self._perf_start(f"{getattr(self, 'plane', '?')} refresh_display_after_contour_update")
        if layer_id and self._contour_layer_id != layer_id:
            self._perf_end(perf_token)
            return
        plane = getattr(self, 'plane', None)
        if plane is None:
            self._perf_end(perf_token)
            return

        # Get state for current plane
        state = self.state

        artists_to_hide: List[object] = []
        if state.vline:
            artists_to_hide.append(state.vline)
        if state.hline:
            artists_to_hide.append(state.hline)
        if state.chlabel:
            artists_to_hide.append(state.chlabel)
        if plane == 'xy':
            if state.hpbw and state.hpbw.ellipse:
                artists_to_hide.append(state.hpbw.ellipse)

            main_window = getattr(self, 'parent', self)
            if hasattr(main_window, 'control_panel') and main_window.control_panel and main_window.control_panel.pvd_panel:
                pvd = main_window.control_panel.pvd_panel
                if pvd.arrow_artist:
                    artists_to_hide.append(pvd.arrow_artist)
                artists_to_hide.extend(pvd.width_indicators)
                if pvd.pos_indicator_on_arrow:
                    artists_to_hide.append(pvd.pos_indicator_on_arrow)

        hidden_regions: List[object] = []
        if hasattr(self, 'region_manager'):
            hidden_regions = self.region_manager.prepare_for_background_capture()

        vis_states: Dict[object, bool] = {}
        for artist in artists_to_hide:
            if artist:
                try:
                    vis_states[artist] = artist.get_visible()
                    artist.set_visible(False)
                except Exception as e:
                    print(f"Warning: Could not hide artist {artist}: {e}")

        contours_active = self._contours_active()
        try:
            if self._contour_layer_id and contours_active:
                target_canvas = state.canvas
                background = state._background

                if target_canvas is not None and background is not None:
                    target_canvas.restore_region(background)
                else:
                    self._draw_canvas_with_image(state)
                if state.ax is not None and state.im is not None:
                    state.ax.draw_artist(state.im)

                manager = ContourManager.instance()
                layer = manager._layers.get(self._contour_layer_id)
                if layer:
                    artists = layer.get_generated_artists()
                    for artist in artists:
                        if hasattr(artist, 'axes') and artist.axes is not None:
                            artist.axes.draw_artist(artist)
                    overlay_artists = layer.get_overlay_artists()
                    for artist in overlay_artists:
                        if hasattr(artist, 'axes') and artist.axes is not None:
                            artist.axes.draw_artist(artist)
                self._draw_axis_foreground(state)
            else:
                # Contours inactive: avoid full draw when possible.
                if state.canvas is not None:
                    self._fast_blit_image_and_overlay()
                    if state._background is None:
                        state.canvas.draw()

            if state.canvas:
                state.update_background(state.copy_overlay_background())

        finally:
            for artist, visible in vis_states.items():
                if artist:
                    try:
                        artist.set_visible(visible)
                    except Exception as e:
                        print(f"Warning: Could not restore artist {artist}: {e}")

            if hasattr(self, 'region_manager') and hidden_regions:
                self.region_manager.restore_after_background_capture(hidden_regions)

        try:
            if plane == 'xy':
                main_window = getattr(self, 'parent', self)
                if hasattr(main_window, 'redraw_main_overlay_and_blit'):
                    main_window.redraw_main_overlay_and_blit()
                else:
                    self.redraw_main_overlay_and_blit()
            elif state.canvas:
                state.overlay_ax.draw_artist(state.hline)
                state.overlay_ax.draw_artist(state.vline)
                cpoint = getattr(state, "cpoint", None)
                if cpoint is not None and cpoint.get_visible():
                    state.overlay_ax.draw_artist(cpoint)
                state.overlay_ax.draw_artist(state.chlabel)
                state.canvas.blit(state.overlay_ax.bbox)
                self._blit_colorbar_foreground_for_state(state, force=False)

        except Exception as e:
            print(f"Warning: Error during final blit after contour update: {e}")
        self._perf_end(perf_token)

    def update_coordinates(self, x, y, z):
        self.world_x, self.world_y, self.world_z = x, y, z
        self.coordinate_updated.emit(float(x), float(y), float(z))
