from astropy.wcs import WCS
import numpy as np
import os
import time
import math
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import QMainWindow, QWidget, QGridLayout, QSlider, QLineEdit, QPushButton, QLabel, QSizePolicy, QMessageBox
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from ui.navigation_toolbar import MyNavigationToolbar
from core.coordinate import CoordinateConverter
from core.common import Common
from core.config import ConfigManager
from core.contour_manager import ContourManager, ContourItem
from ui.display_map import DisplayMap
from core.coordinate import Format_pix_to_wcs
from matplotlib.figure import Figure
from logic.add_hpbw import AddHPBW
from tools.color_scale import ColorSettingsPanel
from astropy import units as u
from astropy.coordinates import Angle
from core.region_manager import RegionManager
from core.marker_manager import MarkerManager

class FITSViewer(QMainWindow):
    data = None
    header = None
    wcs = None
    instance_initialized = False
    wcs_check_initialized = False
    velocity_unit_converted = False
    def __init__(self, data, header, wcs=None, filename="", spectral_metadata=None):
        super().__init__()
        self.data = data
        self.header = header
        self.wcs = wcs
        self.spectral_metadata = spectral_metadata if isinstance(spectral_metadata, dict) else {}
        self.region_manager = RegionManager(self)
        try:
            self.region_manager.selected_region_changed.connect(self._on_region_selection_changed)
        except Exception:
            pass
        self.marker_manager = MarkerManager(self)
        self.marker_mode_enabled = False
        self.marker_panel = None
        self.cutout_dialog = None
        self._pending_region_restore = []
        self._contour_layer_id = None
        self._contour_title_connected = False
        try:
            from ui.subwindow import SubWindow_control
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
        
        naxis = self.wcs.wcs.naxis
        
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
                    #if wcs_unit is not None:
                    #    self.wcs.wcs.cdelt[wcs_axis_idx] = (self.wcs.wcs.cdelt[wcs_axis_idx] * wcs_unit).to(u.km / u.s).value
                    #    self.wcs.wcs.crval[wcs_axis_idx] = (self.wcs.wcs.crval[wcs_axis_idx] * wcs_unit).to(u.km / u.s).value
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
                        print(f"\033[1;36mConverted velocity unit from m/s to km/s.\033[0m")
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
                    #print(f"Converted CDELT{i + 1} and CRVAL{i + 1} to km/s: {self.wcs.wcs.cdelt[i]}, {self.wcs.wcs.crval[i]}")
        
        
        self.original_data = np.array(data, copy=False)
        self._blank_planes = {}
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
        
        config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config')
        os.makedirs(config_dir, exist_ok=True)
        config_file = os.path.join(config_dir, 'config.yaml')
        self.config_manager = ConfigManager(config_file)
        config = self.config_manager.config
        
        self.setWindowTitle(windowtitle)
        self.figure_pos_x = config.get('figure_pos_x', 100)
        self.figure_pos_y = config.get('figure_pos_y', 100)
        self.figure_width = config.get('figure_width', 640)
        self.figure_height = config.get('figure_height', 640)
        self.setGeometry(self.figure_pos_x, self.figure_pos_y, self.figure_width, self.figure_height)


        self.click_label_color = config.get('click_label_color', 'grey')
        self.click_linewidth = config.get('click_linewidth', 0.25)
        self.click_linecolor = config.get('click_linecolor', 'cyan')
        self.decimal = config.get('decimal', True)
        self.number_decimals = config.get('number_decimals', 6)
        self.coord_wrap = config.get('coord_wrap', 180)
        self.scrollspeed = config.get('scrollspeed', 0.1)
        
        self.poslabel_x = config.get('poslabel_x', 0.75)
        self.poslabel_y = config.get('poslabel_y', 0.9)
        self.poslabel_w = config.get('poslabel_w', 250)
        self.poslabel_h = config.get('poslabel_h', 30)
        
        self.beam_facecolor = config.get('beam_facecolor', 'white')
        self.beam_edgecolor = config.get('beam_edgecolor', 'None')
        self.beam_linewidth = config.get('beam_linewidth', 0)
        self.beam_pos_x = config.get('beam_pox_x', 0.1)
        self.beam_pos_y = config.get('beam_pox_y', 0.1)
        
        
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
        from tools.cutout import CutoutSettingsDialog

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
            return int(Common.zpix)
        elif self.plane == 'xz':
            return int(Common.ypix)
        elif self.plane == 'zy':
            return int(Common.xpix)
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
        xstr = ("{:>.%ds}" % (self.number_decimals+6)).format(xstr)
        ystr = ("{:>.%ds}" % (self.number_decimals+6)).format(ystr)
        if self.plane == 'xy': return f'x={xstr}, y={ystr}'
        elif  self.plane == 'xz': return f'x={xstr}, z={ystr}'
        elif  self.plane == 'zy': return f'z={xstr}, y={ystr}'

    def on_motion_redirect(self, event):
        if event.inaxes == self.displaymap.overlay_ax:
            event.inaxes = self.ax
        

    def update_overlay_position(self, event):
        canvas = getattr(event, 'canvas', None)
        if canvas is not None and canvas is not self.canvas:
            return
        if not getattr(self, '_overlay_updates_enabled', True):
            return
        if getattr(self, '_updating_overlay', False):
            return
        self._updating_overlay = True
        
        self.overlay_ax.set_position(self.ax.get_position())

        hidden_regions = []
        hidden_markers = []
        region_manager = getattr(self, 'region_manager', None)
        if region_manager is not None:
            hidden_regions = region_manager.prepare_for_background_capture()
        marker_manager = getattr(self, 'marker_manager', None)
        if marker_manager is not None:
            hidden_markers = marker_manager.prepare_for_background_capture(self.plane)

        if self.plane == 'xy':
            vline_visible = Common.xy_vline.get_visible()
            hline_visible = Common.xy_hline.get_visible()
            Common.xy_vline.set_visible(False)
            Common.xy_hline.set_visible(False)
            Common._background_xy = Common.canvas_xy.copy_from_bbox(Common.overlay_ax_xy.bbox)
        elif self.plane == 'xz':
            vline_visible = Common.xz_vline.get_visible()
            hline_visible = Common.xz_hline.get_visible()
            Common.xz_vline.set_visible(False)
            Common.xz_hline.set_visible(False)
            Common._background_xz = Common.canvas_xz.copy_from_bbox(Common.overlay_ax_xz.bbox)
        elif self.plane == 'zy':
            vline_visible = Common.zy_vline.get_visible()
            hline_visible = Common.zy_hline.get_visible()
            Common.zy_vline.set_visible(False)
            Common.zy_hline.set_visible(False)
            Common._background_zy = Common.canvas_zy.copy_from_bbox(Common.overlay_ax_zy.bbox)
        
        ### update cursor lines ###
        if self.plane == 'xy':
            Common.xy_vline.set_visible(vline_visible)
            Common.xy_hline.set_visible(hline_visible)
            Common.xy_vline.set_xdata([Common.xpix])
            Common.xy_hline.set_ydata([Common.ypix])
            Common.overlay_ax_xy.draw_artist(Common.xy_vline)
            Common.overlay_ax_xy.draw_artist(Common.xy_hline)
        elif self.plane == 'xz':
            Common.xz_vline.set_visible(vline_visible)
            Common.xz_hline.set_visible(hline_visible)
            Common.xz_vline.set_xdata([Common.xpix])
            Common.xz_hline.set_ydata([Common.zpix])
            Common.overlay_ax_xz.draw_artist(Common.xz_vline)
            Common.overlay_ax_xz.draw_artist(Common.xz_hline)
        elif self.plane == 'zy':
            Common.zy_vline.set_visible(vline_visible)
            Common.zy_hline.set_visible(hline_visible)
            Common.zy_vline.set_xdata([Common.zpix])
            Common.zy_hline.set_ydata([Common.ypix])
            Common.overlay_ax_zy.draw_artist(Common.zy_vline)
            Common.overlay_ax_zy.draw_artist(Common.zy_hline)

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

        QTimer.singleShot(0, lambda: setattr(self, '_updating_overlay', False))


    def initUI(self, plane):
        self.plane = plane
        self.displaymap = DisplayMap(self.data, self.header, self.wcs, self.config_manager.config)
        self.im, self.ax = self.displaymap.display(self.fig, plane)
        self.ax.format_coord = self.formatter
        self.overlay_ax = self.displaymap.overlay_ax
        self.fig.canvas.mpl_connect("draw_event", self.update_overlay_position)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion_redirect)
        
        Common.update_im(self.plane, self.im)
        Common.update_ax(self.plane, self.ax)
        Common.update_overlay_ax(self.plane, self.overlay_ax)
        #Common.update_colorbar(self.plane, self.colorbar)
        #Common.update_cax(self.plane, self.cax)
        
        
        self.format_pix = Format_pix_to_wcs(self.wcs, self.displaymap.slices, self.ax, self.plane, self.decimal, self.number_decimals, self.coord_wrap)
        self.format_pix.convert(self.plane, 0, 0)
        

        self.canvas = FigureCanvas(self.fig)
        self.canvas.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas.updateGeometry()
        Common.update_canvas(self.plane, self.canvas)
        self._overlay_updates_enabled = True

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
            layout.addWidget(self.integ_button, 1, 17, 1, 3, alignment=Qt.AlignmentFlag.AlignRight)
            self.spec_button = QPushButton('Spec', self)
            layout.addWidget(self.spec_button, 1, 11, 1, 3, alignment=Qt.AlignmentFlag.AlignRight)

            self.slider.sliderPressed.connect(self.clicked_slider)
            self.chval_box.returnPressed.connect(self.get_chval)
            Common.update_slider(self.plane, self.slider)
            Common.update_chval_box(self.plane, self.chval_box)
        else:
            # No depth axis (2D): still place Spec/Integ buttons but disabled
            self.integ_button = QPushButton('Integ', self)
            self.spec_button = QPushButton('Spec', self)
            self.integ_button.setEnabled(False)
            self.spec_button.setEnabled(False)
            # Keep layout positions consistent with 3D case
            layout.addWidget(self.spec_button, 1, 11, 1, 3, alignment=Qt.AlignmentFlag.AlignRight)
            layout.addWidget(self.integ_button, 1, 17, 1, 3, alignment=Qt.AlignmentFlag.AlignRight)

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
            
            Common.update_xrange_input(self.plane, self.x_min_input, self.x_max_input)
    
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
    
            Common.update_yrange_input(self.plane, self.y_min_input, self.y_max_input)
    
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

            Common.update_zrange_input(self.plane, self.z_min_input, self.z_max_input)

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

        layout.addWidget(self.full_button,  1, 4, 1, 3)
        layout.addWidget(self.canvas, 2, 0, 1, 21)
        layout.addWidget(self.toolbar, 3, 0, 1, 21)
        self.setCentralWidget(central_widget)
        
        
        self.layout = layout
        
        self.label = QLabel(self)
        self.label.setStyleSheet("QLabel { color : %s; }" % self.click_label_color)
        self.label.setAlignment(Qt.AlignmentFlag.AlignRight)
        Common.update_poslabel(self.plane, self.label)
        
        """
        self.label2 = QLabel(self)
        self.label2.setStyleSheet("QLabel { color : %s; }" % self.ch_label_color)
        self.label2.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        #self.label2.setAlignment(Qt.AlignmentFlag.AlignBottom)
        font = QFont(self.ch_label_font, self.ch_label_size)
        self.label2.setFont(font)
        """
        
        self.hpbw = AddHPBW(Common.overlay_ax_xy, self.header, self.config_manager.config)
        self.ch_label = self.overlay_ax.text(self.pos_chlabel_x, self.pos_chlabel_y, "", 
                            transform=self.ax.transAxes, verticalalignment='bottom', horizontalalignment='right', 
                            fontsize=self.ch_label_size, fontfamily=self.ch_label_font, color=self.ch_label_color)
        Common.update_chlabel(self.plane, self.ch_label)
        Common.update_hpbw(self.plane, self.hpbw)
        
        
        self.show()
        self.label.setGeometry(int(self.canvas.width()*self.poslabel_x - self.poslabel_w/2), int(self.canvas.height() - self.canvas.height()*self.poslabel_y + self.poslabel_h/2) , self.poslabel_w, self.poslabel_h)
        #self.position_label_relative_to_axis(self.label2, self.pos_chlabel_x, self.pos_chlabel_y, self.pos_chlabel_w, self.pos_chlabel_h)

        # Capture baseline pixel limits for later full-range resets.
        self.original_xlim = self.ax.get_xlim()
        self.original_ylim = self.ax.get_ylim()
        if self.data.ndim > 2:
            depth_len = self.data.shape[self.data.ndim - 3]
            self.original_zlim = (-0.5, depth_len - 0.5)
        else:
            self.original_zlim = (0.0, 0.0)

        
        if plane == 'xy':
            self.hline = Common.overlay_ax_xy.axhline(y=0, visible = False, lw = self.click_linewidth, c = self.click_linecolor, animated=True)
            self.vline = Common.overlay_ax_xy.axvline(x=0, visible = False, lw = self.click_linewidth, c = self.click_linecolor, animated=True)
            self._background = Common.copy_overlay_background('xy')
        elif plane == 'xz':
            self.hline = Common.overlay_ax_xz.axhline(y=0, visible = False, lw = self.click_linewidth, c = self.click_linecolor, animated=True)
            self.vline = Common.overlay_ax_xz.axvline(x=0, visible = False, lw = self.click_linewidth, c = self.click_linecolor, animated=True)
            self._background = Common.copy_overlay_background('xz')
        elif plane == 'zy':
            self.hline = Common.overlay_ax_zy.axhline(y=0, visible = False, lw = self.click_linewidth, c = self.click_linecolor, animated=True)
            self.vline = Common.overlay_ax_zy.axvline(x=0, visible = False, lw = self.click_linewidth, c = self.click_linecolor, animated=True)
            self._background = Common.copy_overlay_background('zy')

        Common.update_lines(self.plane, self.hline, self.vline)
        Common.update_background(self.plane, self._background)
        
        #self.add_marker = AddMarker(self.overlay_ax, self.canvas)
        
    
    def reload_viewer(self):
        """Reload the viewer based on the updated configuration settings for all windows."""
        config = self.config_manager.config
        self.format_pix.decimal = config.get('decimal')
        self.format_pix.number_decimals = config.get('number_decimals')
        self.format_pix.coord_wrap = config.get('coord_wrap')
        self.scrollspeed = config.get('scrollspeed')
        self.poslabel_x = config.get('poslabel_x')
        self.poslabel_y = config.get('poslabel_y')
        self.poslabel_w = config.get('poslabel_w')
        self.poslabel_h = config.get('poslabel_h')
        self.click_label_color = config.get('click_label_color')
        self.label.setStyleSheet("QLabel { color : %s; }" % self.click_label_color)
        
        self.pos_chlabel_x = config.get('pos_chlabel_x')
        self.pos_chlabel_y = config.get('pos_chlabel_y')
        #self.pos_chlabel_w = config.get('pos_chlabel_w')
        #self.pos_chlabel_h = config.get('pos_chlabel_h')
        
        self.ch_label_color = config.get('ch_label_color')
        self.ch_label_size = config.get('ch_label_size')
        self.ch_label_font = config.get('ch_label_font')
        #self.label2.setStyleSheet("QLabel { color : %s; }" % self.ch_label_color)
        #font = QFont(self.ch_label_font, self.ch_label_size)
        #self.label2.setFont(font)
        
        self.beam_facecolor = config.get('beam_facecolor')
        self.beam_edgecolor = config.get('beam_edgecolor')
        self.beam_linewidth = config.get('beam_linewidth')
        self.beam_pos_x = config.get('beam_pox_x')
        self.beam_pos_y = config.get('beam_pox_y')
        
        
        self.resize(config.get('figure_width'), config.get('figure_height'))
        #self.move(config.get('figure_pos_x'), config.get('figure_pos_y')) 
        self.fig.subplots_adjust( left = config.get('ax_pos_l'), 
                                right = config.get('ax_pos_r'),
                                bottom = config.get('ax_pos_b'), 
                                top = config.get('ax_pos_t'))
        
        if len(self.subwindows) > 1:
            for subwindow in self.subwindows:
                subwindow.fig.subplots_adjust( left = config.get('ax_pos_l'), 
                                right = config.get('ax_pos_r'),
                                bottom = config.get('ax_pos_b'), 
                                top = config.get('ax_pos_t'))
                subwindow.resize(config.get('figure_width'), config.get('figure_height'))
                #subwindow.move(config.get('figure_pos_x'), config.get('figure_pos_y')) 
                subwindow.format_pix.decimal = config.get('decimal')
                subwindow.format_pix.number_decimals = config.get('number_decimals')
                subwindow.format_pix.coord_wrap = config.get('coord_wrap')
                subwindow.scrollspeed = config.get('scrollspeed')
                subwindow.poslabel_x = config.get('poslabel_x')
                subwindow.poslabel_y = config.get('poslabel_y')
                subwindow.poslabel_w = config.get('poslabel_w')
                subwindow.poslabel_h = config.get('poslabel_h')
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
                
        self._apply_config_to_plane()

    def _apply_config_to_plane(self):
        """Apply configuration settings to a specific plane."""
        config = self.config_manager.config
        if self.xlabel is None:
            self.xlabel = Common.ax_coord_xy[0].get_axislabel()
            self.ylabel = Common.ax_coord_xy[1].get_axislabel()
            if self.data.ndim > 2: self.zlabel = Common.ax_coord_xz[1].get_axislabel()        
            
        Common.fig_xy.set_facecolor(config.get('fig_background_color'))
        Common.ax_xy.set_facecolor(config.get('ax_background_color'))
        Common.im_xy.cmap.set_bad(config.get('bad_color'))
        
        xtick_label_position = config.get('xticklabel_position')
        ytick_label_position = config.get('yticklabel_position')
        
        Common.ax_coord_xy[0].set_axislabel(self.xlabel, fontsize=config.get('axislabel_fontsize'),
                           fontfamily=config.get('axislabel_fontfamily'),
                           color=config.get('axislabel_color'))
        Common.ax_coord_xy[0].set_axislabel_position(xtick_label_position)
        Common.ax_coord_xy[1].set_axislabel(self.ylabel, fontsize=config.get('axislabel_fontsize'),
                           fontfamily=config.get('axislabel_fontfamily'),
                           color=config.get('axislabel_color'))
        Common.ax_coord_xy[1].set_axislabel_position(ytick_label_position)
        Common.ax_xy.tick_params(axis='both', which = 'major', direction=config.get('tick_direction'), length=config.get('tick_length'),
                                color=config.get('tick_color'), width = config.get('tick_width'), labelsize = config.get('tick_labelsize'),
                                labelcolor = config.get('tick_labelcolor'))
        for spine in Common.ax_xy.spines.values():
                spine.set_linewidth(config.get('tick_width'))
                spine.set_color(config.get('tick_color'))
        Common.ax_xy.tick_params(which = 'minor', length=config.get('mtick_length'))
        Common.ax_coord_xy[0].set_ticklabel(rotation = config.get('tick_xlabelrotation'), pad = config.get('tick_pad_x'), ha='right', va='top')
        Common.ax_coord_xy[0].set_ticklabel_position(xtick_label_position)
        Common.ax_coord_xy[0].set_ticks_position(config.get('default_ticks_position'))
        Common.ax_coord_xy[1].set_ticklabel(rotation = config.get('tick_ylabelrotation'), pad = config.get('tick_pad_y'), ha='center', va='top')
        Common.ax_coord_xy[1].set_ticklabel_position(ytick_label_position)
        Common.ax_coord_xy[1].set_ticks_position(config.get('default_ticks_position'))
        Common.ax_coord_xy[0].set_minor_frequency(config.get('x_mtick_freq', 5))
        Common.ax_coord_xy[1].set_minor_frequency(config.get('y_mtick_freq', 5))
        
        Common.chlabel_xy.set_position((self.pos_chlabel_x, self.pos_chlabel_y))
        Common.chlabel_xy.set_fontsize(self.ch_label_size)
        Common.chlabel_xy.set_fontfamily(self.ch_label_font)
        Common.chlabel_xy.set_color(self.ch_label_color)
        
        Common.xy_hline.set_color(config.get('click_linecolor'))
        Common.xy_vline.set_color(config.get('click_linecolor'))
        Common.xy_hline.set_linewidth(config.get('click_linewidth'))
        Common.xy_vline.set_linewidth(config.get('click_linewidth'))
        Common.overlay_ax_xy.draw_artist(Common.xy_hline)
        Common.overlay_ax_xy.draw_artist(Common.xy_vline)
        Common.overlay_ax_xy.draw_artist(Common.chlabel_xy)
        
        #Common.hpbw_xy.ellipse.remove()
        #Common.hpbw_xy.ellipse = None
        #Common.hpbw_xy = AddHPBW(Common.ax_xy, self.header, config)
        #Common.hpbw_xy.ellipse.set_facecolor(self.beam_facecolor)
        #Common.hpbw_xy.ellipse.set_edgecolor(self.beam_edgecolor)
        #Common.hpbw_xy.ellipse.set_linewidth(self.beam_linewidth)
        #Common.hpbw_xy.create_beam()
        #Common.hpbw_xy.update_position()
        
        #Common.canvas_xy.draw_idle()
        
        if self.data.ndim > 2:
            Common.fig_xz.set_facecolor(config.get('fig_background_color'))
            Common.ax_xz.set_facecolor(config.get('ax_background_color'))
            Common.fig_zy.set_facecolor(config.get('fig_background_color'))
            Common.ax_zy.set_facecolor(config.get('ax_background_color'))
        
            Common.xz_hline.set_color(config.get('click_linecolor'))
            Common.xz_vline.set_color(config.get('click_linecolor'))
            Common.xz_hline.set_linewidth(config.get('click_linewidth'))
            Common.xz_vline.set_linewidth(config.get('click_linewidth'))
            
            Common.zy_hline.set_color(config.get('click_linecolor'))
            Common.zy_vline.set_color(config.get('click_linecolor'))
            Common.zy_hline.set_linewidth(config.get('click_linewidth'))
            Common.zy_vline.set_linewidth(config.get('click_linewidth'))

            Common.overlay_ax_xz.draw_artist(Common.xz_hline)
            Common.overlay_ax_xz.draw_artist(Common.xz_vline)
            Common.overlay_ax_zy.draw_artist(Common.zy_hline)
            Common.overlay_ax_zy.draw_artist(Common.zy_vline)
            
            Common.im_xz.cmap.set_bad(color=config.get('bad_color'))
            Common.im_xz.cmap.set_bad(color=config.get('bad_color'))
            
            
            Common.ax_coord_xz[0].set_axislabel(self.xlabel, fontsize=config.get('axislabel_fontsize'),
                fontfamily=config.get('axislabel_fontfamily'),
                color=config.get('axislabel_color'))
            Common.ax_coord_xz[0].set_axislabel_position(xtick_label_position)
            Common.ax_coord_xz[1].set_axislabel(self.zlabel, fontsize=config.get('axislabel_fontsize'),
                fontfamily=config.get('axislabel_fontfamily'),
                color=config.get('axislabel_color'))
            Common.ax_coord_xz[1].set_axislabel_position(ytick_label_position)
            Common.ax_xz.tick_params(axis='both', which = 'major', direction=config.get('tick_direction'), length=config.get('tick_length'),
                        color=config.get('tick_color'), width = config.get('tick_width'), labelsize = config.get('tick_labelsize'),
                        labelcolor = config.get('tick_labelcolor'))
            for spine in Common.ax_xz.spines.values():
                spine.set_linewidth(config.get('tick_width'))
                spine.set_color(config.get('tick_color'))
            Common.ax_xz.tick_params(which = 'minor', length=config.get('mtick_length'))
            Common.ax_coord_xz[0].set_ticklabel(rotation = config.get('tick_xlabelrotation'), pad = config.get('tick_pad_x'))
            Common.ax_coord_xz[0].set_ticklabel_position(xtick_label_position)
            Common.ax_coord_xz[0].set_ticks_position(config.get('default_ticks_position'))
            Common.ax_coord_xz[1].set_ticklabel(rotation = config.get('tick_ylabelrotation'), pad = config.get('tick_pad_y'))
            Common.ax_coord_xz[1].set_ticklabel_position(ytick_label_position)
            Common.ax_coord_xz[1].set_ticks_position(config.get('default_ticks_position'))
            Common.ax_coord_xz[0].set_minor_frequency(config.get('x_mtick_freq', 5))
            Common.ax_coord_xz[1].set_minor_frequency(config.get('z_mtick_freq', 5))
            
            #Common.canvas_xz.draw_idle()
            
            
            Common.ax_coord_zy[0].set_axislabel(self.zlabel, fontsize=config.get('axislabel_fontsize'),
                fontfamily=config.get('axislabel_fontfamily'),
                color=config.get('axislabel_color'))
            Common.ax_coord_zy[0].set_axislabel_position(xtick_label_position)
            Common.ax_coord_zy[1].set_axislabel(self.ylabel, fontsize=config.get('axislabel_fontsize'),
                fontfamily=config.get('axislabel_fontfamily'),
                color=config.get('axislabel_color'))
            Common.ax_coord_zy[1].set_axislabel_position(ytick_label_position)
            Common.ax_zy.tick_params(axis='both', which = 'major', direction=config.get('tick_direction'), length=config.get('tick_length'),
                        color=config.get('tick_color'), width = config.get('tick_width'), labelsize = config.get('tick_labelsize'),
                        labelcolor = config.get('tick_labelcolor'))
            for spine in Common.ax_zy.spines.values():
                spine.set_linewidth(config.get('tick_width'))
                spine.set_color(config.get('tick_color'))
            Common.ax_zy.tick_params(which = 'minor', length=config.get('mtick_length'))
            Common.ax_coord_zy[0].set_ticklabel(rotation = config.get('tick_xlabelrotation'), pad = config.get('tick_pad_x'))
            Common.ax_coord_zy[0].set_ticklabel_position(xtick_label_position)
            Common.ax_coord_zy[0].set_ticks_position(config.get('default_ticks_position'))
            Common.ax_coord_zy[1].set_ticklabel(rotation = config.get('tick_ylabelrotation'), pad = config.get('tick_pad_y'))
            Common.ax_coord_zy[1].set_ticklabel_position(ytick_label_position)
            Common.ax_coord_zy[1].set_ticks_position(config.get('default_ticks_position'))
            Common.ax_coord_zy[0].set_minor_frequency(config.get('z_mtick_freq', 5))
            Common.ax_coord_zy[1].set_minor_frequency(config.get('y_mtick_freq', 5))
            
            #Common.canvas_zy.draw_idle()

            Common.chlabel_xz.set_position((self.pos_chlabel_x, self.pos_chlabel_y))
            Common.chlabel_xz.set_fontsize(self.ch_label_size)
            Common.chlabel_xz.set_fontfamily(self.ch_label_font)
            Common.chlabel_xz.set_color(self.ch_label_color)
            Common.chlabel_zy.set_position((self.pos_chlabel_x, self.pos_chlabel_y))
            Common.chlabel_zy.set_fontsize(self.ch_label_size)
            Common.chlabel_zy.set_fontfamily(self.ch_label_font)
            Common.chlabel_zy.set_color(self.ch_label_color)
            Common.overlay_ax_xz.draw_artist(Common.chlabel_xz)
            Common.overlay_ax_zy.draw_artist(Common.chlabel_zy)
            
        if Common.colorbar_xy: 
            Common.colorbar_xy.remove()
            Common.cax_xy = Common.fig_xy.add_axes([config.get('cbar_pos_x'), config.get('cbar_pos_y'), config.get('cbar_width'), config.get('cbar_height')])
            Common.colorbar_xy = Common.fig_xy.colorbar(Common.im_xy, cax = Common.cax_xy, orientation = config.get('colorbar_orientation') )
        
        if Common.colorbar_xz: 
            Common.colorbar_xz.remove()
            Common.cax_xz = Common.fig_xz.add_axes([config.get('cbar_pos_x'), config.get('cbar_pos_y'), config.get('cbar_width'), config.get('cbar_height')])
            Common.colorbar_xz = Common.fig_xz.colorbar(Common.im_xz, cax = Common.cax_xz, orientation = config.get('colorbar_orientation') )
        
        if Common.colorbar_zy: 
            Common.colorbar_zy.remove()
            Common.cax_zy = Common.fig_zy.add_axes([config.get('cbar_pos_x'), config.get('cbar_pos_y'), config.get('cbar_width'), config.get('cbar_height')])
            Common.colorbar_zy = Common.fig_zy.colorbar(Common.im_zy, cax = Common.cax_zy, orientation = config.get('colorbar_orientation') )
        
        color_settings = ColorSettingsPanel(fits_viewer = self, subwindows = self.subwindows)
        color_settings.apply_colorbar_settings_to_all(config)

        # Re-apply the decimal or dms format to the tick labels for all planes
        is_decimal = config.get('decimal', True)
        
        all_axes = [(Common.ax_xy, 'xy'), (Common.ax_xz, 'xz'), (Common.ax_zy, 'zy')]
        
        for ax, plane in all_axes:
            if ax is None:
                continue

            try:
                # Iterate through the coordinates of the axis (e.g., 'glon', 'glat')
                for i in range(ax.wcs.naxis):
                    coord = ax.coords[i]
                    axis_type = self.wcs.world_axis_physical_types[i]
                    
                    # Apply formatting only to angular coordinates
                    if axis_type and ('lon' in axis_type or 'lat' in axis_type):
                        unit = coord.get_format_unit()
                        coord.set_format_unit(unit, decimal=is_decimal)

                # Redraw the canvas to show the changes
                if plane == 'xy' and Common.canvas_xy:
                    Common.canvas_xy.draw_idle()
                elif plane == 'xz' and Common.canvas_xz:
                    Common.canvas_xz.draw_idle()
                elif plane == 'zy' and Common.canvas_zy:
                    Common.canvas_zy.draw_idle()

            except Exception as e:
                # In case of any error, print it and continue
                print(f"Could not update tick format for plane {plane}: {e}")

        self._register_contour_layer()


    def forward_ch(self):
        self.slider.setValue(self.slider.value()+1)
        
    def backward_ch(self):
        self.slider.setValue(self.slider.value()-1)
    
    def get_chval(self):
        chval = self.chval_box.text()
        if self.plane == 'xy':
            x = Common.world_x
            y = Common.world_y
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
                #print("\n\nValueError: invalid value provided")
        
        elif self.plane == 'xz':
            x = Common.world_x
            z = Common.world_z
            try:
                if self.decimal == False: 
                    y = float(Angle(chval).to_string(unit = u.deg, decimal=True))
                else: y = float(chval)
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
                #print("\n\nValueError: invalid value provided")

        elif self.plane == 'zy':
            z = Common.world_z
            y = Common.world_y
            try:
                if self.decimal == False: 
                    x = float(Angle(chval).to_string(unit = u.deg, decimal=True))
                else: x = float(chval)
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
                #print("\n\nValueError: invalid value provided")
        
            
    def clicked_slider(self):
        if self.plane == 'xy': 
            Common.clicked_xy = True
            Common.clicked_xz = False
            Common.clicked_zy = False
        elif self.plane == 'xz': 
            Common.clicked_xy = False
            Common.clicked_xz = True
            Common.clicked_zy = False
        elif self.plane == 'zy': 
            Common.clicked_xy = False
            Common.clicked_xz = False
            Common.clicked_zy = True

    
    def scroll_slider_mpl(self, event):
        if self.plane == 'xy': 
            k = Common.slider_xy.value()
            k_max = self.header['NAXIS3']-1
            self.scroll_accumulation += event.step * self.scrollspeed
            if abs(self.scroll_accumulation) >= 1:
                step = int(self.scroll_accumulation)
                k = k - step
                k = max(0, min(k, k_max))
                Common.clicked_xy = False
                Common.slider_xy.setValue(k)
                self.scroll_accumulation -= step
    
        elif self.plane == 'xz': 
            k = Common.slider_xz.value()
            k_max = self.header['NAXIS2']-1
            self.scroll_accumulation += event.step * self.scrollspeed
            if abs(self.scroll_accumulation) >= 1:
                step = int(self.scroll_accumulation)
                k = k - step
                k = max(0, min(k, k_max))
                Common.clicked_xz = False
                Common.slider_xz.setValue(k)
                self.scroll_accumulation -= step
    
        elif self.plane == 'zy': 
            k = Common.slider_zy.value()
            k_max = self.header['NAXIS1']-1
            self.scroll_accumulation += event.step * self.scrollspeed
            if abs(self.scroll_accumulation) >= 1:
                step = int(self.scroll_accumulation)
                k = k - step
                k = max(0, min(k, k_max))
                Common.clicked_zy = False
                Common.slider_zy.setValue(k)
                self.scroll_accumulation -= step
    
    def scroll_slider(self, event):
        k=0
        if self.plane == 'xy': 
            Common.clicked_xy = False
            k = Common.slider_xy.value()
            Common.zpix = k
        elif self.plane == 'xz':
            Common.clicked_xz = False
            k = Common.slider_xz.value()
            Common.ypix = k
        elif self.plane == 'zy': 
            Common.clicked_zy = False
            k = Common.slider_zy.value()
            Common.xpix = k
        self.update_channel(self.plane, k)
        self.current_value_label.setText(str(k+1)) #current slider (ch) value 

            
    def on_click(self, event):
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
            if hasattr(self, 'control_panel'):
                if self.control_panel.pvd_panel is not None:
                    return
            self.region_manager.draw_regions_for_blit()

            x, y = self.ax.transData.inverted().transform((event.x, event.y))
            xstr, ystr = self.format_pix.convert(self.plane, x, y)
            if self.data.ndim > 2:
                if self.plane == 'xy': 
                    Common.clicked_xy = True
                    if  Common.slider_xy: Common.zpix = Common.slider_xy.value()
                    Common.world_z = self.format_pix.convert_chpix_to_world(self.plane, x, y, Common.zpix)
                    Common.world_z_str = self.format_pix.convert_chval_to_world_str(self.plane, Common.world_z)
                    i, j, k = int(round(x)), int(round(y)), Common.zpix
                    Common.update_world_xyz(x, y, Common.world_z)
                    Common.update_world_xyz_str(xstr, ystr, Common.world_z_str)
                    try: intensity = self.cube[k,j,i]
                    except: return
                    
                    
                elif self.plane == 'xz':
                    Common.clicked_xz = True
                    Common.ypix = Common.slider_xz.value()                
                    Common.world_y = self.format_pix.convert_chpix_to_world(self.plane, x, Common.ypix, y)
                    Common.world_y_str = self.format_pix.convert_chval_to_world_str(self.plane, Common.world_y)
                    i, j, k = int(round(x)), Common.ypix, int(round(y))
                    Common.update_world_xyz(x, Common.world_y, y)
                    Common.update_world_xyz_str(xstr, Common.world_y_str, ystr)
                    try: intensity = self.cube[k,j,i]
                    except: return
                    
                elif self.plane == 'zy':
                    Common.clicked_zy = True
                    Common.xpix = Common.slider_zy.value()
                    Common.world_x = self.format_pix.convert_chpix_to_world(self.plane, Common.xpix, y, x)
                    Common.world_x_str = self.format_pix.convert_chval_to_world_str(self.plane, Common.world_x)                
                    i, j, k = Common.xpix, int(round(y)), int(round(x))
                    Common.update_world_xyz(Common.world_x, y, x)
                    Common.update_world_xyz_str(Common.world_x_str, ystr, xstr)
                    try: intensity = self.cube[k,j,i]
                    except: return
                print('\r Clicked at (%s, %s, %s)              \n Intensity = %s %s            \033[1A'  % (Common.world_x_str, Common.world_y_str, Common.world_z_str, self._format_significant_digits(intensity, 4), self.bunit), end = '')
                
            elif self.data.ndim == 2: 
                i, j = int(round(x)), int(round(y))
                try: intensity =  self.data[j,i]
                except: return
                print('\r Clicked at (%s, %s)             \n%s %s          \033[1A'  % (Common.world_x_str, Common.world_y_str, self._format_significant_digits(intensity, 4), self.bunit), end = '')
                
            intensity = self._format_significant_digits(intensity, 4)
            self.update_clicked_pix(x,y)
            self.label.setText('%s, %s \n[%s]'  % (xstr, ystr, intensity))
            
            self.label.setGeometry(int(self.canvas.width()*self.poslabel_x - self.poslabel_w/2), int(self.canvas.height() - self.canvas.height()*self.poslabel_y + self.poslabel_h/2) , self.poslabel_w, self.poslabel_h)
            #self.label2.setGeometry(int(self.canvas.width()*self.pos_chlabel_x - self.pos_chlabel_w/2), int(self.canvas.height() - self.canvas.height()*self.pos_chlabel_y + self.pos_chlabel_h/2) , self.pos_chlabel_w, self.pos_chlabel_h)
            #self.position_label_relative_to_axis(self.label2, self.pos_chlabel_x, self.pos_chlabel_y, self.pos_chlabel_w, self.pos_chlabel_h)

        elif event.inaxes != self.ax:
            if event.dblclick:
                if Common.plabel_xy:  Common.plabel_xy.hide()
                if Common.plabel_xz:  Common.plabel_xz.hide()
                if Common.plabel_zy:  Common.plabel_zy.hide()
                if Common.chlabel_xy:  Common.chlabel_xy.set_visible(False)
                if Common.chlabel_xz:  Common.chlabel_xz.set_visible(False)
                if Common.chlabel_zy:  Common.chlabel_zy.set_visible(False)                
                if Common.xy_hline:  Common.xy_hline.set_visible(False)
                if Common.xy_vline:  Common.xy_vline.set_visible(False)
                if Common.xz_hline:  Common.xz_hline.set_visible(False)
                if Common.xz_vline:  Common.xz_vline.set_visible(False)
                if Common.zy_hline:  Common.zy_hline.set_visible(False)
                if Common.zy_vline:  Common.zy_vline.set_visible(False)
                if Common.canvas_xy: Common.canvas_xy.draw_idle()
                if Common.canvas_xz: Common.canvas_xz.draw_idle()
                if Common.canvas_zy: Common.canvas_zy.draw_idle()


    def on_release(self, event):
        """Handles mouse release events, delegating to the RegionManager if in region mode."""
        # This is primarily for finalizing region drawing.
        if hasattr(self, 'region_mode_enabled') and self.region_mode_enabled:
            if self.plane == 'xy':
                self.region_manager.handle_release(event)
        if getattr(self, 'marker_mode_enabled', False) and hasattr(self, 'marker_manager') and self.marker_manager is not None:
            self.marker_manager.handle_release(event)

    def position_label_relative_to_axis(self, label, pos_x, pos_y, label_width, label_height):
        bbox = self.ax.get_position()
        ax_x0 = self.canvas.width()*bbox.x0
        ax_x1 = self.canvas.width()*bbox.x1
        ax_y0 = self.canvas.height()*(1- bbox.y0)
        ax_y1 = self.canvas.height()*(1- bbox.y1)
        ax_width = abs(ax_x1 - ax_x0)
        ax_height = abs(ax_y1 - ax_y0)
        relative_x = int(ax_x0 + ax_width * pos_x)
        relative_y = int(ax_y0 - ax_height * pos_y + label_height + 15)
        label.setGeometry(relative_x, relative_y, label_width, label_height)
        
    def cursor_position(self, event):
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
        # Throttle the update to only happen every 0.01 seconds
        if current_time - self.last_update_time < 0.01:
            return  # Skip drawing if 0.01 seconds haven't passed since the last update
        # Update the last update time to the current time
        self.last_update_time = current_time
        #if event.inaxes != self.ax:
        #    return
        

        x, y = self.ax.transData.inverted().transform((event.x, event.y))
        if self.toolbar.mode == 'zoom rect' or self.toolbar.mode =='pan/zoom': return
        elif event.inaxes is self.ax and event.button == 1:
            if hasattr(self, 'control_panel'):
                if self.control_panel.pvd_panel is not None:
                    return
            self.region_manager.draw_regions_for_blit()

            if self.plane == 'xy': 
                # Common.canvas_xy.restore_region(Common._background_xy)
                xstr,  ystr = self.format_pix.convert(self.plane, x, y)
                i, j, k = int(round(x)), int(round(y)), int(round(Common.zpix))
                Common.update_world_xyz_str(xstr, ystr, Common.world_z_str)                
            elif self.plane == 'xz':
                # Common.canvas_xz.restore_region(Common._background_xz)
                xstr,  ystr = self.format_pix.convert(self.plane, x, y)
                i, j, k = int(round(x)), int(round(Common.ypix)), int(round(y))
                Common.update_world_xyz_str(xstr, Common.world_y_str, ystr)
            elif self.plane == 'zy':
                # Common.canvas_zy.restore_region(Common._background_zy)
                xstr,  ystr = self.format_pix.convert(self.plane, x, y)
                i, j, k = int(round(Common.xpix)), int(round(y)), int(round(x))
                Common.update_world_xyz_str(Common.world_x_str, ystr, xstr)
            #self.update_clicked_pix(x, y)
            
            
            if self.data.ndim == 2: 
                try: intensity =  self.data[j,i]
                except: return
            elif self.data.ndim > 2: 
                try: intensity = self.cube[k,j,i]
                except: return
            
            intensity = '{:.4g}'.format(intensity)
            self.update_clicked_pix(x,y)
            self.label.setText('%s, %s \n[%s]'  % (xstr, ystr, intensity))

            #self.label.setGeometry(int(self.canvas.width()-self.poslabel_x), int(self.canvas.height() - self.poslabel_y), self.poslabel_w, self.poslabel_h)
            #self.label2.setGeometry(int(self.canvas.width()-self.poslabel_x), int(self.canvas.height()- self.poslabel_y - 15), self.poslabel_w, self.poslabel_h+15)

            
    def update_clicked_pix(self, x, y):
        if getattr(self, 'marker_mode_enabled', False):
            return
        self.xpix = Common.xpix
        self.ypix = Common.ypix
        self.zpix = Common.zpix
        self.clicked_coords = (x, y)
        i, j = int(round(x)), int(round(y))
        subwindow1 = getattr(self.SubWindow, 'subwindow1', None)
        subwindow2 = getattr(self.SubWindow, 'subwindow2', None)
        sub1_visible = subwindow1 is not None and not subwindow1.isHidden()
        sub2_visible = subwindow2 is not None and not subwindow2.isHidden()

        if self.plane == 'xy': 
            self.xpix = x
            self.ypix = y
            if self.wcs.naxis > 2 and 'NAXIS3' in self.header:
                Common.xz_vline.set_xdata([x])
                Common.zy_hline.set_ydata([y])
                Common.xz_hline.set_visible(True)
                Common.zy_hline.set_visible(True)
                Common.xz_vline.set_visible(True)
                Common.zy_vline.set_visible(True)
            
                if sub1_visible:
                    Common.chlabel_xz.set_visible(True)
                    if 0 <= j < self.cube.shape[1]:
                        Common.im_xz.set_data(self.cube[:, j, :])
                        if Common.clicked_xy:
                            Common.slider_xz.setValue(j)
                        Common.ypix = j
                    else:
                        Common.im_xz.set_data(self._blank_plane('xz'))

                    Common.canvas_xz.restore_region(Common._background_xz)
                    Common.overlay_ax_xz.draw_artist(Common.xz_vline)
                    Common.overlay_ax_xz.draw_artist(Common.xz_hline)
                    Common.overlay_ax_xz.draw_artist(Common.chlabel_xz)
                    marker_mgr = getattr(self, 'marker_manager', None)
                    if marker_mgr is not None:
                        marker_mgr.draw_markers_for_blit('xz')
                    Common.canvas_xz.blit(Common.overlay_ax_xz.bbox)
                else:
                    if 0 <= j < self.cube.shape[1]:
                        Common.xpix = j

                if sub2_visible:
                    Common.chlabel_zy.set_visible(True)
                    if 0 <= i < self.cube.shape[2]:
                        Common.im_zy.set_data(self.cube[:, :, i].T)
                        if Common.clicked_xy:
                            Common.slider_zy.setValue(i)
                        Common.xpix = i
                    else:
                        Common.im_zy.set_data(self._blank_plane('zy'))

                    Common.canvas_zy.restore_region(Common._background_zy)
                    Common.overlay_ax_zy.draw_artist(Common.zy_hline)
                    Common.overlay_ax_zy.draw_artist(Common.zy_vline)
                    Common.overlay_ax_zy.draw_artist(Common.chlabel_zy)
                    marker_mgr = getattr(self, 'marker_manager', None)
                    if marker_mgr is not None:
                        marker_mgr.draw_markers_for_blit('zy')
                    Common.canvas_zy.blit(Common.overlay_ax_zy.bbox)
                
            
            #Common.canvas_xy.restore_region(Common._background_xy)
            Common.xy_vline.set_xdata([x])
            Common.xy_hline.set_ydata([y])
            Common.xy_vline.set_visible(True)
            Common.xy_hline.set_visible(True)
            Common.chlabel_xy.set_text("%s" % Common.world_z_str)
            Common.chlabel_xy.set_visible(True)

            self.redraw_main_overlay_and_blit()

        elif self.plane == 'xz' and self.header['NAXIS3'] > 1:  
            self.xpix = x
            self.zpix = y

            if self.wcs.naxis > 2:
                Common.xy_vline.set_xdata([x])
                Common.zy_vline.set_xdata([y])
                Common.zy_vline.set_visible(True)
                Common.zy_hline.set_visible(True)
                Common.xy_vline.set_visible(True)
                Common.xy_hline.set_visible(True)
                
                Common.chlabel_xy.set_visible(True)
                Common.chlabel_zy.set_visible(True)
                
                if 0 <= j < self.cube.shape[0]:
                    Common.im_xy.set_data(self.cube[j])
                    if Common.clicked_xz:
                        Common.slider_xy.setValue(j)
                    Common.zpix = j
                else:
                    Common.im_xy.set_data(self._blank_plane('xy'))

                if sub2_visible:
                    if 0 <= i < self.cube.shape[2]:
                        Common.im_zy.set_data(self.cube[:, :, i].T)
                        if Common.clicked_xz:
                            Common.slider_zy.setValue(i)
                        Common.xpix = i
                    else:
                        Common.im_zy.set_data(self._blank_plane('zy'))

                if sub2_visible:
                    Common.canvas_zy.restore_region(Common._background_zy)
                    Common.overlay_ax_zy.draw_artist(Common.zy_vline)
                    Common.overlay_ax_zy.draw_artist(Common.zy_hline)
                    Common.overlay_ax_zy.draw_artist(Common.chlabel_zy)
                    marker_mgr = getattr(self, 'marker_manager', None)
                    if marker_mgr is not None:
                        marker_mgr.draw_markers_for_blit('zy')
                    Common.canvas_zy.blit(Common.overlay_ax_zy.bbox)

                Common.main_window.redraw_main_overlay_and_blit()



            Common.canvas_xz.restore_region(Common._background_xz)
            Common.xz_vline.set_xdata([x])
            Common.xz_hline.set_ydata([y])                
            Common.xz_vline.set_visible(True)
            Common.xz_hline.set_visible(True) 

            Common.chlabel_xz.set_visible(True)

            Common.overlay_ax_xz.draw_artist(Common.xz_vline)
            Common.overlay_ax_xz.draw_artist(Common.xz_hline)
            Common.overlay_ax_xz.draw_artist(Common.chlabel_xz)
            marker_mgr = getattr(self, 'marker_manager', None)
            if marker_mgr is not None:
                marker_mgr.draw_markers_for_blit('xz')

            Common.canvas_xz.blit(Common.overlay_ax_xz.bbox)
            
                
        elif self.plane == 'zy' and self.header['NAXIS3']> 1:
            self.zpix = x
            self.ypix = y

            if self.wcs.naxis > 2:
                Common.xz_hline.set_ydata([x])
                Common.xy_hline.set_ydata([y])
                Common.xz_hline.set_visible(True)
                Common.xy_hline.set_visible(True)
                Common.xz_vline.set_visible(True)
                Common.xy_vline.set_visible(True)
                
                Common.chlabel_xz.set_visible(True)
                Common.chlabel_xy.set_visible(True)
                
                
                if 0 <= i < self.cube.shape[0]:
                    Common.im_xy.set_data(self.cube[i])
                    if Common.clicked_zy:
                        Common.slider_xy.setValue(i)
                    Common.zpix = i
                else:
                    Common.im_xy.set_data(self._blank_plane('xy'))

                if sub1_visible:
                    if 0 <= j < self.cube.shape[1]:
                        Common.im_xz.set_data(self.cube[:, j, :])
                        if Common.clicked_zy:
                            Common.slider_xz.setValue(j)
                        Common.ypix = j
                    else:
                        Common.im_xz.set_data(self._blank_plane('xz'))
                        Common.ypix = j
                
                if sub1_visible:
                    Common.canvas_xz.restore_region(Common._background_xz)
                    Common.overlay_ax_xz.draw_artist(Common.xz_hline)
                    Common.overlay_ax_xz.draw_artist(Common.xz_vline)
                    Common.overlay_ax_xz.draw_artist(Common.chlabel_xz)
                    marker_mgr = getattr(self, 'marker_manager', None)
                    if marker_mgr is not None:
                        marker_mgr.draw_markers_for_blit('xz')
                    Common.canvas_xz.blit(Common.overlay_ax_xz.bbox)
                Common.main_window.redraw_main_overlay_and_blit()
                
            
            Common.canvas_zy.restore_region(Common._background_zy)
            Common.zy_vline.set_xdata([x])
            Common.zy_hline.set_ydata([y])                
            Common.zy_vline.set_visible(True)
            Common.zy_hline.set_visible(True)

            Common.chlabel_zy.set_visible(True)

            Common.overlay_ax_zy.draw_artist(Common.zy_vline)
            Common.overlay_ax_zy.draw_artist(Common.zy_hline)
            Common.overlay_ax_zy.draw_artist(Common.chlabel_zy)
            marker_mgr = getattr(self, 'marker_manager', None)
            if marker_mgr is not None:
                marker_mgr.draw_markers_for_blit('zy')

            Common.canvas_zy.blit(Common.overlay_ax_zy.bbox)

        Common.update_pix(self.xpix, self.ypix, self.zpix)
        
        if Common.plabel_xy: 
            Common.plabel_xy.setText('%s, %s'  % (Common.world_x_str, Common.world_y_str))
            if Common.plabel_xy.isVisible() == False: Common.plabel_xy.setVisible(True)
        if Common.plabel_xz: 
            Common.plabel_xz.setText('%s, %s'  %  (Common.world_x_str, Common.world_z_str))
            if Common.plabel_xz.isVisible() == False: Common.plabel_xz.setVisible(True)
        if Common.plabel_zy: 
            Common.plabel_zy.setText('%s, %s'  %  (Common.world_z_str, Common.world_y_str))
            if Common.plabel_zy.isVisible() == False: Common.plabel_zy.setVisible(True)


    def update_channel(self, plane, k):
        ax_xy = Common.ax_xy
        ax_xz = Common.ax_xz
        ax_zy = Common.ax_zy
        im_xy = Common.im_xy
        im_xz = Common.im_xz
        im_zy = Common.im_zy
        marker_mgr = getattr(self, 'marker_manager', None)
        if plane == 'xy':
            z = self.format_pix.convert_chpix_to_world(self.plane, Common.xpix, Common.ypix, k)
            z_str = self.format_pix.convert_chval_to_world_str(self.plane, z)
            if Common.chval_box_xy:
                Common.chlabel_xy.set_text("%s" % z_str)
                Common.chval_box_xy.setText("%s" % z_str)
                Common.chval_box_xy.setCursorPosition(0)
                if Common.chlabel_xy.get_visible() == False: Common.chlabel_xy.set_visible(True)
            Common.update_pix(Common.xpix, Common.ypix, k)
            Common.update_world_xyz(Common.world_x, Common.world_y, z)
            Common.update_world_xyz_str(Common.world_x_str, Common.world_y_str, z_str)
            if self.data.ndim == 3:
                im_xy.set_data(self.data[k])
            elif self.data.ndim == 4:
                im_xy.set_data(self.data[0, k])

            Common.ax_xy.draw_artist(Common.im_xy)
            Common.overlay_ax_xy.draw_artist(Common.chlabel_xy)

            if hasattr(self, 'control_panel') and self.control_panel.pvd_panel:
                if k >= 0:
                    self.control_panel.pvd_panel.update_cursor(k)

            self._refresh_contours()
            self.refresh_display_after_contour_update(self._contour_layer_id)

            if Common.clicked_zy == False: Common.xz_hline.set_ydata([k])
            if Common.clicked_xz == False: Common.zy_vline.set_xdata([k])

            if Common.clicked_xy == False:
                if self.SubWindow.subwindow1.isHidden() == False:
                    if Common.plabel_xy.isVisible() == True:
                        Common.canvas_xz.restore_region(Common._background_xz)
                        Common.overlay_ax_xz.draw_artist(Common.xz_hline)
                        Common.overlay_ax_xz.draw_artist(Common.xz_vline)
                        Common.overlay_ax_xz.draw_artist(Common.chlabel_xz)
                        if marker_mgr is not None:
                            marker_mgr.draw_markers_for_blit('xz')
                        Common.canvas_xz.blit(Common.overlay_ax_xz.bbox)

                if self.SubWindow.subwindow2.isHidden() == False:
                    if Common.plabel_zy.isVisible() == True:
                        Common.canvas_zy.restore_region(Common._background_zy)
                        Common.overlay_ax_zy.draw_artist(Common.zy_hline)
                        Common.overlay_ax_zy.draw_artist(Common.zy_vline)
                        Common.overlay_ax_zy.draw_artist(Common.chlabel_zy)
                        if marker_mgr is not None:
                            marker_mgr.draw_markers_for_blit('zy')
                        Common.canvas_zy.blit(Common.overlay_ax_zy.bbox)

        elif plane == 'xz':
            z = self.format_pix.convert_chpix_to_world(self.plane, Common.xpix, k, Common.ypix)
            z_str = self.format_pix.convert_chval_to_world_str(self.plane, z)
            if Common.chval_box_xz:
                Common.chlabel_xz.set_text("%s" % z_str)
                Common.chval_box_xz.setText("%s" % z_str)
                Common.chval_box_xz.setCursorPosition(0)
                if Common.chlabel_xz.get_visible() == False: Common.chlabel_xz.set_visible(True)
            Common.update_pix(Common.xpix, k, Common.zpix)
            Common.update_world_xyz(Common.world_x, z, Common.world_z)
            Common.update_world_xyz_str(Common.world_x_str, z_str, Common.world_z_str)
            if self.data.ndim == 3:
                im_xz.set_data(self.data[:, k, :])
            elif self.data.ndim == 4:
                im_xz.set_data(self.data[0, :, k, :])
            Common.ax_xz.draw_artist(Common.im_xz)
            Common.overlay_ax_xz.draw_artist(Common.chlabel_xz)

            self._refresh_contours()
            self.refresh_display_after_contour_update(self._contour_layer_id)

            if Common.clicked_zy == False: Common.xy_hline.set_ydata([k])
            if Common.clicked_xy == False: Common.zy_hline.set_ydata([k])

            if Common.clicked_xz == False:
                if Common.plabel_xy.isVisible() == True:
                    Common.main_window.redraw_main_overlay_and_blit()

                    if self.SubWindow.subwindow2.isHidden() == False:
                        Common.canvas_zy.restore_region(Common._background_zy)
                        Common.overlay_ax_zy.draw_artist(Common.zy_hline)
                        Common.overlay_ax_zy.draw_artist(Common.zy_vline)
                        Common.overlay_ax_zy.draw_artist(Common.chlabel_zy)
                        if marker_mgr is not None:
                            marker_mgr.draw_markers_for_blit('zy')
                        Common.canvas_zy.blit(Common.overlay_ax_zy.bbox)

        elif plane == 'zy':
            z = self.format_pix.convert_chpix_to_world(self.plane, k, Common.ypix, Common.xpix)
            z_str = self.format_pix.convert_chval_to_world_str(self.plane, z)
            if Common.chval_box_zy:
                Common.chlabel_zy.set_text("%s" % z_str)
                Common.chval_box_zy.setText("%s" % z_str)
                Common.chval_box_zy.setCursorPosition(0)
                if Common.chlabel_zy.get_visible() == False: Common.chlabel_zy.set_visible(True)
            Common.update_pix(k, Common.ypix, Common.zpix)
            Common.update_world_xyz(z, Common.world_y, Common.world_z)
            Common.update_world_xyz_str(z_str, Common.world_y_str, Common.world_z_str)
            if self.data.ndim == 3:
                im_zy.set_data(self.data[:, :, k].T)
            elif self.data.ndim == 4:
                im_zy.set_data(self.data[0, :, :, k].T)
            Common.ax_zy.draw_artist(Common.im_zy)
            Common.overlay_ax_zy.draw_artist(Common.chlabel_zy)

            self._refresh_contours()
            self.refresh_display_after_contour_update(self._contour_layer_id)

            if Common.clicked_xz == False: Common.xy_vline.set_xdata([k])
            if Common.clicked_xy == False: Common.xz_vline.set_xdata([k])

            if Common.clicked_zy == False:
                if Common.plabel_xy.isVisible() == True:
                    Common.main_window.redraw_main_overlay_and_blit()

                    if self.SubWindow.subwindow1.isHidden() == False:
                        Common.canvas_xz.restore_region(Common._background_xz)
                        Common.overlay_ax_xz.draw_artist(Common.xz_hline)
                        Common.overlay_ax_xz.draw_artist(Common.xz_vline)
                        Common.overlay_ax_xz.draw_artist(Common.chlabel_xz)
                        if marker_mgr is not None:
                            marker_mgr.draw_markers_for_blit('xz')
                        Common.canvas_xz.blit(Common.overlay_ax_xz.bbox)

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

        try:
            manager.contour_updated.disconnect(self.refresh_display_after_contour_update)
        except TypeError:
            pass
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

    def _refresh_contours(self):
        if not self._contour_layer_id:
            return
        ContourManager.instance().refresh_layer(self._contour_layer_id)

    def _unregister_contour_layer(self):
        if not self._contour_layer_id:
            return
        ContourManager.instance().unregister_layer(self._contour_layer_id)
        self._contour_layer_id = None


    def resizeEvent(self, event):
        self.label.setGeometry(int(self.canvas.width()*self.poslabel_x - self.poslabel_w/2), int(self.canvas.height() - self.canvas.height()*self.poslabel_y + self.poslabel_h/2) , self.poslabel_w, self.poslabel_h)
        #self.label.setGeometry(int(self.canvas.width()-self.poslabel_x), int(self.canvas.height() * self.poslabel_y), self.poslabel_w, self.poslabel_h)
        #self.label2.setGeometry(int(self.canvas.width()-self.poslabel_x), int(self.canvas.height()- self.poslabel_y - 15), self.poslabel_w, 15)
        #self.position_label_relative_to_axis(self.label2, self.pos_chlabel_x, self.pos_chlabel_y, self.pos_chlabel_w, self.pos_chlabel_h)
        if self.toolbar._subplot_dialog is not None:
            self.toolbar._subplot_dialog.width_spin.setValue(int(self.window().width()))
            self.toolbar._subplot_dialog.height_spin.setValue(int(self.window().height()))
            
        super().resizeEvent(event)


    def redraw_main_overlay_and_blit(self):
        """
        Redraws overlay artists (crosshairs, PVD, regions, HPBW)
        on the main XY canvas and blits the result.
        Contours are assumed to be part of the background captured in update_channel.
        """
        # Ensure this complex drawing logic only runs on the main window's context.
        if Common.canvas_xy is None or Common._background_xy is None or Common.overlay_ax_xy is None:
            return

        Common.canvas_xy.restore_region(Common._background_xy)

        # Draw artists (on overlay_ax)
        Common.overlay_ax_xy.draw_artist(Common.xy_vline)
        Common.overlay_ax_xy.draw_artist(Common.xy_hline)

        if Common.chlabel_xy:
            Common.overlay_ax_xy.draw_artist(Common.chlabel_xy)
        if hasattr(self, 'control_panel'):
            if self.control_panel.pvd_panel is not None and self.control_panel.pvd_panel.arrow_artist is not None:
                Common.overlay_ax_xy.draw_artist(self.control_panel.pvd_panel.arrow_artist)
                for indicator in self.control_panel.pvd_panel.width_indicators:
                    Common.overlay_ax_xy.draw_artist(indicator)
                if self.control_panel.pvd_panel.pos_indicator_on_arrow is not None:
                    Common.overlay_ax_xy.draw_artist(self.control_panel.pvd_panel.pos_indicator_on_arrow)

        if hasattr(self, 'region_manager'):
            self.region_manager.draw_regions_for_blit()

        Common.hpbw_xy.update_position()

        if hasattr(self, 'marker_manager') and self.marker_manager is not None:
            self.marker_manager.draw_markers_for_blit('xy')

        Common.canvas_xy.blit(Common.overlay_ax_xy.bbox)


    def redraw_overlay_for_plane(self, plane=None):
        """
        Restore overlay background and re-blit crosshair / markers for the specified plane.
        Falls back to the viewer's current plane if not provided.
        """
        plane = plane or getattr(self, 'plane', None)
        if plane not in ('xy', 'xz', 'zy'):
            return

        marker_manager = getattr(self, 'marker_manager', None)
        if plane == 'xy':
            self.redraw_main_overlay_and_blit()
            return

        canvas = getattr(Common, f'canvas_{plane}', None)
        overlay_ax = getattr(Common, f'overlay_ax_{plane}', None)
        background = getattr(Common, f'_background_{plane}', None)
        if canvas is None or overlay_ax is None:
            if canvas is not None:
                canvas.draw_idle()
            return

        if background is None:
            canvas.draw_idle()
            return

        try:
            canvas.restore_region(background)
        except Exception:
            canvas.draw_idle()
            return

        line_h = getattr(Common, f'{plane}_hline', None)
        line_v = getattr(Common, f'{plane}_vline', None)
        label = getattr(Common, f'chlabel_{plane}', None)

        for artist in (line_h, line_v, label):
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


    def open_marker_panel(self):
        if self.marker_panel is None or not self.marker_panel.isVisible():
            from tools.marker_panel import MarkerPanel
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

    def _position_marker_panel(self, panel):
        if panel is None:
            return
        try:
            panel.adjustSize()
        except Exception:
            pass
        anchor = getattr(Common, "main_window", None)
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

    def world_extent(self, plane, cache=True):
        """
        Return the full data extent for the chosen plane in world coordinates.
        """
        if cache and plane in self._full_world_limits:
            return self._full_world_limits[plane].copy()

        plane_viewer = self._get_plane_viewer(plane)
        if plane_viewer is None:
            return {}

        fmt = getattr(plane_viewer, 'format_pix', None)
        if fmt is None:
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

        # Sample at midpoints to minimise rounding discrepancies.
        x_ref = sum(xlim) / 2.0
        y_ref = sum(ylim) / 2.0

        primary_min, _ = fmt.convert(plane_viewer.plane, xlim[0], y_ref)
        primary_max, _ = fmt.convert(plane_viewer.plane, xlim[1], y_ref)

        _, secondary_min = fmt.convert(plane_viewer.plane, x_ref, ylim[0])
        _, secondary_max = fmt.convert(plane_viewer.plane, x_ref, ylim[1])

        extent = {
            f"{primary_label}_min": str(primary_min),
            f"{primary_label}_max": str(primary_max),
            f"{secondary_label}_min": str(secondary_min),
            f"{secondary_label}_max": str(secondary_max),
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

        fmt = getattr(plane_viewer, 'format_pix', None)
        if fmt is None:
            return {}

        axis_labels = {
            'xy': ('x', 'y'),
            'xz': ('x', 'z'),
            'zy': ('z', 'y'),
        }
        primary_axis, secondary_axis = axis_labels.get(plane, ('x', 'y'))

        x_ref = (xlim[0] + xlim[1]) / 2.0
        y_ref = (ylim[0] + ylim[1]) / 2.0

        # Primary axis (first coordinate returned by convert)
        primary_min, _ = fmt.convert(plane_viewer.plane, xlim[0], y_ref)
        primary_max, _ = fmt.convert(plane_viewer.plane, xlim[1], y_ref)

        # Secondary axis (second coordinate)
        _, secondary_min = fmt.convert(plane_viewer.plane, x_ref, ylim[0])
        _, secondary_max = fmt.convert(plane_viewer.plane, x_ref, ylim[1])

        result = {
            f"{primary_axis}_min": str(primary_min),
            f"{primary_axis}_max": str(primary_max),
            f"{secondary_axis}_min": str(secondary_min),
            f"{secondary_axis}_max": str(secondary_max),
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


    def set_x_range(self):
        """Set the X range for both the MainWindow and SubWindow1."""
        try:
            if self.plane == 'xy':
                x_min = Common.xmin_input_xy.text()
                x_max = Common.xmax_input_xy.text()
                y_min = Common.ymin_input_xy.text()
                y_max = Common.ymax_input_xy.text()
            elif self.plane == 'zy':
                x_min = Common.xmin_input_xy.text()
                x_max = Common.xmax_input_xy.text()
                y_min = Common.ymin_input_zy.text()
                y_max = Common.ymax_input_zy.text()
                
            elif self.plane == 'xz':
                x_min = Common.xmin_input_xz.text()
                x_max = Common.xmax_input_xz.text()
                y_min = Common.ymin_input_zy.text()
                y_max = Common.ymax_input_zy.text()
                
            if self.data.ndim == 3:
                xp_min = float(self.converter.world_to_pix(x_min, y_min, self.original_zval)[0])
                xp_max = float(self.converter.world_to_pix(x_max, y_max, self.original_zval)[0])
            elif self.data.ndim == 4:
                xp_min = float(self.converter.world_to_pix(x_min, y_min, self.original_zval, 0)[0])
                xp_max = float(self.converter.world_to_pix(x_max, y_max, self.original_zval, 0)[0])
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


                
            elif self.plane == 'xz' and len(self.parent.subwindows) > 0 and self.parent.subwindows[1]:  
                self.parent.ax.set_xlim(xp_min, xp_max)
                self.parent.overlay_ax.set_position(self.parent.ax.get_position())
                self.parent._suspend_regions_for_full_draw()
                self.parent.canvas.draw_idle()
                self.parent.x_min_input.setText(x_min)
                self.parent.x_max_input.setText(x_max)
                
            


        except ValueError:
            QMessageBox.warning(self, 'Invalid Input', 'Please enter valid numeric values for the X range.')
            return

        root = getattr(Common, 'main_window', self if self.plane == 'xy' else getattr(self, 'parent', self))
        if hasattr(root, 'range_panel'):
            root.range_panel._sync_from_common('xy')

    def set_y_range(self):
        """Set the Y range for both the MainWindow and SubWindow2."""
        try:
            if self.plane == 'xy':
                x_min = Common.xmin_input_xy.text()
                x_max = Common.xmax_input_xy.text()
                y_min = Common.ymin_input_xy.text()
                y_max = Common.ymax_input_xy.text()
            elif self.plane == 'zy':
                x_min = Common.xmin_input_xy.text()
                x_max = Common.xmax_input_xy.text()
                y_min = Common.ymin_input_zy.text()
                y_max = Common.ymax_input_zy.text()
                
            elif self.plane == 'xz':
                x_min = Common.xmin_input_xz.text()
                x_max = Common.xmax_input_xz.text()
                y_min = Common.ymin_input_zy.text()
                y_max = Common.ymax_input_zy.text()
                
            #y_min = self.y_min_input.text()
            #y_max = self.y_max_input.text()
            #x_min = self.x_min_input.text()
            if self.data.ndim == 3:
                yp_min = float(self.converter.world_to_pix(x_min, y_min, self.original_zval)[1])
                yp_max = float(self.converter.world_to_pix(x_max, y_max, self.original_zval)[1])
            elif self.data.ndim == 4:
                yp_min = float(self.converter.world_to_pix(x_min, y_min, self.original_zval, 0)[1])
                yp_max = float(self.converter.world_to_pix(x_max, y_max, self.original_zval, 0)[1])
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

            if self.plane == 'xy' and len(self.subwindows) > 0 and self.subwindows[1]:
                self.subwindows[1].ax.set_ylim(yp_min, yp_max)
                self.subwindows[1].overlay_ax.set_position(self.subwindows[1].ax.get_position())
                self.subwindows[1]._suspend_regions_for_full_draw()
                self.subwindows[1].canvas.draw_idle()
                self.subwindows[1].y_min_input.setText(y_min)
                self.subwindows[1].y_max_input.setText(y_max)
            elif self.plane == 'zy' and len(self.parent.subwindows) > 0 and self.parent.subwindows[0]:  
                self.parent.ax.set_ylim(yp_min, yp_max)
                self.parent.overlay_ax.set_position(self.parent.ax.get_position())
                self.parent._suspend_regions_for_full_draw()
                self.parent.canvas.draw_idle()
                self.parent.y_min_input.setText(y_min)
                self.parent.y_max_input.setText(y_max)

        except ValueError:
            QMessageBox.warning(self, 'Invalid Input', 'Please enter valid numeric values for the Y range.')
            return

        root = getattr(Common, 'main_window', self if self.plane == 'xy' else getattr(self, 'parent', self))
        if hasattr(root, 'range_panel'):
            root.range_panel._sync_from_common('xy')

    def set_z_range(self):
        """Set the Z range for both SubWindow1 (vertical in XZ plane) and SubWindow2 (horizontal in ZY plane)."""
        try:
            if self.plane == 'xy' or self.plane == 'xz':
                z_min = Common.zmin_input_xz.text()
                z_max = Common.zmax_input_xz.text()
            elif self.plane == 'zy':
                z_min = Common.zmin_input_zy.text()
                z_max = Common.zmax_input_zy.text()
            #z_min = self.z_min_input.text()
            #z_max = self.z_max_input.text()
            #x_min = self.x_min_input.text()
            if self.data.ndim == 3:
                zp_min = float(self.converter.world_to_pix(self.original_xval,self.original_yval, z_min)[2])
                zp_max = float(self.converter.world_to_pix(self.original_xval,self.original_yval, z_max)[2])
            elif self.data.ndim == 4:
                zp_min = float(self.converter.world_to_pix(self.original_xval,self.original_yval, z_min, 0)[2])
                zp_max = float(self.converter.world_to_pix(self.original_xval,self.original_yval, z_max, 0)[2])
            if zp_min > zp_max: zp_min, zp_max = zp_max, zp_min

            # Apply Z range to SubWindow1 (if exists)
            if self.plane == 'xy': self.parent = self
            if len(self.parent.subwindows) > 0 and self.parent.subwindows[0]:  # SubWindow1 (XZ plane vertical axis)
                self.parent.subwindows[0].ax.set_ylim(zp_min, zp_max)
                self.parent.subwindows[0].overlay_ax.set_position(self.parent.subwindows[0].ax.get_position())
                self.parent.subwindows[0]._suspend_regions_for_full_draw()
                self.parent.subwindows[0].canvas.draw_idle()
                self.parent.subwindows[0].z_min_input.setText(z_min)
                self.parent.subwindows[0].z_max_input.setText(z_max)

            # Apply Z range to SubWindow2 (if exists)
            if len(self.parent.subwindows) > 1 and self.parent.subwindows[1]:  # SubWindow2 (ZY plane horizontal axis)
                self.parent.subwindows[1].ax.set_xlim(zp_min, zp_max)
                self.parent.subwindows[1].overlay_ax.set_position(self.parent.subwindows[1].ax.get_position())
                self.parent.subwindows[1]._suspend_regions_for_full_draw()
                self.parent.subwindows[1].canvas.draw_idle()
                self.parent.subwindows[1].z_min_input.setText(z_min)
                self.parent.subwindows[1].z_max_input.setText(z_max)

        except ValueError:
            QMessageBox.warning(self, 'Invalid Input', 'Please enter valid numeric values for the Z range.')
            return

        root = getattr(Common, 'main_window', self if self.plane == 'xy' else getattr(self, 'parent', self))
        if hasattr(root, 'range_panel'):
            root.range_panel._sync_from_common('xz')
            root.range_panel._sync_from_common('zy')

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

        is_main = Common.main_window is None or self is Common.main_window

        if plane == "xy" and is_main:
            xmin_val = limit('x_min', new_xlim[0])
            xmax_val = limit('x_max', new_xlim[1])
            ymin_val = limit('y_min', new_ylim[0])
            ymax_val = limit('y_max', new_ylim[1])

            Common.xmin_input_xy.setText(str(xmin_val))
            Common.xmax_input_xy.setText(str(xmax_val))
            Common.ymin_input_xy.setText(str(ymin_val))
            Common.ymax_input_xy.setText(str(ymax_val))

            if self.data.ndim > 2:
                Common.xmin_input_xz.setText(str(xmin_val))
                Common.xmax_input_xz.setText(str(xmax_val))
                Common.ymin_input_zy.setText(str(ymin_val))
                Common.ymax_input_zy.setText(str(ymax_val))
            root = getattr(Common, 'main_window', self)
            if hasattr(root, 'range_panel'):
                root.range_panel._sync_from_common('xy')

        elif plane == "xz" and self.data.ndim > 2 and is_main:
            xmin_val = limit('x_min', new_xlim[0])
            xmax_val = limit('x_max', new_xlim[1])
            zmin_val = limit('z_min', new_ylim[0])
            zmax_val = limit('z_max', new_ylim[1])

            Common.xmin_input_xz.setText(str(xmin_val))
            Common.xmax_input_xz.setText(str(xmax_val))
            Common.zmin_input_xz.setText(str(zmin_val))
            Common.zmax_input_xz.setText(str(zmax_val))

            Common.xmin_input_xy.setText(str(xmin_val))
            Common.xmax_input_xy.setText(str(xmax_val))
            Common.zmin_input_zy.setText(str(zmin_val))
            Common.zmax_input_zy.setText(str(zmax_val))
            root = getattr(Common, 'main_window', self)
            if hasattr(root, 'range_panel'):
                root.range_panel._sync_from_common('xz')

        elif plane == "zy" and self.data.ndim > 2 and is_main:
            zmin_val = limit('z_min', new_xlim[0])
            zmax_val = limit('z_max', new_xlim[1])
            ymin_val = limit('y_min', new_ylim[0])
            ymax_val = limit('y_max', new_ylim[1])

            Common.zmin_input_zy.setText(str(zmin_val))
            Common.zmax_input_zy.setText(str(zmax_val))
            Common.ymin_input_zy.setText(str(ymin_val))
            Common.ymax_input_zy.setText(str(ymax_val))

            Common.ymin_input_xy.setText(str(ymin_val))
            Common.ymax_input_xy.setText(str(ymax_val))
            Common.zmin_input_xz.setText(str(zmin_val))
            Common.zmax_input_xz.setText(str(zmax_val))
            root = getattr(Common, 'main_window', self)
            if hasattr(root, 'range_panel'):
                root.range_panel._sync_from_common('zy')


    def reset_all_ranges(self):
        parent = getattr(self, 'parent', None)
        if parent is not None and hasattr(parent, 'reset_all_ranges') and self is not parent:
             parent.reset_all_ranges()
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
            self.update_ranges('zy', self.original_zlim, self.original_ylim)
        
        # Ensure Range Control panel mirrors the refreshed ranges
        if hasattr(self, 'range_panel'):
            self.range_panel._sync_from_common('xy')
            if self.data.ndim > 2:
                self.range_panel._sync_from_common('xz')
                self.range_panel._sync_from_common('zy')

    
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
        if layer_id and self._contour_layer_id != layer_id:
            return
        plane = getattr(self, 'plane', None)
        if plane is None:
            return

        artists_to_hide: List[object] = []
        if plane == 'xy':
            if Common.xy_vline: artists_to_hide.append(Common.xy_vline)
            if Common.xy_hline: artists_to_hide.append(Common.xy_hline)
            if Common.chlabel_xy: artists_to_hide.append(Common.chlabel_xy)
            if Common.hpbw_xy and Common.hpbw_xy.ellipse: artists_to_hide.append(Common.hpbw_xy.ellipse)
            
            main_window = getattr(self, 'parent', self)
            if hasattr(main_window, 'control_panel') and main_window.control_panel and main_window.control_panel.pvd_panel:
                pvd = main_window.control_panel.pvd_panel
                if pvd.arrow_artist:
                    artists_to_hide.append(pvd.arrow_artist)
                artists_to_hide.extend(pvd.width_indicators)
                if pvd.pos_indicator_on_arrow:
                    artists_to_hide.append(pvd.pos_indicator_on_arrow)
                    
        elif plane == 'xz':
            if Common.xz_vline: artists_to_hide.append(Common.xz_vline)
            if Common.xz_hline: artists_to_hide.append(Common.xz_hline)
            if Common.chlabel_xz: artists_to_hide.append(Common.chlabel_xz)
            
        elif plane == 'zy':
            if Common.zy_vline: artists_to_hide.append(Common.zy_vline)
            if Common.zy_hline: artists_to_hide.append(Common.zy_hline)
            if Common.chlabel_zy: artists_to_hide.append(Common.chlabel_zy)

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
    
        try:
            if self._contour_layer_id:
                target_canvas = None
                background = None
                if plane == 'xy':
                    target_canvas = Common.canvas_xy
                    background = Common._background_xy
                elif plane == 'xz':
                    target_canvas = Common.canvas_xz
                    background = Common._background_xz
                elif plane == 'zy':
                    target_canvas = Common.canvas_zy
                    background = Common._background_zy
                
                if target_canvas is not None and background is not None:
                    target_canvas.restore_region(background)
                else:
                    canvas = getattr(self, 'canvas', None)
                    if canvas is not None:
                        canvas.draw()
                if plane == 'xy' and Common.ax_xy is not None and Common.im_xy is not None:
                    Common.ax_xy.draw_artist(Common.im_xy)
                elif plane == 'xz' and Common.ax_xz is not None and Common.im_xz is not None:
                    Common.ax_xz.draw_artist(Common.im_xz)
                elif plane == 'zy' and Common.ax_zy is not None and Common.im_zy is not None:
                    Common.ax_zy.draw_artist(Common.im_zy)

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
            
            if plane == 'xy' and Common.canvas_xy:
                Common._background_xy = Common.copy_overlay_background('xy')
            elif plane == 'xz' and Common.canvas_xz:
                Common._background_xz = Common.copy_overlay_background('xz')
            elif plane == 'zy' and Common.canvas_zy:
                Common._background_zy = Common.copy_overlay_background('zy')

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
                    self.redraw_main_overlay_and_blit() # Fallback
            elif plane == 'xz' and Common.canvas_xz:
               # Common.canvas_xz.restore_region(Common._background_xz)
                Common.overlay_ax_xz.draw_artist(Common.xz_hline)
                Common.overlay_ax_xz.draw_artist(Common.xz_vline)
                Common.overlay_ax_xz.draw_artist(Common.chlabel_xz)
                Common.canvas_xz.blit(Common.overlay_ax_xz.bbox)
            elif plane == 'zy' and Common.canvas_zy:
               # Common.canvas_zy.restore_region(Common._background_zy)
                Common.overlay_ax_zy.draw_artist(Common.zy_hline)
                Common.overlay_ax_zy.draw_artist(Common.zy_vline)
                Common.overlay_ax_zy.draw_artist(Common.chlabel_zy)
                Common.canvas_zy.blit(Common.overlay_ax_zy.bbox)
                
        except Exception as e:
            print(f"Warning: Error during final blit after contour update: {e}")
