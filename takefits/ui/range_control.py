from PySide6.QtWidgets import QWidget, QGridLayout, QLineEdit, QPushButton, QLabel, QMessageBox
from PySide6.QtCore import Qt
from takefits.core.range_files import (
    build_range_payload,
    build_coordinate_mismatch_message,
    evaluate_range_payload_compatibility,
    extract_native_ranges,
    load_range_payload,
    native_ranges_to_pixel_limits,
    save_range_payload,
)
from takefits.core.wcs_frames import normalize_display_frame, parse_world_value
from takefits.ui.widget_sizing import fit_button_to_text
import os

class RangeControlPanel(QWidget):
    def __init__(self, fits_viewer, subwindows):
        super().__init__()
        self.fits_viewer = fits_viewer
        self.subwindows = subwindows
        self.decimal = self.fits_viewer.decimal
        self.number_decimals = self.fits_viewer.number_decimals
        self.wcs = self.fits_viewer.wcs
        self.coord_wrap = self.fits_viewer.coord_wrap
        
        self.original_xlim = self.fits_viewer.ax.get_xlim()
        self.original_ylim = self.fits_viewer.ax.get_ylim()
        if self.fits_viewer.data.ndim > 2:
            subwindow_xz = self._subwindow(0)
            if subwindow_xz is not None:
                self.original_zlim = subwindow_xz.ax.get_ylim()
            else:
                self.original_zlim = tuple(getattr(self.fits_viewer, 'original_zlim', (0.0, 0.0)))
        self.range_file = self.fits_viewer.range_file
        
        self.initUI()
            

    def initUI(self):
        layout = QGridLayout()
        
        layout.setHorizontalSpacing(4)
        layout.setVerticalSpacing(8)

        # X-axis range input fields
        self.xr_label = QLabel('X:')
        self.xr_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.x_min_input = QLineEdit(self)
        self.x_min_input.setPlaceholderText("X min")
        self.x_min_input.setFixedWidth(100)
        self.x_min_input.returnPressed.connect(self.set_x_range)
        self.x_max_input = QLineEdit(self)
        self.x_max_input.setPlaceholderText("X max")
        self.x_max_input.setFixedWidth(100)
        self.x_max_input.returnPressed.connect(self.set_x_range)
        self.x_button = QPushButton('Set X', self)
        fit_button_to_text(self.x_button)
        self.x_button.clicked.connect(self.set_x_range)

        layout.addWidget(self.xr_label, 0, 0)
        layout.addWidget(self.x_min_input, 0, 1, 1, 2)
        layout.addWidget(self.x_max_input, 0, 3, 1, 2)
        layout.addWidget(self.x_button, 0, 5)

        # Y-axis range input fields
        self.yr_label = QLabel('Y:')
        self.yr_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.y_min_input = QLineEdit(self)
        self.y_min_input.setPlaceholderText("Y min")
        self.y_min_input.setFixedWidth(100)
        self.y_min_input.returnPressed.connect(self.set_y_range)
        self.y_max_input = QLineEdit(self)
        self.y_max_input.setPlaceholderText("Y max")
        self.y_max_input.setFixedWidth(100)
        self.y_max_input.returnPressed.connect(self.set_y_range)
        self.y_button = QPushButton('Set Y', self)
        fit_button_to_text(self.y_button)
        self.y_button.clicked.connect(self.set_y_range)

        layout.addWidget(self.yr_label, 1, 0)
        layout.addWidget(self.y_min_input, 1, 1, 1, 2)
        layout.addWidget(self.y_max_input, 1, 3, 1, 2)
        layout.addWidget(self.y_button, 1, 5)

        # Z-axis range input fields
        if self.fits_viewer.data.ndim > 2:
            self.zr_label = QLabel('Z:')
            self.zr_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.z_min_input = QLineEdit(self)
            self.z_min_input.setPlaceholderText("Z min")
            self.z_min_input.setFixedWidth(100)
            self.z_min_input.returnPressed.connect(self.set_z_range)
            self.z_max_input = QLineEdit(self)
            self.z_max_input.setPlaceholderText("Z max")
            self.z_max_input.setFixedWidth(100)
            self.z_max_input.returnPressed.connect(self.set_z_range)
            self.z_button = QPushButton('Set Z', self)
            fit_button_to_text(self.z_button)
            self.z_button.clicked.connect(self.set_z_range)

            layout.addWidget(self.zr_label, 2, 0)
            layout.addWidget(self.z_min_input, 2, 1, 1, 2)
            layout.addWidget(self.z_max_input, 2, 3, 1, 2)
            layout.addWidget(self.z_button, 2, 5)
            
            xy_limits = self.fits_viewer.compute_world_limits('xy', self.original_xlim, self.original_ylim)
            xmin_val = xy_limits.get('x_min', str(self.original_xlim[0]))
            xmax_val = xy_limits.get('x_max', str(self.original_xlim[1]))
            ymin_val = xy_limits.get('y_min', str(self.original_ylim[0]))
            ymax_val = xy_limits.get('y_max', str(self.original_ylim[1]))

            xz_limits = self.fits_viewer.compute_world_limits('xz', self.original_xlim, self.original_zlim)
            zmin_val = xz_limits.get('z_min', str(self.original_zlim[0]))
            zmax_val = xz_limits.get('z_max', str(self.original_zlim[1]))

            # When ZY is lazily created, avoid overwriting XY world Y-range with pixel fallbacks.
            if len(self.subwindows) > 1 and self.subwindows[1]:
                zy_limits = self.fits_viewer.compute_world_limits('zy', self.original_zlim, self.original_ylim)
                ymin_val = zy_limits.get('y_min', ymin_val)
                ymax_val = zy_limits.get('y_max', ymax_val)
            
            self.z_min_input.setText(str(zmin_val))
            self.z_max_input.setText(str(zmax_val))
            self.original_zval = zmin_val
            self.fits_viewer.original_zval = zmin_val
            
        elif self.fits_viewer.data.ndim == 2:
            xy_limits = self.fits_viewer.compute_world_limits('xy', self.original_xlim, self.original_ylim)
            xmin_val = xy_limits.get('x_min', str(self.original_xlim[0]))
            xmax_val = xy_limits.get('x_max', str(self.original_xlim[1]))
            ymin_val = xy_limits.get('y_min', str(self.original_ylim[0]))
            ymax_val = xy_limits.get('y_max', str(self.original_ylim[1]))
        
        self.reset_button = QPushButton('Full', self)
        self.reset_button.clicked.connect(self.reset_all_ranges)
        layout.addWidget(self.reset_button, 3, 0, 1, 4)

        self.save_range_button = QPushButton('Save', self)
        fit_button_to_text(self.save_range_button)
        self.save_range_button.clicked.connect(self.save_range_button_pressed)
        layout.addWidget(self.save_range_button, 3, 4)

        self.load_range_button = QPushButton('Load', self)
        fit_button_to_text(self.load_range_button)
        self.load_range_button.clicked.connect(self.load_range_button_pressed)
        layout.addWidget(self.load_range_button, 3, 5)
        
        self.x_min_input.setText(str(xmin_val))
        self.x_max_input.setText(str(xmax_val))
        self.y_min_input.setText(str(ymin_val))
        self.y_max_input.setText(str(ymax_val))
        
        self.fits_viewer.x_min_input.setText(str(xmin_val))
        self.fits_viewer.x_max_input.setText(str(xmax_val))        
        self.fits_viewer.y_min_input.setText(str(ymin_val))
        self.fits_viewer.y_max_input.setText(str(ymax_val))
        
        
        if len(self.subwindows) > 0 and self.subwindows[0]:
            self.subwindows[0].x_min_input.setText(str(xmin_val))
            self.subwindows[0].x_max_input.setText(str(xmax_val))
            self.subwindows[0].z_min_input.setText(str(zmin_val))
            self.subwindows[0].z_max_input.setText(str(zmax_val))
            self.subwindows[0].original_xval = xmin_val
            self.subwindows[0].original_yval = ymin_val
            self.subwindows[0].original_zval = zmin_val

        if len(self.subwindows) > 1 and self.subwindows[1]:
            self.subwindows[1].y_min_input.setText(str(ymin_val))
            self.subwindows[1].y_max_input.setText(str(ymax_val))
            self.subwindows[1].z_min_input.setText(str(zmin_val))
            self.subwindows[1].z_max_input.setText(str(zmax_val))
            self.subwindows[1].original_xval = xmin_val
            self.subwindows[1].original_yval = ymin_val
            self.subwindows[1].original_zval = zmin_val
            
        
        self.original_xval = xmin_val
        self.original_yval = ymin_val
        self.fits_viewer.original_xval = xmin_val
        self.fits_viewer.original_yval = ymin_val

        self.setLayout(layout)
        self.setWindowTitle(f'Range Control Panel:{self.fits_viewer.filename}')

        if not self._sync_inputs():
            self.update_ranges('xy', None, None)
            if self.fits_viewer.data.ndim > 2:
                self.update_ranges('xz', None, None)
                self.update_ranges('zy', None, None)

        # Move the window to the right of the main window
        self.move_to_default_position()

        self._update_range_file_buttons()

    def _resolve_range_file_path(self, filename):
        if not filename:
            return None
        if os.path.isabs(filename):
            return filename
        config_file = getattr(self.fits_viewer.config_manager, 'config_file', None)
        if config_file:
            config_dir = os.path.dirname(config_file)
            if config_dir:
                return os.path.join(config_dir, filename)
        return os.path.join(os.getcwd(), filename)

    def _get_range_file_path(self):
        configured = str(self.fits_viewer.config_manager.config.get('range_file', 'takefits.range') or '').strip()
        if not configured:
            configured = 'takefits.range'
            try:
                self.fits_viewer.config_manager.config['range_file'] = configured
            except Exception:
                pass
        self.range_file = configured
        return self._resolve_range_file_path(configured)

    def _update_range_file_buttons(self):
        range_path = self._get_range_file_path()
        has_path = bool(range_path)
        self.save_range_button.setEnabled(has_path)
        self.load_range_button.setEnabled(has_path and os.path.isfile(range_path))

    def _axis_keys(self):
        keys = ['x', 'y']
        if self.fits_viewer.data.ndim > 2:
            keys.append('z')
        return keys

    def _subwindow(self, index):
        if 0 <= int(index) < len(self.subwindows):
            return self.subwindows[int(index)]
        return None

    def _axis_index(self, axis_key):
        return {'x': 0, 'y': 1, 'z': 2}.get(str(axis_key or '').lower(), -1)

    def _axis_ctype(self, axis_key):
        axis_index = self._axis_index(axis_key)
        if axis_index < 0 or self.wcs is None or getattr(self.wcs, 'wcs', None) is None:
            return ''
        ctype = list(getattr(self.wcs.wcs, 'ctype', []) or [])
        if axis_index >= len(ctype):
            return ''
        return str(ctype[axis_index] or '')

    def _current_display_frame(self):
        getter = getattr(self.fits_viewer, 'get_wcs_display_frame', None)
        if callable(getter):
            try:
                return normalize_display_frame(getter())
            except Exception:
                pass
        getter = getattr(self.fits_viewer, '_get_shared_display_frame', None)
        if callable(getter):
            try:
                return normalize_display_frame(getter())
            except Exception:
                pass
        return 'native'

    def _build_dataset_descriptor(self):
        descriptor_builder = getattr(self.fits_viewer, '_dataset_descriptor', None)
        if callable(descriptor_builder):
            try:
                descriptor = descriptor_builder()
            except Exception:
                descriptor = None
            if isinstance(descriptor, dict):
                source = {}
                for key in ('filepath', 'filename', 'wcs_signature'):
                    value = descriptor.get(key)
                    if value not in (None, '', {}):
                        source[key] = value
                return source

        filepath = str(getattr(self.fits_viewer, 'filename_path', '') or '')
        descriptor = {
            'filepath': os.path.abspath(filepath) if filepath else '',
            'filename': os.path.basename(filepath) if filepath else str(getattr(self.fits_viewer, 'filename', '') or ''),
        }
        signature_builder = getattr(self.fits_viewer, '_build_wcs_signature', None)
        if callable(signature_builder):
            try:
                signature = signature_builder(getattr(self.fits_viewer, 'wcs', None))
            except Exception:
                signature = None
            if isinstance(signature, dict):
                descriptor['wcs_signature'] = dict(signature)
        return descriptor

    def _current_wcs_signature(self):
        descriptor = self._build_dataset_descriptor()
        signature = descriptor.get('wcs_signature')
        if isinstance(signature, dict):
            return dict(signature)
        return {}

    def _fallback_native_world(self):
        fallback_world = None
        getter = getattr(self.fits_viewer, '_shared_world_vector', None)
        if callable(getter):
            try:
                fallback_world = list(getter() or [])
            except Exception:
                fallback_world = None
        return fallback_world

    def _parse_axis_value(self, axis_key, value_text, *, axis_ctype=None, frame=None):
        text = str(value_text or '').strip()
        if not text:
            raise ValueError(f'Missing {str(axis_key).upper()} range value.')
        ctype = str(axis_ctype or self._axis_ctype(axis_key) or '')
        frame_name = str(frame or self._current_display_frame() or 'native')
        try:
            return float(parse_world_value(text, ctype, frame_for_longitude=frame_name))
        except Exception as exc:
            raise ValueError(f'Invalid value for {str(axis_key).upper()} range: "{text}"') from exc

    def _collect_range_payload(self):
        payload = {}
        for axis_key in self._axis_keys():
            min_widget = getattr(self, f'{axis_key}_min_input', None)
            max_widget = getattr(self, f'{axis_key}_max_input', None)
            if min_widget is None or max_widget is None:
                continue
            min_text = str(min_widget.text() or '').strip()
            max_text = str(max_widget.text() or '').strip()
            if not min_text or not max_text:
                raise ValueError(f'Missing {axis_key.upper()} range.')
            payload[axis_key] = {
                'ctype': self._axis_ctype(axis_key),
                'min_text': min_text,
                'max_text': max_text,
                'native_min': self._parse_axis_value(axis_key, min_text),
                'native_max': self._parse_axis_value(axis_key, max_text),
            }
        return payload

    def _pixel_limits_from_native_ranges(self, native_ranges):
        return native_ranges_to_pixel_limits(
            self.wcs,
            native_ranges,
            fallback_native_world=self._fallback_native_world(),
        )

    @staticmethod
    def _set_view_limits(viewer, *, xlim=None, ylim=None):
        if viewer is None or not hasattr(viewer, 'ax'):
            return
        if xlim is not None:
            viewer.ax.set_xlim(*xlim)
        if ylim is not None:
            viewer.ax.set_ylim(*ylim)
        overlay_ax = getattr(viewer, 'overlay_ax', None)
        if overlay_ax is not None:
            try:
                overlay_ax.set_position(viewer.ax.get_position())
            except Exception:
                pass
        canvas = getattr(viewer, 'canvas', None)
        if canvas is not None:
            canvas.draw_idle()

    def _build_range_payload_from_inputs(self):
        return build_range_payload(
            source=self._build_dataset_descriptor(),
            ranges=self._collect_range_payload(),
        )

    @staticmethod
    def _set_axis_input_texts(viewer, axis_key, minimum, maximum):
        if viewer is None:
            return
        min_input = getattr(viewer, f'{axis_key}_min_input', None)
        max_input = getattr(viewer, f'{axis_key}_max_input', None)
        if min_input is not None:
            min_input.setText(str(minimum))
        if max_input is not None:
            max_input.setText(str(maximum))

    def _axis_targets(self, axis_key):
        if axis_key == 'x':
            return (
                (self.fits_viewer, 'xlim', 'x'),
                (self._subwindow(0), 'xlim', 'x'),
            )
        if axis_key == 'y':
            return (
                (self.fits_viewer, 'ylim', 'y'),
                (self._subwindow(1), 'ylim', 'y'),
            )
        if axis_key == 'z':
            return (
                (self._subwindow(0), 'ylim', 'z'),
                (self._subwindow(1), 'xlim', 'z'),
            )
        return ()

    def _apply_axis_limit(self, axis_key, pixel_range, minimum, maximum):
        for viewer, limit_name, input_axis in self._axis_targets(axis_key):
            if viewer is None:
                continue
            self._set_view_limits(viewer, **{limit_name: pixel_range})
            self._set_axis_input_texts(viewer, input_axis, minimum, maximum)

    def _native_ranges_from_payload(self, payload, *, include_full_z):
        native_ranges = {
            'x': (payload['x']['native_min'], payload['x']['native_max']),
            'y': (payload['y']['native_min'], payload['y']['native_max']),
        }
        if self.fits_viewer.data.ndim <= 2:
            return native_ranges

        z_entry = payload.get('z')
        if z_entry is None:
            if include_full_z:
                raise ValueError('Missing Z range.')
            return native_ranges

        if include_full_z:
            native_ranges['z'] = (z_entry['native_min'], z_entry['native_max'])
        else:
            z_anchor = z_entry['native_min']
            native_ranges['z'] = (z_anchor, z_anchor)
        return native_ranges

    def _set_z_anchor(self, z_text):
        z_anchor = str(z_text or '').strip()
        if not z_anchor:
            return
        self.original_zval = z_anchor
        self.fits_viewer.original_zval = z_anchor
        for sw in list(self.subwindows or []):
            try:
                sw.original_zval = z_anchor
            except Exception:
                continue

    def _apply_axis_range_from_payload(self, axis_key, payload):
        if axis_key not in payload:
            raise ValueError(f'Missing {axis_key.upper()} range.')

        pixel_limits = self._pixel_limits_from_native_ranges(
            self._native_ranges_from_payload(payload, include_full_z=(axis_key == 'z'))
        )
        if axis_key not in pixel_limits:
            raise ValueError(f'Calculated {axis_key.upper()} pixel limits are unavailable.')

        minimum = payload[axis_key]['min_text']
        maximum = payload[axis_key]['max_text']
        self._apply_axis_limit(axis_key, pixel_limits[axis_key], minimum, maximum)

        if axis_key == 'z':
            self._set_z_anchor(minimum)

    def _sync_planes_for_axis(self, axis_key):
        if axis_key == 'z':
            return ('xz', 'zy')
        return ('xy',)

    def _record_axis_history(self, axis_key):
        record_history = getattr(self.fits_viewer, "_record_shared_view_history", None)
        if not callable(record_history):
            return
        reason = {
            'x': 'range_panel:set_x',
            'y': 'range_panel:set_y',
            'z': 'range_panel:set_z',
        }.get(axis_key)
        if not reason:
            return
        try:
            record_history(reason=reason)
        except Exception:
            pass

    def _set_axis_range(self, axis_key):
        try:
            payload = self._collect_range_payload()
            self._apply_axis_range_from_payload(axis_key, payload)
        except (ValueError, TypeError):
            if not getattr(self, "_suppress_range_warning", False):
                QMessageBox.warning(self, 'Invalid Input', f'Please enter valid values for the {axis_key.upper()} range.')
            return

        for plane in self._sync_planes_for_axis(axis_key):
            self._sync_inputs(plane)
        self._record_axis_history(axis_key)

    def _apply_loaded_native_ranges(self, native_ranges):
        pixel_limits = self._pixel_limits_from_native_ranges(native_ranges)
        xlim = pixel_limits.get('x')
        ylim = pixel_limits.get('y')
        zlim = pixel_limits.get('z')

        if xlim is None or ylim is None:
            raise ValueError('Saved range entry is missing X/Y limits.')

        self._set_view_limits(self.fits_viewer, xlim=xlim, ylim=ylim)
        if len(self.subwindows) > 0 and self.subwindows[0]:
            self._set_view_limits(self.subwindows[0], xlim=xlim, ylim=zlim)
        if len(self.subwindows) > 1 and self.subwindows[1]:
            self._set_view_limits(self.subwindows[1], xlim=zlim, ylim=ylim)

        viewer_update_ranges = getattr(self.fits_viewer, 'update_ranges', None)
        if callable(viewer_update_ranges):
            viewer_update_ranges('xy', xlim, ylim)
            if self.fits_viewer.data.ndim > 2 and zlim is not None:
                viewer_update_ranges('xz', xlim, zlim)
                if len(self.subwindows) > 1 and self.subwindows[1]:
                    viewer_update_ranges('zy', zlim, ylim)

        if self.fits_viewer.data.ndim > 2 and zlim is not None and hasattr(self, 'z_min_input'):
            self._set_z_anchor(self.z_min_input.text())

    def load_range_button_pressed(self):
        range_path = self._get_range_file_path()
        if not range_path:
            QMessageBox.warning(self, 'Range File Missing', 'No range file configured.')
            return
        if not os.path.isfile(range_path):
            QMessageBox.warning(self, 'Range File Missing', f'Range file not found:\n{range_path}')
            self._update_range_file_buttons()
            return

        try:
            payload = load_range_payload(range_path)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, 'Failed to Load Range', f'Could not load range file:\n{exc}')
            return

        current_signature = self._current_wcs_signature()
        compatible, _reason = evaluate_range_payload_compatibility(
            payload,
            current_signature=current_signature,
            data_ndim=int(getattr(self.fits_viewer.data, 'ndim', 0) or 0),
        )
        native_ranges = extract_native_ranges(payload)
        if not native_ranges:
            QMessageBox.warning(
                self,
                'Range Not Found',
                'No saved range entry was found.'
            )
            return

        if not compatible:
            QMessageBox.warning(
                self,
                'Coordinate Mismatch',
                build_coordinate_mismatch_message(payload, current_signature),
            )
            return

        try:
            self._apply_loaded_native_ranges(native_ranges)
        except (ValueError, TypeError) as exc:
            QMessageBox.warning(self, 'Failed to Load Range', str(exc))
            return

        # Record the loaded view so it can be reverted with View Back (Cmd/Ctrl+Z),
        # consistent with typing a range into the panel.
        record_history = getattr(self.fits_viewer, "_record_shared_view_history", None)
        if callable(record_history):
            try:
                record_history(reason='range_panel:load')
            except Exception:
                pass

        source = payload.get('source') if isinstance(payload.get('source'), dict) else {}
        saved_name = str(source.get('filename') or '').strip()
        if saved_name:
            print(f'\n\nRange file "{self.range_file}" loaded from {range_path} using entry "{saved_name}".')
        else:
            print(f'\n\nRange file "{self.range_file}" was loaded from {range_path}.')

    def save_range_button_pressed(self):
        range_path = self._get_range_file_path()
        if not range_path:
            QMessageBox.warning(self, 'Range File Missing', 'No range file configured.')
            return
        try:
            payload = self._build_range_payload_from_inputs()
            save_range_payload(range_path, payload)
        except ValueError as exc:
            QMessageBox.warning(self, 'Failed to Save Range', str(exc))
            return
        except OSError as exc:
            QMessageBox.warning(self, 'Failed to Save Range', f'Could not write range file:\n{exc}')
            return

        self._update_range_file_buttons()
        print(f'\n\nRange file "{self.range_file}" was saved to {range_path}.')
        
    def move_to_default_position(self):
        # Get MainWindow geometry
        mainwindow_geometry = self.fits_viewer.geometry()
        mainwindow_x = mainwindow_geometry.x()
        mainwindow_y = mainwindow_geometry.y()
        mainwindow_width = mainwindow_geometry.width()

        # Move ControlPanel to the right of MainWindow
        self.move(mainwindow_x + mainwindow_width, mainwindow_y - 28)

    def _sync_inputs(self, plane=None):
        updated = False

        def copy_xy():
            # Sync X/Y inputs from Main Window (fits_viewer)
            if hasattr(self.fits_viewer, 'x_min_input') and hasattr(self.fits_viewer, 'x_max_input'):
                 self.x_min_input.setText(self.fits_viewer.x_min_input.text())
                 self.x_max_input.setText(self.fits_viewer.x_max_input.text())
            if hasattr(self.fits_viewer, 'y_min_input') and hasattr(self.fits_viewer, 'y_max_input'):
                 self.y_min_input.setText(self.fits_viewer.y_min_input.text())
                 self.y_max_input.setText(self.fits_viewer.y_max_input.text())
            return True

        def copy_xz():
            # Sync X/Z inputs from SubWindow 1 (XZ)
            # Assuming subwindows[0] is XZ
            if len(self.subwindows) > 0 and self.subwindows[0]:
                 sw = self.subwindows[0]
                 if hasattr(sw, 'x_min_input') and hasattr(sw, 'x_max_input'):
                     self.x_min_input.setText(sw.x_min_input.text())
                     self.x_max_input.setText(sw.x_max_input.text())
                 if hasattr(sw, 'z_min_input') and hasattr(sw, 'z_max_input'):
                     self.z_min_input.setText(sw.z_min_input.text())
                     self.z_max_input.setText(sw.z_max_input.text())
                 return True
            return False

        def copy_zy():
            # Sync Z/Y inputs from SubWindow 2 (ZY)
            # Assuming subwindows[1] is ZY
            if len(self.subwindows) > 1 and self.subwindows[1]:
                 sw = self.subwindows[1]
                 if hasattr(sw, 'z_min_input') and hasattr(sw, 'z_max_input'):
                     self.z_min_input.setText(sw.z_min_input.text())
                     self.z_max_input.setText(sw.z_max_input.text())
                 if hasattr(sw, 'y_min_input') and hasattr(sw, 'y_max_input'):
                     self.y_min_input.setText(sw.y_min_input.text())
                     self.y_max_input.setText(sw.y_max_input.text())
                 return True
            return False

        if plane in (None, 'xy'):
            copy_xy()

        if self.fits_viewer.data.ndim > 2:
            if plane in (None, 'xz'):
                copy_xz()
            if plane in (None, 'zy'):
                copy_zy()

        return True

    def set_x_range(self):
        """Set the X range for both the MainWindow and SubWindow1."""
        self._set_axis_range('x')

    def set_y_range(self):
        """Set the Y range for both the MainWindow and SubWindow2."""
        self._set_axis_range('y')

    def set_z_range(self):
        """Set the Z range for both SubWindow1 (vertical in XZ plane) and SubWindow2 (horizontal in ZY plane)."""
        self._set_axis_range('z')
    def update_ranges(self, plane, xlim, ylim):
        # Always use the main viewer to compute full limits so we mirror its rounding.
        # Prefer values already shown on the main window so we mirror them exactly.
        if plane == "xy" and hasattr(self.fits_viewer, 'x_min_input'):
            self.x_min_input.setText(self.fits_viewer.x_min_input.text())
            self.x_max_input.setText(self.fits_viewer.x_max_input.text())
            self.y_min_input.setText(self.fits_viewer.y_min_input.text())
            self.y_max_input.setText(self.fits_viewer.y_max_input.text())
            return
    
        if plane == "xz" and self.fits_viewer.data.ndim > 2 and len(self.subwindows) > 0 and hasattr(self.subwindows[0], 'x_min_input'):
            self.x_min_input.setText(self.subwindows[0].x_min_input.text())
            self.x_max_input.setText(self.subwindows[0].x_max_input.text())
            self.z_min_input.setText(self.subwindows[0].z_min_input.text())
            self.z_max_input.setText(self.subwindows[0].z_max_input.text())
            return
                
        if plane == "zy" and self.fits_viewer.data.ndim > 2 and len(self.subwindows) > 1 and hasattr(self.subwindows[1], 'z_min_input'):
            z_min_text = self.subwindows[1].z_min_input.text().strip()
            z_max_text = self.subwindows[1].z_max_input.text().strip()
            y_min_text = self.subwindows[1].y_min_input.text().strip()
            y_max_text = self.subwindows[1].y_max_input.text().strip()
            if z_min_text and z_max_text and y_min_text and y_max_text:
                self.z_min_input.setText(z_min_text)
                self.z_max_input.setText(z_max_text)
                self.y_min_input.setText(y_min_text)
                self.y_max_input.setText(y_max_text)
                return

        # Fallback: compute from WCS if shared widgets are not yet initialised.
        if xlim is None or ylim is None:
            world_limits = self.fits_viewer.world_extent(plane)
            if plane == 'xy':
                default_xlim = getattr(self.fits_viewer, 'original_xlim', self.original_xlim)
                default_ylim = getattr(self.fits_viewer, 'original_ylim', self.original_ylim)
            elif plane == 'xz':
                default_xlim = getattr(self.fits_viewer, 'original_xlim', self.original_xlim)
                default_ylim = getattr(self.fits_viewer, 'original_zlim', getattr(self, 'original_zlim', (0.0, 0.0)))
            elif plane == 'zy':
                default_xlim = getattr(self.fits_viewer, 'original_zlim', getattr(self, 'original_zlim', (0.0, 0.0)))
                default_ylim = getattr(self.fits_viewer, 'original_ylim', self.original_ylim)
            else:
                return
        else:
            default_xlim = xlim
            default_ylim = ylim
            world_limits = self.fits_viewer.compute_world_limits(plane, default_xlim, default_ylim)

        def limit(key, fallback):
            if world_limits and key in world_limits:
                return world_limits[key]
            return str(fallback)

        if plane == "xy":
            self.x_min_input.setText(str(limit('x_min', default_xlim[0])))
            self.x_max_input.setText(str(limit('x_max', default_xlim[1])))
            self.y_min_input.setText(str(limit('y_min', default_ylim[0])))
            self.y_max_input.setText(str(limit('y_max', default_ylim[1])))
        elif plane == "xz" and self.fits_viewer.data.ndim > 2:
            self.x_min_input.setText(str(limit('x_min', default_xlim[0])))
            self.x_max_input.setText(str(limit('x_max', default_xlim[1])))
            self.z_min_input.setText(str(limit('z_min', default_ylim[0])))
            self.z_max_input.setText(str(limit('z_max', default_ylim[1])))
        elif plane == "zy" and self.fits_viewer.data.ndim > 2:
            self.z_min_input.setText(str(limit('z_min', default_xlim[0])))
            self.z_max_input.setText(str(limit('z_max', default_xlim[1])))
            self.y_min_input.setText(str(limit('y_min', default_ylim[0])))
            self.y_max_input.setText(str(limit('y_max', default_ylim[1])))


    def reset_all_ranges(self):
        # Delegate to main viewer so a single code path computes full ranges.
        self.fits_viewer.reset_all_ranges()
        self.update_ranges('xy', None, None)
        if self.fits_viewer.data.ndim > 2:
            self.update_ranges('xz', None, None)
            if len(self.subwindows) > 1 and self.subwindows[1]:
                self.update_ranges('zy', None, None)

    
    def reset_ranges(self, plane):
        if plane == 'xy':
            self.update_ranges(plane, None, None)
        elif plane == 'xz':
            self.update_ranges(plane, None, None)
        elif plane == 'zy':
            self.update_ranges(plane, None, None)
                                
    def closeEvent(self, event):
        sync_action = getattr(self.fits_viewer, "_set_panel_toggle_checked", None)
        if callable(sync_action):
            sync_action("range_panel_action", False)
        else:
            self.fits_viewer.menu_bar.range_panel_action.setChecked(False)
