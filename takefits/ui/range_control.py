from PySide6.QtWidgets import QWidget, QGridLayout, QLineEdit, QPushButton, QLabel, QMessageBox
from PySide6.QtCore import Qt
from takefits.core.coordinate import CoordinateConverter
import os
from pathlib import Path

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
            self.original_zlim = self.subwindows[0].ax.get_ylim()
        self.range_file = self.fits_viewer.range_file
        
        self.initUI()
            

    def initUI(self):
        self.converter = CoordinateConverter(self.wcs, self.fits_viewer.config_manager.config)
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
        self.x_button.setFixedWidth(50)
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
        self.y_button.setFixedWidth(50)
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
            self.z_button.setFixedWidth(50)
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
        self.save_range_button.setFixedWidth(50)
        self.save_range_button.clicked.connect(self.save_range_button_pressed)
        layout.addWidget(self.save_range_button, 3, 4)

        self.load_range_button = QPushButton('Load', self)
        self.load_range_button.setFixedWidth(50)
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

    def read_range_file(self, range_file):
        coords = {}
        meta = {'filename': None, 'coordinate_system': None}
        with open(range_file, 'r', encoding='utf-8') as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith('#'):
                    lower = stripped.lower()
                    if lower.startswith('#filename'):
                        parts = stripped.split(':', 1)
                        if len(parts) == 2:
                            meta['filename'] = parts[1].strip()
                    elif lower.startswith('#coordinate'):
                        parts = stripped.split(':', 1)
                        if len(parts) == 2:
                            meta['coordinate_system'] = parts[1].strip()
                    continue

                parts = stripped.split()
                if len(parts) != 4:
                    raise ValueError(f'Invalid line format: "{line.strip()}"')
                key = parts[0].upper()
                ctype = parts[1]
                try:
                    coord1 = float(parts[2])
                    coord2 = float(parts[3])
                except ValueError as err:
                    raise ValueError(f'Invalid numeric values in line: "{line.strip()}"') from err
                coords[key] = (ctype, coord1, coord2)
        if 'X' not in coords or 'Y' not in coords:
            raise ValueError('Range file must contain X and Y entries.')
        return meta, coords

    def _write_range_file(self, range_file):
        axis_order = [('X', 0), ('Y', 1)]
        if self.fits_viewer.data.ndim > 2:
            axis_order.append(('Z', 2))

        entries = []
        for axis_label, axis_index in axis_order:
            input_widget = getattr(self, f"{axis_label.lower()}_min_input", None)
            max_widget = getattr(self, f"{axis_label.lower()}_max_input", None)
            if input_widget is None or max_widget is None:
                continue

            try:
                minimum = float(input_widget.text())
                maximum = float(max_widget.text())
            except ValueError:
                raise ValueError(f'Invalid numeric value for {axis_label} range.')

            ctype = self.wcs.wcs.ctype[axis_index] if self.wcs and self.wcs.wcs and len(self.wcs.wcs.ctype) > axis_index else ''
            entries.append((axis_label, ctype, minimum, maximum))

        summary_parts = []
        for _, ctype, _, _ in entries:
            if ctype:
                summary_parts.append(ctype.split('-')[0])
        coordinate_summary = ', '.join(summary_parts)

        output_lines = []
        output_lines.append(f'#filename: {self.fits_viewer.filename}')
        output_lines.append(f'#Coordinate: {coordinate_summary}')
        for axis_label, ctype, minimum, maximum in entries:
            output_lines.append(f'{axis_label} {ctype} {minimum} {maximum}')

        range_path = Path(range_file)
        if range_path.parent and not range_path.parent.exists():
            range_path.parent.mkdir(parents=True, exist_ok=True)

        with open(range_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines) + '\n')

    def _coordinate_system_matches(self, saved_axes):
        if not self.wcs or not self.wcs.wcs:
            return True

        axis_mapping = [('X', 0), ('Y', 1)]
        if self.fits_viewer.data.ndim > 2:
            axis_mapping.append(('Z', 2))

        # Allow different velocity/frequency types to be compatible for range loading
        velocity_types = {'VELO', 'VRAD', 'VOPT', 'FREQ'}

        for label, idx in axis_mapping:
            if label not in saved_axes or len(self.wcs.wcs.ctype) <= idx:
                continue
            current = self.wcs.wcs.ctype[idx].split('-')[0]
            saved = saved_axes[label][0].split('-')[0] if saved_axes[label][0] else ''

            # If both are velocity/frequency types, consider them a match
            if current in velocity_types and saved in velocity_types:
                continue

            if current != saved:
                return False
        return True

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
            meta, coords = self.read_range_file(range_path)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, 'Failed to Load Range', f'Could not load range file:\n{exc}')
            return

        if not self._coordinate_system_matches(coords):
            QMessageBox.warning(
                self,
                'Coordinate Mismatch',
                'The coordinate system in the range file does not match the current FITS data.'
            )
            return

        x_min, x_max = coords['X'][1], coords['X'][2]
        y_min, y_max = coords['Y'][1], coords['Y'][2]

        self.x_min_input.setText(str(x_min))
        self.x_max_input.setText(str(x_max))
        self.set_x_range()

        self.y_min_input.setText(str(y_min))
        self.y_max_input.setText(str(y_max))
        self.set_y_range()

        if self.fits_viewer.data.ndim > 2 and 'Z' in coords and hasattr(self, 'z_min_input'):
            z_min, z_max = coords['Z'][1], coords['Z'][2]
            self.z_min_input.setText(str(z_min))
            self.z_max_input.setText(str(z_max))
            self.set_z_range()

        print(f'\n\nRange file "{self.range_file}" was loaded from {range_path}.')

    def save_range_button_pressed(self):
        range_path = self._get_range_file_path()
        if not range_path:
            QMessageBox.warning(self, 'Range File Missing', 'No range file configured.')
            return
        try:
            self._write_range_file(range_path)
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
        try:
            x_min = self.x_min_input.text()
            x_max = self.x_max_input.text()
            y_min = self.y_min_input.text()
            y_max = self.y_max_input.text()
            if self.fits_viewer.data.ndim == 3:
                xp_min = float(self.converter.world_to_pix(x_min,y_min,self.original_zval)[0])
                xp_max = float(self.converter.world_to_pix(x_max,y_max,self.original_zval)[0])
            elif self.fits_viewer.data.ndim == 4:
                xp_min = float(self.converter.world_to_pix(x_min,y_min,self.original_zval, 0)[0])
                xp_max = float(self.converter.world_to_pix(x_max,y_max,self.original_zval, 0)[0])
            elif self.fits_viewer.data.ndim == 2:
                xp_min = float(self.converter.world_to_pix(x_min,y_min)[0])
                xp_max = float(self.converter.world_to_pix(x_max,y_max)[0])
                
            if xp_min > xp_max: xp_min, xp_max = xp_max, xp_min
                
            self.fits_viewer.ax.set_xlim(xp_min, xp_max)
            self.fits_viewer.canvas.draw_idle()

            if hasattr(self.fits_viewer, 'x_min_input'):
                self.fits_viewer.x_min_input.setText(str(x_min))
                self.fits_viewer.x_max_input.setText(str(x_max))

            # Apply X range to SubWindow1 (if exists)
            if len(self.subwindows) > 0 and self.subwindows[0]:  # SubWindow1 (XZ plane)
                self.subwindows[0].ax.set_xlim(xp_min, xp_max)
                self.subwindows[0].canvas.draw_idle()
                if hasattr(self.subwindows[0], 'x_min_input'):
                    self.subwindows[0].x_min_input.setText(str(x_min))
                    self.subwindows[0].x_max_input.setText(str(x_max))

        except (ValueError, TypeError):
            if not getattr(self, "_suppress_range_warning", False):
                QMessageBox.warning(self, 'Invalid Input', 'Please enter valid numeric values for the X range.')
            return

        self._sync_inputs('xy')
        record_history = getattr(self.fits_viewer, "_record_shared_view_history", None)
        if callable(record_history):
            try:
                record_history(reason="range_panel:set_x")
            except Exception:
                pass
    def set_y_range(self):
        """Set the Y range for both the MainWindow and SubWindow2."""
        try:
            y_min = self.y_min_input.text()
            y_max = self.y_max_input.text()
            x_min = self.x_min_input.text()
            x_max = self.x_max_input.text()
            if self.fits_viewer.data.ndim == 3:
                yp_min = float(self.converter.world_to_pix(x_min,y_min,self.original_zval)[1])
                yp_max = float(self.converter.world_to_pix(x_max,y_max,self.original_zval)[1])
            elif self.fits_viewer.data.ndim == 4:
                yp_min = float(self.converter.world_to_pix(x_min,y_min,self.original_zval, 0)[1])
                yp_max = float(self.converter.world_to_pix(x_max,y_max,self.original_zval, 0)[1])
            elif self.fits_viewer.data.ndim == 2:
                yp_min = float(self.converter.world_to_pix(x_min,y_min)[1])
                yp_max = float(self.converter.world_to_pix(x_max,y_max)[1])

                
            if yp_min > yp_max: yp_min, yp_max = yp_max, yp_min
            
            self.fits_viewer.ax.set_ylim(yp_min, yp_max)
            self.fits_viewer.canvas.draw_idle()

            if hasattr(self.fits_viewer, 'y_min_input'):
                self.fits_viewer.y_min_input.setText(str(y_min))
                self.fits_viewer.y_max_input.setText(str(y_max))

            # Apply Y range to SubWindow1 (if exists)
            if len(self.subwindows) > 1 and self.subwindows[1]:  # SubWindow2 (ZY plane)
                self.subwindows[1].ax.set_ylim(yp_min, yp_max)
                self.subwindows[1].canvas.draw_idle()
                if hasattr(self.subwindows[1], 'y_min_input'):
                    self.subwindows[1].y_min_input.setText(str(y_min))
                    self.subwindows[1].y_max_input.setText(str(y_max))

        except (ValueError, TypeError):
            if not getattr(self, "_suppress_range_warning", False):
                QMessageBox.warning(self, 'Invalid Input', 'Please enter valid numeric values for the Y range.')
            return

        self._sync_inputs('xy')
        record_history = getattr(self.fits_viewer, "_record_shared_view_history", None)
        if callable(record_history):
            try:
                record_history(reason="range_panel:set_y")
            except Exception:
                pass
    def set_z_range(self):
        """Set the Z range for both SubWindow1 (vertical in XZ plane) and SubWindow2 (horizontal in ZY plane)."""
        try:
            z_min = self.z_min_input.text()
            z_max = self.z_max_input.text()
            y_min = self.y_min_input.text()
            y_max = self.y_max_input.text()
            x_min = self.x_min_input.text()
            x_max = self.x_max_input.text()
            
            if self.fits_viewer.data.ndim == 3:
                zp_min = float(self.converter.world_to_pix(x_min, y_min, z_min)[2])
                zp_max = float(self.converter.world_to_pix(x_max, y_max, z_max)[2])
            elif self.fits_viewer.data.ndim == 4:
                zp_min = float(self.converter.world_to_pix(x_min, y_min, z_min, 0)[2])
                zp_max = float(self.converter.world_to_pix(x_min, y_max, z_max, 0)[2])
                
            if zp_min > zp_max: zp_min, zp_max = zp_max, zp_min

            # Apply Z range to SubWindow1 (if exists)
            if len(self.subwindows) > 0 and self.subwindows[0]:  # SubWindow1 (XZ plane vertical axis)
                self.subwindows[0].ax.set_ylim(zp_min, zp_max)
                self.subwindows[0].canvas.draw_idle()
                if hasattr(self.subwindows[0], 'z_min_input'):
                    self.subwindows[0].z_min_input.setText(str(z_min))
                    self.subwindows[0].z_max_input.setText(str(z_max))

            # Apply Z range to SubWindow2 (if exists)
            if len(self.subwindows) > 1 and self.subwindows[1]:  # SubWindow2 (ZY plane horizontal axis)
                self.subwindows[1].ax.set_xlim(zp_min, zp_max)
                self.subwindows[1].canvas.draw_idle()
                if hasattr(self.subwindows[1], 'z_min_input'):
                    self.subwindows[1].z_min_input.setText(str(z_min))
                    self.subwindows[1].z_max_input.setText(str(z_max))

            # Keep the conversion anchor in sync with the latest Z-range edits.
            self.original_zval = str(z_min)
            self.fits_viewer.original_zval = str(z_min)
            for sw in list(self.subwindows or []):
                try:
                    sw.original_zval = str(z_min)
                except Exception:
                    continue

        except (ValueError, TypeError):
            if not getattr(self, "_suppress_range_warning", False):
                QMessageBox.warning(self, 'Invalid Input', 'Please enter valid numeric values for the Z range.')
            return

        self._sync_inputs('xz')
        self._sync_inputs('zy')
        record_history = getattr(self.fits_viewer, "_record_shared_view_history", None)
        if callable(record_history):
            try:
                record_history(reason="range_panel:set_z")
            except Exception:
                pass
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
        self.fits_viewer.menu_bar.range_panel_action.setChecked(False)
