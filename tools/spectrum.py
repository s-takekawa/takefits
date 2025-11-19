from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QGridLayout, QLineEdit, QCheckBox, QLabel,
    QFrame, QSizePolicy, QMessageBox, QFileDialog, QSpacerItem,
    QPushButton, QHBoxLayout
)
from PyQt6.QtCore import Qt, QTimer
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import time
import os

from core.common import Common
from core.coordinate import CoordinateConverter
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

import warnings
from core.region import CircleRegion, RectangleRegion, EllipseRegion

class SpecWindow(QWidget):
    is_on = False
    
    def __init__(self, fits_viewer):
        super().__init__()
        self.fits_viewer = fits_viewer
        self.wcs = self.fits_viewer.wcs
        self.config = self.fits_viewer.displaymap.config
        self.converter = CoordinateConverter(self.wcs, self.config)

        self.common_instance = Common._get_instance()
        self.common_instance.position_updated.connect(self.update_spectrum)

        if hasattr(self.fits_viewer, 'region_manager'):
            self.fits_viewer.region_manager.selected_region_changed.connect(self.on_region_changed)

        z_axis_size = self.fits_viewer.data.shape[self.fits_viewer.data.ndim-3] 
        self.initial_x_range = (0, z_axis_size - 1)
        self.initial_y_range = (0, 1)
        
        if self.fits_viewer.data.ndim == 3:
            self.slices = (0, 0, 'x')
        elif self.fits_viewer.data.ndim == 4:
            self.slices = (0, 0, 'x', 0)

        self.auto_y_axis = True
        self.last_update_time = 0
        self.spec_axis = self.wcs.wcs.spec
        
        
        n_channels = self.fits_viewer.header[f'NAXIS{self.spec_axis + 1}']
        crval = self.fits_viewer.header[f'CRVAL{self.spec_axis + 1}']
        cdelt = self.fits_viewer.header[f'CDELT{self.spec_axis + 1}'] 
        crpix = self.fits_viewer.header[f'CRPIX{self.spec_axis + 1}'] 
        self.velocity_values = crval + (np.arange(n_channels) - (crpix - 1)) * cdelt
        
        self.is_dragging = False
        #self.pick_radius = 5  # pixels within which a click is considered 'on the line'
        self.active_region = None

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(10, 0, 10, 0)

        self.setLayout(layout)
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection=self.fits_viewer.wcs, slices=self.slices)

        self.canvas = FigureCanvas(self.fig)
        self.fig.subplots_adjust(bottom=0.15) 

        # CODE ADDED: Connect mouse events for dragging
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('button_release_event', self.on_release)

        self.toolbar = SpecNavigationToolbar(self.canvas, self)

        range_layout = QGridLayout()
        range_layout.setHorizontalSpacing(5)
        range_layout.setVerticalSpacing(0)
    
        # Horizontal Range
        range_layout.addWidget(QLabel("Horizontal (X):"), 0, 0)
        self.x_min_input = QLineEdit(str(self.initial_x_range[0]))
        self.x_min_input.setFixedWidth(60)
        range_layout.addWidget(self.x_min_input, 0, 1)
        self.x_min_input.setText(f"{np.nanmin(self.velocity_values):.4g}")
    
        range_layout.addWidget(QLabel("to"), 0, 2)
    
        self.x_max_input = QLineEdit(str(self.initial_x_range[1]))
        self.x_max_input.setFixedWidth(60)
        range_layout.addWidget(self.x_max_input, 0, 3)
        self.x_max_input.setText(f"{np.nanmax(self.velocity_values):.4g}")

        vline1 = QFrame()
        vline1.setFrameShape(QFrame.Shape.VLine)
        vline1.setFrameShadow(QFrame.Shadow.Sunken)
        range_layout.addWidget(vline1, 0, 4, 1, 1)
    
        # Vertical Range
        range_layout.addWidget(QLabel("Vertical (Y):"), 0, 5)
        self.y_min_input = QLineEdit(str(self.initial_y_range[0]))
        self.y_min_input.setFixedWidth(60)
        range_layout.addWidget(self.y_min_input, 0, 6)
    
        range_layout.addWidget(QLabel("to"), 0, 7)
    
        self.y_max_input = QLineEdit(str(self.initial_y_range[1]))
        self.y_max_input.setFixedWidth(60)
        range_layout.addWidget(self.y_max_input, 0, 8)
    
        layout.addLayout(range_layout) 
        layout.addWidget(self.canvas)

        extract_layout = QHBoxLayout()
        extract_layout.addStretch(1) # Spacer to push the button to the right
        #extract_layout.setContentsMargins(0, 5, 0, 5)
        self.extract_button = QPushButton("Extract")
        self.extract_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.extract_button.setFixedWidth(100)
        self.extract_button.clicked.connect(self.extract_spectrum)
        extract_layout.addWidget(self.extract_button)
        layout.addLayout(extract_layout)

        layout.addWidget(self.toolbar)
    
        # Auto checkbox
        self.auto_checkbox = QCheckBox("Auto Y-axis")
        self.auto_checkbox.setChecked(True)
        self.auto_checkbox.stateChanged.connect(self.toggle_auto_y_axis)
        range_layout.addWidget(self.auto_checkbox, 0, 9)
        range_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum), 0, 10)
        
        self.y_min_input.setEnabled(False)
        self.y_max_input.setEnabled(False)
    
        self.x_min_input.returnPressed.connect(self.set_axis_range)
        self.x_max_input.returnPressed.connect(self.set_axis_range)
        self.y_min_input.returnPressed.connect(self.set_axis_range)
        self.y_max_input.returnPressed.connect(self.set_axis_range)
    
        #Initial figure settings
        self.ax.axhline(y=0, color='gray', linewidth=0.5)
        self.line, = self.ax.step([], [], where='mid', color='blue', linewidth=1.5)
        self.cursor_line = self.ax.axvline(x=Common.zpix, color='cyan', linestyle='-', linewidth=0.75)
    
        #Set label
        self.ax.set_xlabel(f"{self.fits_viewer.displaymap.third_axis_label}")
        self.ax.set_ylabel(f"Intensity [{self.fits_viewer.bunit}]", labelpad=10)
    
        SpecWindow.is_on = True

        self.setWindowTitle(f'Spec Panel: {self.fits_viewer.filename}')
        
        
        ### x ticks ###
        x_coord_helper = self.ax.coords[2]
        x_coord_helper.set_minor_frequency(5)
        x_coord_helper.display_minor_ticks(True)
        x_coord_helper.tick_params(
            axis='x', 
            width=1.,
            length=5,
            direction='in', 
            which='major'
        )
        x_coord_helper.tick_params(
            length=3.,
            which='minor'
        )
        
        ### y ticks ###
        self.ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
        self.ax.yaxis.set_tick_params(
            width=1.,
            length=5,
            direction='in',
            which='major',
            color='black'
        )
    
        self.ax.yaxis.set_tick_params(
            length=3,
            width=1.0,
            direction='in',
            which='minor',
            color='black'
        )
        
        self.update_spectrum(Common.xpix, Common.ypix, Common.zpix)

        if self.fits_viewer.region_mode_enabled:
            selected_region = self.fits_viewer.region_manager.selected_region
            if selected_region:
                # Use QTimer.singleShot to ensure the window is fully loaded before updating
                QTimer.singleShot(0, lambda: self.on_region_changed(selected_region))

    # CODE ADDED: Mouse event handlers for dragging the cursor line
    def on_press(self, event):
        """ Handles mouse button press event. """
        # Ignore clicks outside the axes or when a toolbar tool is active
        if event.inaxes != self.ax or self.toolbar.mode != '':
            return
        
        # Check if the click is close to the cursor line
        contains, _ = self.cursor_line.contains(event)
        if contains:
            self.is_dragging = True

    def on_motion(self, event):
        """ Handles mouse motion event. """
        # If dragging is active, handle the line movement
        if self.is_dragging and event.inaxes == self.ax:
            new_channel = int(round(event.xdata))

            # Ensure the new channel is within the valid range
            n_channels = self.fits_viewer.header[f'NAXIS{self.spec_axis + 1}']
            if 0 <= new_channel < n_channels:
                # Update the main window's slider, which triggers all other updates
                if Common.slider_xy:
                    Common.slider_xy.setValue(new_channel)
            return

        # If not dragging, handle the cursor change on hover
        is_over_line = False
        if event.inaxes == self.ax and self.toolbar.mode == '':
            contains, _ = self.cursor_line.contains(event)
            if contains:
                is_over_line = True

        if is_over_line:
            self.canvas.setCursor(Qt.CursorShape.SizeHorCursor)
        else:
            self.canvas.setCursor(Qt.CursorShape.ArrowCursor)

    def on_release(self, event):
        """ Handles mouse button release event. """
        self.is_dragging = False

    def toggle_auto_y_axis(self):
        self.auto_y_axis = self.auto_checkbox.isChecked()
        self.y_min_input.setEnabled(not self.auto_y_axis)
        self.y_max_input.setEnabled(not self.auto_y_axis)

    def set_axis_range(self):
        try:
            x_min = float(self.x_min_input.text())
            x_max = float(self.x_max_input.text())
            if  self.fits_viewer.data.ndim == 3:
                v_min = self.converter.world_to_pix(Common.world_x, Common.world_y, x_min)[2]
                v_max = self.converter.world_to_pix(Common.world_x, Common.world_y, x_max)[2]
    
            elif  self.fits_viewer.data.ndim == 4:
                v_min = self.converter.world_to_pix(Common.world_x, Common.world_y, x_min, 0)[2]
                v_max = self.converter.world_to_pix(Common.world_x, Common.world_y, x_max, 0)[2]
            
            self.ax.set_xlim(v_min, v_max)
            
            y_min = float(self.y_min_input.text())
            y_max = float(self.y_max_input.text())
            self.ax.set_ylim(y_min, y_max)
            
            self.canvas.draw_idle()
        except ValueError:
            QMessageBox.warning(self, 'Invalid Input', ' Please enter numeric values.')
            return

    def update_spectrum(self, x, y, z):
        current_time = time.time()
        if current_time - self.last_update_time < 0.01:
            return
        self.last_update_time = current_time

        if not self.fits_viewer.region_mode_enabled:
            self.active_region = None

        title = ""
        if self.active_region:
            self.spectrum, title = self._calculate_average_spectrum(self.active_region)

        else:
            x_pix = int(round(x))
            y_pix = int(round(y))
            self.x = x_pix
            self.y = y_pix
            self.spectrum = None

            data_cube = self.fits_viewer.data
            if data_cube.ndim == 4:
                data_cube = data_cube[0]

            if (0 <= x_pix < data_cube.shape[2]) and (0 <= y_pix < data_cube.shape[1]):
                self.spectrum = data_cube[:, y_pix, x_pix]

            try:
                if self.fits_viewer.data.ndim == 3:
                    world = self.converter.pix_to_world(x, y, 0)
                else: # ndim == 4
                    world = self.converter.pix_to_world(x, y, 0, 0)
                title = f"Spectrum at ({world[0]}, {world[1]})"
            except Exception:
                title = "Spectrum"

        self.ax.set_title(title, loc='left')
        if self.spectrum is not None:
            self.update_plot(self.spectrum, x, y, z)
            
    def update_plot(self, spectrum, x, y, z):
        pixel_indices = np.arange(len(spectrum))
        self.line.set_data(range(len(spectrum)), spectrum)
        
        self.cursor_line.set_visible(True)
        self.cursor_line.set_xdata([z, z])
        try:
            x_min = float(self.x_min_input.text())
            x_max = float(self.x_max_input.text())
        except ValueError:
            QMessageBox.warning(self, 'Invalid Input', 'Incorrect value entered for the X-axis range.')
            return
        if  self.fits_viewer.data.ndim == 3:
            v_min = self.converter.world_to_pix(Common.world_x, Common.world_y, x_min)[2]
            v_max = self.converter.world_to_pix(Common.world_x, Common.world_y, x_max)[2]

        elif  self.fits_viewer.data.ndim == 4:
            v_min = self.converter.world_to_pix(Common.world_x, Common.world_y, x_min, 0)[2]
            v_max = self.converter.world_to_pix(Common.world_x, Common.world_y, x_max, 0)[2]
        
        self.ax.set_xlim(v_min, v_max)
        if self.auto_y_axis:
            try:
                x_min_pix, x_max_pix = self.ax.get_xlim()
                x_start = max(int(np.floor(x_min_pix)), 0)
                x_end = min(int(np.ceil(x_max_pix)), len(spectrum))
                if x_start >= x_end:
                     spectrum_in_range = spectrum
                else:
                     spectrum_in_range = spectrum[x_start:x_end]

                if spectrum_in_range.size > 0 and not np.all(np.isnan(spectrum_in_range)):
                    y_min, y_max = np.nanmin(spectrum_in_range), np.nanmax(spectrum_in_range)
                    y_range = y_max - y_min
                    if y_range == 0:
                        y_range = 1e-6
                    self.ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
                    self.update_range_textboxes()
            except (ValueError, IndexError):
                pass
        else:
            self.ax.set_ylim(float(self.y_min_input.text()), float(self.y_max_input.text()))
        
        self.canvas.draw_idle()

    def update_range_textboxes(self):
        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()
        if self.fits_viewer.data.ndim == 3:
            v_min = self.converter.pix_to_world(Common.xpix, Common.ypix, x_min)[2]
            v_max = self.converter.pix_to_world(Common.xpix, Common.ypix, x_max)[2]
        elif self.fits_viewer.data.ndim == 4:
            v_min = self.converter.pix_to_world(Common.xpix, Common.ypix, x_min, 0)[2]
            v_max = self.converter.pix_to_world(Common.xpix, Common.ypix, x_max, 0)[2]

        self.x_min_input.setText(f"{float(v_min):.4g}")
        self.x_max_input.setText(f"{float(v_max):.4g}")
        self.y_min_input.setText(f"{y_min:.4g}")
        self.y_max_input.setText(f"{y_max:.4g}")


    def extract_spectrum(self):
        """
        Extracts the current spectrum data to a text file with a detailed,
        context-aware header.
        """
        if self.spectrum is None:
            QMessageBox.warning(self, "No Data", "No spectrum data to extract.")
            return

        base_filename = os.path.basename(self.fits_viewer.filename)
        default_name = os.path.splitext(base_filename)[0] + ".spec.txt"

        desktop_dir = os.path.join(os.path.expanduser("~"), "Desktop")
        initial_path = os.path.join(desktop_dir, default_name)

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Spectrum", initial_path, "Text Files (*.txt);;All Files (*)"
        )

        if not path:
            return

        source_file = os.path.basename(self.fits_viewer.filename)
        xlabel = self.ax.get_xlabel()
        ylabel = self.ax.get_ylabel()

        header_lines = [f"Source FITS: {source_file}"]

        if self.active_region:
            region = self.active_region
            state = region.get_state()
            shape = region.__class__.__name__.replace('Region', '')
            
            editor = self.fits_viewer.region_manager.region_editors.get(region)

            header_lines.append("Spectrum Type: Region Average")
            header_lines.append(f"Region Shape: {shape}")
            if state.get('label'):
                header_lines.append(f"Region Label: {state['label']}")

            center_pix = state.get('center')
            if center_pix:
                try:
                    if self.fits_viewer.data.ndim == 3:
                        world = self.converter.pix_to_world(center_pix[0], center_pix[1], 0)
                    else: # ndim == 4
                        world = self.converter.pix_to_world(center_pix[0], center_pix[1], 0, 0)
                    header_lines.append(f"Center ({self.wcs.wcs.ctype[0]}, {self.wcs.wcs.ctype[1]}) = ({world[0]}, {world[1]})")
                except Exception:
                    header_lines.append(f"Center (X, Y) [pix]: ({center_pix[0]:.2f}, {center_pix[1]:.2f})")

            if isinstance(region, CircleRegion):
                unit = editor._field_units.get('radius', 'pix') if editor else 'pix'
                value = editor.radius_spin.value() if editor else state['radius']
                header_lines.append(f"Radius [{unit}]: {value:.3f}")

            if isinstance(region, (RectangleRegion, EllipseRegion)):
                unit = editor._field_units.get('width', 'pix') if editor else 'pix'
                width = editor.width_spin.value() if editor else state['width']
                height = editor.height_spin.value() if editor else state['height']
                header_lines.append(f"Width [{unit}]: {width:.3f}")
                header_lines.append(f"Height [{unit}]: {height:.3f}")
            
            if 'angle' in state:
                header_lines.append(f"Angle [deg]: {state['angle']:.2f}")

        else:
            if self.fits_viewer.data.ndim == 3:
                world = self.converter.pix_to_world(self.x, self.y, 0)
            else: # ndim == 4
                world = self.converter.pix_to_world(self.x, self.y, 0, 0)
            coord_str = f"({world[0]}, {world[1]})"
            
            header_lines.append("Spectrum Type: Single Pixel")
            header_lines.append(f"World Coordinate = {coord_str}")
            header_lines.append(f"Pixel Coordinate (X, Y) = ({self.x}, {self.y})")
        
        header_lines.extend([
            "",
            f"Column 1: {xlabel}",
            f"Column 2: {ylabel}",
            "------------------------------------"
        ])
        header = "\n".join([f"# {line}" for line in header_lines])

        data_to_save = np.column_stack((self.velocity_values, self.spectrum))

        try:
            np.savetxt(path, data_to_save, fmt='%.6g', delimiter='   ', header=header, comments='')
            QMessageBox.information(self, "Success", f"Spectrum data successfully saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save spectrum data.\nError: {e}")

    def on_region_changed(self, region):
        """
        Slot for the signal from RegionManager. Updates the active region
        and refreshes the spectrum plot.
        """
        self.active_region = region
        # Trigger an update using the main window's current cursor position.
        self.update_spectrum(Common.xpix, Common.ypix, Common.zpix)

    def _calculate_average_spectrum(self, region):
        """
        Calculates the average spectrum within a region using an optimized
        bounding box approach for performance.
        """
        if self.fits_viewer.data.ndim < 3:
            return None, "Data is not a cube"

        data_cube = self.fits_viewer.data
        if data_cube.ndim == 4:
            data_cube = data_cube[0]

        z_dim, height, width = data_cube.shape

        # --- 1. Get the region's bounding box in pixel coordinates ---
        state = region.get_state()
        if isinstance(region, CircleRegion):
            cx, cy = state['center']
            r = state['radius']
            x_min, x_max = cx - r, cx + r
            y_min, y_max = cy - r, cy + r
        elif isinstance(region, (RectangleRegion, EllipseRegion)):
            # For rotated shapes, find the min/max of their vertices
            if isinstance(region, RectangleRegion):
                verts = region._compute_vertices()
            else: # EllipseRegion - approximate with bounding box vertices
                cx, cy = state['center']
                w, h = state['width'] / 2, state['height'] / 2
                angle = np.deg2rad(state.get('angle', 0.0))
                corners = [(-w, -h), (w, -h), (w, h), (-w, h)]
                verts = []
                for dx, dy in corners:
                    x = cx + np.cos(angle) * dx - np.sin(angle) * dy
                    y = cy + np.sin(angle) * dx + np.cos(angle) * dy
                    verts.append((x, y))
            
            x_coords, y_coords = zip(*verts)
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
        else:
            return None, "Unsupported region shape"
        
        # --- 2. Create a small sub-cube based on the bounding box ---
        # Convert to integer indices and clip to the image boundaries
        x_start, x_end = int(np.floor(x_min)), int(np.ceil(x_max))
        y_start, y_end = int(np.floor(y_min)), int(np.ceil(y_max))
        
        x_start = max(0, x_start)
        y_start = max(0, y_start)
        x_end = min(width, x_end)
        y_end = min(height, y_end)

        if x_start >= x_end or y_start >= y_end:
            return None, "Region is outside data bounds"

        sub_cube = data_cube[:, y_start:y_end, x_start:x_end]
        
        # --- 3. Create a local mask for the sub-cube ---
        # Create a coordinate grid relative to the main image
        y_grid, x_grid = np.indices(sub_cube.shape[1:])
        x_grid += x_start
        y_grid += y_start

        # Generate the precise mask for this small grid
        local_mask = region.contains(x_grid, y_grid)
        
        if not np.any(local_mask):
            return None, "Region contains no data"

        # --- 4. Calculate the average on the small, masked sub-cube ---
        # Expand 2D mask to 3D for broadcasting
        local_mask_3d = np.broadcast_to(local_mask, sub_cube.shape)
        masked_sub_cube = np.where(local_mask_3d, sub_cube, np.nan)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            average_spectrum = np.nanmean(masked_sub_cube, axis=(1, 2))

        # --- Generate Title (same as before) ---
        region_label = region.label_text.strip()
        title_part1 = f"Average Spectrum ({region_label})" if region_label else f"Average Spectrum (Region {region.region_id})"
        title_part2 = ""
        if hasattr(region, 'center'):
            try:
                world = self.converter.pix_to_world(*region.center, 0)
                title_part2 = f"around ({world[0]}, {world[1]})"
            except Exception: pass
        title = f"{title_part1}\n{title_part2}"

        return average_spectrum, title

    def closeEvent(self, event):
        try:
            self.common_instance.position_updated.disconnect(self.update_spectrum)
            if hasattr(self.fits_viewer, 'region_manager'):
                self.fits_viewer.region_manager.selected_region_changed.disconnect(self.on_region_changed)
        except (TypeError, RuntimeError):
            pass

        SpecWindow.is_on = False
        super().closeEvent(event)
        self.destroyed.emit()


class SpecNavigationToolbar(NavigationToolbar):
    def __init__(self, canvas, parent=None):
        super().__init__(canvas, parent)
        self.parent = parent
        self._zoom_mode = False
        self._pan_mode = False

        self.cid_release = self.canvas.mpl_connect("button_release_event", self.on_release)
        self.cid_click = self.canvas.mpl_connect("button_press_event", self.on_click)

    def on_release(self, event):
        if self._zoom_mode or self._pan_mode:
            self.parent.update_range_textboxes()
        self.parent.on_release(event)

    def on_click(self, event):
        if event.dblclick:
            self.parent.cursor_line.set_visible(False)
            self.parent.canvas.draw_idle()
            current_mode = self.mode
            if current_mode == 'pan/zoom':
                self.pan() 
                self._active = None 
                release_event = mpl.backend_bases.MouseEvent(
                    name='button_release_event', canvas=self.canvas,
                    x=event.x, y=event.y, button=event.button,
                    key=event.key, step=event.step, dblclick=event.dblclick,
                    guiEvent=event.guiEvent
                )
                self.release_pan(release_event)
    
            elif current_mode == 'zoom rect':
                self.zoom()
                self._active = None
                release_event = mpl.backend_bases.MouseEvent(
                    name='button_release_event', canvas=self.canvas,
                    x=event.x, y=event.y, button=event.button,
                    key=event.key, step=event.step, dblclick=event.dblclick,
                    guiEvent=event.guiEvent
                )
                self.release_zoom(release_event)
            else:
                return
            self._update_buttons_checked()
            self.set_message('')

    def zoom(self, *args):
        self._zoom_mode = not self._zoom_mode
        super().zoom(*args)

    def pan(self, *args):
        self._pan_mode = not self._pan_mode
        super().pan(*args)
        
    def home(self, *args):
        super().home(*args)
        if self.parent.spectrum is not None and self.parent.spectrum.size > 0:
            y_min, y_max = np.nanmin(self.parent.spectrum), np.nanmax(self.parent.spectrum)
            y_range = y_max - y_min
            if y_range == 0: y_range = 1e-6
            self.parent.ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
            self.parent.update_range_textboxes()
        self.parent.canvas.draw_idle()

    def save_figure(self):
        current_dir = os.getcwd()
        desktop_dir = os.path.join(os.path.expanduser("~"), "Desktop")
        initial_dir = desktop_dir if os.access(desktop_dir, os.W_OK) else current_dir
        self.default_image_name = self.parent.fits_viewer.filename
        
        if self.default_image_name.endswith(".fits"):
            self.default_image_name = self.default_image_name[:-5]
            
        default_filename = f"{self.default_image_name}.spec.pdf"

        path, _ = QFileDialog.getSaveFileName(
            self.canvas.parent(), "Save Figure",
            os.path.join(initial_dir, default_filename),
            "PDF Files (*.pdf);;EPS Files (*.eps);;SVG Files (*.svg);;PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;All Files (*)"
        )
        if path:
            connections = self.canvas.callbacks.callbacks
            saved_connections = connections.copy()
            self.canvas.callbacks.callbacks = {}
        
            visibility = []
            for patch in self.parent.ax.patches:
                visibility.append(patch.get_visible())
                patch.set_visible(True)
            self.canvas.figure.savefig(path, transparent=True, dpi = 300)
            for patch, vis in zip(self.parent.ax.patches, visibility):
                patch.set_visible(vis)
            
            self.canvas.callbacks.callbacks = saved_connections
            filename = os.path.basename(path) 
            self.show_save_success_message(path, filename)
            
    def show_save_success_message(self, path, filename):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setText(f'"{filename}" was saved successfully at:\n{path}')
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()