from .region import CircleRegion, RectangleRegion, EllipseRegion, CubeRegion
import numpy as np
import math
from PySide6.QtCore import Qt, Signal as pyqtSignal, QObject
from .region import Region
from astropy.wcs.utils import proj_plane_pixel_scales, wcs_to_celestial_frame
from astropy.coordinates import SkyCoord
import astropy.units as u

class RegionManager(QObject):
    """
    Manages all region objects, including their creation, selection, 
    and interaction with mouse events.
    """
    _global_region_counter = 1
    selected_region_changed = pyqtSignal(object)
    def __init__(self, viewer):
        # Initializes the RegionManager.
        super().__init__()
        self.viewer = viewer
        self.regions = []          # List to store all created region objects.
        self.selected_region = None # The region that is currently selected.
        self.active_region = None  # The region currently being drawn.
        self.is_drawing = False
        self.is_dragging = False   # Flag to indicate if a region is being dragged.
        self.is_resizing = False   # Flag to indicate if a region is being resized.
        self.start_point = None    # To store the (x, y) coordinates on mouse press.
        self.drag_start_region_pos = None # To store the initial position of the region being dragged.
        self.resize_handle = None
        self.resize_initial_state = None
        self.region_mode = "circle" # Default shape
        self.is_rotating = False
        self.rotation_center = None
        self.rotation_initial_angle = None
        self.rotation_reference_angle = None
        self.shift_pressed = False
        self.region_editors = {}
        self.drag_initiated = False
        self._axes_cache = {}
        self._is_redrawing = False  # Guard to prevent recursive repaints


    def set_region_mode(self, mode):
        """Sets the shape of the region to be drawn."""
        if mode in ["circle", "rectangle", "ellipse", "cube"]:
            self.region_mode = mode
            print(f"\n\nRegion mode set to: {self.region_mode}")
        else:
            print(f"\n\nUnknown region mode: {mode}")

    def _request_overlay_redraw(self):
        """Requests an overlay redraw using the viewer's blitting pipeline."""
        # Guard against recursive repaint
        if self._is_redrawing:
            return
        self._is_redrawing = True
        try:
            redraw = getattr(self.viewer, 'redraw_main_overlay_and_blit', None)
            if callable(redraw):
                redraw()
                return

            canvas = getattr(self.viewer, 'canvas', None)
            if canvas is not None:
                canvas.draw_idle()
        finally:
            self._is_redrawing = False

    def _get_resize_tolerance(self):
        canvas = getattr(self.viewer, 'canvas', None)
        if canvas is None:
            return 6
        return max(6, min(canvas.width(), canvas.height()) * 0.015)

    def _get_axis_bounds(self):
        axes = getattr(self.viewer, 'ax', None)
        if axes is None:
            return None
        try:
            xmin, xmax = axes.get_xlim()
            ymin, ymax = axes.get_ylim()
        except Exception:
            return None

        if xmin > xmax:
            xmin, xmax = xmax, xmin
        if ymin > ymax:
            ymin, ymax = ymax, ymin
        return xmin, xmax, ymin, ymax

    def _is_colorbar_axes(self, axes):
        if axes is None:
            return False
        gid = getattr(axes, 'get_gid', lambda: None)()
        if gid == 'colorbar':
            return True
        colorbar = getattr(self.viewer, 'colorbar', None)
        if colorbar is not None and getattr(colorbar, 'ax', None) is axes:
            return True
        cax = getattr(self.viewer, 'cax', None)
        if cax is axes:
            return True
        return False

    def _extract_data_coords(self, event):
        xdata = getattr(event, 'xdata', None)
        ydata = getattr(event, 'ydata', None)
        if self._is_colorbar_axes(getattr(event, 'inaxes', None)):
            xdata = None
            ydata = None
        if xdata is not None and ydata is not None:
            return xdata, ydata

        axes = getattr(event, 'inaxes', None)
        if self._is_colorbar_axes(axes):
            axes = None
        if axes is None and self.selected_region is not None:
            axes = getattr(self.selected_region.mpl_patch, 'axes', None)
        if axes is None:
            axes = getattr(self.viewer, 'overlay_ax', getattr(self.viewer, 'ax', None))
        if self._is_colorbar_axes(axes):
            axes = getattr(self.viewer, 'ax', None)

        if axes is not None and getattr(event, 'x', None) is not None and getattr(event, 'y', None) is not None:
            try:
                xdata, ydata = axes.transData.inverted().transform((event.x, event.y))
            except Exception:
                xdata = getattr(event, 'xdata', None)
                ydata = getattr(event, 'ydata', None)
        return xdata, ydata


    def _handle_resize_motion(self, event):
        xdata, ydata = self._extract_data_coords(event)
        if xdata is None or ydata is None or not self.resize_handle or not self.resize_initial_state:
            return

        if isinstance(self.selected_region, CircleRegion):
            cx, cy = self.resize_initial_state['center']
            dx = xdata - cx
            dy = ydata - cy
            radius = max(np.hypot(dx, dy), 1e-3)
            self.selected_region.center = (cx, cy)
            self.selected_region.radius = radius
            self.selected_region.update_visual()
            self._request_overlay_redraw()
            self._notify_region_changed(self.selected_region)
            # CRITICAL: Do NOT update resize_initial_state here.
            return

        elif isinstance(self.selected_region, (RectangleRegion, CubeRegion)):
            state = self.resize_initial_state # Use the state captured at the beginning of the resize
            if state is None:
                return
            edges = self.resize_handle.get('edges', set())
            if not edges:
                return

            center = state.get('center')
            if center is None:
                cx = state['xy'][0] + state['width'] / 2.0
                cy = state['xy'][1] + state['height'] / 2.0
                center = (cx, cy)

            angle = getattr(self.selected_region, 'angle', state.get('angle', 0.0))
            rad = math.radians(angle)
            cos_a = math.cos(rad)
            sin_a = math.sin(rad)

            dx = xdata - center[0]
            dy = ydata - center[1]
            x_local = cos_a * dx + sin_a * dy
            y_local = -sin_a * dx + cos_a * dy

            half_w = state['width'] / 2.0
            half_h = state['height'] / 2.0
            min_size = 1e-3

            new_left = -half_w
            new_right = half_w
            new_bottom = -half_h
            new_top = half_h

            if 'left' in edges: new_left = min(x_local, new_right - min_size)
            if 'right' in edges: new_right = max(x_local, new_left + min_size)
            if 'bottom' in edges: new_bottom = min(y_local, new_top - min_size)
            if 'top' in edges: new_top = max(y_local, new_bottom + min_size)

            new_width = max(new_right - new_left, min_size)
            new_height = max(new_top - new_bottom, min_size)

            center_local_x = (new_right + new_left) / 2.0
            center_local_y = (new_top + new_bottom) / 2.0
            cx_new = center[0] + cos_a * center_local_x - sin_a * center_local_y
            cy_new = center[1] + sin_a * center_local_x + cos_a * center_local_y

            self.selected_region.width = new_width
            self.selected_region.height = new_height
            self.selected_region.xy = (cx_new - new_width / 2.0, cy_new - new_height / 2.0)
            self.selected_region.set_angle(angle)
            
            if isinstance(self.selected_region, CubeRegion):
                self.selected_region.z_min = state['z_min']
                self.selected_region.z_max = state['z_max']

            self.selected_region.update_visual()
            self._request_overlay_redraw()
            self._notify_region_changed(self.selected_region)
            return

        elif isinstance(self.selected_region, EllipseRegion):
            # (EllipseRegion logic remains unchanged)
            state = self.resize_initial_state
            if state is None: return
            edges = self.resize_handle.get('edges', set())
            if not edges: return

            center = state.get('center')
            angle = getattr(self.selected_region, 'angle', state.get('angle', 0.0))
            rad = math.radians(angle)
            cos_a, sin_a = math.cos(rad), math.sin(rad)

            dx, dy = xdata - center[0], ydata - center[1]
            x_local = cos_a * dx + sin_a * dy
            y_local = -sin_a * dx + cos_a * dy
            
            min_size = 1e-3
            half_w, half_h = state['width'] / 2.0, state['height'] / 2.0
            new_left, new_right = -half_w, half_w
            new_bottom, new_top = -half_h, half_h

            if 'left' in edges: new_left = min(x_local, new_right - min_size)
            if 'right' in edges: new_right = max(x_local, new_left + min_size)
            if 'bottom' in edges: new_bottom = min(y_local, new_top - min_size)
            if 'top' in edges: new_top = max(y_local, new_bottom + min_size)

            new_width = max(new_right - new_left, min_size)
            new_height = max(new_top - new_bottom, min_size)

            center_local_x = (new_right + new_left) / 2.0
            center_local_y = (new_top + new_bottom) / 2.0
            cx_new = center[0] + cos_a * center_local_x - sin_a * center_local_y
            cy_new = center[1] + sin_a * center_local_x + cos_a * center_local_y

            self.selected_region.center = (cx_new, cy_new)
            self.selected_region.width = new_width
            self.selected_region.height = new_height
            self.selected_region.set_angle(angle)
            self.selected_region.update_visual()
            self._request_overlay_redraw()
            self._notify_region_changed(self.selected_region)
            return

    def _notify_region_changed(self, region):
        if region is not None:
            dialog = self.region_editors.get(region)
            if dialog is not None:
                dialog.update_from_region()
        if hasattr(self, 'selected_region_changed'):
            self.selected_region_changed.emit(region)

    def _should_start_rotation(self, event, region):
        if not isinstance(region, (RectangleRegion, EllipseRegion)):
            return False
        if event.button != 1:
            return False
        xdata, ydata = self._extract_data_coords(event)
        if xdata is None or ydata is None:
            return False
        if not region.contains(xdata, ydata):
            return False
        return self.shift_pressed

    def _start_rotation(self, event, region):
        if region is None or not isinstance(region, (RectangleRegion, EllipseRegion)):
            return False
        if region is not self.selected_region:
            self.select_region(region)

        xdata, ydata = self._extract_data_coords(event)
        if xdata is None or ydata is None:
            return False

        cx, cy = region.center
        dx = xdata - cx
        dy = ydata - cy
        if math.hypot(dx, dy) < 1e-6:
            return False

        self.is_rotating = True
        self.rotation_center = (cx, cy)
        self.rotation_initial_angle = getattr(region, 'angle', 0.0)
        self.rotation_reference_angle = math.atan2(dy, dx)
        self.is_dragging = False
        self.is_resizing = False
        self.is_drawing = False
        self.resize_handle = None
        self.resize_initial_state = None
        self.drag_initiated = False
        self.start_point = None

        canvas = getattr(self.viewer, 'canvas', None)
        if canvas is not None:
            canvas.setCursor(Qt.CursorShape.CrossCursor)
        return True

    def _apply_rotation(self, event):
        if not self.is_rotating or self.selected_region is None:
            return
        if event.xdata is None or event.ydata is None:
            return

        xdata, ydata = self._extract_data_coords(event)
        if xdata is None or ydata is None:
            return

        dx = xdata - self.rotation_center[0]
        dy = ydata - self.rotation_center[1]
        if math.hypot(dx, dy) < 1e-6:
            return

        current_angle = math.atan2(dy, dx)
        delta = math.atan2(math.sin(current_angle - self.rotation_reference_angle),
                           math.cos(current_angle - self.rotation_reference_angle))
        new_angle = self.rotation_initial_angle + math.degrees(delta)
        self.selected_region.set_angle(new_angle)
        self._request_overlay_redraw()
        self._notify_region_changed(self.selected_region)

    def update_hover_cursor(self, event):
        canvas = getattr(self.viewer, 'canvas', None)
        if canvas is None:
            return False
        if self.is_drawing or self.is_dragging or self.is_resizing:
            return False
        if event.inaxes is None:
            canvas.setCursor(Qt.CursorShape.ArrowCursor)
            return False

        tolerance = self._get_resize_tolerance()

        if self.shift_pressed and event.xdata is not None and event.ydata is not None:
            for region in reversed(self.regions):
                if isinstance(region, (RectangleRegion, EllipseRegion)) and region.contains(event.xdata, event.ydata):
                    canvas.setCursor(Qt.CursorShape.CrossCursor)
                    return True

        for region in reversed(self.regions):
            handle = region.get_resize_handle(event, tolerance)
            if handle:
                cursor = handle.get('cursor', Qt.CursorShape.SizeAllCursor)
                canvas.setCursor(cursor)
                return True
            if event.xdata is not None and event.ydata is not None and region.contains(event.xdata, event.ydata):
                canvas.setCursor(Qt.CursorShape.OpenHandCursor)
                return True

        canvas.setCursor(Qt.CursorShape.ArrowCursor)
        return False

    def handle_press(self, event):
        if not event.inaxes or event.button != 1:
            return

        xdata, ydata = self._extract_data_coords(event)
        if xdata is None or ydata is None:
            return

        clicked_on_existing = False
        # Iterate in reverse order to select the top-most region if they overlap.
        for region in reversed(self.regions):
            if event.dblclick and region.contains(xdata, ydata):
                self.select_region(region)
                self.open_region_editor(region)
                clicked_on_existing = True
                return

            if self._should_start_rotation(event, region):
                if self._start_rotation(event, region):
                    return

            resize_handle = region.get_resize_handle(event, self._get_resize_tolerance())
            if resize_handle:
                self.select_region(region)
                self.is_resizing = True
                self.is_dragging = False
                self.is_drawing = False
                self.resize_handle = resize_handle
                self.resize_initial_state = region.get_state() # Capture full state
                self.drag_initiated = False
                self.start_point = (xdata, ydata)
                clicked_on_existing = True
                break

            if region.contains(xdata, ydata):
                self.select_region(region)
                self.is_dragging = True
                self.is_resizing = False
                self.is_drawing = False
                self.drag_initiated = False
                self.start_point = (xdata, ydata)
                # Use get_state() to capture all relevant properties, including z-axis for CubeRegion
                self.drag_start_region_pos = region.get_state()
                clicked_on_existing = True
                break

        if not clicked_on_existing:
            self.deselect_all()
            self.is_drawing = True
            self.is_dragging = False
            self.start_point = (xdata, ydata)
            if self.region_mode == "circle":
                new_region = CircleRegion(center=self.start_point, radius=0)
            elif self.region_mode == "rectangle":
                new_region = RectangleRegion(xy=self.start_point, width=0, height=0)
            elif self.region_mode == "ellipse":
                new_region = EllipseRegion(center=self.start_point, width=0, height=0)
            elif self.region_mode == "cube":

                z_min, z_max = 0, self.viewer.data.shape[0] - 1
                if hasattr(self.viewer, 'SubWindow') and self.viewer.SubWindow.subwindow1:
                    try:
                        xz_ax = self.viewer.SubWindow.subwindow1.ax
                        z_min_view, z_max_view = xz_ax.get_ylim()
                        z_min = max(0, z_min_view)
                        z_max = min(self.viewer.data.shape[0] - 1, z_max_view)
                    except Exception:
                        pass
                new_region = CubeRegion(xy=self.start_point, width=0, height=0, z_min=z_min, z_max=z_max)
            else:
                new_region = CircleRegion(center=self.start_point, radius=0)

            target_ax = getattr(self.viewer, 'overlay_ax', self.viewer.ax)
            new_region.add_to_axes(target_ax)
            self.active_region = new_region
            self.regions.append(new_region)

        self._request_overlay_redraw()


    def handle_motion(self, event):
        if getattr(event, 'inaxes', None) is None:
            if self.is_dragging or self.is_resizing or self.is_rotating:
                xdata, ydata = self._extract_data_coords(event)
                if xdata is None or ydata is None:
                    return
            else:
                return

        if self.is_rotating:
            self._apply_rotation(event)
            return

        if self.is_dragging and self.selected_region:
            if not self.drag_initiated:
                canvas = getattr(self.viewer, 'canvas', None)
                if canvas is not None:
                    canvas.setCursor(Qt.CursorShape.ClosedHandCursor)
                self.drag_initiated = True
            
            if self.start_point is None or self.drag_start_region_pos is None:
                return

            xdata, ydata = self._extract_data_coords(event)
            if xdata is None or ydata is None:
                return

            dx = xdata - self.start_point[0]
            dy = ydata - self.start_point[1]
            
            # Check for CubeRegion BEFORE RectangleRegion
            if isinstance(self.selected_region, CubeRegion):
                new_pos_x = self.drag_start_region_pos['xy'][0] + dx
                new_pos_y = self.drag_start_region_pos['xy'][1] + dy
                self.selected_region.xy = (new_pos_x, new_pos_y)
                self.selected_region.z_min = self.drag_start_region_pos['z_min']
                self.selected_region.z_max = self.drag_start_region_pos['z_max']
            elif isinstance(self.selected_region, RectangleRegion):
                new_pos_x = self.drag_start_region_pos['xy'][0] + dx
                new_pos_y = self.drag_start_region_pos['xy'][1] + dy
                self.selected_region.xy = (new_pos_x, new_pos_y)
            elif isinstance(self.selected_region, EllipseRegion):
                new_pos_x = self.drag_start_region_pos['center'][0] + dx
                new_pos_y = self.drag_start_region_pos['center'][1] + dy
                self.selected_region.center = (new_pos_x, new_pos_y)
            elif isinstance(self.selected_region, CircleRegion):
                new_pos_x = self.drag_start_region_pos['center'][0] + dx
                new_pos_y = self.drag_start_region_pos['center'][1] + dy
                self.selected_region.center = (new_pos_x, new_pos_y)

            self.selected_region.update_visual()
            self._request_overlay_redraw()
            self._notify_region_changed(self.selected_region)
            return

        if self.is_resizing and self.selected_region:
            self._handle_resize_motion(event)
            return

        if not self.is_drawing or not self.active_region:
            return

        if self.region_mode == "circle":
            xdata, ydata = self._extract_data_coords(event)
            if xdata is None or ydata is None:
                return
            bounds = self._get_axis_bounds()
            if bounds is not None:
                xmin, xmax, ymin, ymax = bounds
                xdata = min(max(xdata, xmin), xmax)
                ydata = min(max(ydata, ymin), ymax)
            dx = xdata - self.start_point[0]
            dy = ydata - self.start_point[1]
            radius = np.sqrt(dx**2 + dy**2)
            self.active_region.radius = radius
        
        elif self.region_mode == "rectangle" or self.region_mode == "cube":
            xdata, ydata = self._extract_data_coords(event)
            if xdata is None or ydata is None:
                return
            x0, y0 = self.start_point
            bounds = self._get_axis_bounds()
            if bounds is not None:
                xmin, xmax, ymin, ymax = bounds
                x1 = min(max(xdata, xmin), xmax)
                y1 = min(max(ydata, ymin), ymax)
            else:
                x1, y1 = xdata, ydata

            new_x = min(x0, x1)
            new_y = min(y0, y1)
            width = abs(x0 - x1)
            height = abs(y0 - y1)
            
            self.active_region.xy = (new_x, new_y)
            self.active_region.width = width
            self.active_region.height = height

        elif self.region_mode == "ellipse":
            xdata, ydata = self._extract_data_coords(event)
            if xdata is None or ydata is None:
                return
            x0, y0 = self.start_point
            bounds = self._get_axis_bounds()
            if bounds is not None:
                xmin, xmax, ymin, ymax = bounds
                x1 = min(max(xdata, xmin), xmax)
                y1 = min(max(ydata, ymin), ymax)
            else:
                x1, y1 = xdata, ydata

            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
            width = abs(x0 - x1)
            height = abs(y0 - y1)

            self.active_region.center = (cx, cy)
            self.active_region.width = width
            self.active_region.height = height

        self.active_region.update_visual()
        self._request_overlay_redraw()

    def handle_release(self, event):
        # Handles mouse release to finalize drawing or dragging.
        if event.button != 1:
            return

        if self.is_rotating:
            self.is_rotating = False
            self.rotation_center = None
            self.rotation_initial_angle = None
            self.rotation_reference_angle = None
            canvas = getattr(self.viewer, 'canvas', None)
            if canvas is not None:
                canvas.setCursor(Qt.CursorShape.ArrowCursor)
            if self.selected_region:
                self._notify_region_changed(self.selected_region)
            self._request_overlay_redraw()
            return

        if self.is_dragging:
            self.is_dragging = False
            self.drag_initiated = False # Reset the flag
            self.drag_start_region_pos = None
            self.start_point = None
            canvas = getattr(self.viewer, 'canvas', None)
            if canvas is not None:
                canvas.setCursor(Qt.CursorShape.ArrowCursor)
            if self.selected_region:
                self._notify_region_changed(self.selected_region)
                self.update_hover_cursor(event) # Update cursor after drag
            self._request_overlay_redraw()
            return

        if self.is_resizing:
            self.is_resizing = False
            self.resize_handle = None
            self.resize_initial_state = None
            self.start_point = None
            canvas = getattr(self.viewer, 'canvas', None)
            if canvas is not None:
                canvas.setCursor(Qt.CursorShape.ArrowCursor)
            if self.selected_region:
                self._notify_region_changed(self.selected_region)
                self.update_hover_cursor(event) # Update cursor after resize
            self._request_overlay_redraw()
            return

        if not self.is_drawing or not self.active_region:
            return

        is_valid_region = False
        if isinstance(self.active_region, CircleRegion) and self.active_region.radius > 0.5:
            is_valid_region = True
        elif isinstance(self.active_region, RectangleRegion) and self.active_region.width > 0.5 and self.active_region.height > 0.5:
            is_valid_region = True
        elif isinstance(self.active_region, EllipseRegion) and self.active_region.width > 0.5 and self.active_region.height > 0.5:
            is_valid_region = True

        redraw_required = False
        if self.active_region and is_valid_region:
            if self.active_region.region_id is None:
                self.active_region.region_id = RegionManager._global_region_counter
                RegionManager._global_region_counter += 1
            redraw_required = True
            self._notify_region_changed(self.active_region)

        elif self.active_region:
            self.active_region.remove_from_axes()
            self.regions.remove(self.active_region)
            redraw_required = True

        if redraw_required or not self.active_region:
            self._request_overlay_redraw()

        # Finalize internal state.
        self.active_region = None
        self.is_drawing = False
        self.start_point = None

    def select_region(self, region_to_select):
        # Selects a region.
        self.deselect_all(skip_redraw=True)
        self.selected_region = region_to_select
        if self.selected_region:
            self.selected_region.set_selected(True)
            self.selected_region.mpl_patch.set_animated(True) 
            self.selected_region.update_visual()
        self._request_overlay_redraw()
        self._notify_region_changed(self.selected_region)

    def deselect_all(self, *, skip_redraw=False):
        # Deselects all regions.
        for r in self.regions:
            if r.selected:
                r.set_selected(False)
                r.update_visual()
        self.selected_region = None
        self.is_resizing = False
        self.resize_handle = None
        self.resize_initial_state = None
        if not skip_redraw:
            canvas = getattr(self.viewer, 'canvas', None)
            if canvas is not None:
                canvas.setCursor(Qt.CursorShape.ArrowCursor)
            self._request_overlay_redraw()
            self._notify_region_changed(None)
    
    def delete_selected_region(self):
        """
        Deletes the currently selected region.
        """
        if self.selected_region:
            self.delete_region(self.selected_region)
            canvas = getattr(self.viewer, 'canvas', None)
            if canvas is not None:
                canvas.setCursor(Qt.CursorShape.ArrowCursor)

    def handle_key_press(self, event):
        key = getattr(event, 'key', None)
        if not key:
            return
        key_lower = str(key).lower()
        if 'shift' in key_lower:
            self.shift_pressed = True

    def handle_key_release(self, event):
        key = getattr(event, 'key', None)
        if not key:
            return
        key_lower = str(key).lower()
        if 'shift' in key_lower:
            self.shift_pressed = False
            if not (self.is_dragging or self.is_resizing or self.is_drawing or self.is_rotating):
                canvas = getattr(self.viewer, 'canvas', None)
                if canvas is not None:
                    canvas.setCursor(Qt.CursorShape.ArrowCursor)

    def prepare_for_background_capture(self):
        """Temporarily hides regions so they are not baked into the blit background."""
        hidden_items = []
        for region in self.regions:
            patch = getattr(region, 'mpl_patch', None)
            if patch is None:
                continue
            if not patch.get_animated():
                patch.set_animated(True)
            if patch.get_visible():
                patch.set_visible(False)
                hidden_items.append(patch)
            label = getattr(region, 'label_artist', None)
            if label is not None and label.get_visible():
                label.set_visible(False)
                hidden_items.append(label)
        return hidden_items

    def restore_after_background_capture(self, hidden_patches):
        """Restores region visibility after the background has been captured."""
        for patch in hidden_patches:
            patch.set_visible(True)

    def draw_regions_for_blit(self):
        # Draws all region artists for blitting.
        overlay_ax = getattr(self.viewer, 'overlay_ax', getattr(self.viewer, 'ax', None))
        for region in self.regions:
            patch_axes = getattr(region.mpl_patch, 'axes', overlay_ax)
            if patch_axes is not None:
                patch_axes.draw_artist(region.mpl_patch)
            if region.label_artist is not None and region.label_artist.get_visible():
                label_axes = getattr(region.label_artist, 'axes', overlay_ax)
                if label_axes is not None:
                    label_axes.draw_artist(region.label_artist)

    def open_region_editor(self, region):
        if region not in self.regions:
            return
        if region.region_id is None:
            region.region_id = RegionManager._global_region_counter
            RegionManager._global_region_counter += 1
        dialog = self.region_editors.get(region)
        if dialog is None:
            from takefits.ui.region_editor import RegionEditorDialog
            dialog = RegionEditorDialog(self.viewer, region, self)
            self.region_editors[region] = dialog
        self._position_region_editor(dialog)
        dialog.update_from_region()
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def on_editor_closed(self, region, dialog):
        existing = self.region_editors.get(region)
        if existing is dialog:
            self.region_editors.pop(region, None)
        if hasattr(dialog, 'offset_index'):
            dialog.offset_index = None

    def _close_region_editor(self, region):
        dialog = self.region_editors.pop(region, None)
        if dialog is not None:
            dialog.region_manager = None
            dialog.close()

    def update_region_from_editor(self, region, params):
        if region not in self.regions:
            return
        min_size = 1e-3
        label = params.get('label')
        if label is not None:
            region.set_label_text(label)
        if isinstance(region, CircleRegion):
            center = params.get('center', region.center)
            radius = max(params.get('radius', region.radius), min_size)
            region.center = center
            region.radius = radius
        elif isinstance(region, CubeRegion):
            center = params.get('center', region.center)
            width = max(params.get('width', region.width), min_size)
            height = max(params.get('height', region.height), min_size)
            angle = params.get('angle', region.angle)
            cx, cy = center
            region.width = width
            region.height = height
            region.xy = (cx - width / 2.0, cy - height / 2.0)
            region.set_angle(angle)
            region.z_min = params.get('z_min', region.z_min)
            region.z_max = params.get('z_max', region.z_max)

        elif isinstance(region, RectangleRegion):
            center = params.get('center', region.center)
            width = max(params.get('width', region.width), min_size)
            height = max(params.get('height', region.height), min_size)
            angle = params.get('angle', region.angle)
            cx, cy = center
            region.width = width
            region.height = height
            region.xy = (cx - width / 2.0, cy - height / 2.0)
            region.set_angle(angle)

        elif isinstance(region, EllipseRegion):
            center = params.get('center', region.center)
            width = max(params.get('width', region.width), min_size)
            height = max(params.get('height', region.height), min_size)
            angle = params.get('angle', region.angle)
            region.center = center
            region.width = width
            region.height = height
            region.set_angle(angle)
        else:
            return

        style_params = params.get('style')
        if isinstance(style_params, dict):
            region.update_style_attributes(
                color=style_params.get('color'),
                linewidth=style_params.get('linewidth'),
                linestyle=style_params.get('linestyle')
            )

        region.update_visual()
        self._request_overlay_redraw()
        # Notify listeners (spectrum panel, AppState bridge, etc.). Avoid re-entering editor updates.
        try:
            self.selected_region_changed.emit(region)
        except Exception:
            pass

    def delete_region(self, region):
        if region not in self.regions:
            return
        was_selected = region is self.selected_region
        self._close_region_editor(region)
        region.remove_from_axes()
        self.regions.remove(region)
        if was_selected:
            self.selected_region = None
            self.is_dragging = False
            self.is_resizing = False
            self.is_rotating = False
            self.resize_handle = None
            self.resize_initial_state = None
            self.rotation_center = None
            self.rotation_initial_angle = None
            self.rotation_reference_angle = None
        self._request_overlay_redraw()
        self._notify_region_changed(None)

    def _position_region_editor(self, dialog):
        viewer_widget = getattr(self.viewer, 'window', None)
        if callable(viewer_widget):
            viewer_widget = viewer_widget()
        if viewer_widget is None:
            viewer_widget = getattr(self.viewer, 'parentWidget', lambda: None)()
        if viewer_widget is None:
            return

        base_geom = viewer_widget.frameGeometry()
        base_point = base_geom.topRight()
        margin = 20
        step = 30

        visible_dialogs = [dlg for dlg in self.region_editors.values()
                           if dlg is not None and dlg.isVisible() and dlg is not dialog]

        if dialog.offset_index is None:
            indices = sorted(set(getattr(dlg, 'offset_index', idx)
                                  for idx, dlg in enumerate(visible_dialogs)))
            next_index = 0
            while next_index in indices:
                next_index += 1
            dialog.offset_index = next_index

        x = base_point.x() + margin + dialog.offset_index * step
        y = base_point.y() + margin + dialog.offset_index * step
        dialog.move(x, y)

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
            formatted = f"{value:.{digits}f}"
            # Remove trailing zeros and decimal point if not needed
            return formatted.rstrip('0').rstrip('.')
        
        # For numbers < 1, use 4 significant digits
        decimal_places = digits - 1 - magnitude
        decimal_places = max(0, decimal_places)
        
        return f"{value:.{decimal_places}f}"
    

    def delete_all_regions(self):
        """
        Deletes all regions currently managed by the manager.
        """
        for region in list(self.regions):
            self.delete_region(region)

    # ------------------------------------------------------------------
    # Plane/axes helpers for persistence
    def _axes_for_plane(self, plane: str):
        """Resolve overlay axes for a plane name."""
        plane_lower = (plane or "xy").lower()
        viewer = self.viewer
        if viewer is not None:
            if plane_lower == "xy":
                return getattr(viewer, "overlay_ax", getattr(viewer, "overlay_ax_xy", None))
            if plane_lower == "xz":
                return getattr(viewer, "overlay_ax_xz", None)
            if plane_lower == "zy":
                return getattr(viewer, "overlay_ax_zy", None)
                return getattr(viewer, "overlay_ax_zy", None)
        return None

    def _plane_for_axes(self, axes) -> str:
        """Infer plane name from axes reference."""
        if axes is None:
            return "xy"
        viewer = self.viewer
        mapping = {}
        if viewer is not None:
            mapping.update({
                getattr(viewer, "overlay_ax", None): "xy",
                getattr(viewer, "overlay_ax_xy", None): "xy",
                getattr(viewer, "overlay_ax_xz", None): "xz",
                getattr(viewer, "overlay_ax_zy", None): "zy",
            })

        return mapping.get(axes, "xy")

    def _region_kind(self, region: Region) -> str:
        if isinstance(region, CircleRegion):
            return "circle"
        if isinstance(region, CubeRegion):
            return "cube"
        if isinstance(region, RectangleRegion):
            return "rectangle"
        if isinstance(region, EllipseRegion):
            return "ellipse"
        return region.__class__.__name__.lower()

    # ------------------------------------------------------------------
    # Persistence helpers
    def _plane_axis_indices(self, plane: str, wcs) -> tuple[int, int] | None:
        if wcs is None:
            return None
        plane_lower = (plane or "xy").lower()
        mapping = {"xy": (0, 1), "xz": (0, 2), "zy": (2, 1)}
        axes = mapping.get(plane_lower)
        if axes is None:
            return None
        naxis = getattr(wcs, "naxis", 0) or 0
        if naxis <= max(axes):
            return None
        return axes

    def _pixel_scales_deg(self, plane: str, wcs) -> tuple[float, float] | None:
        indices = self._plane_axis_indices(plane, wcs)
        if indices is None:
            return None
        try:
            scales = proj_plane_pixel_scales(wcs)
            if scales is None or len(scales) <= max(indices):
                return None
            vals: list[float] = []
            for idx in indices:
                val = scales[idx]
                try:
                    vals.append(abs(val.to_value("deg")))
                except Exception:
                    vals.append(abs(float(val)))
            return (vals[0], vals[1])
        except Exception:
            return None

    def _world_frame_from_wcs(self, wcs) -> str | None:
        if wcs is None:
            return None
        try:
            frame = wcs_to_celestial_frame(wcs)
            if frame is None:
                return None
            name = getattr(frame, "name", None) or frame.__class__.__name__
            return str(name)
        except Exception:
            return None

    def _context_axis_value(self, world_context, axis_index: int):
        if isinstance(world_context, dict):
            for key in (axis_index, str(axis_index)):
                if key in world_context:
                    try:
                        return float(world_context[key])
                    except Exception:
                        return None
            return None
        if isinstance(world_context, (list, tuple)):
            if 0 <= axis_index < len(world_context):
                try:
                    value = world_context[axis_index]
                    if value is None:
                        return None
                    return float(value)
                except Exception:
                    return None
        return None

    def _shared_world_axis_value(self, axis_index: int):
        viewer = self.viewer
        if viewer is None:
            return None
        getter_names = (
            "_get_shared_world_x",
            "_get_shared_world_y",
            "_get_shared_world_z",
            "_get_shared_world_s",
        )
        if axis_index < 0 or axis_index >= len(getter_names):
            return None
        getter = getattr(viewer, getter_names[axis_index], None)
        if not callable(getter):
            return None
        try:
            value = getter()
        except Exception:
            return None
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None

    def _shared_world_context(self, naxis: int):
        context = {}
        limit = max(int(naxis), 0)
        for axis_index in range(limit):
            value = self._shared_world_axis_value(axis_index)
            if value is None:
                continue
            context[str(axis_index)] = float(value)
        return context

    def _world_center_to_pix(self, world_center, source_frame: str | None, plane: str, wcs, world_context=None):
        """Convert a 2-tuple world center to pixel using the target WCS and plane axes."""
        if world_center is None or wcs is None:
            return None
        axes = self._plane_axis_indices(plane, wcs)
        if axes is None:
            return None
        try:
            target_frame = wcs_to_celestial_frame(wcs)
        except Exception:
            target_frame = None
        lon, lat = world_center[:2]
        # Transform if frames differ and celestial frame is available.
        if target_frame is not None and source_frame:
            try:
                sky = SkyCoord(lon, lat, unit=u.deg, frame=source_frame.lower())
                sky_t = sky.transform_to(target_frame)
                lon = sky_t.spherical.lon.deg
                lat = sky_t.spherical.lat.deg
            except Exception:
                pass
        try:
            naxis = max(getattr(wcs, "naxis", 2), max(axes) + 1)
            vec = [0.0] * naxis
            # Fill non-plane axes from saved world-context first, then current shared world.
            for axis_index in range(naxis):
                value = self._context_axis_value(world_context, axis_index)
                if value is None:
                    value = self._shared_world_axis_value(axis_index)
                vec[axis_index] = 0.0 if value is None else float(value)
            vec[axes[0]] = float(lon)
            vec[axes[1]] = float(lat)
            pix = wcs.wcs_world2pix([vec], 0)[0]
            return (float(pix[axes[0]]), float(pix[axes[1]]))
        except Exception:
            return None

    def _rotation_transform(self, angle_deg: float, source_frame: str | None, target_frame) -> float:
        """
        Approximate rotation transform between frames.
        Assumes small-angle local projection; if frames differ, reuse the same angle.
        """
        if source_frame is None or target_frame is None:
            return angle_deg
        try:
            src_name = str(source_frame).lower()
            tgt_name = str(getattr(target_frame, "name", None) or target_frame.__class__.__name__).lower()
            if src_name == tgt_name:
                return angle_deg
        except Exception:
            return angle_deg
        # Fallback: keep the angle; refining would require Jacobian at center.
        return angle_deg

    def _transform_angle_via_center(
        self,
        angle_deg: float,
        center_world,
        source_frame: str | None,
        plane: str,
        wcs,
        world_context=None,
    ) -> float:
        """Transform an orientation angle using local projection around center."""
        if angle_deg is None or center_world is None or wcs is None:
            return angle_deg
        axes = self._plane_axis_indices(plane, wcs)
        if axes is None:
            return angle_deg
        try:
            # Build a small offset along the angle direction in the source frame.
            lon, lat = center_world[:2]
            src_frame = source_frame.lower() if source_frame else None
            sky_center = SkyCoord(lon * u.deg, lat * u.deg, frame=src_frame) if src_frame else SkyCoord(lon * u.deg, lat * u.deg)
            # 1 arcsec offset along angle
            offset = sky_center.directional_offset_by(angle_deg * u.deg, 1.0 * u.arcsec)
            tgt_frame = wcs_to_celestial_frame(wcs)
            if tgt_frame is not None:
                sky_center = sky_center.transform_to(tgt_frame)
                offset = offset.transform_to(tgt_frame)
            # Convert to pixel in target WCS
            naxis = max(getattr(wcs, "naxis", 2), max(axes) + 1)
            vec_center = [0.0] * naxis
            vec_center[axes[0]] = float(sky_center.spherical.lon.deg)
            vec_center[axes[1]] = float(sky_center.spherical.lat.deg)
            for axis_index in range(naxis):
                if axis_index in axes:
                    continue
                value = self._context_axis_value(world_context, axis_index)
                if value is None:
                    value = self._shared_world_axis_value(axis_index)
                if value is not None:
                    vec_center[axis_index] = float(value)
            pix_c = wcs.wcs_world2pix([vec_center], 0)[0]
            vec_off = [0.0] * naxis
            vec_off[axes[0]] = float(offset.spherical.lon.deg)
            vec_off[axes[1]] = float(offset.spherical.lat.deg)
            for axis_index in range(naxis):
                if axis_index in axes:
                    continue
                value = self._context_axis_value(world_context, axis_index)
                if value is None:
                    value = self._shared_world_axis_value(axis_index)
                if value is not None:
                    vec_off[axis_index] = float(value)
            pix_o = wcs.wcs_world2pix([vec_off], 0)[0]
            dx = float(pix_o[axes[0]] - pix_c[axes[0]])
            dy = float(pix_o[axes[1]] - pix_c[axes[1]])
            if math.isfinite(dx) and math.isfinite(dy) and abs(dx) + abs(dy) > 0:
                return float(math.degrees(math.atan2(dy, dx)))
        except Exception:
            return angle_deg
        return angle_deg

    def export_regions_to_dict(self, plane: str | None = None) -> dict:
        """Serialize regions (all or by plane) to a dict, embedding world info when possible."""
        entries = []
        target_plane = plane.lower() if plane else None
        viewer = self.viewer
        wcs = getattr(viewer, "wcs", None) if viewer is not None else None
        world_frame = self._world_frame_from_wcs(wcs)
        shared_context = self._shared_world_context(max(getattr(wcs, "naxis", 2), 2)) if wcs is not None else {}
        for region in self.regions:
            plane_name = getattr(region, "plane", None) or self._plane_for_axes(region.axes)
            if target_plane and plane_name != target_plane:
                continue
            state = region.get_state()
            world_state = {}
            axes = self._plane_axis_indices(plane_name, wcs)
            # Center in world coordinates
            center = None
            if hasattr(region, "center"):
                center = getattr(region, "center", None)
            elif hasattr(region, "xy") and hasattr(region, "width") and hasattr(region, "height"):
                xy = getattr(region, "xy", None)
                if xy is not None:
                    center = (xy[0] + region.width / 2.0, xy[1] + region.height / 2.0)
            if wcs is not None and axes is not None and center is not None:
                try:
                    pix_vec = [0.0] * max(getattr(wcs, "naxis", 2), 2)
                    pix_vec[axes[0]] = float(center[0])
                    pix_vec[axes[1]] = float(center[1])
                    world_vec = wcs.wcs_pix2world([pix_vec], 0)[0]
                    world_state["center"] = [float(world_vec[axes[0]]), float(world_vec[axes[1]])]
                    if shared_context:
                        world_state["context_world"] = dict(shared_context)
                except Exception:
                    world_state = {}
            scales = self._pixel_scales_deg(plane_name, wcs)
            if scales is not None:
                if isinstance(region, CircleRegion):
                    world_state["radius_deg"] = float(region.radius) * scales[0]
                elif isinstance(region, (RectangleRegion, EllipseRegion, CubeRegion)):
                    world_state["width_deg"] = float(region.width) * scales[0]
                    world_state["height_deg"] = float(region.height) * scales[1]
                    world_state["angle_deg"] = float(getattr(region, "angle", 0.0) or 0.0)
                    # Capture axis endpoints in world to preserve orientation/scale under frame changes
                    try:
                        rad = math.radians(float(getattr(region, "angle", 0.0) or 0.0))
                        half_w = float(region.width) / 2.0
                        half_h = float(region.height) / 2.0
                        dx_w = half_w * math.cos(rad)
                        dy_w = half_w * math.sin(rad)
                        dx_h = -half_h * math.sin(rad)
                        dy_h = half_h * math.cos(rad)
                        cx, cy = center
                        width_pt = (cx + dx_w, cy + dy_w)
                        height_pt = (cx + dx_h, cy + dy_h)
                        vec_w = [0.0] * max(getattr(wcs, "naxis", 2), 2)
                        vec_h = list(vec_w)
                        vec_w[axes[0]], vec_w[axes[1]] = width_pt
                        vec_h[axes[0]], vec_h[axes[1]] = height_pt
                        world_w = wcs.wcs_pix2world([vec_w], 0)[0]
                        world_h = wcs.wcs_pix2world([vec_h], 0)[0]
                        world_state["axes"] = {
                            "width": [float(world_w[axes[0]]), float(world_w[axes[1]])],
                            "height": [float(world_h[axes[0]]), float(world_h[axes[1]])],
                        }
                    except Exception:
                        pass
            entries.append(
                {
                    "id": getattr(region, "region_id", None),
                    "kind": self._region_kind(region),
                    "plane": plane_name,
                    "state": state,
                    "world": world_state,
                }
            )
        return {
            "format": "takefits.region",
            "version": 1,
            "plane": target_plane or "all",
            "regions": entries,
            "world_frame": world_frame,
        }

    def _region_from_state(self, kind: str, state: dict) -> Region | None:
        if kind == "circle":
            region = CircleRegion(state.get("center", (0, 0)), state.get("radius", 1.0), style=state.get("style"))
        elif kind == "rectangle":
            region = RectangleRegion(state.get("xy", (0, 0)), state.get("width", 1.0), state.get("height", 1.0), style=state.get("style"))
            region.set_angle(state.get("angle", 0.0))
        elif kind == "ellipse":
            region = EllipseRegion(state.get("center", (0, 0)), state.get("width", 1.0), state.get("height", 1.0), style=state.get("style"))
            region.set_angle(state.get("angle", 0.0))
        elif kind == "cube":
            region = CubeRegion(state.get("xy", (0, 0)), state.get("width", 1.0), state.get("height", 1.0), state.get("z_min", 0), state.get("z_max", 1), style=state.get("style"))
            region.set_angle(state.get("angle", 0.0))
        else:
            return None
        label = state.get("label", "")
        if label:
            region.set_label_text(label)
        return region

    def import_regions_from_dict(self, payload: dict, clear_existing: bool = True) -> None:
        """Load regions from serialized dict. Falls back to world geometry when available."""
        if payload.get("format") != "takefits.region":
            raise ValueError("Unsupported region file format")
        version = int(payload.get("version", 0))
        if version != 1:
            raise ValueError(f"Unsupported region format version: {version}")
        regions_payload = payload.get("regions") or []
        source_frame = payload.get("world_frame")
        if clear_existing:
            self.delete_all_regions()
        added_any = False
        viewer = self.viewer
        wcs = getattr(viewer, "wcs", None) if viewer is not None else None
        for entry in regions_payload:
            kind = entry.get("kind")
            plane = (entry.get("plane") or payload.get("plane") or "xy").lower()
            state = entry.get("state") or {}
            world_state = entry.get("world") or {}
            world_context = world_state.get("context_world")
            axes = self._plane_axis_indices(plane, wcs)
            scales = self._pixel_scales_deg(plane, wcs) if wcs is not None else None
            # Apply world geometry if available
            center_world = world_state.get("center")
            center_pix = None
            if axes is not None and wcs is not None and center_world and len(center_world) >= 2:
                pix_center = self._world_center_to_pix(center_world, source_frame, plane, wcs, world_context=world_context)
                if pix_center is not None:
                    cx, cy = pix_center
                    center_pix = (cx, cy)
                    if "center" in state:
                        state["center"] = (cx, cy)
                    if "width" in state and "height" in state:
                        state["xy"] = (cx - state["width"] / 2.0, cy - state["height"] / 2.0)
            if scales is not None:
                if "radius_deg" in world_state:
                    try:
                        state["radius"] = float(world_state["radius_deg"]) / scales[0]
                    except Exception:
                        pass
                if "width_deg" in world_state and "height_deg" in world_state:
                    try:
                        state["width"] = float(world_state["width_deg"]) / scales[0]
                        state["height"] = float(world_state["height_deg"]) / scales[1]
                    except Exception:
                        pass
                if "angle_deg" in world_state:
                    ang = float(world_state["angle_deg"])
                    ang = self._transform_angle_via_center(
                        ang,
                        center_world,
                        source_frame,
                        plane,
                        wcs,
                        world_context=world_context,
                    )
                    state["angle"] = ang
            # If axis endpoints are available, prefer them to recover width/height/angle
            axes_world = world_state.get("axes") if isinstance(world_state, dict) else None
            if axes_world and axes is not None and wcs is not None:
                width_world = axes_world.get("width")
                height_world = axes_world.get("height")
                if center_pix is None and center_world is not None:
                    center_pix = self._world_center_to_pix(center_world, source_frame, plane, wcs, world_context=world_context)
                if center_pix is not None:
                    cx, cy = center_pix
                    def _world_pt_to_pix(pt):
                        if pt is None or len(pt) < 2:
                            return None
                        return self._world_center_to_pix(pt, source_frame, plane, wcs, world_context=world_context)
                    w_pix = _world_pt_to_pix(width_world)
                    h_pix = _world_pt_to_pix(height_world)
                    if w_pix is not None and h_pix is not None:
                        wx, wy = w_pix
                        hx, hy = h_pix
                        vx = wx - cx
                        vy = wy - cy
                        ux = hx - cx
                        uy = hy - cy
                        width_len = math.hypot(vx, vy) * 2.0
                        height_len = math.hypot(ux, uy) * 2.0
                        if width_len > 0 and height_len > 0:
                            state["width"] = width_len
                            state["height"] = height_len
                            state["angle"] = math.degrees(math.atan2(vy, vx))
                            state["center"] = (cx, cy)
                            state["xy"] = (cx - width_len / 2.0, cy - height_len / 2.0)
            region = self._region_from_state(kind, state)
            if region is None:
                continue
            axes = self._axes_for_plane(plane)
            if axes is None:
                continue
            region.plane = plane
            region.region_id = RegionManager._global_region_counter
            RegionManager._global_region_counter += 1
            region.add_to_axes(axes)
            region.update_visual() if hasattr(region, "update_visual") else None
            self.regions.append(region)
            added_any = True
        if added_any:
            self._request_overlay_redraw()
        else:
            raise ValueError("No regions could be loaded for this viewer.")
