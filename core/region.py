from abc import ABC, abstractmethod
import math
import numpy as np
from matplotlib.patches import Circle, Polygon, Ellipse
from matplotlib.text import Text
from PyQt6.QtCore import Qt

class Region(ABC):
    """
    Abstract base class for all region types.
    """
    def __init__(self, style=None):
        self.default_style = {'color': 'lime', 'linewidth': 1, 'fill': False}
        self.selected_style = {'color': 'yellow', 'linewidth': 2, 'fill': False}
        self.style = style if style is not None else self.default_style.copy()
        
        self.mpl_patch = None
        self.selected = False
        self.label_text = ""
        self.label_artist: Text | None = None
        self.axes = None
        self.region_id = None

    @abstractmethod
    def contains(self, x, y):
        # Check if a point (x, y) is inside the region.
        pass

    def get_resize_handle(self, event, tolerance):
        return None

    @abstractmethod
    def get_state(self):
        pass
        
    def set_selected(self, selected):
        self.selected = selected
        self.style = self.selected_style.copy() if selected else self.default_style.copy()

    def add_to_axes(self, ax):
        self.axes = ax
        if self.mpl_patch is not None:
            ax.add_patch(self.mpl_patch)
        self._ensure_label_artist()
        self._update_label_position()

    def remove_from_axes(self):
        if self.mpl_patch is not None:
            try:
                self.mpl_patch.remove()
            except ValueError:
                pass
        if self.label_artist is not None:
            try:
                self.label_artist.remove()
            except ValueError:
                pass
            self.label_artist = None
        self.axes = None

    def set_label_text(self, text: str):
        self.label_text = text.strip()
        self._ensure_label_artist()
        self._update_label_position()

    def _ensure_label_artist(self):
        if self.axes is None:
            return
        if self.label_artist is None:
            color = self.style.get('color', 'yellow')
            z = self.mpl_patch.get_zorder() + 1 if self.mpl_patch is not None else 10
            self.label_artist = self.axes.text(0, 0, '', color=color, fontsize=9,
                                               ha='center', va='bottom', visible=False,
                                               zorder=z)

    def _label_position(self):
        raise NotImplementedError

    def _update_label_position(self):
        if self.label_artist is None:
            return
        if not self.label_text:
            self.label_artist.set_visible(False)
            return
        x, y = self._label_position()
        self.label_artist.set_position((x, y))
        self.label_artist.set_text(self.label_text)
        if self.mpl_patch is not None:
            self.label_artist.set_color(self.style.get('color', 'yellow'))
            self.label_artist.set_zorder(self.mpl_patch.get_zorder() + 1)
        self.label_artist.set_visible(True)

    def _calculate_stats(self, region_pixels):
        empty_stats = {'total_pixel_count': 0, 'valid_pixel_count': 0, 'mean': np.nan, 'sum': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan}
        
        if region_pixels.size == 0:
            return empty_stats
            
        region_pixels_non_nan = region_pixels[~np.isnan(region_pixels)]
        
        if region_pixels_non_nan.size == 0:
            return {'total_pixel_count': region_pixels.size, 'valid_pixel_count': 0, 'mean': np.nan, 'sum': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan}

        return {
            'total_pixel_count': region_pixels.size,
            'valid_pixel_count': region_pixels_non_nan.size,
            'mean': np.nanmean(region_pixels),
            'sum': np.nansum(region_pixels),
            'std': np.nanstd(region_pixels),
            'min': np.nanmin(region_pixels),
            'max': np.nanmax(region_pixels)
        }


    def get_moments(self, data):
        """
        Calculates the intensity-weighted 2D moments for the region
        on the given 2D data array.
        """
        if data is None or data.ndim != 2: return {}
        y_coords, x_coords = np.indices(data.shape)
        mask = self.contains(x_coords, y_coords)
        if not np.any(mask): return {}
        
        valid_x, valid_y, intensities = x_coords[mask], y_coords[mask], data[mask]
        non_nan_mask = ~np.isnan(intensities)
        valid_x, valid_y, intensities = valid_x[non_nan_mask], valid_y[non_nan_mask], intensities[non_nan_mask]

        if intensities.size == 0: return {}
        total_intensity = np.sum(intensities)
        if total_intensity == 0: return {}

        mean_x = np.sum(intensities * valid_x) / total_intensity
        mean_y = np.sum(intensities * valid_y) / total_intensity
        var_x = np.sum(intensities * (valid_x - mean_x)**2) / total_intensity
        var_y = np.sum(intensities * (valid_y - mean_y)**2) / total_intensity
        
        return {
            'mean_x_pix': mean_x, 'mean_y_pix': mean_y,
            'sigma_x_pix': np.sqrt(var_x), 'sigma_y_pix': np.sqrt(var_y),
        }
    

class CircleRegion(Region):
    """
    A circular region defined by a center and a radius.
    """
    def __init__(self, center, radius, style=None):
        super().__init__(style)
        self.center = center
        self.radius = radius
        self.mpl_patch = Circle(self.center, self.radius, **self.style, animated=True)

    def add_to_axes(self, ax):
        super().add_to_axes(ax)

    def update_visual(self):
        self.mpl_patch.set_center(self.center)
        self.mpl_patch.set_radius(self.radius)
        self.mpl_patch.set(**self.style)
        self._update_label_position()

    def contains(self, x, y):
        cx, cy = self.center
        return (x - cx)**2 + (y - cy)**2 < self.radius**2

    def get_stats(self, data):
        if self.radius == 0:
            return self._calculate_stats(np.array([]))
        
        y_coords, x_coords = np.indices(data.shape)
        mask = self.contains(x_coords, y_coords)
        region_pixels = data[mask]
        
        return self._calculate_stats(region_pixels)

    def get_resize_handle(self, event, tolerance):
        if event.x is None or event.y is None:
            return None
        axes = self.mpl_patch.axes
        if axes is None:
            return None
        transform = axes.transData
        center_disp = transform.transform(self.center)
        edge_disp = transform.transform((self.center[0] + self.radius, self.center[1]))
        disp_radius = math.hypot(edge_disp[0] - center_disp[0], edge_disp[1] - center_disp[1])
        if disp_radius <= 0:
            return None
        dist = math.hypot(event.x - center_disp[0], event.y - center_disp[1])
        if abs(dist - disp_radius) <= tolerance:
            return {'type': 'circle', 'cursor': Qt.CursorShape.SizeAllCursor}
        return None

    def get_state(self):
        return {'center': tuple(self.center), 'radius': float(self.radius), 'label': self.label_text}

    def _label_position(self):
        cx, cy = self.center
        return (cx, cy + self.radius + 0.5)

    def fit_to_view(self, xlim, ylim):
        """Adjusts the circle to fit within the given x and y limits."""
        center_x = (xlim[0] + xlim[1]) / 2
        center_y = (ylim[0] + ylim[1]) / 2
        self.center = (center_x, center_y)

        width = abs(xlim[1] - xlim[0])
        height = abs(ylim[1] - ylim[0])

        self.radius = min(width, height) / 2

        self.update_visual()


class RectangleRegion(Region):
    """Axis-aligned rectangle that can be rotated around its center."""

    def __init__(self, xy, width, height, style=None):
        super().__init__(style)
        self.xy = xy
        self.width = width
        self.height = height
        self.angle = 0.0  # Degrees
        self._update_cached_trig()
        self.mpl_patch = Polygon(self._compute_vertices(), closed=True, **self.style, animated=True)

    def _update_cached_trig(self):
        rad = math.radians(self.angle)
        self._cos = math.cos(rad)
        self._sin = math.sin(rad)

    def _compute_vertices(self):
        cx, cy = self.center
        half_w = self.width / 2.0
        half_h = self.height / 2.0
        corners = [
            (-half_w, -half_h),
            (half_w, -half_h),
            (half_w, half_h),
            (-half_w, half_h)
        ]
        verts = []
        for dx, dy in corners:
            x = cx + self._cos * dx - self._sin * dy
            y = cy + self._sin * dx + self._cos * dy
            verts.append((x, y))
        return verts

    @property
    def center(self):
        return (self.xy[0] + self.width / 2.0, self.xy[1] + self.height / 2.0)

    @center.setter
    def center(self, value):
        cx, cy = value
        self.xy = (cx - self.width / 2.0, cy - self.height / 2.0)

    def _to_local(self, x, y):
        cx, cy = self.center
        dx = np.asarray(x) - cx
        dy = np.asarray(y) - cy
        x_local = self._cos * dx + self._sin * dy
        y_local = -self._sin * dx + self._cos * dy
        return x_local, y_local

    def _to_global(self, x_local, y_local):
        cx, cy = self.center
        x = cx + self._cos * x_local - self._sin * y_local
        y = cy + self._sin * x_local + self._cos * y_local
        return x, y

    def update_visual(self):
        self._update_cached_trig()
        self.mpl_patch.set(**self.style)
        self.mpl_patch.set_xy(self._compute_vertices())
        self._update_label_position()

    def contains(self, x, y):
        if self.width == 0 or self.height == 0:
            return False
        half_w = self.width / 2.0
        half_h = self.height / 2.0
        x_local, y_local = self._to_local(x, y)
        mask = (np.abs(x_local) <= half_w) & (np.abs(y_local) <= half_h)
        if isinstance(mask, np.ndarray):
            if mask.ndim == 0:
                return bool(mask.item())
            return mask
        if isinstance(mask, np.bool_):
            return bool(mask)
        return mask

    def get_stats(self, data):
        if self.width == 0 or self.height == 0:
            return self._calculate_stats(np.array([]))
        grid_y, grid_x = np.indices(data.shape)
        mask = self.contains(grid_x, grid_y)
        region_pixels = data[mask]
        return self._calculate_stats(region_pixels)

    def get_resize_handle(self, event, tolerance):
        if event.x is None or event.y is None:
            return None
        axes = self.mpl_patch.axes
        if axes is None:
            return None

        verts = self._compute_vertices()
        corners = [axes.transData.transform(v) for v in verts]

        def point_segment_distance(px, py, ax, ay, bx, by):
            dx = bx - ax
            dy = by - ay
            if dx == 0 and dy == 0:
                return math.hypot(px - ax, py - ay)
            t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
            t = max(0.0, min(1.0, t))
            cx = ax + t * dx
            cy = ay + t * dy
            return math.hypot(px - cx, py - cy)

        edge_defs = [
            ('left', corners[0], corners[3]),
            ('right', corners[1], corners[2]),
            ('bottom', corners[0], corners[1]),
            ('top', corners[3], corners[2])
        ]

        edges = []
        for name, start, end in edge_defs:
            if point_segment_distance(event.x, event.y, start[0], start[1], end[0], end[1]) <= tolerance:
                edges.append(name)

        if not edges:
            return None

        edges_set = set(edges)
        horizontal = any(edge in edges_set for edge in ('left', 'right'))
        vertical = any(edge in edges_set for edge in ('top', 'bottom'))

        if horizontal and vertical:
            if ('left' in edges_set and 'top' in edges_set) or ('right' in edges_set and 'bottom' in edges_set):
                cursor = Qt.CursorShape.SizeFDiagCursor
            else:
                cursor = Qt.CursorShape.SizeBDiagCursor
        elif horizontal:
            cursor = Qt.CursorShape.SizeHorCursor
        elif vertical:
            cursor = Qt.CursorShape.SizeVerCursor
        else:
            cursor = Qt.CursorShape.SizeAllCursor

        return {'type': 'rectangle', 'edges': edges_set, 'cursor': cursor}

    def set_angle(self, angle_deg):
        self.angle = angle_deg % 360.0
        self.update_visual()

    def get_state(self):
        return {
            'xy': tuple(self.xy),
            'width': float(self.width),
            'height': float(self.height),
            'angle': float(self.angle),
            'center': self.center,
            'label': self.label_text
        }

    def _label_position(self):
        verts = self._compute_vertices()
        if not verts:
            return self.center
        sorted_by_y = sorted(verts, key=lambda v: v[1], reverse=True)
        top_vertices = sorted_by_y[:2] if len(sorted_by_y) >= 2 else sorted_by_y
        avg_x = sum(v[0] for v in top_vertices) / len(top_vertices)
        max_y = max(v[1] for v in top_vertices)
        return (avg_x, max_y + 0.5)

    def fit_to_view(self, xlim, ylim):
        """Adjusts the rectangle to fit the given x and y limits."""
        self.width = abs(xlim[1] - xlim[0])
        self.height = abs(ylim[1] - ylim[0])

        center_x = (xlim[0] + xlim[1]) / 2
        center_y = (ylim[0] + ylim[1]) / 2
        self.center = (center_x, center_y)

        self.angle = 0.0

        self.update_visual()


class EllipseRegion(Region):
    """Ellipse that supports rotation around its center."""

    def __init__(self, center, width, height, style=None):
        super().__init__(style)
        self.center = center
        self.width = width
        self.height = height
        self.angle = 0.0  # Degrees
        self._update_cached_trig()
        self.mpl_patch = Ellipse(self.center, self.width, self.height, angle=self.angle,
                                 **self.style, animated=True)

    def _update_cached_trig(self):
        rad = math.radians(self.angle)
        self._cos = math.cos(rad)
        self._sin = math.sin(rad)

    def _to_local(self, x, y):
        cx, cy = self.center
        dx = np.asarray(x) - cx
        dy = np.asarray(y) - cy
        x_local = self._cos * dx + self._sin * dy
        y_local = -self._sin * dx + self._cos * dy
        return x_local, y_local

    def _to_global(self, x_local, y_local):
        cx, cy = self.center
        x = cx + self._cos * x_local - self._sin * y_local
        y = cy + self._sin * x_local + self._cos * y_local
        return x, y

    def update_visual(self):
        self._update_cached_trig()
        self.mpl_patch.center = self.center
        self.mpl_patch.width = self.width
        self.mpl_patch.height = self.height
        self.mpl_patch.angle = self.angle
        self.mpl_patch.set(**self.style)
        self._update_label_position()

    def contains(self, x, y):
        if self.width == 0 or self.height == 0:
            return False
        half_w = self.width / 2.0
        half_h = self.height / 2.0
        if half_w == 0 or half_h == 0:
            return False
        x_local, y_local = self._to_local(x, y)
        value = (x_local / half_w) ** 2 + (y_local / half_h) ** 2
        if isinstance(value, np.ndarray):
            return value <= 1.0
        return value <= 1.0

    def get_stats(self, data):
        if self.width == 0 or self.height == 0:
            return self._calculate_stats(np.array([]))
        grid_y, grid_x = np.indices(data.shape)
        mask = self.contains(grid_x, grid_y)
        region_pixels = data[mask]
        return self._calculate_stats(region_pixels)

    def get_resize_handle(self, event, tolerance):
        if event.x is None or event.y is None:
            return None
        axes = self.mpl_patch.axes
        if axes is None:
            return None
        if self.width == 0 or self.height == 0:
            return None

        half_w = self.width / 2.0
        half_h = self.height / 2.0
        handle_defs = [
            ('left', (-half_w, 0.0)),
            ('right', (half_w, 0.0)),
            ('top', (0.0, half_h)),
            ('bottom', (0.0, -half_h))
        ]

        handle_tolerance = tolerance * 1.5
        handles = []
        for name, (hx, hy) in handle_defs:
            gx, gy = self._to_global(hx, hy)
            disp = axes.transData.transform((gx, gy))
            distance = math.hypot(event.x - disp[0], event.y - disp[1])
            if distance <= handle_tolerance:
                handles.append(name)

        if not handles:
            return None

        edges_set = set(handles)
        horizontal = any(edge in edges_set for edge in ('left', 'right'))
        vertical = any(edge in edges_set for edge in ('top', 'bottom'))

        if horizontal and vertical:
            cursor = Qt.CursorShape.SizeAllCursor
        elif horizontal:
            cursor = Qt.CursorShape.SizeHorCursor
        elif vertical:
            cursor = Qt.CursorShape.SizeVerCursor
        else:
            cursor = Qt.CursorShape.SizeAllCursor

        return {'type': 'ellipse', 'edges': edges_set, 'cursor': cursor}

    def set_angle(self, angle_deg):
        self.angle = angle_deg % 360.0
        self.update_visual()

    def get_state(self):
        return {
            'center': tuple(self.center),
            'width': float(self.width),
            'height': float(self.height),
            'angle': float(self.angle),
            'label': self.label_text
        }

    def _label_position(self):
        rad = math.radians(self.angle)
        cx, cy = self.center
        half_h = self.height / 2.0
        offset_x = math.sin(rad) * (half_h + 0.5)
        offset_y = math.cos(rad) * (half_h + 0.5)
        return (cx + offset_x, cy + offset_y)


    def fit_to_view(self, xlim, ylim):
        """Adjusts the ellipse to fit the given x and y limits."""
        self.center = ((xlim[0] + xlim[1]) / 2, (ylim[0] + ylim[1]) / 2)
        self.width = abs(xlim[1] - xlim[0])
        self.height = abs(ylim[1] - ylim[0])

        self.angle = 0.0

        self.update_visual()


class CubeRegion(RectangleRegion):
    """
    A 3D region based on a rectangle, with an added depth along the Z-axis.
    """
    def __init__(self, xy, width, height, z_min=0, z_max=1, style=None):
        # Initialize the parent RectangleRegion
        super().__init__(xy, width, height, style)

        # Add Z-axis attributes
        self.z_min = z_min
        self.z_max = z_max

    def get_state(self):
        """Returns the full state of the cube, including Z-range."""
        # Get the state from the parent RectangleRegion
        state = super().get_state()

        state.update({
            'z_min': float(self.z_min),
            'z_max': float(self.z_max),
        })
        return state

    def get_stats(self, data):
        """
        Calculates statistics for the data within the 3D bounds of the cube.
        Assumes 'data' is a 3D FITS data cube (Z, Y, X).
        """
        if self.width == 0 or self.height == 0 or data.ndim < 3:
            return self._calculate_stats(np.array([]))

        grid_y, grid_x = np.indices((data.shape[1], data.shape[2]))
        xy_mask = self.contains(grid_x, grid_y)

        if not np.any(xy_mask):
            return self._calculate_stats(np.array([]))

        z_start = max(0, int(round(self.z_min)))
        z_end = min(data.shape[0], int(round(self.z_max)) + 1)

        if z_start >= z_end:
            return self._calculate_stats(np.array([]))

        region_pixels = data[z_start:z_end, xy_mask]
        
        return self._calculate_stats(region_pixels.flatten())

    def fit_to_view(self, xlim, ylim, zlim):
        """Adjusts the cube to fit within the given x, y, and z limits."""
        super().fit_to_view(xlim, ylim)
        self.z_min = zlim[0]
        self.z_max = zlim[1]
        self.update_visual()


    def get_moments(self, data):
        """
        Calculates the intensity-weighted 3D moments for the CubeRegion.
        Overrides the base Region's 2D get_moments.
        """
        if data is None or data.ndim < 3:
            return super().get_moments(data) if data is not None and data.ndim == 2 else {}

        z_start, z_end = max(0, int(round(self.z_min))), min(data.shape[0], int(round(self.z_max)) + 1)
        if z_start >= z_end: return {}

        z, y, x = np.indices(data.shape)
        xy_mask_2d = self.contains(x[0], y[0])
        if not np.any(xy_mask_2d): return {}

        full_mask = ((z >= z_start) & (z < z_end)) & np.broadcast_to(xy_mask_2d, data.shape)
        
        vx, vy, vz, intensities = x[full_mask], y[full_mask], z[full_mask], data[full_mask]
        non_nan_mask = ~np.isnan(intensities)
        vx, vy, vz, intensities = vx[non_nan_mask], vy[non_nan_mask], vz[non_nan_mask], intensities[non_nan_mask]

        if intensities.size == 0: return {}
        total_intensity = np.sum(intensities)
        if total_intensity == 0: return {}

        mean_x = np.sum(intensities * vx) / total_intensity
        mean_y = np.sum(intensities * vy) / total_intensity
        mean_z = np.sum(intensities * vz) / total_intensity
        var_x = np.sum(intensities * (vx - mean_x)**2) / total_intensity
        var_y = np.sum(intensities * (vy - mean_y)**2) / total_intensity
        var_z = np.sum(intensities * (vz - mean_z)**2) / total_intensity

        return {
            'mean_x_pix': mean_x, 'mean_y_pix': mean_y, 'mean_z_pix': mean_z,
            'sigma_x_pix': np.sqrt(var_x), 'sigma_y_pix': np.sqrt(var_y), 'sigma_z_pix': np.sqrt(var_z),
        }