from abc import ABC, abstractmethod
import math
import numpy as np
from matplotlib.patches import Circle, Polygon, Ellipse
from matplotlib.text import Text


def _cursor_shapes():
    """Qt cursor shapes, imported lazily.

    Only the interactive hit-test helpers need them, and they all require a
    live Matplotlib mouse event. Keeping the import out of module scope lets
    headless renderers use the region geometry/artists without Qt.
    """
    from PySide6.QtCore import Qt

    return Qt.CursorShape


class Region(ABC):
    """
    Abstract base class for all region types.
    """
    def __init__(self, style=None):
        self.base_style = {'color': 'lime', 'linewidth': 1, 'fill': False, 'linestyle': 'solid'}
        if isinstance(style, dict):
            self._apply_style_overrides(style)
        self.selected_style = {}
        self.selected = False
        self.style = {}

        self.mpl_patch = None
        self.label_text = ""
        self.label_artist: Text | None = None
        self.axes = None
        self.region_id = None
        self._refresh_style_state(initial=True)

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
        self._refresh_style_state()

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

    def _apply_style_overrides(self, overrides):
        for key in ('color', 'linewidth', 'linestyle', 'fill'):
            if key in overrides and overrides[key] is not None:
                self.base_style[key] = overrides[key]

    def _refresh_style_state(self, initial=False):
        linewidth = float(self.base_style.get('linewidth', 1.0) or 1.0)
        linewidth = max(linewidth, 0.1)
        self.base_style['linewidth'] = linewidth
        self.base_style.setdefault('linestyle', 'solid')
        self.base_style.setdefault('fill', False)

        self.selected_style = self.base_style.copy()
        self.selected_style.update({
            'color': 'yellow',
            'linewidth': linewidth,
        })
        self.selected_style['linestyle'] = self.base_style['linestyle']

        if initial:
            self.style = self.base_style.copy()
        else:
            self.style = self.selected_style.copy() if self.selected else self.base_style.copy()

        if self.mpl_patch is not None:
            self.mpl_patch.set(**self.style)
        self._update_label_position()

    def update_style_attributes(self, *, color=None, linewidth=None, linestyle=None):
        changed = False
        if color and color != self.base_style.get('color'):
            self.base_style['color'] = color
            changed = True
        if linewidth is not None:
            linewidth = max(float(linewidth), 0.1)
            if not math.isclose(linewidth, float(self.base_style.get('linewidth', 1.0))):
                self.base_style['linewidth'] = linewidth
                changed = True
        if linestyle and linestyle != self.base_style.get('linestyle'):
            self.base_style['linestyle'] = linestyle
            changed = True
        if changed:
            self._refresh_style_state()

    def get_style_attributes(self):
        return self.base_style.copy()

    def _geometry_bounds(self):
        """Return conservative ``(xmin, xmax, ymin, ymax)`` pixel bounds."""
        return None

    def _bounded_spatial_selection(self, data_shape):
        """Build a region mask only inside its clipped geometry bounds.

        Pixel coordinates refer to pixel centres at integer positions.  The
        one-pixel padding makes the crop conservative at fractional and rotated
        boundaries; ``contains`` remains the sole authority for membership.
        """
        if len(data_shape) != 2:
            raise ValueError("Spatial region selection requires a 2D shape")
        ny, nx = (int(data_shape[0]), int(data_shape[1]))
        bounds = self._geometry_bounds()
        if bounds is None:
            x_start, x_stop, y_start, y_stop = 0, nx, 0, ny
        else:
            xmin, xmax, ymin, ymax = (float(value) for value in bounds)
            if not np.all(np.isfinite([xmin, xmax, ymin, ymax])):
                return slice(0, 0), slice(0, 0), np.zeros((0, 0), dtype=bool)
            if xmin > xmax:
                xmin, xmax = xmax, xmin
            if ymin > ymax:
                ymin, ymax = ymax, ymin
            x_start = max(0, int(math.floor(xmin)) - 1)
            x_stop = min(nx, int(math.ceil(xmax)) + 2)
            y_start = max(0, int(math.floor(ymin)) - 1)
            y_stop = min(ny, int(math.ceil(ymax)) + 2)

        if x_start >= x_stop or y_start >= y_stop:
            return (
                slice(y_start, y_start),
                slice(x_start, x_start),
                np.zeros((0, 0), dtype=bool),
            )

        grid_y, grid_x = np.ogrid[y_start:y_stop, x_start:x_stop]
        mask = np.asarray(self.contains(grid_x, grid_y), dtype=bool)
        expected_shape = (y_stop - y_start, x_stop - x_start)
        if mask.shape != expected_shape:
            mask = np.broadcast_to(mask, expected_shape)
        return slice(y_start, y_stop), slice(x_start, x_stop), mask

    def _region_pixels(self, data):
        """Return selected 2D values without allocating whole-image grids."""
        y_slice, x_slice, mask = self._bounded_spatial_selection(data.shape)
        if mask.size == 0 or not np.any(mask):
            return np.array([])
        plane = np.asanyarray(data[y_slice, x_slice])
        return np.asanyarray(plane[mask]).reshape(-1)


    def get_moments(self, data):
        """
        Calculates the intensity-weighted 2D moments for the region
        on the given 2D data array.
        """
        if data is None or data.ndim != 2: return {}
        y_slice, x_slice, mask = self._bounded_spatial_selection(data.shape)
        if not np.any(mask): return {}

        local_y, local_x = np.nonzero(mask)
        valid_x = local_x.astype(np.float64, copy=False) + int(x_slice.start)
        valid_y = local_y.astype(np.float64, copy=False) + int(y_slice.start)
        plane = np.asanyarray(data[y_slice, x_slice])
        intensities = np.asanyarray(plane[mask]).reshape(-1)
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

    def _geometry_bounds(self):
        cx, cy = self.center
        radius = abs(float(self.radius))
        return cx - radius, cx + radius, cy - radius, cy + radius

    def get_stats(self, data):
        if self.radius == 0:
            return self._calculate_stats(np.array([]))
        return self._calculate_stats(self._region_pixels(data))

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
            return {'type': 'circle', 'cursor': _cursor_shapes().SizeAllCursor}
        return None

    def get_state(self):
        return {
            'center': tuple(self.center),
            'radius': float(self.radius),
            'label': self.label_text,
            'style': self.get_style_attributes()
        }

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

    def _geometry_bounds(self):
        vertices = self._compute_vertices()
        if not vertices:
            return None
        xs, ys = zip(*vertices)
        return min(xs), max(xs), min(ys), max(ys)

    def get_stats(self, data):
        if self.width == 0 or self.height == 0:
            return self._calculate_stats(np.array([]))
        return self._calculate_stats(self._region_pixels(data))

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
                cursor = _cursor_shapes().SizeFDiagCursor
            else:
                cursor = _cursor_shapes().SizeBDiagCursor
        elif horizontal:
            cursor = _cursor_shapes().SizeHorCursor
        elif vertical:
            cursor = _cursor_shapes().SizeVerCursor
        else:
            cursor = _cursor_shapes().SizeAllCursor

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
            'label': self.label_text,
            'style': self.get_style_attributes()
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

    def _geometry_bounds(self):
        cx, cy = self.center
        half_w = abs(float(self.width)) / 2.0
        half_h = abs(float(self.height)) / 2.0
        x_radius = math.hypot(self._cos * half_w, self._sin * half_h)
        y_radius = math.hypot(self._sin * half_w, self._cos * half_h)
        return (
            cx - x_radius,
            cx + x_radius,
            cy - y_radius,
            cy + y_radius,
        )

    def get_stats(self, data):
        if self.width == 0 or self.height == 0:
            return self._calculate_stats(np.array([]))
        return self._calculate_stats(self._region_pixels(data))

    def _outline_display_points(self, axes, half_w, half_h, phis):
        """Return display-space coords of outline points at parametric *phis*."""
        x_local = half_w * np.cos(phis)
        y_local = half_h * np.sin(phis)
        gx, gy = self._to_global(x_local, y_local)
        return axes.transData.transform(np.column_stack([gx, gy]))

    def _nearest_outline_phi(self, axes, ex, ey, half_w, half_h):
        """Parametric angle and display distance of the outline point nearest
        to (*ex*, *ey*). Two-stage sampling keeps the hit band gap-free even on
        large ellipses without an iterative closest-point solve."""
        coarse = np.linspace(0.0, 2.0 * math.pi, 121)[:-1]
        disp = self._outline_display_points(axes, half_w, half_h, coarse)
        d = np.hypot(disp[:, 0] - ex, disp[:, 1] - ey)
        i = int(np.argmin(d))
        step = coarse[1] - coarse[0]
        fine = np.linspace(coarse[i] - step, coarse[i] + step, 41)
        fdisp = self._outline_display_points(axes, half_w, half_h, fine)
        fd = np.hypot(fdisp[:, 0] - ex, fdisp[:, 1] - ey)
        j = int(np.argmin(fd))
        return float(fine[j] % (2.0 * math.pi)), float(fd[j])

    def _outline_cursor(self, axes, half_w, half_h, phi):
        """Pick a resize cursor from the outline point's screen-space direction
        so the feedback stays correct for rotated ellipses."""
        c_disp = axes.transData.transform(self.center)
        p_disp = self._outline_display_points(axes, half_w, half_h, np.array([phi]))[0]
        ang = math.degrees(math.atan2(p_disp[1] - c_disp[1], p_disp[0] - c_disp[0])) % 180.0
        if ang < 22.5 or ang >= 157.5:
            return _cursor_shapes().SizeHorCursor
        if ang < 67.5:
            return _cursor_shapes().SizeBDiagCursor
        if ang < 112.5:
            return _cursor_shapes().SizeVerCursor
        return _cursor_shapes().SizeFDiagCursor

    def get_resize_handle(self, event, tolerance):
        if event.x is None or event.y is None:
            return None
        axes = self.mpl_patch.axes
        if axes is None:
            return None
        if self.width <= 0 or self.height <= 0:
            return None

        half_w = self.width / 2.0
        half_h = self.height / 2.0

        # Keep an interior move zone: clicking well inside (norm < 0.5) is not a
        # resize target, so the body can still be grabbed to move the ellipse.
        if event.xdata is not None and event.ydata is not None:
            xl, yl = self._to_local(event.xdata, event.ydata)
            if (xl / half_w) ** 2 + (yl / half_h) ** 2 < 0.5:
                return None

        # The whole outline is grabbable for resize (not just the 4 axis tips),
        # so a tilted ellipse can be resized without hunting for its axis ends.
        phi, distance = self._nearest_outline_phi(axes, event.x, event.y, half_w, half_h)
        if distance > tolerance:
            return None

        cursor = self._outline_cursor(axes, half_w, half_h, phi)
        return {'type': 'ellipse', 'phi': phi, 'cursor': cursor}

    def set_angle(self, angle_deg):
        self.angle = angle_deg % 360.0
        self.update_visual()

    def get_state(self):
        return {
            'center': tuple(self.center),
            'width': float(self.width),
            'height': float(self.height),
            'angle': float(self.angle),
            'label': self.label_text,
            'style': self.get_style_attributes()
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

        y_slice, x_slice, xy_mask = self._bounded_spatial_selection(data.shape[1:])

        if not np.any(xy_mask):
            return self._calculate_stats(np.array([]))

        z_start = max(0, int(round(self.z_min)))
        z_end = min(data.shape[0], int(round(self.z_max)) + 1)

        if z_start >= z_end:
            return self._calculate_stats(np.array([]))

        total_pixel_count = int(np.count_nonzero(xy_mask)) * (z_end - z_start)
        valid_pixel_count = 0
        total_sum = 0.0
        running_mean = 0.0
        running_m2 = 0.0
        data_min = np.inf
        data_max = -np.inf

        # Read one channel at a time. Advanced indexing over the complete
        # z-range would materialize the selected cube and flatten it again.
        for channel_index in range(z_start, z_end):
            plane = np.asanyarray(data[channel_index])[y_slice, x_slice]
            values = np.asanyarray(plane[xy_mask]).reshape(-1)
            valid_values = values[~np.isnan(values)]
            batch_count = int(valid_values.size)
            if batch_count == 0:
                continue

            batch_mean = float(np.mean(valid_values, dtype=np.float64))
            centered = valid_values - batch_mean
            batch_m2 = float(np.sum(centered * centered, dtype=np.float64))
            batch_sum = float(np.sum(valid_values, dtype=np.float64))

            combined_count = valid_pixel_count + batch_count
            delta = batch_mean - running_mean
            running_mean += delta * batch_count / combined_count
            running_m2 += (
                batch_m2
                + delta * delta
                * valid_pixel_count
                * batch_count
                / combined_count
            )
            valid_pixel_count = combined_count
            total_sum += batch_sum
            data_min = min(data_min, float(np.min(valid_values)))
            data_max = max(data_max, float(np.max(valid_values)))

        if valid_pixel_count == 0:
            return {
                'total_pixel_count': total_pixel_count,
                'valid_pixel_count': 0,
                'mean': np.nan,
                'sum': np.nan,
                'std': np.nan,
                'min': np.nan,
                'max': np.nan,
            }

        return {
            'total_pixel_count': total_pixel_count,
            'valid_pixel_count': valid_pixel_count,
            'mean': running_mean,
            'sum': total_sum,
            'std': np.sqrt(running_m2 / valid_pixel_count),
            'min': data_min,
            'max': data_max,
        }

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

        y_slice, x_slice, xy_mask_2d = self._bounded_spatial_selection(
            data.shape[1:]
        )
        if not np.any(xy_mask_2d): return {}

        selected_y, selected_x = np.nonzero(xy_mask_2d)
        selected_x = (
            selected_x.astype(np.float64, copy=False) + int(x_slice.start)
        )
        selected_y = (
            selected_y.astype(np.float64, copy=False) + int(y_slice.start)
        )
        total_intensity = 0.0
        weighted_x = 0.0
        weighted_y = 0.0
        weighted_z = 0.0
        valid_count = 0

        for channel_index in range(z_start, z_end):
            plane = np.asanyarray(data[channel_index])[y_slice, x_slice]
            intensities = np.asanyarray(plane[xy_mask_2d]).reshape(-1)
            valid = ~np.isnan(intensities)
            if not np.any(valid):
                continue
            values = intensities[valid]
            x_values = selected_x[valid]
            y_values = selected_y[valid]
            channel_sum = float(np.sum(values, dtype=np.float64))
            total_intensity += channel_sum
            weighted_x += float(np.sum(values * x_values, dtype=np.float64))
            weighted_y += float(np.sum(values * y_values, dtype=np.float64))
            weighted_z += channel_sum * channel_index
            valid_count += int(values.size)

        if valid_count == 0: return {}
        if total_intensity == 0: return {}

        mean_x = weighted_x / total_intensity
        mean_y = weighted_y / total_intensity
        mean_z = weighted_z / total_intensity
        x_offset_sq = (selected_x - mean_x) ** 2
        y_offset_sq = (selected_y - mean_y) ** 2
        weighted_var_x = 0.0
        weighted_var_y = 0.0
        weighted_var_z = 0.0

        for channel_index in range(z_start, z_end):
            plane = np.asanyarray(data[channel_index])[y_slice, x_slice]
            intensities = np.asanyarray(plane[xy_mask_2d]).reshape(-1)
            valid = ~np.isnan(intensities)
            if not np.any(valid):
                continue
            values = intensities[valid]
            channel_sum = float(np.sum(values, dtype=np.float64))
            weighted_var_x += float(
                np.sum(values * x_offset_sq[valid], dtype=np.float64)
            )
            weighted_var_y += float(
                np.sum(values * y_offset_sq[valid], dtype=np.float64)
            )
            weighted_var_z += channel_sum * (channel_index - mean_z) ** 2

        var_x = weighted_var_x / total_intensity
        var_y = weighted_var_y / total_intensity
        var_z = weighted_var_z / total_intensity

        return {
            'mean_x_pix': mean_x, 'mean_y_pix': mean_y, 'mean_z_pix': mean_z,
            'sigma_x_pix': np.sqrt(var_x), 'sigma_y_pix': np.sqrt(var_y), 'sigma_z_pix': np.sqrt(var_z),
        }
