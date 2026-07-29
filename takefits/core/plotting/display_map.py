from matplotlib.axes import Axes
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import astropy.units as u
from typing import TYPE_CHECKING

from takefits.logic.data_tools import (
    DEFAULT_LARGE_DATA_DISPLAY_MAX_DIM,
    MEMMAP_THRESHOLD_BYTES,
    downsample_2d_for_display,
    estimate_array_nbytes,
    fast_nanminmax,
    is_lazy_scaled,
    sanitize_slice,
)
from takefits.core.fonts import resolve_mpl_font_family
from takefits.core.wcs_frames import (
    celestial_axis_indices,
    frame_is_available,
    native_celestial_frame,
    normalize_display_frame,
)

if TYPE_CHECKING:
    from takefits.core.viewer_state import ViewerState

class TransparentOverlayAxes(Axes):
    def contains(self, mouseevent):
        return False, {}


def _coord_wrap_quantity(value, default):
    try:
        return float(value) * u.deg
    except Exception:
        return float(default) * u.deg


def _safe_grid_float(value, default, lower, upper):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(default)
    if not math.isfinite(parsed):
        parsed = float(default)
    return min(float(upper), max(float(lower), parsed))


def _optional_nonnegative_float(value):
    if value is None or str(value).strip().lower() in {'', 'auto', 'none'}:
        return None
    try:
        parsed = abs(float(value))
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _grid_bool(value, default):
    if isinstance(value, bool):
        return value
    token = str(value or '').strip().lower()
    if token in {'1', 'true', 'yes', 'on'}:
        return True
    if token in {'0', 'false', 'no', 'off'}:
        return False
    return bool(default)


def _normalized_grid_linestyle(value, default):
    token = str(value or '').strip().lower()
    return {
        '-': 'solid',
        '--': 'dashed',
        ':': 'dotted',
        '-.': 'dashdot',
        'solid': 'solid',
        'dashed': 'dashed',
        'dotted': 'dotted',
        'dash-dot': 'dashdot',
        'dashdot': 'dashdot',
        'dash dot': 'dashdot',
    }.get(token, default)


def _relative_luminance(rgb):
    def _linear(channel):
        channel = float(channel)
        if channel <= 0.04045:
            return channel / 12.92
        return ((channel + 0.055) / 1.055) ** 2.4

    red, green, blue = (_linear(channel) for channel in rgb)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _contrast_ratio(first, second):
    lighter = max(_relative_luminance(first), _relative_luminance(second))
    darker = min(_relative_luminance(first), _relative_luminance(second))
    return (lighter + 0.05) / (darker + 0.05)


def _contrast_adjusted_color(color, background, minimum_ratio=3.0):
    """Keep a requested hue when possible, darkening/lightening for labels."""
    try:
        foreground_rgb = mpl.colors.to_rgb(color)
    except (TypeError, ValueError):
        foreground_rgb = mpl.colors.to_rgb('#008000')
    try:
        background_rgb = mpl.colors.to_rgb(background)
    except (TypeError, ValueError):
        background_rgb = mpl.colors.to_rgb('white')
    if _contrast_ratio(foreground_rgb, background_rgb) >= minimum_ratio:
        return mpl.colors.to_hex(foreground_rgb)

    black = (0.0, 0.0, 0.0)
    white = (1.0, 1.0, 1.0)
    target = (
        black
        if _contrast_ratio(black, background_rgb)
        >= _contrast_ratio(white, background_rgb)
        else white
    )
    for step in range(1, 21):
        amount = step / 20.0
        candidate = tuple(
            (1.0 - amount) * source + amount * destination
            for source, destination in zip(foreground_rgb, target)
        )
        if _contrast_ratio(candidate, background_rgb) >= minimum_ratio:
            return mpl.colors.to_hex(candidate)
    return mpl.colors.to_hex(target)


def _normalized_header_unit_text(value):
    text = str(value or "").strip()
    if not text:
        return ""
    return text.replace(" ", "")


def _build_third_axis_label(header):
    ctype3 = str(header.get("CTYPE3", "") or "").strip().upper()
    specsys = str(header.get("SPECSYS", "") or "").strip().upper()
    cunit3 = _normalized_header_unit_text(header.get("CUNIT3", ""))

    if "FREQ" in ctype3:
        base = "Frequency"
    elif any(token in ctype3 for token in ("VRAD", "VELO", "VOPT")):
        base = "Velocity"
    elif ctype3:
        base = ctype3.split("-")[0].replace("_", " ").title()
    else:
        base = "Velocity"

    if base == "Velocity" and ("LSR" in ctype3 or "LSR" in specsys):
        base = "LSR Velocity"

    if cunit3:
        return f"{base}  [{cunit3}]"
    return base


def _is_pv_fits_header(header) -> bool:
    return bool(str(header.get("PVXAXIS", "") or "").strip() or str(header.get("PVPATH", "") or "").strip())


def _pv_axis_ctype_head(header, axis_index: int) -> str:
    try:
        axis_no = int(axis_index) + 1
    except Exception:
        return ""
    return str(header.get(f"CTYPE{axis_no}", "") or "").strip().upper().split("-")[0]


def _pv_axis_unit(header, axis_index: int) -> str:
    try:
        axis_no = int(axis_index) + 1
    except Exception:
        return ""
    return _normalized_header_unit_text(header.get(f"CUNIT{axis_no}", ""))


def _is_pv_scalar_axis(header, axis_index: int) -> bool:
    if not _is_pv_fits_header(header):
        return False
    return _pv_axis_ctype_head(header, axis_index) in {"PHI", "OFFSET"}


def _pv_scalar_axis_label(header, axis_index: int) -> str:
    ctype = _pv_axis_ctype_head(header, axis_index)
    if ctype == "PHI":
        base = "Phi"
    elif ctype == "OFFSET":
        base = "Position"
    else:
        return ""
    unit = _pv_axis_unit(header, axis_index)
    return f"{base} [{unit}]" if unit else base

class DisplayMap:
    def __init__(
        self,
        data,
        header,
        wcs,
        config,
        viewer_state: 'ViewerState' = None,
        *,
        large_data_mode: bool = False,
        defer_colorbar: bool = False,
    ):
        self.third_axis_label = _build_third_axis_label(header)

        self.coords_dict = {'glon': 'Galactic Longitude',
               'glat': 'Galactic Latitude',
               'ra': 'Right Ascension',
               'dec': 'Declination',
               'vopt': f'{self.third_axis_label}',
               'vrad': f'{self.third_axis_label}',
               'freq': f'{self.third_axis_label}'
               }
        for idx in range(getattr(wcs, "naxis", 0)):
            label = _pv_scalar_axis_label(header, idx)
            if label:
                self.coords_dict[_pv_axis_ctype_head(header, idx).lower()] = label
               
        self.config = config
        self.colorscale = config.get('colorscale', 'Rainbow')  # default color pattern
        self.coord_wrap = config.get('coord_wrap', 180)
        self.axislabel_fontsize = config.get('axislabel_fontsize', 14)
        self.axislabel_fontfamily = resolve_mpl_font_family(config.get('axislabel_fontfamily', 'DejaVu Sans'))
        self.axislabel_color = config.get('axislabel_color', 'black')
        self.default_ticks_position = config.get('default_ticks_position', 'btlr')
        self.xticklabel_position = config.get('xticklabel_position', 'b')
        self.yticklabel_position = config.get('yticklabel_position', 'l')
        self.tick_direction = config.get('tick_direction', 'out')
        self.tick_length = config.get('tick_length', 4)  # default: 4
        self.mtick_length = config.get('mtick_length', 2)  # default: 2
        self.tick_width = config.get('tick_width', 1)
        self.tick_labelsize = config.get('tick_labelsize', 10)
        self.tick_color = config.get('tick_color', 'black')
        self.tick_labelcolor = config.get('tick_labelcolor', 'black')
        self.tick_pad_x = config.get('tick_pad_x', 5)
        self.tick_pad_y = config.get('tick_pad_y', 5)

        self.x_mtick_freq = config.get('x_mtick_freq', 5)
        self.y_mtick_freq = config.get('y_mtick_freq', 5)
        self.z_mtick_freq = config.get('z_mtick_freq', 5)
        self._mtick_freq = {
            'x': self.x_mtick_freq,
            'y': self.y_mtick_freq,
            'z': self.z_mtick_freq,
        }
        # Major tick placement per axis; None leaves astropy WCSAxes in charge.
        self._tick_spacing = {
            key: config.get(f'{key}_tick_spacing') for key in ('x', 'y', 'z')
        }
        self._tick_number = {
            key: config.get(f'{key}_tick_number') for key in ('x', 'y', 'z')
        }
        
        self.tick_xlabelrotation = config.get('tick_xlabelrotation', 0)
        self.tick_ylabelrotation = config.get('tick_ylabelrotation', 0)
        self.fig_background_color = config.get('fig_background_color', '#ececec')
        self.ax_background_color = config.get('ax_background_color', 'white')
        self.bad_color = config.get('bad_color', 'black')
        self.ax_pos_l = config.get('ax_pos_l', 0.15)
        self.ax_pos_r = config.get('ax_pos_r', 0.85)
        self.ax_pos_t = config.get('ax_pos_t', 0.9)
        self.ax_pos_b = config.get('ax_pos_b', 0.12)
        
        self.cbar_pos_x = config.get('cbar_pos_x', 0.9)
        self.cbar_pos_y = config.get('cbar_pos_y', 0.11)
        self.cbar_width = config.get('cbar_width', 0.04)
        self.cbar_height = config.get('cbar_height', 0.77)
        
        self.decimal = config.get('decimal', True)
        
        self.tick_font = resolve_mpl_font_family(config.get('tick_font', 'DejaVu Sans'))
        self.tick_font_weight = config.get('tick_font_weight', 'normal')

        # Coordinate grid overlay (TF-404 / TF-407)
        self.grid_visible = bool(config.get('grid_visible', False))
        self.grid_frame = normalize_display_frame(config.get('grid_frame', 'native'))
        self.grid_keep_native = bool(config.get('grid_keep_native', True))
        self.refresh_grid_style(config)
        self.grid_effective_frame = 'native'
        self.grid_overlay_active = False
        self.active_grid_overlay = None
        self._grid_overlay_cache = {}
        self._grid_overlay_layout_active = False
        self._grid_overlay_base_ax_bounds = None
        self.last_grid_error = None
        
        self.colorbar_orientation = config.get('colorbar_orientation', 'vertical')
        self.colorbar_tick_color = config.get('colorbar_tick_color', 'black')
        self.colorbar_tick_length = config.get('colorbar_tick_length', 2)
        self.colorbar_mtick_length = config.get('colorbar_mtick_length', 1)
        self.colorbar_tick_width = config.get('colorbar_tick_width', 1)
        self.colorbar_tick_direction = config.get('colorbar_tick_direction', 'out')
        self.colorbar_mtick_freq = config.get('colorbar_mtick_freq', 2)
        self.colorbar_tick_labelleft = config.get('colorbar_tick_labelleft', False)
        self.colorbar_tick_labeltop = config.get('colorbar_tick_labeltop', False)
        self.colorbar_tick_left = config.get('colorbar_tick_left', True)
        self.colorbar_tick_right = config.get('colorbar_tick_right', True)
        self.colorbar_tick_top = config.get('colorbar_tick_top', False)
        self.colorbar_tick_bottom = config.get('colorbar_tick_bottom', True)
        self.colorbar_label = config.get('colorbar_label', None)
        self.colorbar_label_fontsize = config.get('colorbar_label_fontsize', 12)
        self.colorbar_label_color = config.get('colorbar_label_color', 'black')
        self.colorbar_label_fontfamily = resolve_mpl_font_family(
            config.get('colorbar_label_fontfamily', 'DejaVu Sans')
        )
        self.colorbar_tick_labelcolor = config.get('colorbar_tick_labelcolor', 'black')
        # None keeps the Matplotlib default so existing configs are unchanged.
        self.colorbar_tick_labelsize = config.get('colorbar_tick_labelsize', None)
        # Defaults to the image tick font, which is what the previous global
        # rcParams write effectively gave the colorbar.
        self.colorbar_tick_labelfontfamily = resolve_mpl_font_family(
            config.get('colorbar_tick_labelfontfamily')
            or config.get('tick_font', 'DejaVu Sans')
        )
        
        self.data = data
        self.wcs = wcs
        self.header = header
        self.viewer_state = viewer_state
        self.large_data_mode = bool(large_data_mode)
        self.defer_colorbar = bool(defer_colorbar)
        self.large_data_display_max_dim = int(
            config.get('large_data_display_max_dim', DEFAULT_LARGE_DATA_DISPLAY_MAX_DIM)
        )

        self.default_cmin, self.default_cmax = self._initial_limits()

        if np.isnan(self.default_cmin) or np.isnan(self.default_cmax):
            self.default_cmin, self.default_cmax = 0.0, 0.0

        self.colorbar = None
        self.cax = None

    def refresh_grid_style(self, config=None):
        """Refresh grid preferences without replacing interactive grid state."""
        if isinstance(config, dict):
            self.config = config
        config = self.config if isinstance(self.config, dict) else {}
        self.grid_color = config.get('grid_color', 'white')
        self.grid_overlay_color = config.get(
            'grid_overlay_color',
            '#00ff66',
        )
        self.grid_alpha = _safe_grid_float(
            config.get('grid_alpha', 0.5),
            0.5,
            0.0,
            1.0,
        )
        self.grid_linestyle = _normalized_grid_linestyle(
            config.get('grid_linestyle', 'solid'),
            'solid',
        )
        self.grid_overlay_linestyle = _normalized_grid_linestyle(
            config.get('grid_overlay_linestyle', 'dashed'),
            'dashed',
        )
        self.grid_linewidth = _safe_grid_float(
            config.get('grid_linewidth', 0.5),
            0.5,
            0.0,
            20.0,
        )
        label_color = str(
            config.get('grid_overlay_label_color', 'auto') or 'auto'
        ).strip()
        self.grid_overlay_label_color = (
            label_color.lower()
            if label_color.lower() in {'auto', 'same'}
            else label_color
        )
        self.grid_overlay_show_lines = _grid_bool(
            config.get('grid_overlay_show_lines'),
            True,
        )
        self.grid_overlay_show_ticklabels = _grid_bool(
            config.get('grid_overlay_show_ticklabels'),
            True,
        )
        placement = str(
            config.get('grid_overlay_axislabel_placement', 'inside')
            or 'inside'
        ).strip().lower()
        if placement not in {'outside', 'inside', 'hidden'}:
            placement = 'inside'
        self.grid_overlay_axislabel_placement = placement
        self.grid_overlay_longitude_axislabel_pad = (
            _optional_nonnegative_float(
                config.get('grid_overlay_longitude_axislabel_pad')
            )
        )
        self.grid_overlay_latitude_axislabel_pad = (
            _optional_nonnegative_float(
                config.get('grid_overlay_latitude_axislabel_pad')
            )
        )
        configured_right_margin = _safe_grid_float(
            config.get('grid_overlay_right_margin_px', 96.0),
            96.0,
            0.0,
            1000.0,
        )
        configured_top_margin = _safe_grid_float(
            config.get('grid_overlay_top_margin_px', 64.0),
            64.0,
            0.0,
            1000.0,
        )
        # Exterior WCSAxes decorations can extend a fraction of a pixel past
        # their nominal text extent. Outside titles need more room than the
        # default inside titles; both retain a small anti-aliasing clearance
        # before colorbars and figure titles.
        has_exterior_ticks = self.grid_overlay_show_ticklabels
        has_exterior_titles = (
            self.grid_overlay_axislabel_placement == 'outside'
        )
        if has_exterior_ticks or has_exterior_titles:
            minimum_right_margin = (
                120.0 if has_exterior_titles else 96.0
            )
            minimum_right_margin = max(
                minimum_right_margin,
                configured_right_margin,
            )
        else:
            minimum_right_margin = 0.0
        self._grid_overlay_min_right_margin_px = (
            minimum_right_margin + 2.0
            if minimum_right_margin > 0.0
            else 0.0
        )
        self.grid_overlay_right_margin_px = (
            self._grid_overlay_min_right_margin_px
        )
        if has_exterior_ticks or has_exterior_titles:
            minimum_top_margin = (
                120.0 if has_exterior_titles else 64.0
            )
            minimum_top_margin = max(
                minimum_top_margin,
                configured_top_margin,
            )
        else:
            minimum_top_margin = 0.0
        self._grid_overlay_min_top_margin_px = (
            minimum_top_margin + 2.0
            if minimum_top_margin > 0.0
            else 0.0
        )
        self.grid_overlay_top_margin_px = (
            self._grid_overlay_min_top_margin_px
        )
        self.axislabel_fontsize = config.get('axislabel_fontsize', 14)
        self.axislabel_fontfamily = resolve_mpl_font_family(
            config.get('axislabel_fontfamily', 'DejaVu Sans')
        )
        self.tick_labelsize = config.get('tick_labelsize', 10)
        self.tick_font = resolve_mpl_font_family(
            config.get('tick_font', 'DejaVu Sans')
        )
        self.tick_font_weight = config.get('tick_font_weight', 'normal')
        self.tick_length = config.get('tick_length', 4)
        self.tick_width = config.get('tick_width', 1)
        self.tick_direction = config.get('tick_direction', 'out')
        self.tick_pad_x = config.get('tick_pad_x', 5)
        self.tick_pad_y = config.get('tick_pad_y', 5)
        self.tick_xlabelrotation = config.get('tick_xlabelrotation', 0)
        self.tick_ylabelrotation = config.get('tick_ylabelrotation', 0)
        self.fig_background_color = config.get(
            'fig_background_color',
            '#ececec',
        )
        self.ax_background_color = config.get('ax_background_color', 'white')
        self.decimal = bool(config.get('decimal', True))
        self.coord_wrap = config.get('coord_wrap', 180)

    def _resolved_overlay_label_color(self, *, axis_title=False) -> str:
        setting = str(self.grid_overlay_label_color or 'auto').strip()
        if setting.lower() == 'same':
            return self.grid_overlay_color
        if setting.lower() != 'auto':
            try:
                mpl.colors.to_rgba(setting)
            except (TypeError, ValueError):
                return self.grid_overlay_color
            return setting
        background = self.fig_background_color
        if (
            axis_title
            and self.grid_overlay_axislabel_placement == 'inside'
        ):
            background = self.ax_background_color
        return _contrast_adjusted_color(
            self.grid_overlay_color,
            background,
        )

    def restore_grid_overlay_layout(self) -> bool:
        """Detach a frame overlay and restore its reserved axes bounds."""
        self._hide_grid_overlays()
        return bool(self.update_grid_overlay_layout())

    def _initial_limits(self):
        """Derive display limits without forcing a full scan on large cubes."""
        datamin = self.header.get('DATAMIN')
        datamax = self.header.get('DATAMAX')
        if datamin is not None and datamax is not None:
            try:
                cmin = float(datamin)
                cmax = float(datamax)
            except (TypeError, ValueError):
                cmin = cmax = np.nan
            else:
                if not np.isfinite(cmin) or not np.isfinite(cmax):
                    cmin = cmax = np.nan
                else:
                    if cmin > cmax:
                        cmin, cmax = cmax, cmin
                    if cmin != cmax:
                        return cmin, cmax

        approx_bytes = estimate_array_nbytes(self.data)
        if approx_bytes and approx_bytes >= MEMMAP_THRESHOLD_BYTES:
            sample = self._subsample_large_array(self.data)
            cmin, cmax = fast_nanminmax(sample)
        else:
            cmin, cmax = fast_nanminmax(self.data)

        if np.isnan(cmin) or np.isnan(cmax):
            # Fallback: try a representative major-plane slice, then give up to (0, 0)
            try:
                slicer = []
                for axis in range(max(self.data.ndim, 2)):
                    if axis < self.data.ndim - 2:
                        slicer.append(0)
                    else:
                        slicer.append(slice(None))
                first_slice = self.data[tuple(slicer[: self.data.ndim])]
            except Exception:
                first_slice = self.data
            cmin, cmax = fast_nanminmax(first_slice)
        if np.isnan(cmin) or np.isnan(cmax):
            cmin = cmax = 0.0
        return cmin, cmax

    def _subsample_large_array(self, array, points_per_axis=24):
        """
        Edge-aware subsampling that keeps the sample size modest while covering the volume.
        """
        if array.ndim == 0:
            return array

        axis_indices = [self._axis_indices(size, points_per_axis) for size in array.shape]
        try:
            mesh = np.ix_(*axis_indices)
            return array[mesh]
        except Exception:
            # As a fallback, use flattened step-sampling.
            flat = array.reshape(-1)
            step = max(1, flat.size // (points_per_axis ** max(array.ndim, 1)))
            return flat[::step]

    @staticmethod
    def _axis_indices(size, max_points):
        if size <= max_points:
            return np.arange(size, dtype=int)
        # Ensure we always include endpoints and central values.
        return np.unique(
            np.round(
                np.linspace(0, size - 1, max_points, dtype=float)
            ).astype(int)
        )
    
    def display(self, fig, plane):
        # A CoordinatesMap returned by get_coords_overlay() is permanently
        # bound to the WCSAxes that created it. If this DisplayMap is rebuilt,
        # discard the old-axis cache rather than reattaching stale coordinates
        # to the new axes.
        if getattr(self, 'ax', None) is not None:
            self._hide_grid_overlays()
            self.update_grid_overlay_layout()
        self._grid_overlay_cache.clear()
        self.grid_effective_frame = 'native'
        self.grid_overlay_active = False
        self.active_grid_overlay = None
        self._grid_overlay_layout_active = False
        self._grid_overlay_base_ax_bounds = None

        if self.wcs is not None:
            ndim = self.wcs.naxis
        else:
            ndim = self.data.ndim

        if ndim == 2:
            self.slices = ('x', 'y')
            self.imdata = self.data
            # Check if the header indicates a position-velocity diagram.
            pv = False
            if hasattr(self, 'header') and self.header is not None:
                for i in [1, 2]:
                    ctype = self.header.get(f'CTYPE{i}', '').upper()
                    if 'VRAD' in ctype or 'VEL' in ctype or 'VOPT' in ctype:
                        pv = True
                        break
            if pv:
                aspect = 'auto'
            else:
                aspect = 'equal'

        elif ndim == 3:
            if plane=='xy':
                self.slices = ('x', 'y', 0)
                if self.data.ndim == 3: self.imdata = self.data[0, :, :]
                elif self.data.ndim == 2: self.imdata = self.data
                aspect = 'equal'
            elif plane=='xz':
                self.slices = ('x', 0, 'y')
                self.imdata = self.data[:, 0, :]
                aspect = 'auto'
            elif plane=='zy':
                self.slices = (0, 'y', 'x')
                self.imdata = self.data[:, :, 0].T
                aspect = 'auto'
        elif ndim ==4:
            if plane=='xy':
                self.slices = ('x', 'y', 0, 0)
                self.imdata = self.data[0, 0, :, :]
                aspect = 'equal'
            elif plane=='xz':
                self.slices = ('x', 0, 'y', 0)
                self.imdata = self.data[0, :, 0, :]
                aspect = 'auto'
            elif plane=='zy':
                self.slices = (0, 'y', 'x', 0)
                self.imdata = self.data[0, :, :, 0].T
                aspect = 'auto'

        self.fig = fig
        self.plane = plane

        # Tick label typography is applied per coordinate below and on the
        # colorbar axes, not through plt.rcParams. A global rcParams write
        # leaked into every artist created afterwards (markers, titles), so a
        # tick font choice silently overrode their own font settings.

        from astropy.visualization.wcsaxes import WCSAxes
        # Calculate axes position as [left, bottom, width, height] in figure coordinates.
        pos = [self.ax_pos_l, self.ax_pos_b, self.ax_pos_r - self.ax_pos_l, self.ax_pos_t - self.ax_pos_b]
        # Create a WCSAxes with slices set.
        self.ax = WCSAxes(self.fig, pos, wcs=self.wcs, slices=self.slices)
        # Add the newly created axes to the figure.
        self.fig.add_axes(self.ax)

        fig.subplots_adjust(left=self.ax_pos_l, right = self.ax_pos_r, bottom = self.ax_pos_b, top= self.ax_pos_t) 
        #self.cax = self.fig.add_axes([self.cbar_pos_x, self.cbar_pos_y, self.cbar_width, self.cbar_height])  # [left, bottom, width, height] in figure coordinates
        
        #from mpl_toolkits.axes_grid1 import make_axes_locatable
        #divider = make_axes_locatable(self.ax)
        # Append axes to the right of ax
        #self.cax = divider.append_axes("right", size=0.1, pad=0.1)
        

        
        cmap = plt.get_cmap(self.colorscale)
        cmap.set_bad(self.bad_color, 1.)
        self.ax.patch.set_zorder(0)
        self.ax.set_axisbelow(False)

        source_imdata = self.imdata
        source_height, source_width = source_imdata.shape[-2], source_imdata.shape[-1]
        display_imdata = source_imdata
        image_kwargs = {}
        if self.large_data_mode and getattr(source_imdata, "ndim", 0) == 2:
            display_imdata = downsample_2d_for_display(
                source_imdata,
                max_dimension=self.large_data_display_max_dim,
            )
            image_kwargs["extent"] = (-0.5, source_width - 0.5, -0.5, source_height - 0.5)
            image_kwargs["interpolation"] = "nearest"

        # Ensure display_imdata is a plain numpy array (LazyScaledArray cannot
        # be passed directly to matplotlib).
        if is_lazy_scaled(display_imdata):
            display_imdata = np.asarray(display_imdata)

        self.im = self.ax.imshow(
            display_imdata,
            cmap=cmap,
            aspect=aspect,
            origin='lower',
            zorder=-1,
            **image_kwargs,
        )
        # Keep initial view stable: lock to full pixel extent and disable autoscale drift.
        try:
            ny, nx = source_height, source_width
            x0, x1 = self.ax.get_xlim()
            y0, y1 = self.ax.get_ylim()
            full_xlim = (-0.5, nx - 0.5) if x1 >= x0 else (nx - 0.5, -0.5)
            full_ylim = (-0.5, ny - 0.5) if y1 >= y0 else (ny - 0.5, -0.5)
            self.ax.set_xlim(*full_xlim)
            self.ax.set_ylim(*full_ylim)
            self.ax.set_autoscale_on(False)
        except Exception:
            pass
        self.im.set_clim(self.default_cmin, self.default_cmax)
        

        # Create an overlay axes
        self.overlay_ax = self.fig.add_axes(self.ax.get_position(), sharex=self.ax, sharey=self.ax, zorder=100, frameon=False)
        self.overlay_ax.__class__ = TransparentOverlayAxes
        self.overlay_ax.patch.set_alpha(0)
        self.overlay_ax.set_xticks([])
        self.overlay_ax.set_yticks([])
        self.overlay_ax.set_autoscale_on(False)
        self.overlay_ax.set_navigate(False)
        self.overlay_ax.set_picker(False)


        if not self.defer_colorbar:
            self.cax = self.fig.add_axes([self.cbar_pos_x, self.cbar_pos_y, self.cbar_width, self.cbar_height])
            self.cax.set_gid('colorbar')
            self.cax.set_zorder(300)
            self.colorbar = self.fig.colorbar(self.im, cax = self.cax, orientation = self.colorbar_orientation )
            self.colorbar.ax.set_zorder(300)
            self.cax.tick_params(axis='y', which='both', left=self.colorbar_tick_left, right=self.colorbar_tick_right, labelleft=self.colorbar_tick_labelleft, labelright=(not self.colorbar_tick_labelleft),
                                width=self.colorbar_tick_width, length=self.colorbar_tick_length, color=self.colorbar_tick_color, direction=self.colorbar_tick_direction, labelcolor=self.colorbar_tick_labelcolor)
            self.cax.tick_params(axis='x', which='both', top=self.colorbar_tick_top, bottom=self.colorbar_tick_bottom, labeltop= self.colorbar_tick_labeltop, labelbottom = (not self.colorbar_tick_labeltop),
                                width=self.colorbar_tick_width, length=self.colorbar_tick_length, color=self.colorbar_tick_color, direction=self.colorbar_tick_direction, labelcolor=self.colorbar_tick_labelcolor)
            # The colorbar used to inherit the tick font from the global
            # rcParams write; apply it explicitly now that the write is gone.
            colorbar_tick_params = {
                'labelfontfamily': self.colorbar_tick_labelfontfamily,
            }
            if self.colorbar_tick_labelsize is not None:
                colorbar_tick_params['labelsize'] = self.colorbar_tick_labelsize
            self.cax.tick_params(axis='both', which='both', **colorbar_tick_params)
            self.colorbar.outline.set_color(self.colorbar_tick_color)
            self.colorbar.outline.set_linewidth(self.colorbar_tick_width)
            self.colorbar.set_label(
                self.colorbar_label,
                fontsize=self.colorbar_label_fontsize,
                color=self.colorbar_label_color,
                fontfamily=self.colorbar_label_fontfamily,
            )
            # Matplotlib puts the label opposite the ticks, so a top-ticked
            # horizontal bar drops its label between the bar and the image.
            # Keep the label on the same side as the tick labels.
            try:
                if self.colorbar_orientation == 'horizontal':
                    self.colorbar.ax.xaxis.set_label_position(
                        'top' if self.colorbar_tick_labeltop else 'bottom'
                    )
                else:
                    self.colorbar.ax.yaxis.set_label_position(
                        'left' if self.colorbar_tick_labelleft else 'right'
                    )
            except Exception:
                pass
            self.colorbar.ax.minorticks_on()
            self.colorbar.ax.tick_params(which='minor', length=self.colorbar_mtick_length, color=self.colorbar_tick_color)
            self.colorbar.ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(self.colorbar_mtick_freq))
            self.colorbar.ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(self.colorbar_mtick_freq))

        # Update ViewerState if provided
        if self.viewer_state is not None:
            self.viewer_state.update_colorbar(self.colorbar)
            self.viewer_state.update_cax(self.cax)
            self.viewer_state.update_fig(self.fig)

        
        self.fig.set_facecolor(self.fig_background_color)
        self.ax.set_facecolor(self.ax_background_color)

        self.update_axes_format()

        disp_idx = tuple(True if i else False for i in self.slices)
        for idx in np.where(np.logical_not(disp_idx))[0]:
            self.ax.coords[idx].set_ticklabel_visible(False)
            self.ax.coords[idx].set_ticks_visible(False) 
        ax_xy = []
        for idx in np.where(disp_idx)[0]:
            self.ax.coords[idx].set_ticks_position(self.default_ticks_position)
            # Carry the tick typography on the coordinate itself so it survives
            # draw-time label regeneration without a global rcParams write.
            self.ax.coords[idx].set_ticklabel(
                exclude_overlapping=True,
                fontfamily=self.tick_font,
                fontweight=self.tick_font_weight,
            )
            self.ax.coords[idx].set_ticklabel_visible(True)
            self.ax.coords[idx].display_minor_ticks(True)
            ax_xy.append(self.ax.coords[idx])

        if self.plane == 'zy': ax_xy.reverse()
        ax_xy[0].set_ticklabel(rotation = self.tick_xlabelrotation)
        ax_xy[1].set_ticklabel(rotation = self.tick_ylabelrotation)

        ax_xy[0].set_ticklabel_position(self.xticklabel_position)
        ax_xy[0].set_axislabel_position(self.xticklabel_position)
        ax_xy[1].set_ticklabel_position(self.yticklabel_position)
        ax_xy[1].set_axislabel_position(self.yticklabel_position)
        
        if self.plane =='xy':
            axis_keys = ('x', 'y')
        elif self.plane =='xz':
            axis_keys = ('x', 'z')
        elif self.plane =='zy':
            axis_keys = ('z', 'y')
        else:
            axis_keys = ()

        for coord, key in zip(ax_xy, axis_keys):
            coord.set_minor_frequency(self._mtick_freq[key])
            self._apply_major_tick_placement(coord, key)

        # Update ViewerState if provided
        if self.viewer_state is not None:
            self.viewer_state.update_ax_coord(ax_xy)
        
        for coord_name, axis_label in self.coords_dict.items():
            if coord_name in self.ax.coords: 
                self.ax.coords[coord_name].set_axislabel(axis_label, fontsize=self.axislabel_fontsize, fontfamily = self.axislabel_fontfamily, color = self.axislabel_color)

        if self.plane == 'xz':
            ax_xy[1].set_axislabel(
                self.third_axis_label,
                fontsize=self.axislabel_fontsize,
                fontfamily=self.axislabel_fontfamily,
                color=self.axislabel_color,
            )
        elif self.plane == 'zy':
            ax_xy[0].set_axislabel(
                self.third_axis_label,
                fontsize=self.axislabel_fontsize,
                fontfamily=self.axislabel_fontfamily,
                color=self.axislabel_color,
            )
        
        self.ax.tick_params(axis='both', which = 'major', direction=self.tick_direction,length=self.tick_length,color=self.tick_color, width = self.tick_width, labelsize = self.tick_labelsize, labelcolor = self.tick_labelcolor)
        
        self.ax.tick_params(axis='x', pad = self.tick_pad_x)
        self.ax.tick_params(axis='y', pad = self.tick_pad_y)
        self.ax.tick_params(which = 'minor', length=self.mtick_length)
        for spine in self.ax.spines.values():
            spine.set_visible(True)
            spine.set_zorder(5)
            spine.set_linewidth(self.tick_width)
            spine.set_color(self.tick_color)

        # Apply the initial coordinate-grid state (TF-404). Calling this on
        # every (re)build keeps a config-enabled grid visible after the figure
        # is recreated, e.g. on workspace restore.
        self._install_safe_grid_contour_clear()
        self.set_grid(self.grid_visible)

        return self.im, self.ax

    def _install_safe_grid_contour_clear(self):
        """Work around an astropy contour-grid teardown bug.

        ``CoordinateHelper._clear_grid_contour`` removes the contour artist but
        never resets ``self._grid``, so toggling a contour-style grid off and on
        again makes the next draw call ``self._grid.remove()`` a second time and
        raise ``ValueError: list.remove(x): x not in list``. We replace the bound
        method per-coordinate with one that drops the reference after removal and
        tolerates an already-removed artist.
        """
        ax = getattr(self, 'ax', None)
        if ax is None:
            return
        try:
            coords = list(ax.coords)
        except Exception:
            return
        for coord in coords:
            if getattr(coord, '_takefits_safe_grid_clear', False):
                continue

            def _safe_clear(_coord=coord):
                grid = getattr(_coord, '_grid', None)
                if grid is not None:
                    try:
                        grid.remove()
                    except (ValueError, AttributeError):
                        pass
                    _coord._grid = None

            try:
                coord._clear_grid_contour = _safe_clear
                coord._takefits_safe_grid_clear = True
            except Exception:
                continue

    def _grid_type_for_plane(self) -> str:
        """Pick the WCSAxes grid algorithm that is safe for this plane.

        WCSAxes' straight-line grid (``grid_type='lines'``) inverts the display
        transform. On the spectral planes (XZ/ZY) the displayed axes come from a
        ``SlicedLowLevelWCS`` with a celestial axis sliced out; inverting that
        raises ``IndexError`` inside astropy (worse for coupled projections such
        as GLS/SFL). The contour-based grid only uses the forward
        pixel->world transform, so it is safe there. Conversely ``'contours'``
        is broken for the all-celestial XY plane in the installed astropy, so
        XY (and plain 2D images) keep ``'lines'``.
        """
        plane = getattr(self, 'plane', 'xy')
        return 'contours' if plane in ('xz', 'zy') else 'lines'

    def _displayed_coord_indices(self):
        """Indices of the two on-screen axes (``slices`` entries 'x'/'y')."""
        slices = getattr(self, 'slices', None)
        if not slices:
            return None
        return {idx for idx, s in enumerate(slices) if s in ('x', 'y')}

    def _set_native_grid(self, visible: bool) -> bool:
        """Apply native grid visibility to the two displayed WCS axes."""
        ax = getattr(self, 'ax', None)
        if ax is None:
            return False
        grid_type = self._grid_type_for_plane()
        displayed = self._displayed_coord_indices()
        try:
            for idx, coord in enumerate(ax.coords):
                draw = bool(visible) and (displayed is None or idx in displayed)
                if draw:
                    coord.grid(
                        draw_grid=True,
                        grid_type=grid_type,
                        color=self.grid_color,
                        alpha=self.grid_alpha,
                        linestyle=self.grid_linestyle,
                        linewidth=self.grid_linewidth,
                    )
                else:
                    coord.grid(draw_grid=False)
        except Exception as exc:
            self.last_grid_error = exc
            return False
        return True

    def _overlay_target_frame(self):
        """Return the non-native frame that can be overlaid on this XY view."""
        if getattr(self, 'plane', 'xy') != 'xy' or self.wcs is None:
            return None
        displayed = self._displayed_coord_indices()
        celestial = celestial_axis_indices(self.wcs)
        if displayed is None or celestial is None or set(celestial) != set(displayed):
            return None

        native_frame = native_celestial_frame(self.wcs)
        requested = normalize_display_frame(self.grid_frame)
        target = native_frame if requested == 'native' else requested
        if native_frame is None or target == native_frame:
            return None
        if not frame_is_available(self.wcs, target):
            return None
        return target

    @staticmethod
    def _overlay_axis_label(frame: str, coord_type: str) -> str:
        if coord_type == 'longitude':
            return 'Galactic Longitude' if frame == 'galactic' else 'Right Ascension'
        if coord_type == 'latitude':
            return 'Galactic Latitude' if frame == 'galactic' else 'Declination'
        return ''

    def _overlay_axis_minpad(
        self,
        coord_type: str,
        colorbar_placement: str = None,
    ) -> float:
        placement = self.grid_overlay_axislabel_placement
        if coord_type == 'longitude':
            configured = self.grid_overlay_longitude_axislabel_pad
            automatic = 1.0 if placement == 'outside' else 2.0
        elif coord_type == 'latitude':
            configured = self.grid_overlay_latitude_axislabel_pad
            automatic = 1.0 if placement == 'outside' else 5.0
        else:
            return 0.0
        magnitude = automatic if configured is None else configured
        if placement == 'inside':
            return -abs(float(magnitude))
        if placement == 'outside':
            return abs(float(magnitude))
        return 0.0

    def _overlay_axis_label_for_placement(
        self,
        frame: str,
        coord_type: str,
        colorbar_placement: str = None,
    ) -> str:
        axis_label = self._overlay_axis_label(frame, coord_type)
        if self.grid_overlay_axislabel_placement == 'hidden':
            return ''
        placement = self._effective_colorbar_placement(
            colorbar_placement
        )
        # A vertical inside bar can span the top title's y-range, while a
        # horizontal inside bar can span the right title's x-range. Keep both
        # numeric tick-label sets, but suppress only the conflicting
        # descriptive title. A left exterior bar can likewise cross the long
        # top title in compact windows. Switching placement restores it.
        if self.grid_overlay_axislabel_placement == 'inside':
            if (
                placement in {'left', 'inside-left', 'inside-right'}
                and coord_type == 'longitude'
            ):
                return ''
            if (
                placement in {'inside-top', 'inside-bottom'}
                and coord_type == 'latitude'
            ):
                return ''
            # On the opposite inside sides, compact figures can make the axes
            # narrow/short enough that the remaining perpendicular title also
            # crosses the bar. Numeric coordinate labels remain visible.
            if placement == 'inside-left' and coord_type == 'latitude':
                return ''
            if placement == 'inside-bottom' and coord_type == 'longitude':
                return ''
        return axis_label

    def _effective_colorbar_placement(
        self,
        explicit_placement: str = None,
    ) -> str:
        """Use actual axes geometry when no GUI layout hint is supplied."""
        if explicit_placement is not None:
            return str(explicit_placement or '').strip().lower()
        ax = getattr(self, 'ax', None)
        cax = getattr(self, 'cax', None)
        if ax is not None and cax is not None:
            try:
                ax_left, ax_bottom, ax_width, ax_height = (
                    float(value)
                    for value in ax.get_position().bounds
                )
                cb_left, cb_bottom, cb_width, cb_height = (
                    float(value)
                    for value in cax.get_position().bounds
                )
                ax_right = ax_left + ax_width
                ax_top = ax_bottom + ax_height
                cb_right = cb_left + cb_width
                cb_top = cb_bottom + cb_height
                if cb_right <= ax_left:
                    return 'left'
                if cb_left >= ax_right:
                    return 'right'
                if cb_top <= ax_bottom:
                    return 'bottom'
                if cb_bottom >= ax_top:
                    return 'top'
                if cb_height >= cb_width:
                    cb_center = cb_left + 0.5 * cb_width
                    ax_center = ax_left + 0.5 * ax_width
                    return (
                        'inside-left'
                        if cb_center <= ax_center
                        else 'inside-right'
                    )
                cb_center = cb_bottom + 0.5 * cb_height
                ax_center = ax_bottom + 0.5 * ax_height
                return (
                    'inside-bottom'
                    if cb_center <= ax_center
                    else 'inside-top'
                )
            except Exception:
                pass
        return str(
            self.config.get('colorbar_placement', '')
        ).strip().lower()

    def _grow_grid_overlay_margins_from_rendered_labels(self) -> None:
        """Grow top/right reservations from the labels actually rendered."""
        overlay = self.active_grid_overlay
        ax = getattr(self, 'ax', None)
        fig = getattr(self, 'fig', None)
        if overlay is None or ax is None or fig is None:
            return
        try:
            renderer = fig.canvas.get_renderer()
            ax_bbox = ax.bbox
            coords = list(overlay)
        except Exception:
            return

        right_margin = float(
            getattr(
                self,
                '_grid_overlay_min_right_margin_px',
                self.grid_overlay_right_margin_px,
            )
        )
        top_margin = float(
            getattr(
                self,
                '_grid_overlay_min_top_margin_px',
                self.grid_overlay_top_margin_px,
            )
        )
        for coord in coords:
            coord_type = str(
                getattr(coord, 'coord_type', '') or ''
            ).lower()
            if coord_type not in {'longitude', 'latitude'}:
                continue
            artists = []
            if self.grid_overlay_show_ticklabels:
                artists.append(getattr(coord, '_ticklabels', None))
            try:
                if coord.get_axislabel():
                    artists.append(getattr(coord, '_axislabels', None))
            except Exception:
                pass
            for artist in artists:
                get_extent = getattr(artist, 'get_window_extent', None)
                if not callable(get_extent):
                    continue
                try:
                    bbox = get_extent(renderer)
                    if coord_type == 'longitude':
                        outward = float(bbox.y1) - float(ax_bbox.y1)
                        top_margin = max(top_margin, outward + 2.0)
                    else:
                        outward = float(bbox.x1) - float(ax_bbox.x1)
                        right_margin = max(right_margin, outward + 2.0)
                except Exception:
                    continue

        # Margins only grow during one active-overlay lifetime, avoiding
        # draw/layout oscillation when WCSAxes selects a slightly different
        # set of ticks after the axes moves. Style refresh or overlay teardown
        # resets them to the configured minima.
        self.grid_overlay_right_margin_px = max(
            float(self.grid_overlay_right_margin_px),
            right_margin,
        )
        self.grid_overlay_top_margin_px = max(
            float(self.grid_overlay_top_margin_px),
            top_margin,
        )

    def update_grid_overlay_label_layout(
        self,
        colorbar_placement: str = None,
    ) -> bool:
        overlay = self.active_grid_overlay
        if not self.grid_overlay_active or overlay is None:
            return False
        label_color = self._resolved_overlay_label_color(axis_title=True)
        try:
            for coord in overlay:
                coord_type = str(
                    getattr(coord, 'coord_type', '') or ''
                ).lower()
                if coord_type not in {'longitude', 'latitude'}:
                    continue
                coord.set_axislabel(
                    self._overlay_axis_label_for_placement(
                        self.grid_effective_frame,
                        coord_type,
                        colorbar_placement,
                    ),
                    fontsize=self.axislabel_fontsize,
                    fontfamily=self.axislabel_fontfamily,
                    color=label_color,
                    minpad=self._overlay_axis_minpad(
                        coord_type, colorbar_placement
                    ),
                )
        except Exception as exc:
            self.last_grid_error = exc
            return False
        return True

    def _set_overlay_grid_visible(self, overlay, frame: str, visible: bool) -> bool:
        """Show/hide one cached coordinate overlay, including its top/right labels."""
        try:
            coords = list(overlay)
        except Exception as exc:
            self.last_grid_error = exc
            return False

        label_color = self._resolved_overlay_label_color()
        axis_title_color = self._resolved_overlay_label_color(
            axis_title=True
        )
        try:
            for coord in coords:
                coord_type = str(getattr(coord, 'coord_type', '') or '').lower()
                if visible:
                    set_visible = getattr(coord, 'set_visible', None)
                    if callable(set_visible):
                        set_visible(True)
                    coord.grid(
                        draw_grid=self.grid_overlay_show_lines,
                        grid_type='lines',
                        color=self.grid_overlay_color,
                        alpha=self.grid_alpha,
                        linestyle=self.grid_overlay_linestyle,
                        linewidth=self.grid_linewidth,
                    )
                    if coord_type == 'longitude':
                        position = 't'
                    elif coord_type == 'latitude':
                        position = 'r'
                    else:
                        position = ''
                    if position:
                        coord.set_ticks_position(position)
                        coord.set_ticklabel_position(position)
                        coord.set_axislabel_position(position)
                        coord.set_ticks(
                            size=self.tick_length,
                            width=self.tick_width,
                            color=label_color,
                            direction=self.tick_direction,
                        )
                        is_longitude = coord_type == 'longitude'
                        coord.set_ticklabel(
                            color=label_color,
                            size=self.tick_labelsize,
                            fontfamily=self.tick_font,
                            fontweight=self.tick_font_weight,
                            pad=self.tick_pad_x if is_longitude else self.tick_pad_y,
                            rotation=(
                                self.tick_xlabelrotation
                                if is_longitude
                                else self.tick_ylabelrotation
                            ),
                            exclude_overlapping=True,
                        )
                        coord.set_axislabel(
                            self._overlay_axis_label_for_placement(
                                frame, coord_type
                            ),
                            fontsize=self.axislabel_fontsize,
                            fontfamily=self.axislabel_fontfamily,
                            color=axis_title_color,
                            minpad=self._overlay_axis_minpad(coord_type),
                        )
                        if coord_type == 'longitude':
                            if frame == 'galactic':
                                coord.set_coord_type(
                                    'longitude',
                                    coord_wrap=_coord_wrap_quantity(
                                        self.coord_wrap, 180
                                    ),
                                )
                            else:
                                coord.set_coord_type(
                                    'longitude',
                                    coord_wrap=360 * u.deg,
                                )
                            if frame in {'icrs', 'fk5', 'fk4'}:
                                if self.decimal:
                                    coord.set_format_unit('deg', decimal=True)
                                else:
                                    coord.set_format_unit('hour', decimal=False)
                            else:
                                coord.set_format_unit('deg', decimal=self.decimal)
                        elif coord_type == 'latitude':
                            coord.set_format_unit('deg', decimal=self.decimal)
                        coord.set_ticks_visible(
                            self.grid_overlay_show_ticklabels
                        )
                        coord.set_ticklabel_visible(
                            self.grid_overlay_show_ticklabels
                        )
                    else:
                        coord.set_ticks_visible(False)
                        coord.set_ticklabel_visible(False)
                        coord.set_axislabel('')
                else:
                    coord.grid(draw_grid=False)
                    coord.set_ticks_visible(False)
                    coord.set_ticklabel_visible(False)
                    coord.set_axislabel('')
                    set_visible = getattr(coord, 'set_visible', None)
                    if callable(set_visible):
                        set_visible(False)
        except Exception as exc:
            self.last_grid_error = exc
            return False
        return True

    def _hide_grid_overlays(self):
        for frame, overlay in tuple(self._grid_overlay_cache.items()):
            self._set_overlay_grid_visible(overlay, frame, False)
            self._detach_grid_overlay(overlay)
        self.grid_overlay_active = False
        self.active_grid_overlay = None

    def _detach_grid_overlay(self, overlay):
        """Keep inactive cached overlays out of WCSAxes' per-draw traversal."""
        ax = getattr(self, 'ax', None)
        all_coords = getattr(ax, '_all_coords', None) if ax is not None else None
        if not isinstance(all_coords, list):
            return
        all_coords[:] = [coord_map for coord_map in all_coords if coord_map is not overlay]
        if getattr(ax, 'overlay_coords', None) is overlay:
            ax.overlay_coords = None
        try:
            if int(getattr(ax, '_display_coords_index', 0)) >= len(all_coords):
                ax._display_coords_index = 0
        except Exception:
            pass

    def _attach_grid_overlay(self, overlay):
        ax = getattr(self, 'ax', None)
        all_coords = getattr(ax, '_all_coords', None) if ax is not None else None
        if isinstance(all_coords, list) and not any(
            coord_map is overlay for coord_map in all_coords
        ):
            all_coords.append(overlay)
        if ax is not None:
            ax.overlay_coords = overlay

    def _grid_overlay_for_frame(self, frame: str):
        overlay = self._grid_overlay_cache.get(frame)
        if overlay is not None:
            self._attach_grid_overlay(overlay)
            return overlay
        ax = getattr(self, 'ax', None)
        if ax is None:
            return None
        try:
            overlay = ax.get_coords_overlay(frame)
        except Exception as exc:
            self.last_grid_error = exc
            return None
        self._grid_overlay_cache[frame] = overlay
        self._attach_grid_overlay(overlay)
        return overlay

    @staticmethod
    def _bounds_close(lhs, rhs, *, tolerance=1e-9) -> bool:
        if lhs is None or rhs is None:
            return False
        try:
            return all(
                abs(float(left) - float(right)) <= tolerance
                for left, right in zip(lhs, rhs)
            )
        except Exception:
            return False

    def update_grid_overlay_layout(
        self,
        *,
        colorbar_placement: str = None,
        colorbar_gap_x_px: float = None,
        colorbar_gap_y_px: float = None,
        colorbar_thickness_px: float = None,
        colorbar_left_decoration_px: float = None,
        colorbar_right_decoration_px: float = None,
        colorbar_bottom_decoration_px: float = None,
        colorbar_top_decoration_px: float = None,
    ) -> bool:
        """Reserve reversible space for top/right overlay coordinate labels.

        WCSAxes places a non-native longitude label above the axes and its
        latitude label to the right. Those decorations are outside ``ax.bbox``;
        a separately positioned colorbar would otherwise overlap them. The
        original user/config axes bounds are retained and restored as soon as
        the overlay is hidden or the selected frame returns to native.

        ``colorbar_placement`` is supplied by the GUI auto-layout path. Without
        it (for example headless export), the current colorbar axes position is
        used as the available boundary.
        """
        ax = getattr(self, 'ax', None)
        fig = getattr(self, 'fig', None)
        if ax is None or fig is None:
            return False

        try:
            current = tuple(float(value) for value in ax.get_position(original=True).bounds)
        except Exception:
            try:
                current = tuple(float(value) for value in ax.get_position().bounds)
            except Exception:
                return False

        active = bool(self.grid_overlay_active and self.active_grid_overlay is not None)
        if not active:
            self.grid_overlay_right_margin_px = float(
                getattr(
                    self,
                    '_grid_overlay_min_right_margin_px',
                    self.grid_overlay_right_margin_px,
                )
            )
            self.grid_overlay_top_margin_px = float(
                getattr(
                    self,
                    '_grid_overlay_min_top_margin_px',
                    self.grid_overlay_top_margin_px,
                )
            )
            base = self._grid_overlay_base_ax_bounds
            self._grid_overlay_layout_active = False
            self._grid_overlay_base_ax_bounds = None
            if base is None or self._bounds_close(current, base):
                return False
            try:
                ax.set_position(base)
                overlay_ax = getattr(self, 'overlay_ax', None)
                if overlay_ax is not None:
                    overlay_ax.set_position(ax.get_position())
            except Exception:
                return False
            return True

        if not self._grid_overlay_layout_active or self._grid_overlay_base_ax_bounds is None:
            self._grid_overlay_base_ax_bounds = current
        base = self._grid_overlay_base_ax_bounds
        self._grid_overlay_layout_active = True

        try:
            fig_width_px = max(1.0, float(fig.bbox.width))
            fig_height_px = max(1.0, float(fig.bbox.height))
        except Exception:
            fig_width_px = max(1.0, float(fig.get_figwidth() * fig.dpi))
            fig_height_px = max(1.0, float(fig.get_figheight() * fig.dpi))

        self._grow_grid_overlay_margins_from_rendered_labels()

        left, bottom, width, height = base
        base_right = left + width
        base_top = bottom + height
        max_right = 1.0 - self.grid_overlay_right_margin_px / fig_width_px
        max_top = 1.0 - self.grid_overlay_top_margin_px / fig_height_px

        placement = (
            str(colorbar_placement or '').strip().lower()
            if colorbar_placement is not None
            else None
        )
        if placement is not None:
            try:
                gap_x = max(0.0, float(colorbar_gap_x_px or 0.0))
            except Exception:
                gap_x = 0.0
            try:
                gap_y = max(0.0, float(colorbar_gap_y_px or 0.0))
            except Exception:
                gap_y = 0.0
            try:
                thickness = max(0.0, float(colorbar_thickness_px or 0.0))
            except Exception:
                thickness = 0.0
            try:
                left_decoration = max(
                    0.0, float(colorbar_left_decoration_px or 0.0)
                )
            except Exception:
                left_decoration = 0.0
            try:
                right_decoration = max(
                    0.0, float(colorbar_right_decoration_px or 0.0)
                )
            except Exception:
                right_decoration = 0.0
            try:
                bottom_decoration = max(
                    0.0, float(colorbar_bottom_decoration_px or 0.0)
                )
            except Exception:
                bottom_decoration = 0.0
            try:
                top_decoration = max(
                    0.0, float(colorbar_top_decoration_px or 0.0)
                )
            except Exception:
                top_decoration = 0.0
            if placement == 'right':
                max_right = 1.0 - (
                    self.grid_overlay_right_margin_px
                    + gap_x
                    + left_decoration
                    + thickness
                    + right_decoration
                ) / fig_width_px
            elif placement == 'top':
                max_top = 1.0 - (
                    self.grid_overlay_top_margin_px
                    + gap_y
                    + bottom_decoration
                    + thickness
                    + top_decoration
                ) / fig_height_px
        else:
            state_cax = getattr(
                getattr(self, 'viewer_state', None), 'cax', None
            )
            cax = state_cax if state_cax is not None else getattr(self, 'cax', None)
            if cax is not None:
                try:
                    active_bounds = tuple(
                        float(value) for value in ax.get_position().bounds
                    )
                    active_left, active_bottom, active_width, active_height = (
                        active_bounds
                    )
                    active_right = active_left + active_width
                    active_top = active_bottom + active_height
                    cbar_bounds = tuple(
                        float(value) for value in cax.get_position().bounds
                    )
                    cbar_left, cbar_bottom, cbar_width, cbar_height = cbar_bounds
                    cbar_right = cbar_left + cbar_width
                    cbar_top = cbar_bottom + cbar_height
                    left_decoration = 0.0
                    bottom_decoration = 0.0
                    try:
                        renderer = fig.canvas.get_renderer()
                        tight_bbox = cax.get_tightbbox(renderer)
                        axes_bbox = cax.bbox
                        left_decoration = max(
                            0.0,
                            float(axes_bbox.x0) - float(tight_bbox.x0),
                        )
                        bottom_decoration = max(
                            0.0,
                            float(axes_bbox.y0) - float(tight_bbox.y0),
                        )
                    except Exception:
                        pass
                    vertical_overlap = (
                        cbar_top > active_bottom and cbar_bottom < active_top
                    )
                    horizontal_overlap = (
                        cbar_right > active_left and cbar_left < active_right
                    )
                    vertical_bar = cbar_height >= cbar_width
                    horizontal_bar = cbar_width > cbar_height
                    right_boundary = (
                        cbar_left >= active_right
                        or (
                            vertical_bar
                            and cbar_left + 0.5 * cbar_width >= active_right
                        )
                    )
                    top_boundary = (
                        cbar_bottom >= active_top
                        or (
                            horizontal_bar
                            and cbar_bottom + 0.5 * cbar_height >= active_top
                        )
                    )
                    if right_boundary and vertical_overlap:
                        max_right = cbar_left - (
                            self.grid_overlay_right_margin_px + left_decoration
                        ) / fig_width_px
                    if top_boundary and horizontal_overlap:
                        max_top = cbar_bottom - (
                            self.grid_overlay_top_margin_px + bottom_decoration
                        ) / fig_height_px
                except Exception:
                    pass

        # Keep a useful image area even in an unusually small window. The
        # configured bounds remain the authority and are restored verbatim.
        min_width = min(width, 0.25)
        min_height = min(height, 0.25)
        target_right = max(left + min_width, min(base_right, max_right))
        target_top = max(bottom + min_height, min(base_top, max_top))
        target = (left, bottom, target_right - left, target_top - bottom)
        if self._bounds_close(current, target):
            return False
        try:
            ax.set_position(target)
            overlay_ax = getattr(self, 'overlay_ax', None)
            if overlay_ax is not None:
                overlay_ax.set_position(ax.get_position())
        except Exception:
            return False
        return True

    def set_grid(
        self,
        visible: bool,
        *,
        frame: str = None,
        keep_native: bool = None,
    ) -> bool:
        """Configure the WCS coordinate grid on this map's WCSAxes.

        The grid is a curved, world-coordinate grid drawn by ``WCSAxes`` itself,
        so it is correct for rotated / non-orthogonal WCS. Returns ``True`` when
        the state was applied, ``False`` when there is no axes/WCS to draw on.

        The grid is enabled only on the two *displayed* axes. Drawing it for a
        sliced-out axis is meaningless and, for coupled projections (GLS/SFL),
        actively wrong: e.g. on the ZY plane the sliced longitude still varies
        with latitude, so contouring it paints spurious, unevenly-spaced lines.

        On an all-celestial XY plane, a non-native display ``frame`` creates a
        cached WCSAxes coordinate overlay. XZ/ZY remain native because their
        celestial+spectral sliced WCS cannot be mapped to a 2-D sky frame.
        """
        visible = bool(visible)
        if frame is not None:
            self.grid_frame = normalize_display_frame(frame)
        if keep_native is not None:
            self.grid_keep_native = bool(keep_native)
        self.grid_visible = visible
        self.last_grid_error = None

        ax = getattr(self, 'ax', None)
        if ax is None:
            return False

        self._hide_grid_overlays()
        # Restore the unreserved bounds before rebuilding the overlay. This
        # keeps repeated frame/style changes idempotent.
        self.update_grid_overlay_layout()
        if self.wcs is None:
            return False
        overlay_frame = self._overlay_target_frame() if visible else None
        native_frame = native_celestial_frame(self.wcs) or 'native'

        if overlay_frame is not None:
            self._set_native_grid(self.grid_keep_native)
            overlay = self._grid_overlay_for_frame(overlay_frame)
            if overlay is not None and self._set_overlay_grid_visible(
                overlay, overlay_frame, True
            ):
                self.grid_overlay_active = True
                self.active_grid_overlay = overlay
                self.grid_effective_frame = overlay_frame
                self.update_grid_overlay_layout()
                return True

            # A target-frame transform can fail for unusual WCS metadata. Keep
            # the user-visible grid functional by falling back to native.
            self._hide_grid_overlays()
            self.grid_effective_frame = native_frame
            applied = self._set_native_grid(True)
            self.update_grid_overlay_layout()
            return applied

        self.grid_effective_frame = native_frame
        applied = self._set_native_grid(visible)
        self.update_grid_overlay_layout()
        return applied

    def _apply_major_tick_placement(self, coord, axis_key: str) -> None:
        """Honour `<axis>_tick_spacing` / `<axis>_tick_number` for one coord.

        `spacing` is interpreted in the axis' displayed format unit, so the
        number the user types matches the numbers on the tick labels. Astropy
        rejects passing both, so spacing wins. Any failure leaves the automatic
        placement untouched rather than breaking the render.
        """
        spacing = self._tick_spacing.get(axis_key)
        number = self._tick_number.get(axis_key)

        if spacing is not None:
            try:
                value = float(spacing)
            except (TypeError, ValueError):
                value = None
            if value is not None and value > 0.0:
                try:
                    unit = coord.get_format_unit()
                except Exception:
                    unit = None
                try:
                    coord.set_ticks(
                        spacing=value * unit if unit is not None else value
                    )
                    return
                except Exception:
                    pass

        if number is not None:
            try:
                count = int(number)
            except (TypeError, ValueError):
                return
            if count > 0:
                try:
                    coord.set_ticks(number=count)
                except Exception:
                    pass

    def update_axes_format(self):
        if not self.wcs:
            return
        self.decimal = self.config.get('decimal', True)
        self.coord_wrap = self.config.get('coord_wrap', 180)

        axis_ctype = self.wcs.wcs.ctype

        for idx, coord in enumerate(self.ax.coords):
            ctype_str = (axis_ctype[idx] or '').upper()
            ctype_head = ctype_str.split('-')[0]
            if _is_pv_scalar_axis(self.header, idx):
                continue
            if ctype_head.startswith('RA'):
                coord.set_coord_type('longitude', coord_wrap=360 * u.deg)
                if self.decimal:
                    coord.set_format_unit('deg', decimal=True)
                else:
                    coord.set_format_unit('hour', decimal=False) # hms
            elif ctype_head.startswith('DEC'):
                coord.set_coord_type('latitude')
                coord.set_format_unit('deg', decimal=self.decimal)
            elif any(keyword in ctype_head for keyword in ['GLON', 'GLAT', 'OFFSET']):
                if 'GLON' in ctype_head:
                    coord.set_coord_type(
                        'longitude',
                        coord_wrap=_coord_wrap_quantity(self.coord_wrap, 180),
                    )
                else:
                    coord.set_coord_type('latitude')
                coord.set_format_unit('deg', decimal=self.decimal)

        if self.grid_overlay_active and self.active_grid_overlay is not None:
            self._set_overlay_grid_visible(
                self.active_grid_overlay,
                self.grid_effective_frame,
                True,
            )
