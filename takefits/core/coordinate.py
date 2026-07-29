from astropy.coordinates import Angle
from astropy import units as u
import numpy as np
from takefits.core.wcs_frames import (
    axis_is_latitude,
    axis_is_longitude,
    axis_value_for_display,
    celestial_axis_indices,
    normalize_display_frame,
    plane_values_for_display,
)


def _format_numeric_value(value: float, decimals: int) -> str:
    """Format numeric values with trimmed trailing zeros for readability."""
    rounded = round(value, decimals)
    formatted = f"{rounded:.{decimals}f}"
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    if formatted in {"-0", "+0"}:
        formatted = "0"
    return formatted


def _unit_text(unit) -> str:
    """Normalize unit objects/strings for robust comparisons."""
    if unit is None:
        return ""
    try:
        if hasattr(unit, "to_string"):
            return str(unit.to_string()).strip().lower()
    except Exception:
        pass
    return str(unit).strip().lower()


def _is_degree_unit(unit) -> bool:
    text = _unit_text(unit)
    return text in {"deg", "degree"}


def _is_pv_scalar_axis(fits_viewer, axis_index: int) -> bool:
    header = getattr(fits_viewer, "header", None)
    if header is None:
        return False
    if not (
        str(header.get("PVXAXIS", "") or "").strip()
        or str(header.get("PVPATH", "") or "").strip()
    ):
        return False
    try:
        axis_no = int(axis_index) + 1
    except Exception:
        return False
    ctype = str(header.get(f"CTYPE{axis_no}", "") or "").strip().upper().split("-")[0]
    return ctype in {"PHI", "OFFSET"}


def _wrap_longitude_value(value: float, axis_type: object, coord_wrap: object = 180) -> float:
    axis_upper = str(axis_type or "").upper()
    wrapped = float(value)

    if axis_upper.startswith("RA"):
        return wrapped % 360.0

    if not axis_is_longitude(axis_type):
        return wrapped

    try:
        wrap_mode = int(coord_wrap)
    except Exception:
        wrap_mode = 180

    if wrap_mode == 360:
        wrapped %= 360.0
        if wrapped < 0.0:
            wrapped += 360.0
        return wrapped

    wrapped = ((wrapped + 180.0) % 360.0) - 180.0
    if wrapped == -180.0 and float(value) > 0.0:
        return 180.0
    return wrapped


def _config_bool(config, key: str, default: bool = True) -> bool:
    if not isinstance(config, dict):
        return bool(default)
    return bool(config.get(key, default))


def _auto_decimals_from_pixel_step(step, fallback: int, *, max_decimals: int = 10) -> int:
    """
    Derive display decimals from pixel scale:
    enough precision for roughly 1/10 pixel in world units.
    """
    try:
        default = int(fallback)
    except Exception:
        default = 6
    default = max(0, default)

    try:
        value = abs(float(step))
    except Exception:
        return default
    if not np.isfinite(value) or value <= 0:
        return default

    try:
        decimals = int(np.ceil(-np.log10(value / 10.0)))
    except Exception:
        return default
    return max(0, min(int(max_decimals), decimals))


def _auto_sexagesimal_precision_from_pixel_step(
    step_deg,
    fallback: int,
    *,
    hourangle: bool = False,
    max_precision: int = 6,
) -> int:
    """
    Derive sexagesimal precision (fractional second digits in hms/dms) from pixel
    scale, targeting roughly 1/10 pixel in angular resolution.
    """
    try:
        default = int(fallback)
    except Exception:
        default = 3
    default = max(0, default)

    try:
        value_deg = abs(float(step_deg))
    except Exception:
        return default
    if not np.isfinite(value_deg) or value_deg <= 0:
        return default

    target_deg = value_deg / 10.0
    if hourangle:
        # 1 second of time = 1/240 degree.
        target_seconds = target_deg * 240.0
    else:
        # dms seconds (arcsec): 1 degree = 3600 arcsec.
        target_seconds = target_deg * 3600.0

    if not np.isfinite(target_seconds) or target_seconds <= 0:
        return default
    try:
        precision = int(np.ceil(-np.log10(target_seconds)))
    except Exception:
        return default
    return max(0, min(int(max_precision), precision))

class CoordinateConverter:
    def __init__(self, wcs, config):
        self.wcs = wcs
        self.config = config

    def _axis_step_from_index(self, axis_index: int):
        try:
            idx = int(axis_index)
        except Exception:
            return None
        try:
            return float(self.wcs.wcs.cdelt[idx])
        except Exception:
            return None

    def _axis_index_from_type(self, axis_type) -> int:
        try:
            axis_types = list(self.get_axis_types() or [])
        except Exception:
            return -1
        if not axis_types:
            return -1

        target = str(axis_type or "").upper()
        if not target:
            return -1
        for idx, candidate in enumerate(axis_types):
            if str(candidate or "").upper() == target:
                return idx

        target_head = target.split("-")[0]
        for idx, candidate in enumerate(axis_types):
            candidate_head = str(candidate or "").upper().split("-")[0]
            if candidate_head == target_head:
                return idx

        # Display-frame transformed celestial labels (e.g. native Galactic -> displayed RA/DEC)
        # do not directly match CTYPE headers. In that case map longitude/latitude labels
        # to the native celestial axis indices so auto precision can still use pixel scale.
        try:
            celestial = celestial_axis_indices(self.wcs)
        except Exception:
            celestial = None
        if celestial:
            if axis_is_longitude(target):
                try:
                    return int(celestial[0])
                except Exception:
                    return -1
            if axis_is_latitude(target):
                try:
                    return int(celestial[1])
                except Exception:
                    return -1
        return -1

    def _effective_decimals(self, axis_type) -> int:
        base = self.config.get('number_decimals', 6)
        try:
            base_int = max(0, int(base))
        except Exception:
            base_int = 6
        if not _config_bool(self.config, 'auto_precision_digits', True):
            return base_int
        axis_index = self._axis_index_from_type(axis_type)
        if axis_index < 0:
            return base_int
        step = self._axis_step_from_index(axis_index)
        if step is None:
            return base_int
        return _auto_decimals_from_pixel_step(step, base)

    def _effective_sexagesimal_precision(self, axis_type) -> int:
        base = self.config.get('number_decimals', 6)
        try:
            base_int = max(0, int(base))
        except Exception:
            base_int = 3
        if not _config_bool(self.config, 'auto_precision_digits', True):
            return base_int
        axis_index = self._axis_index_from_type(axis_type)
        if axis_index < 0:
            return base_int
        step = self._axis_step_from_index(axis_index)
        if step is None:
            return base_int
        axis_upper = str(axis_type or '').upper()
        return _auto_sexagesimal_precision_from_pixel_step(
            step,
            base,
            hourangle=axis_upper.startswith('RA'),
        )

    def get_axis_types(self):
        """Retrieve CTYPE information from the WCS header for each axis."""
        axis_types = [self.wcs.wcs.ctype[i] for i in range(self.wcs.naxis)]
        return axis_types

    def format_world_coordinate(self, coord, axis_type):
        """Format a world coordinate for the active display style."""
        self.decimal = self.config.get('decimal', True)
        numeric_decimals = self._effective_decimals(axis_type)
        sexagesimal_precision = self._effective_sexagesimal_precision(axis_type)
        self.number_decimals = numeric_decimals
        axis_upper = (axis_type or '').upper()
        coord_wrap = self.config.get('coord_wrap', 180)
        value = _wrap_longitude_value(float(coord), axis_type, coord_wrap)
        if self.decimal:
            if any(token in axis_upper for token in ('VRAD', 'VELO', 'VOPT')) or 'FREQ' in axis_upper:
                return _format_numeric_value(value, numeric_decimals)
            return f'{value:.{numeric_decimals}f}'
        else:
            # Handle RA/DEC or GLON/GLAT based on CTYPE
            if axis_type[:2] == 'RA':
                # Convert RA to hourangle format
                return Angle(value, unit=u.deg).to_string(
                    unit=u.hourangle,
                    sep='hms',
                    precision=sexagesimal_precision,
                    pad=True,
                )
            elif 'DEC' in axis_type or 'GLAT' in axis_type or 'OFFSET' in axis_type:
                # Convert DEC or GLAT to dms format
                return Angle(value, unit=u.deg).to_string(
                    unit=u.deg,
                    sep='dms',
                    precision=sexagesimal_precision,
                    alwayssign=True,
                    pad=True,
                )
            elif 'GLON' in axis_type:
                # Convert GLON to dms format
                return Angle(value, unit=u.deg).to_string(
                    unit=u.deg,
                    sep='dms',
                    precision=sexagesimal_precision,
                    pad=True,
                )
            elif 'VELO' in axis_type or 'VRAD' in axis_type or 'VOPT' in axis_type or 'FREQ' in axis_type:
                # Handle velocity or frequency (return as is)
                return _format_numeric_value(value, numeric_decimals)
            else:
                # Default to numeric format if no specific type is matched
                return f'{value:.{numeric_decimals}f}'

    def pix_to_world(self, *pixel_coords):
        """Convert pixel coordinates to world coordinates and format them."""
        pixel_coords = list(pixel_coords)
        while len(pixel_coords) < self.wcs.naxis:
            pixel_coords.append(0)
        pixel_array = np.atleast_2d(pixel_coords)

        world_coords = self.wcs.wcs_pix2world(pixel_array, 0)[0]

        #world_coords = self.wcs.wcs_pix2world(*pixel_coords, 0)
        axis_types = self.get_axis_types()
        formatted_coords = []
        self.coord_wrap = self.config.get('coord_wrap', 180)
        for coord, axis_type in zip(world_coords, axis_types):
            if "GLON" in axis_type:
                if self.coord_wrap == 180:
                    if coord > 180: coord -= 360
                    elif coord < -180: coord += 360
                elif self.coord_wrap == 360:
                    if coord > 360: coord -= 360
                    elif coord < 0: coord += 360
            formatted_coords.append(self.format_world_coordinate(coord, axis_type))

        return formatted_coords

    def world_to_pix(self, *world_coords):
        """Convert formatted world coordinates back to pixel coordinates."""
        pixel_coords = []
        axis_types = self.get_axis_types()

        for coord, axis_type in zip(world_coords, axis_types):
            if isinstance(coord, str):
                # Check for sexagesimal format indicators
                if any(c in coord for c in ['h', 'm', 's', 'd', ':', ' ']):
                    try:
                        if axis_type[:2] == 'RA':
                            coord = Angle(coord, unit=u.hourangle).degree
                        else: # Handles DEC, GLON, GLAT
                            coord = Angle(coord, unit=u.deg).degree
                    except u.UnitsError:
                        # Fallback for cases where unit is not explicitly hourangle
                        try:
                            coord = Angle(coord, unit=u.deg).degree
                        except Exception as e:
                            raise ValueError(f"Could not parse coordinate string: {coord}") from e
            
            pixel_coords.append(float(coord))

        while len(pixel_coords) < self.wcs.naxis:
            pixel_coords.append(0)
            
        return self.wcs.wcs_world2pix([pixel_coords], 0)[0]


class Format_pix_to_wcs:
    def __init__(
        self,
        wcs,
        slices,
        ax,
        plane,
        decimal,
        number_decimals,
        wrap,
        fits_viewer=None,
        auto_precision_digits=True,
    ):
        self.fits_viewer = fits_viewer
        self.decimal = decimal
        self.number_decimals = number_decimals
        self.coord_wrap = wrap
        self.auto_precision_digits = bool(auto_precision_digits)
        self.slices = slices
        self.wcs = wcs

        if self.wcs is None:
            naxis = 2
        else:
            naxis = self.wcs.naxis

        units = [ax.coords[i].get_format_unit() for i in range(naxis) ]
        xaxis_unit, yaxis_unit = units[slices.index('x')], units[slices.index('y')]

        if self.wcs is None:
            xaxis_type, yaxis_type = 'Cartesian', 'Cartesian'
        else:
            # Use the slices to determine axis types.
            try:
                xaxis_type = self.wcs.axis_type_names[self.slices.index('x')]
                yaxis_type = self.wcs.axis_type_names[self.slices.index('y')]
            except Exception:
                xaxis_type, yaxis_type = 'Unknown', 'Unknown'

        # Update ViewerState with axis units and types
        if self.fits_viewer is not None and hasattr(self.fits_viewer, 'state'):
            self.fits_viewer.state.update_ax_units(xaxis_unit, yaxis_unit)
            self.fits_viewer.state.update_ax_types(xaxis_type, yaxis_type)

    def _auto_precision_enabled(self) -> bool:
        if self.fits_viewer is not None:
            config_manager = getattr(self.fits_viewer, "config_manager", None)
            config = getattr(config_manager, "config", None)
            if isinstance(config, dict):
                return _config_bool(config, "auto_precision_digits", self.auto_precision_digits)

            displaymap = getattr(self.fits_viewer, "displaymap", None)
            config = getattr(displaymap, "config", None)
            if isinstance(config, dict):
                return _config_bool(config, "auto_precision_digits", self.auto_precision_digits)
        return bool(self.auto_precision_digits)

    def _effective_decimals_for_axis(self, axis_index: int) -> int:
        base = self.number_decimals
        try:
            base_int = max(0, int(base))
        except Exception:
            base_int = 6
        if not self._auto_precision_enabled():
            return base_int
        if axis_index is None or axis_index < 0:
            return base_int
        try:
            step = self.wcs.wcs.cdelt[int(axis_index)]
        except Exception:
            return base_int
        return _auto_decimals_from_pixel_step(step, base)

    def _sexagesimal_precision_for_axis(self, axis_index: int, *, hourangle: bool = False) -> int:
        base = self.number_decimals
        try:
            base_int = max(0, int(base))
        except Exception:
            base_int = 3
        if not self._auto_precision_enabled():
            return base_int
        if axis_index is None or axis_index < 0:
            return base_int
        try:
            step = self.wcs.wcs.cdelt[int(axis_index)]
        except Exception:
            return base_int
        return _auto_sexagesimal_precision_from_pixel_step(
            step,
            base,
            hourangle=bool(hourangle),
        )

    def _plane_axis_indices(self, plane: str):
        x_axis = None
        y_axis = None
        try:
            if isinstance(self.slices, (tuple, list)):
                if 'x' in self.slices:
                    x_axis = int(self.slices.index('x'))
                if 'y' in self.slices:
                    y_axis = int(self.slices.index('y'))
        except Exception:
            x_axis = None
            y_axis = None

        if x_axis is None or y_axis is None:
            fallback = {
                'xy': (0, 1),
                'xz': (0, 2),
                'zy': (2, 1),
            }.get(str(plane or '').lower(), (0, 1))
            if x_axis is None:
                x_axis = int(fallback[0])
            if y_axis is None:
                y_axis = int(fallback[1])
        return x_axis, y_axis
    
    def convert(self, plane, xdata, ydata):
        x, y = self.pix_to_wcs(self.wcs, xdata, ydata, plane)
        x_axis_index, y_axis_index = self._plane_axis_indices(plane)
        x_decimals = self._effective_decimals_for_axis(x_axis_index)
        y_decimals = self._effective_decimals_for_axis(y_axis_index)

        # Get world strings from coordinator via fits_viewer
        str_x, str_y, str_z = "", "", ""
        if self.fits_viewer is not None:
            coord = self.fits_viewer.get_coordinator()
            if coord is not None:
                str_x = coord.world_x_str
                str_y = coord.world_y_str
                str_z = coord.world_z_str

        x = np.round(x, x_decimals)
        y = np.round(y, y_decimals)
        xval, yval = x, y

        # Get axis units/types from ViewerState
        xunit, yunit, xtype, ytype = None, None, None, None
        if self.fits_viewer is not None:
            state = self.fits_viewer.get_viewer_state(plane)
            if state is not None:
                xunit, yunit = state.ax_xunit, state.ax_yunit
                xtype, ytype = state.ax_xtype, state.ax_ytype
        x_unit_is_deg = _is_degree_unit(xunit)
        y_unit_is_deg = _is_degree_unit(yunit)
        x_is_pv_scalar = _is_pv_scalar_axis(self.fits_viewer, x_axis_index)
        y_is_pv_scalar = _is_pv_scalar_axis(self.fits_viewer, y_axis_index)

        display_frame = "native"
        fallback_native_world = None
        use_display_frame = plane == "xy"
        if use_display_frame and self.fits_viewer is not None and hasattr(self.fits_viewer, "_get_shared_display_frame"):
            display_frame = normalize_display_frame(self.fits_viewer._get_shared_display_frame())
            fallback_native_world = []
            for getter in (
                getattr(self.fits_viewer, "_get_shared_world_x", None),
                getattr(self.fits_viewer, "_get_shared_world_y", None),
                getattr(self.fits_viewer, "_get_shared_world_z", None),
                getattr(self.fits_viewer, "_get_shared_world_s", None),
            ):
                if callable(getter):
                    try:
                        fallback_native_world.append(float(getter()))
                    except Exception:
                        fallback_native_world.append(None)
                else:
                    fallback_native_world.append(None)

        display_x = xval
        display_y = yval
        display_xtype = xtype
        display_ytype = ytype
        if display_frame != "native":
            try:
                transformed_x, transformed_y, transformed_xtype, transformed_ytype = plane_values_for_display(
                    self.wcs,
                    plane,
                    xval,
                    yval,
                    frame=display_frame,
                    fallback_native_world=fallback_native_world,
                    coord_wrap=self.coord_wrap,
                )
                display_x = transformed_x
                display_y = transformed_y
                if transformed_xtype:
                    display_xtype = transformed_xtype
                if transformed_ytype:
                    display_ytype = transformed_ytype
            except Exception:
                display_x = xval
                display_y = yval

        xtype_upper = (display_xtype or '').upper()
        ytype_upper = (display_ytype or '').upper()
        x_is_angular_axis = (
            axis_is_longitude(display_xtype)
            or axis_is_latitude(display_xtype)
            or (x_unit_is_deg and not x_is_pv_scalar)
        )
        y_is_angular_axis = (
            axis_is_longitude(display_ytype)
            or axis_is_latitude(display_ytype)
            or (y_unit_is_deg and not y_is_pv_scalar)
        )

        if x_is_angular_axis:
            display_x = _wrap_longitude_value(display_x, display_xtype, self.coord_wrap)
            xval = display_x
            x_is_hourangle = (not self.decimal) and xtype_upper.startswith("RA")
            x_sexagesimal_precision = self._sexagesimal_precision_for_axis(
                x_axis_index,
                hourangle=x_is_hourangle,
            )
            x = Angle(display_x*u.deg).to_string(
                unit=u.deg,
                decimal=self.decimal,
                precision=(x_decimals if self.decimal else x_sexagesimal_precision),
            )
        else:
            x = display_x
        if y_is_angular_axis:
            y_is_hourangle = (not self.decimal) and ytype_upper.startswith("RA")
            y_sexagesimal_precision = self._sexagesimal_precision_for_axis(
                y_axis_index,
                hourangle=y_is_hourangle,
            )
            y = Angle(display_y*u.deg).to_string(
                unit=u.deg,
                decimal=self.decimal,
                precision=(y_decimals if self.decimal else y_sexagesimal_precision),
            )
        else:
            y = display_y
        if xtype_upper.startswith('RA') and self.decimal == False:
            x_ra_precision = self._sexagesimal_precision_for_axis(x_axis_index, hourangle=True)
            x = Angle(xval, unit=u.deg).to_string(
                unit=u.hourangle,
                sep='hms',
                precision=x_ra_precision,
                pad=True,
            )
        if any(token in xtype_upper for token in ('VRAD', 'VOPT', 'VELO')) or 'FREQ' in xtype_upper:
            x = _format_numeric_value(display_x, x_decimals)
        if any(token in ytype_upper for token in ('VRAD', 'VOPT', 'VELO')) or 'FREQ' in ytype_upper:
            y = _format_numeric_value(display_y, y_decimals)
        # Keep numeric output precision aligned with config when value is not angle-formatted text.
        try:
            if not isinstance(x, str):
                x = _format_numeric_value(float(x), x_decimals)
        except Exception:
            pass
        try:
            if not isinstance(y, str):
                y = _format_numeric_value(float(y), y_decimals)
        except Exception:
            pass

        # Update world strings through fits_viewer
        if self.fits_viewer is not None:
            if plane == 'xy':
                self.fits_viewer._update_shared_world_xyz_str(str(x), str(y), str_z)
            elif plane == 'xz':
                self.fits_viewer._update_shared_world_xyz_str(str(x), str_y, str(y))
            elif plane == 'zy':
                self.fits_viewer._update_shared_world_xyz_str(str_x, str(y), str(x))

        return str(x), str(y)
    
    def convert_chpix_to_world(self, plane, x, y, z):
        wcs = self.wcs
        if self.wcs.naxis == 2: return
        
        if self.wcs.naxis == 3:
            if plane == 'xy':
                xpix, ypix, zpix = float(x), float(y), float(z)
                xcoord, ycoord, zcoord = wcs.wcs_pix2world(xpix, ypix, zpix, 0)
                return zcoord
            elif plane == 'xz':
                xpix, ypix, zpix = float(x), float(y), float(z)
                xcoord, ycoord, zcoord = wcs.wcs_pix2world(xpix, ypix, zpix, 0)
                return ycoord
            elif plane == 'zy':
                xpix, ypix, zpix = float(z), float(y), float(x)
                xcoord, ycoord, zcoord = wcs.wcs_pix2world(xpix, ypix, zpix, 0)
                return xcoord
            
        if self.wcs.naxis == 4:
            if plane == 'xy':
                xpix, ypix, zpix, spix = float(x), float(y), float(z), 0.0
                xcoord, ycoord, zcoord, val = wcs.wcs_pix2world(xpix, ypix, zpix, spix, 0)
                return zcoord
            elif plane == 'xz':
                xpix, ypix, zpix, spix = float(x), float(y), float(z), 0.0
                xcoord, ycoord, zcoord, val = wcs.wcs_pix2world(xpix, ypix, zpix, spix, 0)
                return ycoord
            elif plane == 'zy':
                xpix, ypix, zpix, spix = float(z), float(y), float(x), 0.0
                xcoord, ycoord, zcoord, val = wcs.wcs_pix2world(xpix, ypix, zpix, spix, 0)
                return xcoord
                
        

    def convert_chval_to_world_str(self, plane, z):
            zval = float(z)
            # Get axis type from ViewerState
            ztype = None
            if self.fits_viewer is not None:
                if plane == 'xy':
                    state = self.fits_viewer.get_viewer_state('zy')
                    if state is not None:
                        ztype = state.ax_xtype
                elif plane == 'xz':
                    state = self.fits_viewer.get_viewer_state('zy')
                    if state is not None:
                        ztype = state.ax_ytype
                elif plane == 'zy':
                    state = self.fits_viewer.get_viewer_state('xy')
                    if state is not None:
                        ztype = state.ax_xtype

            display_frame = "native"
            fallback_native_world = None
            # For xz/zy slice values, keep native-axis semantics even when display frame changes.
            use_display_frame = plane == "xy"
            if use_display_frame and self.fits_viewer is not None and hasattr(self.fits_viewer, "_get_shared_display_frame"):
                display_frame = normalize_display_frame(self.fits_viewer._get_shared_display_frame())
                fallback_native_world = []
                for getter in (
                    getattr(self.fits_viewer, "_get_shared_world_x", None),
                    getattr(self.fits_viewer, "_get_shared_world_y", None),
                    getattr(self.fits_viewer, "_get_shared_world_z", None),
                    getattr(self.fits_viewer, "_get_shared_world_s", None),
                ):
                    if callable(getter):
                        try:
                            fallback_native_world.append(float(getter()))
                        except Exception:
                            fallback_native_world.append(None)
                    else:
                        fallback_native_world.append(None)

            axis_index = None
            if plane == "xy":
                axis_index = 2
            elif plane == "xz":
                axis_index = 1
            elif plane == "zy":
                axis_index = 0
            if axis_index is not None and self.wcs is not None:
                try:
                    value, display_axis = axis_value_for_display(
                        self.wcs,
                        axis_index,
                        zval,
                        frame=display_frame,
                        fallback_native_world=fallback_native_world,
                        coord_wrap=self.coord_wrap,
                    )
                    ztype = display_axis or ztype
                except Exception:
                    value = zval
            else:
                value = zval
            axis_decimals = self._effective_decimals_for_axis(axis_index if axis_index is not None else -1)
            axis_is_hourangle = (not self.decimal) and str(ztype or '').upper().startswith('RA')
            axis_sexagesimal_precision = self._sexagesimal_precision_for_axis(
                axis_index if axis_index is not None else -1,
                hourangle=axis_is_hourangle,
            )

            axis_type = (ztype or '').upper()

            value = _wrap_longitude_value(value, axis_type, self.coord_wrap)

            if axis_type.startswith('RA'):
                angle = Angle(value, unit=u.deg)
                if self.decimal:
                    formatted = angle.to_string(unit=u.deg, decimal=True, precision=axis_decimals)
                else:
                    formatted = angle.to_string(
                        unit=u.hourangle,
                        sep='hms',
                        precision=axis_sexagesimal_precision,
                        pad=True,
                    )
            elif axis_type.startswith('DEC') or 'GLAT' in axis_type or 'LAT' in axis_type:
                angle = Angle(value, unit=u.deg)
                formatted = angle.to_string(
                    unit=u.deg,
                    decimal=self.decimal,
                    precision=(axis_decimals if self.decimal else axis_sexagesimal_precision),
                    pad=True,
                )
            elif any(token in axis_type for token in ('VRAD', 'VELO', 'VOPT')) or 'FREQ' in axis_type:
                formatted = _format_numeric_value(value, axis_decimals)
            else:
                formatted = f"{value:.{axis_decimals}f}"

            return formatted
            
    def pix_to_wcs(self, wcs, xpix, ypix, plane):
        ndim = wcs.naxis
        xpix = float(xpix)
        ypix = float(ypix)

        # Get fixed pixel values from coordinator
        xfixed, yfixed, zfixed, stokes_fixed = 0.0, 0.0, 0.0, 0.0
        if self.fits_viewer is not None:
            xfixed = float(self.fits_viewer._get_shared_xpix())
            yfixed = float(self.fits_viewer._get_shared_ypix())
            zfixed = float(self.fits_viewer._get_shared_zpix())

        if ndim == 0:
            return xpix, ypix

        # Build pixel coordinates for each WCS axis.
        pixel_args = []
        for axis in range(ndim):
            if axis == 0:
                if plane == 'zy' and ndim >= 1:
                    pixel_args.append(xfixed)
                else:
                    pixel_args.append(xpix)
            elif axis == 1:
                if plane == 'xz' and ndim >= 2:
                    pixel_args.append(yfixed)
                else:
                    pixel_args.append(ypix)
            elif axis == 2:
                if plane == 'xy':
                    pixel_args.append(zfixed)
                elif plane == 'xz':
                    pixel_args.append(ypix)
                elif plane == 'zy':
                    pixel_args.append(xpix)
                else:
                    pixel_args.append(zfixed)
            elif axis == 3:
                pixel_args.append(stokes_fixed)
            else:
                pixel_args.append(0.0)

        # Perform the WCS transformation.
        world = wcs.wcs_pix2world(*pixel_args, 0)
        world = np.atleast_1d(np.asarray(world, dtype=float))

        if self.fits_viewer:
            if ndim >= 3:
                 self.fits_viewer.update_coordinates(world[0], world[1], world[2])
            elif ndim == 2:
                 self.fits_viewer.update_coordinates(world[0], world[1], self.fits_viewer.world_z if self.fits_viewer.world_z is not None else 0.0)
            else:
                 self.fits_viewer.update_coordinates(world[0], self.fits_viewer.world_y if self.fits_viewer.world_y is not None else 0.0, self.fits_viewer.world_z if self.fits_viewer.world_z is not None else 0.0)



        if plane == 'xy':
            xcoord = world[0]
            ycoord = world[1] if ndim >= 2 else 0.0
        elif plane == 'xz':
            xcoord = world[0]
            ycoord = world[2] if ndim >= 3 else (world[1] if ndim >= 2 else 0.0)
        elif plane == 'zy':
            xcoord = world[2] if ndim >= 3 else world[0]
            ycoord = world[1] if ndim >= 2 else 0.0
        else:
            xcoord = world[0]
            ycoord = world[1] if ndim >= 2 else 0.0

        return float(xcoord), float(ycoord)
