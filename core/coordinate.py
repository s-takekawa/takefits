from astropy.coordinates import Angle
from astropy import units as u
import numpy as np
from functools import lru_cache
from core.common import Common


def _format_numeric_value(value: float, decimals: int) -> str:
    """Format numeric values with trimmed trailing zeros for readability."""
    rounded = round(value, decimals)
    formatted = f"{rounded:.{decimals}f}"
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    if formatted in {"-0", "+0"}:
        formatted = "0"
    return formatted

class CoordinateConverter:
    def __init__(self, wcs, config):
        self.wcs = wcs
        self.config = config

    def get_axis_types(self):
        """Retrieve CTYPE information from the WCS header for each axis."""
        axis_types = [self.wcs.wcs.ctype[i] for i in range(self.wcs.naxis)]
        return axis_types

    def format_world_coordinate(self, coord, axis_type):
        self.decimal = self.config.get('decimal', True)
        self.number_decimals = self.config.get('number_decimals', 6)
        """Format world coordinates based on axis type (RA/DEC, GLON/GLAT, etc.) and decimal option."""
        axis_upper = (axis_type or '').upper()
        if self.decimal:
            if any(token in axis_upper for token in ('VRAD', 'VELO', 'VOPT')) or 'FREQ' in axis_upper:
                return _format_numeric_value(coord, self.number_decimals)
            return f'{coord:.{self.number_decimals}f}'
        else:
            # Handle RA/DEC or GLON/GLAT based on CTYPE
            if axis_type[:2] == 'RA':
                # Convert RA to hourangle format
                return Angle(coord, unit=u.deg).to_string(unit=u.hourangle, sep='hms', precision=self.number_decimals, pad=True)
            elif 'DEC' in axis_type or 'GLAT' in axis_type or 'OFFSET' in axis_type:
                # Convert DEC or GLAT to dms format
                return Angle(coord, unit=u.deg).to_string(unit=u.deg, sep='dms', precision=self.number_decimals, alwayssign=True, pad=True)
            elif 'GLON' in axis_type:
                # Convert GLON to dms format
                return Angle(coord, unit=u.deg).to_string(unit=u.deg, sep='dms', precision=self.number_decimals, pad=True)
            elif 'VELO' in axis_type or 'VRAD' in axis_type or 'VOPT' in axis_type or 'FREQ' in axis_type:
                # Handle velocity or frequency (return as is)
                return _format_numeric_value(coord, self.number_decimals)
            else:
                # Default to numeric format if no specific type is matched
                return f'{coord:.{self.number_decimals}f}'

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
    def __init__(self, wcs, slices, ax, plane, decimal, number_decimals, wrap):
        self.decimal = decimal
        self.number_decimals = number_decimals
        self.coord_wrap = wrap
        self.slices = slices
        self.wcs = wcs

        if self.wcs is None:
            naxis = 2
            print("Warning: WCS is None. Using fallback value naxis=2 for coordinate formatting.")
        else:
            naxis = self.wcs.naxis

        units = [ax.coords[i].get_format_unit() for i in range(naxis) ]
        xaxis_unit, yaxis_unit = units[slices.index('x')], units[slices.index('y')]

        if self.wcs is None:
            print("Warning: WCS is None. Using fallback Cartesian coordinate types.")
            # For fallback, assume Cartesian types for both axes.
            xaxis_type, yaxis_type = 'Cartesian', 'Cartesian'
        else:
            # Use the slices to determine axis types.
            try:
                xaxis_type = self.wcs.axis_type_names[self.slices.index('x')]
                yaxis_type = self.wcs.axis_type_names[self.slices.index('y')]
            except Exception as e:
                print(f"Error determining axis types: {e}")
                xaxis_type, yaxis_type = 'Unknown', 'Unknown'

        Common.update_ax_units(plane, xaxis_unit, yaxis_unit )
        Common.update_ax_types(plane, xaxis_type, yaxis_type )
    
    def convert(self, plane, xdata, ydata):
        x, y = self.pix_to_wcs(self.wcs, xdata, ydata, plane)
        str_x, str_y, str_z = Common.world_x_str, Common.world_y_str, Common.world_z_str
        x = np.round(x, self.number_decimals)
        y = np.round(y, self.number_decimals)
        xval, yval = x, y
        if plane == 'xy':
            xunit, yunit = Common.ax_xy_xunit, Common.ax_xy_yunit
            xtype, ytype = Common.ax_xy_xtype, Common.ax_xy_ytype
        elif plane == 'xz':
            xunit, yunit = Common.ax_xz_xunit, Common.ax_xz_yunit
            xtype, ytype = Common.ax_xz_xtype, Common.ax_xz_ytype
        elif plane == 'zy':
            xunit, yunit = Common.ax_zy_xunit, Common.ax_zy_yunit
            xtype, ytype = Common.ax_zy_xtype, Common.ax_zy_ytype
        
        if xunit == 'deg':
            if self.coord_wrap == 180:
                if x < -180: x += 360
                elif x > 180: x -= 360
            elif self.coord_wrap == 360:
                if x < 0: x += 360
                elif x > 360: x -= 360
            xval = x
            x = Angle(x*u.deg).to_string(unit=u.deg, decimal = self.decimal)
        if yunit == 'deg' : y = Angle(y*u.deg).to_string(unit=u.deg, decimal = self.decimal)
        if xtype == 'GLON' and self.decimal == True:
            x = np.round(xval, self.number_decimals)
        if xtype == 'RA' and self.decimal == False: x = Angle(xval, unit=u.deg).to_string(unit=u.hourangle)
        xtype_upper = (xtype or '').upper()
        ytype_upper = (ytype or '').upper()
        if any(token in xtype_upper for token in ('VRAD', 'VOPT', 'VELO')) or 'FREQ' in xtype_upper:
            x = _format_numeric_value(xval, self.number_decimals)
        if any(token in ytype_upper for token in ('VRAD', 'VOPT', 'VELO')) or 'FREQ' in ytype_upper:
            y = _format_numeric_value(yval, self.number_decimals)
        
        if plane == 'xy':  Common.update_world_xyz_str(str(x), str(y), str_z)
        elif plane == 'xz':  Common.update_world_xyz_str(str(x), str_y, str(y))
        elif plane == 'zy':  Common.update_world_xyz_str(str_x, str(y), str(x))
        
        return str(x), str(y)
    
    @lru_cache(maxsize=None)
    def convert_chpix_to_world(self, plane, x, y, z):
        wcs = self.wcs
        if self.wcs.naxis == 2: return
        
        if self.wcs.naxis == 3:
            if plane == 'xy':
                xpix, ypix, zpix = float(x), float(y), float(z)
                xcoord, ycoord, zcoord = wcs.wcs_pix2world(xpix, ypix, zpix, 0)
                return zcoord
            elif plane == 'xz':
                xpix, ypix, zpix = float(x), float(z), float(y)
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
                xpix, ypix, zpix, spix = float(x), float(z), float(y), 0.0
                xcoord, ycoord, zcoord, val = wcs.wcs_pix2world(xpix, ypix, zpix, spix, 0)
                return ycoord
            elif plane == 'zy':
                xpix, ypix, zpix, spix = float(z), float(y), float(x), 0.0
                xcoord, ycoord, zcoord, val = wcs.wcs_pix2world(xpix, ypix, zpix, spix, 0)
                return xcoord
                
        

    def convert_chval_to_world_str(self, plane, z):
            zval = float(z)
            if plane == 'xy':
                ztype = Common.ax_zy_xtype
            elif plane == 'xz':
                ztype = Common.ax_zy_ytype
            elif plane == 'zy':
                ztype = Common.ax_xy_xtype
            else:
                ztype = None

            axis_type = (ztype or '').upper()
            value = zval

            if 'GLON' in axis_type:
                if self.coord_wrap == 180:
                    if value < -180:
                        value += 360
                    elif value > 180:
                        value -= 360
                elif self.coord_wrap == 360:
                    if value < 0:
                        value += 360
                    elif value > 360:
                        value -= 360

            if axis_type.startswith('RA'):
                angle = Angle(value, unit=u.deg)
                if self.decimal:
                    formatted = angle.to_string(unit=u.deg, decimal=True, precision=self.number_decimals)
                else:
                    formatted = angle.to_string(unit=u.hourangle, sep='hms', precision=self.number_decimals, pad=True)
            elif axis_type.startswith('DEC') or 'GLAT' in axis_type or 'LAT' in axis_type:
                angle = Angle(value, unit=u.deg)
                formatted = angle.to_string(
                    unit=u.deg,
                    decimal=self.decimal,
                    precision=self.number_decimals,
                    pad=True,
                )
            elif any(token in axis_type for token in ('VRAD', 'VELO', 'VOPT')) or 'FREQ' in axis_type:
                formatted = _format_numeric_value(value, self.number_decimals)
            else:
                formatted = f"{value:.{self.number_decimals}f}"

            return formatted
            
    def pix_to_wcs(self, wcs, xpix, ypix, plane):
        ndim = wcs.naxis
        xpix = float(xpix)
        ypix = float(ypix)
        xfixed = float(getattr(Common, 'xpix', 0))
        yfixed = float(getattr(Common, 'ypix', 0))
        zfixed = float(getattr(Common, 'zpix', 0))
        stokes_fixed = float(getattr(Common, 'spix', 0))

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

        if ndim >= 3:
            Common.update_world_xyz(world[0], world[1], world[2])
        elif ndim == 2:
            Common.update_world_xyz(world[0], world[1], Common.world_z)
        else:
            Common.update_world_xyz(world[0], Common.world_y, Common.world_z)

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
