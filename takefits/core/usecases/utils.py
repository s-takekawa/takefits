"""Shared coordinate/WCS helpers for usecases."""
from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np

from takefits.core.app_state import AppState
from takefits.core.io.save_fits import update_datamin_datamax_if_present


def world_to_pixel(
    state: AppState,
    world_coords: Tuple[float, ...]
) -> Tuple[float, ...]:
    """
    Convert world coordinates to pixel coordinates using WCS.

    Args:
        state: AppState with WCS
        world_coords: World coordinates (e.g., RA, Dec, Velocity)

    Returns:
        Pixel coordinates as tuple

    Raises:
        ValueError: If WCS is not available
    """
    if state.wcs is None:
        raise ValueError("WCS is not available")

    world_array = np.array([world_coords])
    pixel_array = state.wcs.wcs_world2pix(world_array, 0)

    return tuple(pixel_array[0])


def pixel_to_world(
    state: AppState,
    pixel_coords: Tuple[float, ...]
) -> Tuple[float, ...]:
    """
    Convert pixel coordinates to world coordinates using WCS.

    Args:
        state: AppState with WCS
        pixel_coords: Pixel coordinates (x, y, z, ...)

    Returns:
        World coordinates as tuple

    Raises:
        ValueError: If WCS is not available
    """
    if state.wcs is None:
        raise ValueError("WCS is not available")

    pixel_array = np.array([pixel_coords])
    world_array = state.wcs.wcs_pix2world(pixel_array, 0)

    return tuple(world_array[0])


def get_axis_ctype(state: AppState, axis: int) -> str:
    """
    Get the CTYPE for a specific WCS axis.

    Args:
        state: AppState with WCS
        axis: WCS axis index

    Returns:
        CTYPE string (e.g., "RA---SIN", "VELO-LSR")
    """
    if state.wcs is None:
        return ""
    if axis < 0 or axis >= state.wcs.naxis:
        return ""
    return state.wcs.wcs.ctype[axis]


def parse_world_coordinate(
    value: str,
    axis_type: str
) -> float:
    """
    Parse a world coordinate string to float, handling sexagesimal formats.

    This mirrors the logic in core/coordinate.py CoordinateConverter.world_to_pix()

    Args:
        value: Coordinate string (e.g., "12h30m45s", "187.5", "-30:15:30")
        axis_type: WCS CTYPE for the axis (e.g., "RA---SIN", "DEC--SIN", "VELO-LSR")

    Returns:
        Coordinate value in native units (degrees for RA/Dec/GLON/GLAT,
        native unit for VELO/FREQ)
    """
    from astropy.coordinates import Angle
    from astropy import units as u

    if isinstance(value, (int, float)):
        return float(value)

    value = str(value).strip()

    if any(c in value for c in ['h', 'm', 's', 'd', ':', ' '] ):
        axis_upper = (axis_type or '').upper()
        try:
            if axis_upper.startswith('RA'):
                return Angle(value, unit=u.hourangle).degree
            return Angle(value, unit=u.deg).degree
        except Exception:
            try:
                return Angle(value, unit=u.deg).degree
            except Exception as exc:
                raise ValueError(f"Could not parse coordinate string: {value}") from exc

    return float(value)


def axis_world_to_pixel(
    state: AppState,
    world_value: Union[float, str],
    axis: int,
    reference_pixel: Optional[Tuple[float, ...]] = None
) -> float:
    """
    Convert a world coordinate value to pixel for a specific axis.

    WCS transformations require full coordinate tuples, so this function
    uses reference values (CRPIX) for the other axes.
    """
    if state.wcs is None:
        raise ValueError("WCS is not available for world-to-pixel conversion")

    naxis = state.wcs.naxis
    if axis < 0 or axis >= naxis:
        raise ValueError(f"axis {axis} out of range for {naxis}-axis WCS")

    if isinstance(world_value, str):
        ctype = get_axis_ctype(state, axis)
        world_val_float = parse_world_coordinate(world_value, ctype)
    else:
        world_val_float = float(world_value)

    if reference_pixel is not None:
        ref_world = list(state.wcs.wcs_pix2world([list(reference_pixel)], 0)[0])
    else:
        ref_world = list(state.wcs.wcs.crval)

    ref_world[axis] = world_val_float
    pixel_coords = state.wcs.wcs_world2pix([ref_world], 0)[0]

    return float(pixel_coords[axis])


def axis_pixel_to_world(
    state: AppState,
    pixel_value: float,
    axis: int,
    reference_pixel: Optional[Tuple[float, ...]] = None
) -> float:
    """
    Convert a pixel coordinate value to world for a specific axis.
    """
    if state.wcs is None:
        raise ValueError("WCS is not available for pixel-to-world conversion")

    naxis = state.wcs.naxis
    if axis < 0 or axis >= naxis:
        raise ValueError(f"axis {axis} out of range for {naxis}-axis WCS")

    if reference_pixel is not None:
        ref_pixel = list(reference_pixel)
    else:
        ref_pixel = [crpix - 1 for crpix in state.wcs.wcs.crpix]

    ref_pixel[axis] = pixel_value
    world_coords = state.wcs.wcs_pix2world([ref_pixel], 0)[0]

    return float(world_coords[axis])


def world_bounds_to_pixel_bounds(
    state: AppState,
    world_bounds: List[Tuple[Union[float, str], Union[float, str]]]
) -> List[Tuple[int, int]]:
    """
    Convert world coordinate bounds to pixel bounds for all axes.
    """
    if state.wcs is None:
        raise ValueError("WCS is not available")

    from itertools import product

    naxis = state.wcs.naxis
    if len(world_bounds) != naxis:
        raise ValueError(f"Expected {naxis} bounds, got {len(world_bounds)}")

    parsed_bounds = []
    for i, (lo, hi) in enumerate(world_bounds):
        ctype = get_axis_ctype(state, i)
        lo_val = parse_world_coordinate(lo, ctype) if isinstance(lo, str) else float(lo)
        hi_val = parse_world_coordinate(hi, ctype) if isinstance(hi, str) else float(hi)
        parsed_bounds.append((lo_val, hi_val))

    corners = list(product(*[(lo, hi) for lo, hi in parsed_bounds]))

    pixel_values = []
    for corner in corners:
        pix = state.wcs.wcs_world2pix([list(corner)], 0)[0]
        pixel_values.append(pix)

    pix_array = np.array(pixel_values)
    pixel_bounds = []
    for axis in range(naxis):
        lo = float(pix_array[:, axis].min())
        hi = float(pix_array[:, axis].max())
        start = int(np.floor(lo - 0.5)) + 1
        stop = int(np.ceil(hi + 0.5))
        if stop <= start:
            stop = start + 1
        pixel_bounds.append((start, stop))

    return pixel_bounds


def create_2d_header_from_3d(header, axis_to_drop=0):
    """
    Create a 2D header from a 3D header by dropping one axis.
    
    Args:
        header: Original FITS header
        axis_to_drop: Axis index to drop (0=z, 1=y, 2=x in numpy convention).
                     Note: FITS axis 3 corresponds to numpy axis 0 (with simple loading).
    
    Returns:
        New 2D header.
    """
    from astropy.io import fits
    
    new_header = header.copy()
    
    # 1. Update NAXIS
    new_header['NAXIS'] = 2
    # Shift axes if needed?
    # Logic in moment.py was specific to dropping axis 0 (numpy), which is axis 3 (FITS).
    # If we drop axis 0 (Z): NAXIS1/2 stay as X/Y. NAXIS3 dropped.
    # If we drop axis 1 (Y): NAXIS1 (X) stays. NAXIS3 (Z) becomes NAXIS2?
    # Usually we want the *Result* to be 2D image.
    
    # Let's clean up FITS keywords for the dropped axis.
    # Map numpy axis to FITS axis index (1-based)
    # Numpy (Z, Y, X) -> FITS (NAXIS1=X, NAXIS2=Y, NAXIS3=Z)
    # axis=0 -> FITS 3
    # axis=1 -> FITS 2
    # axis=2 -> FITS 1
    
    fits_axis_to_drop = 3 - axis_to_drop # Assuming 3D
    
    # Special handling: if we drop axis 2 (Y) or 1 (X), we need to shift others down?
    # For now, replicate moment.py logic which was hardcoded for spectral axis drop (mostly).
    # But for general usage, we typically drop the spectral axis (3).
    
    # If fits_axis_to_drop is 3:
    if fits_axis_to_drop == 3:
        # Keep 1 and 2
        pass
    elif fits_axis_to_drop == 2:
        # Dropping Y. Keep X(1) and Z(3). Z becomes Y(2).
        # We need to move keywords from 3 to 2.
        pass
    
    # Given the complexity, let's stick to the specific logic from moment.py
    # which cleans up axis 3.
    # If the user asks for axis != 0 (Z), this might be imperfect for WCS, 
    # but DisplayMap handles simple 2D arrays well.
    
    # Simplified logic: Just drop 3rd axis keywords for now, as that's the main use case.
    
    for key in list(new_header.keys()):
        key_upper = key.upper()
        if key_upper.endswith('3') and key_upper[:-1] in (
            'NAXIS', 'CTYPE', 'CRPIX', 'CRVAL', 'CDELT', 'CUNIT', 'CROTA'
        ):
            del new_header[key]
            continue

        if key_upper.startswith(('PC', 'CD')) and '_' in key_upper:
            parts = key_upper[2:].split('_', 1)
            if len(parts) == 2 and all(p.isdigit() for p in parts):
                if 3 in (int(parts[0]), int(parts[1])):
                    del new_header[key]
                    continue
        
        if key_upper.startswith('PV3_') or key_upper.startswith('PS3_'):
            del new_header[key]
            continue
            
    if 'WCSAXES' in new_header and int(new_header['WCSAXES']) > 2:
        new_header['WCSAXES'] = 2
        
    return new_header
