"""PV diagram usecases."""
from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
from scipy.ndimage import map_coordinates

from takefits.core.app_state import AppState
from .utils import get_axis_ctype, parse_world_coordinate, update_datamin_datamax_if_present


def set_pv_endpoints(
    state: AppState,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    width: float = 0.0
) -> AppState:
    """
    Set the PV diagram slice endpoints.

    Args:
        state: The AppState to update
        x0, y0: Start point (pixel coordinates)
        x1, y1: End point (pixel coordinates)
        width: Slice width in pixels (0 = single pixel width)

    Returns:
        The updated AppState
    """
    state.pv_x0 = x0
    state.pv_y0 = y0
    state.pv_x1 = x1
    state.pv_y1 = y1
    state.pv_width = width
    return state


def compute_pv(
    state: AppState,
    x0: Optional[float] = None,
    y0: Optional[float] = None,
    x1: Optional[float] = None,
    y1: Optional[float] = None,
    width: Optional[float] = None,
    num_samples: Optional[int] = None,
    weight_mode: int = 0,
    start_world: Optional[List[Union[float, str]]] = None,
    end_world: Optional[List[Union[float, str]]] = None
) -> np.ndarray:
    """
    Extract a position-velocity diagram from the data cube.

    This is a headless implementation of the PV extraction from
    tools/pv_diagram.py.

    Args:
        state: AppState with loaded data
        x0, y0, x1, y1: Slice endpoints in pixel coordinates.
            If not provided, uses values from state.pv_x0/y0/x1/y1
            (Unless start_world/end_world are provided)
        width: Slice width in pixels (0 = single pixel width)
        num_samples: Number of samples along the slice (default: line length)
        weight_mode: Interpolation mode (0=bilinear/average, 1=gaussian weighted)
        start_world: Start point in world coordinates [x_world, y_world].
                     Overrides x0, y0.
        end_world: End point in world coordinates [x_world, y_world].
                   Overrides x1, y1.

    Returns:
        2D numpy array with shape (n_channels, num_samples)
    """
    if state.data is None:
        raise ValueError("No data loaded")

    # Handle world coordinates if provided
    if start_world is not None:
        if state.wcs is None:
            raise ValueError("WCS required for start_world")
        if len(start_world) < 2:
            raise ValueError("start_world must have at least 2 coordinates (x, y)")

        # Parse strings
        w0 = []
        for i, val in enumerate(start_world):
            # Map input indices to WCS axes 0, 1
            if i >= state.wcs.naxis:
                break
            ctype = get_axis_ctype(state, i)
            w0.append(parse_world_coordinate(val, ctype))

        # Pad with 0s to match naxis
        while len(w0) < state.wcs.naxis:
            w0.append(0.0)

        pix0 = state.wcs.wcs_world2pix([w0], 0)[0]
        x0 = float(pix0[0])
        y0 = float(pix0[1])

    if end_world is not None:
        if state.wcs is None:
            raise ValueError("WCS required for end_world")
        if len(end_world) < 2:
            raise ValueError("end_world must have at least 2 coordinates (x, y)")

        w1 = []
        for i, val in enumerate(end_world):
            if i >= state.wcs.naxis:
                break
            ctype = get_axis_ctype(state, i)
            w1.append(parse_world_coordinate(val, ctype))

        while len(w1) < state.wcs.naxis:
            w1.append(0.0)

        pix1 = state.wcs.wcs_world2pix([w1], 0)[0]
        x1 = float(pix1[0])
        y1 = float(pix1[1])

    # Use state values if not provided
    if x0 is None:
        x0 = state.pv_x0
    if y0 is None:
        y0 = state.pv_y0
    if x1 is None:
        x1 = state.pv_x1
    if y1 is None:
        y1 = state.pv_y1
    if width is None:
        width = state.pv_width

    if any(v is None for v in [x0, y0, x1, y1]):
        raise ValueError("PV slice endpoints not specified")

    data = state.data

    # Handle 4D data by selecting current S slice
    if data.ndim == 4:
        data = data[state.current_s]

    if data.ndim != 3:
        raise ValueError(f"Expected 3D data cube, got {data.ndim}D")

    n_vel = data.shape[0]

    # Calculate line length
    line_length_px = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

    if num_samples is None:
        num_samples = max(1, int(round(line_length_px)))

    if line_length_px == 0:
        num_samples = 1
        pv = np.full((n_vel, num_samples), np.nan)
    else:
        # Sample points along the line
        xs = np.linspace(x0, x1, num_samples)
        ys = np.linspace(y0, y1, num_samples)

        # Helper for direction vectors
        dx = x1 - x0
        dy = y1 - y0
        length = np.sqrt(dx ** 2 + dy ** 2)
        if length > 0:
            perp_x = -dy / length
            perp_y = dx / length
        else:
            perp_x, perp_y = 0, 1

        pv = np.zeros((n_vel, num_samples), dtype=np.float64)

        if width <= 0:
            # Simple interpolation along the line (no width)
            for v in range(n_vel):
                pv[v, :] = map_coordinates(
                    data[v], [ys, xs],
                    order=1, mode='constant', cval=np.nan
                )

        elif weight_mode == 0:
            # Bilinear interpolation averaged over width
            n_width = max(1, int(round(width)))
            offsets = np.linspace(-width / 2, width / 2, n_width)

            for v in range(n_vel):
                values = np.full((n_width, num_samples), np.nan, dtype=np.float64)
                for i, offset in enumerate(offsets):
                    # Use perp vectors directly like in original code
                    # Original: off_x = -np.sin(theta) * off
                    # theta = arctan2(dy, dx), so cos(theta)=dx/len, sin(theta)=dy/len
                    # off_x = - (dy/len) * off = perp_x * off
                    xs_offset = xs + offset * perp_x
                    ys_offset = ys + offset * perp_y
                    values[i, :] = map_coordinates(
                        data[v], [ys_offset, xs_offset],
                        order=1, mode='constant', cval=np.nan
                    )
                # Avoid RuntimeWarning from np.nanmean on all-NaN columns.
                valid = ~np.isnan(values)
                count = valid.sum(axis=0)
                sums = np.nansum(values, axis=0)
                row = np.full(num_samples, np.nan, dtype=np.float64)
                valid_cols = count > 0
                row[valid_cols] = sums[valid_cols] / count[valid_cols]
                pv[v, :] = row

        elif weight_mode == 1:
            # Gaussian weighted interpolation
            sigma = width / (2.0 * np.sqrt(2.0 * np.log(2)))
            n_offsets = int(np.ceil(width * 2)) + 1
            offsets = np.linspace(-width / 2, width / 2, n_offsets)
            weights = np.exp(-0.5 * (offsets / sigma) ** 2)
            weights /= weights.sum()

            for v in range(n_vel):
                prof_sum = np.zeros(num_samples)
                weight_sum = np.zeros(num_samples)

                for offset, w in zip(offsets, weights):
                    xs_offset = xs + offset * perp_x
                    ys_offset = ys + offset * perp_y
                    prof = map_coordinates(
                        data[v], [ys_offset, xs_offset],
                        order=1, mode='constant', cval=np.nan
                    )

                    valid = ~np.isnan(prof)
                    prof_sum[valid] += w * prof[valid]
                    weight_sum[valid] += w

                result = np.full(num_samples, np.nan)
                valid_mask = weight_sum > 0
                result[valid_mask] = prof_sum[valid_mask] / weight_sum[valid_mask]
                pv[v, :] = result

    return pv


def export_pv_fits(
    state: AppState,
    pv_data: np.ndarray,
    output_path: str,
    x0: float, y0: float, x1: float, y1: float,
    is_swapped: bool = False,
    history_entries: Optional[list] = None
) -> str:
    """
    Export PV diagram to a FITS file.

    Args:
        state: AppState with original header/WCS info
        pv_data: 2D PV data array
        output_path: Path for output FITS file
        x0, y0, x1, y1: Slice endpoints in pixel coordinates
        is_swapped: Whether axes are swapped (True: X=Vel, Y=Pos)
        history_entries: Optional list of strings to add as HISTORY keywords

    Returns:
        The output file path
    """
    from astropy.io import fits
    from astropy.wcs.utils import proj_plane_pixel_scales
    import re

    # --- 1. Prepare Data Array ---
    data_to_save = pv_data
    if isinstance(data_to_save, np.ma.MaskedArray):
        data_to_save = data_to_save.filled(np.nan)

    # --- 2. Create FITS Header and WCS ---
    header = fits.Header()
    header['NAXIS'] = 2
    header['NAXIS1'] = data_to_save.shape[1]  # X-axis length
    header['NAXIS2'] = data_to_save.shape[0]  # Y-axis length
    if state.header:
        header['BUNIT'] = state.header.get('BUNIT', 'UNKNOWN')

    # Add history
    if history_entries:
        for entry in history_entries:
            header.add_history(entry)

    # --- Define Position Axis (Offset in degrees) ---
    line_length_px = np.hypot(x1 - x0, y1 - y0)

    # Calculate pixel scale in degrees
    pixel_scale_deg = None
    if state.wcs:
        try:
            if state.wcs.is_celestial:
                scales = proj_plane_pixel_scales(state.wcs)
                pixel_scale_deg = (abs(scales[0]) + abs(scales[1])) / 2.0

            if pixel_scale_deg is None and state.wcs.wcs.cdelt is not None:
                cdelt = state.wcs.wcs.cdelt
                # Use first two axes
                if len(cdelt) >= 2:
                    cdelt1 = abs(cdelt[0])
                    cdelt2 = abs(cdelt[1])
                    unit_str = str(state.wcs.wcs.cunit[0]).lower() if state.wcs.wcs.cunit else ''
                    if unit_str in ('deg', 'degree', 'degrees'):
                        pixel_scale_deg = (cdelt1 + cdelt2) / 2.0
        except Exception:
            pass

    length_deg = line_length_px
    if pixel_scale_deg is not None:
        length_deg = line_length_px * pixel_scale_deg

    if is_swapped:
        num_pos_pixels = data_to_save.shape[0]  # Position is NAXIS2
    else:
        num_pos_pixels = data_to_save.shape[1]  # Position is NAXIS1

    if num_pos_pixels > 1:
        pos_cdelt = length_deg / num_pos_pixels
    else:
        pos_cdelt = length_deg

    pos_crval = 0.0
    pos_crpix = 0.5
    pos_ctype = 'OFFSET'
    pos_cunit = 'deg'

    # --- Define Velocity Axis ---
    vel_axis_index = 2  # Default to 3rd axis (0-indexed 2)
    vel_ctype = 'VELOCITY'
    vel_cunit = 'km/s'
    vel_crval = 0.0
    vel_cdelt = 1.0
    vel_crpix = 1.0

    if state.wcs and state.wcs.wcs.naxis >= 3:
        # Try to find spectral axis
        spec_axis = -1
        for i, ctype in enumerate(state.wcs.wcs.ctype):
            if any(x in ctype.upper() for x in ['VEL', 'FREQ', 'VRAD', 'VOPT']):
                spec_axis = i
                break

        if spec_axis != -1:
            vel_axis_index = spec_axis
            vel_ctype = state.wcs.wcs.ctype[vel_axis_index]
            vel_crval = state.wcs.wcs.crval[vel_axis_index]
            vel_cdelt = state.wcs.wcs.cdelt[vel_axis_index]
            vel_crpix = state.wcs.wcs.crpix[vel_axis_index]
            if state.wcs.wcs.cunit:
                vel_cunit = state.wcs.wcs.cunit[vel_axis_index].to_string()

    # Attempt to use pretty unit from spectral_metadata if available
    # (This handles the case where fits_loader converted units but WCS might be raw)
    if state.spectral_metadata and 'current_axis_unit' in state.spectral_metadata:
        # If we have a formatted unit string like "Velocity [km/s]" or just "km/s"
        meta_unit = state.spectral_metadata['current_axis_unit']
        # Try to extract content inside brackets if present
        match = re.search(r'\[(.*?)\]', meta_unit)
        if match:
            vel_cunit = match.group(1).strip()
        elif meta_unit.strip():
            vel_cunit = meta_unit.strip()

    # --- Assign WCS keywords based on swap state ---
    if is_swapped:
        # NAXIS1 = Velocity, NAXIS2 = Position
        header['CTYPE1'] = vel_ctype
        header['CUNIT1'] = vel_cunit
        header['CRVAL1'] = vel_crval
        header['CDELT1'] = vel_cdelt
        header['CRPIX1'] = vel_crpix

        header['CTYPE2'] = pos_ctype
        header['CUNIT2'] = pos_cunit
        header['CRVAL2'] = pos_crval
        header['CDELT2'] = pos_cdelt
        header['CRPIX2'] = pos_crpix
    else:
        # NAXIS1 = Position, NAXIS2 = Velocity
        header['CTYPE1'] = pos_ctype
        header['CUNIT1'] = pos_cunit
        header['CRVAL1'] = pos_crval
        header['CDELT1'] = pos_cdelt
        header['CRPIX1'] = pos_crpix

        header['CTYPE2'] = vel_ctype
        header['CUNIT2'] = vel_cunit
        header['CRVAL2'] = vel_crval
        header['CDELT2'] = vel_cdelt
        header['CRPIX2'] = vel_crpix

    update_datamin_datamax_if_present(header, data_to_save)

    # Write file
    hdu = fits.PrimaryHDU(data=data_to_save.astype(np.float32), header=header)
    hdu.writeto(output_path, overwrite=True)

    return output_path
