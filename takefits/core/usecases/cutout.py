"""Cutout usecases."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
from astropy.wcs import WCS

from takefits.core.app_state import AppState, RegionSpec
from .utils import world_bounds_to_pixel_bounds, update_datamin_datamax_if_present


@dataclass
class CutoutResult:
    """Result from cutout operation."""
    data: np.ndarray
    header: any
    wcs: any
    pixel_bounds: List[Tuple[int, int]]  # bounds per WCS axis (start, stop)


def compute_cutout(
    state: AppState,
    pixel_bounds: Optional[List[Tuple[int, int]]] = None,
    world_bounds: Optional[List[Tuple[Union[float, str], Union[float, str]]]] = None,
    region: Optional[RegionSpec] = None,
) -> CutoutResult:
    """
    Extract a cutout from the data cube.

    Args:
        state: AppState with data/header/wcs
        pixel_bounds: Optional pixel bounds per axis [(min,max), ...]
        world_bounds: Optional world bounds per axis [(min,max), ...]
        region: Optional RegionSpec to mask outside region

    Returns:
        CutoutResult with data/header/wcs/bounds
    """
    if state.data is None:
        raise ValueError("No data loaded")

    data = state.data
    header = state.header
    wcs = state.wcs

    if header is None:
        raise ValueError("No header available")

    # Map world bounds to pixel bounds if provided
    if world_bounds is not None:
        if wcs is None:
            raise ValueError("WCS is required for world_bounds")
        pixel_bounds = world_bounds_to_pixel_bounds(state, world_bounds)

    if pixel_bounds is None:
        raise ValueError("Either pixel_bounds or world_bounds must be provided")

    # Ensure bounds list matches WCS axes
    num_wcs_axes = wcs.naxis if wcs is not None else data.ndim
    if len(pixel_bounds) != num_wcs_axes:
        raise ValueError(f"Expected {num_wcs_axes} bounds, got {len(pixel_bounds)}")

    # Build slices for numpy data order (reverse of WCS axis order)
    data_ndim = data.ndim
    data_slices = [slice(None)] * data_ndim
    actual_bounds = []

    for wcs_axis, (start, stop) in enumerate(pixel_bounds):
        # Map WCS axis to data axis (reverse order)
        data_axis = data_ndim - wcs_axis - 1

        if data_axis < 0 or data_axis >= data_ndim:
            actual_bounds.append((start, stop))
            continue

        axis_len = data.shape[data_axis]
        clip_start = max(0, int(start))
        clip_stop = min(axis_len, int(stop))

        if clip_stop <= clip_start:
            raise ValueError(f"Invalid bounds for axis {wcs_axis}: ({start}, {stop})")

        data_slices[data_axis] = slice(clip_start, clip_stop)
        actual_bounds.append((clip_start, clip_stop))

    # Extract cutout data
    cutout_data = data[tuple(data_slices)].astype(np.float32, copy=True)

    # Update header for cutout
    new_header = header.copy()

    for wcs_axis, (clip_start, clip_stop) in enumerate(actual_bounds):
        data_axis = data_ndim - wcs_axis - 1

        # Update CRPIX
        crpix_key = f"CRPIX{wcs_axis + 1}"
        if crpix_key in new_header:
            new_header[crpix_key] = float(new_header[crpix_key]) - clip_start

        # Update NAXIS
        naxis_key = f"NAXIS{wcs_axis + 1}"
        if 0 <= data_axis < cutout_data.ndim:
            new_header[naxis_key] = cutout_data.shape[data_axis]
        else:
            new_header[naxis_key] = clip_stop - clip_start

    # Update NAXIS count
    new_header['NAXIS'] = cutout_data.ndim

    # Update data range (if header already has DATAMIN/DATAMAX)
    update_datamin_datamax_if_present(new_header, cutout_data)

    # Create new WCS
    new_wcs = WCS(new_header) if wcs is not None else None

    # Apply region mask if provided
    if region is not None:
        cutout_data = _apply_region_mask(cutout_data, region, actual_bounds)

    return CutoutResult(
        data=cutout_data,
        header=new_header,
        wcs=new_wcs,
        pixel_bounds=actual_bounds
    )


def _apply_region_mask(
    data: np.ndarray,
    region: RegionSpec,
    pixel_bounds: List[Tuple[int, int]]
) -> np.ndarray:
    """Apply a region mask to cutout data."""
    if data.ndim < 2:
        return data

    # Determine spatial dimensions (last two in numpy order)
    y_dim = data.ndim - 2
    x_dim = data.ndim - 1

    # Get spatial bounds (WCS axis 0=X, 1=Y)
    if len(pixel_bounds) >= 2:
        x_start, x_stop = pixel_bounds[0]
        y_start, y_stop = pixel_bounds[1]
    else:
        x_start, x_stop = 0, data.shape[x_dim]
        y_start, y_stop = 0, data.shape[y_dim]

    # Create grid for spatial mask
    y_grid, x_grid = np.indices(data.shape[-2:])
    x_grid = x_grid + x_start
    y_grid = y_grid + y_start

    # Create region mask
    if region.type == "circle":
        cx, cy = region.center_x, region.center_y
        r = region.params.get('radius', 0)
        mask = (x_grid - cx)**2 + (y_grid - cy)**2 <= r**2
    elif region.type == "rectangle":
        cx, cy = region.center_x, region.center_y
        w = region.params.get('width', 0) / 2
        h = region.params.get('height', 0) / 2
        angle = np.deg2rad(region.params.get('angle', 0.0))

        dx = x_grid - cx
        dy = y_grid - cy

        x_rot = dx * np.cos(-angle) - dy * np.sin(-angle)
        y_rot = dx * np.sin(-angle) + dy * np.cos(-angle)

        mask = (np.abs(x_rot) <= w) & (np.abs(y_rot) <= h)
    elif region.type == "ellipse":
        cx, cy = region.center_x, region.center_y
        w = region.params.get('width', 0) / 2
        h = region.params.get('height', 0) / 2
        angle = np.deg2rad(region.params.get('angle', 0.0))

        dx = x_grid - cx
        dy = y_grid - cy

        x_rot = dx * np.cos(-angle) - dy * np.sin(-angle)
        y_rot = dx * np.sin(-angle) + dy * np.cos(-angle)

        mask = ((x_rot / w)**2 + (y_rot / h)**2 <= 1.0)
    else:
        raise ValueError(f"Unsupported region shape: {region.type}")

    # Expand mask to data dimensions
    while mask.ndim < data.ndim:
        mask = mask[np.newaxis, ...]

    mask = np.broadcast_to(mask, data.shape)
    result = data.copy()
    result[mask] = np.nan

    return result


def export_cutout_fits(
    result: CutoutResult,
    output_path: str,
    source_filename: Optional[str] = None,
    history_entries: Optional[list] = None
) -> str:
    """
    Export cutout result to a FITS file.

    Args:
        result: CutoutResult from compute_cutout
        output_path: Path for output FITS file
        source_filename: Original source filename for HISTORY
        history_entries: Optional list of HISTORY entries

    Returns:
        The output file path
    """
    from astropy.io import fits
    from datetime import datetime

    header = result.header.copy()

    if history_entries:
        # history_entries already contains verbose + provenance lines
        # from build_processing_history_lines(), so skip hand-crafted lines
        # to avoid duplication.
        for entry in history_entries:
            header.add_history(entry)
    else:
        # Standalone call without pre-built history: emit our own lines.
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header.add_history(f"Cutout generated by takefits on {timestamp}")

        if source_filename:
            header.add_history(f"Source file: {source_filename}")

        for i, (start, stop) in enumerate(result.pixel_bounds):
            axis_label = str(header.get(f"CTYPE{i+1}", "") or "").split("-")[0].strip() or f"Axis {i+1}"
            header.add_history(f"{axis_label}: pixels {start} to {stop}")

    update_datamin_datamax_if_present(header, result.data)

    # Write file
    hdu = fits.PrimaryHDU(data=result.data.astype(np.float32), header=header)
    hdu.writeto(output_path, overwrite=True)

    return output_path
