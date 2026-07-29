"""Cutout usecases."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import math
import numpy as np
from astropy.wcs import WCS

from takefits.core.app_state import AppState, RegionSpec
from takefits.logic.data_tools import (
    LazyScaledArray,
    ensure_operation_memory_budget,
    is_lazy_scaled,
    sanitize_slice,
)
from .utils import world_bounds_to_pixel_bounds, update_datamin_datamax_if_present


_CUTOUT_LAZY_TILE_BYTES = 16 * 1024 * 1024


def _cutout_tile_shape(height: int, width: int) -> Tuple[int, int]:
    """Return a 2-D tile bounded by the lazy float64 materialization target."""
    target_pixels = max(
        1, _CUTOUT_LAZY_TILE_BYTES // np.dtype(np.float64).itemsize
    )
    width_for_rows = min(max(1, int(width)), target_pixels)
    tile_rows = min(
        max(1, int(height)),
        max(1, target_pixels // width_for_rows),
    )
    tile_cols = min(
        max(1, int(width)),
        max(1, target_pixels // tile_rows),
    )
    return tile_rows, tile_cols


def _materialize_lazy_cutout(selected, shape: Tuple[int, ...]) -> np.ndarray:
    """Scale only bounded 2-D tiles from a selected LazyScaledArray view."""
    result = np.empty(shape, dtype=np.float32)
    if len(shape) == 0:
        result[...] = sanitize_slice(np.asarray(selected))[()]
        return result

    if len(shape) == 1:
        target_items = max(
            1, _CUTOUT_LAZY_TILE_BYTES // np.dtype(np.float64).itemsize
        )
        for start in range(0, shape[0], target_items):
            stop = min(start + target_items, shape[0])
            result[start:stop] = sanitize_slice(selected[start:stop])
        return result

    tile_rows, tile_cols = _cutout_tile_shape(shape[-2], shape[-1])
    leading_indices = np.ndindex(shape[:-2])
    for leading_index in leading_indices:
        for y_start in range(0, shape[-2], tile_rows):
            y_stop = min(y_start + tile_rows, shape[-2])
            for x_start in range(0, shape[-1], tile_cols):
                x_stop = min(x_start + tile_cols, shape[-1])
                tile_key = leading_index + (
                    slice(y_start, y_stop),
                    slice(x_start, x_stop),
                )
                result[tile_key] = sanitize_slice(selected[tile_key])
    return result


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
    region: Optional[Union[RegionSpec, dict]] = None,
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

    if region is not None and not isinstance(region, RegionSpec):
        region = RegionSpec.from_dict(region)

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

    cutout_shape = tuple(
        int(
            (data.shape[axis] if data_slice.stop is None else data_slice.stop)
            - (0 if data_slice.start is None else data_slice.start)
        )
        for axis, data_slice in enumerate(data_slices)
    )
    output_pixels = math.prod(cutout_shape)
    output_bytes = output_pixels * np.dtype(np.float32).itemsize
    lazy_tile_pixels = min(
        output_pixels,
        _CUTOUT_LAZY_TILE_BYTES // np.dtype(np.float64).itemsize,
    )
    lazy_tile_scratch = (
        lazy_tile_pixels
        * (np.dtype(np.float64).itemsize + np.dtype(np.float32).itemsize)
        if is_lazy_scaled(data)
        else 0
    )
    region_scratch = 0
    if region is not None and len(cutout_shape) >= 2:
        spatial_pixels = cutout_shape[-2] * cutout_shape[-1]
        region_scratch = spatial_pixels * (
            4 * np.dtype(np.float64).itemsize
            + 2 * np.dtype(np.bool_).itemsize
        )

    ensure_operation_memory_budget(
        output_bytes + lazy_tile_scratch + region_scratch,
        operation_name="Cutout",
        guidance=(
            "Select smaller pixel/world bounds or cut fewer axes at once."
        ),
    )

    # Extract only the selected range.  A lazy scaled cube is filled from
    # bounded 2-D tiles so neither the source cube nor the selected sub-cube is
    # first expanded into a full float64 temporary.
    if is_lazy_scaled(data):
        # Index the raw memmap first and keep scaling lazy.  Calling
        # LazyScaledArray.__getitem__ with a 2-D result would otherwise scale
        # the complete selected image before the tile loop can bound it.
        selected_data = LazyScaledArray(
            data._raw[tuple(data_slices)],
            data._bzero,
            data._bscale,
            data._blank,
        )
        cutout_data = _materialize_lazy_cutout(selected_data, cutout_shape)
    else:
        selected_data = data[tuple(data_slices)]
        cutout_data = np.asarray(selected_data).astype(np.float32, copy=True)

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
        cutout_data = _apply_region_mask(
            cutout_data,
            region,
            actual_bounds,
            copy_data=False,
        )

    return CutoutResult(
        data=cutout_data,
        header=new_header,
        wcs=new_wcs,
        pixel_bounds=actual_bounds
    )


def _apply_region_mask(
    data: np.ndarray,
    region: RegionSpec,
    pixel_bounds: List[Tuple[int, int]],
    *,
    copy_data: bool = True,
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

    # Create broadcast coordinate vectors for the spatial mask.
    y_grid = np.arange(y_start, y_stop, dtype=np.float64)[:, np.newaxis]
    x_grid = np.arange(x_start, x_stop, dtype=np.float64)[np.newaxis, :]

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

    result = data.copy() if copy_data else data
    invalid = ~mask
    if result.ndim == 2:
        result[invalid] = np.nan
    else:
        for leading_index in np.ndindex(result.shape[:-2]):
            result[leading_index][invalid] = np.nan

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
    from takefits.core.io.save_fits import atomic_write_fits

    hdu = fits.PrimaryHDU(
        data=result.data.astype(np.float32, copy=False),
        header=header,
    )
    atomic_write_fits(
        output_path,
        lambda temporary: hdu.writeto(temporary, overwrite=True),
    )

    return output_path
