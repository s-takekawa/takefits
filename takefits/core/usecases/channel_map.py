"""Channel map usecases."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import math
import numpy as np

from takefits.core.app_state import AppState
from .utils import axis_world_to_pixel


@dataclass
class ChannelMapResult:
    """Result from channel map generation."""
    images: List[np.ndarray]  # List of 2D images
    labels: List[Tuple[float, float, float]]  # (from, center, to) for each image
    display_labels: List[Tuple[float, float, float]]  # Requested bin edges and midpoint
    pixel_ranges: List[Tuple[int, int]]  # (start, end) pixel for each image


def compute_channel_map(
    state: AppState,
    start_channel: Optional[float] = None,
    end_channel: Optional[float] = None,
    interval: float = 1.0,
    mode: str = "average",
    axis: int = 0,
    start_world: Optional[Union[float, str]] = None,
    end_world: Optional[Union[float, str]] = None,
    interval_world: Optional[float] = None
) -> ChannelMapResult:
    """
    Generate channel map images from a data cube.

    Args:
        state: AppState with data
        start_channel: Starting channel (0-indexed). Required unless start_world specified.
        end_channel: Ending channel (exclusive). Required unless end_world specified.
        interval: Number of channels per map panel (default: 1.0). Can be fractional.
        mode: "slice" (single channel), "average", or "integrate" (sum)
        axis: Axis along which to extract channels (0=z for XY plane in numpy indexing)
        start_world: Starting position in world coordinates. Requires WCS.
        end_world: Ending position in world coordinates. Requires WCS.
        interval_world: Interval in world coordinates (e.g., km/s). If provided,
                       overrides interval (pixel-based).

    Returns:
        ChannelMapResult with images and labels
    """
    if state.data is None:
        raise ValueError("No data loaded")

    data = state.data
    if data.ndim == 4:
        data = data[state.current_s]

    if data.ndim != 3:
        raise ValueError(f"Expected 3D data cube, got {data.ndim}D")

    n_channels = data.shape[axis]
    wcs = state.wcs

    # Convert world coordinates to pixel if provided
    if start_world is not None or end_world is not None:
        if wcs is None:
            raise ValueError("WCS is required for world coordinate specification")

        # WCS axis index (numpy axis 0 = WCS axis 2 for 3D)
        wcs_axis = data.ndim - 1 - axis

        if start_world is not None:
            # Use float for precision
            start_channel = axis_world_to_pixel(state, start_world, wcs_axis)
        if end_world is not None:
            end_channel = axis_world_to_pixel(state, end_world, wcs_axis)

        # Handle inverted axis (world coords may be decreasing with pixel)
        if start_channel is not None and end_channel is not None:
            if start_channel > end_channel:
                start_channel, end_channel = end_channel, start_channel

        # Convert interval from world to pixel if specified
        if interval_world is not None:
            cdelt = abs(wcs.wcs.cdelt[wcs_axis])
            interval = abs(interval_world) / cdelt

    # Ensure we have valid logical channel range
    if start_channel is None:
        start_channel = 0.0
    if end_channel is None:
        end_channel = float(n_channels)

    # Validate inputs
    start_channel = float(start_channel)
    end_channel = float(end_channel)

    if start_channel >= end_channel:
        # Allow tiny tolerance? No, just raise if empty.
        raise ValueError(f"Invalid channel range: {start_channel} to {end_channel}")

    if interval <= 0:
        raise ValueError("Interval must be positive")

    images = []
    labels = []
    display_labels = []
    pixel_ranges = []

    current = start_channel

    # Pre-calculate slice for other axes
    base_slices = [slice(None)] * data.ndim

    while current < end_channel:
        # Define fractional range for this slab
        ch_end = min(current + interval, end_channel)

        # Avoid creating tiny slivers at the end due to float precision
        if ch_end - current < 1e-6:
            break

        if mode == "slice":
            # For slice mode, we pick the nearest integer channel to the START of the current step
            # This matches original behavior: idx = floor(slice_pix - 0.5) where slice_pix = current + 1.0
            idx = int(math.floor(current + 0.5))
            idx = max(0, min(n_channels - 1, idx))

            slices = list(base_slices)
            slices[axis] = slice(idx, idx + 1)
            img = np.squeeze(data[tuple(slices)], axis=axis)

        else:  # average or integrate
            # Implementing fractional implementation logic
            # Range: [current, ch_end)

            # Identify integer pixels involved
            # If current=2.5, it starts in pixel 2 (center 2.0? or 2.5 is edge?)
            # Standard FITS/Numpy: pixel i covers [i-0.5, i+0.5) ??
            # OR pixel i covers [i, i+1) ??
            # Usually image coordinates are 0-indexed at center of pixel (0,0) is center of first pixel.
            # So pixel 0 covers [-0.5, 0.5].
            # Wait, `get_spectrum` uses `data[y, x]`.
            # Let's assume standard integer indexing logic where index `i` is the value at integer coordinate `i`.
            # Integrating strictly over range [A, B] usually assumes pixels are finite areas.
            # Let's assume pixel i extends from i-0.5 to i+0.5.
            # So if range is [0.0, 1.0], it covers pixel 0 ([-0.5, 0.5]) and pixel 1?
            # NO. "Channel Map" usually implies "Sum of channels i..j".
            # If use asks for channels 0 to 5.
            # It sums indices 0, 1, 2, 3, 4.
            # This corresponds to range [0, 5] in slice notation.
            # Slice(0, 5) includes 0,1,2,3,4.
            # So "integer coordinate" i corresponds to the block [i, i+1) in slice logic?
            # Yes, slice(start, stop) works on indices.
            # So let's stick to SLICE logic: Index i covers domain [i, i+1).

            # Slicing logic:
            # Slab [current, ch_end).
            # Min integer index: floor(current)
            # Max integer index: ceil(ch_end)

            min_idx = int(math.floor(current))
            max_idx = int(math.ceil(ch_end))
            max_idx = min(max_idx, n_channels)

            # Prepare weighted sum
            weighted_sum = None
            total_weight = 0.0

            for i in range(min_idx, max_idx):
                # Bounds check: skip invalid pixel indices
                if i < 0 or i >= n_channels:
                    continue

                # Pixel i covers range [i, i+1)
                # Overlap with [current, ch_end)
                pix_start = float(i)
                pix_end = float(i + 1)

                overlap_start = max(pix_start, current)
                overlap_end = min(pix_end, ch_end)

                weight = max(0.0, overlap_end - overlap_start)

                if weight > 0:
                    slices = list(base_slices)
                    slices[axis] = i  # Select single index to get 2D array
                    val = data[tuple(slices)]

                    # Handle NaNs
                    val_clean = np.nan_to_num(val, nan=0.0)

                    if weighted_sum is None:
                        weighted_sum = val_clean * weight
                    else:
                        weighted_sum += val_clean * weight

                    total_weight += weight

            if weighted_sum is None:
                # Should not happen if range is valid and overlaps at least one valid pixel
                # But if range is entirely outside data, weighted_sum is None.
                # Create zero array of correct shape
                img = np.zeros_like(np.squeeze(data.take([0], axis=axis), axis=axis))
            elif mode == "integrate":
                img = weighted_sum
            else:  # average
                if total_weight > 0:
                    img = weighted_sum / total_weight
                else:
                    img = np.zeros_like(weighted_sum)

        images.append(img)

        if mode == "slice":
            # For slice mode, we want the label to reflect exactly the selected index
            # Re-calculate idx just to be safe/clear (or could have stored it)
            idx = int(math.floor(current + 0.5))
            idx = max(0, min(n_channels - 1, idx))
            labels.append((float(idx), float(idx), float(idx)))
            display_labels.append((float(idx), float(idx), float(idx)))
            pixel_ranges.append((idx, idx))
        else:
            display_center = (current + ch_end) / 2.0
            display_labels.append((float(current), float(display_center), float(ch_end)))
            is_integer_range = (
                abs(current - round(current)) < 1e-6
                and abs(ch_end - round(ch_end)) < 1e-6
            )

            if is_integer_range:
                start_idx = int(round(current))
                end_idx = int(round(ch_end)) - 1
                if start_idx < 0:
                    start_idx = 0
                if start_idx >= n_channels:
                    start_idx = n_channels - 1
                if end_idx < 0:
                    end_idx = 0
                if end_idx >= n_channels:
                    end_idx = n_channels - 1
                if end_idx < start_idx:
                    end_idx = start_idx

                center = (start_idx + end_idx) / 2.0
                labels.append((float(start_idx), float(center), float(end_idx)))
                pixel_ranges.append((start_idx, end_idx))
            else:
                center = (current + ch_end) / 2.0
                labels.append((float(current), float(center), float(ch_end)))
                pixel_ranges.append((current, ch_end))

        current = ch_end

    return ChannelMapResult(
        images=images,
        labels=labels,
        display_labels=display_labels,
        pixel_ranges=pixel_ranges
    )


def channel_labels_to_world(
    state: AppState,
    labels: List[Tuple[float, float, float]],
    axis: int = 0
) -> List[Tuple[str, str, str]]:
    """
    Convert channel labels from pixel to world coordinates.

    Args:
        state: AppState with WCS
        labels: List of (from, center, to) in pixel coordinates
        axis: WCS axis index (0=X, 1=Y, 2=Z)

    Returns:
        List of (from_str, center_str, to_str) in world coordinates
    """
    if state.wcs is None:
        return [(f"{f:.1f}", f"{c:.1f}", f"{t:.1f}") for f, c, t in labels]

    wcs = state.wcs
    world_labels = []

    for from_pix, center_pix, to_pix in labels:
        # Convert each to world coordinates
        coords_from = np.zeros(wcs.naxis)
        coords_center = np.zeros(wcs.naxis)
        coords_to = np.zeros(wcs.naxis)

        coords_from[axis] = from_pix
        coords_center[axis] = center_pix
        coords_to[axis] = to_pix

        try:
            world_from = wcs.wcs_pix2world([coords_from], 0)[0][axis]
            world_center = wcs.wcs_pix2world([coords_center], 0)[0][axis]
            world_to = wcs.wcs_pix2world([coords_to], 0)[0][axis]

            world_labels.append((
                f"{world_from:.4g}",
                f"{world_center:.4g}",
                f"{world_to:.4g}"
            ))
        except Exception:
            world_labels.append((
                f"{from_pix:.1f}",
                f"{center_pix:.1f}",
                f"{to_pix:.1f}"
            ))

    return world_labels
