"""Smoothing helpers for usecases."""
from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np

from takefits.core.app_state import AppState


SmoothingKernel = Literal["gaussian", "boxcar", "hanning"]


def compute_smoothed(
    data: np.ndarray,
    kernel_type: str = "gaussian",
    smoothness_x: float = 1.0,
    smoothness_y: float = 1.0,
    smoothness_z: float = 0.0,
    handle_nan: bool = True
) -> np.ndarray:
    """
    Compute smoothed data array (pure function).

    Args:
        data: Input data array (2D/3D/4D)
        kernel_type: "gaussian", "boxcar", or "hanning" (velocity-axis only)
        smoothness_x: Kernel width in pixels (sigma for gaussian, half-width for boxcar)
        smoothness_y: Kernel width in pixels
        smoothness_z: Kernel width in pixels (for 3D data)
        handle_nan: If True, preserve NaNs and interpolate neighbors via weighted FFT

    Returns:
        Smoothed data array (same shape as input)
    """
    from scipy.signal import fftconvolve
    from scipy.ndimage import convolve1d
    from astropy.convolution import Gaussian1DKernel, Gaussian2DKernel, CustomKernel

    if kernel_type != "hanning" and smoothness_x == 0 and smoothness_y == 0 and smoothness_z == 0:
        return data.copy()

    ndim = data.ndim

    has_nan = np.isnan(data).any()
    use_nan_aware = handle_nan and bool(has_nan)

    if kernel_type == "hanning":
        if data.ndim < 3:
            raise ValueError("Hanning smoothing is available only for 3D/4D cubes")

        # FITS cubes are expected as (z, y, x) or (s, z, y, x); smooth along z.
        spectral_axis = data.ndim - 3
        hanning_weights = np.array([0.25, 0.5, 0.25], dtype=np.float64)

        if use_nan_aware:
            valid_mask = np.isfinite(data)
            if not np.any(valid_mask):
                return np.full(data.shape, np.nan, dtype=np.float64)
            data_filled = np.where(valid_mask, data, 0.0)
            numer = convolve1d(data_filled, hanning_weights, axis=spectral_axis, mode="reflect")
            denom = convolve1d(valid_mask.astype(np.float32), hanning_weights, axis=spectral_axis, mode="reflect")
            out = np.full(data.shape, np.nan, dtype=np.float64)
            np.divide(numer, denom, out=out, where=denom > 1e-12)
            out[~valid_mask] = np.nan
            return out

        data_clean = np.nan_to_num(data, nan=0.0) if has_nan else np.asarray(data)
        return convolve1d(data_clean, hanning_weights, axis=spectral_axis, mode="reflect")

    # Build kernel for spatial smoothing
    if kernel_type == "gaussian":
        if ndim == 2 or smoothness_z == 0:
            # 2D Gaussian kernel
            if smoothness_x > 0 and smoothness_y > 0:
                kernel = Gaussian2DKernel(x_stddev=smoothness_x, y_stddev=smoothness_y)
            elif smoothness_x > 0:
                kx = Gaussian1DKernel(smoothness_x).array
                kernel_array = np.outer(np.array([1]), kx)
                kernel_array /= kernel_array.sum()
                kernel = CustomKernel(kernel_array)
            elif smoothness_y > 0:
                ky = Gaussian1DKernel(smoothness_y).array
                kernel_array = np.outer(ky, np.array([1]))
                kernel_array /= kernel_array.sum()
                kernel = CustomKernel(kernel_array)
            else:
                return data.copy()
        else:
            # 3D Gaussian kernel
            kx = Gaussian1DKernel(smoothness_x).array if smoothness_x > 0 else np.array([1.0])
            ky = Gaussian1DKernel(smoothness_y).array if smoothness_y > 0 else np.array([1.0])
            kz = Gaussian1DKernel(smoothness_z).array if smoothness_z > 0 else np.array([1.0])

            kernel_array = kz[:, None, None] * ky[None, :, None] * kx[None, None, :]
            kernel_array /= kernel_array.sum()
            kernel = CustomKernel(kernel_array)

    elif kernel_type == "boxcar":
        size_x = int(2 * smoothness_x + 1) if smoothness_x > 0 else 1
        size_y = int(2 * smoothness_y + 1) if smoothness_y > 0 else 1
        size_z = int(2 * smoothness_z + 1) if smoothness_z > 0 else 1

        # Ensure odd sizes
        if size_x % 2 == 0:
            size_x += 1
        if size_y % 2 == 0:
            size_y += 1
        if size_z % 2 == 0:
            size_z += 1

        if ndim == 2:
            kernel_array = np.ones((size_y, size_x))
        else:
            kernel_array = np.ones((size_z, size_y, size_x))

        kernel_array /= kernel_array.sum()
        kernel = CustomKernel(kernel_array)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    # Calculate padding and kernel dimensions for FFT convolution
    kernel_shape = kernel.shape
    if data.ndim != len(kernel_shape):
        pad_width = [(0, 0)] * (data.ndim - len(kernel_shape)) + [(s // 2, s // 2) for s in kernel_shape]
    else:
        pad_width = [(s // 2, s // 2) for s in kernel_shape]

    kernel_nd = kernel.array
    while kernel_nd.ndim < data.ndim:
        kernel_nd = kernel_nd[None, ...]

    slices = tuple(slice(p[0], -p[1] if p[1] != 0 else None) for p in pad_width)

    if use_nan_aware:
        # Fast NaN-aware smoothing via weighted convolution:
        # smoothed = conv(data * valid) / conv(valid)
        valid_mask = np.isfinite(data)
        if not np.any(valid_mask):
            return np.full(data.shape, np.nan, dtype=np.float64)

        data_filled = np.where(valid_mask, data, 0.0)
        data_padded = np.pad(data_filled, pad_width=pad_width, mode='reflect')
        valid_padded = np.pad(valid_mask.astype(np.float32), pad_width=pad_width, mode='reflect')

        smoothed_padded = fftconvolve(data_padded, kernel_nd, mode='same')
        weight_padded = fftconvolve(valid_padded, kernel_nd, mode='same')
        smoothed = smoothed_padded[slices]
        weights = weight_padded[slices]

        out = np.full(data.shape, np.nan, dtype=np.float64)
        np.divide(smoothed, weights, out=out, where=weights > 1e-12)
        out[~valid_mask] = np.nan
        return out

    # Fast convolution without NaN interpolation
    data_clean = np.nan_to_num(data, nan=0.0) if has_nan else np.asarray(data)
    data_padded = np.pad(data_clean, pad_width=pad_width, mode='reflect')
    smoothed_padded = fftconvolve(data_padded, kernel_nd, mode='same')
    return smoothed_padded[slices]


def compute_smoothed_to_resolution(
    data: np.ndarray,
    header: Any,
    target_bmaj: float,
    target_bmin: float,
    target_bpa: float = 0.0,
    current_bmaj: Optional[float] = None,
    current_bmin: Optional[float] = None,
    current_bpa: Optional[float] = None
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Smooth data to a target resolution.

    Args:
        data: Input data array (2D or 3D)
        header: FITS header (for CDELT and current beam if not specified)
        target_bmaj: Target major axis FWHM in arcsec
        target_bmin: Target minor axis FWHM in arcsec
        target_bpa: Target position angle in degrees
        current_bmaj: Current major axis FWHM in arcsec (reads from header if None)
        current_bmin: Current minor axis FWHM in arcsec (reads from header if None)
        current_bpa: Current position angle in degrees (reads from header if None)

    Returns:
        (smoothed_data, new_beam_params) where new_beam_params has keys
        'BMAJ', 'BMIN', 'BPA' in degrees.

    Raises:
        ValueError: If target resolution is smaller than current resolution
    """
    from astropy.convolution import Gaussian2DKernel
    from scipy.signal import fftconvolve

    # Get current beam from header if not provided
    if current_bmaj is None:
        current_bmaj = header.get('BMAJ', 0) * 3600  # Convert deg to arcsec
    if current_bmin is None:
        current_bmin = header.get('BMIN', 0) * 3600
    if current_bpa is None:
        current_bpa = header.get('BPA', 0)

    # Ensure major >= minor
    if target_bmaj < target_bmin:
        target_bmaj, target_bmin = target_bmin, target_bmaj

    # Get pixel scale
    pixel_scale_x = abs(header['CDELT1']) * 3600  # arcsec/pixel
    pixel_scale_y = abs(header['CDELT2']) * 3600

    # Calculate convolving kernel parameters
    FWHM_to_sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2)))

    sigma_current_maj = current_bmaj * FWHM_to_sigma
    sigma_current_min = current_bmin * FWHM_to_sigma
    sigma_target_maj = target_bmaj * FWHM_to_sigma
    sigma_target_min = target_bmin * FWHM_to_sigma

    sigma_kernel_maj_sq = sigma_target_maj ** 2 - sigma_current_maj ** 2
    sigma_kernel_min_sq = sigma_target_min ** 2 - sigma_current_min ** 2

    if sigma_kernel_maj_sq <= 0 or sigma_kernel_min_sq <= 0:
        raise ValueError("Target resolution must be larger than current resolution")

    sigma_kernel_maj = np.sqrt(sigma_kernel_maj_sq)
    sigma_kernel_min = np.sqrt(sigma_kernel_min_sq)

    # Convert to pixels
    sigma_kernel_x = sigma_kernel_min / pixel_scale_x
    sigma_kernel_y = sigma_kernel_maj / pixel_scale_y
    theta = np.deg2rad(target_bpa - current_bpa)

    # Create kernel
    kernel = Gaussian2DKernel(x_stddev=sigma_kernel_x, y_stddev=sigma_kernel_y, theta=theta)

    kernel_array = kernel.array
    pad_width = [(s // 2, s // 2) for s in kernel_array.shape]
    slices = tuple(slice(p[0], -p[1] if p[1] != 0 else None) for p in pad_width)

    def _convolve_2d_preserve_nan(slice_data: np.ndarray) -> np.ndarray:
        valid_mask = np.isfinite(slice_data)
        if not np.any(valid_mask):
            return np.full(slice_data.shape, np.nan, dtype=np.float64)

        slice_filled = np.where(valid_mask, slice_data, 0.0)
        data_padded = np.pad(slice_filled, pad_width=pad_width, mode='reflect')
        valid_padded = np.pad(valid_mask.astype(np.float32), pad_width=pad_width, mode='reflect')

        smoothed_padded = fftconvolve(data_padded, kernel_array, mode='same')
        weight_padded = fftconvolve(valid_padded, kernel_array, mode='same')
        smoothed = smoothed_padded[slices]
        weights = weight_padded[slices]

        out = np.full(slice_data.shape, np.nan, dtype=np.float64)
        np.divide(smoothed, weights, out=out, where=weights > 1e-12)
        out[~valid_mask] = np.nan
        return out

    # Perform convolution
    if data.ndim == 2:
        smoothed = _convolve_2d_preserve_nan(np.asarray(data))
    elif data.ndim == 3:
        smoothed = np.empty(data.shape, dtype=np.float64)
        for i in range(data.shape[0]):
            smoothed[i] = _convolve_2d_preserve_nan(data[i])
    elif data.ndim == 4:
        smoothed = np.empty(data.shape, dtype=np.float64)
        for s in range(data.shape[0]):
            for z in range(data.shape[1]):
                smoothed[s, z] = _convolve_2d_preserve_nan(data[s, z])
    else:
        raise ValueError(f"Unsupported data dimensionality: {data.ndim}")

    # New beam parameters (in degrees for FITS header)
    new_beam = {
        'BMAJ': target_bmaj / 3600,
        'BMIN': target_bmin / 3600,
        'BPA': target_bpa
    }

    return smoothed, new_beam


def apply_smoothing(
    state: AppState,
    kernel_type: str = "gaussian",
    smoothness_x: float = 1.0,
    smoothness_y: float = 1.0,
    smoothness_z: float = 0.0,
    handle_nan: bool = True
) -> AppState:
    """
    Apply smoothing to state.data in-place.

    Args:
        state: The AppState to update
        kernel_type: "gaussian", "boxcar", or "hanning"
        smoothness_x: Kernel width in pixels
        smoothness_y: Kernel width in pixels
        smoothness_z: Kernel width in pixels (for 3D data)
        handle_nan: If True, preserve NaNs and interpolate neighbors via weighted FFT

    Returns:
        The updated AppState
    """
    if state.data is None:
        raise ValueError("No data loaded")

    state.data = compute_smoothed(
        state.data,
        kernel_type=kernel_type,
        smoothness_x=smoothness_x,
        smoothness_y=smoothness_y,
        smoothness_z=smoothness_z,
        handle_nan=handle_nan
    )

    return state


def apply_smoothing_to_resolution(
    state: AppState,
    target_bmaj: float,
    target_bmin: float,
    target_bpa: float = 0.0,
    current_bmaj: Optional[float] = None,
    current_bmin: Optional[float] = None,
    current_bpa: Optional[float] = None,
) -> AppState:
    """
    Apply Gaussian smoothing to a target beam resolution and update header keys.
    """
    if state.data is None:
        raise ValueError("No data loaded")
    if state.header is None:
        raise ValueError("No FITS header loaded")

    smoothed_data, new_beam = compute_smoothed_to_resolution(
        state.data,
        state.header,
        target_bmaj=target_bmaj,
        target_bmin=target_bmin,
        target_bpa=target_bpa,
        current_bmaj=current_bmaj,
        current_bmin=current_bmin,
        current_bpa=current_bpa,
    )
    state.data = smoothed_data
    state.header["BMAJ"] = float(new_beam["BMAJ"])
    state.header["BMIN"] = float(new_beam["BMIN"])
    state.header["BPA"] = float(new_beam["BPA"])
    return state
