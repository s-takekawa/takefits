"""Smoothing helpers for usecases."""
from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np

from takefits.core.app_state import AppState


SmoothingKernel = Literal["gaussian", "boxcar", "hanning"]


def _is_per_beam_bunit(bunit: Any) -> bool:
    """Return True for FITS BUNIT strings expressed per synthesized beam."""
    text = "".join(str(bunit or "").lower().split())
    return "/beam" in text or "beam-1" in text or "beam^-1" in text


def _beam_area_ratio_for_bunit(
    header: Any,
    *,
    current_bmaj: float,
    current_bmin: float,
    target_bmaj: float,
    target_bmin: float,
) -> float:
    """Scale Jy/beam-like data from current-beam to target-beam units."""
    if not _is_per_beam_bunit(header.get("BUNIT", "")):
        return 1.0

    if current_bmaj <= 0 or current_bmin <= 0:
        raise ValueError(
            "Current BMAJ and BMIN are required for Jy/beam target-resolution smoothing"
        )
    if target_bmaj <= 0 or target_bmin <= 0:
        raise ValueError("Target BMAJ and BMIN must be positive")

    return (target_bmaj * target_bmin) / (current_bmaj * current_bmin)


def beam_unit_scale_for_target_resolution(
    header: Any,
    *,
    current_bmaj: float,
    current_bmin: float,
    target_bmaj: float,
    target_bmin: float,
) -> float:
    """Return the intensity scale needed when target smoothing changes beam units."""
    return _beam_area_ratio_for_bunit(
        header,
        current_bmaj=current_bmaj,
        current_bmin=current_bmin,
        target_bmaj=target_bmaj,
        target_bmin=target_bmin,
    )


def _centered(arr: np.ndarray, newshape) -> np.ndarray:
    """Crop ``arr`` to ``newshape`` about its centre (matches scipy.signal)."""
    currshape = np.array(arr.shape)
    newshape = np.asarray(newshape)
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    sl = tuple(slice(startind[k], endind[k]) for k in range(len(endind)))
    return arr[sl]


class _ReflectFFTConvolver:
    """Reflect-padded 2D FFT convolution with a fixed kernel.

    The kernel's frequency transform is computed once and reused for every
    plane, and planes free of NaNs skip the weight-normalisation pass.  This is
    numerically equivalent (to FFT round-off) to the previous per-plane
    ``scipy.signal.fftconvolve(..., mode="same")`` calls, but avoids re-running
    the kernel FFT on each plane and never materialises a padded copy of the
    whole cube.
    """

    def __init__(self, kernel_2d: np.ndarray, plane_shape: Tuple[int, int]) -> None:
        from scipy import fft as sp_fft

        self._sp_fft = sp_fft
        kernel_2d = np.asarray(kernel_2d)
        self._pad_width = [(s // 2, s // 2) for s in kernel_2d.shape]
        self._slices = tuple(
            slice(p[0], -p[1] if p[1] != 0 else None) for p in self._pad_width
        )
        self._padded_shape = tuple(
            int(d + p[0] + p[1]) for d, p in zip(plane_shape, self._pad_width)
        )
        self._plane_shape = tuple(int(d) for d in plane_shape)
        self._clean_weight: Optional[np.ndarray] = None
        full_shape = np.array(self._padded_shape) + np.array(kernel_2d.shape) - 1
        self._fshape = [sp_fft.next_fast_len(int(d), True) for d in full_shape]
        self._fslice = tuple(slice(0, int(d)) for d in full_shape)
        self._kernel_fft = sp_fft.rfftn(kernel_2d, self._fshape)

    def _fft_same(self, padded_plane: np.ndarray) -> np.ndarray:
        # Match scipy.signal.fftconvolve: a float32 plane transforms to
        # complex64, but the (out-of-place) product with the complex128 kernel
        # promotes back to double precision before the inverse transform.
        sp_fft = self._sp_fft
        spec = sp_fft.rfftn(padded_plane, self._fshape) * self._kernel_fft
        full = sp_fft.irfftn(spec, self._fshape)[self._fslice]
        return _centered(full, self._padded_shape)

    def _clean_plane_weight(self) -> np.ndarray:
        """conv(valid) for a fully-valid plane, computed once and reused.

        Matches the per-plane weight the previous implementation produced from a
        float32 validity mask, so NaN-free planes reproduce the old weighted
        normalisation without running a weight convolution on every plane.
        """
        if self._clean_weight is None:
            ones = np.ones(self._plane_shape, dtype=np.float32)
            padded = np.pad(ones, self._pad_width, mode="reflect")
            self._clean_weight = self._fft_same(padded)[self._slices].copy()
        return self._clean_weight

    def convolve_plane(self, plane: np.ndarray, *, nan_aware: bool = True) -> np.ndarray:
        plane = np.asarray(plane)

        if not nan_aware:
            padded = np.pad(plane, self._pad_width, mode="reflect")
            return self._fft_same(padded)[self._slices].copy()

        valid_mask = np.isfinite(plane)
        if valid_mask.all():
            padded = np.pad(plane, self._pad_width, mode="reflect")
            smoothed = self._fft_same(padded)[self._slices]
            return smoothed / self._clean_plane_weight()
        if not valid_mask.any():
            return np.full(plane.shape, np.nan, dtype=np.float64)

        filled = np.where(valid_mask, plane, 0.0)
        data_padded = np.pad(filled, self._pad_width, mode="reflect")
        valid_padded = np.pad(valid_mask.astype(np.float32), self._pad_width, mode="reflect")
        smoothed = self._fft_same(data_padded)[self._slices]
        weights = self._fft_same(valid_padded)[self._slices]
        out = np.full(plane.shape, np.nan, dtype=np.float64)
        np.divide(smoothed, weights, out=out, where=weights > 1e-12)
        out[~valid_mask] = np.nan
        return out


def _apply_planewise(
    convolver: "_ReflectFFTConvolver",
    data: np.ndarray,
    *,
    nan_aware: bool,
) -> np.ndarray:
    """Convolve each trailing 2D plane of ``data`` one at a time."""
    if data.ndim == 2:
        return convolver.convolve_plane(data, nan_aware=nan_aware)

    out = np.empty(data.shape, dtype=np.float64)
    for idx in np.ndindex(*data.shape[:-2]):
        out[idx] = convolver.convolve_plane(data[idx], nan_aware=nan_aware)
    return out


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

    kernel_array = kernel.array

    # Planar (2D) kernels convolve each image plane independently, so iterate
    # plane-by-plane: this reuses one kernel FFT, skips weight normalisation on
    # NaN-free planes, and never materialises a padded copy of the whole cube.
    # 3D kernels mix the spectral axis and stay on the whole-array FFT path.
    if kernel_array.ndim == 2 and data.ndim >= 2:
        convolver = _ReflectFFTConvolver(kernel_array, data.shape[-2:])
        if use_nan_aware:
            return _apply_planewise(convolver, data, nan_aware=True)
        data_clean = np.nan_to_num(data, nan=0.0) if has_nan else np.asarray(data)
        return _apply_planewise(convolver, data_clean, nan_aware=False)

    # Whole-array FFT convolution for 3D (spectral-mixing) kernels.
    kernel_shape = kernel_array.shape
    pad_width = [(0, 0)] * (data.ndim - len(kernel_shape)) + [(s // 2, s // 2) for s in kernel_shape]

    kernel_nd = kernel_array
    while kernel_nd.ndim < data.ndim:
        kernel_nd = kernel_nd[None, ...]

    slices = tuple(slice(p[0], -p[1] if p[1] != 0 else None) for p in pad_width)

    if use_nan_aware:
        # NaN-aware smoothing via weighted convolution:
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

    # Whole-array convolution without NaN interpolation
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

    beam_unit_scale = beam_unit_scale_for_target_resolution(
        header,
        current_bmaj=float(current_bmaj),
        current_bmin=float(current_bmin),
        target_bmaj=float(target_bmaj),
        target_bmin=float(target_bmin),
    )

    sigma_kernel_maj = np.sqrt(sigma_kernel_maj_sq)
    sigma_kernel_min = np.sqrt(sigma_kernel_min_sq)

    # Convert to pixels
    sigma_kernel_x = sigma_kernel_min / pixel_scale_x
    sigma_kernel_y = sigma_kernel_maj / pixel_scale_y
    theta = np.deg2rad(target_bpa - current_bpa)

    # Create kernel
    kernel = Gaussian2DKernel(x_stddev=sigma_kernel_x, y_stddev=sigma_kernel_y, theta=theta)

    # Convolve each spatial plane independently (NaN-preserving), reusing one
    # kernel FFT across planes and bounding peak memory to a single plane.
    if data.ndim not in (2, 3, 4):
        raise ValueError(f"Unsupported data dimensionality: {data.ndim}")

    convolver = _ReflectFFTConvolver(kernel.array, data.shape[-2:])
    smoothed = _apply_planewise(convolver, np.asarray(data), nan_aware=True)

    if beam_unit_scale != 1.0:
        smoothed = smoothed * beam_unit_scale

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
