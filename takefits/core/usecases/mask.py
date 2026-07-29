"""Masking helpers for usecases."""
from __future__ import annotations

from typing import Dict, Literal, Optional

import numpy as np
from scipy import ndimage

from takefits.core.app_state import AppState
from takefits.core.io.save_fits import write_fits
from takefits.logic.data_tools import (
    ensure_operation_memory_budget,
    estimate_materialized_nbytes,
    is_lazy_scaled,
    materialize_elementwise_inputs,
)


MaskCondition = Literal["less_than", "greater_than"]
MomentMaskAlgorithm = Literal["smoothed_hysteresis", "moment_masking"]
MaskPolarity = Literal["emission", "absorption"]
MomentMaskPreset = Literal["faint", "normal", "strict"]
NoiseMethod = Literal["diff_mad", "mad", "std"]


MOMENT_MASK_PRESETS: Dict[str, Dict[str, float | int]] = {
    "faint": {
        "smooth_xy_pix": 1.5,
        "smooth_v_chan": 1.0,
        "seed_sigma": 3.0,
        "grow_sigma": 2.0,
        "clip_sigma": 4.0,
        "expand_xy_pix": 2,
        "expand_v_chan": 1,
        "min_channels": 2,
        "min_voxels": 0,
        "connectivity": 26,
    },
    "normal": {
        "smooth_xy_pix": 1.2,
        "smooth_v_chan": 0.8,
        "seed_sigma": 3.5,
        "grow_sigma": 2.5,
        "clip_sigma": 5.0,
        "expand_xy_pix": 1,
        "expand_v_chan": 1,
        "min_channels": 2,
        "min_voxels": 0,
        "connectivity": 26,
    },
    "strict": {
        "smooth_xy_pix": 0.8,
        "smooth_v_chan": 0.6,
        "seed_sigma": 4.5,
        "grow_sigma": 3.0,
        "clip_sigma": 6.0,
        "expand_xy_pix": 1,
        "expand_v_chan": 0,
        "min_channels": 3,
        "min_voxels": 0,
        "connectivity": 18,
    },
}

_MAX_NOISE_SAMPLE_VALUES = 1_000_000


def _array_element_count(data) -> int:
    """Return an array-like's size without coercing lazy scaled FITS data."""
    size = getattr(data, "size", None)
    if size is not None:
        return max(0, int(size))
    shape = tuple(int(dim) for dim in getattr(data, "shape", ()) or ())
    return int(np.prod(shape, dtype=np.int64)) if shape else 0


def _masked_output_nbytes(data) -> int:
    count = _array_element_count(data)
    dtype = np.dtype(getattr(data, "dtype", np.float64))
    output_dtype = dtype if np.issubdtype(dtype, np.floating) else np.dtype(np.float32)
    return count * int(output_dtype.itemsize)


def _prepare_mask_source(
    data,
    *,
    operation_name: str,
    required_bytes: int,
    output_array_count: float = 1.0,
) -> np.ndarray:
    """Preflight a mask operation and materialize a lazy source exactly once."""
    ensure_operation_memory_budget(
        required_bytes,
        operation_name=operation_name,
        guidance=(
            "Use Tools > Cutout, fewer channels, or a smaller spatial region "
            "before applying this mask."
        ),
    )
    prepared, = materialize_elementwise_inputs(
        data,
        operation_name=operation_name,
        output_array_count=output_array_count,
    )
    return prepared


def _moment_mask_working_bytes(data, *, include_masked_output: bool) -> int:
    """Estimate the peak extra storage used by a moment-mask recipe.

    Moment masking runs in phases: smoothing, thresholding, component
    filtering, then (for the panel action) creation of the masked output.
    Arrays from those phases are not all alive at once, so the safety check
    must use the largest phase rather than their sum.
    """
    full_count = _array_element_count(data)
    source_bytes = int(estimate_materialized_nbytes(data) or 0)
    lazy_input_bytes = source_bytes if is_lazy_scaled(data) else 0
    shape = tuple(int(dim) for dim in getattr(data, "shape", ()) or ())
    if len(shape) == 4 and shape[0] > 0:
        slice_count = full_count // shape[0]
        slice_source_bytes = source_bytes // shape[0]
    else:
        slice_count = full_count
        slice_source_bytes = source_bytes

    bool_bytes = slice_count * np.dtype(bool).itemsize
    full_mask_bytes = full_count * np.dtype(bool).itemsize

    # NaN-aware smoothing reuses its float64 numerator in place and keeps one
    # float32 weights cube.  Thresholding retains the smoothed cube plus a
    # handful of boolean masks.
    smoothing_peak = slice_source_bytes + (slice_count * 4) + bool_bytes
    threshold_peak = slice_source_bytes + (5 * bool_bytes)

    # Component filtering can retain int32 labels, first/last/extents, and
    # input/output/keep masks.  This intentionally assumes the worst case of
    # one connected-component id per voxel.
    component_peak = slice_count * (
        np.dtype(np.int32).itemsize * 4 + np.dtype(bool).itemsize * 3
    )
    sample_peak = min(slice_count, _MAX_NOISE_SAMPLE_VALUES) * 32
    processing_peak = max(
        smoothing_peak,
        threshold_peak,
        component_peak,
        slice_source_bytes + bool_bytes + sample_peak,
    )

    # Allow for SciPy line buffers and short-lived NumPy temporaries without
    # reverting to the former all-phases-at-once overestimate.
    processing_peak = int(processing_peak * 1.15)
    result_peak = full_mask_bytes
    if include_masked_output:
        result_peak += _masked_output_nbytes(data)
    return int(lazy_input_bytes + max(full_mask_bytes + processing_peak, result_peak))


def get_moment_mask_preset(preset: str = "normal") -> Dict[str, float | int]:
    """
    Return a copy of preset parameters for automatic moment masking.

    Args:
        preset: Preset name ('faint', 'normal', 'strict')

    Returns:
        Parameter dictionary
    """
    preset_key = str(preset or "normal").strip().lower()
    if preset_key not in MOMENT_MASK_PRESETS:
        raise ValueError(
            f"Unknown moment mask preset '{preset}'. "
            f"Available: {sorted(MOMENT_MASK_PRESETS.keys())}"
        )
    return dict(MOMENT_MASK_PRESETS[preset_key])


def _robust_sigma(values: np.ndarray) -> float:
    """Estimate sigma using MAD (scaled for Gaussian noise)."""
    if values.size == 0:
        return float("nan")
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    return float(1.4826 * mad)


def _bounded_noise_sample(
    array: np.ndarray,
    *,
    difference_axis: Optional[int] = None,
    max_values: int = _MAX_NOISE_SAMPLE_VALUES,
) -> np.ndarray:
    """Return a deterministic, bounded sample for robust noise estimation.

    For difference noise, complete spectra are sampled so the differences
    remain between adjacent channels.  Small arrays retain the established
    exact behaviour.
    """
    limit = max(1, int(max_values))
    if difference_axis is None:
        flat = np.ravel(array)
        if flat.size <= limit:
            return flat
        indices = np.linspace(0, flat.size - 1, num=limit, dtype=np.int64)
        return flat[indices]

    axis = int(difference_axis)
    moved = np.moveaxis(array, axis, 0)
    spectral_size = int(moved.shape[0])
    if spectral_size <= 1:
        return np.empty(0, dtype=array.dtype)
    spectra = moved.reshape(spectral_size, -1)
    diff_size = (spectral_size - 1) * int(spectra.shape[1])
    if diff_size <= limit:
        return np.diff(spectra, axis=0).ravel()

    spectrum_count = max(1, limit // (spectral_size - 1))
    spectrum_count = min(spectrum_count, int(spectra.shape[1]))
    indices = np.linspace(
        0,
        spectra.shape[1] - 1,
        num=spectrum_count,
        dtype=np.int64,
    )
    sampled_spectra = spectra[:, indices]
    return np.diff(sampled_spectra, axis=0).ravel()


def estimate_noise_sigma(
    data: np.ndarray,
    method: str = "diff_mad",
    spectral_axis: int = -3,
) -> float:
    """
    Estimate noise sigma from data using a robust method.

    Args:
        data: Input array
        method: 'diff_mad', 'mad', or 'std'
        spectral_axis: Spectral axis index for diff-MAD (default -3 for z in z,y,x)

    Returns:
        Estimated 1-sigma noise level
    """
    array = np.asarray(data)

    method_key = str(method or "diff_mad").strip().lower()
    if method_key == "diff_mad":
        axis = spectral_axis if spectral_axis >= 0 else (array.ndim + spectral_axis)
        if 0 <= axis < array.ndim and array.shape[axis] > 1:
            diff = _bounded_noise_sample(array, difference_axis=axis)
            diff_values = diff[np.isfinite(diff)]
            if diff_values.size > 0:
                sigma = _robust_sigma(diff_values) / np.sqrt(2.0)
                if np.isfinite(sigma) and sigma > 0:
                    return float(sigma)
        method_key = "mad"

    sample = _bounded_noise_sample(array)
    finite = np.isfinite(sample)
    if not np.any(finite):
        raise ValueError("No finite values available for noise estimation")
    values = sample[finite]
    if method_key == "mad":
        sigma = _robust_sigma(values)
        if np.isfinite(sigma) and sigma > 0:
            return float(sigma)
        method_key = "std"

    if method_key == "std":
        sigma = float(np.std(values))
        if np.isfinite(sigma) and sigma > 0:
            return sigma

    raise ValueError("Could not estimate a valid noise sigma")


def _resolve_connectivity_rank(connectivity: int, ndim: int) -> int:
    if ndim == 2:
        return 1 if connectivity <= 4 else 2
    if ndim == 3:
        if connectivity <= 6:
            return 1
        if connectivity <= 18:
            return 2
        return 3
    return 1


def _make_connectivity_structure(ndim: int, connectivity: int) -> np.ndarray:
    rank = _resolve_connectivity_rank(int(connectivity), ndim)
    return ndimage.generate_binary_structure(ndim, rank)


def _make_expansion_structure(expand_xy_pix: int, expand_v_chan: int) -> np.ndarray:
    xy = max(0, int(expand_xy_pix))
    vel = max(0, int(expand_v_chan))
    return np.ones((2 * vel + 1, 2 * xy + 1, 2 * xy + 1), dtype=bool)


def _filter_connected_components(
    mask: np.ndarray,
    min_voxels: int,
    min_channels: int,
    connectivity: int,
) -> np.ndarray:
    if not np.any(mask):
        return mask

    min_voxels_int = max(0, int(min_voxels))
    min_channels_int = max(1, int(min_channels))
    if min_voxels_int <= 0 and min_channels_int <= 1:
        return mask

    structure = _make_connectivity_structure(mask.ndim, connectivity)
    labels, n_labels = ndimage.label(mask, structure=structure)
    if n_labels == 0:
        return mask

    keep_components = np.ones(n_labels + 1, dtype=bool)
    keep_components[0] = False
    if min_voxels_int > 0:
        # Count only when requested; the standard presets use min_voxels=0.
        counts = np.bincount(labels.ravel(), minlength=n_labels + 1)
        keep_components &= counts >= min_voxels_int
        keep_components[0] = False
        del counts

    # Track first/last channel without creating one Python slice tuple per
    # component.  A noisy 1 GiB cube can contain millions of components.
    if min_channels_int > 1 and mask.ndim >= 3:
        first_channel = np.full(n_labels + 1, -1, dtype=np.int32)
        last_channel = np.full(n_labels + 1, -1, dtype=np.int32)
        for channel_index in range(labels.shape[0]):
            present = np.unique(labels[channel_index])
            present = present[present != 0]
            if present.size == 0:
                continue
            unseen = first_channel[present] < 0
            first_channel[present[unseen]] = channel_index
            last_channel[present] = channel_index
        spectral_extent = last_channel - first_channel + 1
        keep_components &= spectral_extent >= min_channels_int
        keep_components[0] = False
        del first_channel, last_channel, spectral_extent

    return keep_components[labels]


def _resolve_moment_mask_params(
    preset: str,
    smooth_xy_pix: Optional[float],
    smooth_v_chan: Optional[float],
    seed_sigma: Optional[float],
    grow_sigma: Optional[float],
    clip_sigma: Optional[float],
    expand_xy_pix: Optional[int],
    expand_v_chan: Optional[int],
    min_channels: Optional[int],
    min_voxels: Optional[int],
    connectivity: Optional[int],
) -> Dict[str, float | int]:
    params = get_moment_mask_preset(preset)

    if smooth_xy_pix is not None:
        params["smooth_xy_pix"] = float(smooth_xy_pix)
    if smooth_v_chan is not None:
        params["smooth_v_chan"] = float(smooth_v_chan)
    if seed_sigma is not None:
        params["seed_sigma"] = float(seed_sigma)
    if grow_sigma is not None:
        params["grow_sigma"] = float(grow_sigma)
    if clip_sigma is not None:
        params["clip_sigma"] = float(clip_sigma)
    if expand_xy_pix is not None:
        params["expand_xy_pix"] = int(expand_xy_pix)
    if expand_v_chan is not None:
        params["expand_v_chan"] = int(expand_v_chan)
    if min_channels is not None:
        params["min_channels"] = int(min_channels)
    if min_voxels is not None:
        params["min_voxels"] = int(min_voxels)
    if connectivity is not None:
        params["connectivity"] = int(connectivity)

    params["smooth_xy_pix"] = max(0.0, float(params["smooth_xy_pix"]))
    params["smooth_v_chan"] = max(0.0, float(params["smooth_v_chan"]))
    params["seed_sigma"] = max(0.1, float(params["seed_sigma"]))
    params["grow_sigma"] = max(0.1, float(params["grow_sigma"]))
    params["clip_sigma"] = max(0.1, float(params["clip_sigma"]))
    params["expand_xy_pix"] = max(0, int(params["expand_xy_pix"]))
    params["expand_v_chan"] = max(0, int(params["expand_v_chan"]))
    params["min_channels"] = max(1, int(params["min_channels"]))
    params["min_voxels"] = max(0, int(params["min_voxels"]))
    params["connectivity"] = int(params["connectivity"])

    return params


def _smooth_for_moment_mask(
    cube: np.ndarray,
    smooth_xy_pix: float,
    smooth_v_chan: float,
    finite: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Fast Gaussian smoothing specialized for moment-mask generation.

    Uses scipy.ndimage directly and performs weighted smoothing when NaNs exist:
      smoothed = gaussian(data * valid) / gaussian(valid)
    """
    sigma = (
        max(0.0, float(smooth_v_chan)),
        max(0.0, float(smooth_xy_pix)),
        max(0.0, float(smooth_xy_pix)),
    )
    if sigma[0] == 0.0 and sigma[1] == 0.0 and sigma[2] == 0.0:
        return cube.copy()

    finite_mask = np.isfinite(cube) if finite is None else finite
    if np.all(finite_mask):
        return ndimage.gaussian_filter(cube, sigma=sigma, mode="reflect")

    smoothed = np.where(finite_mask, cube, 0.0)
    weights = finite_mask.astype(np.float32, copy=True)
    ndimage.gaussian_filter(smoothed, sigma=sigma, mode="reflect", output=smoothed)
    ndimage.gaussian_filter(weights, sigma=sigma, mode="reflect", output=weights)

    eps = np.finfo(weights.dtype).eps
    np.divide(smoothed, weights, out=smoothed, where=weights > eps)
    smoothed[weights <= eps] = np.nan
    return smoothed


def _compute_moment_mask_cube(
    cube: np.ndarray,
    algorithm: str,
    polarity: str,
    noise_method: str,
    params: Dict[str, float | int],
) -> np.ndarray:
    if cube.ndim != 3:
        raise ValueError(f"Expected 3D cube, got ndim={cube.ndim}")

    finite = np.isfinite(cube)
    if not np.any(finite):
        return np.zeros_like(cube, dtype=bool)

    polarity_key = str(polarity or "emission").strip().lower()
    if polarity_key not in ("emission", "absorption"):
        raise ValueError("polarity must be 'emission' or 'absorption'")

    smooth_xy_pix = float(params["smooth_xy_pix"])
    smooth_v_chan = float(params["smooth_v_chan"])

    smoothed = _smooth_for_moment_mask(
        cube,
        smooth_xy_pix=smooth_xy_pix,
        smooth_v_chan=smooth_v_chan,
        finite=finite,
    )
    if polarity_key == "absorption":
        np.negative(smoothed, out=smoothed)

    sigma_original = estimate_noise_sigma(cube, method=noise_method, spectral_axis=0)
    sigma_smoothed = estimate_noise_sigma(smoothed, method=noise_method, spectral_axis=0)

    eps = np.finfo(float).eps
    sigma_original = max(float(sigma_original), eps)
    sigma_smoothed = max(float(sigma_smoothed), eps)

    algorithm_key = str(algorithm or "smoothed_hysteresis").strip().lower()
    connectivity = int(params["connectivity"])

    if algorithm_key == "smoothed_hysteresis":
        seed_sigma = float(params["seed_sigma"])
        grow_sigma = float(params["grow_sigma"])

        seed_mask = smoothed >= (seed_sigma * sigma_smoothed)
        if polarity_key == "emission":
            grow_mask = cube >= (grow_sigma * sigma_original)
        else:
            grow_mask = cube <= (-grow_sigma * sigma_original)
        candidate_mask = np.logical_or(seed_mask, grow_mask)
        candidate_mask &= finite
        del grow_mask
        structure = _make_connectivity_structure(3, connectivity)
        mask = ndimage.binary_propagation(seed_mask, structure=structure, mask=candidate_mask)
        mask &= finite
        del seed_mask, candidate_mask
    elif algorithm_key == "moment_masking":
        clip_sigma = float(params["clip_sigma"])
        expand_xy_pix = int(params["expand_xy_pix"])
        expand_v_chan = int(params["expand_v_chan"])

        seed_mask = smoothed >= (clip_sigma * sigma_smoothed)
        if expand_xy_pix > 0 or expand_v_chan > 0:
            expansion_structure = _make_expansion_structure(expand_xy_pix, expand_v_chan)
            mask = ndimage.binary_dilation(seed_mask, structure=expansion_structure)
        else:
            mask = seed_mask
        mask &= finite
        if mask is not seed_mask:
            del seed_mask
    else:
        raise ValueError(
            f"Unknown moment mask algorithm '{algorithm}'. "
            "Expected 'smoothed_hysteresis' or 'moment_masking'."
        )

    del smoothed, finite
    return _filter_connected_components(
        mask.astype(bool, copy=False),
        min_voxels=int(params["min_voxels"]),
        min_channels=int(params["min_channels"]),
        connectivity=connectivity,
    )


def compute_moment_mask(
    data: np.ndarray,
    algorithm: str = "smoothed_hysteresis",
    polarity: str = "emission",
    preset: str = "normal",
    smooth_xy_pix: Optional[float] = None,
    smooth_v_chan: Optional[float] = None,
    seed_sigma: Optional[float] = None,
    grow_sigma: Optional[float] = None,
    clip_sigma: Optional[float] = None,
    expand_xy_pix: Optional[int] = None,
    expand_v_chan: Optional[int] = None,
    min_channels: Optional[int] = None,
    min_voxels: Optional[int] = None,
    connectivity: Optional[int] = None,
    noise_method: str = "diff_mad",
) -> np.ndarray:
    """
    Create an automatic moment-analysis mask from a data cube.

    Supports two standard recipes:
      - smoothed_hysteresis: seed+grow mask from smoothed/original S/N thresholds
      - moment_masking: clip in smoothed cube, then expand back to original cube

    Args:
        data: Input 3D cube (z, y, x) or 4D cube (s, z, y, x)
        algorithm: 'smoothed_hysteresis' or 'moment_masking'
        polarity: 'emission' (positive lines) or 'absorption' (negative lines)
        preset: 'faint', 'normal', or 'strict'
        smooth_xy_pix, smooth_v_chan: Gaussian smoothing sigma in pixels/channels
        seed_sigma, grow_sigma: Thresholds for hysteresis recipe
        clip_sigma: Threshold for moment_masking recipe
        expand_xy_pix, expand_v_chan: Expansion radius for moment_masking recipe
        min_channels: Minimum spectral extent for connected components
        min_voxels: Minimum voxel count for connected components
        connectivity: Component connectivity (3D: 6, 18, 26)
        noise_method: 'diff_mad', 'mad', or 'std'

    Returns:
        Boolean mask with the same shape as data (True=keep, False=mask)
    """
    ndim = int(getattr(data, "ndim", 0))
    if ndim not in (3, 4):
        raise ValueError("compute_moment_mask expects 3D or 4D data")

    array = _prepare_mask_source(
        data,
        operation_name="Moment masking",
        required_bytes=_moment_mask_working_bytes(
            data,
            include_masked_output=False,
        ),
    )
    return _compute_moment_mask_prepared(
        array,
        algorithm=algorithm,
        polarity=polarity,
        preset=preset,
        smooth_xy_pix=smooth_xy_pix,
        smooth_v_chan=smooth_v_chan,
        seed_sigma=seed_sigma,
        grow_sigma=grow_sigma,
        clip_sigma=clip_sigma,
        expand_xy_pix=expand_xy_pix,
        expand_v_chan=expand_v_chan,
        min_channels=min_channels,
        min_voxels=min_voxels,
        connectivity=connectivity,
        noise_method=noise_method,
    )


def _compute_moment_mask_prepared(
    array: np.ndarray,
    *,
    algorithm: str,
    polarity: str,
    preset: str,
    smooth_xy_pix: Optional[float],
    smooth_v_chan: Optional[float],
    seed_sigma: Optional[float],
    grow_sigma: Optional[float],
    clip_sigma: Optional[float],
    expand_xy_pix: Optional[int],
    expand_v_chan: Optional[int],
    min_channels: Optional[int],
    min_voxels: Optional[int],
    connectivity: Optional[int],
    noise_method: str,
) -> np.ndarray:
    """Compute a moment mask from an already checked NumPy array."""
    params = _resolve_moment_mask_params(
        preset=preset,
        smooth_xy_pix=smooth_xy_pix,
        smooth_v_chan=smooth_v_chan,
        seed_sigma=seed_sigma,
        grow_sigma=grow_sigma,
        clip_sigma=clip_sigma,
        expand_xy_pix=expand_xy_pix,
        expand_v_chan=expand_v_chan,
        min_channels=min_channels,
        min_voxels=min_voxels,
        connectivity=connectivity,
    )

    if array.ndim == 3:
        return _compute_moment_mask_cube(array, algorithm, polarity, noise_method, params)

    result = np.zeros_like(array, dtype=bool)
    for stokes_index in range(array.shape[0]):
        result[stokes_index] = _compute_moment_mask_cube(
            array[stokes_index],
            algorithm,
            polarity,
            noise_method,
            params,
        )
    return result


def compute_moment_masked(data, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Return masked data and its moment mask after one source materialization."""
    ndim = int(getattr(data, "ndim", 0))
    if ndim not in (3, 4):
        raise ValueError("compute_moment_mask expects 3D or 4D data")

    array = _prepare_mask_source(
        data,
        operation_name="Moment masking",
        required_bytes=_moment_mask_working_bytes(
            data,
            include_masked_output=True,
        ),
    )
    mask = _compute_moment_mask_prepared(
        array,
        algorithm=kwargs.get("algorithm", "smoothed_hysteresis"),
        polarity=kwargs.get("polarity", "emission"),
        preset=kwargs.get("preset", "normal"),
        smooth_xy_pix=kwargs.get("smooth_xy_pix"),
        smooth_v_chan=kwargs.get("smooth_v_chan"),
        seed_sigma=kwargs.get("seed_sigma"),
        grow_sigma=kwargs.get("grow_sigma"),
        clip_sigma=kwargs.get("clip_sigma"),
        expand_xy_pix=kwargs.get("expand_xy_pix"),
        expand_v_chan=kwargs.get("expand_v_chan"),
        min_channels=kwargs.get("min_channels"),
        min_voxels=kwargs.get("min_voxels"),
        connectivity=kwargs.get("connectivity"),
        noise_method=kwargs.get("noise_method", "diff_mad"),
    )
    masked = array.copy()
    if not np.issubdtype(masked.dtype, np.floating):
        masked = masked.astype(np.float32, copy=False)
    masked[~mask] = np.nan
    return masked, mask


def apply_mask_moment_recipe(
    state: AppState,
    algorithm: str = "smoothed_hysteresis",
    polarity: str = "emission",
    preset: str = "normal",
    smooth_xy_pix: Optional[float] = None,
    smooth_v_chan: Optional[float] = None,
    seed_sigma: Optional[float] = None,
    grow_sigma: Optional[float] = None,
    clip_sigma: Optional[float] = None,
    expand_xy_pix: Optional[int] = None,
    expand_v_chan: Optional[int] = None,
    min_channels: Optional[int] = None,
    min_voxels: Optional[int] = None,
    connectivity: Optional[int] = None,
    noise_method: str = "diff_mad",
) -> AppState:
    """
    Apply automatic moment-analysis mask to state.data in-place.

    Args:
        state: The AppState to update
        algorithm: 'smoothed_hysteresis' or 'moment_masking'
        polarity: 'emission' or 'absorption'
        preset: 'faint', 'normal', or 'strict'
        smooth_xy_pix, smooth_v_chan: Optional smoothing overrides
        seed_sigma, grow_sigma, clip_sigma: Optional threshold overrides
        expand_xy_pix, expand_v_chan: Optional expansion overrides
        min_channels, min_voxels, connectivity: Component filters
        noise_method: 'diff_mad', 'mad', or 'std'

    Returns:
        Updated AppState
    """
    if state.data is None:
        raise ValueError("No data loaded")

    masked, _mask = compute_moment_masked(
        state.data,
        algorithm=algorithm,
        polarity=polarity,
        preset=preset,
        smooth_xy_pix=smooth_xy_pix,
        smooth_v_chan=smooth_v_chan,
        seed_sigma=seed_sigma,
        grow_sigma=grow_sigma,
        clip_sigma=clip_sigma,
        expand_xy_pix=expand_xy_pix,
        expand_v_chan=expand_v_chan,
        min_channels=min_channels,
        min_voxels=min_voxels,
        connectivity=connectivity,
        noise_method=noise_method,
    )

    state.data = masked
    return state


def compute_masked(
    data: np.ndarray,
    threshold: float,
    condition: str = "less_than"
) -> np.ndarray:
    """
    Apply threshold mask to data (pure function).

    Pixels meeting the condition are set to NaN.

    Args:
        data: Input data array
        threshold: Threshold value
        condition: "less_than" or "greater_than"

    Returns:
        Masked data array (copy with NaN where condition met)
    """
    if condition not in ("less_than", "greater_than"):
        raise ValueError(f"Unknown condition: {condition}")

    count = _array_element_count(data)
    source_bytes = int(estimate_materialized_nbytes(data) or 0)
    required = _masked_output_nbytes(data) + count * np.dtype(bool).itemsize
    if is_lazy_scaled(data):
        required += source_bytes
    data = _prepare_mask_source(
        data,
        operation_name="Threshold masking",
        required_bytes=required,
    )

    result = data.copy()
    if np.issubdtype(result.dtype, np.integer):
        result = result.astype(np.float32, copy=False)

    if condition == "less_than":
        result[result < threshold] = np.nan
    else:
        result[result > threshold] = np.nan

    return result


def compute_masked_from_keep_mask(
    data,
    keep_mask: np.ndarray,
    *,
    operation_name: str = "External masking",
) -> np.ndarray:
    """Copy data and set pixels outside a precomputed boolean mask to NaN."""
    expected_shape = tuple(int(dim) for dim in getattr(data, "shape", ()) or ())
    keep = np.asarray(keep_mask, dtype=bool)
    if tuple(keep.shape) != expected_shape:
        raise ValueError(
            f"Mask shape {keep.shape} doesn't match data shape {expected_shape}"
        )

    count = _array_element_count(data)
    source_bytes = int(estimate_materialized_nbytes(data) or 0)
    required = _masked_output_nbytes(data) + count * np.dtype(bool).itemsize
    if is_lazy_scaled(data):
        required += source_bytes
    array = _prepare_mask_source(
        data,
        operation_name=operation_name,
        required_bytes=required,
    )
    masked = array.copy()
    if not np.issubdtype(masked.dtype, np.floating):
        masked = masked.astype(np.float32, copy=False)
    masked[~keep] = np.nan
    return masked


def apply_mask_threshold(
    state: AppState,
    threshold: float,
    condition: str = "less_than"
) -> AppState:
    """
    Apply threshold mask to state.data in-place.

    Args:
        state: The AppState to update
        threshold: Threshold value
        condition: "less_than" or "greater_than"

    Returns:
        The updated AppState
    """
    if state.data is None:
        raise ValueError("No data loaded")

    state.data = compute_masked(state.data, threshold, condition)
    return state


def apply_mask_external(
    state: AppState,
    mask_path: str,
    mask_value: float = 0.0
) -> AppState:
    """
    Apply external FITS mask to state.data in-place.

    Pixels where mask == mask_value are set to NaN.

    Args:
        state: The AppState to update
        mask_path: Path to the mask FITS file
        mask_value: Value in mask that indicates masked pixels (default 0.0)

    Returns:
        The updated AppState

    Raises:
        ValueError: If mask shape doesn't match data shape
    """
    from astropy.io import fits

    if state.data is None:
        raise ValueError("No data loaded")

    with fits.open(mask_path) as hdul:
        mask_data = hdul[0].data

    if mask_data.shape != state.data.shape:
        raise ValueError(
            f"Mask shape {mask_data.shape} doesn't match data shape {state.data.shape}"
        )

    # Preserve the established semantics: NaN values in an external mask do
    # not equal mask_value and therefore remain unmasked.
    state.data = compute_masked_from_keep_mask(
        state.data,
        mask_data != mask_value,
        operation_name="External masking",
    )

    return state


def export_mask_fits(
    state: AppState,
    mask_data: np.ndarray,
    output_path: str,
    threshold: Optional[float] = None,
    condition: Optional[str] = None,
    history_entries: Optional[list] = None,
    mask_as_float: bool = False,
    nan_for_mask: bool = False,
) -> str:
    """
    Export a mask array to FITS file.

    Args:
        state: AppState with original header info
        mask_data: Mask array (1=unmasked, 0=masked)
        output_path: Path for output FITS file
        threshold: Optional threshold value (for HISTORY)
        condition: Optional condition (for HISTORY)
        history_entries: Optional list of HISTORY entries
        mask_as_float: Save mask as float32 instead of int16
        nan_for_mask: If saving as float, convert masked pixels (0) to NaN

    Returns:
        The output file path
    """
    from astropy.io import fits

    header = state.header.copy() if state.header is not None else fits.Header()

    # Remove header keys that conflict with mask data typing/scaling.
    original_naxis = int(header.get('NAXIS', 0) or 0)
    for key in ('SIMPLE', 'BITPIX', 'NAXIS', 'EXTEND', 'BSCALE', 'BZERO', 'BLANK', 'BUNIT'):
        if key in header:
            del header[key]
    for i in range(1, original_naxis + 1):
        key = f'NAXIS{i}'
        if key in header:
            del header[key]

    output_dtype = np.dtype(np.float32 if mask_as_float else np.int16)
    output_bytes = _array_element_count(mask_data) * output_dtype.itemsize
    ensure_operation_memory_budget(
        2 * output_bytes,
        operation_name="Mask FITS export",
        guidance=(
            "Free memory or export a mask from a smaller cutout before saving."
        ),
    )

    if mask_as_float:
        data_to_save = np.asarray(mask_data).astype(np.float32, copy=True)
        if nan_for_mask:
            data_to_save[data_to_save == 0] = np.nan
    else:
        data_to_save = np.asarray(mask_data).astype(np.int16, copy=False)

    # Update shape info
    header['NAXIS'] = data_to_save.ndim
    for i, dim in enumerate(data_to_save.shape[::-1], 1):
        header[f'NAXIS{i}'] = dim

    # Add masking info to header
    if threshold is not None:
        header['MASKTHRS'] = (threshold, 'Mask threshold value')
    if condition is not None:
        header['MASKCOND'] = (condition, 'Mask condition')

    # Add history
    if history_entries:
        for entry in history_entries:
            header.add_history(entry)

    return write_fits(output_path, data_to_save, header)


def estimate_noise(
    state: AppState,
    method: str = "diff_mad",
    spectral_axis: int = -3,
) -> float:
    """Estimate the 1-sigma noise of the loaded cube (TF-303).

    State-based wrapper around :func:`estimate_noise_sigma` so the value is
    reachable from an action / CLI manifest. It is the natural way to express
    contour levels or mask thresholds as a multiple of the rms.

    Args:
        state: AppState with data loaded.
        method: 'diff_mad' (default), 'mad', or 'std'.
        spectral_axis: Spectral axis index for diff-MAD.

    Returns:
        Estimated 1-sigma noise level.
    """
    if state.data is None:
        raise ValueError("No data loaded")
    return float(
        estimate_noise_sigma(
            state.data,
            method=method,
            spectral_axis=spectral_axis,
        )
    )
