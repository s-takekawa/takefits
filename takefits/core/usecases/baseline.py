"""Polynomial baseline subtraction usecases."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union
import warnings

import numpy as np

from takefits.core.app_state import AppState
from takefits.core.io.save_fits import write_fits
from takefits.logic.data_tools import (
    ensure_operation_memory_budget,
    estimate_materialized_nbytes,
    is_lazy_scaled,
    materialize_elementwise_inputs,
)
from .utils import axis_world_to_pixel, get_axis_ctype, parse_world_coordinate


WorldRange = Tuple[Union[float, str], Union[float, str]]


@dataclass
class BaselineSubtractionResult:
    """Container for baseline subtraction outputs."""

    subtracted_data: np.ndarray
    baseline_model: np.ndarray
    order: int
    world_ranges: List[Tuple[float, float]]
    pixel_ranges: List[Tuple[int, int]]
    n_total_spectra: int
    n_fitted_spectra: int

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(np.asarray(self.subtracted_data).shape)


def _baseline_working_bytes(data) -> int:
    """Estimate materialization plus the corrected/model output cubes."""
    count = int(getattr(data, "size", 0) or 0)
    source_bytes = int(estimate_materialized_nbytes(data) or 0)
    dtype = np.dtype(getattr(data, "dtype", np.float64))
    output_dtype = dtype if np.issubdtype(dtype, np.floating) else np.dtype(np.float32)
    output_bytes = count * int(output_dtype.itemsize)
    return (source_bytes if is_lazy_scaled(data) else 0) + (2 * output_bytes)


def _default_reference_pixel(state: AppState) -> Optional[Tuple[float, ...]]:
    if state.wcs is None:
        return None
    naxis = int(getattr(state.wcs, "naxis", 0) or 0)
    if naxis <= 0:
        return None

    ref = [0.0] * naxis
    try:
        crpix = np.asarray(state.wcs.wcs.crpix, dtype=float).reshape(-1)
        for idx in range(min(naxis, crpix.size)):
            ref[idx] = float(crpix[idx] - 1.0)
    except Exception:
        pass

    if naxis >= 1:
        ref[0] = float(getattr(state.cursor, "xpix", ref[0]))
    if naxis >= 2:
        ref[1] = float(getattr(state.cursor, "ypix", ref[1]))
    if naxis >= 3:
        ref[2] = float(getattr(state, "current_z", ref[2]))
    if naxis >= 4:
        ref[3] = float(getattr(state, "current_s", ref[3]))
    return tuple(ref)


def _normalize_reference_pixel(
    state: AppState,
    reference_pixel: Optional[Sequence[float]],
) -> Optional[Tuple[float, ...]]:
    fallback = _default_reference_pixel(state)
    if reference_pixel is None:
        return fallback
    if state.wcs is None:
        return None
    naxis = int(getattr(state.wcs, "naxis", 0) or 0)
    if naxis <= 0:
        return None

    values = list(fallback or ([0.0] * naxis))
    try:
        supplied = list(reference_pixel)
    except Exception:
        supplied = []
    for idx in range(min(naxis, len(supplied))):
        try:
            values[idx] = float(supplied[idx])
        except Exception:
            continue
    return tuple(values)


def _spectral_wcs_axis(state: AppState, *, data_ndim: int) -> int:
    spectral_meta = getattr(state, "spectral_metadata", {}) or {}
    axis_index = spectral_meta.get("axis_index")
    try:
        axis_index_int = int(axis_index)
        if axis_index_int >= 1:
            return axis_index_int - 1
    except Exception:
        pass

    wcs = getattr(state, "wcs", None)
    if wcs is not None and hasattr(wcs, "wcs"):
        try:
            for idx, ctype in enumerate(list(getattr(wcs.wcs, "ctype", []) or [])):
                token = str(ctype or "").upper()
                if any(tag in token for tag in ("VRAD", "VELO", "VOPT", "FREQ", "WAVE")):
                    return int(idx)
        except Exception:
            pass

    # numpy axis 0 (z) corresponds to WCS axis (ndim-1) for (z, y, x) ordering
    return int(max(0, data_ndim - 1))


def _world_ranges_to_channel_mask(
    state: AppState,
    *,
    world_ranges: Sequence[WorldRange],
    n_channels: int,
    spectral_wcs_axis: int,
    reference_pixel: Optional[Tuple[float, ...]],
) -> Tuple[np.ndarray, List[Tuple[float, float]], List[Tuple[int, int]]]:
    if state.wcs is None:
        raise ValueError("WCS is required for world-range baseline subtraction.")
    if not world_ranges:
        raise ValueError("At least one world range is required.")
    if n_channels <= 0:
        raise ValueError("No spectral channels available.")

    ctype = get_axis_ctype(state, spectral_wcs_axis)
    mask = np.zeros(int(n_channels), dtype=bool)
    normalized_world_ranges: List[Tuple[float, float]] = []
    pixel_ranges: List[Tuple[int, int]] = []

    for idx, entry in enumerate(world_ranges, start=1):
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError(f"Invalid world range at index {idx}: expected (min, max).")
        raw_lo, raw_hi = entry
        try:
            world_lo = (
                parse_world_coordinate(raw_lo, ctype)
                if isinstance(raw_lo, str)
                else float(raw_lo)
            )
            world_hi = (
                parse_world_coordinate(raw_hi, ctype)
                if isinstance(raw_hi, str)
                else float(raw_hi)
            )
        except Exception as exc:
            raise ValueError(f"Invalid world range values at index {idx}: {entry}") from exc
        if not (np.isfinite(world_lo) and np.isfinite(world_hi)):
            raise ValueError(f"Non-finite world range at index {idx}: {entry}")

        pix_lo = axis_world_to_pixel(
            state,
            world_lo,
            spectral_wcs_axis,
            reference_pixel=reference_pixel,
        )
        pix_hi = axis_world_to_pixel(
            state,
            world_hi,
            spectral_wcs_axis,
            reference_pixel=reference_pixel,
        )

        p0 = int(np.floor(min(float(pix_lo), float(pix_hi))))
        p1 = int(np.ceil(max(float(pix_lo), float(pix_hi))))
        p0 = max(0, p0)
        p1 = min(int(n_channels) - 1, p1)
        if p0 > p1:
            continue
        mask[p0 : p1 + 1] = True
        normalized_world_ranges.append((float(world_lo), float(world_hi)))
        pixel_ranges.append((int(p0), int(p1)))

    if not np.any(mask):
        raise ValueError("No valid channels selected from world ranges.")
    return mask, normalized_world_ranges, pixel_ranges


def _fit_baseline_single_cube(
    cube: np.ndarray,
    *,
    channel_mask: np.ndarray,
    order: int,
    subtracted_out: Optional[np.ndarray] = None,
    model_out: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    if cube.ndim != 3:
        raise ValueError(f"Expected 3D cube, got {cube.ndim}D.")
    n_channels = int(cube.shape[0])
    if channel_mask.shape != (n_channels,):
        raise ValueError("channel_mask shape does not match spectral axis length.")

    out_dtype = cube.dtype if np.issubdtype(cube.dtype, np.floating) else np.float32
    if subtracted_out is None:
        subtracted = np.asarray(cube, dtype=out_dtype).copy()
    else:
        if subtracted_out.shape != cube.shape:
            raise ValueError("subtracted_out shape does not match cube shape.")
        subtracted = subtracted_out
        np.copyto(subtracted, cube, casting="unsafe")
    if model_out is None:
        model = np.full(cube.shape, np.nan, dtype=out_dtype)
    else:
        if model_out.shape != cube.shape:
            raise ValueError("model_out shape does not match cube shape.")
        model = model_out
        model.fill(np.nan)
    x_axis = np.arange(n_channels, dtype=float)

    total_spectra = int(np.prod(cube.shape[1:]))
    fitted_spectra = 0

    rank_warning = getattr(np, "RankWarning", Warning)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", rank_warning)
        for iy, ix in np.ndindex(cube.shape[1], cube.shape[2]):
            spectrum = np.asarray(cube[:, iy, ix], dtype=float)
            finite = np.isfinite(spectrum)
            fit_mask = finite & channel_mask
            if int(np.count_nonzero(fit_mask)) < (int(order) + 1):
                continue
            try:
                coeffs = np.polyfit(x_axis[fit_mask], spectrum[fit_mask], int(order))
                baseline = np.polyval(coeffs, x_axis)
            except Exception:
                continue

            corrected = spectrum - baseline
            corrected[~finite] = np.nan

            model[:, iy, ix] = np.asarray(baseline, dtype=out_dtype)
            subtracted[:, iy, ix] = np.asarray(corrected, dtype=out_dtype)
            fitted_spectra += 1

    return subtracted, model, total_spectra, fitted_spectra


def compute_polynomial_baseline_subtraction(
    state: AppState,
    *,
    world_ranges: Sequence[WorldRange],
    order: int = 1,
    reference_pixel: Optional[Sequence[float]] = None,
) -> BaselineSubtractionResult:
    """
    Compute polynomial baseline subtraction from world-coordinate line-free ranges.

    Args:
        state: App state with loaded data/WCS.
        world_ranges: List of (min_world, max_world) ranges.
        order: Polynomial order (>=0).
        reference_pixel: Optional full WCS reference pixel tuple for world->pixel conversion.

    Returns:
        BaselineSubtractionResult with subtracted data and baseline model.
    """
    if state.data is None:
        raise ValueError("No data loaded.")
    if int(order) < 0:
        raise ValueError("Polynomial order must be >= 0.")

    source_data = state.data
    data_ndim = int(getattr(source_data, "ndim", 0))
    if data_ndim < 3:
        raise ValueError("Baseline subtraction requires 3D/4D cube data.")

    working_ndim = 3 if data_ndim == 4 else data_ndim
    spectral_wcs_axis = _spectral_wcs_axis(state, data_ndim=working_ndim)
    ref_pixel = _normalize_reference_pixel(state, reference_pixel)
    source_shape = tuple(int(dim) for dim in source_data.shape)
    n_channels = int(source_shape[1] if data_ndim == 4 else source_shape[0])
    channel_mask, normalized_world_ranges, pixel_ranges = _world_ranges_to_channel_mask(
        state,
        world_ranges=world_ranges,
        n_channels=n_channels,
        spectral_wcs_axis=spectral_wcs_axis,
        reference_pixel=ref_pixel,
    )

    ensure_operation_memory_budget(
        _baseline_working_bytes(source_data),
        operation_name="Polynomial baseline subtraction",
        guidance=(
            "Use Tools > Cutout, fewer channels, or a smaller spatial region "
            "before subtracting a baseline."
        ),
    )
    data, = materialize_elementwise_inputs(
        source_data,
        operation_name="Polynomial baseline subtraction",
        output_array_count=2.0,
    )

    order_int = int(order)
    if data.ndim == 3:
        subtracted, model, n_total, n_fit = _fit_baseline_single_cube(
            data,
            channel_mask=channel_mask,
            order=order_int,
        )
    else:
        out_dtype = data.dtype if np.issubdtype(data.dtype, np.floating) else np.float32
        subtracted = np.asarray(data, dtype=out_dtype).copy()
        model = np.full(data.shape, np.nan, dtype=out_dtype)
        n_total = 0
        n_fit = 0
        for s_idx in range(int(data.shape[0])):
            _, _, s_total, s_fit = _fit_baseline_single_cube(
                np.asarray(data[s_idx]),
                channel_mask=channel_mask,
                order=order_int,
                subtracted_out=subtracted[s_idx],
                model_out=model[s_idx],
            )
            n_total += int(s_total)
            n_fit += int(s_fit)

    return BaselineSubtractionResult(
        subtracted_data=subtracted,
        baseline_model=model,
        order=order_int,
        world_ranges=normalized_world_ranges,
        pixel_ranges=pixel_ranges,
        n_total_spectra=int(n_total),
        n_fitted_spectra=int(n_fit),
    )


def _apply_baseline_result_to_state(
    state: AppState,
    result: BaselineSubtractionResult,
) -> AppState:
    state.data = result.subtracted_data
    spectral_meta = dict(getattr(state, "spectral_metadata", {}) or {})
    spectral_meta["baseline_last_order"] = int(result.order)
    spectral_meta["baseline_last_world_ranges"] = [list(pair) for pair in result.world_ranges]
    spectral_meta["baseline_last_pixel_ranges"] = [list(pair) for pair in result.pixel_ranges]
    state.spectral_metadata = spectral_meta
    return state


def apply_baseline_subtraction(
    state: AppState,
    *,
    world_ranges: Sequence[WorldRange],
    order: int = 1,
    reference_pixel: Optional[Sequence[float]] = None,
) -> AppState:
    """Apply polynomial baseline subtraction in-place to state.data."""
    result = compute_polynomial_baseline_subtraction(
        state,
        world_ranges=world_ranges,
        order=order,
        reference_pixel=reference_pixel,
    )
    return _apply_baseline_result_to_state(state, result)


def export_baseline_model_fits(
    state: AppState,
    output_path: str,
    history_entries: Optional[list] = None,
    result: Optional[BaselineSubtractionResult] = None,
    baseline_model: Optional[np.ndarray] = None,
) -> str:
    """
    Export a polynomial baseline model cube to FITS.

    Args:
        state: AppState with original header info.
        output_path: Path for output FITS file.
        history_entries: Optional list of HISTORY entries.
        result: Optional baseline result whose model should be exported.
        baseline_model: Optional explicit baseline model array.

    Returns:
        The output file path.
    """
    from datetime import datetime
    from astropy.io import fits

    model = baseline_model
    if model is None and result is not None:
        model = result.baseline_model
    if model is None:
        raise ValueError("No baseline model available for export.")

    data_to_save = np.asarray(model)
    header = state.header.copy() if state.header is not None else fits.Header()
    header["NAXIS"] = data_to_save.ndim
    for idx, dim in enumerate(data_to_save.shape[::-1], start=1):
        header[f"NAXIS{idx}"] = int(dim)

    header.add_history(
        f"Baseline model exported using takefits on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    if history_entries:
        for entry in history_entries:
            header.add_history(entry)

    return write_fits(
        output_path,
        data_to_save,
        header,
        ensure_datamin_datamax=True,
        drop_extrema_if_all_invalid=True,
    )
