import copy
import math
import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed,  ProcessPoolExecutor
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Sequence, Tuple, NamedTuple

import numpy as np
from astropy import units as u
from astropy.coordinates import FK5, Galactic, ICRS
from astropy.time import Time
from astropy.io import fits
from astropy.wcs import WCS

import os as _os_for_threads
_os_for_threads.environ.setdefault("OMP_NUM_THREADS", "1")
_os_for_threads.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os_for_threads.environ.setdefault("MKL_NUM_THREADS", "1")
_os_for_threads.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
_os_for_threads.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from reproject import reproject_interp

_SPLINE_ORDER_MAP = {
    "nearest": 0,
    "nearest-neighbor": 0,
    "bilinear": 1,
    "biquadratic": 2,
    "bicubic": 3,
}
_MASK_DIVIDE_EPS = 1e-6


def _normalize_interpolation_label(order: str | None) -> Optional[str]:
    if order is None:
        return None
    return str(order).lower()


def _interpolation_label_to_order(order: str | None) -> Optional[int]:
    label = _normalize_interpolation_label(order)
    if not label:
        return None
    return _SPLINE_ORDER_MAP.get(label)


def _reproject_with_nan_support(
    array: "np.ndarray",
    source_wcs: WCS,
    target_wcs: WCS,
    shape_out: Tuple[int, int] | Tuple[int, ...],
    order: str,
    work_dtype: np.dtype,
    *,
    output_array: Optional[np.ndarray] = None,
    need_coverage: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray], bool, bool]:
    """
    Reproject while handling NaNs for high-order splines by normalizing with a mask.
    Returns (reprojected_data, coverage_map or None, used_nan_fix_flag).
    """
    arr = np.asarray(array, dtype=work_dtype)
    has_nan = np.isnan(arr).any()
    used_nan_fix = False
    downgraded_to_bilinear = False
    coverage = None

    if has_nan:
        used_nan_fix = True
        mask = np.isfinite(arr).astype(work_dtype, copy=False)
        filled = np.where(mask, arr, 0.0).astype(work_dtype, copy=False)

        if output_array is None:
            data_out = np.empty(shape_out, dtype=work_dtype)
        else:
            data_out = output_array
            data_out.fill(0.0)

        data_out = reproject_interp(
            (filled, source_wcs),
            target_wcs,
            shape_out=shape_out,
            order=order,
            output_array=data_out,
            return_footprint=False,
        )

        coverage = np.empty(shape_out, dtype=work_dtype)
        coverage = reproject_interp(
            (mask, source_wcs),
            target_wcs,
            shape_out=shape_out,
            order=order,
            output_array=coverage,
            return_footprint=False,
        )

        if not np.issubdtype(coverage.dtype, np.floating):
            coverage = coverage.astype(work_dtype, copy=False)
        coverage = np.clip(coverage, 0.0, 1.0)
        valid_mask = coverage > _MASK_DIVIDE_EPS
        with np.errstate(invalid="ignore", divide="ignore"):
            np.divide(data_out, coverage, out=data_out, where=valid_mask)
        data_out[~valid_mask] = np.nan

        if np.isnan(data_out).all():
            fallback_order = "bilinear"
            downgraded_to_bilinear = True
            if output_array is None:
                fallback_out = np.empty(shape_out, dtype=work_dtype)
            else:
                fallback_out = output_array
                fallback_out.fill(0.0)
            data_out = reproject_interp(
                (filled, source_wcs),
                target_wcs,
                shape_out=shape_out,
                order=fallback_order,
                output_array=fallback_out,
                return_footprint=False,
            )
            coverage = reproject_interp(
                (mask, source_wcs),
                target_wcs,
                shape_out=shape_out,
                order=fallback_order,
                output_array=np.empty(shape_out, dtype=work_dtype),
                return_footprint=False,
            )
            coverage = np.clip(coverage, 0.0, 1.0)
            valid_mask = coverage > _MASK_DIVIDE_EPS
            with np.errstate(invalid="ignore", divide="ignore"):
                np.divide(data_out, coverage, out=data_out, where=valid_mask)
            data_out[~valid_mask] = np.nan
    else:
        if output_array is None:
            data_out = np.empty(shape_out, dtype=work_dtype)
        else:
            data_out = output_array

        data_out = reproject_interp(
            (arr, source_wcs),
            target_wcs,
            shape_out=shape_out,
            order=order,
            output_array=data_out,
            return_footprint=False,
        )
        if need_coverage:
            coverage = np.asarray(np.isfinite(data_out), dtype=work_dtype)

    if not need_coverage:
        coverage = None

    return data_out, coverage, used_nan_fix, downgraded_to_bilinear


def _reproject_plane_worker(
    plane_2d: "np.ndarray",
    src_wcs_header: dict,
    tgt_wcs_header: dict,
    shape_out_2d: Tuple[int, int],
    order: str,
    footprint_thresh: float | None = 0.5,
    ) -> "np.ndarray":
    """Reproject a single 2D plane in a separate process.
    Rebuild WCS objects from headers to avoid shared-state races.
    Returns float64 output with low-coverage masked to NaN.
    """
    
    # Ensure contiguous float64 input (robust for C extensions)
    plane = np.ascontiguousarray(plane_2d, dtype=np.float64)

    # Recreate WCS locally in the worker process
    src_wcs = WCS(src_wcs_header)
    tgt_wcs = WCS(tgt_wcs_header)

    # Do the interpolation with NaN-aware handling and collect coverage
    out, coverage, used_nan_fix, downgraded = _reproject_with_nan_support(
        plane,
        src_wcs,
        tgt_wcs,
        shape_out_2d,
        order,
        plane.dtype,
        output_array=np.empty(shape_out_2d, dtype=plane.dtype),
        need_coverage=True,
    )

    # Optionally mask low-coverage pixels (tunable threshold)
    if footprint_thresh is not None and coverage is not None:
        thresh = _MASK_DIVIDE_EPS if used_nan_fix else float(footprint_thresh)
        out[coverage < thresh] = np.nan

    return out, bool(used_nan_fix), bool(downgraded)  # float64 data, flags

# ---- Helpers: obstime parsing & preflight mapping ----
def parse_obstime_from_header(hdr):
    """Return astropy Time or None from DATE-OBS / MJD-OBS robustly."""
    from astropy.time import Time
    if hdr is None:
        return None
    obstime = None
    date_obs = hdr.get("DATE-OBS")
    mjd_obs = hdr.get("MJD-OBS")
    if date_obs:
        s = str(date_obs).strip()
        for fmt in ("fits", "isot"):
            try:
                obstime = Time(s, format=fmt, scale="utc")
                break
            except Exception:
                pass
    if obstime is None and mjd_obs is not None:
        try:
            obstime = Time(float(mjd_obs), format="mjd", scale="utc")
        except Exception:
            pass
    return obstime

def preflight_center_mapping(src_wcs_2d, tgt_wcs_2d, ny, nx):
    """Fail fast if target->source mapping yields non-finite pixels (helps catch FK4 issues)."""
    try:
        cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
        world = tgt_wcs_2d.pixel_to_world(cx, cy)
        from astropy.wcs.utils import wcs_to_celestial_frame
        src_frame = wcs_to_celestial_frame(src_wcs_2d)
        world_src = world.transform_to(src_frame)
        sx, sy = src_wcs_2d.world_to_pixel(world_src)
        import numpy as _np
        if not _np.isfinite([sx, sy]).all():
            raise RuntimeError("Target WCS maps to non-finite source pixels; check FK4 obstime/E-terms.")
    except Exception as e:
        raise RuntimeError(f"Preflight WCS mapping failed: {e}")


from astropy.wcs.utils import wcs_to_celestial_frame
from scipy.interpolate import interp1d

from reproject import reproject_interp

try:  # pragma: no cover - optional dependency
    from scipy.ndimage import map_coordinates
except Exception:  # pragma: no cover - scipy.ndimage unavailable
    map_coordinates = None

try:  # pragma: no cover - optional dependency
    import dask.array as da
except Exception:  # pragma: no cover - dask unavailable
    da = None

try:
    from reproject.utils import determine_optimal_celestial_wcs
except ImportError:  # pragma: no cover - compatibility fallback
    from reproject.mosaicking import find_optimal_celestial_wcs as determine_optimal_celestial_wcs


class RegridEngine:
    """Pure regridding engine (no Qt dependencies)."""

    def __init__(
        self,
        original_data,
        original_wcs,
        original_header: Optional[fits.Header] = None,
        filename: Optional[str] = None,
        progress_callback: Optional[Callable[[int], None]] = None,
        finished_callback: Optional[Callable[[object, object], None]] = None,
        error_callback: Optional[Callable[[str], None]] = None,
    ):
        self.filename = filename
        self._progress_callback = progress_callback
        self._finished_callback = finished_callback
        self._error_callback = error_callback
        
        self._last_frame_hint = None  # Remember last requested celestial frame
        self.original_data = original_data
        self._dask_data = self._maybe_wrap_data(original_data)
        self._slice_cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._slice_cache_lock = threading.Lock()
        self._max_cached_slices = 6
        self._spectral_axis_cache: Optional[int] = None
        self._spectral_numpy_axis_cache: Optional[int] = None
        self._nan_high_order_fallback_used = False
        self._nan_high_order_downgraded = False

        # Sanitize the incoming WCS object, which may be inconsistent from fits_loader.py
        wcs_sanitized = copy.deepcopy(original_wcs)
        try:
            # Heuristic to detect if cunit is m/s while cdelt/crval are in km/s
            if wcs_sanitized.wcs.naxis >= 3:
                cunit = wcs_sanitized.wcs.cunit[2]
                cdelt = wcs_sanitized.wcs.cdelt[2]
                ctype = wcs_sanitized.wcs.ctype[2]
                is_velocity = any(token in ctype.upper() for token in ("VRAD", "VELO", "VOPT"))

                if is_velocity and cunit.is_equivalent(u.m / u.s) and abs(cdelt) < 100:
                    wcs_sanitized.wcs.cunit[2] = "km/s"
        except Exception as e:
            # Non-critical warning, can be logged if a logging system is available
            pass

        self.original_wcs = wcs_sanitized
        self.original_header = original_header.copy() if original_header is not None else None
        self._had_datamin = bool(self.original_header and "DATAMIN" in self.original_header)
        self._had_datamax = bool(self.original_header and "DATAMAX" in self.original_header)
        self._beam_metadata = self._extract_beam_metadata(self.original_header)
        original_bunit = self._beam_metadata.get("BUNIT", (None, None))[0]
        self._original_bunit_is_jybeam = self._is_jansky_per_beam(original_bunit)
        self._ensure_wcs_pc(self.original_wcs)
        self._ensure_wcs_units(self.original_wcs)
        self._original_celestial_matrix = self._celestial_cd_matrix(self.original_wcs)
        self._original_celestial_rotation = self._rotation_from_matrix(self._original_celestial_matrix)
        try:
            self.original_wcs.array_shape = tuple(int(dim) for dim in np.shape(self.original_data))
        except Exception:
            self.original_wcs.array_shape = None

    def _emit_progress(self, value: int) -> None:
        if self._progress_callback is not None:
            try:
                self._progress_callback(int(value))
            except Exception:
                pass

    def _emit_finished(self, data, header) -> None:
        if self._finished_callback is not None:
            try:
                self._finished_callback(data, header)
            except Exception:
                pass

    def _emit_error(self, message: str, exc: Optional[Exception] = None) -> None:
        if self._error_callback is not None:
            try:
                self._error_callback(message)
            except Exception:
                pass
        if exc is not None:
            raise exc
        raise RuntimeError(message)

    # ------------------------------------------------------------------
    def perform_regrid(self, params: Dict):
        """Entry point invoked from a worker thread."""
        try:
            mode = params.get("mode")
            interpolation_method = params.get("interpolation", "Bilinear")
            method = self._interpolation_to_method(interpolation_method)
            self._nan_high_order_fallback_used = False
            self._nan_high_order_downgraded = False
            self._emit_progress(10)

            should_trim_nan_edges = True
            new_wcs: Optional[WCS] = None

            if mode == "manual":
                # Always use the fast scipy-based method by default now.
                data, header, new_wcs = self._regrid_manual(params, method)
            elif mode == "template_fits":
                data, header, new_wcs = self._regrid_from_template_fits(params, method)
                should_trim_nan_edges = False
            elif mode == "reproject_system":
                # This mode still needs the high-precision reproject library.
                data, header, new_wcs = self._reproject_to_system(params, method)
            else:
                raise ValueError(f"Unknown regrid mode: {mode}")

            if should_trim_nan_edges:
                data, header = self._trim_nan_edges(data, header)
            self._update_data_extrema(header, data)
            self._finalize_header(
                header,
                data.shape,
                preserve_existing_units=(mode == "template_fits"),
            )

                    
            # Preserve beam metadata for non-manual modes by default
            preserve_opt = params.get("preserve_beam_metadata") if mode == "manual" else True
            beam_preserved = self._apply_beam_metadata(header, preserve_opt, new_wcs=new_wcs)
            self._annotate_history(header, mode, params, beam_preserved=beam_preserved)

            self._emit_progress(95)
            self._emit_finished(data, header)
            self._emit_progress(100)
            return data, header
        except Exception as exc:  # pragma: no cover - error path propagated to UI
            self._emit_error(str(exc), exc)

    # ------------------------------------------------------------------


    def _world_corners_for_axis(self, axis_index: int, data_shape: Sequence[int]) -> np.ndarray:
        """Return world coordinates for the first and last pixel along the given WCS axis."""
        naxis = self.original_wcs.wcs.naxis
        if axis_index < 0 or axis_index >= naxis:
            raise ValueError(f"WCS axis index {axis_index} is out of bounds for naxis={naxis}.")

        # Keep every axis at the reference pixel, then sweep the selected axis endpoints.
        reference_pix = np.asarray(self.original_wcs.wcs.crpix, dtype=float) - 1
        pixel_samples = np.tile(reference_pix, (2, 1))
        numpy_axis = naxis - 1 - axis_index
        if numpy_axis < 0 or numpy_axis >= len(data_shape):
            raise ValueError(f"Data shape does not cover WCS axis {axis_index + 1}.")

        axis_length = max(int(data_shape[numpy_axis]), 1)
        pixel_samples[0, axis_index] = 0
        pixel_samples[1, axis_index] = axis_length - 1

        world_samples = self.original_wcs.wcs_pix2world(pixel_samples, 0)
        return world_samples[:, axis_index]

    def _regrid_manual_legacy(self, params: Dict, method: str) -> Tuple[np.ndarray, fits.Header]:
        """
        Legacy manual regridding pipeline, rewritten for robustness.
        This version creates a clean WCS from scratch to avoid inconsistencies.
        """
        anchor_world = params["anchor_world"]
        grid_cdelt = params["grid_cdelt"]
        naxis = self.original_wcs.wcs.naxis
        spectral_wcs_idx = self._spectral_axis_index()

        if spectral_wcs_idx is None or self.original_wcs.wcs.naxis != 3:
            # This path is for non-spectral or non-3D data, which was not the issue.
            # We keep it for completeness, but the main logic is below.
            final_data, final_header, _ = self._regrid_manual_wcs_only(params, method)
            return final_data, final_header
        
        self._emit_progress(20)

        # --- 1. NEW: Build a clean, new 2D celestial WCS from scratch ---
        # This is the core of the fix. Instead of patching a copied WCS, we build a
        # new, internally consistent one from the user's parameters.
        wcs_2d_celestial = WCS(naxis=2)
        celestial_indices = self._celestial_axis_indices()
        
        # Determine the shape of the new 2D celestial grid
        shape_orig = self.original_data.shape
        shape_out_2d_list = [0, 0]

        for i_2d, i_nd in enumerate(celestial_indices):
            wcs_2d_celestial.wcs.crval[i_2d] = anchor_world[i_nd]
            wcs_2d_celestial.wcs.cdelt[i_2d] = grid_cdelt[i_nd]
            wcs_2d_celestial.wcs.ctype[i_2d] = self.original_wcs.wcs.ctype[i_nd]
            wcs_2d_celestial.wcs.cunit[i_2d] = self.original_wcs.wcs.cunit[i_nd]
            
            world_corners = self._world_corners_for_axis(i_nd, shape_orig)
            dist = np.max(np.abs(world_corners - anchor_world[i_nd]))
            new_size = int(np.ceil(dist * 2 / abs(grid_cdelt[i_nd])))
            new_size = max(new_size, 1)
            shape_out_2d_list[i_2d] = new_size
            wcs_2d_celestial.wcs.crpix[i_2d] = new_size / 2.0
        
        shape_out_2d = tuple(reversed(shape_out_2d_list))
        wcs_2d_celestial.wcs.set()
        
        # --- 2. NEW: Combine the new 2D WCS with the original spectral axis ---
        # Use the robust _build_combined_wcs helper to create a complete and correct 3D WCS.
        target_wcs_3d = self._build_combined_wcs(wcs_2d_celestial, celestial_indices)
        
        # --- 3. Perform plane-by-plane spatial reprojection ---
        n_spec_orig = shape_orig[0]
        source_wcs_2d = self._drop_axis_safe(self.original_wcs, spectral_wcs_idx)
        target_wcs_2d_final = self._drop_axis_safe(target_wcs_3d, spectral_wcs_idx)

        if source_wcs_2d is None or target_wcs_2d_final is None:
            raise RuntimeError("Failed to create 2D WCS for plane-by-plane reprojection.")

        shape_out_spatial = (n_spec_orig,) + shape_out_2d
        work_dtype = self._reproject_float_dtype()
        data_spatial_regridded = np.empty(shape_out_spatial, dtype=work_dtype)
        # Preflight mapping check
        try:
            ny, nx = shape_out_2d[0], shape_out_2d[1]
        except Exception:
            ny, nx = shape_out_2d
        preflight_center_mapping(source_wcs_2d, target_wcs_2d_final, ny, nx)

        for i in range(n_spec_orig):
            plane_data = self.original_data[i, :, :]
            if not isinstance(plane_data, np.ndarray):
                plane_data = np.asarray(plane_data)
            if plane_data.dtype != work_dtype:
                plane_data = plane_data.astype(work_dtype, copy=False)
            output_plane = np.empty(shape_out_2d, dtype=work_dtype)
            reprojected_plane, _, used_fix, downgraded = _reproject_with_nan_support(
                plane_data,
                source_wcs_2d,
                target_wcs_2d_final,
                shape_out_2d,
                method,
                work_dtype,
                output_array=output_plane,
                need_coverage=False,
            )
            if used_fix:
                self._nan_high_order_fallback_used = True
            if downgraded:
                self._nan_high_order_downgraded = True
            data_spatial_regridded[i, :, :] = reprojected_plane
        
        self._emit_progress(60)

        # --- 4. Perform 1D spectral interpolation (logic unchanged) ---
        # This part was already correct.
        spec_idx_numpy = naxis - 1 - spectral_wcs_idx
        crpix_orig = self.original_wcs.wcs.crpix
        crval_orig = self.original_wcs.wcs.crval
        cdelt_orig = self.original_wcs.wcs.cdelt
        cdelt_new = np.array(grid_cdelt)
        crval_new = np.array(anchor_world)

        n_spec_orig = shape_orig[spec_idx_numpy]
        crpix_spec_orig = crpix_orig[spectral_wcs_idx]
        crval_spec_orig = crval_orig[spectral_wcs_idx]
        cdelt_spec_orig = cdelt_orig[spectral_wcs_idx]
        cdelt_spec_new = cdelt_new[spectral_wcs_idx]
        crval_spec_new = crval_new[spectral_wcs_idx]
        
        orig_spec_coords = (np.arange(n_spec_orig) - (crpix_spec_orig - 1)) * cdelt_spec_orig + crval_spec_orig
        velocity_scale = self._spectral_velocity_scale_factor(spectral_wcs_idx, cdelt_spec_orig)
        if velocity_scale != 1.0:
            orig_spec_coords *= velocity_scale

        spectral_min, spectral_max = np.nanmin(orig_spec_coords), np.nanmax(orig_spec_coords)
        step = float(cdelt_spec_new)
        tolerance = max(abs(step) * 1e-8, 1e-9)
        if step > 0:
            k_min = int(math.ceil((spectral_min - crval_spec_new - tolerance) / step))
            k_max = int(math.floor((spectral_max - crval_spec_new + tolerance) / step))
        else:
            k_min = int(math.ceil((spectral_max - crval_spec_new + tolerance) / step))
            k_max = int(math.floor((spectral_min - crval_spec_new - tolerance) / step))
        
        k_indices = np.arange(k_min, k_max + 1)

        # SAFEGUARD: Check if the new spectral range is empty.
        if k_indices.size == 0:
            raise ValueError(
                "No spectral overlap found. The requested spectral range does not "
                "overlap with the original data's spectral range. "
                "No output channels could be generated."
            )
        n_spec_new = len(k_indices)
        crpix_spec_new = 1.0 - k_min
        new_spec_coords = crval_spec_new + k_indices * step

        data_reshaped = data_spatial_regridded.reshape(n_spec_orig, -1)
        interpolator = interp1d(
            orig_spec_coords, data_reshaped, axis=0, bounds_error=False,
            fill_value=np.nan, kind=method if method in ("nearest", "linear") else "linear",
            assume_sorted=True
        )
        regridded_spec_data = interpolator(new_spec_coords)
        
        final_shape = (n_spec_new,) + shape_out_2d
        final_data = regridded_spec_data.reshape(final_shape)
        preferred_dtype = self._preferred_float_dtype()
        if final_data.dtype != preferred_dtype:
            final_data = final_data.astype(preferred_dtype, copy=False)

        self._emit_progress(90)

        # --- 5. Construct Final Header using the corrected 3D WCS ---
        # Update the target_wcs_3d with the new spectral axis solution
        target_wcs_3d.wcs.crpix[spectral_wcs_idx] = crpix_spec_new
        target_wcs_3d.wcs.crval[spectral_wcs_idx] = crval_spec_new
        target_wcs_3d.wcs.cdelt[spectral_wcs_idx] = cdelt_spec_new
        target_wcs_3d.wcs.set()
        
        final_header = self._base_header_copy()
        self._apply_wcs_to_header(final_header, target_wcs_3d)
        final_header[f"NAXIS{spectral_wcs_idx+1}"] = n_spec_new

        self._ensure_header_pc(final_header)
        self._ensure_header_units(final_header)

        return final_data, final_header


    def _regrid_manual_wcs_only(self, params: Dict, method: str) -> Tuple[np.ndarray, fits.Header, WCS]:
        """Original WCS-based regridding for non-3D or non-spectral cubes."""
        anchor_world: Sequence[float] = params["anchor_world"]
        grid_cdelt: Sequence[float] = params["grid_cdelt"]
        naxis = self.original_wcs.wcs.naxis

        new_header = self._base_header_copy()
        self._apply_wcs_to_header(new_header, self.original_wcs)
        new_cdelt_arr = np.array(grid_cdelt, dtype=float)
        anchor_arr = np.array(anchor_world, dtype=float)
        if np.any(new_cdelt_arr == 0):
            raise ValueError("Grid width (world) must be non-zero for every axis.")

        axis_changed = [self._axis_requires_regrid(axis, anchor_arr[axis], new_cdelt_arr[axis]) for axis in range(naxis)]
        if not any(axis_changed):
            data_copy = self._copy_original_cube()
            header = self._base_header_copy()
            self._apply_wcs_to_header(header, self.original_wcs)
            self._ensure_header_pc(header)
            self._ensure_header_units(header)
            return data_copy, header, self.original_wcs.copy()

        for idx in range(naxis):
            new_header[f"CDELT{idx + 1}"] = new_cdelt_arr[idx]
            new_header[f"CRVAL{idx + 1}"] = anchor_arr[idx]
            
        data_shape = self._data_shape_for_wcs(naxis)


        for axis in range(naxis):
            world_corners = self._world_corners_for_axis(axis, data_shape)
            world_min, world_max = np.min(world_corners), np.max(world_corners)
            
            crval_new = anchor_arr[axis]
            cdelt_new = new_cdelt_arr[axis]
            
            step = float(cdelt_new)
            tolerance = max(abs(step) * 1e-8, 1e-9)

            if step > 0:
                k_min = int(math.ceil((world_min - crval_new - tolerance) / step))
                k_max = int(math.floor((world_max - crval_new + tolerance) / step))
            else:
                k_min = int(math.ceil((world_max - crval_new + tolerance) / step))
                k_max = int(math.floor((world_min - crval_new - tolerance) / step))

            if k_max < k_min:
                new_size = 1
                new_crpix = 1.0
            else:
                new_size = k_max - k_min + 1
                new_crpix = 1.0 - k_min
            
            new_header[f"NAXIS{axis + 1}"] = new_size
            new_header[f"CRPIX{axis + 1}"] = new_crpix

        new_header["NAXIS"] = naxis
        self._ensure_header_pc(new_header)
        self._ensure_header_units(new_header)

        target_wcs = self._create_wcs_safely(new_header)
        shape_out = tuple(int(new_header[f"NAXIS{i+1}"]) for i in reversed(range(new_header["NAXIS"])))

        regridded_data = self._reproject_to_target(
            target_wcs,
            shape_out,
            method,
        )
        
        header = self._base_header_copy()
        self._apply_wcs_to_header(header, target_wcs)
        self._ensure_header_pc(header)
        self._ensure_header_units(header)
        
        return regridded_data, header, target_wcs

    def _create_wcs_safely(self, header):
        try:
            return WCS(header)
        except Exception as e:
            if "Unmatched celestial axes" not in str(e):
                raise e

            h = header.copy()
            naxis = self._infer_wcs_axis_count(h)

            if naxis == 2:
                # This logic is borrowed from fits_loader.py to handle PV diagrams
                velocity_indices = []
                non_velocity_indices = []
                for i in range(1, 3):
                    ctype = h.get(f'CTYPE{i}', '').upper()
                    if 'VRAD' in ctype or 'VEL' in ctype or 'VOPT' in ctype:
                        velocity_indices.append(i)
                    else:
                        non_velocity_indices.append(i)
                
                if len(velocity_indices) == 1 and len(non_velocity_indices) == 1:
                    non_vel_idx = non_velocity_indices[0]
                    orig_ctype = h.get(f'CTYPE{non_vel_idx}', '')
                    if '-' in orig_ctype:
                        new_ctype = orig_ctype.split('-')[0]
                        h[f'CTYPE{non_vel_idx}'] = new_ctype
                        try:
                            return WCS(h)
                        except Exception:
                            pass

            if self._linearize_unmatched_celestial_axes(h):
                try:
                    return WCS(h)
                except Exception:
                    pass
            # If we get here the fallback failed as well, so re-raise the original error.
            raise e

    def _linearize_unmatched_celestial_axes(self, header: fits.Header) -> bool:
        """Fallback: strip projection codes from lone celestial axes."""
        axes = self._infer_wcs_axis_count(header)
        if axes <= 0:
            return False

        axis_info = []
        for axis in range(1, axes + 1):
            key = f'CTYPE{axis}'
            original = header.get(key, "")
            upper = (original or "").upper()
            base = upper.split('-')[0] if upper else ""
            axis_info.append((axis, original, base))

        def _linearize(axes_to_fix: Sequence[int]) -> bool:
            changed = False
            for axis_idx in axes_to_fix:
                key = f'CTYPE{axis_idx}'
                before = header.get(key, "")
                if not before:
                    continue
                after = before.split('-')[0].strip()
                if not after or after == before:
                    continue
                header[key] = after
                changed = True
            return changed

        def _filter(predicate):
            return [axis for axis, _, base in axis_info if predicate(base)]

        ra_like = _filter(lambda base: base.startswith("RA"))
        dec_like = _filter(lambda base: base.startswith("DEC"))
        lon_like = _filter(lambda base: base.endswith("LON"))
        lat_like = _filter(lambda base: base.endswith("LAT"))

        changed = False
        if ra_like and not dec_like:
            changed |= _linearize(ra_like)
        if dec_like and not ra_like:
            changed |= _linearize(dec_like)
        if lon_like and not lat_like:
            changed |= _linearize(lon_like)
        if lat_like and not lon_like:
            changed |= _linearize(lat_like)

        if not changed:
            fallback_axes = [
                axis
                for axis, _, base in axis_info
                if base.startswith("RA")
                or base.startswith("DEC")
                or base.endswith("LON")
                or base.endswith("LAT")
            ]
            changed |= _linearize(fallback_axes)

        return changed

    @staticmethod
    def _infer_wcs_axis_count(header: fits.Header) -> int:
        axes = header.get("WCSAXES")
        if axes is None:
            axes = header.get("NAXIS")
        if axes is None:
            axes = 0
            for key in header:
                if not key.startswith("CTYPE"):
                    continue
                suffix = key[5:]
                if suffix.isdigit():
                    axes = max(axes, int(suffix))
        try:
            return int(axes)
        except (TypeError, ValueError):
            return 0



    def _regrid_from_template_fits(self, params: Dict, method: str) -> Tuple[np.ndarray, fits.Header, WCS]:
        template_path = params.get("template_path")
        if not template_path:
            raise ValueError("Template FITS path is required for template regridding.")

        template_header, shape_out = self._load_template_header(template_path)
        self._harmonize_spectral_axis(template_header)
        self._ensure_header_units(
            template_header,
            preserve_original_cunit_presence=False,
        )
        self._ensure_header_pc(template_header)

        target_wcs = WCS(template_header)

        spectral_wcs_idx = self._spectral_axis_index()
        is_3d_cube = self.original_data.ndim > 2 and spectral_wcs_idx is not None

        if not is_3d_cube:
            # For 2D data, the original simple reprojection is correct.
            regridded_data = self._reproject_to_target(target_wcs, shape_out, method)
            return regridded_data, template_header, target_wcs

        self._emit_progress(20)

        # The output of this step will have the spatial grid of the template,
        # but will retain the original spectral axis.
        
        np_spectral_axis = self.original_data.ndim - 1 - spectral_wcs_idx
        n_spec_orig = self.original_data.shape[np_spectral_axis]

        source_wcs_2d = self._drop_axis_safe(self.original_wcs, spectral_wcs_idx)
        target_wcs_2d = self._drop_axis_safe(target_wcs, spectral_wcs_idx)

        if source_wcs_2d is None or target_wcs_2d is None:
            raise ValueError("Failed to create 2D WCS for plane-by-plane reprojection.")

        spatial_shape_out = list(shape_out)
        del spatial_shape_out[np_spectral_axis]
        spatial_shape_out = tuple(spatial_shape_out)

        intermediate_shape = list(self.original_data.shape)
        
        celestial_indices = self._celestial_axis_indices()
        numpy_celestial_axes = sorted([self.original_data.ndim - 1 - i for i in celestial_indices])
        
        intermediate_shape[numpy_celestial_axes[0]] = spatial_shape_out[0]
        intermediate_shape[numpy_celestial_axes[1]] = spatial_shape_out[1]
        
        work_dtype = self._reproject_float_dtype()
        data_spatial_regridded = np.empty(tuple(intermediate_shape), dtype=work_dtype)

        src_hdr_2d = source_wcs_2d.to_header()
        tgt_hdr_2d = target_wcs_2d.to_header()

        cpu_count = os.cpu_count() or 1
        max_workers = min(8, max(1, cpu_count))
        
        regridded_planes = [None] * n_spec_orig
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for i in range(n_spec_orig):
                slicer = [slice(None)] * self.original_data.ndim
                slicer[np_spectral_axis] = i
                plane_data = self.original_data[tuple(slicer)]
                plane_array = np.asarray(plane_data)

                fut = executor.submit(
                    _reproject_plane_worker,
                    plane_array,
                    src_hdr_2d,
                    tgt_hdr_2d,
                    tuple(spatial_shape_out),
                    method,
                    0.5,
                )
                futures[fut] = i

            completed_count = 0
            for fut in as_completed(futures):
                idx = futures[fut]
                reprojected_plane, used_fix, downgraded = fut.result()
                
                if used_fix:
                    self._nan_high_order_fallback_used = True
                if downgraded:
                    self._nan_high_order_downgraded = True
                
                slicer = [slice(None)] * data_spatial_regridded.ndim
                slicer[np_spectral_axis] = idx
                data_spatial_regridded[tuple(slicer)] = reprojected_plane
                
                completed_count += 1
                self._emit_progress(20 + int(40 * (completed_count / n_spec_orig)))

        self._emit_progress(60)
        
        # Get spectral coordinates in the same velocity unit. Astropy normalizes
        # km/s FITS velocity axes to m/s inside WCS, so prefer normalized header
        # values here to avoid unit-mismatched interpolation.
        orig_spec_coords = self._spectral_axis_coordinates(
            n_spec_orig,
            spectral_wcs_idx,
            header=self.original_header,
            wcs_obj=self.original_wcs,
        )

        n_spec_new = shape_out[np_spectral_axis]
        new_spec_coords = self._spectral_axis_coordinates(
            n_spec_new,
            spectral_wcs_idx,
            header=template_header,
            wcs_obj=target_wcs,
        )
        
        data_reshaped = np.moveaxis(data_spatial_regridded, np_spectral_axis, 0)
        original_shape = data_reshaped.shape
        data_reshaped = data_reshaped.reshape(n_spec_orig, -1)

        interpolator = interp1d(
            orig_spec_coords, data_reshaped, axis=0, bounds_error=False,
            fill_value=np.nan, kind=method if method in ("nearest", "linear") else "linear"
        )
        regridded_spec_data = interpolator(new_spec_coords)
        
        final_data_reshaped = regridded_spec_data.reshape((n_spec_new,) + original_shape[1:])
        final_data = np.moveaxis(final_data_reshaped, 0, np_spectral_axis)
        preferred_dtype = self._preferred_float_dtype()
        if final_data.dtype != preferred_dtype:
            final_data = final_data.astype(preferred_dtype, copy=False)
        
        self._emit_progress(90)
        
        return final_data, template_header, target_wcs


    def _load_template_header(self, template_path: str) -> Tuple[fits.Header, Tuple[int, ...]]: 
        with fits.open(template_path, memmap=False) as hdulist:
            image_hdu = None
            for hdu in hdulist:
                if hdu.data is not None:
                    image_hdu = hdu
                    break
            if image_hdu is None:
                raise ValueError("Template FITS does not contain an image HDU with data.")

            header = image_hdu.header.copy()
            if not header.get("NAXIS"):
                raise ValueError("Template FITS header is missing NAXIS information.")

            # This is the new line that fixes the unit scaling issue.
            self._normalize_template_velocity_axis(header)
            
            shape = tuple(int(dim) for dim in np.shape(image_hdu.data))
            if any(dim <= 0 for dim in shape):
                raise ValueError("Template FITS data has an invalid shape.")

            return header, shape

    def _normalize_template_velocity_axis(self, header: fits.Header):
        axes = self._header_axis_count(header)
        if axes <= 0:
            return

        for axis in range(1, axes + 1):
            ctype = header.get(f"CTYPE{axis}", "")
            if not self._is_velocity_axis(ctype):
                continue

            cdelt_key = f"CDELT{axis}"
            cdelt_value = header.get(cdelt_key)
            try:
                cdelt = float(cdelt_value)
            except (TypeError, ValueError):
                continue

            if not np.isfinite(cdelt):
                continue

            original_unit = header.get(f"CUNIT{axis}", "")
            should_scale = self._velocity_values_need_kms_scaling(
                original_unit,
                cdelt,
                assume_missing_unit=True,
            )
            if not should_scale and not self._velocity_unit_is_missing(original_unit):
                continue

            if should_scale:
                header[cdelt_key] = cdelt * 1e-3

                crval_key = f"CRVAL{axis}"
                if crval_key in header:
                    try:
                        header[crval_key] = float(header[crval_key]) * 1e-3
                    except (TypeError, ValueError):
                        pass

                axis_count = self._header_axis_count(header)
                for pixel_axis in range(1, axis_count + 1):
                    cd_key = f"CD{axis}_{pixel_axis}"
                    if cd_key in header:
                        try:
                            header[cd_key] = float(header[cd_key]) * 1e-3
                        except (TypeError, ValueError):
                            continue

            self._set_axis_unit(header, axis, "km/s")


    def _reproject_to_system(self, params: Dict, method: str) -> Tuple[np.ndarray, fits.Header, WCS]:

        target_system = params.get("target_system", "ICRS")
        frame_option = self._resolve_coordinate_frame(target_system)

        celestial_indices = self._celestial_axis_indices()
        if len(celestial_indices) != 2:
            raise ValueError(
                "Celestial reprojection requires exactly two celestial axes (e.g., RA/Dec)."
            )

        # Instead of manually dropping axes, we use the .celestial property, which is
        # the recommended way to get a 2D celestial-only WCS from a higher-dimensional one.
        # We pair it with the shape of the celestial axes.
        celestial_wcs = self.original_wcs.celestial
        celestial_shape = celestial_wcs.array_shape
        
        if celestial_shape is None or len(celestial_shape) != 2:
             # Fallback if celestial shape is not available from WCS object
            numpy_celestial_axes = sorted([self.original_data.ndim - 1 - i for i in celestial_indices])
            celestial_shape = (self.original_data.shape[numpy_celestial_axes[0]], self.original_data.shape[numpy_celestial_axes[1]])

        input_for_wcs_determination = [(celestial_shape, celestial_wcs)]
        
        target_wcs_celestial, shape_celestial = self._determine_target_wcs(
            input_for_wcs_determination, frame_option
        )
        spectral_wcs_idx = self._spectral_axis_index()
        is_3d_cube = self.original_data.ndim > 2 and spectral_wcs_idx is not None

        # Determine the input for shape calculation.
        # For 3D cubes, we use the SHAPE of a 2D slice, not the data itself.
        # This avoids both the "NaN latitude" and "not 2-dimensional" errors.
        if is_3d_cube:
            np_spectral_axis = self.original_data.ndim - 1 - spectral_wcs_idx
            
            # Get the shape of the spatial axes
            spatial_shape = list(self.original_data.shape)
            del spatial_shape[np_spectral_axis]
            
            wcs_2d_for_shape = self._drop_axis_safe(self.original_wcs, spectral_wcs_idx)
            if wcs_2d_for_shape is None:
                raise ValueError("Failed to create a 2D WCS for shape determination.")
            
            # Provide (shape, wcs) tuple, which is a valid input
            input_for_wcs_determination = [(tuple(spatial_shape), wcs_2d_for_shape)]
        else:
            # For 2D data, we can pass the data and WCS directly
            input_for_wcs_determination = [(self.original_data, self.original_wcs)]

        target_wcs_celestial, shape_celestial = self._determine_target_wcs(
            input_for_wcs_determination, frame_option
        )

        shape_out = shape_celestial
        target_wcs_full = target_wcs_celestial
        
        if is_3d_cube:
            target_wcs_full = self._build_combined_wcs(target_wcs_celestial, celestial_indices)
            final_shape = list(self.original_data.shape)
            numpy_celestial_axes = sorted([self.original_data.ndim - 1 - i for i in celestial_indices])
            if len(numpy_celestial_axes) == 2:
                final_shape[numpy_celestial_axes[0]] = shape_celestial[0]
                final_shape[numpy_celestial_axes[1]] = shape_celestial[1]
            shape_out = tuple(final_shape)

        if not is_3d_cube:
            regridded_data = self._reproject_to_target(target_wcs_full, shape_out, method)
        else:
            self._emit_progress(20)
            np_spectral_axis = self.original_data.ndim - 1 - spectral_wcs_idx
            n_planes = self.original_data.shape[np_spectral_axis]

            # 2D WCSs for reprojection of each spatial plane
            source_wcs_2d = self._drop_axis_safe(self.original_wcs, spectral_wcs_idx)
            target_wcs_2d = self._drop_axis_safe(target_wcs_full, spectral_wcs_idx)

            # Output shape for 2D plane
            shape_out_2d = list(shape_out)
            del shape_out_2d[np_spectral_axis]
            shape_out_2d = tuple(shape_out_2d)

            # Serialize WCS to headers for cross-process safety
            src_hdr_2d = source_wcs_2d.to_header()
            tgt_hdr_2d = target_wcs_2d.to_header()

            cpu_count = os.cpu_count() or 1
            max_workers = min(8, max(1, cpu_count))
            progress_start, progress_end = 30, 90
            regridded_planes = [None] * n_planes

            # Submit jobs per plane (no shared WCS objects; headers only)
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                for i in range(n_planes):
                    slicer = [slice(None)] * self.original_data.ndim
                    slicer[np_spectral_axis] = i
                    plane_data = self.original_data[tuple(slicer)]
                    plane_array = np.asarray(plane_data)

                    fut = executor.submit(
                        _reproject_plane_worker,
                        plane_array,  # copied into worker; stable
                        src_hdr_2d,
                        tgt_hdr_2d,
                        shape_out_2d,
                        method,
                        0.5,                     # footprint threshold (tunable)
                    )
                    futures[fut] = i

                completed_count = 0
                for fut in as_completed(futures):
                    idx = futures[fut]
                    plane_result, used_fix, downgraded = fut.result()
                    regridded_planes[idx] = plane_result
                    if used_fix:
                        self._nan_high_order_fallback_used = True
                    if downgraded:
                        self._nan_high_order_downgraded = True
                    completed_count += 1
                    ratio = completed_count / n_planes
                    self._emit_progress(int(progress_start + (progress_end - progress_start) * ratio))

            # Stack planes back along the spectral axis
            regridded_data = np.stack(regridded_planes, axis=np_spectral_axis)
            preferred_dtype = self._preferred_float_dtype()
            if regridded_data.dtype != preferred_dtype:
                regridded_data = regridded_data.astype(preferred_dtype, copy=False)
            self._emit_progress(90)

        header = self._base_header_copy()
        self._apply_wcs_to_header(header, target_wcs_full)
        self._ensure_header_pc(header)
        self._ensure_header_units(header)
        
        return regridded_data, header, target_wcs_full


    def _determine_target_wcs(self, datasets, frame_option):
        try:
            return determine_optimal_celestial_wcs(datasets, frame=frame_option)
        except TypeError:
            return determine_optimal_celestial_wcs(datasets, target_frame=frame_option)

    def _interpolation_to_method(self, label: str) -> str:
        mapping = {
            "nearest-neighbor": "nearest-neighbor",
            "nearest": "nearest-neighbor",
            "bilinear": "bilinear",
            "biquadratic": "biquadratic",
            "bicubic": "bicubic",
        }
        return mapping.get(label.lower(), "bilinear")

    def _data_shape_for_wcs(self, naxis: int) -> Tuple[int, ...]:
        data_shape = self.original_data.shape
        if len(data_shape) == naxis:
            return data_shape
        return data_shape[-naxis:]

    def _resolve_coordinate_frame(self, target_system: str):
        system = target_system.strip().lower()
        obstime = parse_obstime_from_header(self.original_header)
        self._last_frame_hint = None

        if system == "galactic":
            self._last_frame_hint = None
            return Galactic()
        if system == "fk5":
            self._last_frame_hint = "fk5"
            return FK5(equinox="J2000")
        self._last_frame_hint = "icrs"
        return ICRS()

    # ------------------------------------------------------------------
    # Header helpers
    def _base_header_copy(self) -> fits.Header:
        if self.original_header is not None:
            return self.original_header.copy()
        return fits.Header()

    @staticmethod
    def _first_commentary_keyword(header: fits.Header) -> Optional[str]:
        for card in header.cards:
            if card.keyword in {"HISTORY", "COMMENT"}:
                return card.keyword
        return None

    @classmethod
    def _set_header_value_before_commentary(
        cls,
        header: fits.Header,
        key: str,
        value,
        comment=None,
        *,
        after: Optional[str] = None,
    ):
        if key in header:
            header[key] = value
            if comment not in (None, ""):
                try:
                    header.comments[key] = comment
                except Exception:
                    pass
            return

        commentary_key = cls._first_commentary_keyword(header)
        if after and after in header:
            try:
                after_index = header.index(after)
                commentary_index = header.index(commentary_key) if commentary_key else None
                if commentary_index is None or after_index < commentary_index:
                    header.set(key, value, comment=comment, after=after)
                    return
            except Exception:
                pass

        if commentary_key:
            header.set(key, value, comment=comment, before=commentary_key)
        else:
            header.set(key, value, comment=comment)

    def _set_axis_unit(self, header: fits.Header, axis_number: int, unit_value):
        key = f"CUNIT{axis_number}"
        anchors = (
            f"CDELT{axis_number}",
            f"CRVAL{axis_number}",
            f"CRPIX{axis_number}",
            f"CTYPE{axis_number}",
            f"NAXIS{axis_number}",
        )
        after = next((anchor for anchor in anchors if anchor in header), None)
        self._set_header_value_before_commentary(header, key, unit_value, after=after)

    def _apply_wcs_to_header(self, header: fits.Header, wcs_obj: WCS, *, frame_hint: str | None = None):
        # Define a comprehensive list of WCS-related keyword prefixes to remove.
        # This ensures a clean slate before writing the new WCS.
        WCS_KEYWORD_PREFIXES = [
            "CTYPE", "CRVAL", "CRPIX", "CDELT", "CUNIT", "CROTA",
            "PC", "CD", "PV", "PS",
            "WCSAXES", "LONPOLE", "LATPOLE", "RADESYS", "EQUINOX"
        ]

        # First, aggressively remove all potentially conflicting old WCS keywords.
        for key in list(header.keys()):
            for prefix in WCS_KEYWORD_PREFIXES:
                if key.startswith(prefix):
                    try:
                        # Remove the keyword if it matches any prefix.
                        del header[key]
                    except KeyError:
                        # This can happen in rare cases, safe to ignore.
                        pass
                    break  # Move to the next header key once a prefix is matched.

        # Now that the header is clean, apply the new WCS information.
        wcs_header = wcs_obj.to_header(relax=True)

        # The to_header() can sometimes include blank keys, which are invalid.
        if '' in wcs_header:
            del wcs_header['']

        last_inserted_key = None
        for card in wcs_header.cards:
            if not card.keyword:
                continue
            self._set_header_value_before_commentary(
                header,
                card.keyword,
                card.value,
                card.comment,
                after=last_inserted_key,
            )
            last_inserted_key = card.keyword

        # Ensure WCS is initialized
        try:
            wcs_obj.wcs.set()
        except Exception:
            pass

        # Resolve intended frame
        frame_name = (frame_hint or self._last_frame_hint or "").lower() if (frame_hint or self._last_frame_hint) else None
        if not frame_name:
            try:
                wcs_for_detect = getattr(wcs_obj, "celestial", wcs_obj)
                frame = wcs_to_celestial_frame(wcs_for_detect)
                frame_name = getattr(frame, "name", None)
                frame_name = frame_name.lower() if frame_name else None
            except Exception:
                frame_name = None

        if not frame_name:
            radesys = str(header.get("RADESYS","")).strip().upper()
            equinox = header.get("EQUINOX", None)
            ctype1  = str(header.get("CTYPE1","")).upper()
            if radesys in ("FK4","FK5","ICRS"):
                frame_name = radesys.lower()
            elif "RA---" in ctype1 or "DEC--" in ctype1:
                try:
                    eq = float(equinox) if equinox is not None else float("nan")
                except Exception:
                    eq = float("nan")
                frame_name = "fk4" if (eq == eq and eq < 1984) else "fk5"
            else:
                frame_name = None

        # Coerce celestial CTYPE/units for FK4/FK5
        def _coerce_radec(header, ra: bool, ctype_val: str):
            s = (ctype_val or "").strip().upper()
            proj = "TAN"
            if "-" in s:
                proj = s.split("-", 1)[-1][:3]
            return ("RA---" if ra else "DEC--") + proj

        if frame_name in ("fk4","fk5"):
            ctype1 = str(header.get("CTYPE1","")).upper()
            ctype2 = str(header.get("CTYPE2","")).upper()
            if not ctype1.startswith("RA---"):
                header["CTYPE1"] = _coerce_radec(header, True, ctype1)
            if not ctype2.startswith("DEC--"):
                header["CTYPE2"] = _coerce_radec(header, False, ctype2)
            self._set_axis_unit(header, 1, "deg")
            self._set_axis_unit(header, 2, "deg")

        # Enforce RADESYS/EQUINOX/EPOCH
        if frame_name == "fk4":
            header["RADESYS"] = "FK4"
            header["EQUINOX"] = 1950.0
            header["EPOCH"]   = 1950.0
        elif frame_name == "fk5":
            header["RADESYS"] = "FK5"
            header["EQUINOX"] = 2000.0
            if "EPOCH" in header:
                del header["EPOCH"]
        elif frame_name == "icrs":
            header["RADESYS"] = "ICRS"
            if "EQUINOX" in header:
                del header["EQUINOX"]
            if "EPOCH" in header:
                del header["EPOCH"]

        return header

    def _finalize_header(
        self,
        header: fits.Header,
        data_shape: Tuple[int, ...],
        *,
        preserve_existing_units: bool = False,
    ):
        header["NAXIS"] = len(data_shape)
        for idx in range(len(data_shape)):
            axis_number = idx + 1
            data_axis = len(data_shape) - axis_number
            header[f"NAXIS{axis_number}"] = int(data_shape[data_axis])
            unit_key = f"CUNIT{axis_number}"
            if preserve_existing_units and header.get(unit_key):
                continue
            original_unit = ""
            if self.original_header is not None:
                original_unit = self.original_header.get(unit_key, "")
            self._set_axis_unit(header, axis_number, original_unit)

    def _annotate_history(
        self,
        header: fits.Header,
        mode: str,
        params: Dict,
        *,
        beam_preserved: bool = False,
    ):
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        interpolation = params.get("interpolation", "unknown")
        header.add_history(f"Regrid executed by takefits on {timestamp} UTC")

        source_file = self.filename
        if not source_file:
            if self.original_header and 'FILENAME' in self.original_header:
                source_file = self.original_header['FILENAME']

        if source_file:
            # FITS headers must be ASCII. Sanitize the path.
            safe_source = source_file.encode("ascii", "replace").decode("ascii")
            header.add_history(f"Source file: {safe_source}")

        header.add_history(f"Mode: {mode}, Interpolation: {interpolation}")

        if mode == "manual":
            anchors = params.get("anchor_world", [])
            spacings = params.get("grid_cdelt", [])
            anchor_str = ", ".join(f"{val:.6f}" for val in anchors)
            spacing_str = ", ".join(f"{val:.6f}" for val in spacings)
            header.add_history(f"Anchor world coords (deg): {anchor_str}")
            header.add_history(f"Grid spacing (deg): {spacing_str}")
        elif mode == "template_fits":
            template_path = params.get("template_path", "")
            if template_path:
                # Sanitize template path as well
                safe_template = template_path.encode("ascii", "replace").decode("ascii")
                template_name = os.path.basename(safe_template)
            else:
                template_name = 'N/A'
                
            header.add_history(f"Template applied: {template_name}")
        elif mode == "reproject_system":
            header.add_history(f"Target frame: {params.get('target_system', 'Unknown')}")
        if beam_preserved:
            header.add_history("Preserved original beam metadata (BUNIT/BMAJ/BMIN/BPA).")
        if self._nan_high_order_fallback_used:
            header.add_history("High-order interpolation normalized with finite-weight mask due to NaN input pixels.")
        if self._nan_high_order_downgraded:
            header.add_history("High-order interpolation degraded to bilinear where normalization still yielded all-NaN output.")

    def _update_data_extrema(self, header: fits.Header, data: np.ndarray):
        if not isinstance(header, fits.Header):
            return

        update_min = "DATAMIN" in header or self._had_datamin
        update_max = "DATAMAX" in header or self._had_datamax

        if not (update_min or update_max):
            return
            
        if not isinstance(data, np.ndarray) or data.size == 0:
            return

        finite_mask = np.isfinite(data)
        if not np.any(finite_mask):
            if update_min:
                self._remove_header_key(header, "DATAMIN")
            if update_max:
                self._remove_header_key(header, "DATAMAX")
            return

        finite_values = np.asarray(data[finite_mask], dtype=float)
        if finite_values.size == 0:
            if update_min:
                self._remove_header_key(header, "DATAMIN")
            if update_max:
                self._remove_header_key(header, "DATAMAX")
            return

        if update_min:
            header["DATAMIN"] = float(np.min(finite_values))
        if update_max:
            header["DATAMAX"] = float(np.max(finite_values))

    @staticmethod
    def _remove_header_key(header: fits.Header, key: str):
        if key not in header:
            return
        try:
            header.remove(key, remove_all=True, ignore_missing=True)
        except Exception:
            del header[key]

    # ------------------------------------------------------------------
    # WCS helpers
    def _ensure_wcs_pc(self, wcs_obj: WCS):
        """Ensures the WCS object has a PC matrix, defaulting to identity."""
        try:
            # The only way to know if pc is missing is to try to access it.
            # If it exists, we don't need to do anything, unless it's None.
            if wcs_obj.wcs.pc is not None:
                return
        except AttributeError:
            # It's missing, so we'll add it below.
            pass

        # If we're here, pc is missing or is None.
        try:
            naxis = wcs_obj.wcs.naxis
            if naxis > 0:
                wcs_obj.wcs.pc = np.identity(naxis)
        except Exception:  # pragma: no cover
            # In case naxis is not available or other issues.
            pass

    def _celestial_axis_indices(self) -> List[int]:
        indices = []
        for idx, ctype in enumerate(self.original_wcs.wcs.ctype):
            upper = (ctype or "").upper()
            if "RA" in upper or "DEC" in upper or "GLON" in upper or "GLAT" in upper or "ELON" in upper or "ELAT" in upper:
                indices.append(idx)
        return indices[:2]

    def _celestial_axis_indices_from_header(self, header: fits.Header) -> List[int]:
        axes = header.get("WCSAXES", header.get("NAXIS", 0))
        indices: List[int] = []
        for idx in range(axes):
            ctype = (header.get(f"CTYPE{idx + 1}", "") or "").upper()
            if "RA" in ctype or "DEC" in ctype or "GLON" in ctype or "GLAT" in ctype or "ELON" in ctype or "ELAT" in ctype:
                indices.append(idx)
        return indices[:2]

    def _spectral_axis_index(self) -> Optional[int]:
        if self._spectral_axis_cache is not None:
            return self._spectral_axis_cache
        for idx, ctype in enumerate(self.original_wcs.wcs.ctype):
            upper = (ctype or "").upper()
            if any(token in upper for token in ("VRAD", "VELO", "FREQ", "VOPT", "WAVE")):
                self._spectral_axis_cache = idx
                return idx
        self._spectral_axis_cache = None
        return self._spectral_axis_cache

    def _spectral_numpy_axis(self) -> Optional[int]:
        if self._spectral_numpy_axis_cache is not None:
            return self._spectral_numpy_axis_cache
        spectral_wcs_idx = self._spectral_axis_index()
        if spectral_wcs_idx is None:
            self._spectral_numpy_axis_cache = None
            return None
        np_axis = self.original_data.ndim - 1 - spectral_wcs_idx
        self._spectral_numpy_axis_cache = np_axis
        return np_axis

    def _spectral_axis_coordinates(
        self,
        size: int,
        axis_index: int,
        *,
        header: Optional[fits.Header],
        wcs_obj: WCS,
    ) -> np.ndarray:
        axis_number = axis_index + 1
        coords: Optional[np.ndarray] = None
        cdelt_value: Optional[float] = None
        unit_value: object = None
        ctype_value = ""

        if header is not None:
            try:
                crpix = float(header.get(f"CRPIX{axis_number}"))
                crval = float(header.get(f"CRVAL{axis_number}"))
                cdelt = float(header.get(f"CDELT{axis_number}"))
                coords = (np.arange(size, dtype=float) - (crpix - 1.0)) * cdelt + crval
                cdelt_value = cdelt
                unit_value = header.get(f"CUNIT{axis_number}")
                ctype_value = header.get(f"CTYPE{axis_number}", "")
            except (TypeError, ValueError):
                coords = None

        if coords is None:
            crpix = float(wcs_obj.wcs.crpix[axis_index])
            crval = float(wcs_obj.wcs.crval[axis_index])
            cdelt = float(wcs_obj.wcs.cdelt[axis_index])
            coords = (np.arange(size, dtype=float) - (crpix - 1.0)) * cdelt + crval
            cdelt_value = cdelt
            if axis_index < len(wcs_obj.wcs.cunit):
                unit_value = wcs_obj.wcs.cunit[axis_index]
            if axis_index < len(wcs_obj.wcs.ctype):
                ctype_value = wcs_obj.wcs.ctype[axis_index]

        if self._is_velocity_axis(str(ctype_value)):
            if self._velocity_values_need_kms_scaling(
                unit_value,
                float(cdelt_value or 0.0),
                assume_missing_unit=True,
            ):
                coords = coords * 1e-3

        return coords

    def _get_spectral_slice(self, index: int) -> np.ndarray:
        if index < 0:
            raise IndexError("Spectral slice index must be non-negative.")
        np_axis = self._spectral_numpy_axis()
        if np_axis is None:
            raise RuntimeError("Spectral axis is undefined for this dataset.")

        with self._slice_cache_lock:
            cached = self._slice_cache.get(index)
            if cached is not None:
                self._slice_cache.move_to_end(index)
                return cached

        slice_data: np.ndarray
        if self._dask_data is not None:
            try:
                slice_data = np.asarray(self._dask_data.take(index, axis=np_axis).compute())
            except Exception:
                slice_data = np.take(self.original_data, index, axis=np_axis)
        else:
            slice_data = np.take(self.original_data, index, axis=np_axis)

        slice_data = np.asarray(slice_data, dtype=self._preferred_float_dtype())

        with self._slice_cache_lock:
            self._slice_cache[index] = slice_data
            if len(self._slice_cache) > self._max_cached_slices:
                self._slice_cache.popitem(last=False)
        return slice_data

    def _spectral_plane_at(
        self,
        world_value: float,
        method: str,
        orig_spec_coords: np.ndarray,
    ) -> Optional[np.ndarray]:
        coords = np.asarray(orig_spec_coords, dtype=float)
        if coords.size == 0:
            return None

        interp_mode = method.lower()
        if interp_mode in ("nearest", "nearest-neighbor"):
            nearest_idx = int(np.argmin(np.abs(coords - world_value)))
            return np.array(self._get_spectral_slice(nearest_idx), copy=True)

        increasing = coords[-1] >= coords[0]
        coords_for_search = coords if increasing else coords[::-1]
        idx = int(np.searchsorted(coords_for_search, world_value))
        size = coords_for_search.size
        if size == 0:
            return None

        if increasing:
            first_val = float(coords_for_search[0])
            last_val = float(coords_for_search[-1])
        else:
            first_val = float(coords_for_search[-1])
            last_val = float(coords_for_search[0])

        if self._almost_equal(world_value, first_val):
            target_idx = 0 if increasing else size - 1
            return np.array(self._get_spectral_slice(target_idx), copy=True)
        if self._almost_equal(world_value, last_val):
            target_idx = size - 1 if increasing else 0
            return np.array(self._get_spectral_slice(target_idx), copy=True)

        if idx <= 0 or idx >= size:
            return None

        lower_idx = idx - 1
        upper_idx = idx

        if not increasing:
            lower_idx = size - idx
            upper_idx = size - idx - 1

        x0 = float(coords_for_search[idx - 1])
        x1 = float(coords_for_search[idx])

        if self._almost_equal(world_value, x0):
            target_idx = lower_idx
            return np.array(self._get_spectral_slice(target_idx), copy=True)
        if self._almost_equal(world_value, x1):
            target_idx = upper_idx
            return np.array(self._get_spectral_slice(target_idx), copy=True)

        if not np.isfinite(x0) or not np.isfinite(x1) or math.isclose(x0, x1, rel_tol=1e-12, abs_tol=1e-12):
            nearest_idx = lower_idx if abs(world_value - x0) <= abs(world_value - x1) else upper_idx
            return np.array(self._get_spectral_slice(nearest_idx), copy=True)

        weight = (world_value - x0) / (x1 - x0)
        weight = min(max(weight, 0.0), 1.0)
        slice_lo = self._get_spectral_slice(lower_idx)
        slice_hi = self._get_spectral_slice(upper_idx)
        dtype = self._preferred_float_dtype()
        plane = (1.0 - weight) * slice_lo + weight * slice_hi
        if plane.dtype != dtype:
            plane = plane.astype(dtype, copy=False)
        return plane

    def _interpolation_to_spline_order(self, method: str) -> Optional[int]:
        order_map = {
            "nearest": 0,
            "nearest-neighbor": 0,
            "bilinear": 1,
            "biquadratic": 2,
            "bicubic": 3,
        }
        return order_map.get(method.lower())

    def _reproject_2d_plane(
        self,
        plane: Optional[np.ndarray],
        source_wcs: Optional[WCS],
        target_wcs: Optional[WCS],
        shape_out: Tuple[int, int],
        order: str,
    ) -> np.ndarray:
        preferred_dtype = self._preferred_float_dtype()
        work_dtype = self._reproject_float_dtype()
        if plane is None or source_wcs is None or target_wcs is None:
            return np.full(shape_out, np.nan, dtype=preferred_dtype)

        # Ensure the source plane matches the dtype we will hand to reproject_interp.
        if not isinstance(plane, np.ndarray):
            plane = np.asarray(plane)
        if plane.dtype != work_dtype:
            plane = plane.astype(work_dtype, copy=False)

        output_array = np.empty(shape_out, dtype=work_dtype)
        result, _, used_fix, downgraded = _reproject_with_nan_support(
            plane,
            source_wcs,
            target_wcs,
            shape_out,
            order,
            work_dtype,
            output_array=output_array,
            need_coverage=False,
        )
        if used_fix:
            self._nan_high_order_fallback_used = True
        if downgraded:
            self._nan_high_order_downgraded = True
        if preferred_dtype != work_dtype:
            result = result.astype(preferred_dtype, copy=False)
        return result

    class _SpatialMapping(NamedTuple):
        coords: Optional[np.ndarray]
        valid_mask: Optional[np.ndarray]
        output_shape: Tuple[int, int]
        integer_shift: Optional[Tuple[int, int]]

    def _prepare_spatial_resampler(
        self,
        source_wcs: WCS,
        target_wcs: WCS,
        output_shape: Tuple[int, int],
        source_shape: Tuple[int, int],
        unit_step_axes: Optional[Tuple[bool, ...]] = None,
    ) -> Optional["_SpatialMapping"]:
        if map_coordinates is None and not (unit_step_axes and any(unit_step_axes)):
            return None
        if len(output_shape) != 2 or len(source_shape) != 2:
            return None

        try:
            y_grid, x_grid = np.indices(output_shape, dtype=float)
            target_pixels = np.column_stack((x_grid.ravel(), y_grid.ravel()))
            world_coords = target_wcs.wcs_pix2world(target_pixels, 0)
            source_pixels = source_wcs.wcs_world2pix(world_coords, 0)
        except Exception:
            return None

        if not isinstance(source_pixels, np.ndarray):
            source_pixels = np.asarray(source_pixels, dtype=float)
        source_x = np.asarray(source_pixels[:, 0], dtype=float)
        source_y = np.asarray(source_pixels[:, 1], dtype=float)

        valid = np.isfinite(source_x) & np.isfinite(source_y)
        height, width = int(source_shape[0]), int(source_shape[1])
        valid &= (source_x >= -0.5) & (source_x <= width - 0.5)
        valid &= (source_y >= -0.5) & (source_y <= height - 0.5)
        valid = valid.astype(bool, copy=False)

        safe_x = np.where(valid, source_x, 0.0)
        safe_y = np.where(valid, source_y, 0.0)
        coords = np.vstack((safe_y, safe_x))
        integer_shift: Optional[Tuple[int, int]] = None

        def _detect_shift(deltas: np.ndarray) -> Optional[int]:
            if deltas.size == 0:
                return None
            base = float(deltas[0])
            if np.max(np.abs(deltas - base)) > 1e-4:
                return None
            nearest = int(round(base))
            if abs(base - nearest) > 1e-4:
                return None
            return nearest

        if unit_step_axes and valid.any():
            shifts: List[Optional[int]] = [None, None]
            diffs_x = source_x - x_grid.ravel()
            diffs_y = source_y - y_grid.ravel()
            valid_indices = valid

            if len(unit_step_axes) > 0 and unit_step_axes[0]:
                shift_x = _detect_shift(diffs_x[valid_indices])
                if shift_x is not None:
                    shifts[1] = shift_x
            if len(unit_step_axes) > 1 and unit_step_axes[1]:
                shift_y = _detect_shift(diffs_y[valid_indices])
                if shift_y is not None:
                    shifts[0] = shift_y

            if all(shift is not None for shift in shifts):
                integer_shift = (int(shifts[0]), int(shifts[1]))

        return self._SpatialMapping(
            coords=coords,
            valid_mask=valid,
            output_shape=(int(output_shape[0]), int(output_shape[1])),
            integer_shift=integer_shift,
        )

    def _apply_spatial_resampler(
        self,
        plane: np.ndarray,
        mapping: "_SpatialMapping",
        order: int,
        dtype: np.dtype,
        source_shape: Tuple[int, int],
        ) -> np.ndarray:
        if map_coordinates is None:
            raise RuntimeError("Spatial resampler requested but scipy.ndimage is unavailable.")

        coords = mapping.coords
        valid_flat = mapping.valid_mask
        output_shape = mapping.output_shape

        if coords is None:
            return np.full(output_shape, np.nan, dtype=dtype)

        if plane.shape != tuple(int(dim) for dim in source_shape):
            plane = np.reshape(plane, tuple(int(dim) for dim in source_shape))

        # Handle NaNs correctly for high-order interpolation
        
        nan_mask = np.isnan(plane)
        filled_plane = plane
        
        has_nans = False
        prefilter = bool(order > 1)
        if prefilter and np.any(nan_mask):
            has_nans = True
            # Use nan_to_num which replaces NaN with 0.0
            filled_plane = np.nan_to_num(plane)

        # We use cval=0.0 because any out-of-bounds will be masked
        # to NaN by the reprojected masks later.
        sampled = map_coordinates(
            filled_plane,
            coords,
            order=int(order),
            mode="constant",
            cval=0.0, 
            prefilter=prefilter,
        )

        if has_nans:
            # Resample the NaN mask (True=NaN, False=Valid)
            # We sample nan_mask.astype(float) (1.0=NaN, 0.0=Valid)
            # order=0 (nearest), mode='constant', cval=1.0 (out-of-bounds is invalid/NaN)
            nan_mask_sampled = map_coordinates(
                nan_mask.astype(float),
                coords,
                order=0,
                mode="constant",
                cval=1.0, 
            )
            # Apply the NaN mask
            sampled[nan_mask_sampled > 0.5] = np.nan
        
        reshaped = sampled.reshape(output_shape)

        # This mask handles pixels that mapped outside the original image bounds
        # *regardless* of whether they were NaN or not.
        if valid_flat is not None and valid_flat.size == sampled.size:
            invalid_mask = ~valid_flat.reshape(output_shape)
            if np.any(invalid_mask):
                reshaped[invalid_mask] = np.nan

        if reshaped.dtype != dtype:
            reshaped = reshaped.astype(dtype, copy=False)
        return reshaped

    @staticmethod
    def _compute_shift_overlap(
        source_len: int,
        output_len: int,
        shift: int,
    ) -> Tuple[int, int, int]:
        if shift >= 0:
            src_start = shift
            dst_start = 0
        else:
            src_start = 0
            dst_start = -shift
        available_src = source_len - src_start
        available_dst = output_len - dst_start
        length = min(available_src, available_dst)
        if length <= 0:
            return src_start, dst_start, 0
        return src_start, dst_start, length

    def _apply_integer_shift(
        self,
        plane: np.ndarray,
        shift: Tuple[int, int],
        dtype: np.dtype,
        source_shape: Tuple[int, int],
        output_shape: Tuple[int, int],
    ) -> np.ndarray:
        if plane.shape != tuple(int(dim) for dim in source_shape):
            plane = np.reshape(plane, tuple(int(dim) for dim in source_shape))

        shift_y, shift_x = int(shift[0]), int(shift[1])
        out = np.full(output_shape, np.nan, dtype=dtype)

        src_y_start, dst_y_start, y_len = self._compute_shift_overlap(
            source_shape[0],
            output_shape[0],
            shift_y,
        )
        src_x_start, dst_x_start, x_len = self._compute_shift_overlap(
            source_shape[1],
            output_shape[1],
            shift_x,
        )

        if y_len <= 0 or x_len <= 0:
            return out

        out[
            dst_y_start : dst_y_start + y_len,
            dst_x_start : dst_x_start + x_len,
        ] = plane[
            src_y_start : src_y_start + y_len,
            src_x_start : src_x_start + x_len,
        ]
        return out

    def _build_combined_wcs(self, target_wcs2d: WCS, celestial_indices: List[int]) -> WCS:
        """
        Builds a full-dimensional WCS by combining a new 2D celestial WCS with
        the non-celestial axes of the original WCS.

        This robust implementation builds a new WCS object from scratch to ensure
        internal consistency, avoiding the pitfalls of modifying a copied WCS.
        """
        original_wcs = self.original_wcs
        naxis = original_wcs.wcs.naxis

        final_wcs = WCS(naxis=naxis)

        target_w = target_wcs2d.wcs
        for i_2d, i_nd in enumerate(celestial_indices):
            final_wcs.wcs.crpix[i_nd] = target_w.crpix[i_2d]
            final_wcs.wcs.crval[i_nd] = target_w.crval[i_2d]
            final_wcs.wcs.ctype[i_nd] = target_w.ctype[i_2d]
            final_wcs.wcs.cunit[i_nd] = target_w.cunit[i_2d]
            final_wcs.wcs.cdelt[i_nd] = target_w.cdelt[i_2d]

        non_celestial_indices = [i for i in range(naxis) if i not in celestial_indices]
        for i_nd in non_celestial_indices:
            final_wcs.wcs.crpix[i_nd] = original_wcs.wcs.crpix[i_nd]
            final_wcs.wcs.crval[i_nd] = original_wcs.wcs.crval[i_nd]
            final_wcs.wcs.ctype[i_nd] = original_wcs.wcs.ctype[i_nd]
            final_wcs.wcs.cunit[i_nd] = original_wcs.wcs.cunit[i_nd]
            final_wcs.wcs.cdelt[i_nd] = original_wcs.wcs.cdelt[i_nd]

        final_pc = np.identity(naxis)
        target_pc = target_wcs2d.wcs.get_pc()

        for i_2d, i_nd_row in enumerate(celestial_indices):
            for j_2d, j_nd_col in enumerate(celestial_indices):
                final_pc[i_nd_row, j_nd_col] = target_pc[i_2d, j_2d]
        
        final_wcs.wcs.pc = final_pc
        
        if hasattr(target_w, 'get_pv'):
            pv_list = target_w.get_pv()
            new_pv_list = []
            for i_2d, m, value in pv_list:
                # Map the 2D axis index to the N-D axis index
                axis_in_nd = celestial_indices[i_2d]
                new_pv_list.append((axis_in_nd, m, value))
            if new_pv_list:
                final_wcs.wcs.set_pv(new_pv_list)
        
        final_wcs.wcs.set()
        
        self._ensure_wcs_units(final_wcs)
        return final_wcs

    def _spectral_velocity_scale_factor(
        self,
        spectral_axis_index: int,
        cdelt_spec_orig: float,
    ) -> float:
        try:
            unit_value = (
                self.original_wcs.wcs.cunit[spectral_axis_index]
                if spectral_axis_index < len(self.original_wcs.wcs.cunit)
                else None
            )
            if self._velocity_values_need_kms_scaling(
                unit_value,
                cdelt_spec_orig,
                assume_missing_unit=False,
            ):
                return 1e-3
        except Exception:
            return 1.0
        return 1.0


    def _harmonize_spectral_axis(self, template_header: fits.Header):
        original_ctypes = list(self.original_wcs.wcs.ctype)
        template_axes = template_header.get(
            "WCSAXES", template_header.get("NAXIS", len(original_ctypes))
        )
        template_ctypes = [
            template_header.get(f"CTYPE{i}", "") for i in range(1, template_axes + 1)
        ]
        for idx, original_type in enumerate(original_ctypes):
            if idx >= len(template_ctypes):
                continue
            template_type = template_ctypes[idx]
            if self._is_velocity_axis(original_type) and self._is_velocity_axis(template_type):
                template_header[f"CTYPE{idx + 1}"] = original_type
                orig_unit = ""
                if idx < len(self.original_wcs.wcs.cunit):
                    orig_unit = self._sanitize_unit_value(
                        self.original_wcs.wcs.cunit[idx],
                        is_velocity=True,
                    )
                if orig_unit:
                    self._set_axis_unit(template_header, idx + 1, orig_unit)

        self._synchronize_rest_metadata(template_header)

    def _synchronize_rest_metadata(self, header: fits.Header):
        rest_frequency = self._resolve_rest_frequency()
        rest_wavelength = self._resolve_rest_wavelength()

        self._apply_rest_value(header, ("RESTFRQ", "RESTFREQ"), rest_frequency)
        self._apply_rest_value(header, ("RESTWAV", "RESTWAVE", "RESTWVL"), rest_wavelength)

    def _resolve_rest_frequency(self) -> Optional[float]:
        candidates = [
            getattr(self.original_wcs.wcs, "restfrq", None),
        ]
        if self.original_header is not None:
            for key in ("RESTFRQ", "RESTFREQ"):
                candidates.append(self.original_header.get(key))
        return self._first_positive_float(candidates, unit=u.Hz)

    def _resolve_rest_wavelength(self) -> Optional[float]:
        candidates = [
            getattr(self.original_wcs.wcs, "restwav", None),
        ]
        if self.original_header is not None:
            for key in ("RESTWAV", "RESTWAVE", "RESTWVL"):
                candidates.append(self.original_header.get(key))
        return self._first_positive_float(candidates, unit=u.m)

    def _apply_rest_value(self, header: fits.Header, keys: Sequence[str], value: Optional[float]):
        if value is not None:
            for key in keys:
                header[key] = float(value)
            return
        for key in keys:
            if key in header:
                try:
                    header.remove(key, remove_all=True, ignore_missing=True)
                except Exception:
                    del header[key]

    def _first_positive_float(
        self, values: Sequence[object], *, unit: Optional[u.UnitBase] = None
    ) -> Optional[float]:
        for raw in values:
            value = self._coerce_positive_float(raw, unit=unit)
            if value is not None:
                return value
        return None

    def _coerce_positive_float(
        self, value: object, *, unit: Optional[u.UnitBase] = None
    ) -> Optional[float]:
        if value is None:
            return None
        try:
            if isinstance(value, u.Quantity):
                if unit is not None:
                    numeric = value.to_value(unit)
                else:
                    numeric = value.to_value(value.unit)
            else:
                numeric = float(value)
        except Exception:
            return None
        if not np.isfinite(numeric) or numeric <= 0:
            return None
        return float(numeric)

    def _coerce_float_value(
        self, value: object, *, unit: Optional[u.UnitBase] = None
    ) -> Optional[float]:
        if value is None:
            return None
        try:
            if isinstance(value, u.Quantity):
                if unit is not None:
                    numeric = value.to_value(unit)
                else:
                    numeric = value.to_value(value.unit)
            else:
                numeric = float(value)
        except Exception:
            return None
        if not np.isfinite(numeric):
            return None
        return float(numeric)


    def _celestial_cd_matrix(self, wcs_obj: WCS) -> Optional[np.ndarray]:
        """
        Robustly extracts the celestial transformation matrix from a WCS object.
        
        This method uses wcs.celestial.pixel_scale_matrix, which is the
        recommended astropy approach to get the matrix that correctly describes
        the on-sky pixel orientation, regardless of whether the WCS uses
        PC, CD, or CROTA keywords.
        """
        try:
            # astropyから最も信頼性の高い方法で celestial matrix を取得します
            matrix = wcs_obj.celestial.pixel_scale_matrix
            
            # FIX: The block that modified the matrix based on its determinant was
            # incorrect. The matrix returned by pixel_scale_matrix from astropy
            # already has the correct parity and orientation for the given WCS.
            # Manually "fixing" it was the likely cause of the 180-degree rotation.
            # We should trust the matrix as-is.
            #
            # The incorrect block has been removed.
            
            return np.array(matrix, dtype=float)
            
        except Exception:
            return None


    def _compute_celestial_rotation_delta(self, header: fits.Header, new_wcs: Optional[WCS] = None) -> Optional[float]:
        """
        Calculates the true rotation angle between the original celestial system's
        North direction and the new celestial system's North direction at the
        image center. This is the value that must be added to the original BPA.
        """
        # Import necessary components within the function for robustness
        from astropy import units as u
        from astropy.wcs.utils import wcs_to_celestial_frame

        try:
            if new_wcs is None:
                new_wcs = WCS(header)

            # 1. Get the center of the image in the original coordinate system.
            original_frame = wcs_to_celestial_frame(self.original_wcs)
            # Use pixel_to_world for robustness instead of relying on crval directly
            center_pix_x = self.original_wcs.wcs.crpix[0] -1
            center_pix_y = self.original_wcs.wcs.crpix[1] - 1
            p1 = self.original_wcs.celestial.pixel_to_world(center_pix_x, center_pix_y)

            # 2. Define a point (P2) slightly "north" in the original system.
            # This version is more robust against internal representation changes.
            p2_coords = self.original_wcs.celestial.pixel_to_world(center_pix_x, center_pix_y + 1) # 1 pixel north

            # 3. Transform both points into the new coordinate system.
            new_frame = wcs_to_celestial_frame(new_wcs)
            p1_new = p1.transform_to(new_frame)
            p2_new = p2_coords.transform_to(new_frame)

            # 4. The position angle of the transformed vector IS the true rotation angle.
            delta = p1_new.position_angle(p2_new).deg
            
            return self._wrap_angle_degrees(delta)

        except Exception:
            return 0.0 # Return 0.0 on failure to avoid crashing

    @staticmethod
    def _rotation_from_matrix(matrix: Optional[np.ndarray]) -> Optional[float]:
        if matrix is None:
            return None
        if matrix.shape[0] < 2 or matrix.shape[1] < 2:
            return None
        
        # Use matrix elements with explicit indexing
        m10 = matrix[1, 0]
        m00 = matrix[0, 0]
        
        angle = math.degrees(math.atan2(m10, m00))
        
        if not np.isfinite(angle):
            return None
            
        return float(angle)


    @staticmethod
    def _wrap_angle_degrees(angle: float) -> float:
        if not np.isfinite(angle):
            return angle
        wrapped = (angle + 180.0) % 360.0
        if wrapped < 0:
            wrapped += 360.0
        return wrapped - 180.0

    @staticmethod
    def _normalize_beam_position_angle(angle: float, prefer_nonnegative: bool) -> float:
        if not np.isfinite(angle):
            return angle
        if prefer_nonnegative:
            angle = angle % 180.0
            if angle < 0:
                angle += 180.0
            return angle
        return RegridEngine._wrap_angle_degrees(angle)

    def _extract_beam_metadata(
        self, header: Optional[fits.Header]
    ) -> Dict[str, Tuple[object, Optional[str]]]:
        result: Dict[str, Tuple[object, Optional[str]]] = {}
        if header is None:
            return result
        for key in ("BUNIT", "BMAJ", "BMIN", "BPA", "BTYPE"):
            if key in header:
                try:
                    comment = header.comments[key]
                except Exception:
                    comment = None
                result[key] = (header[key], comment)
        return result

    def _set_header_value(
        self,
        header: fits.Header,
        key: str,
        value: object,
        comment: Optional[str],
    ) -> bool:
        if value is None:
            return False
        existing = header.get(key)
        same_value = False
        if isinstance(value, (int, float)) and isinstance(existing, (int, float, np.floating)):
            same_value = math.isclose(float(existing), float(value), rel_tol=1e-9, abs_tol=1e-12)
        else:
            same_value = existing == value
        same_comment = True
        if comment is not None and key in header:
            try:
                same_comment = header.comments[key] == comment
            except Exception:
                same_comment = False
        if same_value and same_comment:
            return False
        if comment is not None:
            header.set(key, value, comment=comment)
        else:
            header[key] = value
        return True

    def _apply_beam_metadata(
        self,
        header: fits.Header,
        preserve_option: Optional[bool] = None,
        new_wcs: Optional[WCS] = None,
    ) -> bool:
        if not isinstance(header, fits.Header):
            return False
        if not getattr(self, "_beam_metadata", None):
            return False

        beam_meta = self._beam_metadata

        preserve_beam = preserve_option
        if preserve_beam is None:
            preserve_beam = self._original_bunit_is_jybeam
        if not preserve_beam:
            return False

        changed = False
        original_bunit_value, original_bunit_comment = beam_meta.get("BUNIT", (None, None))
        if original_bunit_value is not None:
            current_bunit = header.get("BUNIT")
            original_is_jybeam = self._is_jansky_per_beam(original_bunit_value)
            should_override_bunit = current_bunit is None
            if not should_override_bunit and original_is_jybeam:
                should_override_bunit = not self._is_jansky_per_beam(current_bunit)
            if should_override_bunit:
                if self._set_header_value(header, "BUNIT", original_bunit_value, original_bunit_comment):
                    changed = True

        for key in ("BMAJ", "BMIN"):
            value, comment = beam_meta.get(key, (None, None))
            numeric = self._coerce_positive_float(value, unit=u.deg)
            if numeric is None:
                numeric = self._coerce_positive_float(value)
            if numeric is None:
                continue
            if self._set_header_value(header, key, numeric, comment):
                changed = True

        value, comment = beam_meta.get("BPA", (None, None))
        angle = self._coerce_float_value(value, unit=u.deg)
        if angle is None:
            angle = self._coerce_float_value(value)
        # Adjust BPA by WCS rotation difference between original and new header
        if angle is not None:
            prefer_nonnegative = 0.0 <= angle <= 180.0
            rotation_delta = self._compute_celestial_rotation_delta(header, new_wcs=new_wcs)
            if rotation_delta is not None:
                angle = angle + rotation_delta
                angle = self._normalize_beam_position_angle(angle, prefer_nonnegative)
            if self._set_header_value(header, "BPA", angle, comment):
                changed = True

        value, comment = beam_meta.get("BTYPE", (None, None))
        if value is not None:
            if self._set_header_value(header, "BTYPE", value, comment):
                changed = True

        return changed

    @staticmethod
    def _is_jansky_per_beam(value: object) -> bool:
        if value is None:
            return False
        text = str(value).strip().lower()
        if not text:
            return False
        normalized = text.replace(" ", "")
        return ("jy" in normalized or "jansky" in normalized) and "beam" in normalized

    @staticmethod
    def _header_axis_count(header: fits.Header) -> int:
        axes = header.get("WCSAXES", header.get("NAXIS", 0))
        try:
            return max(int(axes), 0)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _looks_like_meter_per_second_unit(value: object) -> bool:
        if value is None:
            return False
        text = str(value).strip().lower().replace(" ", "")
        return text in {"m/s", "ms-1", "meter/second", "metre/second"}

    @staticmethod
    def _velocity_unit_is_missing(value: object) -> bool:
        return value is None or str(value).strip() == ""

    @classmethod
    def _velocity_values_need_kms_scaling(
        cls,
        unit_value: object,
        cdelt_value: float,
        *,
        assume_missing_unit: bool,
    ) -> bool:
        try:
            cdelt = float(cdelt_value)
        except (TypeError, ValueError):
            return False
        if not np.isfinite(cdelt) or abs(cdelt) < 100.0:
            return False
        return cls._looks_like_meter_per_second_unit(unit_value) or (
            assume_missing_unit and cls._velocity_unit_is_missing(unit_value)
        )

    @staticmethod
    def _is_velocity_axis(ctype: str) -> bool:
        upper = (ctype or "").upper()
        return any(token in upper for token in ("VRAD", "VELO", "VOPT"))

    def _ensure_header_units(
        self,
        header: fits.Header,
        *,
        preserve_original_cunit_presence: bool = True,
    ):
        original_ctypes = list(self.original_wcs.wcs.ctype)
        original_cunits = list(self.original_wcs.wcs.cunit)
        axes = header.get("WCSAXES", header.get("NAXIS", len(original_cunits)))

        original_cunits_present = {}
        if self.original_header:
            # Check for CUNIT presence in the original header
            original_naxis = self.original_header.get("NAXIS", 0)
            for i in range(original_naxis):
                original_cunits_present[i] = f"CUNIT{i+1}" in self.original_header

        for idx in range(axes):
            key = f"CUNIT{idx + 1}"

            # If original header exists and CUNIT was not present for this axis, ensure it's not in the new header.
            # The default for get is False, meaning if the axis didn't exist in original, we treat it as "no CUNIT".
            if (
                preserve_original_cunit_presence
                and self.original_header
                and not original_cunits_present.get(idx, False)
            ):
                if key in header:
                    del header[key]
                continue

            ctype = header.get(f"CTYPE{idx + 1}", "")
            is_velocity = self._is_velocity_axis(ctype) or (
                idx < len(original_ctypes) and self._is_velocity_axis(original_ctypes[idx])
            )
            current = header.get(key)
            sanitized = self._sanitize_unit_value(current, is_velocity=is_velocity)
            
            if not sanitized and idx < len(original_cunits) and original_cunits[idx]:
                sanitized = self._sanitize_unit_value(
                    original_cunits[idx], is_velocity=is_velocity
                )
            
            if not sanitized and is_velocity:
                # Only add default "km/s" if we don't have an original header to check against,
                # or if the original header HAD a CUNIT for this axis.
                if not self.original_header or original_cunits_present.get(idx, False):
                     sanitized = "km/s"

            if sanitized:
                self._set_axis_unit(header, idx + 1, sanitized)
            elif key in header:
                # If sanitization results in empty string, remove the key if it exists.
                del header[key]

    def _sanitize_unit_value(self, value, *, is_velocity: bool = False) -> str:
        if value in ("", None):
            return "km/s" if is_velocity else ""
        if isinstance(value, u.UnitBase):
            unit = value
        else:
            text = str(value).strip()
            if text.upper().startswith("UNIT("):
                first = text.find('"')
                last = text.rfind('"')
                if first != -1 and last != -1 and last > first:
                    text = text[first + 1 : last]
            try:
                unit = u.Unit(text)
            except (ValueError, TypeError):
                return "km/s" if is_velocity else ""
        
        if is_velocity:
            if unit.is_equivalent(u.m / u.s):
                try:
                    meters_per_second = (1.0 * unit).to(u.m / u.s).value
                    if np.isclose(meters_per_second, 1000.0):
                        return "km/s"
                    if np.isclose(meters_per_second, 1.0):
                        return "m/s"
                except Exception:
                    pass
                try:
                    unit_text = unit.to_string(format="fits").replace(" ", "")
                    return unit_text or "km/s"
                except Exception:
                    return "km/s"
        
        try:
            return unit.to_string(format="fits").replace(" ", "")
        except Exception:
            return "km/s" if is_velocity else ""


    def _ensure_header_pc(self, header: fits.Header):
        axes = header.get("WCSAXES", header.get("NAXIS", len(self.original_wcs.wcs.ctype)))
        header["WCSAXES"] = axes
        has_cd = any(header.get(f"CD{i}_{j}") is not None for i in range(1, axes + 1) for j in range(1, axes + 1))
        if has_cd:
            return
        for i in range(1, axes + 1):
            for j in range(1, axes + 1):
                key = f"PC{i}_{j}"
                if key not in header:
                    header[key] = 1.0 if i == j else 0.0

    def _ensure_wcs_units(self, wcs_obj: WCS):
        ctype_list = list(wcs_obj.wcs.ctype)
        for idx in range(wcs_obj.wcs.naxis):
            ctype = ctype_list[idx] if idx < len(ctype_list) else ""
            is_velocity = self._is_velocity_axis(ctype)
            current = wcs_obj.wcs.cunit[idx] if idx < len(wcs_obj.wcs.cunit) else ""
            sanitized = self._sanitize_unit_value(current, is_velocity=is_velocity)
            if not sanitized and is_velocity:
                sanitized = "km/s"
            wcs_obj.wcs.cunit[idx] = sanitized or ""

    def _preferred_float_dtype(self) -> np.dtype:
        data = self.original_data
        dtype = getattr(data, "dtype", None)
        if dtype is not None:
            if np.issubdtype(dtype, np.floating):
                return np.dtype("float32") if dtype.itemsize <= 4 else np.dtype("float64")
            if np.issubdtype(dtype, np.integer):
                return np.dtype("float32")
        return np.dtype("float32")

    @staticmethod
    def _reproject_float_dtype() -> np.dtype:
        """Data type expected by reproject (maps to np.float64)."""
        return np.dtype("float64")

    @staticmethod
    def _almost_equal(a: float, b: float, tol: float = 1e-9) -> bool:
        if a == b:
            return True
        if not (np.isfinite(a) and np.isfinite(b)):
            return False
        scale = max(1.0, abs(a), abs(b))
        return abs(a - b) <= tol * scale

    def _axis_requires_regrid(self, axis: int, crval_new: float, cdelt_new: float) -> bool:
        try:
            orig_crval = float(self.original_wcs.wcs.crval[axis])
            orig_cdelt = float(self.original_wcs.wcs.cdelt[axis])
        except Exception:
            return True
        if not self._almost_equal(orig_crval, float(crval_new)):
            return True
        if not self._almost_equal(orig_cdelt, float(cdelt_new)):
            return True
        return False

    def _copy_original_cube(self) -> np.ndarray:
        preferred_dtype = self._preferred_float_dtype()
        if self._dask_data is not None:
            try:
                data_np = np.asarray(self._dask_data.compute(), dtype=preferred_dtype)
            except Exception:
                data_np = np.asarray(self.original_data, dtype=preferred_dtype)
        else:
            data_np = np.asarray(self.original_data, dtype=preferred_dtype)
        return np.array(data_np, copy=True, dtype=preferred_dtype)

    def _maybe_wrap_data(self, data):
        if da is None:
            return None
        if isinstance(data, np.ndarray):
            try:
                chunks = self._suggest_chunks(data.shape)
                return da.from_array(data, chunks=chunks, asarray=False)
            except Exception:
                return None
        if hasattr(data, "chunks"):
            try:
                return da.asarray(data)
            except Exception:
                return None
        return None

    @staticmethod
    def _suggest_chunks(shape: Sequence[int]) -> Tuple[int, ...]:
        if not shape:
            return tuple()
        chunks = []
        for idx, axis_len in enumerate(shape):
            if axis_len <= 0:
                chunks.append(1)
                continue
            if idx == 0:
                chunk = min(max(1, axis_len // 16), 64)
            else:
                chunk = axis_len
            chunks.append(int(chunk))
        return tuple(chunks)



    @staticmethod
    def _drop_axis_safe(wcs_obj: WCS, axis: int) -> Optional[WCS]:
        try:
            dropped = wcs_obj.dropaxis(axis)
        except Exception:
            return None
        try:
            if wcs_obj.array_shape:
                dropped.array_shape = tuple(
                    int(dim) for idx, dim in enumerate(wcs_obj.array_shape) if idx != axis
                )
        except Exception:
            dropped.array_shape = None
        return dropped

    def _celestial_has_projection(self, wcs_obj: WCS) -> bool:
        """Checks if the celestial axes have a projection code (e.g., -SIN, -TAN)."""
        celestial_indices = self._celestial_axis_indices()
        if not celestial_indices:
            return False

        for i in celestial_indices:
            try:
                ctype = wcs_obj.wcs.ctype[i]
                if '-' in ctype:
                    return True
            except (IndexError, AttributeError):
                continue
        return False

    def _is_axis_coupled(self, wcs_obj: WCS, axis: int, tol: float = 1e-10) -> bool:
        try:
            pc_matrix = wcs_obj.wcs.get_pc()
        except Exception:
            pc_matrix = None
        if pc_matrix is not None:
            for idx in range(pc_matrix.shape[0]):
                if idx == axis:
                    continue
                if abs(pc_matrix[axis, idx]) > tol or abs(pc_matrix[idx, axis]) > tol:
                    return True

        cd_matrix = self._extract_cd_matrix(wcs_obj.wcs)
        if cd_matrix is not None:
            for idx in range(cd_matrix.shape[0]):
                if idx == axis:
                    continue
                if abs(cd_matrix[axis, idx]) > tol or abs(cd_matrix[idx, axis]) > tol:
                    return True
        return False

    def _allocate_output_array(self, shape_out: Tuple[int, ...]) -> np.ndarray:
        dtype = self._preferred_float_dtype()
        normalized_shape = tuple(int(dim) for dim in shape_out)
        return np.empty(normalized_shape, dtype=dtype, order="C")

    def _reproject_to_target(self, target_projection, shape_out: Tuple[int, ...], order: str) -> np.ndarray:
        normalized_shape = tuple(int(dim) for dim in shape_out) if shape_out is not None else None
        output_array = None
        work_dtype = self._reproject_float_dtype()
        preferred_dtype = self._preferred_float_dtype()
        source_data = self.original_data
        if not isinstance(source_data, np.ndarray):
            source_data = np.asarray(source_data)
        if source_data.dtype != work_dtype:
            source_data = source_data.astype(work_dtype, copy=False)
        if normalized_shape is not None:
            output_array = np.empty(normalized_shape, dtype=work_dtype, order="C")
            result, _, used_fix, downgraded = _reproject_with_nan_support(
                source_data,
                self.original_wcs,
                target_projection,
                normalized_shape,
                order,
                work_dtype,
                output_array=output_array,
                need_coverage=False,
            )
        else:
            result = reproject_interp(
                (source_data, self.original_wcs),
                target_projection,
                shape_out=None,
                order=order,
                output_array=None,
                return_footprint=False,
            )
            used_fix = False
            downgraded = False
        if used_fix:
            self._nan_high_order_fallback_used = True
        if downgraded:
            self._nan_high_order_downgraded = True
        if preferred_dtype != work_dtype:
            result = np.asarray(result, dtype=preferred_dtype)
        return result

    def _trim_nan_edges(self, data: np.ndarray, header: fits.Header) -> Tuple[np.ndarray, fits.Header]:
        if not isinstance(data, np.ndarray) or data.size == 0:
            return data, header
        if not np.issubdtype(data.dtype, np.floating):
            data = data.astype(self._preferred_float_dtype(), copy=False)
        if not np.isnan(data).any():
            return data, header

        axes_to_check = self._celestial_axis_indices_from_header(header)
        if not axes_to_check:
            axes_to_check = self._celestial_axis_indices()
        if not axes_to_check:
            return data, header

        for wcs_axis in sorted(set(axes_to_check)):
            np_axis = data.ndim - 1 - wcs_axis
            if np_axis < 0 or np_axis >= data.ndim:
                continue

            nan_mask = np.isnan(data)
            axis_nan = nan_mask.all(axis=tuple(ax for ax in range(data.ndim) if ax != np_axis))
            valid = np.flatnonzero(~axis_nan)
            if valid.size == 0:
                continue
            start = int(valid[0])
            end = int(valid[-1]) + 1

            if start == 0 and end == data.shape[np_axis]:
                continue

            slicer = [slice(None)] * data.ndim
            slicer[np_axis] = slice(start, end)
            data = data[tuple(slicer)]

            crpix_key = f"CRPIX{wcs_axis + 1}"
            try:
                crpix_value = float(header.get(crpix_key, 1.0))
            except (TypeError, ValueError):
                crpix_value = 1.0
            header[crpix_key] = crpix_value - start

            naxis_key = f"NAXIS{wcs_axis + 1}"
            header[naxis_key] = data.shape[data.ndim - 1 - wcs_axis]

            wcs_axes = header.get("WCSAXES")
            if wcs_axes is None:
                header["WCSAXES"] = data.ndim
            else:
                header["WCSAXES"] = max(int(wcs_axes), data.ndim)

        return data, header

    def _extract_cd_matrix(self, wcsprm) -> Optional[np.ndarray]:
        # Build a full CD matrix for any NAXIS using the FITS-WCS rule:
        #   CD_ij = CDELT_i * PC_ij  (i.e., CD = diag(CDELT) @ PC)
        # Prefer an existing CD if it is present and valid; otherwise synthesize.
        try:
            # 1) Use existing CD if it is a proper 2D square matrix with data
            cd = getattr(wcsprm, "cd", None)
            if cd is not None:
                arr = np.asarray(cd, dtype=float)
                if arr.ndim == 2 and arr.shape[0] == arr.shape[1] and arr.size > 0 and np.any(arr):
                    return arr
            # Fall through to synthesize from PC & CDELT
        except Exception:
            # If anything odd happens, just synthesize below.
            pass

        # 2) Determine dimensionality
        naxis = getattr(wcsprm, "naxis", None)
        if not naxis or naxis <= 0:
            return None

        # 3) Get PC matrix; fallback to identity if missing/bad shape
        try:
            pc = wcsprm.get_pc()
        except Exception:
            pc = None
        if pc is None or np.size(pc) == 0:
            pc_mat = np.eye(naxis, dtype=float)
        else:
            pc = np.asarray(pc, dtype=float)
            if pc.ndim != 2:
                pc_mat = np.eye(naxis, dtype=float)
            elif pc.shape != (naxis, naxis):
                # Pad/trim to square (naxis x naxis) if needed
                pc_mat = np.eye(naxis, dtype=float)
                m = min(pc.shape[0], naxis)
                n = min(pc.shape[1], naxis)
                pc_mat[:m, :n] = pc[:m, :n]
            else:
                pc_mat = pc

        # 4) Get CDELT vector; fallback to ones if missing/short
        try:
            cdelt = np.asarray(getattr(wcsprm, "cdelt", None), dtype=float)
        except Exception:
            cdelt = None
        if cdelt is None or cdelt.size == 0:
            cdelt_vec = np.ones(naxis, dtype=float)
        else:
            cdelt_vec = np.ones(naxis, dtype=float)
            m = min(cdelt.size, naxis)
            cdelt_vec[:m] = cdelt[:m]

        # 5) Synthesize CD = diag(CDELT) @ PC
        cd_mat = np.diag(cdelt_vec) @ pc_mat
        return cd_mat



    def _regrid_manual(self, params: Dict, method: str) -> Tuple[np.ndarray, fits.Header, WCS]:
        """
        Performs a fast, memory-efficient, and parallelized manual regridding.
        It processes the data plane-by-plane to avoid large memory allocations
        and uses a thread pool to accelerate the computation on multi-core CPUs.
        """
        if map_coordinates is None:
            raise RuntimeError("The SciPy library is required for high-speed resampling.")

        self._emit_progress(20)

        target_header, target_wcs = self._calculate_manual_target_wcs(params)
        shape_out = tuple(int(target_header[f"NAXIS{i+1}"]) for i in reversed(range(target_header["NAXIS"])))

        # Handle NaNs for high-order spline interpolation
        order = self._interpolation_to_spline_order(method) or 1
        prefilter = order > 1
        
        # Prepare data: fill NaNs if necessary for pre-filtering
        input_data = self.original_data
        nan_mask = None
        filled_data = input_data
        has_nans = False

        # Only create filled data if pre-filtering is active and NaNs are present
        if prefilter:
            try:
                # Check for NaNs. This might be slow on large dask arrays
                # but is necessary for correctness.
                if np.isnan(np.asarray(input_data)).any():
                    has_nans = True
                    nan_mask = np.isnan(input_data)
                    # Replace NaNs with 0.0 for spline filtering
                    filled_data = np.nan_to_num(input_data) 
            except Exception:
                # Fallback for complex/unsupported data types
                pass 

        if len(shape_out) < 3:
            # 2D case
            grids = np.indices(shape_out, dtype=np.float32)
            pixel_coords_out = np.vstack([g.ravel() for g in reversed(grids)]).T
            world_coords = target_wcs.wcs_pix2world(pixel_coords_out, 0)
            source_pixel_coords_flat = self.original_wcs.wcs_world2pix(world_coords, 0)
            source_pixel_coords_for_scipy = source_pixel_coords_flat.T[::-1]

            # Use filled_data for interpolation
            regridded_data_flat = map_coordinates(
                filled_data, source_pixel_coords_for_scipy,
                order=order, mode='nearest', prefilter=prefilter
            )
            
            # Remap the NaN mask if we had NaNs
            if has_nans and nan_mask is not None:
                # Use order=0 (nearest) for mask reprojection
                regridded_mask_flat = map_coordinates(
                    nan_mask.astype(float), source_pixel_coords_for_scipy,
                    order=0, mode='nearest'
                )
                # Apply the mask
                regridded_data_flat[regridded_mask_flat > 0.5] = np.nan
            
            regridded_data = regridded_data_flat.reshape(shape_out)
        else:
            # 3D case
            self._emit_progress(30)
            regridded_data = np.empty(shape_out, dtype=self._preferred_float_dtype())
            
            spectral_wcs_idx = self._spectral_axis_index()
            if spectral_wcs_idx is None:
                spectral_wcs_idx = self.original_wcs.wcs.naxis - 1

            # Note: order and prefilter already defined above
            numpy_spectral_axis = len(shape_out) - 1 - spectral_wcs_idx
            
            spatial_shape = list(shape_out)
            del spatial_shape[numpy_spectral_axis]
            spatial_grids = np.indices(spatial_shape, dtype=np.float32)

            # 3D case optimization:
            # Check if spectral axis is coupled to spatial axes.
            spectral_coupled = False
            for ax_idx in range(self.original_wcs.wcs.naxis):
                if ax_idx == spectral_wcs_idx:
                    continue
                if self._is_axis_coupled(self.original_wcs, spectral_wcs_idx, tol=1e-6):
                    spectral_coupled = True
                    break
            
            # If target WCS has coupling, we might also need to be careful, 
            # but usually for 'manual' we are building a separable grid. 
            # We assume target Grid is separable by construction in _calculate_manual_target_wcs.

            # Pre-calculate spatial coordinates if uncoupled
            precomputed_spatial_coords_scipy = None
            
            if not spectral_coupled:
                # Calculate spatial coordinates for a single plane (z=0)
                # The spatial part of the grid is constant for all z if uncoupled.
                
                # Create a grid for z=0
                coords_for_wcs_axes = [None] * target_wcs.wcs.naxis
                # Set spectral pixel to 0 (arbitrary, as it shouldn't affect spatial lookups if uncoupled)
                coords_for_wcs_axes[spectral_wcs_idx] = np.zeros(spatial_shape, dtype=np.float32)
                
                spatial_wcs_indices = [i for i in range(target_wcs.wcs.naxis) if i != spectral_wcs_idx]
                coords_for_wcs_axes[spatial_wcs_indices[0]] = spatial_grids[1] 
                coords_for_wcs_axes[spatial_wcs_indices[1]] = spatial_grids[0]
                
                pixel_coords_out = np.vstack([c.ravel() for c in coords_for_wcs_axes]).T
                world_coords = target_wcs.wcs_pix2world(pixel_coords_out, 0)
                
                # We need full world2pix to get all pixel coordinates
                source_pixel_coords_flat = self.original_wcs.wcs_world2pix(world_coords, 0)
                
                # Extract spatial components (scipy uses [z, y, x] order, so we need y, x here)
                # The source_pixel_coords_flat is (N, 3) with [x, y, z] order in WCS convention
                # We want [y, x] for the spatial part of map_coordinates input.
                # Recall: map_coordinates input is (ndim, n_points). 
                # For 3D it expects [z, y, x] coords.
                
                # Let's verify standard numpy/scipy ordering: 
                # data[z, y, x] -> coordinates should be [z_coords, y_coords, x_coords]
                # data axis 0: spectral (numpy_spectral_axis)
                # data axis 1: y
                # data axis 2: x
                
                # source_pixel_coords_flat column mapping:
                # col 0: source X (data axis 2)
                # col 1: source Y (data axis 1)
                # col 2: source Z (spectral)
                
                # So for map_coordinates we need rows: [col 2, col 1, col 0]
                # We pre-calculate col 1 and col 0. col 2 will be calculated per plane.
                
                # Note: The 'spectral_wcs_idx' might not correspond to 'col 2' if the WCS axis order is permuted.
                # But _is_axis_coupled check above logic should hold.
                # For safety, let's map WCS axes to Numpy axes.
                
                # We can't easily decouple the WCS mapping entirely if PC matrix mixes them, 
                # but _is_axis_coupled checks PC matrix.
                
                # So we extract the columns corresponding to spatial numpy axes.
                
                # We'll just store the full (3, N) array and update the spectral row per plane?
                # No, source_pixel_coords_flat calculation is the expensive part (wcs_world2pix).
                # If uncoupled, the spatial output pixels (col 0, col 1...) only depend on spatial input pixels.
                
                # ...Wait, wcs_world2pix takes (N, 3) world coords.
                # If spectral axis is uncoupled, changing world_z shouldn't change pixel_x, pixel_y.
                # Changing pixel_z_target -> world_z_target -> (uncoupled) -> world_x, world_y unchaged -> pixel_x_source, pixel_y_source unchanged.
                
                # So yes, source X and Y pixels are constant.
                # Source Z pixel depends primarily on World Z.
                
                # We can pre-calculate the spatial columns of 'source_pixel_coords_flat'.
                precomputed_spatial_coords_scipy = source_pixel_coords_flat.T[::-1] # Now (3, N) in [z, y, x] order
                
                # However, the Z row of this precomputed array is for target z=0. 
                # We will need to replace it with the correct source Z for each target plane.
                
                # Optimization for Source Z:
                # If uncoupled, Source Z only depends on Target Z.
                # We can compute the 1D mapping: Target Z index -> Source Z index.
                # But wcs_world2pix might be non-linear.
                # We can basically map the Z-axis 1D array.
                
                target_z_indices = np.arange(shape_out[numpy_spectral_axis])
                
                # Create a probe vector for Z
                # We fix spatial pixel to 0,0 (or center) for this probe
                probe_coords = np.zeros((len(target_z_indices), target_wcs.wcs.naxis))
                # Set spatial to crpix or 0
                
                # Actually, simpler:
                # For each plane, we know the World Z.
                # We can map World Z -> Source Pixel Z directly if 1D.
                # But we have the full machinery. 
                
                # Let's keep it robust:
                # We rely on the fact that `map_coordinates` needs (3, N) coordinates.
                # We have precomputed [Z0, Y, X].
                # We just need to calculate [Z_new, Y, X].
                # Since spatial is constant, Y and X are effectively cached.
                # We only need to compute Z_new.
            
            def _process_plane(z_idx: int):
                # Optimize: If uncoupled, reconstruct source coordinates efficiently
                if not spectral_coupled and precomputed_spatial_coords_scipy is not None:
                     # 1. Calculate the Z-coordinate for this plane. 
                     # Since we established uncoupling, we can just calculate the center pixel's Z transformation
                     # and assume it's constant for the whole plane (if separable).
                     # OR, more safely, allow for spectral variation across the field?
                     # If uncoupled, spectral coordinate shouldn't vary with spatial position.
                     
                     # Let's perform a lightweight WCS transform for just the spectral axis to get the Z-plane value.
                     # We pick the center pixel of the plane to represent the Z-shift.
                     
                     # Construct a single pixel coordinate for the current plane center
                     # center_spatial = [s//2 for s in spatial_shape]
                     # pixel_probe = [ ... ]
                     # This feels risky if there's any tilt.
                     
                     # Safe approach using the fact that we confirmed uncoupled:
                     # The source Z index should be constant for the entire plane 
                     # ONLY IF the spectral axis allows it (e.g. constant frequency plane).
                     # But _is_axis_coupled only checks cross-terms. 
                     # It doesn't guarantee that World Z is constant across the pixel plane?
                     # In a standard datacube, target grid Z=k corresponds to a single spectral value.
                     
                     # Let's recalculate specific Z-coordinates for the generated grid 
                     # but reuse X and Y.
                     
                     # Actually, if we just want to avoid `wcs_world2pix` for all points:
                     # If uncoupled, `world_coords` spatial parts are constant. 
                     # `world_coords` spectral part is constant for the plane (since we fill z_idx).
                     
                     # So `world_coords` = [const_x, const_y, const_z] for the whole plane?
                     # NO, `world_coords` varies spatially.
                     # But `world_z` is constant for the whole plane (because pixel_z is constant).
                     
                     # So Source Pixels:
                     # pix_x = f(world_x, world_y)  <-- constant per plane
                     # pix_y = g(world_x, world_y)  <-- constant per plane
                     # pix_z = h(world_z)           <-- constant per plane (if uncoupled)
                     
                     # So `source_pixel_coords_flat` columns:
                     # Col X: reused from cache
                     # Col Y: reused from cache
                     # Col Z: calculated once per plane?
                     
                     # Yes! If uncoupled, the source Z pixel index is essentially constant for the entire target plane 
                     # (ignoring small numerical variations or higher order distortions which we assume negligible if 'uncoupled').
                     # Wait, if there is velocity variation across the field this assumption fails.
                     # But `_is_axis_coupled` checks PC matrix off-diagonals.
                     
                     # If PC matrix is diagonal, then:
                     # pix = CD_inv * (world - ref)
                     # if CD_inv is diagonal (or block diagonal), then Z_pix only depends on Z_world.
                     
                     # So, we need to determine the single `source_z_value` for this plane.
                     # We can do this by transforming one point (e.g. center) and using that Z for the whole array.
                     
                     # Let's do that.
                     cx = spatial_shape[1] // 2
                     cy = spatial_shape[0] // 2
                     
                     # Target Pixel
                     pix_probe = np.zeros((1, target_wcs.wcs.naxis))
                     pix_probe[0, spectral_wcs_idx] = z_idx
                     spatial_indices = [i for i in range(target_wcs.wcs.naxis) if i != spectral_wcs_idx]
                     pix_probe[0, spatial_indices[0]] = cx
                     pix_probe[0, spatial_indices[1]] = cy
                     
                     # To World
                     world_probe = target_wcs.wcs_pix2world(pix_probe, 0)
                     # To Source Pixel
                     src_pix_probe = self.original_wcs.wcs_world2pix(world_probe, 0)
                     
                     source_z_val = src_pix_probe[0, spectral_wcs_idx]
                     
                     # Copy precomputed cached coords
                     # We only need to update the Z-row (which is row 0 in [z, y, x] order if spectral is axis 0)
                     # Wait, we need to know which row in 'source_pixel_coords_for_scipy' corresponds to spectral.
                     
                     # source_pixel_coords_for_scipy is (3, N).
                     # It's constructed as source_pixel_coords_flat.T[::-1].
                     # source_pixel_coords_flat is (N, 3) ordered by WCS axis index (usually RA, DEC, VELO or similar).
                     # T makes it (3, N).
                     # [::-1] reverses the axes to match numpy order (usually VELO, DEC, RA).
                     
                     # So if spectral axis is WCS axis 2 (last), it becomes Numpy axis 0 (first).
                     # So row 0 is spectral.
                     
                     coords_for_interp = precomputed_spatial_coords_scipy.copy()
                     
                     # Identify which row corresponds to spectral axis
                     # spectral_wcs_idx is the WCS index.
                     # In the reversed list (numpy order), the index is: naxis - 1 - spectral_wcs_idx
                     numpy_z_row_idx =  target_wcs.wcs.naxis - 1 - spectral_wcs_idx
                     
                     coords_for_interp[numpy_z_row_idx, :] = source_z_val
                     
                else:
                    # Fallback to full recalculation
                    coords_for_wcs_axes = [None] * target_wcs.wcs.naxis
                    coords_for_wcs_axes[spectral_wcs_idx] = np.full(spatial_shape, z_idx, dtype=np.float32)

                    spatial_wcs_indices = [i for i in range(target_wcs.wcs.naxis) if i != spectral_wcs_idx]
                    
                    coords_for_wcs_axes[spatial_wcs_indices[0]] = spatial_grids[1]
                    coords_for_wcs_axes[spatial_wcs_indices[1]] = spatial_grids[0]
                    
                    pixel_coords_out = np.vstack([c.ravel() for c in coords_for_wcs_axes]).T

                    world_coords = target_wcs.wcs_pix2world(pixel_coords_out, 0)
                    source_pixel_coords_flat = self.original_wcs.wcs_world2pix(world_coords, 0)
                    coords_for_interp = source_pixel_coords_flat.T[::-1]
                
                # Use filled_data for interpolation
                resampled_plane_flat = map_coordinates(
                    filled_data, coords_for_interp,
                    order=order, mode='nearest', prefilter=prefilter
                )
                
                # Remap the NaN mask if we had NaNs
                if has_nans and nan_mask is not None:
                    regridded_mask_flat = map_coordinates(
                        nan_mask.astype(float), coords_for_interp,
                        order=0, mode='nearest'
                    )
                    resampled_plane_flat[regridded_mask_flat > 0.5] = np.nan
                
                return z_idx, resampled_plane_flat.reshape(spatial_shape)

            n_planes = shape_out[numpy_spectral_axis]
            progress_start = 40
            progress_end = 90
            cpu_count = os.cpu_count() or 1
            max_workers = min(8, max(cpu_count, 1)) if cpu_count > 1 else 1
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_process_plane, i) for i in range(n_planes)]
                completed = 0
                for future in as_completed(futures):
                    z_idx, plane_data = future.result()
                    slicer = [slice(None)] * len(shape_out)
                    slicer[numpy_spectral_axis] = z_idx
                    regridded_data[tuple(slicer)] = plane_data
                    
                    completed += 1
                    ratio = completed / n_planes
                    self._emit_progress(int(progress_start + (progress_end - progress_start) * ratio))

        self._emit_progress(90)
        final_header = self._base_header_copy()
        self._apply_wcs_to_header(final_header, target_wcs)

        return regridded_data, final_header, target_wcs


    def _calculate_manual_target_wcs(self, params: Dict) -> Tuple[fits.Header, WCS]:
        """
        Calculates the target Header and WCS for manual regridding without performing the regrid.
        This is a helper for the fast scipy-based regridding.
        """
        anchor_world: Sequence[float] = params["anchor_world"]
        grid_cdelt: Sequence[float] = params["grid_cdelt"]
        naxis = self.original_wcs.wcs.naxis
        data_shape = self._data_shape_for_wcs(naxis)
        new_header = fits.Header()
        new_cdelt_arr = np.array(grid_cdelt, dtype=float)
        anchor_arr = np.array(anchor_world, dtype=float)
        new_header["NAXIS"] = naxis
        for i in range(naxis):
            ctype = self.original_wcs.wcs.ctype[i]
            unit = self.original_wcs.wcs.cunit[i]
            is_velocity = self._is_velocity_axis(ctype)
            sanitized_unit = self._sanitize_unit_value(unit, is_velocity=is_velocity)
            new_header[f"CTYPE{i + 1}"] = ctype
            self._set_axis_unit(new_header, i + 1, sanitized_unit)
            new_header[f"CRVAL{i + 1}"] = anchor_arr[i]
            new_header[f"CDELT{i + 1}"] = new_cdelt_arr[i]

        for axis in range(naxis):
            world_corners = self._world_corners_for_axis(axis, data_shape)
            world_min, world_max = np.min(world_corners), np.max(world_corners)
            
            ctype = self.original_wcs.wcs.ctype[axis].upper()

            is_lon = 'RA-' in ctype or 'GLON-' in ctype
            if is_lon and (world_max - world_min) > 180:
                wrapped_corners = (world_corners + 180) % 360 - 180
                world_min, world_max = np.min(wrapped_corners), np.max(wrapped_corners)

            crval_new = anchor_arr[axis]
            cdelt_new = new_cdelt_arr[axis]
            step = float(cdelt_new)

            if abs(step) < 1e-12:
                np_axis = naxis - 1 - axis
                new_size = data_shape[np_axis] if np_axis < len(data_shape) else 1
                new_crpix = self.original_wcs.wcs.crpix[axis]
            else:
                tolerance = max(abs(step) * 1e-6, 1e-12)
                if step > 0:
                    k_min = int(math.ceil((world_min - crval_new - tolerance) / step))
                    k_max = int(math.floor((world_max - crval_new + tolerance) / step))
                else:
                    k_min = int(math.ceil((world_max - crval_new + tolerance) / step))
                    k_max = int(math.floor((world_min - crval_new - tolerance) / step))
                
                if k_max < k_min:
                    new_size = 1
                    new_crpix = 1.0
                else:
                    new_size = k_max - k_min + 1
                    new_crpix = 1.0 - k_min
            
            new_header[f"NAXIS{axis + 1}"] = new_size
            new_header[f"CRPIX{axis + 1}"] = new_crpix

        self._ensure_header_pc(new_header)
        target_wcs = self._create_wcs_safely(new_header)

        return new_header, target_wcs


    def _regrid_manual_reproject(self, params: Dict, method: str) -> Tuple[np.ndarray, fits.Header, WCS]:
        """High-precision manual regridding using reproject library.
        Kept as a backend but not exposed to the GUI by default.
        """
        spectral_wcs_idx = self._spectral_axis_index()
        if spectral_wcs_idx is None or self.original_wcs.wcs.naxis != 3:
            # Note: _regrid_manual_wcs_only now returns a WCS object as the third element.
            return self._regrid_manual_wcs_only(params, method)

        if self._is_axis_coupled(self.original_wcs, spectral_wcs_idx) or self._celestial_has_projection(self.original_wcs):
            # Fall back to the legacy implementation when axes are strongly coupled.
            # This path is less common; for simplicity, we create WCS from the final header here.
            # A more robust solution would refactor _regrid_manual_legacy as well.
            final_data, final_header = self._regrid_manual_legacy(params, method)
            final_wcs = WCS(final_header)
            return final_data, final_header, final_wcs

        cache_override = params.get("slice_cache_size")
        previous_cache_limit = self._max_cached_slices
        if isinstance(cache_override, int) and cache_override > 0:
            self._max_cached_slices = cache_override

        try:
            anchor_world = params["anchor_world"]
            grid_cdelt = params["grid_cdelt"]
            naxis = self.original_wcs.wcs.naxis

            self._emit_progress(20)

            # --- Build spatial target WCS without touching the spectral axis ---
            wcs_spatial = copy.deepcopy(self.original_wcs)
            shape_orig = self.original_data.shape
            shape_spatial = list(shape_orig)
            crpix_orig = np.array(self.original_wcs.wcs.crpix, dtype=float)
            crval_orig = np.array(self.original_wcs.wcs.crval, dtype=float)
            cdelt_orig = np.array(self.original_wcs.wcs.cdelt, dtype=float)
            crval_new = np.array(anchor_world, dtype=float)
            cdelt_new = np.array(grid_cdelt, dtype=float)

            unit_pixel_steps: List[bool] = []
            for axis in range(naxis):
                try:
                    orig_step = float(cdelt_orig[axis])
                    new_step = float(cdelt_new[axis])
                except Exception:
                    unit_pixel_steps.append(False)
                    continue
                if not np.isfinite(orig_step) or not np.isfinite(new_step):
                    unit_pixel_steps.append(False)
                    continue
                if abs(orig_step) <= 0:
                    unit_pixel_steps.append(False)
                    continue
                unit_pixel_steps.append(
                    self._almost_equal(abs(orig_step), abs(new_step), tol=1e-6)
                )

            axis_changed = [
                self._axis_requires_regrid(axis, crval_new[axis], cdelt_new[axis])
                for axis in range(naxis)
            ]

            if not any(axis_changed):
                data_copy = self._copy_original_cube()
                header = self._base_header_copy()
                self._apply_wcs_to_header(header, self.original_wcs)
                self._ensure_header_pc(header)
                self._ensure_header_units(header)
                return data_copy, header, self.original_wcs.copy()

            spectral_changed = axis_changed[spectral_wcs_idx]
            needs_spatial_reproject = any(
                axis_changed[axis] for axis in range(naxis) if axis != spectral_wcs_idx
            )

            for axis in range(naxis):
                if axis == spectral_wcs_idx:
                    continue

                np_axis = naxis - 1 - axis
                if np_axis < 0 or np_axis >= len(shape_spatial):
                    final_data, final_header = self._regrid_manual_legacy(params, method)
                    final_wcs = WCS(final_header)
                    return final_data, final_header, final_wcs

                if not axis_changed[axis]:
                    shape_spatial[np_axis] = shape_orig[np_axis]
                    wcs_spatial.wcs.crval[axis] = crval_orig[axis]
                    wcs_spatial.wcs.cdelt[axis] = cdelt_orig[axis]
                    wcs_spatial.wcs.crpix[axis] = crpix_orig[axis]
                    continue

                wcs_spatial.wcs.crval[axis] = crval_new[axis]
                wcs_spatial.wcs.cdelt[axis] = cdelt_new[axis]

                world_corners = self._world_corners_for_axis(axis, shape_orig)
                dist = np.max(np.abs(world_corners - crval_new[axis]))
                new_size = int(np.ceil(dist * 2 / max(abs(cdelt_new[axis]), 1e-12)))
                new_size = max(new_size, 1)
                shape_spatial[np_axis] = new_size
                wcs_spatial.wcs.crpix[axis] = new_size / 2.0

            wcs_spatial.wcs.set()
            shape_out_spatial = tuple(shape_spatial)
            wcs_spatial.array_shape = shape_out_spatial

            source_wcs_2d = self._drop_axis_safe(self.original_wcs, spectral_wcs_idx)
            target_wcs_2d = self._drop_axis_safe(wcs_spatial, spectral_wcs_idx)
            if source_wcs_2d is None or target_wcs_2d is None:
                final_data, final_header = self._regrid_manual_legacy(params, method)
                final_wcs = WCS(final_header)
                return final_data, final_header, final_wcs

            # --- Prepare spectral coordinates ---
            spec_idx_numpy = naxis - 1 - spectral_wcs_idx
            if spec_idx_numpy < 0 or spec_idx_numpy >= len(shape_orig):
                final_data, final_header = self._regrid_manual_legacy(params, method)
                final_wcs = WCS(final_header)
                return final_data, final_header, final_wcs

            n_spec_orig = shape_orig[spec_idx_numpy]
            if n_spec_orig <= 0:
                raise ValueError("Spectral axis has zero length.")

            crpix_spec_orig = crpix_orig[spectral_wcs_idx]
            crval_spec_orig = self.original_wcs.wcs.crval[spectral_wcs_idx]
            cdelt_spec_orig = self.original_wcs.wcs.cdelt[spectral_wcs_idx]
            cdelt_spec_new = cdelt_new[spectral_wcs_idx]
            if cdelt_spec_new == 0:
                raise ValueError("Spectral grid spacing must be non-zero.")
            crval_spec_new = crval_new[spectral_wcs_idx]

            orig_spec_coords = (
                (np.arange(n_spec_orig) - (crpix_spec_orig - 1)) * cdelt_spec_orig + crval_spec_orig
            )
            velocity_scale = self._spectral_velocity_scale_factor(
                spectral_wcs_idx,
                cdelt_spec_orig,
            )
            if velocity_scale != 1.0:
                orig_spec_coords = orig_spec_coords * velocity_scale

            if spectral_changed:
                spectral_min = float(np.nanmin(orig_spec_coords))
                spectral_max = float(np.nanmax(orig_spec_coords))
                step = float(cdelt_spec_new)
                tolerance = max(abs(step) * 1e-8, 1e-9)

                if step > 0:
                    k_min = int(math.ceil((spectral_min - crval_spec_new - tolerance) / step))
                    k_max = int(math.floor((spectral_max - crval_spec_new + tolerance) / step))
                else:
                    k_min = int(math.ceil((spectral_max - crval_spec_new + tolerance) / step))
                    k_max = int(math.floor((spectral_min - crval_spec_new - tolerance) / step))

                if k_max < k_min:
                    k_min = k_max = 0

                k_indices = np.arange(k_min, k_max + 1, dtype=int)
                if k_indices.size == 0:
                    k_indices = np.array([0], dtype=int)
                    k_min = k_max = 0

                n_spec_new = k_indices.size
                crpix_spec_new = float(1 - k_min)
                new_spec_coords = crval_spec_new + k_indices.astype(float) * step
            else:
                n_spec_new = n_spec_orig
                crpix_spec_new = float(crpix_spec_orig)
                new_spec_coords = orig_spec_coords

            target_shape_2d = tuple(shape_out_spatial[1:])
            preferred_dtype = self._preferred_float_dtype()
            final_data = np.empty((n_spec_new,) + target_shape_2d, dtype=preferred_dtype)

            source_shape_2d: Optional[Tuple[int, int]] = None
            spatial_mapping: Optional["_SpatialMapping"] = None
            integer_shift: Optional[Tuple[int, int]] = None
            spline_order: Optional[int] = None
            if needs_spatial_reproject:
                source_shape_candidate = tuple(
                    int(dim) for idx, dim in enumerate(shape_orig) if idx != spec_idx_numpy
                )
                if len(source_shape_candidate) == 2:
                    source_shape_2d = source_shape_candidate
                    spatial_axes = [axis for axis in range(naxis) if axis != spectral_wcs_idx]
                    unit_step_flags_spatial = tuple(unit_pixel_steps[axis] for axis in spatial_axes)
                    spatial_mapping = self._prepare_spatial_resampler(
                        source_wcs_2d,
                        target_wcs_2d,
                        target_shape_2d,
                        source_shape_2d,
                        unit_step_flags_spatial,
                    )
                    if spatial_mapping is not None:
                        integer_shift = spatial_mapping.integer_shift
                    if map_coordinates is not None:
                        spline_order = self._interpolation_to_spline_order(method)

            enable_parallel = bool(params.get("enable_parallel", True))
            requested_workers = params.get("workers") or params.get("parallel_workers")
            if isinstance(requested_workers, int) and requested_workers > 0:
                max_workers = requested_workers
            else:
                cpu_count = os.cpu_count() or 1
                max_workers = min(8, max(1, cpu_count - 1))

            # Ensure at least one worker is active.
            max_workers = max(1, max_workers)
            worker_items = list(enumerate(new_spec_coords))
            total_planes = len(worker_items)

            spectral_identity = not spectral_changed

            def _process_plane(item):
                idx, world_value = item
                if spectral_identity:
                    plane = self._get_spectral_slice(idx)
                else:
                    plane = self._spectral_plane_at(world_value, method, orig_spec_coords)
                if plane is None:
                    plane = np.full(target_shape_2d, np.nan, dtype=preferred_dtype)
                elif plane.dtype != preferred_dtype:
                    plane = plane.astype(preferred_dtype, copy=False)

                if needs_spatial_reproject:
                    if (
                        integer_shift is not None
                        and source_shape_2d is not None
                    ):
                        plane = self._apply_integer_shift(
                            plane,
                            integer_shift,
                            preferred_dtype,
                            source_shape_2d,
                            target_shape_2d,
                        )
                    elif (
                        spatial_mapping is not None
                        and source_shape_2d is not None
                        and spline_order is not None
                        and map_coordinates is not None
                    ):
                        plane = self._apply_spatial_resampler(
                            plane,
                            spatial_mapping,
                            spline_order,
                            preferred_dtype,
                            source_shape_2d,
                        )
                    else:
                        plane = self._reproject_2d_plane(
                            plane,
                            source_wcs_2d,
                            target_wcs_2d,
                            target_shape_2d,
                            method,
                        )
                else:
                    if plane.shape != target_shape_2d:
                        plane = np.reshape(plane, target_shape_2d)
                return idx, plane

            progress_start = 20
            progress_end = 90

            if enable_parallel and total_planes > 1 and max_workers > 1:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(_process_plane, item) for item in worker_items]
                    completed = 0
                    for future in as_completed(futures):
                        idx, plane = future.result()
                        final_data[idx] = plane
                        completed += 1
                        ratio = completed / total_planes
                        self._emit_progress(
                            int(progress_start + (progress_end - progress_start) * ratio)
                        )
            else:
                for completed, item in enumerate(worker_items, start=1):
                    idx, plane = _process_plane(item)
                    final_data[idx] = plane
                    ratio = completed / total_planes
                    self._emit_progress(
                        int(progress_start + (progress_end - progress_start) * ratio)
                    )

            self._emit_progress(90)

            # --- Build final header and WCS ---
            final_header = self._base_header_copy()
            # The wcs_spatial object contains the correct new spatial WCS information.
            # We will now update it with the new spectral axis parameters to make it complete.
            final_wcs = wcs_spatial
            final_wcs.wcs.crpix[spectral_wcs_idx] = crpix_spec_new
            final_wcs.wcs.crval[spectral_wcs_idx] = crval_spec_new
            final_wcs.wcs.cdelt[spectral_wcs_idx] = cdelt_spec_new
            final_wcs.wcs.set()
            
            # Apply the completed WCS to the header
            self._apply_wcs_to_header(final_header, final_wcs)
            final_header[f"NAXIS{spectral_wcs_idx + 1}"] = n_spec_new # Ensure NAXIS is correct

            self._ensure_header_pc(final_header)
            self._ensure_header_units(final_header)

            return final_data, final_header, final_wcs
        finally:
            self._max_cached_slices = previous_cache_limit
