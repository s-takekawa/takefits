"""Spectrum extraction usecases."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_widths
import warnings

try:
    from scipy.signal import PeakPropertyWarning
except Exception:  # pragma: no cover - SciPy compatibility fallback
    try:
        from scipy.signal._peak_finding_utils import PeakPropertyWarning
    except Exception:
        PeakPropertyWarning = Warning

from takefits.core.app_state import AppState, RegionSpec
from takefits.logic.data_tools import ensure_operation_memory_budget, sanitize_slice
from .utils import get_axis_ctype, parse_world_coordinate


_AVERAGED_SPECTRUM_TILE_PIXELS = 1_048_576


def _averaged_spectrum_tile_shape(height: int, width: int) -> Tuple[int, int]:
    """Choose a spatial tile containing at most the configured pixel target."""
    width_for_rows = min(max(1, int(width)), _AVERAGED_SPECTRUM_TILE_PIXELS)
    tile_rows = min(
        max(1, int(height)),
        max(1, _AVERAGED_SPECTRUM_TILE_PIXELS // width_for_rows),
    )
    tile_cols = min(
        max(1, int(width)),
        max(1, _AVERAGED_SPECTRUM_TILE_PIXELS // tile_rows),
    )
    return tile_rows, tile_cols


def get_spectrum(
    state: AppState,
    x: Optional[int] = None,
    y: Optional[int] = None,
    world_x: Optional[Union[float, str]] = None,
    world_y: Optional[Union[float, str]] = None
) -> np.ndarray:
    """
    Extract a spectrum at a given spatial position.

    Args:
        state: AppState with loaded data
        x, y: Pixel coordinates (uses cursor position if not specified)
        world_x, world_y: World coordinates (float or str).
                          Overrides x, y if provided.

    Returns:
        1D numpy array with the spectrum
    """
    if state.data is None:
        raise ValueError("No data loaded")

    # Handle world coordinates
    if world_x is not None or world_y is not None:
        if state.wcs is None:
            raise ValueError("WCS required for world coordinates")

        # We need both world coordinates for accurate conversion (rotation)
        if world_x is None or world_y is None:
            raise ValueError("Both world_x and world_y must be provided")

        # Parse inputs
        ctype_x = get_axis_ctype(state, 0)  # Assumes spatial X is WCS 0
        ctype_y = get_axis_ctype(state, 1)  # Assumes spatial Y is WCS 1

        wx = parse_world_coordinate(world_x, ctype_x)
        wy = parse_world_coordinate(world_y, ctype_y)

        # Prepare full world coordinate list
        w_coords = [0.0] * state.wcs.naxis
        w_coords[0] = wx
        w_coords[1] = wy

        # Convert to pixel
        pix = state.wcs.wcs_world2pix([w_coords], 0)[0]
        x = int(round(pix[0]))
        y = int(round(pix[1]))

    if x is None:
        x = state.cursor.xpix
    if y is None:
        y = state.cursor.ypix

    data = state.data

    # Handle 4D data
    if data.ndim == 4:
        data = data[state.current_s]

    if data.ndim != 3:
        if data.ndim == 2:
            # Handle 2D data (no spectral axis, just return single point?)
            # Usually get_spectrum implies 3D.
            # But logic allows 2D returns array of 1.
            y = max(0, min(y, data.shape[0] - 1))
            x = max(0, min(x, data.shape[1] - 1))
            return sanitize_slice(np.array([data[y, x]]))
        raise ValueError(f"Expected 3D data cube, got {data.ndim}D")

    # Clamp to valid range
    y = max(0, min(y, data.shape[1] - 1))
    x = max(0, min(x, data.shape[2] - 1))

    return sanitize_slice(data[:, y, x]).copy()


def get_averaged_spectrum(
    state: AppState,
    region: RegionSpec
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Calculate the average spectrum within a region.

    Args:
        state: AppState containing data and WCS.
        region: RegionSpec defining the region of interest.

    Returns:
        velocity_values: Array of velocity/frequency values.
        spectrum_values: Array of averaged spectrum values.
        unit_string: String representation of the spectral unit.
    """
    if state.data is None or state.data.ndim < 3:
        raise ValueError("Data is not a cube")

    data_cube = state.data
    if data_cube.ndim == 4:
        data_cube = data_cube[state.current_s]

    # Get dimensions
    z_dim, height, width = data_cube.shape

    # --- 1. Get the region's bounding box in pixel coordinates ---
    # Note: logic adapted from tools/spectrum.py and ui/region_manager.py
    if region.type == "circle":
        cx, cy = region.center_x, region.center_y
        r = region.params.get('radius', 0)
        x_min, x_max = cx - r, cx + r
        y_min, y_max = cy - r, cy + r
    elif region.type in ("rectangle", "ellipse"):
        cx, cy = region.center_x, region.center_y
        w = region.params.get('width', 0) / 2
        h = region.params.get('height', 0) / 2
        angle = np.deg2rad(region.params.get('angle', 0.0))

        # Calculate vertices for bounding box
        corners = [(-w, -h), (w, -h), (w, h), (-w, h)]
        verts = []
        for dx, dy in corners:
            x = cx + np.cos(angle) * dx - np.sin(angle) * dy
            y = cy + np.sin(angle) * dx + np.cos(angle) * dy
            verts.append((x, y))

        x_coords, y_coords = zip(*verts)
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
    else:
        raise ValueError(f"Unsupported region shape: {region.type}")

    # --- 2. Clip the region's spatial bounding box ---
    x_start, x_end = int(np.floor(x_min)), int(np.ceil(x_max))
    y_start, y_end = int(np.floor(y_min)), int(np.ceil(y_max))

    x_start = max(0, x_start)
    y_start = max(0, y_start)
    x_end = min(width, x_end)
    y_end = min(height, y_end)

    if x_start >= x_end or y_start >= y_end:
        raise ValueError("Region is outside data bounds")

    region_height = y_end - y_start
    region_width = x_end - x_start
    region_pixels = region_height * region_width

    # The only region-sized state retained below is a 2-D boolean mask.  The
    # coordinate expressions used to construct rotated shapes can briefly hold
    # a few float64 planes, so bound that peak before creating them.
    ensure_operation_memory_budget(
        region_pixels * (4 * np.dtype(np.float64).itemsize + np.dtype(np.bool_).itemsize)
        + z_dim * 2 * np.dtype(np.float64).itemsize,
        operation_name="Region-averaged Spectrum",
        guidance=(
            "Select a smaller spatial region or make a cutout before averaging "
            "the spectrum."
        ),
    )

    # --- 3. Create one 2-D local mask (never a channel-broadcast 3-D mask) ---
    y_grid = np.arange(y_start, y_end, dtype=np.float64)[:, np.newaxis]
    x_grid = np.arange(x_start, x_end, dtype=np.float64)[np.newaxis, :]

    # Region check logic (using RegionSpec params)
    if region.type == "circle":
        cx, cy = region.center_x, region.center_y
        r = region.params.get('radius', 0)
        # Optimization: work with squared distances
        local_mask = (x_grid - cx) ** 2 + (y_grid - cy) ** 2 <= r ** 2
    elif region.type == "rectangle":
        cx, cy = region.center_x, region.center_y
        w_half = region.params.get('width', 0) / 2
        h_half = region.params.get('height', 0) / 2
        angle_rad = np.deg2rad(region.params.get('angle', 0.0))

        dx = x_grid - cx
        dy = y_grid - cy

        # Rotate point back to aligned coordinates
        x_rot = dx * np.cos(-angle_rad) - dy * np.sin(-angle_rad)
        y_rot = dx * np.sin(-angle_rad) + dy * np.cos(-angle_rad)

        local_mask = (np.abs(x_rot) <= w_half) & (np.abs(y_rot) <= h_half)
    elif region.type == "ellipse":
        cx, cy = region.center_x, region.center_y
        w_half = region.params.get('width', 0) / 2
        h_half = region.params.get('height', 0) / 2
        angle_rad = np.deg2rad(region.params.get('angle', 0.0))

        dx = x_grid - cx
        dy = y_grid - cy

        x_rot = dx * np.cos(-angle_rad) - dy * np.sin(-angle_rad)
        y_rot = dx * np.sin(-angle_rad) + dy * np.cos(-angle_rad)

        # Ellipse equation
        local_mask = ((x_rot / w_half) ** 2 + (y_rot / h_half) ** 2 <= 1.0)

    if not np.any(local_mask):
        raise ValueError("Region contains no data")

    # --- 4. Average finite values channel-by-channel and spatial-tile-by-tile ---
    # A LazyScaledArray therefore scales only one bounded 2-D tile at a time,
    # while a normal ndarray follows the same finite-value averaging semantics.
    spectrum = np.full(z_dim, np.nan, dtype=np.float64)
    tile_rows, tile_cols = _averaged_spectrum_tile_shape(
        region_height, region_width
    )
    for channel in range(z_dim):
        channel_sum = 0.0
        channel_count = 0
        for local_y in range(0, region_height, tile_rows):
            local_y_end = min(local_y + tile_rows, region_height)
            source_y = slice(y_start + local_y, y_start + local_y_end)
            for local_x in range(0, region_width, tile_cols):
                local_x_end = min(local_x + tile_cols, region_width)
                source_x = slice(x_start + local_x, x_start + local_x_end)
                region_tile = local_mask[
                    local_y:local_y_end,
                    local_x:local_x_end,
                ]
                if not np.any(region_tile):
                    continue

                data_tile = data_cube[channel, source_y, source_x]
                if np.ma.isMaskedArray(data_tile):
                    values = sanitize_slice(
                        np.array(np.ma.getdata(data_tile), copy=True)
                    )
                    valid = ~np.ma.getmaskarray(data_tile)
                    valid = valid & region_tile & np.isfinite(values)
                else:
                    values = sanitize_slice(np.array(data_tile, copy=True))
                    valid = region_tile & np.isfinite(values)

                valid_count = int(np.count_nonzero(valid))
                if valid_count:
                    channel_sum += float(np.sum(values[valid], dtype=np.float64))
                    channel_count += valid_count

        if channel_count:
            spectrum[channel] = channel_sum / channel_count

    # --- 5. Get spectral axis ---
    velocity_values = spectral_axis_values(state, len(spectrum))
    unit = spectral_axis_unit(state, len(spectrum), fallback="pixel")

    return velocity_values, spectrum, unit


def _spectral_axis_values_with_status(
    state: AppState, n_channels: int
) -> Tuple[np.ndarray, bool]:
    """Return spectral values and whether WCS conversion actually succeeded."""
    values = np.arange(int(n_channels), dtype=float)
    wcs = getattr(state, "wcs", None)
    if wcs is None:
        return values, False
    try:
        spectral = wcs.spectral
        if spectral.naxis != 1:
            return values, False
        world = np.asarray(
            spectral.pixel_to_world_values(values), dtype=float
        ).ravel()
        if world.size == values.size and np.all(np.isfinite(world)):
            return world, True
    except Exception:
        pass
    return values, False


def spectral_axis_values(state: AppState, n_channels: int) -> np.ndarray:
    """World values along the spectral axis, falling back to channel index.

    Using the one-dimensional spectral WCS keeps extraction, fitting, and
    plotting on the same axis, including non-trivial WCS transformations.
    """
    return _spectral_axis_values_with_status(state, n_channels)[0]


def spectral_axis_unit(
    state: AppState, n_channels: int, *, fallback: str = "pixel"
) -> str:
    """Unit matching :func:`spectral_axis_values`, including its fallback."""
    _values, used_world = _spectral_axis_values_with_status(state, n_channels)
    if not used_world:
        return fallback
    unit = spectral_unit_string(state)
    if unit:
        return unit
    try:
        spectral_unit = state.wcs.spectral.wcs.cunit[0]
        return spectral_unit.to_string() if spectral_unit else fallback
    except Exception:
        return fallback


def spectral_unit_string(state: AppState) -> str:
    """Converted spectral-axis unit from `spectral_metadata`, or ''.

    Accepts either a bare unit (``"km/s"``) or a decorated label
    (``"Velocity [km/s]"``), matching what `export_pv_fits` parses.
    """
    metadata = getattr(state, "spectral_metadata", None) or {}
    raw = str(metadata.get("current_axis_unit", "") or "").strip()
    if not raw:
        return ""
    import re

    match = re.search(r"\[(.*?)\]", raw)
    return (match.group(1) if match else raw).strip()


@dataclass
class GaussianFitComponent:
    """Single Gaussian component in a fitted spectrum model."""
    amplitude: float
    center: float
    sigma: float
    fwhm: float
    amplitude_err: float = np.nan
    center_err: float = np.nan
    sigma_err: float = np.nan
    fwhm_err: float = np.nan


@dataclass
class GaussianFitResult:
    """Result container for Gaussian spectrum fitting."""
    success: bool
    auto_components: bool
    requested_components: int
    n_components: int
    baseline: float
    baseline_err: float
    fit_baseline: bool
    baseline_fixed: float
    components: List[GaussianFitComponent]
    bic: float
    aic: float
    rss: float
    rchi2: float
    noise_sigma: float
    x: np.ndarray
    y: np.ndarray
    model: np.ndarray
    residual: np.ndarray

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the fitted model at arbitrary *x*.

        ``model`` is sampled only at the input channels, so plotting it draws a
        polyline through a handful of points and a narrow line looks angular.
        Re-evaluating on a finer grid gives the smooth curve a figure wants.
        """
        params: List[float] = []
        for component in self.components:
            params.extend(
                [component.amplitude, component.center, component.sigma]
            )
        return _multi_gaussian_model(
            np.asarray(x, dtype=float), float(self.baseline), params
        )

    def sample_curve(self, oversample: int = 20, n_points: Optional[int] = None):
        """Return ``(x, y)`` of the fitted model on a smooth grid.

        The grid spans the fitted x range with ``oversample`` points per input
        channel, capped so a long spectrum does not produce a huge array.
        """
        x_values = np.asarray(self.x, dtype=float)
        if x_values.size < 2:
            return x_values, self.evaluate(x_values)
        if n_points is None:
            n_points = int(min(4000, max(x_values.size * int(oversample), 200)))
        fine = np.linspace(float(x_values.min()), float(x_values.max()), n_points)
        return fine, self.evaluate(fine)


def _estimate_noise_sigma(values: np.ndarray, *, clip_sigma: float = 4.0, max_iter: int = 3) -> float:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 1.0

    work = finite.copy()
    for _ in range(max(0, int(max_iter))):
        if work.size < 8:
            break
        median = float(np.median(work))
        mad = float(np.median(np.abs(work - median)))
        sigma = 1.4826 * mad
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = float(np.nanstd(work))
        if not np.isfinite(sigma) or sigma <= 0:
            break
        keep = np.abs(work - median) <= float(clip_sigma) * sigma
        if keep.all():
            break
        if np.count_nonzero(keep) < 8:
            break
        work = work[keep]

    median = float(np.median(work))
    mad = float(np.median(np.abs(work - median)))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.nanstd(work))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    return sigma


def _multi_gaussian_model(x: np.ndarray, baseline: float, params: Sequence[float]) -> np.ndarray:
    y = np.full_like(x, float(baseline), dtype=float)
    for i in range(0, len(params), 3):
        amp = float(params[i])
        cen = float(params[i + 1])
        sig = float(params[i + 2])
        sig = max(abs(sig), 1e-8)
        y += amp * np.exp(-0.5 * ((x - cen) / sig) ** 2)
    return y


def _multi_gaussian_curve_fit(x: np.ndarray, baseline: float, *params: float) -> np.ndarray:
    return _multi_gaussian_model(x, baseline, params)


def _peak_sigma_guesses(
    x: np.ndarray,
    signal: np.ndarray,
    peak_indices: np.ndarray,
    min_sigma: float,
    max_sigma: float
) -> Dict[int, float]:
    width_map: Dict[int, float] = {}
    if peak_indices.size == 0:
        return width_map

    if x.size > 1:
        dx = float(np.nanmedian(np.diff(x)))
        if not np.isfinite(dx) or dx == 0:
            dx = 1.0
    else:
        dx = 1.0
    dx = abs(dx)

    default_width = 2.355 * min_sigma / max(dx, 1e-8)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PeakPropertyWarning)
            widths = peak_widths(signal, peak_indices, rel_height=0.5)[0]
    except Exception:
        widths = np.full(peak_indices.shape, default_width)

    widths = np.asarray(widths, dtype=float)
    invalid = (~np.isfinite(widths)) | (widths <= 0.0)
    if np.any(invalid):
        widths = widths.copy()
        widths[invalid] = default_width

    for idx, width in zip(peak_indices.tolist(), widths.tolist()):
        sigma = abs(float(width)) * dx / 2.355
        sigma = float(np.clip(sigma, min_sigma, max_sigma))
        width_map[int(idx)] = sigma
    return width_map


def _select_peak_indices(signal: np.ndarray, n_components: int, noise_sigma: float) -> List[int]:
    if signal.size == 0:
        return []

    prominence = max(float(noise_sigma) * 0.5, 1e-8)
    try:
        peak_indices, props = find_peaks(signal, prominence=prominence)
    except Exception:
        peak_indices = np.array([], dtype=int)
        props = {}

    ordered: List[int] = []
    if peak_indices.size > 0:
        prominences = props.get("prominences")
        if prominences is None or len(prominences) != len(peak_indices):
            prominences = signal[peak_indices]
        order = np.argsort(prominences)[::-1]
        ordered = [int(peak_indices[i]) for i in order.tolist()]

    if len(ordered) < n_components:
        fallback = np.argsort(signal)[::-1]
        for idx in fallback.tolist():
            idx = int(idx)
            if idx not in ordered:
                ordered.append(idx)
            if len(ordered) >= n_components:
                break

    return ordered[:n_components]


def _fit_fixed_gaussians(
    x: np.ndarray,
    y: np.ndarray,
    n_components: int,
    *,
    fit_baseline: bool,
    baseline_fixed: float,
    allow_negative: bool,
    noise_sigma: float,
    min_sigma: float,
    max_sigma: Optional[float],
    seed_components: Optional[Sequence[Tuple[float, float, float]]] = None,
) -> Optional[GaussianFitResult]:
    def _param_errors(pcov: Any, expected: int) -> np.ndarray:
        errs = np.full(int(expected), np.nan, dtype=float)
        if pcov is None:
            return errs
        arr = np.asarray(pcov, dtype=float)
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
            return errs
        diag = np.diag(arr)
        if diag.size == 0:
            return errs
        count = min(int(expected), int(diag.size))
        diag = diag[:count]
        valid = np.isfinite(diag) & (diag >= 0.0)
        errs[:count] = np.where(valid, np.sqrt(diag), np.nan)
        return errs

    n_params = (1 if fit_baseline else 0) + 3 * n_components
    if x.size <= n_params:
        return None

    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    x_span = max(abs(x_max - x_min), min_sigma * 4.0)
    max_sigma_local = float(max_sigma) if max_sigma is not None else max(min_sigma * 1.5, x_span * 0.5)
    max_sigma_local = max(max_sigma_local, min_sigma * 1.01)

    y_min = float(np.nanmin(y))
    y_max = float(np.nanmax(y))
    y_span = max(abs(y_max - y_min), noise_sigma * 6.0, 1e-6)
    amp_cap = max(abs(y_max), abs(y_min), y_span) * 3.0
    amp_cap = max(amp_cap, noise_sigma * 10.0, 1e-6)

    baseline_fixed_local = float(baseline_fixed)
    baseline0 = float(np.nanmedian(y)) if fit_baseline else baseline_fixed_local
    residual = y - baseline0
    search_signal = np.abs(residual) if allow_negative else np.clip(residual, 0.0, None)

    peak_idx = _select_peak_indices(search_signal, n_components, noise_sigma)
    width_map = _peak_sigma_guesses(x, search_signal, np.asarray(peak_idx, dtype=int), min_sigma, max_sigma_local)
    default_sigma = float(np.clip(x_span / max(6.0 * n_components, 4.0), min_sigma, max_sigma_local))

    p0: List[float] = []
    lower: List[float] = []
    upper: List[float] = []
    if fit_baseline:
        p0.append(baseline0)
        lower.append(y_min - y_span)
        upper.append(y_max + y_span)

    seed = list(seed_components or [])
    for i in range(n_components):
        if i < len(seed):
            amp0, cen0, sig0 = seed[i]
            amp0 = float(amp0)
            cen0 = float(cen0)
            sig0 = float(sig0)
        else:
            idx = peak_idx[i] if i < len(peak_idx) else int(np.argmax(search_signal))
            idx = max(0, min(int(idx), x.size - 1))
            cen0 = float(x[idx])
            sig0 = float(width_map.get(idx, default_sigma))
            local_amp = float(residual[idx])
            if allow_negative:
                amp0 = local_amp
                if abs(amp0) < noise_sigma:
                    amp0 = float(np.sign(amp0) * max(noise_sigma, 1e-6))
            else:
                amp0 = max(local_amp, noise_sigma, 1e-6)

        if allow_negative:
            amp0 = float(np.clip(amp0, -amp_cap * 0.95, amp_cap * 0.95))
            amp_lower = -amp_cap
            amp_upper = amp_cap
        else:
            amp0 = float(np.clip(abs(amp0), 1e-8, amp_cap * 0.95))
            amp_lower = 0.0
            amp_upper = amp_cap

        cen0 = float(np.clip(cen0, x_min, x_max))
        sig0 = float(np.clip(abs(sig0), min_sigma, max_sigma_local))

        p0.extend([amp0, cen0, sig0])
        lower.extend([amp_lower, x_min, min_sigma])
        upper.extend([amp_upper, x_max, max_sigma_local])

    try:
        if fit_baseline:
            popt, pcov = curve_fit(
                _multi_gaussian_curve_fit,
                x,
                y,
                p0=np.asarray(p0, dtype=float),
                bounds=(np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)),
                maxfev=40000,
            )
            perr = _param_errors(pcov, len(popt))
            baseline = float(popt[0])
            baseline_err = float(perr[0]) if perr.size > 0 else float("nan")
            params_flat = np.asarray(popt[1:], dtype=float)
            params_err_flat = np.asarray(perr[1:], dtype=float)
        else:
            def _model_fixed_baseline(x_local, *params):
                return _multi_gaussian_model(x_local, baseline_fixed_local, params)

            popt, pcov = curve_fit(
                _model_fixed_baseline,
                x,
                y,
                p0=np.asarray(p0, dtype=float),
                bounds=(np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)),
                maxfev=40000,
            )
            perr = _param_errors(pcov, len(popt))
            baseline = baseline_fixed_local
            baseline_err = 0.0
            params_flat = np.asarray(popt, dtype=float)
            params_err_flat = np.asarray(perr, dtype=float)
    except Exception:
        return None

    params = params_flat.reshape(-1, 3)
    if params_err_flat.size != params_flat.size:
        params_err_flat = np.full(params_flat.size, np.nan, dtype=float)
    param_errs = params_err_flat.reshape(-1, 3)
    components = [
        GaussianFitComponent(
            amplitude=float(a),
            center=float(c),
            sigma=max(float(abs(s)), min_sigma),
            fwhm=2.355 * max(float(abs(s)), min_sigma),
            amplitude_err=float(abs(ae)) if np.isfinite(ae) else float("nan"),
            center_err=float(abs(ce)) if np.isfinite(ce) else float("nan"),
            sigma_err=float(abs(se)) if np.isfinite(se) else float("nan"),
            fwhm_err=2.355 * float(abs(se)) if np.isfinite(se) else float("nan"),
        )
        for (a, c, s), (ae, ce, se) in zip(params, param_errs)
    ]
    components.sort(key=lambda comp: comp.center)

    sorted_params: List[float] = []
    for comp in components:
        sorted_params.extend([comp.amplitude, comp.center, comp.sigma])

    model = _multi_gaussian_model(x, baseline, sorted_params)
    residual_arr = y - model
    rss = float(np.sum(residual_arr ** 2))
    if not np.isfinite(rss) or rss <= 0:
        rss = float(np.finfo(float).tiny)

    n_points = int(x.size)
    n_params = int((1 if fit_baseline else 0) + 3 * n_components)
    rss_norm = max(rss / max(n_points, 1), float(np.finfo(float).tiny))
    bic = float(n_points * np.log(rss_norm) + n_params * np.log(max(n_points, 1)))
    aic = float(n_points * np.log(rss_norm) + 2 * n_params)

    fit_noise = _estimate_noise_sigma(residual_arr)
    if np.isfinite(noise_sigma) and noise_sigma > 0:
        fit_noise = max(fit_noise, 0.5 * float(noise_sigma))
    dof = max(1, n_points - n_params)
    rchi2 = float(rss / (dof * max(fit_noise ** 2, np.finfo(float).tiny)))

    return GaussianFitResult(
        success=True,
        auto_components=False,
        requested_components=n_components,
        n_components=n_components,
        baseline=baseline,
        baseline_err=float(baseline_err) if np.isfinite(baseline_err) else float("nan"),
        fit_baseline=bool(fit_baseline),
        baseline_fixed=baseline_fixed_local,
        components=components,
        bic=bic,
        aic=aic,
        rss=rss,
        rchi2=rchi2,
        noise_sigma=float(fit_noise),
        x=x.copy(),
        y=y.copy(),
        model=model,
        residual=residual_arr,
    )


def fit_gaussian_spectrum(
    x_values: np.ndarray,
    y_values: np.ndarray,
    n_components: Optional[int] = None,
    *,
    auto_components: bool = True,
    max_components: int = 3,
    fit_baseline: bool = False,
    baseline_fixed: float = 0.0,
    allow_negative: bool = False,
    noise_sigma: Optional[float] = None,
    min_sigma: float = 0.5,
    max_sigma: Optional[float] = None,
) -> GaussianFitResult:
    """
    Fit one or more Gaussian components to a 1D spectrum.

    Args:
        x_values: 1D x-axis values (channel or world coordinate).
        y_values: 1D intensity values.
        n_components: Fixed component count when auto_components=False.
        auto_components: If True, try 1..max_components and pick minimum BIC.
        max_components: Maximum components for auto mode.
        fit_baseline: If True, fit baseline as a free parameter.
        baseline_fixed: Fixed baseline value used when fit_baseline=False.
        allow_negative: Allow negative amplitudes (e.g., absorption lines).
        noise_sigma: Optional known RMS noise. If omitted, estimated robustly.
        min_sigma: Lower bound for sigma in x-axis units.
        max_sigma: Optional upper bound for sigma in x-axis units.

    Returns:
        GaussianFitResult with best-fit parameters and diagnostics.
    """
    x = np.asarray(x_values, dtype=float).reshape(-1)
    y = np.asarray(y_values, dtype=float).reshape(-1)
    if x.size != y.size:
        raise ValueError("x_values and y_values must have the same length")

    finite = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite) < 8:
        raise ValueError("Not enough finite samples for Gaussian fitting")
    x = x[finite]
    y = y[finite]

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    min_sigma = float(min_sigma)
    if min_sigma <= 0:
        raise ValueError("min_sigma must be positive")

    noise = float(noise_sigma) if noise_sigma is not None else _estimate_noise_sigma(y)
    if not np.isfinite(noise) or noise <= 0:
        noise = _estimate_noise_sigma(y)

    max_components = int(max_components)
    if max_components < 1:
        raise ValueError("max_components must be >= 1")

    baseline_fixed = float(baseline_fixed)
    n_base_params = 1 if fit_baseline else 0
    max_allowed_components = max(1, (x.size - (n_base_params + 1)) // 3)
    if auto_components:
        requested = min(max_components, max_allowed_components)
        candidate_counts = list(range(1, requested + 1))
    else:
        if n_components is None:
            raise ValueError("n_components is required when auto_components=False")
        requested = int(n_components)
        if requested < 1:
            raise ValueError("n_components must be >= 1")
        if requested > max_allowed_components:
            raise ValueError("n_components is too large for the available sample count")
        candidate_counts = [requested]

    candidates: List[GaussianFitResult] = []
    for n_comp in candidate_counts:
        fit = _fit_fixed_gaussians(
            x,
            y,
            n_comp,
            fit_baseline=fit_baseline,
            baseline_fixed=baseline_fixed,
            allow_negative=allow_negative,
            noise_sigma=noise,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
        )
        if fit is not None:
            candidates.append(fit)

    if not candidates:
        raise RuntimeError("Gaussian fitting failed for all tested component counts")

    best = min(candidates, key=lambda item: item.bic)

    # In auto mode, weak components can be pruned and refit.
    # In fixed mode, keep the exact requested component count.
    if auto_components:
        min_amp = 3.0 * best.noise_sigma
        strong = [
            (comp.amplitude, comp.center, comp.sigma)
            for comp in best.components
            if abs(comp.amplitude) >= min_amp and comp.sigma >= min_sigma
        ]
        if 1 <= len(strong) < best.n_components:
            refined = _fit_fixed_gaussians(
                x,
                y,
                len(strong),
                fit_baseline=fit_baseline,
                baseline_fixed=baseline_fixed,
                allow_negative=allow_negative,
                noise_sigma=noise,
                min_sigma=min_sigma,
                max_sigma=max_sigma,
                seed_components=strong,
            )
            if refined is not None:
                best = refined

    best.auto_components = bool(auto_components)
    best.requested_components = int(requested)
    return best


def export_spectrum(
    spectrum_data: np.ndarray,
    velocity_values: np.ndarray,
    output_path: str,
    xlabel: str,
    ylabel: str,
    metadata: Dict[str, Any]
) -> str:
    """
    Export spectrum data to a text file.

    Args:
        spectrum_data: 1D array of spectrum values.
        velocity_values: 1D array of velocity/frequency values.
        output_path: Path to save the file.
        xlabel: Label for the X axis (velocity/frequency).
        ylabel: Label for the Y axis (intensity).
        metadata: Dictionary of metadata to include in the header.

    Returns:
        The output file path.
    """
    header_lines = []

    # Metadata
    if 'filename' in metadata:
        header_lines.append(f"Source: {metadata['filename']}")

    if 'region_type' in metadata and metadata['region_type']:
        header_lines.append(f"Region: {metadata['region_type'].capitalize()}")
        for k, v in metadata.get('region_params', {}).items():
            header_lines.append(f"{k}: {v}")
    elif 'pixel_coord' in metadata:
        header_lines.append("Spectrum Type: Single Pixel")
        if 'world_coord' in metadata:
            header_lines.append(f"World Coordinate: {metadata['world_coord']}")
        header_lines.append(f"Pixel Coordinate: {metadata['pixel_coord']}")

    fit_info = metadata.get('fit_info_lines')
    fit_lines: List[str] = []
    if isinstance(fit_info, str):
        fit_lines = [line.strip() for line in fit_info.splitlines() if line.strip()]
    elif isinstance(fit_info, (list, tuple)):
        for item in fit_info:
            if item is None:
                continue
            text = str(item)
            for line in text.splitlines():
                line = line.strip()
                if line:
                    fit_lines.append(line)
    if fit_lines:
        header_lines.append("")
        header_lines.append("Gaussian Fit:")
        header_lines.extend(fit_lines)

    header_lines.extend([
        "",
        f"Column 1: {xlabel}",
        f"Column 2: {ylabel}",
        "------------------------------------"
    ])

    header = "\n".join([f"# {line}" for line in header_lines])

    data_to_save = np.column_stack((velocity_values, spectrum_data))

    np.savetxt(output_path, data_to_save, fmt='%.6g', delimiter='   ', header=header, comments='')

    return output_path


def get_region_spectrum(
    state: AppState,
    region: Dict[str, Any] | RegionSpec,
) -> Dict[str, Any]:
    """Region-averaged spectrum as a JSON-serializable payload (TF-303).

    `get_averaged_spectrum` is headless but returns bare arrays, which an
    action cannot hand back through a manifest. This wrapper accepts a region
    payload and returns plain lists plus the spectral unit.

    Args:
        state: AppState with data and WCS.
        region: RegionSpec, or a payload dict as used by `add_region`.

    Returns:
        ``{"velocity": [...], "spectrum": [...], "unit": "km/s", "n_channels": N}``
    """
    spec = region if isinstance(region, RegionSpec) else RegionSpec.from_dict(region)
    velocity_values, spectrum_values, unit_string = get_averaged_spectrum(state, spec)
    return {
        "velocity": [float(v) for v in np.asarray(velocity_values).ravel()],
        "spectrum": [float(v) for v in np.asarray(spectrum_values).ravel()],
        "unit": str(unit_string),
        "n_channels": int(np.asarray(spectrum_values).size),
    }


def fit_spectrum_gaussian(
    state: AppState,
    x: Optional[int] = None,
    y: Optional[int] = None,
    region: Optional[Dict[str, Any] | RegionSpec] = None,
    n_components: Optional[int] = None,
    **fit_kwargs: Any,
) -> "GaussianFitResult":
    """Fit Gaussians to a spectrum taken from *state* (TF-303).

    Chooses the spectrum source in this order: an explicit ``region`` (averaged
    spectrum), else the pixel at ``(x, y)``, else the cursor position that
    `get_spectrum` defaults to. `fit_gaussian_spectrum` itself takes bare
    arrays, which is not reachable from a manifest.
    """
    if region is not None:
        spec = region if isinstance(region, RegionSpec) else RegionSpec.from_dict(region)
        x_values, y_values, _unit = get_averaged_spectrum(state, spec)
    else:
        y_values = get_spectrum(state, x=x, y=y)
        x_values = spectral_axis_values(state, np.asarray(y_values).size)

    return fit_gaussian_spectrum(
        np.asarray(x_values, dtype=float),
        np.asarray(y_values, dtype=float),
        n_components=n_components,
        **fit_kwargs,
    )
