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
from .utils import get_axis_ctype, parse_world_coordinate


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
            return np.array([data[y, x]])
        raise ValueError(f"Expected 3D data cube, got {data.ndim}D")

    # Clamp to valid range
    y = max(0, min(y, data.shape[1] - 1))
    x = max(0, min(x, data.shape[2] - 1))

    return data[:, y, x].copy()


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
        data_cube = data_cube[0]

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

    # --- 2. Create a small sub-cube based on the bounding box ---
    x_start, x_end = int(np.floor(x_min)), int(np.ceil(x_max))
    y_start, y_end = int(np.floor(y_min)), int(np.ceil(y_max))

    x_start = max(0, x_start)
    y_start = max(0, y_start)
    x_end = min(width, x_end)
    y_end = min(height, y_end)

    if x_start >= x_end or y_start >= y_end:
        raise ValueError("Region is outside data bounds")

    sub_cube = data_cube[:, y_start:y_end, x_start:x_end]

    # --- 3. Create a local mask for the sub-cube ---
    y_grid, x_grid = np.indices(sub_cube.shape[1:])
    x_grid = x_grid + x_start
    y_grid = y_grid + y_start

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

    # --- 4. Calculate average ---
    mask_3d = np.broadcast_to(local_mask, sub_cube.shape)

    # Apply NAN masking if needed
    if np.ma.is_masked(sub_cube):
        mask_3d = mask_3d & (~sub_cube.mask)
    else:
        # Handle NaNs in numpy array
        nan_mask = np.isnan(sub_cube)
        mask_3d = mask_3d & (~nan_mask)

    masked_data = np.ma.array(sub_cube, mask=~mask_3d)
    spectrum = masked_data.mean(axis=(1, 2))

    # --- 5. Clean up spectrum ---
    if isinstance(spectrum, np.ma.MaskedArray):
        spectrum = spectrum.filled(np.nan)

    # --- 6. Get velocity axis ---
    spectrum_len = len(spectrum)
    if state.wcs and state.wcs.wcs.naxis >= 3:
        # Generate pixel indices for spectral axis
        spec_axis = 2  # Default 3rd axis
        # Try to find spectral axis index
        for i, ctype in enumerate(state.wcs.wcs.ctype):
            if any(x in ctype.upper() for x in ['VEL', 'FREQ', 'VRAD', 'VOPT']):
                spec_axis = i
                break

        pix_coords = np.arange(spectrum_len)
        # Convert to world coordinates (spectral only)
        # This is a bit simplified; ideally we use pixel_to_world_values logic
        # But we can use crval/cdelt/crpix approximation if WCS is linear,
        # or use WCS methods if complex.
        # Fallback to linear approx for headless speed and simplicity in this bridging phase
        crval = state.wcs.wcs.crval[spec_axis]
        cdelt = state.wcs.wcs.cdelt[spec_axis]
        crpix = state.wcs.wcs.crpix[spec_axis]
        # X = (p - crpix) * cdelt + crval
        velocity_values = (pix_coords - (crpix - 1)) * cdelt + crval

        # Unit
        unit = state.wcs.wcs.cunit[spec_axis].to_string() if state.wcs.wcs.cunit else "unknown"
    else:
        velocity_values = np.arange(spectrum_len)
        unit = "pixel"

    return velocity_values, spectrum, unit


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
