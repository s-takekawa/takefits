"""
Usecases: PyQt-free headless operations for FITS data processing.

This module provides usecase functions that operate on AppState without
requiring PyQt. These can be called headlessly for testing or scripting.
"""
from typing import Callable, Optional, Tuple, Literal, Dict, Any, List, Union
import numpy as np

from takefits.core.app_state import AppState, create_app_state, MarkerSpec, RegionSpec
from .regrid import RegridResult, compute_regrid
from .export import export_data_fits, export_figure
from .moment import MomentType, compute_moment, export_moment_fits, export_moment_map_fits
from .pv import (
    CirclePathGeometry,
    EllipsePathGeometry,
    PathSamples,
    PolylinePathGeometry,
    POSITION_ORIGIN_CENTER,
    POSITION_ORIGIN_START,
    PV_X_AXIS_PHI,
    PV_X_AXIS_POSITION,
    PV_SPLINE_BSPLINE,
    PV_SPLINE_CATMULL_ROM,
    PV_SPLINE_NONE,
    StraightPathGeometry,
    anchored_straight_line,
    bspline_densify,
    catmull_rom_densify,
    clamp_pv_smoothness,
    compute_pv,
    export_pv_fits,
    fraction_from_position,
    normalize_position_origin,
    normalize_pv_spline_type,
    normalize_pv_x_axis_mode,
    position_axis_bounds,
    position_from_fraction,
    sample_count_from_spacing,
    sample_path_points,
    set_pv_endpoints,
    straight_line_from_center,
)
from .visualization import export_moment_image, export_channel_map_image
from .smoothing import (
    SmoothingKernel,
    beam_unit_scale_for_target_resolution,
    compute_smoothed,
    compute_smoothed_to_resolution,
    apply_smoothing,
    apply_smoothing_to_resolution,
)
from .mask import (
    MaskCondition,
    MomentMaskAlgorithm,
    MaskPolarity,
    MomentMaskPreset,
    NoiseMethod,
    MOMENT_MASK_PRESETS,
    get_moment_mask_preset,
    estimate_noise_sigma,
    compute_masked,
    compute_moment_mask,
    apply_mask_threshold,
    apply_mask_external,
    apply_mask_moment_recipe,
    export_mask_fits,
)
from .spectrum import (
    GaussianFitComponent,
    GaussianFitResult,
    get_spectrum,
    get_averaged_spectrum,
    fit_gaussian_spectrum,
    export_spectrum,
)
from .baseline import (
    BaselineSubtractionResult,
    compute_polynomial_baseline_subtraction,
    apply_baseline_subtraction,
    export_baseline_model_fits,
)
from .channel_map import ChannelMapResult, compute_channel_map, channel_labels_to_world
from .cutout import CutoutResult, compute_cutout, export_cutout_fits
from .arithmetic import ArithmeticOp, compute_arithmetic, apply_arithmetic
from .unit_conversion import IntensityUnit, convert_intensity_unit, apply_unit_conversion
from .scaling import compute_scaled, apply_scaling
from .clump import (
    ClumpResult,
    run_clumpfind,
    run_fellwalker,
    run_dendrogram,
    check_scimes_availability,
    generate_catalog,
    export_clump_mask,
    export_clump_catalog,
)
from .utils import (
    world_to_pixel,
    pixel_to_world,
    axis_world_to_pixel,
    axis_pixel_to_world,
    world_bounds_to_pixel_bounds,
    parse_world_coordinate,
    get_axis_ctype,
    update_datamin_datamax_if_present,
)
from .annotations import (
    add_marker,
    add_region,
    clear_markers,
    clear_regions,
    delete_marker,
    delete_region,
    set_markers,
    set_regions,
    update_marker,
    update_region,
)


def load_fits_data(
    filepath: str,
    hdu: int = 0,
    compute_wcs: bool = True
) -> AppState:
    """
    Load a FITS file and return an AppState.

    This is a headless wrapper around logic.fits_loader.load_fits.

    Args:
        filepath: Path to the FITS file
        hdu: HDU index to load (default 0)
        compute_wcs: Whether to compute WCS (default True)

    Returns:
        AppState with loaded data, header, wcs, and spectral metadata
    """
    from takefits.core.io.fits import load_fits

    data, header, wcs, spectral_metadata = load_fits(filepath, compute_wcs=compute_wcs)

    return create_app_state(
        data=data,
        header=header,
        wcs=wcs,
        filepath=filepath,
        spectral_metadata=spectral_metadata,
    )


def set_slice(
    state: AppState,
    z: Optional[int] = None,
    s: Optional[int] = None
) -> AppState:
    """
    Update the current slice position in the state.

    Args:
        state: The AppState to update
        z: New z (channel) index, or None to keep current
        s: New s (4th axis) index, or None to keep current

    Returns:
        The updated AppState (same instance, modified in place)
    """
    if z is not None:
        if state.data is not None:
            n_channels = state.n_channels
            state.current_z = max(0, min(z, n_channels - 1))
        else:
            state.current_z = z

    if s is not None:
        if state.data is not None and state.has_4th_axis:
            n_s = state.data.shape[0]
            state.current_s = max(0, min(s, n_s - 1))
        else:
            state.current_s = s

    return state


def set_cursor(
    state: AppState,
    xpix: Optional[int] = None,
    ypix: Optional[int] = None,
    zpix: Optional[int] = None,
    spix: Optional[int] = None
) -> AppState:
    """
    Update the cursor position in the state.

    Args:
        state: The AppState to update
        xpix, ypix, zpix, spix: New pixel coordinates, or None to keep current

    Returns:
        The updated AppState (same instance, modified in place)
    """
    if xpix is not None:
        state.cursor.xpix = xpix
    if ypix is not None:
        state.cursor.ypix = ypix
    if zpix is not None:
        state.cursor.zpix = zpix
    if spix is not None:
        state.cursor.spix = spix
    return state


def set_integration_range(
    state: AppState,
    min_pix: Optional[int] = None,
    max_pix: Optional[int] = None,
    min_world: Optional[float] = None,
    max_world: Optional[float] = None
) -> AppState:
    """
    Set the integration range for moment map calculations.

    Args:
        state: The AppState to update
        min_pix, max_pix: Pixel range (takes precedence if both pixel and world given)
        min_world, max_world: World coordinate range (e.g., km/s)

    Returns:
        The updated AppState
    """
    if min_pix is not None:
        state.integ_min_pix = min_pix
    if max_pix is not None:
        state.integ_max_pix = max_pix
    if min_world is not None:
        state.integ_min = min_world
    if max_world is not None:
        state.integ_max = max_world
    return state


# ============================================================================
# Phase 9a: Tier 1 Usecases
# ============================================================================

# --------------------------------------------------------------------------
# Color Settings
# --------------------------------------------------------------------------

def set_color_settings(
    state: AppState,
    plane: str = 'xy',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: Optional[str] = None,
    log_scale: Optional[bool] = None,
    gamma: Optional[float] = None,
    invert_cmap: Optional[bool] = None
) -> AppState:
    """
    Update color/display settings for a view plane.

    Args:
        state: The AppState to update
        plane: Which plane to update ('xy', 'xz', 'zy')
        vmin: Minimum value for color normalization
        vmax: Maximum value for color normalization
        cmap: Colormap name (e.g., 'viridis', 'inferno')
        log_scale: Whether to use logarithmic scaling
        gamma: Gamma correction factor (1.0 = linear)
        invert_cmap: Whether to invert the colormap

    Returns:
        The updated AppState
    """
    view_state = state.get_view_state(plane)

    if vmin is not None:
        view_state.vmin = vmin
    if vmax is not None:
        view_state.vmax = vmax
    if cmap is not None:
        view_state.cmap = cmap
    if log_scale is not None:
        view_state.log_scale = log_scale
    if gamma is not None:
        view_state.gamma = gamma
    if invert_cmap is not None:
        view_state.invert_cmap = invert_cmap

    return state


# --------------------------------------------------------------------------
# View Range
# --------------------------------------------------------------------------

def set_view_range(
    state: AppState,
    plane: str = 'xy',
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None
) -> AppState:
    """
    Set view range for a plane (pixel coordinates).

    Args:
        state: The AppState to update
        plane: Which plane to update ('xy', 'xz', 'zy')
        xlim: X-axis limits as (min, max) in pixel coordinates
        ylim: Y-axis limits as (min, max) in pixel coordinates

    Returns:
        The updated AppState
    """
    view_state = state.get_view_state(plane)

    if xlim is not None:
        view_state.xlim = xlim
    if ylim is not None:
        view_state.ylim = ylim

    return state
