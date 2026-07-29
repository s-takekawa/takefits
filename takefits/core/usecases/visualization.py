"""Headless visualization usecases."""
from __future__ import annotations

from typing import Optional, Tuple, Union, List

import numpy as np

from takefits.core.app_state import AppState
from takefits.core.usecases.moment import compute_moment
from takefits.core.usecases.channel_map import compute_channel_map, channel_labels_to_world
from takefits.core.usecases.export import export_figure
from takefits.core.usecases.utils import create_2d_header_from_3d
from takefits.core.config import ConfigManager
from takefits.core.fonts import resolve_mpl_font_family
from takefits.core.colorbar_layout import (
    compute_colorbar_geometry,
    orientation_for_placement,
)
from takefits.core.plotting.annotations import (
    draw_markers_on_axes,
    draw_regions_on_axes,
)
from takefits.core.plotting.beam import draw_beam_on_axes
from takefits.core.plotting.contours import draw_contour_specs_on_axes
from takefits.core.usecases.render_config import (
    colorbar_auto_layout_requested,
    resolve_render_config,
)
from takefits.logic.data_tools import ensure_operation_memory_budget
from astropy.wcs import WCS
import astropy.units as u


def _get_pyplot():
    import matplotlib.pyplot as plt

    return plt


def _channel_map_global_extrema(images) -> Tuple[float, float]:
    """Return exact finite extrema without stacking every retained panel."""
    data_min = np.inf
    data_max = -np.inf
    found = False
    for image in images:
        if image is None:
            continue
        array = np.asanyarray(image)
        if array.size == 0:
            continue
        finite = np.isfinite(array)
        if not np.any(finite):
            continue
        tile_min = float(np.min(array, where=finite, initial=np.inf))
        tile_max = float(np.max(array, where=finite, initial=-np.inf))
        data_min = min(data_min, tile_min)
        data_max = max(data_max, tile_max)
        found = True
    if not found:
        return np.nan, np.nan
    return data_min, data_max


def _channel_map_figure_working_bytes(
    *,
    n_images: int,
    nrows: int,
    ncols: int,
    dpi: int,
    figsize: Tuple[float, float],
) -> int:
    """Estimate Matplotlib canvas and per-panel artist storage."""
    width_in, height_in = float(figsize[0]), float(figsize[1])
    dpi_value = int(dpi)
    if (
        not np.isfinite(width_in)
        or not np.isfinite(height_in)
        or width_in <= 0.0
        or height_in <= 0.0
        or dpi_value <= 0
    ):
        raise ValueError("Channel Map figure size and DPI must be positive and finite")
    width_px = max(1, int(np.ceil(width_in * dpi_value)))
    height_px = max(1, int(np.ceil(height_in * dpi_value)))
    rgba_canvas = width_px * height_px * 4
    # Rendering/export can retain an Agg RGBA canvas and a second encoded or
    # compositing buffer. Axes, ticks, labels, and image artists also have a
    # nontrivial per-panel Python footprint.
    artist_bytes = max(0, int(n_images)) * 16 * 1024
    return int(2 * rgba_canvas + artist_bytes)


def _colorbar_decoration_overhang(fig, cax):
    """Pixels by which the colorbar's labels spill outside its own axes box.

    Same measurement as the GUI's `_colorbar_decoration_overhang`; it needs a
    realized renderer, so the caller draws first.
    """
    try:
        renderer = fig.canvas.get_renderer()
        tight_bbox = cax.get_tightbbox(renderer)
        axes_bbox = cax.bbox
        return (
            max(0.0, float(axes_bbox.x0) - float(tight_bbox.x0)),
            max(0.0, float(tight_bbox.x1) - float(axes_bbox.x1)),
            max(0.0, float(axes_bbox.y0) - float(tight_bbox.y0)),
            max(0.0, float(tight_bbox.y1) - float(axes_bbox.y1)),
        )
    except Exception:
        return (0.0, 0.0, 0.0, 0.0)


def _apply_colorbar_auto_layout(dm, fig, config, overrides=None) -> None:
    """Place the colorbar with the same semantic layout the GUI uses.

    `core/colorbar_layout.py` is Qt-free, so the headless renderer can run the
    identical computation, including the GUI's rule that an `inside-*` bar over
    a coordinate overlay reserves the larger of the configured margin and the
    measured label overhang.
    """
    cax = getattr(dm, 'cax', None)
    ax = getattr(dm, 'ax', None)
    if cax is None or ax is None:
        return

    placement = str(config.get('colorbar_placement', 'right') or 'right')
    gap_px, gap_x_px, gap_y_px = _resolved_colorbar_gaps(config, overrides)

    # Mirror the GUI: a non-native coordinate overlay reserves the strip its
    # tick labels and titles occupy, so the bar does not land on top of them.
    # An inside-* bar reserves the larger of that margin and its own measured
    # label overhang, which needs a realized renderer.
    if bool(getattr(dm, 'grid_overlay_active', False)):
        normalized = placement.strip().lower()
        right_margin = float(getattr(dm, 'grid_overlay_right_margin_px', 96.0))
        top_margin = float(getattr(dm, 'grid_overlay_top_margin_px', 64.0))
        if normalized == 'right':
            gap_x_px = float(gap_x_px) + right_margin
        elif normalized == 'top':
            gap_y_px = float(gap_y_px) + top_margin
        elif normalized in ('inside-right', 'inside-top'):
            try:
                fig.canvas.draw()
            except Exception:
                pass
            (
                _left_decoration,
                right_decoration,
                _bottom_decoration,
                top_decoration,
            ) = _colorbar_decoration_overhang(fig, cax)
            if normalized == 'inside-right':
                gap_x_px = float(gap_x_px) + max(right_margin, right_decoration)
            else:
                gap_y_px = float(gap_y_px) + max(top_margin, top_decoration)

    try:
        ax_bounds = tuple(float(v) for v in ax.get_position().bounds)
        fig_w_px = float(fig.bbox.width)
        fig_h_px = float(fig.bbox.height)
    except Exception:
        return
    if len(ax_bounds) != 4 or ax_bounds[2] <= 0.0 or ax_bounds[3] <= 0.0:
        return

    pos_x, pos_y, width, height, _orientation = compute_colorbar_geometry(
        ax_bounds,
        fig_w_px,
        fig_h_px,
        placement=placement,
        align=str(config.get('colorbar_align', 'center') or 'center'),
        gap_px=gap_px,
        gap_x_px=gap_x_px,
        gap_y_px=gap_y_px,
        thickness_px=config.get('colorbar_thickness_px', 24.0),
        length_mode=str(config.get('colorbar_length_mode', 'ratio') or 'ratio'),
        length_value=config.get('colorbar_length_value', 1.0),
    )
    cax.set_position([pos_x, pos_y, width, height])
    cax.set_gid('colorbar')


def _resolved_colorbar_gaps(config, overrides):
    """Gap in x/y px, with an explicit shared `colorbar_gap_px` taking priority."""
    gap_px = config.get('colorbar_gap_px', 24.0)
    overrides = overrides or {}
    gap_override = 'colorbar_gap_px' in overrides
    gap_x_px = (
        overrides['colorbar_gap_x_px']
        if 'colorbar_gap_x_px' in overrides
        else (gap_px if gap_override else config.get('colorbar_gap_x_px', gap_px))
    )
    gap_y_px = (
        overrides['colorbar_gap_y_px']
        if 'colorbar_gap_y_px' in overrides
        else (gap_px if gap_override else config.get('colorbar_gap_y_px', gap_px))
    )
    return gap_px, gap_x_px, gap_y_px


def _apply_channel_map_colorbar_layout(fig, cax, tile_axes, config, overrides=None):
    """Place a channel-map colorbar relative to the whole tile grid.

    The grid has no single image axes, so the reference rectangle is the union
    of the tile axes. Everything else is the shared
    `compute_colorbar_geometry` used by the GUI and the moment export.
    """
    if cax is None or not tile_axes:
        return

    try:
        bounds = [tuple(float(v) for v in ax.get_position().bounds)
                  for ax in tile_axes.values()]
        left = min(b[0] for b in bounds)
        bottom = min(b[1] for b in bounds)
        right = max(b[0] + b[2] for b in bounds)
        top = max(b[1] + b[3] for b in bounds)
        fig_w_px = float(fig.bbox.width)
        fig_h_px = float(fig.bbox.height)
    except Exception:
        return
    if right <= left or top <= bottom:
        return

    gap_px, gap_x_px, gap_y_px = _resolved_colorbar_gaps(config, overrides)
    pos_x, pos_y, width, height, _orientation = compute_colorbar_geometry(
        (left, bottom, right - left, top - bottom),
        fig_w_px,
        fig_h_px,
        placement=str(config.get('colorbar_placement', 'right') or 'right'),
        align=str(config.get('colorbar_align', 'center') or 'center'),
        gap_px=gap_px,
        gap_x_px=gap_x_px,
        gap_y_px=gap_y_px,
        thickness_px=config.get('colorbar_thickness_px', 24.0),
        length_mode=str(config.get('colorbar_length_mode', 'ratio') or 'ratio'),
        length_value=config.get('colorbar_length_value', 1.0),
    )
    cax.set_position([pos_x, pos_y, width, height])
    cax.set_gid('colorbar')


def _label_pv_spectral_axis(dm, header, config) -> None:
    """Show the PV spectral axis in the header's unit, with that unit labelled.

    Astropy normalizes a spectral WCS to SI internally, so a km/s header renders
    m/s tick values and `DisplayMap` labels the axis "Velocity" with no unit.
    Both are unhelpful in a publication figure.
    """
    ax = getattr(dm, 'ax', None)
    if ax is None or header is None:
        return

    for index in (0, 1):
        ctype = str(header.get(f'CTYPE{index + 1}', '') or '').upper()
        if ctype in ('OFFSET', 'PHI', ''):
            continue
        unit = str(header.get(f'CUNIT{index + 1}', '') or '').strip()
        try:
            coord = ax.coords[index]
        except Exception:
            continue
        if unit:
            try:
                coord.set_format_unit(unit)
            except Exception:
                pass
        base = 'Frequency' if 'FREQ' in ctype else 'Velocity'
        try:
            coord.set_axislabel(
                f'{base} [{unit}]' if unit else base,
                fontsize=config.get('axislabel_fontsize', 14),
                fontfamily=resolve_mpl_font_family(
                    config.get('axislabel_fontfamily', 'DejaVu Sans')
                ),
                color=config.get('axislabel_color', 'black'),
            )
        except Exception:
            pass


def _spectral_axis_values(state, n_channels: int):
    """World values along the spectral axis, falling back to channel index."""
    from takefits.core.usecases.spectrum import spectral_axis_values

    return spectral_axis_values(state, n_channels)


def register_custom_colormaps():
    """Register custom colormaps (Rainbow, Cool) if not present."""
    from matplotlib import colormaps as mpl_colormaps
    from takefits.core.custom_colormap import CustomColormap, ColorDefinitions

    # Check if 'Rainbow' is already registered
    if 'Rainbow' not in mpl_colormaps:
        rainbow_cdict = ColorDefinitions.rainbow()
        rainbow = CustomColormap('Rainbow', rainbow_cdict)
        mpl_colormaps.register(rainbow.get_colormap())
        mpl_colormaps.register(rainbow.reversed_colormap().get_colormap())

    if 'Cool' not in mpl_colormaps:
        cool_cdict = ColorDefinitions.cool()
        cool = CustomColormap('Cool', cool_cdict)
        mpl_colormaps.register(cool.get_colormap())
        mpl_colormaps.register(cool.reversed_colormap().get_colormap())


def export_moment_image(
    state: AppState,
    output_path: str,
    moment_type: str = "moment0",
    axis: int = 0,
    pixel_range: Optional[Tuple[float, float]] = None,
    world_range: Optional[Tuple[Union[float, str], Union[float, str]]] = None,
    cmap: Optional[str] = None,
    origin: str = "lower",
    dpi: int = 150,
    title: Optional[str] = None,
    grid: Optional[bool] = None,
    grid_frame: Optional[str] = None,
    grid_keep_native: Optional[bool] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Optional[Tuple[float, float]] = None,
    draw_markers: bool = True,
    draw_regions: bool = True,
    draw_beam: bool = True,
    draw_contours: bool = True,
) -> str:
    """
    Compute a moment map and export it as an image using CLI/GUI styling.

    Args:
        state: AppState with data.
        output_path: Path to save PNG/PDF/etc.
        moment_type: Type of moment to compute (moment0, moment1, etc.)
        axis: Axis to integrate along (0=z).
        pixel_range: Integration range in pixels.
        world_range: Integration range in world coords.
        cmap: Colormap name. ``None`` uses a ``colorscale`` render override if
            one is set, else the historical headless default ``viridis``.
        origin: Image origin ('lower' or 'upper').
        dpi: Output resolution.
        title: Optional title for the plot.
        grid: Override the WCS coordinate-grid overlay. When ``None`` the value
            falls back to ``state.display_grid`` (TF-404).
        grid_frame: Override the display frame followed by the XY grid. When
            ``None`` this falls back to ``state.display_grid_frame`` (TF-407).
        grid_keep_native: Override whether the native grid remains visible
            beneath a non-native XY overlay.
        vmin: Lower intensity limit. ``None`` keeps the renderer's autoscale.
        vmax: Upper intensity limit. ``None`` keeps the renderer's autoscale.
        figsize: Figure size in inches. Defaults to ``(8, 6)``.
        draw_markers: Draw ``state.markers`` for the XY plane onto the image
            (TF-302). Set False for a bare map.
        draw_regions: Draw ``state.regions`` for the XY plane onto the image
            (TF-302). Set False for a bare map.
        draw_beam: Draw the HPBW beam ellipse when the header carries one
            (TF-303), matching the GUI's XY viewer.
        draw_contours: Draw ``state.contours`` overlays (TF-303).

    Returns:
        Path to saved file.
    """
    # 1. Compute moment map data
    moment_data = compute_moment(
        state=state,
        moment_type=moment_type,
        axis=axis,
        pixel_range=pixel_range,
        world_range=world_range
    )
    
    # 2. Prepare 2D Header/WCS
    if state.header:
        header_2d = create_2d_header_from_3d(state.header, axis_to_drop=axis)
        try:
            wcs_2d = WCS(header_2d)
        except Exception:
            wcs_2d = None
    else:
        header_2d = None
        wcs_2d = None

    # 3. Configure DisplayMap
    # Load stored config, then apply the state's render overrides (TF-302) so
    # tick/label/font/colorbar styling is action- and CLI-driveable.
    config_manager = ConfigManager()
    config = resolve_render_config(state, config_manager.config)

    # An explicit cmap wins; otherwise a `colorscale` render override applies;
    # otherwise keep the historical headless default of viridis.
    if cmap:
        config['colorscale'] = cmap
    elif 'colorscale' not in (getattr(state, 'render_config', None) or {}):
        config['colorscale'] = 'viridis'

    # Semantic colorbar placement is opt-in so existing exports keep their
    # manual `cbar_pos_*` rectangle. Deriving the orientation before DisplayMap
    # builds the colorbar avoids rebuilding it afterwards.
    auto_colorbar = colorbar_auto_layout_requested(state)
    if auto_colorbar:
        config['colorbar_orientation'] = orientation_for_placement(
            config.get('colorbar_placement', 'right'),
            fallback=config.get('colorbar_orientation', 'vertical'),
        )

    # Coordinate grid (TF-404): explicit arg wins, else mirror the app state so
    # the headless render matches what the GUI would show.
    grid_on = grid if grid is not None else bool(getattr(state, 'display_grid', False))
    config['grid_visible'] = bool(grid_on)
    config['grid_frame'] = (
        grid_frame
        if grid_frame is not None
        else str(getattr(state, 'display_grid_frame', 'native') or 'native')
    )
    config['grid_keep_native'] = (
        bool(grid_keep_native)
        if grid_keep_native is not None
        else bool(getattr(state, 'display_grid_keep_native', True))
    )
    if title:
        # A non-native overlay uses the top edge for longitude tick labels.
        # Those numeric labels remain outside even when the descriptive axis
        # title is inside, so a titled export always needs the larger strip.
        # The reservation is inert for native grids.
        required_top_margin = 120.0
        config['grid_overlay_top_margin_px'] = max(
            required_top_margin,
            float(
                config.get(
                    'grid_overlay_top_margin_px',
                    required_top_margin,
                )
            ),
        )

    # Only force white background if user hasn't customized it? 
    # Actually, for publication plots (headless), user likely *wants* config settings OR specific overrides.
    # We will let the config file dictate styles unless explicitly overridden by CLI args.
    
    # Note: display_map.py uses config keys directly.
    # The 'config' dict now contains everything from config.yaml.
    
    # 4. Render
    # `figsize` defaults to the historical 8x6; config `figure_width/height` are
    # UI window dimensions and deliberately not reused here.
    plt = _get_pyplot()
    fig = plt.figure(figsize=tuple(figsize) if figsize else (8, 6))

    # Create DisplayMap instance
    # DisplayMap handles WCSAxes, colorbars, ticks, etc.
    from takefits.core.plotting.display_map import DisplayMap

    dm = DisplayMap(moment_data, header_2d, wcs_2d, config)

    # Intensity limits: DisplayMap.display() applies default_cmin/cmax to the
    # image, so overriding them here also keeps the colorbar consistent.
    if vmin is not None:
        dm.default_cmin = float(vmin)
    if vmax is not None:
        dm.default_cmax = float(vmax)

    # We display on 'xy' plane because we have reduced it to 2D
    dm.display(fig, plane='xy')

    # Reposition after display so the axes bounds are final.
    if auto_colorbar:
        _apply_colorbar_auto_layout(
            dm, fig, config, getattr(state, 'render_config', None)
        )

    # Contour overlays (TF-303): the image's own contours, and/or external
    # FITS overlaid by world coordinate. Drawn under the beam and annotations.
    if draw_contours and getattr(state, 'contours', None):
        draw_contour_specs_on_axes(
            dm.ax,
            state.contours,
            data=moment_data,
            target_wcs=wcs_2d,
            plane='xy',
        )

    # HPBW beam, as the GUI draws on its XY viewer. No-op when the header
    # carries no usable beam, so non-interferometric data is unaffected.
    if draw_beam:
        draw_beam_on_axes(dm.ax, header_2d if header_2d is not None else state.header, config)

    # Annotations live in AppState and are drawn with the same artists the GUI
    # uses, so a CLI export matches what the viewer shows (TF-302). Regions go
    # first so markers stay on top, matching the GUI's zorder.
    if draw_regions and getattr(state, 'regions', None):
        draw_regions_on_axes(dm.ax, state.regions, plane='xy')

    if draw_markers and getattr(state, 'markers', None):
        draw_markers_on_axes(
            dm.ax,
            state.markers,
            plane='xy',
            wcs=wcs_2d,
            shape=getattr(moment_data, 'shape', None),
        )

    if title:
        fig.suptitle(title, y=0.95)

    # 5. Save
    try:
        export_figure(fig, output_path, dpi=dpi)
    finally:
        plt.close(fig)
        
    return output_path


def export_channel_map_image(
    state: AppState,
    output_path: str,
    start_channel: Optional[float] = None,
    end_channel: Optional[float] = None,
    interval: float = 1.0,
    mode: str = "average",
    axis: int = 0,
    ncols: int = 4,
    cmap: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    start_world: Optional[float] = None,
    end_world: Optional[float] = None,
    interval_world: Optional[float] = None,
    dpi: int = 150,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    draw_markers: bool = True,
    draw_regions: bool = True,
    draw_beam: bool = True,
    draw_contours: bool = True,
    marker_plane_prefix: str = "channel_xy",
) -> str:
    """
    Export a specific channel map or grid of channel maps to an image file.

    Args:
        state: Application state
        output_path: Path to save image
        start_channel: First channel index (0-based)
        end_channel: Last channel index (0-based)
        interval: Step size for channel grid
        mode: 'average' (binning) or 'slice' (step)
        axis: WCS axis index for spectral dimension (default 0)
        ncols: Number of columns in grid
        cmap: Colormap name (default: Rainbow)
        vmin: Minimum intensity value
        vmax: Maximum intensity value
        dpi: Output DPI
        figsize: Figure size in inches. Defaults to ``(ncols * 3, nrows * 3)``.
        title: Optional figure title.
        draw_markers: Draw ``state.markers`` onto the tiles (TF-302).
        draw_regions: Draw ``state.regions`` onto the tiles (TF-302).
        draw_beam: Draw the HPBW beam ellipse on one tile (TF-303). The
            tile follows the GUI's bottom-left panel unless
            ``chmap_beam_tile`` overrides it.
        marker_plane_prefix: Plane-id prefix identifying channel-map
            annotations. A marker on ``"<prefix>_global_<i>"`` is drawn on tile
            ``i`` only, matching the GUI's page-independent channel planes; one
            on the bare ``"<prefix>"`` is drawn on every tile.

    Returns:
        Path to saved file
    """
    register_custom_colormaps()

    # `cmap` stays None here so a `colorscale` render override can win; the
    # historical Rainbow default is applied further down only when neither an
    # explicit cmap nor an override is present.

    # Convert world coordinates to channels if provided
    if start_world is not None and end_world is not None and state.wcs:
        try:
             # Assume spectral axis is axis 2 (0-based index 2 for 3rd axis) for typical cubes
             # We need to construct a world coordinate array.
             # Get CRVALs for spatial axes to fix them
             crval1 = state.header.get('CRVAL1', 0)
             crval2 = state.header.get('CRVAL2', 0)
             
             # Check if CUNIT3 is 'm/s' and input is likely km/s?
             # User standard: "interval 25 km/s code [-150, 150] km/s"
             # If WCS is m/s, we must scale input 1000x.
             cunit3 = state.header.get('CUNIT3', '').lower()
             scale_factor = 1.0
             if 'm/s' in cunit3 and 'km/s' not in cunit3 and abs(start_world) < 10000:
                  # Heuristic: if WCS is m/s but input is small, assume input is km/s
                  scale_factor = 1000.0

             w_start = start_world * scale_factor
             w_end = end_world * scale_factor
             
             # Convert to pixel
             # wcs_world2pix takes (ra, dec, vel) usually
             # We want the pixel index along the spectral axis.
             # We can probe the spectral axis.
             
             # Safe Conversion: Use spectral_coord conversion if possible, or full wcs
             coords_start = [[crval1, crval2, w_start]]
             coords_end = [[crval1, crval2, w_end]]
             
             pix_start = state.wcs.wcs_world2pix(coords_start, 0)[0][2]
             pix_end = state.wcs.wcs_world2pix(coords_end, 0)[0][2]
             
             start_channel = min(pix_start, pix_end)
             end_channel = max(pix_start, pix_end)
             
             if interval_world is not None:
                 # Calculate interval in pixels
                 # Estimate pixel scale at the center velocity?
                 # Or just diff between start and start+interval
                 w_next = (start_world + interval_world) * scale_factor
                 coords_next = [[crval1, crval2, w_next]]
                 pix_next = state.wcs.wcs_world2pix(coords_next, 0)[0][2]
                 interval = abs(pix_next - pix_start)
                 
             # Round to nearest integer for safe slicing/averaging?
             # compute_channel_map handles floats?
             # Usually best to round for clear boundaries unless sub-pixel
             # But 'compute_channel_map' might expect floats for 'slice' mode interpolation?
             # Existing usage implies int/float.
             # Let's leave as float for precision, but compute_channel_map likely casts or uses slicing.
             
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to convert world coordinates: {e}")

    # 1. Compute channel maps
    result = compute_channel_map(
        state=state,
        start_channel=start_channel,
        end_channel=end_channel,
        interval=interval,
        mode=mode,
        axis=axis
    )
    
    images = result.images
    raw_labels = list(getattr(result, "display_labels", result.labels))
    
    # Convert labels to world coordinates if WCS exists
    # If using world coords, we want format like "-150.0"
    # Helper uses axis 0, 1, 2 depending on WCS structure.
    # We need to identify the spectral axis index in the WCS object.
    # 'axis' argument to this function is the DATA axis (usually 0=z).
    # In WCS list, axis order is reversed (x, y, z).
    # So data axis 0 corresponds to WCS axis 2 (usually).
    
    wcs_axis_index = state.data.ndim - 1 - axis if state.data is not None else 2
    
    # However, WCS object might be 2D or 3D.
    # state.wcs usually matches data dimensions.
    
    labels = channel_labels_to_world(state, raw_labels, axis=wcs_axis_index)
    n_images = len(images)
    
    if n_images == 0:
        raise ValueError("No channel maps generated with given parameters.")
        
    # 2. Setup Grid
    # Load config for channel maps (colors, fonts), then apply the state's
    # render overrides (TF-302).
    config_manager = ConfigManager()
    config = resolve_render_config(state, config_manager.config)
    # Explicit cmap wins; then a `colorscale` render override; then the
    # historical Rainbow default for this export.
    if cmap:
        effective_cmap = cmap
    elif 'colorscale' in (getattr(state, 'render_config', None) or {}):
        effective_cmap = config.get('colorscale', 'Rainbow')
    else:
        effective_cmap = 'Rainbow'
    nrows = int(np.ceil(n_images / ncols))

    figure_size = tuple(figsize) if figsize else (ncols * 3, nrows * 3)
    figure_working_bytes = _channel_map_figure_working_bytes(
        n_images=n_images,
        nrows=nrows,
        ncols=ncols,
        dpi=dpi,
        figsize=figure_size,
    )
    ensure_operation_memory_budget(
        figure_working_bytes,
        operation_name="Channel Map image export",
        guidance=(
            "Export fewer panels, use a smaller figure size or DPI, or split "
            "the channel range across multiple images."
        ),
    )

    plt = _get_pyplot()
    fig = plt.figure(figsize=figure_size)
    gs = fig.add_gridspec(nrows, ncols)

    
    # Global normalization. Scan panels one at a time: ``images`` already
    # retains every result, so stacking them here would transiently duplicate
    # the complete Channel Map result set.
    if vmin is None or vmax is None:
        global_min, global_max = _channel_map_global_extrema(images)
        if vmin is None:
            vmin = global_min
        if vmax is None:
            vmax = global_max

    # Prepare WCS projection
    wcs = state.wcs
    slices = None
    if wcs:
        # Determine slices based on axis
        # axis=0 (z) -> xy plane (plane_num=0)
        # axis=1 (y) -> xz plane (plane_num=1)
        # axis=2 (x) -> zy plane (plane_num=2)
        naxis = wcs.naxis
        if naxis == 3:
            if axis == 0: slices = ('x', 'y', 0)
            elif axis == 1: slices = ('x', 0, 'y')
            elif axis == 2: slices = (0, 'y', 'x')
        elif naxis == 4:
            if axis == 0: slices = ('x', 'y', 0, 0)
            elif axis == 1: slices = ('x', 0, 'y', 0)
            elif axis == 2: slices = (0, 'y', 'x', 0)

    # Config values
    tick_color = config.get('tick_color', 'black')
    tick_width = config.get('tick_width', 1)
    tick_labelsize = config.get('tick_labelsize', 10)
    tick_labelcolor = config.get('tick_labelcolor', 'black')
    tick_font = resolve_mpl_font_family(config.get('tick_font', 'DejaVu Sans'))
    axislabel_fontfamily = resolve_mpl_font_family(config.get('axislabel_fontfamily', 'DejaVu Sans'))
    tick_direction = config.get('tick_direction', 'out')
    tick_length = config.get('tick_length', 4)
    tick_pad_x = config.get('tick_pad_x', 5)
    tick_pad_y = config.get('tick_pad_y', 5)
    
    im_list = []
    # Tile axes keyed by their page-independent global index, so annotations
    # can follow the GUI's `<prefix>_global_<i>` channel planes.
    tile_axes = {}
    # Axis-label text per displayed direction, for the shared-label style.
    shared_axis_labels = {}
    
    for i in range(nrows):
        for j in range(ncols):
            idx = i * ncols + j
            
            # WCSAxes if available
            if wcs and slices:
                ax = fig.add_subplot(gs[i, j], projection=wcs, slices=slices)
            else:
                ax = fig.add_subplot(gs[i, j])
                
            if idx < n_images:
                img = images[idx]
                lbl = labels[idx]
                
                # Aspect handling
                if axis == 0: aspect = 'equal' # xy
                else: aspect = 'auto'
                
                im = ax.imshow(img, cmap=effective_cmap, origin='lower', vmin=vmin, vmax=vmax, aspect=aspect)
                im_list.append(im)
                tile_axes[idx] = ax
                
                # Label
                if len(lbl) == 3:
                    if mode == "integrate":
                        # If string, use as is. If float, format.
                        if isinstance(lbl[1], str): label_text = lbl[1]
                        else: label_text = f"{lbl[1]:.1f}"
                    elif mode == "slice":
                         # Single value
                         if isinstance(lbl[1], (float, int)): label_text = f"{lbl[1]:.1f}"
                         else: label_text = str(lbl[1])
                    else:
                        if isinstance(lbl[1], str): label_text = lbl[1]
                        else: label_text = f"{lbl[1]:.1f}"
                else:
                    label_text = str(lbl)
                
                # Use config font/color for label
                ch_label_font = resolve_mpl_font_family(config.get('ch_label_font', 'DejaVu Sans'))
                ch_label_color = config.get('ch_label_color', 'grey')
                ch_label_size = config.get('ch_label_size', 10)
                pos_x = config.get('pos_chlabel_x', 0.98)
                pos_y = config.get('pos_chlabel_y', 0.02)
                
                ax.text(pos_x, pos_y, label_text, transform=ax.transAxes, 
                        color=ch_label_color, fontsize=ch_label_size, fontfamily=ch_label_font,
                        va='bottom', ha='right', fontweight='bold')
                
                # Styling
                ax.tick_params(which='major', direction=tick_direction,
                               length=tick_length, color=tick_color,
                               width=tick_width, labelsize=tick_labelsize,
                               labelcolor=tick_labelcolor)
                
                for spine in ax.spines.values():
                    spine.set_linewidth(tick_width)
                    spine.set_color(tick_color)

                # Consistent tick behavior
                # Logic: Show ticks/labels only on the left edge (col 0) and bottom edge (bottom-most tile in column)
                
                if wcs:
                    # Identify X and Y axes indices in WCS
                    # slices has 'x', 'y' and integers
                    try:
                        x_wcs_axis = slices.index('x')
                        y_wcs_axis = slices.index('y')
                    except ValueError:
                        # Fallback if slices structure is unexpected
                        x_wcs_axis = 0
                        y_wcs_axis = 1

                    is_bottom_tile = (idx + ncols >= n_images)
                    is_left_tile = (j == 0)

                    
                    # Define label mapping (from DisplayMap)
                    coords_dict = {
                        'glon': 'Galactic Longitude',
                        'glat': 'Galactic Latitude',
                        'ra': 'Right Ascension',
                        'dec': 'Declination',
                        # Add frequency/velocity if needed for 3rd axis, though usually not shown on 2D map axes
                    }
                    
                    # Helper to apply format (hms vs deg) based on config/ctype
                    # Similar to DisplayMap.update_axes_format
                    def configure_coord_format(coord, ctype_index):
                        ctype_str = (state.header.get(f'CTYPE{ctype_index+1}', '')).upper() if state.header else ''
                        # Basic formatting logic
                        is_decimal = config.get('decimal', True)
                        
                        if ctype_str.startswith('RA'):
                            wrap_val = config.get('coord_wrap', 360.0)
                            if isinstance(wrap_val, (int, float)):
                                coord.set_coord_type('longitude', coord_wrap=float(wrap_val) * u.deg)
                            else:
                                coord.set_coord_type('longitude', coord_wrap=360.0 * u.deg)
                            
                            if is_decimal:
                                coord.set_format_unit(u.deg, decimal=True)
                            else:
                                coord.set_format_unit(u.hourangle, decimal=False)
                                
                        elif ctype_str.startswith('DEC'):
                             coord.set_coord_type('latitude')
                             if is_decimal:
                                 coord.set_format_unit(u.deg, decimal=True)
                             else:
                                 coord.set_format_unit(u.deg, decimal=False)
                        
                        elif any(keyword in ctype_str for keyword in ['GLON', 'GLAT', 'OFFSET']):
                            if 'GLON' in ctype_str:
                                wrap_val = config.get('coord_wrap', 180)
                                if isinstance(wrap_val, (int, float)):
                                     coord.set_coord_type('longitude', coord_wrap=float(wrap_val) * u.deg)
                                else:
                                     coord.set_coord_type('longitude', coord_wrap=180 * u.deg)
                            else: 
                                coord.set_coord_type('latitude')
                            
                            coord.set_format_unit('deg', decimal=is_decimal)

                    for k in range(len(slices)):
                        configure_coord_format(ax.coords[k], k)

                    # Apply Labels Globally to the axes object first (so the text is correct)
                    for alias, label in coords_dict.items():
                        if alias in ax.coords:
                            ax.coords[alias].set_axislabel(label,
                                fontsize=config.get('axislabel_fontsize', 14),
                                fontfamily=axislabel_fontfamily,
                                color=config.get('axislabel_color', 'black'))
                            # Remember which label belongs to which displayed
                            # direction, for the shared-axis-label style.
                            coord_index = list(ax.coords).index(ax.coords[alias])
                            if coord_index == x_wcs_axis:
                                shared_axis_labels.setdefault('x', label)
                            elif coord_index == y_wcs_axis:
                                shared_axis_labels.setdefault('y', label)

                    # Apply Visibility
                    for k in range(len(slices)):
                        coord = ax.coords[k]
                        
                        # Handle X Axis (Horizontal)
                        if k == x_wcs_axis:
                            coord.set_ticklabel(
                                rotation=config.get('tick_xlabelrotation', 0),
                                pad=tick_pad_x,
                                size=tick_labelsize,
                                color=tick_labelcolor,
                                fontfamily=tick_font,
                                exclude_overlapping=True,
                            )
                            if is_bottom_tile:
                                coord.set_ticklabel_visible(True)
                                coord.set_ticks_position(config.get('xticklabel_position', 'b'))
                                coord.set_ticklabel_position(config.get('xticklabel_position', 'b'))
                            else:
                                coord.set_axislabel('') # Hide label text on internal
                                coord.set_ticklabel_visible(False)
                        
                        # Handle Y Axis (Vertical)
                        elif k == y_wcs_axis:
                            coord.set_ticklabel(
                                rotation=config.get('tick_ylabelrotation', 0),
                                pad=tick_pad_y,
                                size=tick_labelsize,
                                color=tick_labelcolor,
                                fontfamily=tick_font,
                                exclude_overlapping=True,
                            )
                            if is_left_tile:
                                coord.set_ticklabel_visible(True)
                                coord.set_ticks_position(config.get('yticklabel_position', 'l'))
                                coord.set_ticklabel_position(config.get('yticklabel_position', 'l'))
                            else:
                                coord.set_axislabel('') # Hide label text on internal
                                coord.set_ticklabel_visible(False)
                        
                        # Handle Sliced Axes (should be hidden)
                        else:
                             coord.set_axislabel('')
                             coord.set_ticklabel_visible(False)
                             coord.set_ticks_visible(False)

                    # Ensure ticks themselves are visible everywhere

                    # Ensure ticks themselves are visible everywhere (often desired) or just edges?
                    # GUI "default_ticks_position" usually means ticks on all sides?
                    # If config says 'btlr', we put ticks everywhere.
                    ticks_pos = config.get('default_ticks_position', 'btlr')
                    for coord in ax.coords:
                        coord.set_ticks_position(ticks_pos)
                        coord.display_minor_ticks(True)

            else:
                ax.axis('off')

    # Publication style: one shared axis label per figure rather than one per
    # edge tile. Collect the per-tile labels first so the shared text matches
    # whatever frame the WCS is in, then clear them.
    if bool(config.get('chmap_shared_axislabels', False)):
        shared_x = shared_axis_labels.get('x', '')
        shared_y = shared_axis_labels.get('y', '')
        for tile_ax in tile_axes.values():
            for coord in tile_ax.coords:
                coord.set_axislabel('')

        label_kwargs = dict(
            fontsize=config.get('axislabel_fontsize', 14),
            fontfamily=resolve_mpl_font_family(
                config.get('axislabel_fontfamily', 'DejaVu Sans')
            ),
            color=config.get('axislabel_color', 'black'),
        )
        if shared_x:
            fig.supxlabel(shared_x, **label_kwargs)
        if shared_y:
            fig.supylabel(shared_y, **label_kwargs)

    # Contour overlays per tile (TF-303). A spec on the bare prefix or on
    # `<prefix>_global_<i>` follows the same plane rules as the annotations.
    prefix = str(marker_plane_prefix or 'channel_xy')
    if draw_contours and getattr(state, 'contours', None):
        tile_wcs = wcs.celestial if wcs and wcs.has_celestial else None
        for tile_index, tile_ax in tile_axes.items():
            for plane_id in (prefix, f'{prefix}_global_{int(tile_index)}'):
                draw_contour_specs_on_axes(
                    tile_ax,
                    state.contours,
                    data=images[tile_index],
                    target_wcs=tile_wcs,
                    plane=plane_id,
                )

    # HPBW beam on one panel, as the GUI does. It places the beam on the
    # bottom-left tile, so mirror that: the first tile of the last row.
    if draw_beam and tile_axes:
        beam_index = config.get('chmap_beam_tile')
        if beam_index is None:
            beam_index = ncols * (nrows - 1)
        beam_ax = tile_axes.get(int(beam_index))
        if beam_ax is not None:
            draw_beam_on_axes(beam_ax, state.header, config)

    # Annotations (TF-302). A marker/region on `<prefix>_global_<i>` belongs to
    # tile `i`, matching the GUI's page-independent channel planes; one on the
    # bare prefix is repeated on every tile (scale bars, source positions).
    for tile_index, tile_ax in tile_axes.items():
        tile_plane = f'{prefix}_global_{int(tile_index)}'
        tile_shape = getattr(images[tile_index], 'shape', None)
        for plane_id in (prefix, tile_plane):
            if draw_regions and getattr(state, 'regions', None):
                draw_regions_on_axes(tile_ax, state.regions, plane=plane_id)
            if draw_markers and getattr(state, 'markers', None):
                draw_markers_on_axes(
                    tile_ax,
                    state.markers,
                    plane=plane_id,
                    wcs=wcs.celestial if wcs and wcs.has_celestial else None,
                    shape=tile_shape,
                )

    # Global Axis Labels (Fake it by placing text on the figure edge?)
    # Or rely on WCSAxes labels on the edge subplots.
    # WCSAxes automatically adds labels if set_axislabel is not empty.
    # We cleared them above. Let's restore them just for the "center" of the edges?
    # Or just let every edge plot have the label? 
    # Usually "Declination" appears on the Y axis of the left column.
    
    # Let's simple enable axis labels on the edges where we enabled tick labels.
    # But this might be redundant if every row has "Declination".
    # Standard publication style: Label on the center-left and center-bottom.
    
    # We will let WCSAxes show labels on all edge plots for now, as that's robust.
    # Re-loop to set axis labels on edges?
    # Or just do it in the loop above.
    
    # Refined loop logic for edges:
    # We need to access ax.coords again.
    
    # Let's rely on standard WCS defaults but just hide them internally.
    # (The loop above mostly did this, but let's be explicit about text labels)
    
    # Colorbar
    # Create separate axes for colorbar like GUI
    cbar_pos_x = config.get('cbar_pos_x', 0.9)
    cbar_pos_y = config.get('cbar_pos_y', 0.11)
    cbar_width = config.get('cbar_width', 0.04)
    cbar_height = config.get('cbar_height', 0.77)
    
    # Note: These GUI positions are relative to window/figure. 
    # With tight_layout/constrained_layout, fixed axes might overlap or look weird.
    # However, user requested GUI style matching.
    # The GUI uses fixed axes positions for the colorbar on the figure.
    # If we do fig.add_axes([...]) it is absolute figure coordinates (0-1).
    
    # We used subplots inside GridSpec above.
    # To match GUI exactly, we might need to be careful.
    
    # Let's try adding it to the right of the grid using standard matplotlib first?
    # NO, user wants "SAME STYLE". GUI has it at fixed position on right.
    
    # But wait, GUI layout logic:
    # layout.addWidget(self.canvas, 2, 0, 1, 16)
    # The canvas is the widget. The figure is inside.
    # The config coords (0.9, 0.11) seem to be Figure coordinates.
    
    if im_list:
        cbar_orientation = config.get('colorbar_orientation', 'vertical')
        # With semantic placement the orientation follows it, so the colorbar
        # is built the right way round instead of being rebuilt afterwards.
        if colorbar_auto_layout_requested(state):
            cbar_orientation = orientation_for_placement(
                config.get('colorbar_placement', 'right'),
                fallback=cbar_orientation,
            )
        cax = fig.add_axes([cbar_pos_x, cbar_pos_y, cbar_width, cbar_height])
        
        cbar = fig.colorbar(im_list[0], cax=cax, orientation=cbar_orientation)
        
        # Colorbar Ticks
        cax.tick_params(
             axis='y', which='both',
             left=config.get('colorbar_tick_left', False),
             right=config.get('colorbar_tick_right', True),
             labelleft=config.get('colorbar_tick_labelleft', False),
             labelright=not config.get('colorbar_tick_labelleft', False),
             width=config.get('colorbar_tick_width', 1),
             length=config.get('colorbar_tick_length', 2),
             color=config.get('colorbar_tick_color', 'black'),
             direction=config.get('colorbar_tick_direction', 'out'),
             labelcolor=config.get('colorbar_tick_labelcolor', 'black')
        )
        # (Add x axis params if needed, omitted for brevity as vertical is standard)

        cbar_tick_params = {
            'labelfontfamily': resolve_mpl_font_family(
                config.get('colorbar_tick_labelfontfamily')
                or config.get('tick_font', 'DejaVu Sans')
            )
        }
        cbar_tick_labelsize = config.get('colorbar_tick_labelsize', None)
        if cbar_tick_labelsize is not None:
            cbar_tick_params['labelsize'] = cbar_tick_labelsize
        cax.tick_params(axis='both', which='both', **cbar_tick_params)

        cbar.outline.set_color(config.get('colorbar_tick_color', 'black'))
        cbar.outline.set_linewidth(config.get('colorbar_tick_width', 1))

        cbar_label = config.get('colorbar_label', None)
        if cbar_label:
            cbar.set_label(
                cbar_label,
                fontsize=config.get('colorbar_label_fontsize', 12),
                color=config.get('colorbar_label_color', 'black'),
                fontfamily=resolve_mpl_font_family(
                    config.get('colorbar_label_fontfamily', 'DejaVu Sans')
                ),
            )

    if im_list:
        right_margin = cbar_pos_x - 0.02 if cbar_pos_x < 1.0 else 0.88
    else:
        right_margin = 0.95
    right_margin = float(np.clip(right_margin, 0.55, 0.98))

    def _layout(key, fallback):
        value = config.get(key)
        return fallback if value is None else float(value)

    fig.subplots_adjust(
        left=_layout('chmap_left', 0.08),
        right=_layout('chmap_right', right_margin),
        bottom=_layout('chmap_bottom', 0.08),
        top=_layout('chmap_top', 0.95),
        wspace=_layout('chmap_wspace', 0.12),
        hspace=_layout('chmap_hspace', 0.12),
    )

    if title:
        fig.suptitle(
            title,
            fontsize=config.get('axislabel_fontsize', 14),
            fontfamily=resolve_mpl_font_family(
                config.get('axislabel_fontfamily', 'DejaVu Sans')
            ),
            color=config.get('axislabel_color', 'black'),
        )

    # Semantic colorbar placement across the whole tile grid (TF-302). Opt-in,
    # so exports that do not name a placement keep the manual rectangle above.
    if im_list and colorbar_auto_layout_requested(state):
        _apply_channel_map_colorbar_layout(
            fig, cax, tile_axes, config, getattr(state, 'render_config', None)
        )

    # 3. Save
    try:
        export_figure(fig, output_path, dpi=dpi)
    finally:
        plt.close(fig)
        
    return output_path


def export_pv_image(
    state: AppState,
    output_path: str,
    x0: Optional[float] = None,
    y0: Optional[float] = None,
    x1: Optional[float] = None,
    y1: Optional[float] = None,
    width: Optional[float] = None,
    is_swapped: bool = False,
    cmap: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    dpi: int = 150,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    draw_beam: bool = False,
    draw_contours: bool = True,
    **pv_kwargs,
) -> str:
    """Compute a PV diagram and export it as an image (TF-303).

    `compute_pv` and `export_pv_fits` existed, but a PV *figure* could only be
    made from the Qt-bound PV window. The position/velocity axes come from
    `build_pv_header`, the same construction `export_pv_fits` writes, so the
    image and the FITS describe identical axes.

    Args:
        state: AppState with data loaded.
        output_path: Path to save PNG/PDF/etc.
        x0, y0, x1, y1: Slice endpoints in pixel coordinates.
        width: Slit width in pixels.
        is_swapped: True puts velocity on the horizontal axis.
        cmap: Colormap. None uses a `colorscale` render override, else viridis.
        vmin, vmax: Intensity limits; None autoscales.
        dpi: Output resolution.
        figsize: Figure size in inches. Defaults to ``(8, 6)``.
        title: Optional figure title.
        draw_beam: Off by default; a beam ellipse is not meaningful on
            position-velocity axes.
        draw_contours: Draw contour specs whose plane is ``pv`` (TF-303).
        **pv_kwargs: Forwarded to `compute_pv` / `build_pv_header`
            (`position_origin`, `path_type`, `x_axis_mode`, `position_unit`, ...).

    Returns:
        Path to saved file.
    """
    from takefits.core.usecases.pv import (
        CirclePathGeometry,
        EllipsePathGeometry,
        PolylinePathGeometry,
        StraightPathGeometry,
        build_pv_header,
        compute_pv,
        normalize_pv_x_axis_mode,
        resolve_pv_path_geometry,
        sample_path_points,
    )

    geometry_keys = {
        "start_world",
        "end_world",
        "path_geometry",
        "path_type",
        "center",
        "semi_major_px",
        "semi_minor_px",
        "pa_rad",
        "start_phi_rad",
        "end_phi_rad",
        "vertices",
        "vertices_world",
        "spline_type",
        "smoothness",
        "smooth",
    }
    sampling_keys = {
        "num_samples",
        "sample_spacing_pix",
        "weight_mode",
        "sample_axis",
    }
    header_keys = {
        "history_entries",
        "position_origin",
        "path_length_px",
        "x_axis_mode",
        "phi_start_deg",
        "phi_end_deg",
        "position_unit",
    }
    supported_keys = geometry_keys | sampling_keys | header_keys
    unknown_keys = sorted(set(pv_kwargs) - supported_keys)
    if unknown_keys:
        names = ", ".join(unknown_keys)
        raise TypeError(f"Unsupported PV image argument(s): {names}")

    geometry_kwargs = {
        key: pv_kwargs[key] for key in geometry_keys if key in pv_kwargs
    }
    sampling_kwargs = {
        key: pv_kwargs[key] for key in sampling_keys if key in pv_kwargs
    }
    header_kwargs = {
        key: pv_kwargs[key] for key in header_keys if key in pv_kwargs
    }
    if "sample_axis" not in sampling_kwargs and "x_axis_mode" in header_kwargs:
        sampling_kwargs["sample_axis"] = header_kwargs["x_axis_mode"]
    path = resolve_pv_path_geometry(
        state,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        **geometry_kwargs,
    )

    pv_data = compute_pv(
        state=state,
        path_geometry=path,
        width=width,
        **sampling_kwargs,
    )
    if is_swapped:
        pv_data = np.asarray(pv_data).T

    samples = sample_path_points(
        path,
        num_samples=sampling_kwargs.get("num_samples"),
        sample_spacing_pix=sampling_kwargs.get("sample_spacing_pix"),
        sample_axis=sampling_kwargs.get("sample_axis", "position"),
    )
    endpoints = (
        float(samples.xs[0]),
        float(samples.ys[0]),
        float(samples.xs[-1]),
        float(samples.ys[-1]),
    )

    requested_path_type = geometry_kwargs.get("path_type")
    if requested_path_type is None:
        requested_path_type = {
            StraightPathGeometry: "straight",
            CirclePathGeometry: "circle",
            EllipsePathGeometry: "ellipse",
            PolylinePathGeometry: "polyline",
        }.get(type(path), "straight")
    header_kwargs.setdefault("path_type", requested_path_type)
    header_kwargs.setdefault("path_length_px", float(samples.length_px))
    header_kwargs.setdefault(
        "x_axis_mode", sampling_kwargs.get("sample_axis", "position")
    )
    if (
        isinstance(path, EllipsePathGeometry)
        and normalize_pv_x_axis_mode(header_kwargs["x_axis_mode"]) == "phi"
    ):
        header_kwargs.setdefault("phi_start_deg", np.degrees(path.start_phi_rad))
        header_kwargs.setdefault("phi_end_deg", np.degrees(path.end_phi_rad))

    data_2d, header_2d = build_pv_header(
        state,
        pv_data,
        *endpoints,
        is_swapped=is_swapped,
        **header_kwargs,
    )

    try:
        wcs_2d = WCS(header_2d)
    except Exception:
        wcs_2d = None

    config_manager = ConfigManager()
    config = resolve_render_config(state, config_manager.config)
    if cmap:
        config['colorscale'] = cmap
    elif 'colorscale' not in (getattr(state, 'render_config', None) or {}):
        config['colorscale'] = 'viridis'

    auto_colorbar = colorbar_auto_layout_requested(state)
    if auto_colorbar:
        config['colorbar_orientation'] = orientation_for_placement(
            config.get('colorbar_placement', 'right'),
            fallback=config.get('colorbar_orientation', 'vertical'),
        )

    plt = _get_pyplot()
    fig = plt.figure(figsize=tuple(figsize) if figsize else (8, 6))

    from takefits.core.plotting.display_map import DisplayMap

    dm = DisplayMap(data_2d, header_2d, wcs_2d, config)
    if vmin is not None:
        dm.default_cmin = float(vmin)
    if vmax is not None:
        dm.default_cmax = float(vmax)

    dm.display(fig, plane='xy')
    _label_pv_spectral_axis(dm, header_2d, config)

    # Contour specs on the `pv` plane overlay the diagram itself (TF-303).
    if draw_contours and getattr(state, 'contours', None):
        draw_contour_specs_on_axes(
            dm.ax, state.contours, data=data_2d, target_wcs=wcs_2d, plane='pv'
        )

    if auto_colorbar:
        _apply_colorbar_auto_layout(
            dm, fig, config, getattr(state, 'render_config', None)
        )

    if draw_beam:
        draw_beam_on_axes(dm.ax, header_2d, config)

    if title:
        fig.suptitle(title, y=0.95)

    try:
        export_figure(fig, output_path, dpi=dpi)
    finally:
        plt.close(fig)

    return output_path


def export_spectrum_image(
    state: AppState,
    output_path: str,
    x: Optional[int] = None,
    y: Optional[int] = None,
    region: Optional[dict] = None,
    dpi: int = 150,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    color: Optional[str] = None,
    linewidth: Optional[float] = None,
    drawstyle: str = "steps-mid",
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    show_zero_line: bool = True,
    fit: bool = False,
    fit_kwargs: Optional[dict] = None,
    fit_color: Optional[str] = None,
    fit_linewidth: Optional[float] = None,
) -> str:
    """Plot a spectrum and export it as an image (TF-303).

    `export_spectrum` only wrote a text table; the plotting lived in the
    Qt-bound spectrum tool. Tick/label/font styling comes from the same
    `set_render_config` overrides the image exports use.

    Args:
        state: AppState with data loaded.
        output_path: Path to save PNG/PDF/etc.
        x, y: Pixel position. Ignored when ``region`` is given.
        region: Region spec; averages the spectrum over it.
        dpi: Output resolution.
        figsize: Figure size in inches. Defaults to ``(8, 4.5)``.
        title: Optional figure title.
        xlabel, ylabel: Axis labels. Default to the spectral unit and BUNIT.
        color, linewidth: Line style. Default to the config tick colour and 1.2.
        drawstyle: Matplotlib drawstyle; ``steps-mid`` is the spectral default.
        xlim, ylim: Axis limits.
        show_zero_line: Draw a horizontal zero reference.
        fit: Overlay a Gaussian fit of the same spectrum. The model is drawn
            on a finely sampled grid, not at the channel centres.
        fit_kwargs: Forwarded to `fit_spectrum_gaussian`.
        fit_color, fit_linewidth: Style of the fit curve.

    Returns:
        Path to saved file.
    """
    from takefits.core.app_state import RegionSpec
    from takefits.core.usecases.spectrum import (
        fit_spectrum_gaussian,
        get_averaged_spectrum,
        get_spectrum,
    )

    if region is not None:
        spec = region if isinstance(region, RegionSpec) else RegionSpec.from_dict(region)
        x_values, y_values, unit_string = get_averaged_spectrum(state, spec)
        x_values = np.asarray(x_values, dtype=float)
    else:
        from takefits.core.usecases.spectrum import spectral_axis_unit

        y_values = get_spectrum(state, x=x, y=y)
        unit_string = spectral_axis_unit(
            state, np.asarray(y_values).size, fallback='channel'
        )
        x_values = _spectral_axis_values(state, np.asarray(y_values).size)
    y_values = np.asarray(y_values, dtype=float)

    config_manager = ConfigManager()
    config = resolve_render_config(state, config_manager.config)

    plt = _get_pyplot()
    fig = plt.figure(figsize=tuple(figsize) if figsize else (8, 4.5))
    fig.set_facecolor(config.get('fig_background_color', '#ececec'))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor(config.get('ax_background_color', 'white'))

    ax.plot(
        x_values,
        y_values,
        color=color or config.get('tick_color', 'black'),
        linewidth=1.2 if linewidth is None else float(linewidth),
        drawstyle=drawstyle,
    )
    if show_zero_line:
        ax.axhline(
            0.0,
            color=config.get('tick_color', 'black'),
            linewidth=0.6,
            linestyle='dotted',
        )

    if fit:
        result = fit_spectrum_gaussian(
            state, x=x, y=y, region=region, **(fit_kwargs or {})
        )
        # Plot the model on a fine grid: `result.model` is sampled only at the
        # channels, so a narrow line drawn through it looks angular.
        try:
            fit_x, fit_y = result.sample_curve()
        except Exception:
            fit_x, fit_y = None, None
        if fit_x is None or np.size(fit_y) == 0:
            model = getattr(result, 'model', None)
            if model is not None and np.size(model) == np.size(y_values):
                fit_x, fit_y = x_values, np.asarray(model, dtype=float)
        if fit_x is not None and np.size(fit_y):
            ax.plot(
                fit_x,
                np.asarray(fit_y, dtype=float),
                color=fit_color or 'red',
                linewidth=1.2 if fit_linewidth is None else float(fit_linewidth),
            )

    bunit = ''
    if state.header is not None:
        try:
            bunit = str(state.header.get('BUNIT', '') or '')
        except Exception:
            bunit = ''

    axislabel_kwargs = dict(
        fontsize=config.get('axislabel_fontsize', 14),
        fontfamily=resolve_mpl_font_family(
            config.get('axislabel_fontfamily', 'DejaVu Sans')
        ),
        color=config.get('axislabel_color', 'black'),
    )
    ax.set_xlabel(xlabel if xlabel is not None else unit_string, **axislabel_kwargs)
    ax.set_ylabel(ylabel if ylabel is not None else bunit, **axislabel_kwargs)

    ax.tick_params(
        axis='both', which='major',
        direction=config.get('tick_direction', 'out'),
        length=config.get('tick_length', 4),
        width=config.get('tick_width', 1),
        color=config.get('tick_color', 'black'),
        labelsize=config.get('tick_labelsize', 10),
        labelcolor=config.get('tick_labelcolor', 'black'),
        labelfontfamily=resolve_mpl_font_family(
            config.get('tick_font', 'DejaVu Sans')
        ),
    )
    ax.minorticks_on()
    ax.tick_params(
        axis='both', which='minor',
        direction=config.get('tick_direction', 'out'),
        length=config.get('mtick_length', 2),
        color=config.get('tick_color', 'black'),
    )
    for spine in ax.spines.values():
        spine.set_linewidth(config.get('tick_width', 1))
        spine.set_color(config.get('tick_color', 'black'))

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if title:
        ax.set_title(title, **axislabel_kwargs)

    fig.tight_layout()
    try:
        export_figure(fig, output_path, dpi=dpi)
    finally:
        plt.close(fig)

    return output_path
