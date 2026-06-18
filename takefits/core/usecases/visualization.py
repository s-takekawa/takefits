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
from astropy.wcs import WCS
import astropy.units as u


def _get_pyplot():
    import matplotlib.pyplot as plt

    return plt


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
    cmap: str = "viridis",
    origin: str = "lower",
    dpi: int = 150,
    title: Optional[str] = None
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
        cmap: Colormap name.
        origin: Image origin ('lower' or 'upper').
        dpi: Output resolution.
        title: Optional title for the plot.
        
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
    # Load default config
    config_manager = ConfigManager()
    config = config_manager.config.copy()
    
    # Override with user args
    if cmap:
        config['colorscale'] = cmap
    
    # Only force white background if user hasn't customized it? 
    # Actually, for publication plots (headless), user likely *wants* config settings OR specific overrides.
    # We will let the config file dictate styles unless explicitly overridden by CLI args.
    
    # Note: display_map.py uses config keys directly.
    # The 'config' dict now contains everything from config.yaml.
    
    # 4. Render
    # Use config dimension? Or keep fixed size for CLI?
    # GUI config has 'figure_width', 'figure_height', but those are for UI window size.
    # Export usually needs standard size or specific DPI.
    # We stick to fixed figsize for now, but respect Config for fonts/colors/ticks.
    
    plt = _get_pyplot()
    fig = plt.figure(figsize=(8, 6))
    
    # Create DisplayMap instance
    # DisplayMap handles WCSAxes, colorbars, ticks, etc.
    from takefits.core.plotting.display_map import DisplayMap

    dm = DisplayMap(moment_data, header_2d, wcs_2d, config)
    
    # We display on 'xy' plane because we have reduced it to 2D
    dm.display(fig, plane='xy')
    
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

    Returns:
        Path to saved file
    """
    register_custom_colormaps()
    
    if cmap is None:
        cmap = 'Rainbow'
    if cmap is None:
        cmap = 'Rainbow'

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
    # Load config for channel maps (colors, fonts)
    config_manager = ConfigManager()
    config = config_manager.config
    effective_cmap = cmap if cmap else config.get('colorscale', 'viridis')
    nrows = int(np.ceil(n_images / ncols))

    plt = _get_pyplot()
    fig = plt.figure(figsize=(ncols * 3, nrows * 3))
    gs = fig.add_gridspec(nrows, ncols)

    
    # Global normalization
    all_data = np.array([img for img in images if img is not None])
    
    if vmin is None:
        vmin = np.nanmin(all_data)
    if vmax is None:
        vmax = np.nanmax(all_data)

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
        
        cbar.outline.set_color(config.get('colorbar_tick_color', 'black'))
        cbar.outline.set_linewidth(config.get('colorbar_tick_width', 1))

    if im_list:
        right_margin = cbar_pos_x - 0.02 if cbar_pos_x < 1.0 else 0.88
    else:
        right_margin = 0.95
    right_margin = float(np.clip(right_margin, 0.55, 0.98))
    fig.subplots_adjust(left=0.08, right=right_margin, bottom=0.08, top=0.95, wspace=0.12, hspace=0.12)

    # 3. Save
    try:
        export_figure(fig, output_path, dpi=dpi)
    finally:
        plt.close(fig)
        
    return output_path
