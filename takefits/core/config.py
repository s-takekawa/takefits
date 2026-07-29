import copy
import math
import yaml
import os

from takefits.app_paths import app_config_path


AXES_POSITION_KEYS = ("ax_pos_l", "ax_pos_r", "ax_pos_t", "ax_pos_b")


def axes_positions_are_valid(config):
    """Return whether the configured axes rectangle has positive dimensions."""
    try:
        raw_values = [config[key] for key in AXES_POSITION_KEYS]
        if any(isinstance(value, bool) for value in raw_values):
            return False
        left, right, top, bottom = (float(value) for value in raw_values)
    except (KeyError, TypeError, ValueError):
        return False
    if not all(math.isfinite(value) for value in (left, right, top, bottom)):
        return False
    return left < right and bottom < top


def _normalize_grid_linestyle(value, default):
    token = str(value or "").strip().lower()
    mapping = {
        "-": "solid",
        "--": "dashed",
        ":": "dotted",
        "-.": "dashdot",
        "solid": "solid",
        "dashed": "dashed",
        "dotted": "dotted",
        "dash-dot": "dashdot",
        "dashdot": "dashdot",
        "dash dot": "dashdot",
    }
    return mapping.get(token, default)


def _clamped_float(value, default, lower, upper):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(default)
    if not math.isfinite(parsed):
        parsed = float(default)
    return min(float(upper), max(float(lower), parsed))


def _optional_nonnegative_float(value):
    if value is None or str(value).strip().lower() in {"", "auto", "none"}:
        return None
    try:
        parsed = abs(float(value))
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _config_bool(value, default):
    if isinstance(value, bool):
        return value
    token = str(value or "").strip().lower()
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def build_default_config():
    """Return a fresh copy of the built-in config defaults.

    Kept module-level so callers can inspect the known key set without
    reading or creating the user's config file.
    """
    return {
            'colorscale': 'Rainbow',  # Default color pattern setting
        
            # Background color settings
            'fig_background_color': '#ececec',  # Background color of the figure
            'ax_background_color': 'white',     # Background color of the axes
            'bad_color': 'black',               # NaN pixel color
        
            # Figure position and size settings
            'figure_pos_x': 100,  # X position of the figure
            'figure_pos_y': 100,  # Y position of the figure
            'figure_width': 640,  # Width of the figure
            'figure_height': 640, # Height of the figure
            'startup_show_subwindow1': True,   # Show XZ subwindow at startup (3D)
            'startup_show_subwindow2': False,  # Show ZY subwindow at startup (3D)
            
            'ax_pos_l': 0.15,
            'ax_pos_r': 0.85,
            'ax_pos_t': 0.9,
            'ax_pos_b': 0.12,
            
            
            # Colorbar settings
            'colorbar_orientation': 'vertical',
            'colorbar_auto_layout': True,
            'colorbar_placement': 'right',
            'colorbar_align': 'center',
            'colorbar_gap_px': 24.0,
            'colorbar_gap_x_px': 24.0,
            'colorbar_gap_y_px': 24.0,
            'colorbar_thickness_px': 24.0,
            'colorbar_length_mode': 'ratio',
            'colorbar_length_value': 1.0,
            'cbar_pos_x': 0.9,
            'cbar_pos_y': 0.11,
            'cbar_width': 0.04,
            'cbar_height': 0.77,
            
            'colorbar_tick_color': 'black',
            'colorbar_tick_length': 4,
            'colorbar_mtick_length': 2,
            'colorbar_tick_width': 1,
            'colorbar_tick_direction': 'out',
            'colorbar_mtick_freq': 2,
            
            'colorbar_tick_left': False,
            'colorbar_tick_right': True,
            'colorbar_tick_top': False,
            'colorbar_tick_bottom': True,

            'colorbar_tick_labelcolor': 'black',
            'colorbar_tick_labelsize': None, # None inherits the Matplotlib default
            'colorbar_tick_labelfontfamily': None, # None follows tick_font
            'colorbar_tick_labelleft': False,
            'colorbar_tick_labeltop': False,

            'colorbar_label': None,
            'colorbar_label_fontsize': 12,
            'colorbar_label_color': 'black',
            'colorbar_label_fontfamily': 'DejaVu Sans',
        
            # Axis label settings
            'axislabel_fontsize': 14,        # Font size of the axis labels
            'axislabel_fontfamily': 'DejaVu Sans', # Font family of the axis labels
            'axislabel_color': 'black',      # Color of the axis labels
        
            # Tick position and appearance
            'default_ticks_position': 'btlr', # Default tick positions (bottom, top, left, right)
        
            # Tick label position settings
            'xticklabel_position': 'b', # X-axis tick label position (bottom)
            'yticklabel_position': 'l', # Y-axis tick label position (left)
        
            # Tick appearance settings
            'tick_direction': 'out', # Tick direction ('in', 'out')
            'tick_length': 4,        # Length of the major ticks
            'tick_width': 1,         # Width of the major ticks
            'mtick_length': 2,       # Length of the minor ticks
            
            'x_mtick_freq': 5,       #minor tick frequency
            'y_mtick_freq': 5,
            'z_mtick_freq': 5,

            # Major tick placement. None lets astropy WCSAxes choose.
            # `spacing` is expressed in the axis' displayed format unit
            # (deg for decimal celestial axes, hourangle for sexagesimal RA,
            # km/s or m/s for a spectral axis). `spacing` wins over `number`.
            'x_tick_spacing': None,
            'y_tick_spacing': None,
            'z_tick_spacing': None,
            'x_tick_number': None,
            'y_tick_number': None,
            'z_tick_number': None,
            
            # Tick label settings
            'tick_labelsize': 10,     # Font size of the tick labels
            'tick_color': 'black',    # Color of the ticks
            'tick_labelcolor': 'black', # Color of the tick labels
            'tick_pad_x': 5,         # Distance between ticks and tick labels
            'tick_pad_y': 5,         # Distance between ticks and tick labels
            'tick_xlabelrotation': 0, # Rotation angle of X-axis tick labels
            'tick_ylabelrotation': 0, # Rotation angle of Y-axis tick labels
            'tick_font': 'DejaVu Sans',
            'tick_font_weight': 'normal',

            # Coordinate grid overlay (TF-404)
            'grid_visible': False,       # Show the WCS coordinate grid by default
            'grid_color': 'white',       # Grid line color
            'grid_alpha': 0.5,           # Grid line opacity (0-1)
            'grid_linestyle': 'solid',   # Grid line style ('solid', 'dashed', ...)
            'grid_linewidth': 0.5,       # Grid line width
            'grid_keep_native': True,    # Keep the native grid under a frame overlay
            'grid_overlay_color': '#00ff66', # Non-native XY overlay grid color
            'grid_overlay_linestyle': 'dashed', # Distinguish overlay without color alone
            'grid_overlay_label_color': 'auto', # auto, same, or a Matplotlib color
            'grid_overlay_show_lines': True,
            'grid_overlay_show_ticklabels': True,
            'grid_overlay_axislabel_placement': 'inside', # outside, inside, hidden
            'grid_overlay_longitude_axislabel_pad': None, # None chooses a safe automatic gap
            'grid_overlay_latitude_axislabel_pad': None,
            'grid_overlay_right_margin_px': 96.0, # Internal space for right-side decorations
            'grid_overlay_top_margin_px': 64.0, # Internal space for top-side decorations

            # Coordinate and decimal settings
            'decimal': True,            # Coordinate format (True: decimal degrees, False: sexagesimal)
            'auto_precision_digits': True,  # Auto precision based on pixel scale (1/10 pixel)
            'number_decimals': 6,       # Number of decimals to display
            'coord_wrap': 180,        # Coordinate wrap
        
            # Scroll speed setting
            'scrollspeed': 0.1,         # Speed of scrolling
            'invert_wheel_direction': False,  # Reverse mouse wheel channel direction

            # Large Data Mode thresholds (MiB)
            # "auto" uses 25% of physical RAM, clamped to 2–8 GiB.
            'large_data_mode_threshold_mb': 'auto',
            'large_data_no_memmap_threshold_mb': 2048,
            'large_data_threshold_policy': 2,
        
            # Coordinates label position settings
            'poslabel_x': 0.99,  # X position of the coordinate label
            'poslabel_y': 0.99,  # Y position of the coordinate label
            'poslabel_w': 250,  # Width of the coordinate label
            'poslabel_h': 30,   # Height of the coordinate label

            'pos_chlabel_x': 0.98,  # X position of the coordinate label
            'pos_chlabel_y': 0.02,  # Y position of the coordinate label
            #'pos_chlabel_w': 250,  # Width of the coordinate label
            #'pos_chlabel_h': 20,   # Height of the coordinate label

            'ch_label_color': 'grey',
            'ch_label_font': 'DejaVu Sans',
            'ch_label_size': 10,

            # Channel-map grid layout for headless export. None keeps the
            # historical automatic value.
            'chmap_left': 0.08,
            'chmap_right': None,   # None derives room for the colorbar
            'chmap_bottom': 0.08,
            'chmap_top': 0.95,
            'chmap_wspace': 0.12,
            'chmap_hspace': 0.12,
            # One shared axis label per figure instead of one per edge tile,
            # which is the usual publication style for a channel-map grid.
            'chmap_shared_axislabels': False,
            # Tile index carrying the HPBW beam. None follows the GUI, which
            # draws it on the bottom-left panel.
            'chmap_beam_tile': None,
            
            # Click-related settings
            'click_label_color': 'grey',   # Color of the label when clicking
            'click_linewidth': 0.5,        # Line width for the click indicator
            'click_linecolor': 'cyan',      # Line color for the click indicator
            'click_linestyle': '-',        # Line style for click crosshair
            'click_alpha': 1.0,            # Alpha for click crosshair
            'click_show_crosshair': True,  # Show click crosshair lines
            'click_crosshair_mode': 'both',  # Crosshair mode: both/vertical/horizontal
            'click_show_center_marker': False,  # Show center point marker
            
            # HPBW
            'beam_facecolor': 'white',
            'beam_edgecolor': 'None',
            'beam_linewidth': 0,
            'beam_pos_x': 0.1,
            'beam_pos_y': 0.1,

            # Range file
            'range_file': 'takefits.range'  # Range file path
    }


DEFAULT_CONFIG_KEYS = frozenset(build_default_config())


class ConfigManager:
    def __init__(self, config_file=None):
        if config_file is None:
            config_file = app_config_path('config.yaml')
        self.config_file = config_file
        self.default_config = build_default_config()
        self.config = self.load_config()
        self.config_bu = copy.deepcopy(self.config)

    def load_config(self):
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            config = copy.deepcopy(self.default_config)
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            print(f"\033[91mFailed to load config file: {e}\033[0m")
            config = copy.deepcopy(self.default_config)
        if isinstance(config, dict):
            # Version 1 shipped a fixed 8192 MiB value as if it were a user
            # preference, overriding the RAM-aware default on every machine.
            # Migrate that exact legacy default once; explicit post-migration
            # overrides remain untouched.
            if (
                'large_data_threshold_policy' not in config
                and config.get('large_data_mode_threshold_mb') == 8192
            ):
                config['large_data_mode_threshold_mb'] = 'auto'
            config['large_data_threshold_policy'] = 2
            # Migrate the short-lived inward-only title-offset experiment to
            # semantic placement before defaults are merged. Existing custom
            # files retain their visual intent under the semantic Inside
            # placement used by current defaults.
            legacy_lon_pad = config.pop(
                'grid_overlay_longitude_label_minpad',
                None,
            )
            legacy_lat_pad = config.pop(
                'grid_overlay_latitude_label_minpad',
                None,
            )
            if (
                'grid_overlay_axislabel_placement' not in config
                and (legacy_lon_pad is not None or legacy_lat_pad is not None)
            ):
                config['grid_overlay_axislabel_placement'] = 'inside'
                config['grid_overlay_longitude_axislabel_pad'] = (
                    _optional_nonnegative_float(legacy_lon_pad)
                )
                config['grid_overlay_latitude_axislabel_pad'] = (
                    _optional_nonnegative_float(legacy_lat_pad)
                )
            if 'range_file' not in config and 'region_file' in config:
                config['range_file'] = config.pop('region_file')
            merged_config = copy.deepcopy(self.default_config)
            merged_config.update(config)
            config = merged_config
            if not axes_positions_are_valid(config):
                invalid_values = {
                    key: config.get(key) for key in AXES_POSITION_KEYS
                }
                for key in AXES_POSITION_KEYS:
                    config[key] = self.default_config[key]
                print(
                    "\033[93mInvalid axes positions in config file "
                    f"'{self.config_file}': {invalid_values}. "
                    "Using defaults for the axes rectangle.\033[0m"
                )
            # Backward compatibility: legacy "match" behaves as full-length ratio.
            mode = str(config.get('colorbar_length_mode', '') or '').strip().lower()
            if mode == 'match':
                config['colorbar_length_mode'] = 'ratio'
                config['colorbar_length_value'] = 1.0
            config['grid_linestyle'] = _normalize_grid_linestyle(
                config.get('grid_linestyle'),
                'solid',
            )
            config['grid_overlay_linestyle'] = _normalize_grid_linestyle(
                config.get('grid_overlay_linestyle'),
                'dashed',
            )
            config['grid_alpha'] = _clamped_float(
                config.get('grid_alpha'),
                0.5,
                0.0,
                1.0,
            )
            config['grid_linewidth'] = _clamped_float(
                config.get('grid_linewidth'),
                0.5,
                0.0,
                20.0,
            )
            placement = str(
                config.get('grid_overlay_axislabel_placement', 'inside')
                or 'inside'
            ).strip().lower()
            if placement not in {'outside', 'inside', 'hidden'}:
                placement = 'inside'
            config['grid_overlay_axislabel_placement'] = placement
            config['grid_overlay_longitude_axislabel_pad'] = (
                _optional_nonnegative_float(
                    config.get('grid_overlay_longitude_axislabel_pad')
                )
            )
            config['grid_overlay_latitude_axislabel_pad'] = (
                _optional_nonnegative_float(
                    config.get('grid_overlay_latitude_axislabel_pad')
                )
            )
            label_color = str(
                config.get('grid_overlay_label_color', 'auto') or 'auto'
            ).strip()
            if label_color.lower() in {'auto', 'same'}:
                label_color = label_color.lower()
            config['grid_overlay_label_color'] = label_color
            config['grid_overlay_show_lines'] = _config_bool(
                config.get('grid_overlay_show_lines'),
                True,
            )
            config['grid_overlay_show_ticklabels'] = _config_bool(
                config.get('grid_overlay_show_ticklabels'),
                True,
            )
        else:
            config = copy.deepcopy(self.default_config)
        return config
