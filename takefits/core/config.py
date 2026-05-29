import copy
import yaml
import os

from takefits.app_paths import app_config_path

class ConfigManager:
    def __init__(self, config_file=None):
        if config_file is None:
            config_file = app_config_path('config.yaml')
        self.config_file = config_file
        self.default_config = {
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
            'colorbar_tick_labelleft': False, 
            'colorbar_tick_labeltop': False,
            
            'colorbar_label': None,
            'colorbar_label_fontsize': 12,
            'colorbar_label_color': 'black',
        
            # Axis label settings
            'axislabel_fontsize': 14,        # Font size of the axis labels
            'axislabel_fontfamily': 'Arial', # Font family of the axis labels
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
            
            # Tick label settings
            'tick_labelsize': 10,     # Font size of the tick labels
            'tick_color': 'black',    # Color of the ticks
            'tick_labelcolor': 'black', # Color of the tick labels
            'tick_pad_x': 5,         # Distance between ticks and tick labels
            'tick_pad_y': 5,         # Distance between ticks and tick labels
            'tick_xlabelrotation': 0, # Rotation angle of X-axis tick labels
            'tick_ylabelrotation': 0, # Rotation angle of Y-axis tick labels
            'tick_font': 'Arial',
            'tick_font_weight': 'normal',
        
            # Coordinate and decimal settings
            'decimal': True,            # Coordinate format (True: decimal degrees, False: sexagesimal)
            'auto_precision_digits': True,  # Auto precision based on pixel scale (1/10 pixel)
            'number_decimals': 6,       # Number of decimals to display
            'coord_wrap': 180,        # Coordinate wrap
        
            # Scroll speed setting
            'scrollspeed': 0.1,         # Speed of scrolling
            'invert_wheel_direction': False,  # Reverse mouse wheel channel direction

            # Large Data Mode thresholds (MiB)
            'large_data_mode_threshold_mb': 8192,
            'large_data_no_memmap_threshold_mb': 2048,
        
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
            'ch_label_font': 'Arial',
            'ch_label_size': 10,
            
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
            if 'range_file' not in config and 'region_file' in config:
                config['range_file'] = config.pop('region_file')
            merged_config = copy.deepcopy(self.default_config)
            merged_config.update(config)
            config = merged_config
            # Backward compatibility: legacy "match" behaves as full-length ratio.
            mode = str(config.get('colorbar_length_mode', '') or '').strip().lower()
            if mode == 'match':
                config['colorbar_length_mode'] = 'ratio'
                config['colorbar_length_value'] = 1.0
        else:
            config = copy.deepcopy(self.default_config)
        return config
