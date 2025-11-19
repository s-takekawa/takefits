from PyQt6.QtWidgets import QWidget, QGridLayout, QLineEdit, QPushButton, QLabel, QMessageBox, QTabWidget, QHBoxLayout, QVBoxLayout, QScrollArea, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QCompleter, QRadioButton, QButtonGroup
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFontDatabase
import matplotlib as mpl
from matplotlib import colormaps
import yaml
import os
import math



class CircularSpinBox(QSpinBox):
    def __init__(self):
        super().__init__()
        self.setRange(-180, 180)
        self.setSingleStep(1)
        self.setValue(0)
        self.setWrapping(True)
        self.setKeyboardTracking(False)
        self.valueChanged.connect(self._on_value_changed)
        self._last_value = 0

    def _on_value_changed(self, value):
        if abs(value - self._last_value) > 180:
            if value > self._last_value:
                self.setValue(value - 360)
            else:
                self.setValue(value + 360)
        self._last_value = self.value()

class ConfigPanel(QWidget):
    def __init__(self, config_manager, fits_viewer):
        super().__init__()
        self.config_manager = config_manager
        self.fits_viewer = fits_viewer
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()


        self.tabs = QTabWidget()

        # Add tabs for different configuration categories
        self.tabs.addTab(self.create_general_tab(), "General")
        self.tabs.addTab(self.create_display_tab(), "Display")
        self.tabs.addTab(self.create_ticks_tab(), "Ticks")
        self.tabs.addTab(self.create_click_tab(), "Click")
        self.tabs.addTab(self.create_colorbar_tab(), "Colorbar")

        layout.addWidget(self.tabs)


        button_layout = QHBoxLayout()
        

        self.apply_button = QPushButton('Apply', self)
        self.apply_button.clicked.connect(self.apply_changes)
        self.apply_button.setFixedWidth(50) 
        self.apply_button.setAutoDefault(True)
        self.apply_button.setDefault(True)
        button_layout.addWidget(self.apply_button)

        self.reset_button = QPushButton('Reset', self)
        self.reset_button.clicked.connect(self.reset_to_loaded_config)
        self.reset_button.setFixedWidth(50) 
        button_layout.addWidget(self.reset_button)
        
        
        button_layout.addSpacing(50)  
        self.default_button = QPushButton('Default', self)
        self.default_button.clicked.connect(self.reset_to_default)
        self.default_button.setFixedWidth(80) 
        button_layout.addWidget(self.default_button)

        self.save_button = QPushButton('Save', self)
        self.save_button.clicked.connect(self.save_config)
        self.save_button.setFixedWidth(80)
        button_layout.addWidget(self.save_button)
        

        layout.addLayout(button_layout)

        self.setLayout(layout)
        self.setWindowTitle('Configuration Settings')

    def create_general_tab(self):
        """Create the general settings tab."""
        general_layout = QGridLayout()
    
        # Colorscale setting with a combo box
        self.colorscale_input = QComboBox()
        self.colorscale_input.setEditable(True) 
        colormap_names = [name for name in colormaps.keys() if not name.endswith('_r')]
        self.colorscale_input.addItems(sorted(colormap_names))
        current_colorscale = self.config_manager.config.get('colorscale', 'Rainbow')
        self.colorscale_input.setFixedWidth(100)
        self.colorscale_input.setCurrentText(current_colorscale)
        general_layout.addWidget(QLabel('Default Colorscale:'), 0, 0)
        general_layout.addWidget(self.colorscale_input, 0, 1)
    
        # Coordinate format selection
        coord_format_layout = QHBoxLayout()
        coord_format_layout.setContentsMargins(0, 0, 0, 0)
        coord_format_layout.setSpacing(12)
        coord_format_label = QLabel('Coordinate Format:')
        coord_format_layout.addWidget(coord_format_label)
        self.coord_format_group = QButtonGroup(self)
        self.decimal_radio = QRadioButton('Decimal')
        self.sexagesimal_radio = QRadioButton('Sexagesimal')
        self.coord_format_group.addButton(self.decimal_radio)
        self.coord_format_group.addButton(self.sexagesimal_radio)
        coord_format_layout.addWidget(self.decimal_radio)
        coord_format_layout.addWidget(self.sexagesimal_radio)
        coord_format_layout.addStretch()
        self.set_coordinate_format(self.config_manager.config.get('decimal', True))
        general_layout.addLayout(coord_format_layout, 1, 0, 1, 2)
        
        # Add the number of decimals input
        digits_label = QLabel('Precision digits:')
        digits_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.number_decimals_input = QSpinBox()
        self.number_decimals_input.setValue(self.config_manager.config.get('number_decimals', 6))
        self.number_decimals_input.setFixedWidth(70)
        general_layout.addWidget(digits_label, 2, 0)
        general_layout.addWidget(self.number_decimals_input, 2, 1)
        
        self.wrap_angle_input = QSpinBox()
        self.wrap_angle_input.setSingleStep(180) 
        self.wrap_angle_input.setRange(180, 360) 
        self.wrap_angle_input.setValue(self.config_manager.config.get('coord_wrap', 180))
        self.wrap_angle_input.setFixedWidth(100) 
        wrap_angle_label = QLabel('Wrap Angle:')
        wrap_angle_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        general_layout.addWidget(wrap_angle_label, 3, 0)
        general_layout.addWidget(self.wrap_angle_input, 3, 1)
        
    
        # Scroll speed setting
        self.scrollspeed_input = QDoubleSpinBox()
        self.scrollspeed_input.setValue(self.config_manager.config.get('scrollspeed', 0.1))
        self.scrollspeed_input.setSingleStep(0.1) 
        self.scrollspeed_input.setFixedWidth(70)
        self.scrollspeed_input.setAlignment(Qt.AlignmentFlag.AlignRight)
        scrollspeed_label = QLabel('Scroll Speed:')
        scrollspeed_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        general_layout.addWidget(scrollspeed_label, 4, 0)
        general_layout.addWidget(self.scrollspeed_input, 4, 1)
    
        # Range file input
        self.range_file_input = QLineEdit(self.config_manager.config.get('range_file', 'takefits.range'))
        range_file_label = QLabel('Range File:')
        range_file_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        general_layout.addWidget(range_file_label, 5, 0)
        general_layout.addWidget(self.range_file_input, 5, 1)
    
        general_widget = QWidget()
        general_widget.setLayout(general_layout)
        return general_widget

    def create_display_tab(self):
        """Create the display settings tab."""
        display_layout = QGridLayout()
    

        self.fig_background_color_input = self.create_color_combobox(self.config_manager.config.get('fig_background_color', '#ececec'))
        self.fig_background_color_input.setFixedWidth(100)
        display_layout.addWidget(QLabel('Figure Background Color:'), 0, 0)
        display_layout.addWidget(self.fig_background_color_input, 0, 1)
    

        self.ax_background_color_input = self.create_color_combobox(self.config_manager.config.get('ax_background_color', 'white'))
        self.ax_background_color_input.setFixedWidth(100)
        display_layout.addWidget(QLabel('Axes Background Color:'), 1, 0)
        display_layout.addWidget(self.ax_background_color_input, 1, 1)
    

        self.bad_color_input = self.create_color_combobox(self.config_manager.config.get('bad_color', 'black'))
        self.bad_color_input.setFixedWidth(100)
        display_layout.addWidget(QLabel('Bad (NaN) Color:'), 2, 0)
        display_layout.addWidget(self.bad_color_input, 2, 1)
    

        self.figure_pos_x_input = QSpinBox()
        self.figure_pos_x_input.setRange(0, 9999)
        self.figure_pos_x_input.setValue(self.config_manager.config.get('figure_pos_x', 100))
        self.figure_pos_x_input.setFixedWidth(100)
        display_layout.addWidget(QLabel('Figure X Position:'), 3, 0)
        display_layout.addWidget(self.figure_pos_x_input, 3, 1)
        
        self.figure_pos_y_input = QSpinBox()
        self.figure_pos_y_input.setRange(0, 9999)
        self.figure_pos_y_input.setValue(self.config_manager.config.get('figure_pos_y', 100))
        self.figure_pos_y_input.setFixedWidth(100)
        display_layout.addWidget(QLabel('Figure Y Position:'), 4, 0)
        display_layout.addWidget(self.figure_pos_y_input, 4, 1)
        
        self.figure_width_input = QSpinBox()
        self.figure_width_input.setRange(640, 9999)
        self.figure_width_input.setValue(self.config_manager.config.get('figure_width', 640))
        self.figure_width_input.setFixedWidth(100)
        display_layout.addWidget(QLabel('Figure Width:'), 5, 0)
        display_layout.addWidget(self.figure_width_input, 5, 1)
        
        self.figure_height_input = QSpinBox()
        self.figure_height_input.setRange(115, 9999)
        self.figure_height_input.setValue(self.config_manager.config.get('figure_height', 640))
        self.figure_height_input.setFixedWidth(100)
        display_layout.addWidget(QLabel('Figure Height:'), 6, 0)
        display_layout.addWidget(self.figure_height_input, 6, 1)
        
        self.axislabel_fontsize_input = QSpinBox()
        self.axislabel_fontsize_input.setValue(self.config_manager.config.get('axislabel_fontsize', 14))
        self.axislabel_fontsize_input.setRange(1, 100) 
        self.axislabel_fontsize_input.setFixedWidth(100)
        display_layout.addWidget(QLabel('Axis Label Font Size:'), 7, 0)
        display_layout.addWidget(self.axislabel_fontsize_input, 7, 1)
        
        self.axislabel_fontfamily_input = self.create_font_combobox(self.config_manager.config.get('axislabel_fontfamily', 'Arial'))
        self.axislabel_fontfamily_input.setFixedWidth(100)
        display_layout.addWidget(QLabel('Axis Label Font Family:'), 8, 0)
        display_layout.addWidget(self.axislabel_fontfamily_input, 8, 1)
        
        self.axislabel_color_input = self.create_color_combobox(self.config_manager.config.get('axislabel_color', 'black'))
        self.axislabel_color_input.setFixedWidth(100)
        display_layout.addWidget(QLabel('Axis Label Color:'), 9, 0)
        display_layout.addWidget(self.axislabel_color_input, 9, 1)
        
        # Axis Left Position
        self.axis_left_spinbox = QDoubleSpinBox()
        self.axis_left_spinbox.setRange(0.0, 2.0)
        self.axis_left_spinbox.setSingleStep(0.01)
        self.axis_left_spinbox.setValue(self.config_manager.config.get('ax_pos_l', 0.15))
        display_layout.addWidget(QLabel("Axis Left Position"), 10, 0)
        display_layout.addWidget(self.axis_left_spinbox, 10, 1)
        
        # Axis Right Position
        self.axis_right_spinbox = QDoubleSpinBox()
        self.axis_right_spinbox.setRange(0.0, 2.0)
        self.axis_right_spinbox.setSingleStep(0.01)
        self.axis_right_spinbox.setValue(self.config_manager.config.get('ax_pos_r', 0.85))
        display_layout.addWidget(QLabel("Axis Right Position"), 11, 0)
        display_layout.addWidget(self.axis_right_spinbox, 11, 1)
        
        # Axis Top Position
        self.axis_top_spinbox = QDoubleSpinBox()
        self.axis_top_spinbox.setRange(0.0, 2.0)
        self.axis_top_spinbox.setSingleStep(0.01)
        self.axis_top_spinbox.setValue(self.config_manager.config.get('ax_pos_t', 0.9))
        display_layout.addWidget(QLabel("Axis Top Position"), 12, 0)
        display_layout.addWidget(self.axis_top_spinbox, 12, 1)
        
        # Axis Bottom Position
        self.axis_bottom_spinbox = QDoubleSpinBox()
        self.axis_bottom_spinbox.setRange(0.0, 2.0)
        self.axis_bottom_spinbox.setSingleStep(0.01)
        self.axis_bottom_spinbox.setValue(self.config_manager.config.get('ax_pos_b', 0.12))
        display_layout.addWidget(QLabel("Axis Bottom Position"), 13, 0)
        display_layout.addWidget(self.axis_bottom_spinbox, 13, 1)
        
        #Beam
        self.beam_facecolor_input = self.create_color_combobox(self.config_manager.config.get('beam_facecolor', 'white'))
        self.beam_facecolor_input.setFixedWidth(100)
        display_layout.addWidget(QLabel('Beam FaceColor:'), 14, 0)
        display_layout.addWidget(self.beam_facecolor_input, 14, 1)

        self.beam_edgecolor_input = self.create_color_combobox(self.config_manager.config.get('beam_edgecolor', 'None'))
        self.beam_edgecolor_input.setFixedWidth(100)
        display_layout.addWidget(QLabel('Beam EdgeColor:'), 15, 0)
        display_layout.addWidget(self.beam_edgecolor_input, 15, 1)

        self.beam_linewidth_input = QDoubleSpinBox()
        self.beam_linewidth_input.setValue(self.config_manager.config.get('beam_linewidth', 0.0))
        self.beam_linewidth_input.setSingleStep(0.5) 
        display_layout.addWidget(QLabel('Beam Line Width:'), 16, 0)
        display_layout.addWidget(self.beam_linewidth_input, 16, 1)

        self.beam_pos_x_spinbox = QDoubleSpinBox()
        self.beam_pos_x_spinbox.setRange(0.0, 1.0)
        self.beam_pos_x_spinbox.setSingleStep(0.01)
        self.beam_pos_x_spinbox.setValue(self.config_manager.config.get('beam_pos_x', 0.10))
        display_layout.addWidget(QLabel("Beam Position X"), 17, 0)
        display_layout.addWidget(self.beam_pos_x_spinbox, 17, 1)

        self.beam_pos_y_spinbox = QDoubleSpinBox()
        self.beam_pos_y_spinbox.setRange(0.0, 1.0)
        self.beam_pos_y_spinbox.setSingleStep(0.01)
        self.beam_pos_y_spinbox.setValue(self.config_manager.config.get('beam_pos_y', 0.10))
        display_layout.addWidget(QLabel("Beam Position Y"), 18, 0)
        display_layout.addWidget(self.beam_pos_y_spinbox, 18, 1)
        
        
    
        display_widget = QWidget()
        display_widget.setLayout(display_layout)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)  # Allows resizing of scroll area
        scroll_area.setWidget(display_widget)
        return scroll_area
        
        

    def create_ticks_tab(self):
        """Create the ticks settings tab."""
        ticks_layout = QGridLayout()
        
        
        self.tick_length_input = QSpinBox()
        self.tick_length_input.setValue(self.config_manager.config.get('tick_length', 4))
        self.tick_length_input.setFixedWidth(100) 
        ticks_layout.addWidget(QLabel('Tick Length:'), 1, 0)
        ticks_layout.addWidget(self.tick_length_input, 1, 1)
        
        self.mtick_length_input = QSpinBox()
        self.mtick_length_input.setValue(self.config_manager.config.get('mtick_length', 2))
        self.mtick_length_input.setFixedWidth(100)
        ticks_layout.addWidget(QLabel('Minor Tick Length:'), 2, 0)
        ticks_layout.addWidget(self.mtick_length_input, 2, 1)
        
        self.tick_width_input = QDoubleSpinBox()
        self.tick_width_input.setValue(self.config_manager.config.get('tick_width', 1))
        self.tick_width_input.setFixedWidth(100)
        self.tick_width_input.setSingleStep(0.5) 
        ticks_layout.addWidget(QLabel('Tick Line Width:'), 3, 0)
        ticks_layout.addWidget(self.tick_width_input, 3, 1)
        
        
        self.x_mtick_freq_input = QSpinBox()
        self.x_mtick_freq_input.setValue(self.config_manager.config.get('x_mtick_freq', 5))
        self.x_mtick_freq_input.setFixedWidth(100)
        self.x_mtick_freq_input.setSingleStep(1) 
        self.x_mtick_freq_input.setRange(1, 10)
        ticks_layout.addWidget(QLabel('X Minor Tick Freq.:'), 4, 0)
        ticks_layout.addWidget(self.x_mtick_freq_input, 4, 1)
        
        self.y_mtick_freq_input = QSpinBox()
        self.y_mtick_freq_input.setValue(self.config_manager.config.get('y_mtick_freq', 5))
        self.y_mtick_freq_input.setFixedWidth(100)
        self.y_mtick_freq_input.setSingleStep(1)
        self.y_mtick_freq_input.setRange(1, 10)
        ticks_layout.addWidget(QLabel('Y Minor Tick Freq.:'), 5, 0)
        ticks_layout.addWidget(self.y_mtick_freq_input, 5, 1)
        
        self.z_mtick_freq_input = QSpinBox()
        self.z_mtick_freq_input.setValue(self.config_manager.config.get('z_mtick_freq', 5))
        self.z_mtick_freq_input.setFixedWidth(100)
        self.z_mtick_freq_input.setSingleStep(1)
        self.z_mtick_freq_input.setRange(1, 10)
        ticks_layout.addWidget(QLabel('Z Minor Tick Freq.:'), 6, 0)
        ticks_layout.addWidget(self.z_mtick_freq_input, 6, 1)
        
        
        self.tick_color_input = self.create_color_combobox(self.config_manager.config.get('tick_color', 'black'))
        self.tick_color_input.setFixedWidth(100)  
        ticks_layout.addWidget(QLabel('Tick Color:'), 7, 0)
        ticks_layout.addWidget(self.tick_color_input, 7, 1)


        self.tick_labelsize_input = QSpinBox()
        self.tick_labelsize_input.setValue(self.config_manager.config.get('tick_labelsize', 10))
        self.tick_labelsize_input.setFixedWidth(100) 
        ticks_layout.addWidget(QLabel('Tick Label Font Size:'), 8, 0)
        ticks_layout.addWidget(self.tick_labelsize_input, 8, 1)
        
        
        self.tick_labelcolor_input = self.create_color_combobox(self.config_manager.config.get('tick_labelcolor', 'black'))
        self.tick_labelcolor_input.setFixedWidth(100)  
        ticks_layout.addWidget(QLabel('Tick Label Color:'), 9, 0)
        ticks_layout.addWidget(self.tick_labelcolor_input, 9, 1)
        

        self.tick_direction_input = QComboBox()
        self.tick_direction_input.addItems(['in', 'out'])  
        self.tick_direction_input.setCurrentText(self.config_manager.config.get('tick_direction', 'out'))
        self.tick_direction_input.setFixedWidth(100)  
        ticks_layout.addWidget(QLabel('Tick Direction:'), 10, 0)
        ticks_layout.addWidget(self.tick_direction_input, 10, 1)
        
        
        # Default ticks position (bottom, top, left, right)
        self.default_ticks_position_input = QComboBox()
        self.default_ticks_position_input.addItems(['None', 'b', 't', 'l', 'r', 'bt', 'bl', 'br', 'tl', 'tr', 'lr', 'btlr'])
        self.default_ticks_position_input.setCurrentText(self.config_manager.config.get('default_ticks_position', 'btlr'))
        self.default_ticks_position_input.setFixedWidth(100)
        ticks_layout.addWidget(QLabel('Tick Position:'), 11, 0)
        ticks_layout.addWidget(self.default_ticks_position_input, 11, 1)
    
        # X-axis tick label position (bottom)
        self.xticklabel_position_input = QComboBox()
        self.xticklabel_position_input.addItems(['t', 'b'])
        self.xticklabel_position_input.setCurrentText(self.config_manager.config.get('xticklabel_position', 'b'))
        self.xticklabel_position_input.setFixedWidth(100)
        ticks_layout.addWidget(QLabel('X Tick Label Position:'), 12, 0)
        ticks_layout.addWidget(self.xticklabel_position_input, 12, 1)
    
        # Y-axis tick label position (left)
        self.yticklabel_position_input = QComboBox()
        self.yticklabel_position_input.addItems(['l', 'r'])
        self.yticklabel_position_input.setCurrentText(self.config_manager.config.get('yticklabel_position', 'l'))
        self.yticklabel_position_input.setFixedWidth(100)
        ticks_layout.addWidget(QLabel('Y Tick Label Position:'), 13, 0)
        ticks_layout.addWidget(self.yticklabel_position_input, 13, 1)
        
        
        self.tick_xlabelrotation_input = CircularSpinBox()
        self.tick_xlabelrotation_input.setValue(self.config_manager.config.get('tick_xlabelrotation', 0))
        self.tick_xlabelrotation_input.setFixedWidth(100)
        #self.tick_xlabelrotation_input.setRange(0, 359)
        ticks_layout.addWidget(QLabel('X Tick Label Rotation:'), 14, 0)
        ticks_layout.addWidget(self.tick_xlabelrotation_input, 14, 1)
        
        self.tick_ylabelrotation_input = CircularSpinBox()
        self.tick_ylabelrotation_input.setValue(self.config_manager.config.get('tick_ylabelrotation', 0))
        self.tick_ylabelrotation_input.setFixedWidth(100)
        #self.tick_ylabelrotation_input.setRange(0, 359)
        ticks_layout.addWidget(QLabel('Y Tick Label Rotation:'), 15, 0)
        ticks_layout.addWidget(self.tick_ylabelrotation_input, 15, 1)
        
        self.tick_xpad_input = QSpinBox()
        self.tick_xpad_input.setValue(self.config_manager.config.get('tick_pad_x', 5))
        self.tick_xpad_input.setFixedWidth(100)
        ticks_layout.addWidget(QLabel('X Tick Label Space:'), 16, 0)
        ticks_layout.addWidget(self.tick_xpad_input, 16, 1)
        
        self.tick_ypad_input = QSpinBox()
        self.tick_ypad_input.setValue(self.config_manager.config.get('tick_pad_y', 5))
        self.tick_ypad_input.setFixedWidth(100)
        ticks_layout.addWidget(QLabel('Y Tick Label Space:'), 17, 0)
        ticks_layout.addWidget(self.tick_ypad_input, 17, 1)


        self.ticklabel_fontfamily_input = self.create_font_combobox(self.config_manager.config.get('tick_font', 'Arial'))
        self.ticklabel_fontfamily_input.setFixedWidth(100)
        ticks_layout.addWidget(QLabel('Tick Label Font Family: (updated next)'), 18, 0)
        ticks_layout.addWidget(self.ticklabel_fontfamily_input, 18, 1)
        


        ticks_widget = QWidget()
        ticks_widget.setLayout(ticks_layout)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(ticks_widget)
        return scroll_area

    def create_click_tab(self):
        """Create the click settings tab."""
        click_layout = QGridLayout()
    
        self.click_linecolor_input = self.create_color_combobox(self.config_manager.config.get('click_linecolor', 'cyan'))
        self.click_linecolor_input.setFixedWidth(100) 
        click_layout.addWidget(QLabel('Click Line Color:'), 0, 0)
        click_layout.addWidget(self.click_linecolor_input, 0, 1)


        self.click_linewidth_input = QDoubleSpinBox()
        self.click_linewidth_input.setValue(self.config_manager.config.get('click_linewidth', 0.25))
        self.click_linewidth_input.setFixedWidth(100)
        self.click_linewidth_input.setSingleStep(0.25) 
        click_layout.addWidget(QLabel('Click Line Width:'), 1, 0)
        click_layout.addWidget(self.click_linewidth_input, 1, 1)
        
        
        self.click_label_color_input = self.create_color_combobox(self.config_manager.config.get('click_label_color', 'grey'))
        self.click_label_color_input.setFixedWidth(100) 
        click_layout.addWidget(QLabel('Click Label Color:'), 2, 0)
        click_layout.addWidget(self.click_label_color_input, 2, 1)
        

        self.poslabel_x_input = QDoubleSpinBox()
        self.poslabel_x_input.setValue(self.config_manager.config.get('poslabel_x', 0.75))
        self.poslabel_x_input.setFixedWidth(100)
        self.poslabel_x_input.setRange(0.0, 1.0)
        self.poslabel_x_input.setSingleStep(0.01) 
        click_layout.addWidget(QLabel('Click Label X Position:'), 3, 0)
        click_layout.addWidget(self.poslabel_x_input, 3, 1)
        

        self.poslabel_y_input = QDoubleSpinBox()
        self.poslabel_y_input.setValue(self.config_manager.config.get('poslabel_y', 0.9))
        self.poslabel_y_input.setFixedWidth(100)
        self.poslabel_y_input.setSingleStep(0.01) 
        self.poslabel_y_input.setFixedWidth(100) 
        self.poslabel_y_input.setRange(0.0, 1.0)
        click_layout.addWidget(QLabel('Click Label Y Position:'), 4, 0)
        click_layout.addWidget(self.poslabel_y_input, 4, 1)
        

        self.poslabel_w_input = QSpinBox()
        self.poslabel_w_input.setFixedWidth(100) 
        self.poslabel_w_input.setRange(0,999)
        self.poslabel_w_input.setValue(self.config_manager.config.get('poslabel_w', 250))
        click_layout.addWidget(QLabel('Click Label Width (pixel value):'), 5, 0)
        click_layout.addWidget(self.poslabel_w_input, 5, 1)
        
        self.poslabel_h_input = QSpinBox()
        self.poslabel_h_input.setFixedWidth(100)
        self.poslabel_h_input.setRange(0,999)
        self.poslabel_h_input.setValue(self.config_manager.config.get('poslabel_h', 250))
        click_layout.addWidget(QLabel('Click Label Height (pixel value):'), 6, 0)
        click_layout.addWidget(self.poslabel_h_input, 6, 1)

        # ch_label_color
        self.ch_label_color_input = self.create_color_combobox(self.config_manager.config.get('ch_label_color', 'grey'))
        self.ch_label_color_input.setEditable(True)
        self.ch_label_color_input.setFixedWidth(100)
        click_layout.addWidget(QLabel('Ch. Label Color:'), 7, 0)
        click_layout.addWidget(self.ch_label_color_input, 7, 1)
        
        # ch_label_size
        self.ch_label_size_input = QSpinBox()
        self.ch_label_size_input.setFixedWidth(100)
        self.ch_label_size_input.setRange(1, 100)
        self.ch_label_size_input.setValue(self.config_manager.config.get('ch_label_size', 12))
        click_layout.addWidget(QLabel('Ch. Label Font Size:'), 8, 0)
        click_layout.addWidget(self.ch_label_size_input, 8, 1)
        
        # ch_label_font
        self.ch_label_font_input = self.create_font_combobox(self.config_manager.config.get('ch_label_font', 'Arial'))
        self.ch_label_font_input.setEditable(True)
        self.ch_label_font_input.setFixedWidth(100)
        click_layout.addWidget(QLabel('Ch. Label Font:'), 9, 0)
        click_layout.addWidget(self.ch_label_font_input, 9, 1)

        self.pos_chlabel_x_input = QDoubleSpinBox()
        self.pos_chlabel_x_input.setFixedWidth(100)
        self.pos_chlabel_x_input.setRange(0.0, 2.0) 
        self.pos_chlabel_x_input.setSingleStep(0.01)
        self.pos_chlabel_x_input.setValue(self.config_manager.config.get('pos_chlabel_x', 0.98))
        click_layout.addWidget(QLabel('Ch. Label X Position (relative):'), 10, 0)
        click_layout.addWidget(self.pos_chlabel_x_input, 10, 1)

        self.pos_chlabel_y_input = QDoubleSpinBox()
        self.pos_chlabel_y_input.setFixedWidth(100)
        self.pos_chlabel_y_input.setRange(0.0, 2.0)
        self.pos_chlabel_y_input.setSingleStep(0.01)
        self.pos_chlabel_y_input.setValue(self.config_manager.config.get('pos_chlabel_y', 0.02))
        click_layout.addWidget(QLabel('Ch. Label Y Position (relative):'), 11, 0)
        click_layout.addWidget(self.pos_chlabel_y_input, 11, 1)
        
        """
        self.pos_chlabel_w_input = QSpinBox()
        self.pos_chlabel_w_input.setFixedWidth(100)
        self.pos_chlabel_w_input.setRange(0, 999)
        self.pos_chlabel_w_input.setValue(self.config_manager.config.get('pos_chlabel_w', 250))
        click_layout.addWidget(QLabel('Ch. Label Width (pixel value):'), 12, 0)
        click_layout.addWidget(self.pos_chlabel_w_input, 12, 1)

        self.pos_chlabel_h_input = QSpinBox()
        self.pos_chlabel_h_input.setFixedWidth(100)
        self.pos_chlabel_h_input.setRange(0, 999)
        self.pos_chlabel_h_input.setValue(self.config_manager.config.get('pos_chlabel_h', 20))
        click_layout.addWidget(QLabel('Ch. Label Height (pixel value):'), 13, 0)
        click_layout.addWidget(self.pos_chlabel_h_input, 13, 1)
        """
    
        click_widget = QWidget()
        click_widget.setLayout(click_layout)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(click_widget)
        return scroll_area
        
        return click_widget

    def create_color_combobox(self, current_color):
        """Helper method to create a combo box for color selection."""
        color_combobox = QComboBox()
        colors = list(mpl.colors.CSS4_COLORS.keys())
        colors.append('#323232')
        colors.append('#ececec')
        colors.append('None')
        color_combobox.addItems(colors)
        color_combobox.setEditable(True)
        color_combobox.setCompleter(QCompleter(colors))
        if current_color in colors:
            color_combobox.setCurrentText(current_color)
        return color_combobox

    def create_font_combobox(self, current_font):
        """Helper method to create a combo box for font selection."""
        font_combobox = QComboBox()
        fonts = QFontDatabase.families()
        font_combobox.addItems(fonts)
        font_combobox.setEditable(True)
        font_combobox.setCompleter(QCompleter(fonts))
        if current_font in fonts:
            font_combobox.setCurrentText(current_font)
        return font_combobox
        
        
    def create_colorbar_tab(self):
        """Create the colorbar settings tab."""
        colorbar_layout = QGridLayout()
        
        # Colorbar Orientation
        self.cbar_orientation_label = QLabel("Colorbar Orientation:")
        self.cbar_orientation_combo = QComboBox()
        self.cbar_orientation_combo.addItems(["vertical", "horizontal"])
        self.cbar_orientation_combo.setCurrentText(self.config_manager.config.get('colorbar_orientation', 'vertical'))
        colorbar_layout.addWidget(self.cbar_orientation_label, 0, 0)
        colorbar_layout.addWidget(self.cbar_orientation_combo, 0, 1)
        
        # Colorbar X Position
        self.cbar_x_label = QLabel("Colorbar X Position:")
        self.cbar_x_input = QDoubleSpinBox()
        self.cbar_x_input.setRange(0.0, 1.0)
        self.cbar_x_input.setSingleStep(0.01)
        self.cbar_x_input.setFixedWidth(100)
        self.cbar_x_input.setValue(self.config_manager.config.get('cbar_pos_x', 0.9))
        colorbar_layout.addWidget(self.cbar_x_label, 1, 0)
        colorbar_layout.addWidget(self.cbar_x_input, 1, 1)
        
        # Colorbar Y Position
        self.cbar_y_label = QLabel("Colorbar Y Position:")
        self.cbar_y_input = QDoubleSpinBox()
        self.cbar_y_input.setRange(0.0, 1.0)
        self.cbar_y_input.setSingleStep(0.01)
        self.cbar_y_input.setFixedWidth(100)
        self.cbar_y_input.setValue(self.config_manager.config.get('cbar_pos_y', 0.11))
        colorbar_layout.addWidget(self.cbar_y_label, 2, 0)
        colorbar_layout.addWidget(self.cbar_y_input, 2, 1)
        
        # Colorbar Width
        self.cbar_width_label = QLabel("Colorbar Width:")
        self.cbar_width_input = QDoubleSpinBox()
        self.cbar_width_input.setRange(0.0, 1.0)
        self.cbar_width_input.setSingleStep(0.01)
        self.cbar_width_input.setFixedWidth(100)
        self.cbar_width_input.setValue(self.config_manager.config.get('cbar_width', 0.02))
        colorbar_layout.addWidget(self.cbar_width_label, 3, 0)
        colorbar_layout.addWidget(self.cbar_width_input, 3, 1)
        
        # Colorbar Height
        self.cbar_height_label = QLabel("Colorbar Height:")
        self.cbar_height_input = QDoubleSpinBox()
        self.cbar_height_input.setRange(0.0, 1.0)
        self.cbar_height_input.setSingleStep(0.01)
        self.cbar_height_input.setFixedWidth(100)
        self.cbar_height_input.setValue(self.config_manager.config.get('cbar_height', 0.77))
        colorbar_layout.addWidget(self.cbar_height_label, 4, 0)
        colorbar_layout.addWidget(self.cbar_height_input, 4, 1)
        
        
        # Colorbar Tick Direction
        self.cbar_tick_direction_label = QLabel("Tick Direction:")
        self.cbar_tick_direction_combo = QComboBox()
        self.cbar_tick_direction_combo.addItems(["in", "out", "inout"])
        self.cbar_tick_direction_combo.setCurrentText(self.config_manager.config.get('colorbar_tick_direction', 'out'))
        colorbar_layout.addWidget(self.cbar_tick_direction_label, 5, 0)
        colorbar_layout.addWidget(self.cbar_tick_direction_combo, 5, 1)
        

        # Colorbar Ticks - Left
        self.cbar_tick_left_checkbox = QCheckBox("Tick on Left")
        self.cbar_tick_left_checkbox.setChecked(self.config_manager.config.get('colorbar_tick_left', True))
        colorbar_layout.addWidget(self.cbar_tick_left_checkbox, 6, 0)
        
        # Colorbar Ticks - Right
        self.cbar_tick_right_checkbox = QCheckBox("Tick on Right")
        self.cbar_tick_right_checkbox.setChecked(self.config_manager.config.get('colorbar_tick_right', True))
        colorbar_layout.addWidget(self.cbar_tick_right_checkbox, 6, 1)
        
        # Colorbar Ticks - Top
        self.cbar_tick_top_checkbox = QCheckBox("Tick on Top")
        self.cbar_tick_top_checkbox.setChecked(self.config_manager.config.get('colorbar_tick_top', False))
        colorbar_layout.addWidget(self.cbar_tick_top_checkbox, 7, 0)
        
        # Colorbar Ticks - Bottom
        self.cbar_tick_bottom_checkbox = QCheckBox("Tick on Bottom")
        self.cbar_tick_bottom_checkbox.setChecked(self.config_manager.config.get('colorbar_tick_bottom', True))
        colorbar_layout.addWidget(self.cbar_tick_bottom_checkbox, 7, 1)
        
        # Colorbar Tick Length
        self.cbar_tick_length_label = QLabel("Colorbar Tick Length:")
        self.cbar_tick_length_input = QDoubleSpinBox()
        self.cbar_tick_length_input.setRange(0, 20)
        self.cbar_tick_length_input.setSingleStep(0.5)
        self.cbar_tick_length_input.setFixedWidth(100)
        self.cbar_tick_length_input.setValue(self.config_manager.config.get('colorbar_tick_length', 2))
        colorbar_layout.addWidget(self.cbar_tick_length_label, 8, 0)
        colorbar_layout.addWidget(self.cbar_tick_length_input, 8, 1)
        
        # Colorbar Tick Width
        self.cbar_tick_width_label = QLabel("Tick Width:")
        self.cbar_tick_width_input = QDoubleSpinBox()
        self.cbar_tick_width_input.setRange(0, 20)
        self.cbar_tick_width_input.setSingleStep(0.5)
        self.cbar_tick_width_input.setFixedWidth(100)
        self.cbar_tick_width_input.setValue(self.config_manager.config.get('colorbar_tick_width', 1))
        colorbar_layout.addWidget(self.cbar_tick_width_label, 9, 0)
        colorbar_layout.addWidget(self.cbar_tick_width_input, 9, 1)
        
        # Colorbar Minor Tick Frequency
        self.cbar_mtick_freq_label = QLabel("Minor Tick Frequency:")
        self.cbar_mtick_freq_input = QSpinBox()
        self.cbar_mtick_freq_input.setRange(1, 10)
        self.cbar_mtick_freq_input.setFixedWidth(100)
        self.cbar_mtick_freq_input.setValue(self.config_manager.config.get('colorbar_mtick_freq', 2))
        colorbar_layout.addWidget(self.cbar_mtick_freq_label, 10, 0)
        colorbar_layout.addWidget(self.cbar_mtick_freq_input, 10, 1)
        
        # Colorbar Minor Tick Length
        self.cbar_mtick_length_label = QLabel("Minor Tick Length:")
        self.cbar_mtick_length_input = QDoubleSpinBox()
        self.cbar_mtick_length_input.setRange(0, 20)
        self.cbar_mtick_length_input.setSingleStep(0.5)
        self.cbar_mtick_length_input.setFixedWidth(100)
        self.cbar_mtick_length_input.setValue(self.config_manager.config.get('colorbar_mtick_length', 1))
        colorbar_layout.addWidget(self.cbar_mtick_length_label, 11, 0)
        colorbar_layout.addWidget(self.cbar_mtick_length_input, 11, 1)
        
        """
        # Colorbar Label Font Size
        self.cbar_label_fontsize_label = QLabel("Label Font Size:")
        self.cbar_label_fontsize_input = QSpinBox()
        self.cbar_label_fontsize_input.setRange(1, 100)
        self.cbar_label_fontsize_input.setValue(self.config_manager.config.get('colorbar_label_fontsize', 12))
        colorbar_layout.addWidget(self.cbar_label_fontsize_label, 11, 0)
        colorbar_layout.addWidget(self.cbar_label_fontsize_input, 11, 1)
        
        # Colorbar Label Color
        self.cbar_label_color_label = QLabel("Label Color:")
        self.cbar_label_color_input = self.create_color_combobox(self.config_manager.config.get('colorbar_label_color', 'black'))
        self.cbar_label_color_input.setFixedWidth(100)
        colorbar_layout.addWidget(self.cbar_label_color_label, 12, 0)
        colorbar_layout.addWidget(self.cbar_label_color_input, 12, 1)
        """
        
        self.cbar_tick_color_label = QLabel("Tick Color:")
        self.cbar_tick_color_input = self.create_color_combobox(self.config_manager.config.get('colorbar_tick_color', 'black'))
        self.cbar_tick_color_input.setFixedWidth(100)
        colorbar_layout.addWidget(self.cbar_tick_color_label, 12, 0)
        colorbar_layout.addWidget(self.cbar_tick_color_input, 12, 1)
        
        # Colorbar Tick Label Color
        self.cbar_tick_label_color_label = QLabel("Tick Label Color:")
        self.cbar_tick_label_color_input = self.create_color_combobox(self.config_manager.config.get('colorbar_tick_labelcolor', 'black'))
        self.cbar_tick_label_color_input.setFixedWidth(100)
        colorbar_layout.addWidget(self.cbar_tick_label_color_label, 13, 0)
        colorbar_layout.addWidget(self.cbar_tick_label_color_input, 13, 1)
        
        # Colorbar Tick Labels - Left
        self.cbar_tick_labelleft_checkbox = QCheckBox("Tick Label Left (if unchecked, Right)")
        self.cbar_tick_labelleft_checkbox.setChecked(self.config_manager.config.get('colorbar_tick_labelleft', False))
        colorbar_layout.addWidget(self.cbar_tick_labelleft_checkbox, 14, 0)
        
        # Colorbar Tick Labels - Top
        self.cbar_tick_labeltop_checkbox = QCheckBox("Tick Label Top (if unchecked, Bottom)")
        self.cbar_tick_labeltop_checkbox.setChecked(self.config_manager.config.get('colorbar_tick_labeltop', False))
        colorbar_layout.addWidget(self.cbar_tick_labeltop_checkbox, 15, 0)
        
        # Put the layout into a QWidget

        colorbar_settings_widget = QWidget()
        colorbar_settings_widget.setLayout(colorbar_layout)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(colorbar_settings_widget)
        return scroll_area
        

    def set_coordinate_format(self, use_decimal: bool):
        """Select the coordinate format radio button based on the config flag."""
        if use_decimal is None:
            use_decimal = True
        if use_decimal:
            self.decimal_radio.setChecked(True)
        else:
            self.sexagesimal_radio.setChecked(True)

    @staticmethod
    def _values_equivalent(current, new):
        """Check whether two config values should be considered identical."""
        if isinstance(current, bool) or isinstance(new, bool):
            return current == new
        if isinstance(current, (int, float)) and isinstance(new, (int, float)):
            return math.isclose(float(current), float(new), rel_tol=1e-9, abs_tol=1e-9)
        return current == new

    def _update_config_value(self, key, value, changed_keys):
        """Write a config value only when it actually changes."""
        current = self.config_manager.config.get(key)
        if self._values_equivalent(current, value):
            return
        self.config_manager.config[key] = value
        changed_keys.add(key)

    def keyPressEvent(self, event):
        """Allow pressing Enter/Return anywhere in the panel to trigger Apply."""
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.apply_button.click()
            event.accept()
            return
        super().keyPressEvent(event)

    def apply_changes(self):
        """Apply the changes to all open FITS viewers."""
        updated_keys = set()

        updates = [
            # General settings
            ('colorscale', self.colorscale_input.currentText()),
            ('decimal', self.decimal_radio.isChecked()),
            ('number_decimals', self.number_decimals_input.value()),
            ('coord_wrap', self.wrap_angle_input.value()),
            ('scrollspeed', self.scrollspeed_input.value()),
            ('range_file', self.range_file_input.text()),

            # Display settings
            ('fig_background_color', self.fig_background_color_input.currentText()),
            ('ax_background_color', self.ax_background_color_input.currentText()),
            ('bad_color', self.bad_color_input.currentText()),
            ('figure_pos_x', int(self.figure_pos_x_input.text())),
            ('figure_pos_y', int(self.figure_pos_y_input.text())),
            ('figure_width', int(self.figure_width_input.text())),
            ('figure_height', int(self.figure_height_input.text())),
            ('axislabel_fontsize', self.axislabel_fontsize_input.value()),
            ('axislabel_fontfamily', self.axislabel_fontfamily_input.currentText()),
            ('axislabel_color', self.axislabel_color_input.currentText()),

            # Axis positions
            ('ax_pos_l', self.axis_left_spinbox.value()),
            ('ax_pos_r', self.axis_right_spinbox.value()),
            ('ax_pos_t', self.axis_top_spinbox.value()),
            ('ax_pos_b', self.axis_bottom_spinbox.value()),

            # Beam
            ('beam_facecolor', self.beam_facecolor_input.currentText()),
            ('beam_edgecolor', self.beam_edgecolor_input.currentText()),
            ('beam_linewidth', self.beam_linewidth_input.value()),
            ('beam_pos_x', self.beam_pos_x_spinbox.value()),
            ('beam_pos_y', self.beam_pos_y_spinbox.value()),

            # Colorbar settings
            ('colorbar_orientation', self.cbar_orientation_combo.currentText()),
            ('cbar_pos_x', self.cbar_x_input.value()),
            ('cbar_pos_y', self.cbar_y_input.value()),
            ('cbar_width', self.cbar_width_input.value()),
            ('cbar_height', self.cbar_height_input.value()),
            ('colorbar_tick_direction', self.cbar_tick_direction_combo.currentText()),
            ('colorbar_tick_left', self.cbar_tick_left_checkbox.isChecked()),
            ('colorbar_tick_right', self.cbar_tick_right_checkbox.isChecked()),
            ('colorbar_tick_top', self.cbar_tick_top_checkbox.isChecked()),
            ('colorbar_tick_bottom', self.cbar_tick_bottom_checkbox.isChecked()),
            ('colorbar_tick_length', self.cbar_tick_length_input.value()),
            ('colorbar_tick_width', self.cbar_tick_width_input.value()),
            ('colorbar_mtick_freq', self.cbar_mtick_freq_input.value()),
            ('colorbar_mtick_length', self.cbar_mtick_length_input.value()),
            ('colorbar_tick_color', self.cbar_tick_color_input.currentText()),
            ('colorbar_tick_labelcolor', self.cbar_tick_label_color_input.currentText()),
            ('colorbar_tick_labelleft', self.cbar_tick_labelleft_checkbox.isChecked()),
            ('colorbar_tick_labeltop', self.cbar_tick_labeltop_checkbox.isChecked()),

            # Tick settings
            ('tick_labelsize', self.tick_labelsize_input.value()),
            ('tick_color', self.tick_color_input.currentText()),
            ('tick_labelcolor', self.tick_labelcolor_input.currentText()),
            ('tick_direction', self.tick_direction_input.currentText()),
            ('default_ticks_position', self.default_ticks_position_input.currentText()),
            ('xticklabel_position', self.xticklabel_position_input.currentText()),
            ('yticklabel_position', self.yticklabel_position_input.currentText()),
            ('tick_length', self.tick_length_input.value()),
            ('mtick_length', self.mtick_length_input.value()),
            ('tick_width', self.tick_width_input.value()),
            ('tick_xlabelrotation', self.tick_xlabelrotation_input.value()),
            ('tick_ylabelrotation', self.tick_ylabelrotation_input.value()),
            ('tick_pad_x', self.tick_xpad_input.value()),
            ('tick_pad_y', self.tick_ypad_input.value()),
            ('tick_font', self.ticklabel_fontfamily_input.currentText()),
            ('x_mtick_freq', self.x_mtick_freq_input.value()),
            ('y_mtick_freq', self.y_mtick_freq_input.value()),
            ('z_mtick_freq', self.z_mtick_freq_input.value()),

            # Click label settings
            ('click_label_color', self.click_label_color_input.currentText()),
            ('click_linewidth', self.click_linewidth_input.value()),
            ('click_linecolor', self.click_linecolor_input.currentText()),
            ('poslabel_x', self.poslabel_x_input.value()),
            ('poslabel_y', self.poslabel_y_input.value()),
            ('poslabel_w', self.poslabel_w_input.value()),
            ('poslabel_h', self.poslabel_h_input.value()),
            ('ch_label_color', self.ch_label_color_input.currentText()),
            ('ch_label_size', self.ch_label_size_input.value()),
            ('ch_label_font', self.ch_label_font_input.currentText()),
            ('pos_chlabel_x', self.pos_chlabel_x_input.value()),
            ('pos_chlabel_y', self.pos_chlabel_y_input.value()),
            # ('pos_chlabel_w', self.pos_chlabel_w_input.value()),
            # ('pos_chlabel_h', self.pos_chlabel_h_input.value()),
        ]

        for key, value in updates:
            self._update_config_value(key, value, updated_keys)

        if not updated_keys:
            return

        if updated_keys.intersection({'decimal', 'number_decimals', 'coord_wrap'}):
            self.fits_viewer.refresh_coordinate_format()

        # Apply the changes to all currently open FITS viewers
        try:
            self.fits_viewer.reload_viewer()
        except ValueError as e:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Icon.Critical)
            msg_box.setText(f"Error: Invalid config parameter(s) detected.\nDetails: {e}")
            msg_box.setWindowTitle("Parameter Error")
            msg_box.exec()
        



    def save_config(self):
        self.apply_changes()
        """Save the current config to the config.yaml file."""
        if os.path.exists(self.config_manager.config_file):
            reply = QMessageBox.question(self, 'Overwrite Confirmation',
                                        'The config.yaml file already exists. Do you want to overwrite it?',
                                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                        QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No:
                return
        try:
            with open(self.config_manager.config_file, 'w') as f:
                yaml.safe_dump(self.config_manager.config, f, default_flow_style=False, sort_keys=False)
            QMessageBox.information(self, 'Success', 'Configuration saved successfully.')
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to save configuration: {str(e)}')

    def reset_to_loaded_config(self):
        """Reset the configuration to default values."""
        # General settings
        self.colorscale_input.setCurrentText(self.config_manager.config_bu.get('colorscale'))
        self.set_coordinate_format(self.config_manager.config_bu.get('decimal'))
        self.number_decimals_input.setValue(self.config_manager.config_bu.get('number_decimals'))
        self.wrap_angle_input.setValue(self.config_manager.config_bu.get('coord_wrap'))
        self.scrollspeed_input.setValue(self.config_manager.config_bu.get('scrollspeed'))
        self.range_file_input.setText(self.config_manager.config_bu.get('range_file'))
        
        # Display settings
        self.fig_background_color_input.setCurrentText(self.config_manager.config_bu.get('fig_background_color'))
        self.ax_background_color_input.setCurrentText(self.config_manager.config_bu.get('ax_background_color'))
        self.bad_color_input.setCurrentText(self.config_manager.config_bu.get('bad_color'))
        self.figure_pos_x_input.setValue(self.config_manager.config_bu.get('figure_pos_x'))
        self.figure_pos_y_input.setValue(self.config_manager.config_bu.get('figure_pos_y'))
        self.figure_width_input.setValue(self.config_manager.config_bu.get('figure_width'))
        self.figure_height_input.setValue(self.config_manager.config_bu.get('figure_height'))
        self.axislabel_fontsize_input.setValue(self.config_manager.config_bu.get('axislabel_fontsize'))
        self.axislabel_fontfamily_input.setCurrentText(self.config_manager.config_bu.get('axislabel_fontfamily'))
        self.axislabel_color_input.setCurrentText(self.config_manager.config_bu.get('axislabel_color'))
        
        # Axis positions
        self.axis_left_spinbox.setValue(self.config_manager.config_bu.get('ax_pos_l'))
        self.axis_right_spinbox.setValue(self.config_manager.config_bu.get('ax_pos_r'))
        self.axis_top_spinbox.setValue(self.config_manager.config_bu.get('ax_pos_t'))
        self.axis_bottom_spinbox.setValue(self.config_manager.config_bu.get('ax_pos_b'))
        
        #Beam
        self.beam_facecolor_input.setCurrentText(self.config_manager.config_bu.get('beam_facecolor'))
        self.beam_edgecolor_input.setCurrentText(self.config_manager.config_bu.get('beam_edgecolor'))
        self.beam_linewidth_input.setValue(self.config_manager.config_bu.get('beam_linewidth'))
        self.beam_pos_x_spinbox.setValue(self.config_manager.config_bu.get('beam_pos_x'))
        self.beam_pos_y_spinbox.setValue(self.config_manager.config_bu.get('beam_pos_y'))
        
        # Colorbar settings
        self.cbar_orientation_combo.setCurrentText(self.config_manager.config_bu.get('colorbar_orientation'))
        self.cbar_x_input.setValue(self.config_manager.config_bu.get('cbar_pos_x'))
        self.cbar_y_input.setValue(self.config_manager.config_bu.get('cbar_pos_y'))
        self.cbar_width_input.setValue(self.config_manager.config_bu.get('cbar_width'))
        self.cbar_height_input.setValue(self.config_manager.config_bu.get('cbar_height'))
        
        self.cbar_tick_direction_combo.setCurrentText(self.config_manager.config_bu.get('colorbar_tick_direction'))
        self.cbar_tick_left_checkbox.setChecked(self.config_manager.config_bu.get('colorbar_tick_left'))
        self.cbar_tick_right_checkbox.setChecked(self.config_manager.config_bu.get('colorbar_tick_right'))
        self.cbar_tick_top_checkbox.setChecked(self.config_manager.config_bu.get('colorbar_tick_top'))
        self.cbar_tick_bottom_checkbox.setChecked(self.config_manager.config_bu.get('colorbar_tick_bottom'))
        
        self.cbar_tick_length_input.setValue(self.config_manager.config_bu.get('colorbar_tick_length'))
        self.cbar_tick_width_input.setValue(self.config_manager.config_bu.get('colorbar_tick_width'))
        
        self.cbar_mtick_freq_input.setValue(self.config_manager.config_bu.get('colorbar_mtick_freq'))
        self.cbar_mtick_length_input.setValue(self.config_manager.config_bu.get('colorbar_mtick_length'))
        # self.cbar_label_fontsize_input.setValue(self.config_manager.config_bu.get('colorbar_label_fontsize'))
        # self.cbar_label_color_input.setCurrentText(self.config_manager.config_bu.get('colorbar_label_color'))
        self.cbar_tick_color_input.setCurrentText(self.config_manager.config_bu.get('colorbar_tick_color'))
        self.cbar_tick_label_color_input.setCurrentText(self.config_manager.config_bu.get('colorbar_tick_labelcolor'))
        self.cbar_tick_labelleft_checkbox.setChecked(self.config_manager.config_bu.get('colorbar_tick_labelleft'))
        self.cbar_tick_labeltop_checkbox.setChecked(self.config_manager.config_bu.get('colorbar_tick_labeltop'))
        
        # Ticks settings
        self.tick_labelsize_input.setValue(self.config_manager.config_bu.get('tick_labelsize'))
        self.tick_color_input.setCurrentText(self.config_manager.config_bu.get('tick_color'))
        self.tick_labelcolor_input.setCurrentText(self.config_manager.config_bu.get('tick_labelcolor'))
        self.tick_direction_input.setCurrentText(self.config_manager.config_bu.get('tick_direction'))
        self.default_ticks_position_input.setCurrentText(self.config_manager.config_bu.get('default_ticks_position'))
        self.xticklabel_position_input.setCurrentText(self.config_manager.config_bu.get('xticklabel_position'))
        self.yticklabel_position_input.setCurrentText(self.config_manager.config_bu.get('yticklabel_position'))
        self.tick_length_input.setValue(self.config_manager.config_bu.get('tick_length'))
        self.mtick_length_input.setValue(self.config_manager.config_bu.get('mtick_length'))
        self.tick_width_input.setValue(self.config_manager.config_bu.get('tick_width'))
        self.tick_xlabelrotation_input.setValue(self.config_manager.config_bu.get('tick_xlabelrotation'))
        self.tick_ylabelrotation_input.setValue(self.config_manager.config_bu.get('tick_ylabelrotation'))
        self.tick_xpad_input.setValue(self.config_manager.config_bu.get('tick_pad_x'))
        self.tick_ypad_input.setValue(self.config_manager.config_bu.get('tick_pad_y'))
        self.ticklabel_fontfamily_input.setCurrentText(self.config_manager.config_bu.get('tick_font'))
        
        self.x_mtick_freq_input.setValue(self.config_manager.config_bu.get('x_mtick_freq'))
        self.y_mtick_freq_input.setValue(self.config_manager.config_bu.get('y_mtick_freq'))
        self.z_mtick_freq_input.setValue(self.config_manager.config_bu.get('z_mtick_freq'))
        
        # Click settings
        self.click_label_color_input.setCurrentText(self.config_manager.config_bu.get('click_label_color'))
        self.click_linewidth_input.setValue(self.config_manager.config_bu.get('click_linewidth'))
        self.click_linecolor_input.setCurrentText(self.config_manager.config_bu.get('click_linecolor'))
        self.poslabel_x_input.setValue(self.config_manager.config_bu.get('poslabel_x'))
        self.poslabel_y_input.setValue(self.config_manager.config_bu.get('poslabel_y'))
        self.poslabel_w_input.setValue(self.config_manager.config_bu.get('poslabel_w'))
        self.poslabel_h_input.setValue(self.config_manager.config_bu.get('poslabel_h'))
        
        self.ch_label_color_input.setCurrentText(self.config_manager.config_bu.get('ch_label_color'))
        self.ch_label_size_input.setValue(self.config_manager.config_bu.get('ch_label_size'))
        self.ch_label_font_input.setCurrentText(self.config_manager.config_bu.get('ch_label_font'))
        self.pos_chlabel_x_input.setValue(self.config_manager.config_bu.get('pos_chlabel_x'))
        self.pos_chlabel_y_input.setValue(self.config_manager.config_bu.get('pos_chlabel_y'))
        #self.pos_chlabel_w_input.setValue(self.config_manager.config_bu.get('pos_chlabel_w'))
        #self.pos_chlabel_h_input.setValue(self.config_manager.config_bu.get('pos_chlabel_h'))


    def reset_to_default(self):
        """Reset the configuration to default values."""
        #self.config_manager.reset_to_default()
        # General settings
        self.colorscale_input.setCurrentText(self.config_manager.default_config.get('colorscale'))
        self.set_coordinate_format(self.config_manager.default_config.get('decimal'))
        self.number_decimals_input.setValue(self.config_manager.default_config.get('number_decimals'))
        self.wrap_angle_input.setValue(self.config_manager.default_config.get('coord_wrap'))
        self.scrollspeed_input.setValue(self.config_manager.default_config.get('scrollspeed'))
        self.range_file_input.setText(self.config_manager.default_config.get('range_file'))
        
        # Display settings
        self.fig_background_color_input.setCurrentText(self.config_manager.default_config.get('fig_background_color'))
        self.ax_background_color_input.setCurrentText(self.config_manager.default_config.get('ax_background_color'))
        self.bad_color_input.setCurrentText(self.config_manager.default_config.get('bad_color'))
        self.figure_pos_x_input.setValue(self.config_manager.default_config.get('figure_pos_x'))
        self.figure_pos_y_input.setValue(self.config_manager.default_config.get('figure_pos_y'))
        self.figure_width_input.setValue(self.config_manager.default_config.get('figure_width'))
        self.figure_height_input.setValue(self.config_manager.default_config.get('figure_height'))
        self.axislabel_fontsize_input.setValue(self.config_manager.default_config.get('axislabel_fontsize'))
        self.axislabel_fontfamily_input.setCurrentText(self.config_manager.default_config.get('axislabel_fontfamily'))
        self.axislabel_color_input.setCurrentText(self.config_manager.default_config.get('axislabel_color'))
        
        # Axis positions
        self.axis_left_spinbox.setValue(self.config_manager.default_config.get('ax_pos_l'))
        self.axis_right_spinbox.setValue(self.config_manager.default_config.get('ax_pos_r'))
        self.axis_top_spinbox.setValue(self.config_manager.default_config.get('ax_pos_t'))
        self.axis_bottom_spinbox.setValue(self.config_manager.default_config.get('ax_pos_b'))
        
        #Beam
        self.beam_facecolor_input.setCurrentText(self.config_manager.default_config.get('beam_facecolor'))
        self.beam_edgecolor_input.setCurrentText(self.config_manager.default_config.get('beam_edgecolor'))
        self.beam_linewidth_input.setValue(self.config_manager.default_config.get('beam_linewidth'))
        self.beam_pos_x_spinbox.setValue(self.config_manager.default_config.get('beam_pos_x'))
        self.beam_pos_y_spinbox.setValue(self.config_manager.default_config.get('beam_pos_y'))
        
        # Colorbar settings
        self.cbar_orientation_combo.setCurrentText(self.config_manager.default_config.get('colorbar_orientation'))
        self.cbar_x_input.setValue(self.config_manager.default_config.get('cbar_pos_x'))
        self.cbar_y_input.setValue(self.config_manager.default_config.get('cbar_pos_y'))
        self.cbar_width_input.setValue(self.config_manager.default_config.get('cbar_width'))
        self.cbar_height_input.setValue(self.config_manager.default_config.get('cbar_height'))
        
        self.cbar_tick_direction_combo.setCurrentText(self.config_manager.default_config.get('colorbar_tick_direction'))
        self.cbar_tick_left_checkbox.setChecked(self.config_manager.default_config.get('colorbar_tick_left'))
        self.cbar_tick_right_checkbox.setChecked(self.config_manager.default_config.get('colorbar_tick_right'))
        self.cbar_tick_top_checkbox.setChecked(self.config_manager.default_config.get('colorbar_tick_top'))
        self.cbar_tick_bottom_checkbox.setChecked(self.config_manager.default_config.get('colorbar_tick_bottom'))
        
        self.cbar_tick_length_input.setValue(self.config_manager.default_config.get('colorbar_tick_length'))
        self.cbar_tick_width_input.setValue(self.config_manager.default_config.get('colorbar_tick_width'))
        
        self.cbar_mtick_freq_input.setValue(self.config_manager.default_config.get('colorbar_mtick_freq'))
        self.cbar_mtick_length_input.setValue(self.config_manager.default_config.get('colorbar_mtick_length'))
        # self.cbar_label_fontsize_input.setValue(self.config_manager.default_config.get('colorbar_label_fontsize'))
        # self.cbar_label_color_input.setCurrentText(self.config_manager.default_config.get('colorbar_label_color'))
        self.cbar_tick_color_input.setCurrentText(self.config_manager.default_config.get('colorbar_tick_color'))
        self.cbar_tick_label_color_input.setCurrentText(self.config_manager.default_config.get('colorbar_tick_labelcolor'))
        self.cbar_tick_labelleft_checkbox.setChecked(self.config_manager.default_config.get('colorbar_tick_labelleft'))
        self.cbar_tick_labeltop_checkbox.setChecked(self.config_manager.default_config.get('colorbar_tick_labeltop'))
        
        
        # Ticks settings
        self.tick_labelsize_input.setValue(self.config_manager.default_config.get('tick_labelsize'))
        self.tick_color_input.setCurrentText(self.config_manager.default_config.get('tick_color'))
        self.tick_labelcolor_input.setCurrentText(self.config_manager.default_config.get('tick_labelcolor'))
        self.tick_direction_input.setCurrentText(self.config_manager.default_config.get('tick_direction'))
        self.default_ticks_position_input.setCurrentText(self.config_manager.default_config.get('default_ticks_position'))
        self.xticklabel_position_input.setCurrentText(self.config_manager.default_config.get('xticklabel_position'))
        self.yticklabel_position_input.setCurrentText(self.config_manager.default_config.get('yticklabel_position'))
        self.tick_length_input.setValue(self.config_manager.default_config.get('tick_length'))
        self.mtick_length_input.setValue(self.config_manager.default_config.get('mtick_length'))
        self.tick_width_input.setValue(self.config_manager.default_config.get('tick_width'))
        self.tick_xlabelrotation_input.setValue(self.config_manager.default_config.get('tick_xlabelrotation'))
        self.tick_ylabelrotation_input.setValue(self.config_manager.default_config.get('tick_ylabelrotation'))
        self.tick_xpad_input.setValue(self.config_manager.default_config.get('tick_pad_x'))
        self.tick_ypad_input.setValue(self.config_manager.default_config.get('tick_pad_y'))
        self.ticklabel_fontfamily_input.setCurrentText(self.config_manager.default_config.get('tick_font'))
        self.x_mtick_freq_input.setValue(self.config_manager.default_config.get('x_mtick_freq'))
        self.y_mtick_freq_input.setValue(self.config_manager.default_config.get('y_mtick_freq'))
        self.z_mtick_freq_input.setValue(self.config_manager.default_config.get('z_mtick_freq'))
        
        # Click settings
        self.click_label_color_input.setCurrentText(self.config_manager.default_config.get('click_label_color'))
        self.click_linewidth_input.setValue(self.config_manager.default_config.get('click_linewidth'))
        self.click_linecolor_input.setCurrentText(self.config_manager.default_config.get('click_linecolor'))
        self.poslabel_x_input.setValue(self.config_manager.default_config.get('poslabel_x'))
        self.poslabel_y_input.setValue(self.config_manager.default_config.get('poslabel_y'))
        self.poslabel_w_input.setValue(self.config_manager.default_config.get('poslabel_w'))
        self.poslabel_h_input.setValue(self.config_manager.default_config.get('poslabel_h'))
        
        self.ch_label_color_input.setCurrentText(self.config_manager.default_config.get('ch_label_color'))
        self.ch_label_size_input.setValue(self.config_manager.default_config.get('ch_label_size'))
        self.ch_label_font_input.setCurrentText(self.config_manager.default_config.get('ch_label_font'))
        self.pos_chlabel_x_input.setValue(self.config_manager.default_config.get('pos_chlabel_x'))
        self.pos_chlabel_y_input.setValue(self.config_manager.default_config.get('pos_chlabel_y'))
        #self.pos_chlabel_w_input.setValue(self.config_manager.default_config.get('pos_chlabel_w'))
        #self.pos_chlabel_h_input.setValue(self.config_manager.default_config.get('pos_chlabel_h'))
