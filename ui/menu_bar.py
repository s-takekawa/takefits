from PyQt6.QtWidgets import QMenu
from PyQt6.QtGui import QAction, QActionGroup
from core.config import ConfigPanel
from logic.show_header import ShowHeader


class MenuBar:
    _instance = None 
    @staticmethod
    def get_instance(parent):
        if MenuBar._instance is None:
            MenuBar._instance = MenuBar(parent)
        return MenuBar._instance

    def __init__(self, parent):
        self.parent = parent
        self.make_menu()
        
    def make_menu(self):
        menubar = self.parent.menuBar()
        
        # Preferences
        dummy_menu = menubar.addMenu("Preferences")
        preferences_action = QAction("Preferences", self.parent)
        dummy_menu.addAction(preferences_action)
        preferences_action.triggered.connect(self.open_config_panel)
        

        # Window Menu
        window_menu = menubar.addMenu("Window")
        
        header_action = QAction('Show Header', self.parent)
        window_menu.addAction(header_action)
        header_action.triggered.connect(self.open_header_panel)
        
        self.control_panel_action = QAction("ToolsPanel", self.parent, checkable=True)
        window_menu.addAction(self.control_panel_action)
        self.control_panel_action.triggered.connect(self.toggle_control_panel)
        self.control_panel_action.setChecked(True)
        
        self.range_panel_action = QAction("RangePanel", self.parent, checkable=True)
        window_menu.addAction(self.range_panel_action)
        self.range_panel_action.triggered.connect(self.toggle_range_panel)
        self.range_panel_action.setChecked(True)
        
        self.plane_menu = QMenu("Plane", self.parent)
        window_menu.addMenu(self.plane_menu)
        self.main_action = QAction("X-Y (Main)", self.parent, checkable=True)
        self.sub1_action = QAction("X-Z (Sub1)", self.parent, checkable=True)
        self.sub2_action = QAction("Z-Y (Sub2)", self.parent, checkable=True)
        
        self.main_action.setChecked(True)
        
        self.main_action.triggered.connect(self.toggle_main_window)
        self.sub1_action.triggered.connect(self.toggle_sub1_window)
        self.sub2_action.triggered.connect(self.toggle_sub2_window)
        
        self.plane_menu.addAction(self.main_action)
        self.plane_menu.addAction(self.sub1_action)
        self.plane_menu.addAction(self.sub2_action)
        
        # Tools Menu
        tools_menu = menubar.addMenu("Tools")
        tools_menu.addSeparator() # Add a visual separator in the menu

        chmap_action = QAction("Channel Map", self.parent)
        tools_menu.addAction(chmap_action)
        chmap_action.triggered.connect(self.open_chmap_panel)
        if self.parent.wcs.wcs.naxis < 3: chmap_action.setEnabled(False)
        
        actions = []

        contour_action = QAction("Contours", self.parent)
        contour_action.triggered.connect(self.open_contour_panel)
        actions.append(contour_action)

        marker_action = QAction("Markers", self.parent)
        marker_action.triggered.connect(self.parent.open_marker_panel)
        actions.append(marker_action)

        colorscale_action = QAction("Color Settings", self.parent)
        colorscale_action.triggered.connect(self.open_colorscale_panel)
        actions.append(colorscale_action)

        cutout_action = QAction("Cut Out", self.parent)
        cutout_action.triggered.connect(self.open_cutout_dialog)
        actions.append(cutout_action)

        integ_action = QAction("Integration", self.parent)
        integ_action.triggered.connect(self.open_integ_panel)
        if self.parent.wcs.wcs.naxis < 3:
            integ_action.setEnabled(False)
        actions.append(integ_action)

        mask_panel_action = QAction("Mask", self.parent)
        mask_panel_action.triggered.connect(self.open_mask_panel)
        actions.append(mask_panel_action)

        pvd_action = QAction("PV diagram", self.parent)
        pvd_action.triggered.connect(self.open_pvd_panel)
        if self.parent.wcs.wcs.naxis < 3:
            pvd_action.setEnabled(False)
        actions.append(pvd_action)

        regrid_action = QAction("Regrid", self.parent)
        regrid_action.triggered.connect(self.open_regrid_panel)
        actions.append(regrid_action)

        scaling_action = QAction("Scaling", self.parent)
        scaling_action.triggered.connect(self.open_scaling_panel)
        actions.append(scaling_action)

        smooth_action = QAction("Smoothing", self.parent)
        smooth_action.triggered.connect(self.open_smooth_panel)
        actions.append(smooth_action)

        spec_action = QAction("Spectrum", self.parent)
        spec_action.triggered.connect(self.open_spec_window)
        if self.parent.wcs.wcs.naxis < 3:
            spec_action.setEnabled(False)
        actions.append(spec_action)

        unit_conversion_action = QAction("Unit Conversion", self.parent)
        unit_conversion_action.triggered.connect(self.open_unit_conversion_panel)
        actions.append(unit_conversion_action)

        for action in sorted(actions, key=lambda a: a.text()):
            tools_menu.addAction(action)


        region_menu = menubar.addMenu("Region")
        self.circle_action = QAction("Circle", self.parent, checkable=True)
        self.rectangle_action = QAction("Rectangle", self.parent, checkable=True)
        self.ellipse_action = QAction("Ellipse", self.parent, checkable=True)
        self.cube_action = QAction("Cube", self.parent, checkable=True) 
        
        region_group = QActionGroup(self.parent)
        region_group.setExclusive(True)
        region_group.addAction(self.circle_action)
        region_group.addAction(self.rectangle_action)
        region_group.addAction(self.ellipse_action)
        region_group.addAction(self.cube_action) 
        
        region_menu.addAction(self.circle_action)
        region_menu.addAction(self.rectangle_action)
        region_menu.addAction(self.ellipse_action)
        region_menu.addAction(self.cube_action)

        if self.parent.wcs.wcs.naxis < 3:
            self.cube_action.setEnabled(False)

        region_menu.addSeparator()
        clear_all_action = QAction("Clear All", self.parent)
        clear_all_action.triggered.connect(self.parent.clear_all_regions_globally)
        region_menu.addAction(clear_all_action)
        
    def enable_plane_menu(self, enabled):
        self.plane_menu.setEnabled(enabled)

    def open_regrid_panel(self):
        if hasattr(self.parent, "open_regrid_panel"):
            self.parent.open_regrid_panel()
        
    def toggle_control_panel(self):
        if self.control_panel_action.isChecked():
            self.parent.show_control_panel()
        else:
            self.parent.hide_control_panel()
            
    def toggle_range_panel(self):
        if self.range_panel_action.isChecked():
            self.parent.show_range_panel()
        else:
            self.parent.hide_range_panel()


    def toggle_main_window(self):
        if self.main_action.isChecked():
            self.parent.show_main_window()
        else:
            self.parent.hide_main_window()

    def toggle_sub1_window(self):
        if self.sub1_action.isChecked():
            self.parent.SubWindow.subwindow1.show()
        else:
            self.parent.SubWindow.subwindow1.hide()
    
    def toggle_sub2_window(self):
        if self.sub2_action.isChecked():
            self.parent.SubWindow.subwindow2.show()
        else:
            self.parent.SubWindow.subwindow2.hide()
        
        
    def open_colorscale_panel(self):
        if self.parent.control_panel.color_settings_panel is None:
            self.parent.control_panel.open_color_settings()
        else:
            self.parent.control_panel.color_settings_panel.raise_()
            self.parent.control_panel.color_settings_panel.activateWindow()
            
    def open_scaling_panel(self):
        if self.parent.control_panel.scaling_panel is None:
            self.parent.control_panel.open_scaling_panel()
        else:
            self.parent.control_panel.scaling_panel.raise_()
            self.parent.control_panel.scaling_panel.activateWindow()

    def open_unit_conversion_panel(self):
        if self.parent.control_panel.unit_conversion_panel is None:
            self.parent.control_panel.open_unit_conversion_panel()
        else:
            self.parent.control_panel.unit_conversion_panel.raise_()
            self.parent.control_panel.unit_conversion_panel.activateWindow()
    
    def open_chmap_panel(self):
        if self.parent.control_panel.chmap_settings_panel is None:
            self.parent.control_panel.open_chmap_settings()
        else:
            self.parent.control_panel.chmap_settings_panel.raise_()
            self.parent.control_panel.chmap_settings_panel.activateWindow()
    
    def open_integ_panel(self):
        if self.parent.control_panel.integ_settings_panel is None:
            self.parent.control_panel.open_integ_settings()
        else:
            self.parent.control_panel.integ_settings_panel.raise_()
            self.parent.control_panel.integ_settings_panel.activateWindow()

    def open_cutout_dialog(self):
        if hasattr(self.parent, 'open_cutout_dialog'):
            self.parent.open_cutout_dialog(use_view_bounds=True)

    def open_mask_panel(self):
        if self.parent.control_panel.mask_settings_panel is None:
            self.parent.control_panel.open_mask_settings()
        else:
            self.parent.control_panel.mask_settings_panel.raise_()
            self.parent.control_panel.mask_settings_panel.activateWindow()            

    def open_pvd_panel(self):
        if self.parent.control_panel.pvd_panel is None:
            self.parent.control_panel.open_pvd_settings()
        else:
            self.parent.control_panel.pvd_panel.raise_()
            self.parent.control_panel.pvd_panel.activateWindow()
    
    def open_spec_window(self):
        if self.parent.control_panel.spec_window is None:
            self.parent.control_panel.open_spec_window()
        else:
            self.parent.control_panel.spec_window.raise_()
            self.parent.control_panel.spec_window.activateWindow()

    def open_smooth_panel(self):
        if self.parent.control_panel.smooth_settings_panel is None:
            self.parent.control_panel.open_smooth_settings()
        else:
            self.parent.control_panel.smooth_settings_panel.raise_()
            self.parent.control_panel.smooth_settings_panel.activateWindow()

    def open_contour_panel(self):
        if self.parent.control_panel is None:
            return
        if self.parent.control_panel.contour_panel is None:
            self.parent.control_panel.open_contour_panel()
        else:
            self.parent.control_panel.contour_panel.raise_()
            self.parent.control_panel.contour_panel.activateWindow()


    def open_config_panel(self):
        self.config_panel = ConfigPanel(self.parent.config_manager, self.parent)
        self.config_panel.show()
        
    def open_header_panel(self):
        self.header_panel = ShowHeader(self.parent.header)
        self.header_panel.resize(300, 400)
        self.header_panel.show()
