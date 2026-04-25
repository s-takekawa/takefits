from PySide6.QtWidgets import QMenu, QMessageBox
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtCore import Qt
from takefits.ui.config_panel import ConfigPanel
from takefits.logic.show_header import ShowHeader
from takefits.core.version import APP_DISPLAY_VERSION, APP_NAME
from takefits.core.wcs_frames import (
    available_display_frames,
    display_frame_label,
    normalize_display_frame,
    preferred_display_frame,
)


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

        # File Menu
        file_menu = menubar.addMenu("File")
        self.save_workspace_action = QAction("Save Workspace", self.parent)
        self.load_workspace_action = QAction("Load Workspace...", self.parent)
        self.save_recipe_action = QAction("Save Recipe...", self.parent)
        self.load_recipe_action = QAction("Load Recipe...", self.parent)
        self.save_workspace_action.setShortcut(QKeySequence.StandardKey.Save)
        self.load_workspace_action.setShortcut(QKeySequence.StandardKey.Open)
        self.save_recipe_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        self.load_recipe_action.setShortcut(QKeySequence("Ctrl+Shift+O"))
        for action in (
            self.save_workspace_action,
            self.load_workspace_action,
            self.save_recipe_action,
            self.load_recipe_action,
        ):
            action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.save_workspace_action.triggered.connect(self.save_workspace)
        self.load_workspace_action.triggered.connect(self.load_workspace)
        self.save_recipe_action.triggered.connect(self.save_recipe)
        self.load_recipe_action.triggered.connect(self.load_recipe)
        file_menu.addAction(self.save_workspace_action)
        file_menu.addAction(self.load_workspace_action)
        file_menu.addSeparator()
        file_menu.addAction(self.save_recipe_action)
        file_menu.addAction(self.load_recipe_action)
        file_menu.addSeparator()
        self.show_header_action = QAction("Show Header", self.parent)
        self.show_header_action.triggered.connect(self.open_header_panel)
        file_menu.addAction(self.show_header_action)

        # Actions Menu (avoid macOS auto-injected Edit entries like Writing Tools / Emoji & Symbols)
        edit_menu = menubar.addMenu("Actions")
        self.undo_action = QAction("Undo Analysis", self.parent)
        self.redo_action = QAction("Redo Analysis", self.parent)
        self.undo_action.setShortcut(QKeySequence("Alt+Left"))
        self.redo_action.setShortcut(QKeySequence("Alt+Right"))
        self.undo_action.triggered.connect(self.undo_action_triggered)
        self.redo_action.triggered.connect(self.redo_action_triggered)
        edit_menu.addAction(self.undo_action)
        edit_menu.addAction(self.redo_action)
        self.set_undo_redo_enabled(False, False)
        edit_menu.addSeparator()
        self.view_back_action = QAction("View Back", self.parent)
        self.view_forward_action = QAction("View Forward", self.parent)
        self.view_back_action.setShortcut(QKeySequence.StandardKey.Undo)
        self.view_forward_action.setShortcut(QKeySequence.StandardKey.Redo)
        self.view_back_action.triggered.connect(self.view_back_triggered)
        self.view_forward_action.triggered.connect(self.view_forward_triggered)
        edit_menu.addAction(self.view_back_action)
        edit_menu.addAction(self.view_forward_action)
        self.set_view_navigation_enabled(False, False)
        
        # Preferences
        dummy_menu = menubar.addMenu("Preferences")
        preferences_action = QAction("Preferences", self.parent)
        dummy_menu.addAction(preferences_action)
        preferences_action.triggered.connect(self.open_config_panel)
        

        # Window Menu
        window_menu = menubar.addMenu("Window")

        self.control_panel_action = QAction("ToolsPanel", self.parent, checkable=True)
        window_menu.addAction(self.control_panel_action)
        self.control_panel_action.triggered.connect(self.toggle_control_panel)
        self.control_panel_action.setChecked(True)
        
        self.range_panel_action = QAction("RangePanel", self.parent, checkable=True)
        window_menu.addAction(self.range_panel_action)
        self.range_panel_action.triggered.connect(self.toggle_range_panel)
        self.range_panel_action.setChecked(True)
        
        window_menu.addSeparator()
        self.main_action = QAction("X-Y (Main)", self.parent, checkable=True)
        self.sub1_action = QAction("X-Z (Sub1)", self.parent, checkable=True)
        self.sub2_action = QAction("Z-Y (Sub2)", self.parent, checkable=True)
        
        self.main_action.setChecked(True)
        
        self.main_action.triggered.connect(self.toggle_main_window)
        self.sub1_action.triggered.connect(self.toggle_sub1_window)
        self.sub2_action.triggered.connect(self.toggle_sub2_window)
        
        window_menu.addAction(self.main_action)
        window_menu.addAction(self.sub1_action)
        window_menu.addAction(self.sub2_action)

        # Tools Menu
        tools_menu = menubar.addMenu("Tools")
        tools_menu.addSeparator() # Add a visual separator in the menu

        actions = []

        chmap_action = QAction("Channel Map", self.parent)
        chmap_action.triggered.connect(self.open_chmap_panel)
        if self.parent.wcs.wcs.naxis < 3: chmap_action.setEnabled(False)
        actions.append(chmap_action)

        clump_finding_action = QAction("Clump Finding", self.parent)
        clump_finding_action.triggered.connect(self.open_clump_finding_panel)
        actions.append(clump_finding_action)

        contour_action = QAction("Contours", self.parent)
        contour_action.triggered.connect(self.open_contour_panel)
        actions.append(contour_action)

        marker_action = QAction("Markers", self.parent)
        marker_action.triggered.connect(self.parent.open_marker_panel)
        actions.append(marker_action)

        colorscale_action = QAction("Color Settings", self.parent)
        colorscale_action.triggered.connect(self.open_colorscale_panel)
        actions.append(colorscale_action)
        
        arithmetic_action = QAction("Arithmetic", self.parent)
        arithmetic_action.triggered.connect(self.open_arithmetic_panel)
        actions.append(arithmetic_action)

        baseline_action = QAction("Baseline", self.parent)
        baseline_action.triggered.connect(self.open_baseline_panel)
        if self.parent.wcs.wcs.naxis < 3:
            baseline_action.setEnabled(False)
        actions.append(baseline_action)

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
        region_menu.addSeparator()
        self.save_regions_action = QAction("Save Regions...", self.parent)
        self.save_regions_action.triggered.connect(self.parent.save_regions_dialog)
        region_menu.addAction(self.save_regions_action)
        self.load_regions_action = QAction("Load Regions...", self.parent)
        self.load_regions_action.triggered.connect(self.parent.load_regions_dialog)
        region_menu.addAction(self.load_regions_action)

        wcs_menu = menubar.addMenu("WCS")
        self.wcs_frame_group = QActionGroup(self.parent)
        self.wcs_frame_group.setExclusive(True)
        self.wcs_frame_actions = {}
        for frame in available_display_frames(getattr(self.parent, "wcs", None)):
            action = QAction(display_frame_label(frame), self.parent, checkable=True)
            action.triggered.connect(lambda checked=False, frame_name=frame: self.set_wcs_frame(frame_name))
            self.wcs_frame_group.addAction(action)
            wcs_menu.addAction(action)
            self.wcs_frame_actions[frame] = action
        if self.wcs_frame_actions:
            wcs_menu.addSeparator()
        self.wcs_decimal_group = QActionGroup(self.parent)
        self.wcs_decimal_group.setExclusive(True)
        self.wcs_decimal_action = QAction("Decimal", self.parent, checkable=True)
        self.wcs_sexagesimal_action = QAction("Sexagesimal", self.parent, checkable=True)
        self.wcs_decimal_action.triggered.connect(lambda checked=False: self.set_wcs_decimal_mode(True))
        self.wcs_sexagesimal_action.triggered.connect(lambda checked=False: self.set_wcs_decimal_mode(False))
        self.wcs_decimal_group.addAction(self.wcs_decimal_action)
        self.wcs_decimal_group.addAction(self.wcs_sexagesimal_action)
        wcs_menu.addAction(self.wcs_decimal_action)
        wcs_menu.addAction(self.wcs_sexagesimal_action)

        current_frame = preferred_display_frame(getattr(self.parent, "wcs", None))
        getter = getattr(self.parent, "get_wcs_display_frame", None)
        if callable(getter):
            try:
                current_frame = normalize_display_frame(getter())
            except Exception:
                current_frame = preferred_display_frame(getattr(self.parent, "wcs", None))
        self.set_wcs_frame_checked(current_frame)
        decimal_mode = True
        decimal_getter = getattr(self.parent, "get_wcs_decimal_mode", None)
        if callable(decimal_getter):
            try:
                decimal_mode = bool(decimal_getter())
            except Exception:
                decimal_mode = True
        self.set_wcs_decimal_checked(decimal_mode)

        # About (placed in macOS application menu via AboutRole)
        about_action = QAction("About...", self.parent)
        about_action.setMenuRole(QAction.MenuRole.AboutRole)
        about_action.triggered.connect(self.show_about)
        file_menu.addAction(about_action)

    def enable_plane_menu(self, enabled):
        available = bool(enabled)
        self.sub1_action.setEnabled(available)
        self.sub2_action.setEnabled(available)

    def set_undo_redo_enabled(
        self,
        can_undo: bool,
        can_redo: bool,
        *,
        undo_label: str | None = None,
        redo_label: str | None = None,
    ):
        if hasattr(self, "undo_action") and self.undo_action is not None:
            self.undo_action.setEnabled(bool(can_undo))
            text = "Undo Analysis"
            if can_undo and undo_label:
                text = f"Undo Analysis ({undo_label})"
            self.undo_action.setText(text)
        if hasattr(self, "redo_action") and self.redo_action is not None:
            self.redo_action.setEnabled(bool(can_redo))
            text = "Redo Analysis"
            if can_redo and redo_label:
                text = f"Redo Analysis ({redo_label})"
            self.redo_action.setText(text)

    def set_view_navigation_enabled(self, can_back: bool, can_forward: bool):
        if hasattr(self, "view_back_action") and self.view_back_action is not None:
            self.view_back_action.setEnabled(bool(can_back))
        if hasattr(self, "view_forward_action") and self.view_forward_action is not None:
            self.view_forward_action.setEnabled(bool(can_forward))

    def set_wcs_frame_checked(self, frame: str):
        normalized = normalize_display_frame(frame)
        target = self.wcs_frame_actions.get(normalized)
        if target is None:
            target = self.wcs_frame_actions.get(preferred_display_frame(getattr(self.parent, "wcs", None)))
        if target is not None:
            target.setChecked(True)

    def set_wcs_frame(self, frame: str):
        setter = getattr(self.parent, "set_wcs_display_frame", None)
        if callable(setter):
            setter(frame)
        self.set_wcs_frame_checked(frame)

    def set_wcs_decimal_checked(self, use_decimal: bool):
        if use_decimal:
            self.wcs_decimal_action.setChecked(True)
        else:
            self.wcs_sexagesimal_action.setChecked(True)

    def set_wcs_decimal_mode(self, use_decimal: bool):
        setter = getattr(self.parent, "set_wcs_decimal_mode", None)
        if callable(setter):
            setter(bool(use_decimal))
        self.set_wcs_decimal_checked(bool(use_decimal))

    def undo_action_triggered(self):
        if hasattr(self.parent, "undo_last_action"):
            self.parent.undo_last_action()

    def redo_action_triggered(self):
        if hasattr(self.parent, "redo_last_action"):
            self.parent.redo_last_action()

    def view_back_triggered(self):
        if hasattr(self.parent, "view_back"):
            self.parent.view_back()

    def view_forward_triggered(self):
        if hasattr(self.parent, "view_forward"):
            self.parent.view_forward()

    def save_recipe(self):
        if hasattr(self.parent, "save_recipe_dialog"):
            self.parent.save_recipe_dialog()

    def load_recipe(self):
        if hasattr(self.parent, "load_recipe_dialog"):
            self.parent.load_recipe_dialog()

    def save_workspace(self):
        if hasattr(self.parent, "save_workspace"):
            self.parent.save_workspace()
        elif hasattr(self.parent, "save_workspace_dialog"):
            self.parent.save_workspace_dialog()

    def load_workspace(self):
        if hasattr(self.parent, "load_workspace_dialog"):
            self.parent.load_workspace_dialog()

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
            ensure = getattr(self.parent, "ensure_subwindow1", None)
            if callable(ensure):
                sub1 = ensure()
            else:
                sub1 = getattr(self.parent.SubWindow, "subwindow1", None)
            if sub1 is not None:
                sub1.show()
                sub1.raise_()
                sub1.activateWindow()
            else:
                self.sub1_action.setChecked(False)
        else:
            sub1 = getattr(self.parent.SubWindow, "subwindow1", None)
            if sub1 is not None:
                sub1.hide()
    
    def toggle_sub2_window(self):
        if self.sub2_action.isChecked():
            ensure = getattr(self.parent, "ensure_subwindow2", None)
            if callable(ensure):
                sub2 = ensure()
            else:
                sub2 = getattr(self.parent.SubWindow, "subwindow2", None)
            if sub2 is not None:
                sub2.show()
                sub2.raise_()
                sub2.activateWindow()
            else:
                self.sub2_action.setChecked(False)
        else:
            sub2 = getattr(self.parent.SubWindow, "subwindow2", None)
            if sub2 is not None:
                sub2.hide()
        
        
    def _ensure_control_panel(self, *, visible: bool = True):
        ensure = getattr(self.parent, "ensure_control_panel", None)
        if callable(ensure):
            try:
                return ensure(visible=visible)
            except TypeError:
                return ensure()
        return getattr(self.parent, "control_panel", None)

    def _open_control_panel_child(self, panel_attr: str, opener_name: str):
        control_panel = self._ensure_control_panel(visible=False)
        if control_panel is None:
            return
        panel = getattr(control_panel, panel_attr, None)
        if panel is None:
            opener = getattr(control_panel, opener_name, None)
            if callable(opener):
                opener()
            return
        panel.raise_()
        panel.activateWindow()

    def open_colorscale_panel(self):
        self._open_control_panel_child("color_settings_panel", "open_color_settings")
            
    def open_scaling_panel(self):
        self._open_control_panel_child("scaling_panel", "open_scaling_panel")

    def open_unit_conversion_panel(self):
        self._open_control_panel_child("unit_conversion_panel", "open_unit_conversion_panel")
    
    def open_chmap_panel(self):
        self._open_control_panel_child("chmap_settings_panel", "open_chmap_settings")

    def open_clump_finding_panel(self):
        self._open_control_panel_child("clump_finding_panel", "open_clump_finding_panel")

    def open_integ_panel(self):
        self._open_control_panel_child("integ_settings_panel", "open_integ_settings")

    def open_cutout_dialog(self):
        if hasattr(self.parent, 'open_cutout_dialog'):
            self.parent.open_cutout_dialog(use_view_bounds=True)

    def open_mask_panel(self):
        self._open_control_panel_child("mask_settings_panel", "open_mask_settings")

    def open_pvd_panel(self):
        self._open_control_panel_child("pvd_panel", "open_pvd_settings")
    
    def open_spec_window(self):
        self._open_control_panel_child("spec_window", "open_spec_window")

    def open_baseline_panel(self):
        self._open_control_panel_child("baseline_panel", "open_baseline_panel")

    def open_smooth_panel(self):
        self._open_control_panel_child("smooth_settings_panel", "open_smooth_settings")

    def open_contour_panel(self):
        self._open_control_panel_child("contour_panel", "open_contour_panel")


    def open_config_panel(self):
        self.config_panel = ConfigPanel(self.parent.config_manager, self.parent)
        self.config_panel.show()
        
    def open_header_panel(self):
        self.header_panel = ShowHeader(self.parent.header)
        self.header_panel.resize(300, 400)
        self.header_panel.show()
        
    def show_about(self):
        dlg = QMessageBox(self.parent)
        dlg.setWindowTitle("About")
        dlg.setTextFormat(Qt.TextFormat.RichText)
        dlg.setText(
            f'<table cellpadding="4">'
            f'<tr><td><b>{APP_NAME}</b>&nbsp; version {APP_DISPLAY_VERSION}</td>'
            f'<td><a href="https://github.com/s-takekawa/takefits">GitHub</a></td></tr>'
            # f'<tr><td>&copy; Shunya Takekawa</td>'
            # f'<td><a href="https://orcid.org/0000-0001-8147-6817">ORCID</a></td></tr>'
            f'</table>'
        )
        dlg.exec()

    def open_arithmetic_panel(self):
        self._open_control_panel_child("arithmetic_panel", "open_arithmetic_panel")


def _copy_menu_contents(source_menu: QMenu, target_menu: QMenu):
    for source_action in list(source_menu.actions() or []):
        if source_action is None:
            continue
        if source_action.isSeparator():
            target_menu.addSeparator()
            continue
        source_submenu = source_action.menu()
        if source_submenu is not None:
            title = str(source_submenu.title() or source_action.text() or "").strip()
            submenu = target_menu.addMenu(title or "Menu")
            submenu.menuAction().setEnabled(source_action.isEnabled())
            submenu.menuAction().setVisible(source_action.isVisible())
            _copy_menu_contents(source_submenu, submenu)
            continue
        target_menu.addAction(source_action)


def mirror_menu_bar_to_window(source_window, target_window) -> bool:
    if source_window is None or target_window is None:
        return False
    if source_window is target_window:
        return False
    source_getter = getattr(source_window, "menuBar", None)
    target_getter = getattr(target_window, "menuBar", None)
    if not callable(source_getter) or not callable(target_getter):
        return False
    try:
        source_bar = source_window.menuBar()
        target_bar = target_window.menuBar()
    except Exception:
        return False
    if source_bar is None or target_bar is None:
        return False
    try:
        target_bar.clear()
    except Exception:
        return False

    for source_action in list(source_bar.actions() or []):
        if source_action is None:
            continue
        source_menu = source_action.menu()
        if source_menu is None:
            target_bar.addAction(source_action)
            continue
        title = str(source_menu.title() or source_action.text() or "").strip()
        mirrored_menu = target_bar.addMenu(title or "Menu")
        mirrored_menu.menuAction().setEnabled(source_action.isEnabled())
        mirrored_menu.menuAction().setVisible(source_action.isVisible())
        _copy_menu_contents(source_menu, mirrored_menu)
    return True
