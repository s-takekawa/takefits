from ui.viewer import FITSViewer
from ui.menu_bar import MenuBar
from ui.subwindow import SubWindow, SubWindow_control

from ui.control_panel import ControlPanel
from ui.range_control import RangeControlPanel
from tools.regrid_panel import RegridPanel
from logic.regridder import Regridder
from core.save_fits import SaveFITS
from PyQt6.QtWidgets import QApplication, QMessageBox, QFileDialog, QInputDialog
from PyQt6.QtCore import Qt, QThread
import sys
import os
import json
from functools import partial
from core.common import Common


class MainWindow(FITSViewer):
    def __init__(self, plane, windowtitle, data, header, wcs, filename, spectral_metadata=None):
        super().__init__(data, header, wcs, filename, spectral_metadata)
        self.data = data
        self.header = header
        self.wcs = wcs
        
        self.plane = plane
        self.integ_result_windows = []
        self.initUI(plane)
        self.setWindowTitle(windowtitle)
        self.original_window_title = windowtitle
        
        self.filename = os.path.basename(filename)

        self.menu_bar = MenuBar.get_instance(self) #Make menubar
    
        self.region_mode_enabled = False
        self.region_shape = None  # Add this line to store the current shape

        # Connect the new actions to the handler
        self.menu_bar.circle_action.triggered.connect(lambda: self.set_region_shape("circle"))
        self.menu_bar.rectangle_action.triggered.connect(lambda: self.set_region_shape("rectangle"))
        self.menu_bar.ellipse_action.triggered.connect(lambda: self.set_region_shape("ellipse"))
        self.menu_bar.cube_action.triggered.connect(lambda: self.set_region_shape('cube')) 

        # Initialize the control panel
        self.subwindows = []
        if self.data.ndim > 2:
            self.subwindow1 = SubWindow('xz', "SubWindow1: %s" % filename, self) 
            self.subwindow2 = SubWindow('zy', "SubWindow2: %s" % filename, self) 
            self.subwindows.extend([self.subwindow1, self.subwindow2])
            SubWindow_control.update_subwindow(self.subwindow1, self.subwindow2)
            #self.subwindow_UI(self.subwindow1, self.subwindow2)
            self.menu_bar.enable_plane_menu(True)
            self.menu_bar.sub1_action.setChecked(True)
            self.menu_bar.sub2_action.setChecked(False) ##hiding Subwindow 2
            self.subwindow2.hide()
            self.subwindow1.raise_()
        else:
            self.menu_bar.enable_plane_menu(False)
        
        self.menu_bar.main_action.setChecked(True)

        self.control_panel = ControlPanel(self, self.subwindows)
        self.range_panel = RangeControlPanel(self, self.subwindows)
        
        self.range_panel.show()

        # Normalise range displays so all panels share the same baseline values.
        self.reset_all_ranges()
        self.range_panel.update_ranges('xy', None, None)
        if self.data.ndim > 2:
            self.range_panel.update_ranges('xz', None, None)
            self.range_panel.update_ranges('zy', None, None)
        
        #Color Scale connect
        self.color_button.clicked.connect(self.control_panel.open_color_settings)
        self.smooth_button.clicked.connect(self.control_panel.open_smooth_settings)
        if self.data.ndim > 2:
            #Integ connect
            self.integ_button.clicked.connect(self.control_panel.open_integ_settings)
            self.spec_button.clicked.connect(self.control_panel.open_spec_window)
        
        Common.main_window = self

        for subwindow in self.subwindows:
            subwindow.color_button.clicked.connect(self.control_panel.open_color_settings)
            subwindow.integ_button.clicked.connect(self.control_panel.open_integ_settings)
            subwindow.smooth_button.clicked.connect(self.control_panel.open_smooth_settings)
            subwindow.spec_button.clicked.connect(self.control_panel.open_spec_window)
        self.raise_()
        self.activateWindow()

        self._regrid_panel = None
        self._regrid_thread = None
        self._regrid_worker = None

    def set_region_shape(self, shape):
        """
        Sets the shape for the region selection mode.
        If the selected shape's action is already checked, it disables the region mode.
        """
        action = None
        if shape == "circle":
            action = self.menu_bar.circle_action
        elif shape == "rectangle":
            action = self.menu_bar.rectangle_action
        elif shape == "ellipse":
            action = self.menu_bar.ellipse_action
        elif shape == "cube":
            action = self.menu_bar.cube_action

        # If the current shape is re-selected, toggle region mode off.
        if self.region_shape == shape and self.region_mode_enabled:
            self.region_mode_enabled = False
            self.region_shape = None
            action.setChecked(False)  # Uncheck the menu item
            self.setWindowTitle(self.original_window_title)
        # Otherwise, enable region mode with the selected shape.
        elif action and action.isChecked():
            self.region_mode_enabled = True
            self.region_shape = shape
            self.setWindowTitle(f"[REGION MODE: {shape.upper()}] {self.original_window_title}")
            self.region_manager.set_region_mode(shape)
            if hasattr(self, 'set_marker_mode'):
                self.set_marker_mode(False)
            self._reset_navigation_mode()
        # This case handles unchecking via the action group
        else:
            self.region_mode_enabled = False
            self.region_shape = None
            self.setWindowTitle(self.original_window_title)

        if not self.region_mode_enabled:
            for region_action in (self.menu_bar.circle_action,
                                   self.menu_bar.rectangle_action,
                                   self.menu_bar.ellipse_action):
                region_action.setChecked(False)


        # Propagate the state to subwindows
        for subwindow in self.subwindows:
            subwindow.region_mode_enabled = self.region_mode_enabled
            subwindow.region_shape = self.region_shape 

        # Notify other windows like IntegResultWindow
        self.integ_result_windows = [ref for ref in self.integ_result_windows if ref() is not None]
        for window_ref in self.integ_result_windows:
            window = window_ref()
            if window:
                window.set_region_mode(self.region_mode_enabled)


    def disable_region_mode(self):
        super().disable_region_mode()
        if hasattr(self, 'menu_bar'):
            for action in (self.menu_bar.circle_action,
                           self.menu_bar.rectangle_action,
                           self.menu_bar.ellipse_action,
                           self.menu_bar.cube_action):
                if action is not None:
                    action.setChecked(False)




    def show_main_window(self):
        self.show()  # Main window should always be visible

    def hide_main_window(self):
        self.hide()  # Hide the main window

    def show_control_panel(self):
        if not self.control_panel.isVisible():
            self.control_panel.show()
    
    def show_range_panel(self):
        if not self.range_panel.isVisible():
            self.range_panel.show()

    def hide_control_panel(self):
        self.control_panel.hide()
    
    def hide_range_panel(self):
        self.range_panel.hide()

    def on_control_panel_closed(self):
        # When control panel is closed, uncheck the menu action
        self.menu_bar.control_panel_action.setChecked(False)
    
    def subwindow_UI(self, subWindow1, subWindow2):
        True


    def clear_all_regions_globally(self):
        """
        Clears all regions from the main viewer and any open integration result windows.
        """
        if hasattr(self, 'region_manager'):
            self.region_manager.delete_all_regions()

        if hasattr(self, 'integ_result_windows'):
            live_windows = []
            for window_ref in self.integ_result_windows:
                window = window_ref()
                if window is not None:
                    if hasattr(window, 'region_manager'):
                        window.region_manager.delete_all_regions()
                    live_windows.append(window_ref)

            self.integ_result_windows = live_windows

    def closeEvent(self, event):
        self._unregister_contour_layer()
        print("\n\nProgram exited.")
        QApplication.instance().quit()
        sys.exit()

    # ------------------------------------------------------------------
    # Region save/load
    def save_regions_dialog(self):
        if not hasattr(self, "region_manager"):
            return
        # Collect candidates (manager, plane, label) that have regions
        candidates = []
        # Main viewer
        main_regions = getattr(self.region_manager, "regions", [])
        if main_regions:
            planes = {getattr(r, "plane", None) or getattr(self, "plane", "xy") for r in main_regions}
            for plane_name in planes:
                candidates.append((self.region_manager, plane_name.lower(), "Main"))
        # Integ windows
        if hasattr(self, "integ_result_windows"):
            for window_ref in list(self.integ_result_windows):
                window = window_ref()
                if window is None:
                    continue
                mgr = getattr(window, "region_manager", None)
                regs = getattr(mgr, "regions", []) if mgr is not None else []
                if not regs:
                    continue
                planes = {getattr(r, "plane", None) or getattr(window, "plane", "xy") for r in regs}
                label_attr = getattr(window, "original_window_title", None)
                label = None
                if label_attr:
                    label = label_attr
                else:
                    label_attr = getattr(window, "windowTitle", None)
                    if callable(label_attr):
                        try:
                            label = label_attr()
                        except Exception:
                            label = None
                    else:
                        label = label_attr
                if isinstance(label, str) and label.startswith("[REGION MODE"):
                    closing = label.find("]")
                    if closing != -1:
                        label = label[closing + 1 :].strip()
                if not label:
                    label = f"Integ ({getattr(window, 'plane', 'xy').upper()})"
                for plane_name in planes:
                    candidates.append((mgr, plane_name.lower(), label))

        if not candidates:
            QMessageBox.information(self, "Regions", "No regions to save.")
            return

        # Deduplicate manager/plane combos
        unique = []
        seen = set()
        for mgr, plane_name, label in candidates:
            key = (id(mgr), plane_name)
            if key in seen:
                continue
            seen.add(key)
            unique.append((mgr, plane_name, label))

        if len(unique) == 1:
            target_manager, target_plane, _label = unique[0]
        else:
            items = []
            for _, pl, lbl in unique:
                short_lbl = str(lbl)
                if len(short_lbl) > 24:
                    short_lbl = short_lbl[:21] + "..."
                items.append(f"{short_lbl} - {pl.upper()}")
            choice, ok = QInputDialog.getItem(self, "Save Regions", "Select panel/plane to save:", items, 0, False)
            if not ok or not choice:
                return
            idx = items.index(choice)
            target_manager, target_plane, _label = unique[idx]
        default_dir = os.path.dirname(getattr(self, "filename_path", "")) or os.getcwd()
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Regions",
            os.path.join(default_dir, f"regions_{target_plane}.region"),
            "Region Files (*.region.json *.region *.json);;All Files (*)",
        )
        if not path:
            return
        if not path.lower().endswith((".region.json", ".region", ".json")):
            path += ".region"
        try:
            payload = target_manager.export_regions_to_dict(target_plane)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            QMessageBox.information(self, "Regions", f"Saved {len(payload.get('regions', []))} region(s).")
        except Exception as exc:
            QMessageBox.warning(self, "Regions", f"Failed to save regions: {exc}")

    def load_regions_dialog(self):
        if not hasattr(self, "region_manager"):
            return
        default_dir = os.path.dirname(getattr(self, "filename_path", "")) or os.getcwd()
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Regions",
            default_dir,
            "Region Files (*.region.json *.region *.json);;All Files (*)",
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            # Load into main viewer
            self.region_manager.import_regions_from_dict(payload, clear_existing=True)
            # If region mode was off, enable it with circle mode for immediate interaction
            if not getattr(self, "region_mode_enabled", False):
                try:
                    try:
                        self.menu_bar.circle_action.setChecked(True)
                    except Exception:
                        pass
                    self.set_region_shape("circle")
                except Exception:
                    pass
            # Also load into any open integration result windows on the XY plane
            if hasattr(self, "integ_result_windows"):
                for window_ref in list(self.integ_result_windows):
                    window = window_ref()
                    if window is None:
                        continue
                    try:
                        if getattr(window, "plane", "").lower() != "xy":
                            continue
                        if hasattr(window, "region_manager"):
                            window.region_manager.import_regions_from_dict(payload, clear_existing=True)
                    except Exception:
                        continue
            QMessageBox.information(self, "Regions", "Regions loaded.")
        except Exception as exc:
            QMessageBox.warning(self, "Regions", f"Failed to load regions: {exc}")

    def open_regrid_panel(self):
        if self._regrid_panel is None:
            self._regrid_panel = RegridPanel(fits_viewer=self)
            self._regrid_panel.regrid_requested.connect(self._start_regrid_job)
            self._regrid_panel.closed.connect(self._clear_regrid_panel)
        self._regrid_panel.show()
        self._regrid_panel.raise_()
        self._regrid_panel.activateWindow()

    def _clear_regrid_panel(self):
        self._regrid_panel = None

    def _start_regrid_job(self, params):
        if self._regrid_thread is not None and self._regrid_thread.isRunning():
            QMessageBox.information(
                self,
                "Regrid In Progress",
                "A regrid operation is already running. Please wait for it to finish.",
            )
            return

        self._last_regrid_params = dict(params)
        self._regrid_thread = QThread(self)
        header_copy = self.header.copy() if hasattr(self, "header") and self.header is not None else None
        self._regrid_worker = Regridder(self.data, self.wcs, header_copy, filename=self.filename)
        self._regrid_worker.moveToThread(self._regrid_thread)

        self._regrid_worker.progress.connect(self._handle_regrid_progress)
        self._regrid_worker.finished.connect(self._handle_regrid_finished)
        self._regrid_worker.error.connect(self._handle_regrid_error)

        self._regrid_thread.started.connect(
            partial(self._regrid_worker.perform_regrid, params)
        )
        self._regrid_worker.finished.connect(self._regrid_thread.quit)
        self._regrid_worker.error.connect(self._regrid_thread.quit)
        self._regrid_thread.finished.connect(self._cleanup_regrid_thread)
        self._regrid_thread.start()

        if self._regrid_panel:
            self._regrid_panel.on_regrid_started()

    def _handle_regrid_progress(self, value):
        if self._regrid_panel:
            self._regrid_panel.update_progress(value)

    def _handle_regrid_finished(self, data, header):
        if self._regrid_panel:
            self._regrid_panel.on_regrid_finished(True)
        saver = SaveFITS(data, header, self.filename, original_header=self.header)
        suffix = self._resolve_regrid_suffix()
        saver.save(suffix=suffix)

    def _handle_regrid_error(self, message):
        if self._regrid_panel:
            self._regrid_panel.on_regrid_finished(False)
            QMessageBox.critical(self._regrid_panel, "Regrid Failed", message)
        else:
            QMessageBox.critical(self, "Regrid Failed", message)

    def _cleanup_regrid_thread(self):
        if self._regrid_worker:
            self._regrid_worker.deleteLater()
            self._regrid_worker = None
        thread = self._regrid_thread
        self._regrid_thread = None
        if thread:
            thread.deleteLater()

    def _resolve_regrid_suffix(self) -> str:
        params = getattr(self, "_last_regrid_params", {}) or {}
        mode = params.get("mode", "").lower()
        if mode == "reproject_system":
            target = (params.get("target_system") or "").strip().lower()
            mapping = {
                "galactic": "lb",
                "icrs": "icrs",
                "fk5": "fk5",
                 "fk4": "fk4",
                "ecliptic": "ecliptic",
            }
            if target in mapping:
                return mapping[target]
        elif mode == "template_fits":
            return "reproject"
        return "reg"
