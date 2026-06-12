from takefits.ui.viewer import FITSViewer
from takefits.core.contour_manager import ContourManager
from PySide6.QtCore import QTimer

class SubWindow(FITSViewer):
    def __init__(self, plane, windowtitle, parent):
        self.main_viewer = parent
        # Preset before super().__init__() so FITSViewer keeps this value.
        self.main_window = parent
        # Share one marker manager across main/sub windows so placement/drag
        # state is consistent regardless of which window is clicked.
        self.marker_manager = parent.marker_manager
        super().__init__(parent.data, parent.header, parent.wcs, parent.filename_path, parent.spectral_metadata)
        self.plane = plane
        self._first_show_pending = True
        self.initUI(plane)
        self.setWindowTitle(windowtitle)
        
    def showEvent(self, event):
        super().showEvent(event)
        # Refresh contours when the subwindow becomes visible
        # This handles the case where contours were applied while the window was hidden
        if self._contour_layer_id:
            manager = ContourManager.instance()
            layer = manager._layers.get(self._contour_layer_id)
            if layer and layer._last_parameters is not None:
                manager.refresh_layer(self._contour_layer_id)
        
        # Synchronize slice with current shared coordinates
        if self.plane == 'xz':
            current_y = int(round(self.main_viewer._get_shared_ypix()))
            if hasattr(self, 'slider'):
                self.slider.setValue(current_y)
        elif self.plane == 'zy':
            current_x = int(round(self.main_viewer._get_shared_xpix()))
            if hasattr(self, 'slider'):
                self.slider.setValue(current_x)
            if getattr(self, "_first_show_pending", False):
                QTimer.singleShot(0, self._sync_initial_cursor_visibility_for_zy)
        self._first_show_pending = False

    def closeEvent(self, event):
        if self.toolbar._subplot_dialog is not None:
            self.toolbar._subplot_dialog.close()
        super().closeEvent(event)
        if self.plane == 'xz':
            self.main_viewer.menu_bar.sub1_action.setChecked(False)
        elif self.plane == 'zy':
            self.main_viewer.menu_bar.sub2_action.setChecked(False)

    def _sync_initial_cursor_visibility_for_zy(self):
        if self.plane != 'zy' or self.main_viewer is None:
            return
        try:
            lines = [
                self.main_viewer._get_plane_vline('xy'),
                self.main_viewer._get_plane_hline('xy'),
                self.main_viewer._get_plane_vline('xz'),
                self.main_viewer._get_plane_hline('xz'),
            ]
            show_cursor = any(line is not None and line.get_visible() for line in lines)
            if show_cursor:
                # Set ZY cursor lines directly to avoid slice refresh and timing races
                # with the XY-driven update path during first show.
                zpix = float(self.main_viewer._get_shared_zpix())
                ypix = float(self.main_viewer._get_shared_ypix())
                self.vline.set_xdata([zpix])
                self.hline.set_ydata([ypix])
                self._update_plane_cursor('zy', x=zpix, y=ypix)
                self._set_crosshair_point_for_plane('zy', x=zpix, y=ypix)
                self._set_crosshair_visibility_for_plane('zy', True)
                chlabel = self.main_viewer._get_plane_chlabel('zy')
                if chlabel is not None:
                    chlabel.set_visible(True)
                redraw_overlay = getattr(self.main_viewer, "redraw_overlay_for_plane", None)
                if callable(redraw_overlay):
                    redraw_overlay('zy')
                else:
                    self.canvas.draw_idle()
            else:
                self.canvas.draw_idle()
        except Exception:
            self.canvas.draw_idle()
        
class SubWindow_control:
    def __init__(self):
        subwindow1 = None
        subwindow2 = None
        
    @classmethod
    def update_subwindow(cls, sub1, sub2):
            cls.subwindow1, cls.subwindow2 = sub1, sub2
